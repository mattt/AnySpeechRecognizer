import CoreML
import Foundation

/// The greedy next token after the tokens that a decoder has read, and its log probability.
struct QwenPrediction: Sendable, Equatable {
    let token: Int
    let logProbability: Float
}

/// Runs the audio encoder and text decoder of a Qwen3-ASR model.
///
/// The recognizer computes features, builds prompts, and decodes; the engine only runs models.
/// Tests use a fake engine.
protocol QwenInferenceEngine: AnyObject {
    var configuration: QwenModelConfiguration { get }

    /// Encodes one window of mel features into audio embeddings.
    /// - Parameters:
    ///   - window: `[melBins × windowFrames]` features in mel-major order, zero after the valid frames.
    ///   - validFrameCount: The number of valid frames, from 1 to `windowFrames`.
    ///   - output: `[windowTokens × hiddenSize]` embeddings in token-major order.
    func encode(
        window: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: UnsafeMutableBufferPointer<Float>) throws

    /// Empties the decoder's cache for a new prompt.
    func resetDecoder() throws

    /// Appends tokens at the next cache positions and predicts the token after the last one.
    /// - Parameters:
    ///   - tokens: Token IDs. Audio placeholders read their embedding from `audioEmbeddings`.
    ///   - audioEmbeddings: `[tokens.count × hiddenSize]` values, or `nil` without placeholders.
    func append(_ tokens: [Int], audioEmbeddings: UnsafeBufferPointer<Float>?) throws
        -> QwenPrediction
}

/// Runs `AudioEncoder.mlmodelc` and `TextDecoder.mlmodelc` with Core ML.
///
/// The decoder reads a fixed block of tokens in each prediction and keeps its KV cache in
/// Core ML states, one per layer for keys and one for values. A partial block repeats the
/// end-of-text token after the real tokens, at the following cache positions; later blocks
/// overwrite those positions before any token attends to them, because each row's mask allows
/// only positions up to its own. Each prediction returns the greedy token after every row,
/// and the engine reads the row of the last real token.
///
/// The engine owns its models and arrays, so one owner uses it at a time.
final class CoreMLQwenEngine: QwenInferenceEngine {
    static let encoderName = "AudioEncoder.mlmodelc"
    static let decoderName = "TextDecoder.mlmodelc"
    static let vocabularyName = "vocab.json"
    static let configurationName = "qwen3_asr_config.json"
    /// The additive mask value for positions that a row must not attend to.
    static let maskedValue: Float = -1e4

    let configuration: QwenModelConfiguration
    private let encoder: MLModel
    private let decoder: MLModel
    private var state: MLState
    private var position = 0
    private let mel: MLMultiArray
    private let melLength: MLMultiArray
    private let encoderInput: MLDictionaryFeatureProvider
    private let inputIds: MLMultiArray
    private let audioEmbeddings: MLMultiArray
    private let positionIds: MLMultiArray
    private let causalMask: MLMultiArray
    private let decoderInput: MLDictionaryFeatureProvider

    /// Loads the models from a directory without downloading.
    init(
        directory: URL, configuration: QwenModelConfiguration,
        encoderComputeUnits: MLComputeUnits, decoderComputeUnits: MLComputeUnits
    ) throws {
        func load(_ name: String, units: MLComputeUnits) throws -> MLModel {
            let url = directory.appending(path: name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SpeechRecognitionError.missingAssets("Missing Qwen3-ASR model \(url.path).")
            }
            let options = MLModelConfiguration()
            options.computeUnits = units
            return try MLModel(contentsOf: url, configuration: options)
        }
        self.configuration = configuration
        encoder = try load(Self.encoderName, units: encoderComputeUnits)
        decoder = try load(Self.decoderName, units: decoderComputeUnits)
        state = decoder.makeState()
        let audio = configuration.audio
        let block = NSNumber(value: configuration.blockLength)
        mel = try MLMultiArray(
            shape: [1, NSNumber(value: audio.melBins), NSNumber(value: audio.windowFrames)],
            dataType: .float32)
        melLength = try MLMultiArray(shape: [1], dataType: .int32)
        encoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "mel_length": MLFeatureValue(multiArray: melLength),
        ])
        inputIds = try MLMultiArray(shape: [1, block], dataType: .int32)
        audioEmbeddings = try MLMultiArray(
            shape: [1, block, NSNumber(value: configuration.hiddenSize)], dataType: .float32)
        positionIds = try MLMultiArray(shape: [block], dataType: .int32)
        causalMask = try MLMultiArray(
            shape: [1, 1, block, NSNumber(value: configuration.contextLength)], dataType: .float32)
        decoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "audio_embeddings": MLFeatureValue(multiArray: audioEmbeddings),
            "position_ids": MLFeatureValue(multiArray: positionIds),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
        ])
        try validateShapes()
    }

    func encode(
        window: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: UnsafeMutableBufferPointer<Float>
    ) throws {
        let audio = configuration.audio
        let hidden = configuration.hiddenSize
        precondition(window.count == audio.melBins * audio.windowFrames)
        precondition(output.count == audio.windowTokens * hidden)
        mel.withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, _ in
            pointer.baseAddress!.update(from: window.baseAddress!, count: window.count)
        }
        melLength[0] = NSNumber(value: Int32(validFrameCount))
        let result = try encoder.prediction(from: encoderInput)
        guard let embeddings = result.featureValue(for: "audio_embeddings")?.multiArrayValue,
            embeddings.shape.map(\.intValue) == [1, audio.windowTokens, hidden]
        else {
            throw SpeechRecognitionError.invalidResult("The Qwen3-ASR encoder output is invalid.")
        }
        let values = MLShapedArray<Float>(converting: embeddings).scalars
        output.baseAddress!.update(from: values, count: output.count)
    }

    func resetDecoder() throws {
        state = decoder.makeState()
        position = 0
    }

    func append(_ tokens: [Int], audioEmbeddings audio: UnsafeBufferPointer<Float>?) throws
        -> QwenPrediction
    {
        let block = configuration.blockLength
        let context = configuration.contextLength
        let hidden = configuration.hiddenSize
        guard !tokens.isEmpty else {
            throw SpeechRecognitionError.invalidResult("A Qwen3-ASR decoder step has no tokens.")
        }
        guard position + tokens.count <= context else {
            throw SpeechRecognitionError.invalidResult(
                "The Qwen3-ASR prompt and output exceed the \(context)-token context.")
        }
        var prediction = QwenPrediction(
            token: configuration.specialTokens.endOfText, logProbability: 0)
        var start = 0
        while start < tokens.count {
            let count = min(block, tokens.count - start)
            inputIds.withUnsafeMutableBufferPointer(ofType: Int32.self) { ids, _ in
                for row in 0..<block {
                    ids[row] = Int32(
                        row < count ? tokens[start + row] : configuration.specialTokens.endOfText)
                }
            }
            audioEmbeddings.withUnsafeMutableBufferPointer(ofType: Float.self) { values, _ in
                values.update(repeating: 0)
                if let audio {
                    values.baseAddress!.update(
                        from: audio.baseAddress! + start * hidden, count: count * hidden)
                }
            }
            positionIds.withUnsafeMutableBufferPointer(ofType: Int32.self) { positions, _ in
                for row in 0..<block { positions[row] = Int32(min(position + row, context - 1)) }
            }
            causalMask.withUnsafeMutableBufferPointer(ofType: Float.self) { mask, _ in
                for row in 0..<block {
                    let allowed = min(position + row, context - 1) + 1
                    let values = mask.baseAddress! + row * context
                    values.update(repeating: 0, count: allowed)
                    (values + allowed).update(repeating: Self.maskedValue, count: context - allowed)
                }
            }
            let result = try decoder.prediction(from: decoderInput, using: state)
            guard let ids = result.featureValue(for: "token_ids")?.multiArrayValue,
                let logProbabilities = result.featureValue(for: "token_log_probs")?.multiArrayValue,
                ids.count == block, logProbabilities.count == block
            else {
                throw SpeechRecognitionError.invalidResult(
                    "The Qwen3-ASR decoder output is invalid.")
            }
            let token = MLShapedArray<Int32>(converting: ids).scalars[count - 1]
            let logProbability = MLShapedArray<Float>(converting: logProbabilities).scalars[
                count - 1]
            guard (0..<configuration.vocabularySize).contains(Int(token)) else {
                throw SpeechRecognitionError.invalidResult(
                    "The Qwen3-ASR decoder returned an invalid token.")
            }
            prediction = QwenPrediction(token: Int(token), logProbability: logProbability)
            position += count
            start += count
        }
        return prediction
    }

    private func validateShapes() throws {
        func shape(_ model: MLModel, input name: String) -> [Int]? {
            model.modelDescription.inputDescriptionsByName[name]?.multiArrayConstraint?.shape
                .map(\.intValue)
        }
        let audio = configuration.audio
        let block = configuration.blockLength
        guard shape(encoder, input: "mel") == [1, audio.melBins, audio.windowFrames],
            shape(decoder, input: "input_ids") == [1, block],
            shape(decoder, input: "audio_embeddings") == [1, block, configuration.hiddenSize],
            shape(decoder, input: "causal_mask") == [1, 1, block, configuration.contextLength]
        else {
            throw SpeechRecognitionError.missingAssets(
                "The Qwen3-ASR models do not match the shapes in \(Self.configurationName).")
        }
    }
}
