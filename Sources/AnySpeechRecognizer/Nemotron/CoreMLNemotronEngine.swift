import CoreML
import Foundation

/// Runs Nemotron's cache-aware encoder, prediction network, and joint decision with Core ML.
///
/// `Encoder.mlmodelc` is a multifunction model with one function per chunk size, such as
/// `chunk_560ms`, which share their weights. Each function takes `mel`, `mel_length`, the
/// caches `cache_channel` `[24, 1, 56, 1024]`, `cache_time` `[24, 1, 1024, 8]`, and
/// `cache_length`, and a one-hot `prompt`, and returns `encoder`, `encoder_length`, and the
/// next caches. `Decoder.mlmodelc` and `JointDecision.mlmodelc` have Parakeet's contract,
/// without durations; see ``CoreMLTransducer``.
///
/// The engine keeps two sets of cache arrays. Each step reads one set and writes the next
/// caches into the other through output backings, and the sets swap roles, so the caches do
/// not pass through Swift. The arrays have the type that the encoder declares, float32 or
/// float16. A step that does not update the caches leaves the roles unchanged,
/// and the next step overwrites what it wrote.
final class CoreMLNemotronEngine: NemotronInferenceEngine {
    struct Configuration: Sendable {
        /// The encoder function, such as `chunk_560ms`.
        var function: String
        var encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
        var predictionComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    }

    static let encoderName = "Encoder.mlmodelc"
    static let decoderName = "Decoder.mlmodelc"
    static let jointName = "JointDecision.mlmodelc"

    var model: ParakeetModelDescription { transducer.model }
    let chunkFrames: Int
    private let encoder: MLModel
    private let transducer: CoreMLTransducer
    private let mel: MLMultiArray
    private let melLength: MLMultiArray
    private let prompt: MLMultiArray
    private var promptSlot: Int?
    /// The two cache sets: channel, time, and length arrays.
    private let caches: [(channel: MLMultiArray, time: MLMultiArray, length: MLMultiArray)]
    private let inputs: [FeatureProvider]
    private let options: [MLPredictionOptions]
    private let encoded: MLMultiArray
    /// The cache set that the next step reads.
    private var current = 0

    /// Loads the models from a directory without downloading.
    init(directory: URL, configuration: Configuration) throws {
        func load(_ name: String, units: MLComputeUnits, function: String? = nil) throws
            -> MLModel
        {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SpeechRecognitionError.missingAssets("Missing Nemotron model \(url.path).")
            }
            let options = MLModelConfiguration()
            options.computeUnits = units
            options.functionName = function
            return try MLModel(contentsOf: url, configuration: options)
        }
        do {
            encoder = try load(
                Self.encoderName, units: configuration.encoderComputeUnits,
                function: configuration.function)
        } catch let error as SpeechRecognitionError {
            throw error
        } catch {
            // Core ML reports a missing function as an invalid configuration.
            throw SpeechRecognitionError.missingAssets(
                "The Nemotron encoder cannot load the function \(configuration.function); "
                    + "convert the encoder with that chunk size. \(error.localizedDescription)")
        }
        transducer = try CoreMLTransducer(
            decoder: load(Self.decoderName, units: configuration.predictionComputeUnits),
            joint: load(Self.jointName, units: configuration.predictionComputeUnits),
            model: NemotronModel.transducer, name: "Nemotron")

        let description = encoder.modelDescription
        func shape(input name: String) -> [Int]? {
            description.inputDescriptionsByName[name]?.multiArrayConstraint?.shape.map(\.intValue)
        }
        func shape(output name: String) -> [Int]? {
            description.outputDescriptionsByName[name]?.multiArrayConstraint?.shape.map(\.intValue)
        }
        let model = NemotronModel.transducer
        guard let melShape = shape(input: "mel"), let encoderShape = shape(output: "encoder"),
            let channelShape = shape(input: "cache_channel"),
            let timeShape = shape(input: "cache_time"),
            melShape.count == 3, encoderShape.count == 3, melShape[1] == model.melBinCount,
            encoderShape[1] == model.encoderHiddenSize, encoderShape[2] > 0,
            melShape[2]
                == NemotronModel.preEncodeFrames + NemotronModel.subsampling * encoderShape[2],
            shape(output: "cache_channel_out") == channelShape,
            shape(output: "cache_time_out") == timeShape,
            shape(input: "prompt") == [1, NemotronModel.promptCount]
        else {
            throw SpeechRecognitionError.missingAssets(
                "The Nemotron encoder function \(configuration.function) does not have the "
                    + "expected cache-aware shapes.")
        }
        func array(_ shape: [Int], _ type: MLMultiArrayDataType = .float32) throws -> MLMultiArray {
            try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
        }
        chunkFrames = encoderShape[2]
        let mel = try array(melShape)
        let melLength = try array([1], .int32)
        let prompt = try array([1, NemotronModel.promptCount])
        let encoded = try array(encoderShape)
        // The caches keep the type that the encoder declares, float32 or float16.
        let cacheType =
            description.inputDescriptionsByName["cache_channel"]?.multiArrayConstraint?.dataType
            ?? .float32
        let caches = try (0..<2).map { _ in
            (
                channel: try array(channelShape, cacheType),
                time: try array(timeShape, cacheType), length: try array([1], .int32)
            )
        }
        (self.mel, self.melLength, self.prompt, self.encoded, self.caches) = (
            mel, melLength, prompt, encoded, caches
        )
        inputs = caches.map { set in
            FeatureProvider([
                "mel": mel, "mel_length": melLength, "prompt": prompt,
                "cache_channel": set.channel, "cache_time": set.time, "cache_length": set.length,
            ])
        }
        options = caches.indices.map { index in
            let next = caches[1 - index]
            let options = MLPredictionOptions()
            options.outputBackings = [
                "encoder": encoded, "cache_channel_out": next.channel,
                "cache_time_out": next.time, "cache_length_out": next.length,
            ]
            return options
        }
        resetCaches()
    }

    func resetCaches() {
        for set in caches {
            for array in [set.channel, set.time] {
                array.withUnsafeMutableBytes { pointer, _ in
                    _ = pointer.initializeMemory(as: UInt8.self, repeating: 0)
                }
            }
            set.length[0] = 0
        }
        current = 0
    }

    func encode(
        mel features: UnsafeBufferPointer<Float>, validFrameCount: Int, prompt slot: Int,
        updatesCaches: Bool, into output: inout ParakeetEncoderOutput
    ) throws {
        guard (0..<NemotronModel.promptCount).contains(slot) else {
            throw SpeechRecognitionError.unsupportedLanguage("Nemotron has no prompt slot \(slot).")
        }
        if promptSlot != slot {
            prompt.withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, _ in
                pointer.update(repeating: 0)
            }
            prompt[[0, NSNumber(value: slot)]] = 1
            promptSlot = slot
        }
        try mel.writeLogical(features)
        melLength[0] = NSNumber(value: Int32(validFrameCount))
        let result = try encoder.prediction(from: inputs[current], options: options[current])
        guard let frames = result.featureValue(for: "encoder")?.multiArrayValue,
            let length = result.featureValue(for: "encoder_length")?.multiArrayValue
        else {
            throw SpeechRecognitionError.invalidResult("The Nemotron encoder output is invalid.")
        }
        if updatesCaches { current = 1 - current }
        if output.hiddenSize != model.encoderHiddenSize || output.frameCount != chunkFrames {
            output = ParakeetEncoderOutput(
                hiddenSize: model.encoderHiddenSize, frameCount: chunkFrames)
        }
        try frames.transposeChannels(into: &output.features)
        output.validFrameCount = min(max(0, length[0].intValue), chunkFrames)
    }

    func predict(token: Int, state: inout ParakeetDecoderState, output: inout [Float]) throws {
        try transducer.predict(token: token, state: &state, output: &output)
    }

    func joint(
        encoderFrame: UnsafeBufferPointer<Float>, decoderOutput: UnsafeBufferPointer<Float>
    ) throws -> ParakeetJointDecision {
        try transducer.joint(encoderFrame: encoderFrame, decoderOutput: decoderOutput)
    }
}
