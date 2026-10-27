import Foundation

/// Transcribes one segment of audio with a Qwen3-ASR engine and greedy decoding.
///
/// A segment is transcribed as the processor and `generate()` in transformers transcribe one
/// clip: features over the whole segment, audio embeddings of each 8-second window,
/// the chat prompt with one placeholder per embedding, an optional forced-language prefill,
/// and greedy decoding until an end token. The output is parsed into a language and a transcript.
final class QwenRecognizer {
    struct Transcript: Sendable, Equatable {
        /// The language that the model named, such as `English`, or `nil` when it named none
        /// or the language was forced.
        var language: String?
        var text: String
        /// The generated tokens, without the end token.
        var tokens: [Int]
        /// Whether decoding stopped at the end of the decoder's context instead of at an end token.
        var reachedContextLimit: Bool
    }

    let engine: any QwenInferenceEngine
    let vocabulary: QwenVocabulary
    /// The forced-language prefill, or `nil` for automatic language identification.
    let prefill: [Int]?
    private let frontend = QwenMelFrontend()

    init(engine: any QwenInferenceEngine, vocabulary: QwenVocabulary, prefill: [Int]?) {
        self.engine = engine
        self.vocabulary = vocabulary
        self.prefill = prefill
    }

    var configuration: QwenModelConfiguration { engine.configuration }

    /// The audio embeddings of a segment, `[count × hiddenSize]` in token-major order.
    func audioEmbeddings(_ samples: [Float]) throws -> (values: [Float], count: Int) {
        let configuration = configuration
        let audio = configuration.audio
        let hidden = configuration.hiddenSize
        let features = samples.withUnsafeBufferPointer { frontend.compute($0) }
        let frames = QwenMelFrontend.frameCount(sampleCount: samples.count)
        let total = configuration.audioTokenCount(melFrames: frames)
        var values = [Float](repeating: 0, count: total * hidden)
        var window = [Float](repeating: 0, count: audio.melBins * audio.windowFrames)
        var encoded = [Float](repeating: 0, count: audio.windowTokens * hidden)
        var written = 0
        for start in stride(from: 0, to: frames, by: audio.windowFrames) {
            let length = min(audio.windowFrames, frames - start)
            window.withUnsafeMutableBufferPointer { window in
                window.update(repeating: 0)
                features.withUnsafeBufferPointer { features in
                    for bin in 0..<audio.melBins {
                        (window.baseAddress! + bin * audio.windowFrames).update(
                            from: features.baseAddress! + bin * frames + start, count: length)
                    }
                }
            }
            try window.withUnsafeBufferPointer { window in
                try encoded.withUnsafeMutableBufferPointer {
                    try engine.encode(window: window, validFrameCount: length, into: $0)
                }
            }
            let tokens = configuration.audioTokenCount(melFrames: length)
            values.withUnsafeMutableBufferPointer { values in
                (values.baseAddress! + written * hidden).update(
                    from: encoded, count: tokens * hidden)
            }
            written += tokens
        }
        return (values, total)
    }

    /// Transcribes 16 kHz samples. Segments shorter than 0.5 seconds are padded with silence,
    /// as the feature extractor pads them.
    func transcribe(_ samples: [Float], maximumNewTokens: Int = .max) throws -> Transcript {
        let configuration = configuration
        var samples = samples
        if samples.count < configuration.audio.minimumSamples {
            samples += [Float](
                repeating: 0, count: configuration.audio.minimumSamples - samples.count)
        }
        let (audio, audioCount) = try audioEmbeddings(samples)
        let prompt = configuration.prompt
        let ids =
            prompt.prefix
            + [Int](repeating: configuration.specialTokens.audioPad, count: audioCount)
            + prompt.suffix + (prefill ?? [])
        let hidden = configuration.hiddenSize
        var embeddings = [Float](repeating: 0, count: ids.count * hidden)
        embeddings.withUnsafeMutableBufferPointer { embeddings in
            (embeddings.baseAddress! + prompt.prefix.count * hidden).update(
                from: audio, count: audio.count)
        }
        guard ids.count < configuration.contextLength else {
            throw SpeechRecognitionError.invalidAudio(
                "A Qwen3-ASR segment of \(samples.count) samples does not fit the decoder's context."
            )
        }
        try engine.resetDecoder()
        var prediction = try embeddings.withUnsafeBufferPointer {
            try engine.append(ids, audioEmbeddings: $0)
        }
        let ends = Set(configuration.endTokens)
        var tokens: [Int] = []
        var position = ids.count
        var reachedContextLimit = false
        while !ends.contains(prediction.token) && tokens.count < maximumNewTokens {
            try Task.checkCancellation()
            tokens.append(prediction.token)
            guard position < configuration.contextLength else {
                reachedContextLimit = true
                break
            }
            prediction = try engine.append([prediction.token], audioEmbeddings: nil)
            position += 1
        }
        let parsed = QwenOutput.parse(vocabulary.text(for: tokens))
        return Transcript(
            language: prefill == nil ? parsed.language : nil, text: parsed.transcript,
            tokens: tokens,
            reachedContextLimit: reachedContextLimit)
    }
}

/// Chooses where to split long audio into segments, as the original Qwen3-ASR library does:
/// at the quietest sample of the quietest 100 ms within a search window around a target length.
enum QwenSegmenter {
    static let sampleRate = 16_000
    /// Audio of up to 40 seconds is one segment: 520 audio embeddings, which leave about 480
    /// tokens of the 1,024-token context for the prompt and transcript.
    static let maximumSamples = 40 * sampleRate
    /// Longer audio is cut near 30 seconds, within 5 seconds either way.
    static let targetSamples = 30 * sampleRate
    static let searchSamples = 5 * sampleRate
    static let windowSamples = sampleRate / 10

    /// The sample at which to end the first segment.
    /// - Parameters:
    ///   - samples: The audio to split.
    ///   - target: The preferred segment length in samples.
    ///   - search: How far from the target the boundary can be, in samples.
    /// - Returns: A boundary from 1 to `samples.count`.
    static func boundary(
        in samples: UnsafeBufferPointer<Float>, target: Int = targetSamples,
        search: Int = searchSamples
    ) -> Int {
        let count = samples.count
        let cut = min(target, count)
        let lower = max(0, cut - search)
        let upper = min(count, cut + search)
        guard upper - lower > windowSamples else { return max(1, cut) }
        // Window sums of absolute values, from prefix sums in double precision.
        var prefix = [Double](repeating: 0, count: upper - lower + 1)
        for index in lower..<upper {
            prefix[index - lower + 1] = prefix[index - lower] + Double(abs(samples[index]))
        }
        var quietest = 0
        var smallest = Double.infinity
        for start in 0...(upper - lower - windowSamples) {
            let sum = prefix[start + windowSamples] - prefix[start]
            if sum < smallest {
                smallest = sum
                quietest = start
            }
        }
        var inner = 0
        for offset in 0..<windowSamples
        where abs(samples[lower + quietest + offset]) < abs(samples[lower + quietest + inner]) {
            inner = offset
        }
        return min(count, max(1, lower + quietest + inner))
    }
}
