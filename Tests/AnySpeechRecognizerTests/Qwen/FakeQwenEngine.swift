import Foundation

@testable import AnySpeechRecognizer

/// A Qwen3-ASR engine whose decoder replies with scripted tokens.
///
/// Each prompt, from one `resetDecoder` call to the next, receives the next script in
/// `scripts`, or the last script again when they run out. The decoder returns a script's
/// tokens one per `append` call, then the end-of-text token. With `endless`, it never returns
/// an end token. The encoder writes each window's valid frame count into every embedding.
/// Tests read its records after the executor that uses it finishes, so it is unchecked `Sendable`.
final class FakeQwenEngine: QwenInferenceEngine, @unchecked Sendable {
    let configuration: QwenModelConfiguration
    var scripts: [[Int]]
    var endless = false
    /// The tokens of each prompt's first `append` call.
    private(set) var prompts: [[Int]] = []
    /// The audio embedding rows of each prompt's first `append` call, by token.
    private(set) var promptAudio: [[Float]] = []
    /// The valid frame count of each encoded window.
    private(set) var encodedWindows: [Int] = []
    /// The number of tokens appended to the current prompt, including the prompt.
    private(set) var position = 0
    private var reply: [Int] = []
    private var promptCount = 0

    init(configuration: QwenModelConfiguration = QwenTestSupport.configuration(), scripts: [[Int]])
    {
        self.configuration = configuration
        self.scripts = scripts
    }

    func encode(
        window: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: UnsafeMutableBufferPointer<Float>
    ) throws {
        encodedWindows.append(validFrameCount)
        output.update(repeating: Float(validFrameCount))
    }

    func resetDecoder() throws {
        position = 0
        reply = scripts[min(promptCount, scripts.count - 1)]
        promptCount += 1
    }

    func append(_ tokens: [Int], audioEmbeddings: UnsafeBufferPointer<Float>?) throws
        -> QwenPrediction
    {
        if position == 0 {
            prompts.append(tokens)
            let hidden = configuration.hiddenSize
            promptAudio.append(
                (0..<tokens.count).map { row in audioEmbeddings.map { $0[row * hidden] } ?? 0 })
        }
        position += tokens.count
        precondition(position <= configuration.contextLength)
        if endless { return QwenPrediction(token: QwenTestSupport.hello, logProbability: -0.1) }
        let token = reply.isEmpty ? configuration.specialTokens.endOfText : reply.removeFirst()
        return QwenPrediction(token: token, logProbability: -0.1)
    }
}

enum QwenTestSupport {
    // Token IDs of the test vocabulary.
    static let audioPad = 1
    static let asrText = 2
    static let endOfText = 3
    static let imEnd = 4
    static let prefix = 5
    static let suffix = 6
    static let language = 7
    static let english = 8
    static let hello = 9
    static let world = 10
    static let ni = 11
    static let hao = 12
    static let none = 13

    /// Token strings in GPT-2's byte encoding: `Ġ` is a space, and `ä½ł` and `å¥½` are the
    /// UTF-8 bytes of 你 and 好.
    static let strings = [
        "!", "<|audio_pad|>", "<asr_text>", "<|endoftext|>", "<|im_end|>", "<|im_start|>",
        "assistant", "language", "ĠEnglish", "Hello", "Ġworld", "ä½ł", "å¥½", "ĠNone",
    ]
    static let skipped: Set<Int> = [audioPad, endOfText, imEnd, prefix]

    static var vocabulary: QwenVocabulary { QwenVocabulary(strings: strings, skipping: skipped) }

    /// "language English<asr_text>Hello world"
    static let englishReply = [language, english, asrText, hello, world]
    /// "language Chinese<asr_text>你好", with the name spelled with the English token.
    static let chineseReply = [language, english, asrText, ni, hao]

    /// A configuration with the real audio settings and a small decoder.
    static func configuration(context: Int = 1_024) -> QwenModelConfiguration {
        let json = """
            {
              "context_length": \(context), "block_length": 4, "hidden_size": 2,
              "vocabulary_size": \(strings.count),
              "special_tokens": {"audio_pad": \(audioPad), "asr_text": \(asrText),
                "end_of_text": \(endOfText)},
              "skipped_token_ids": \(skipped.sorted()),
              "end_tokens": [\(endOfText), \(imEnd)],
              "prompt": {"prefix": [\(prefix)], "suffix": [\(suffix)],
                "languages": {"en": {"name": "English", "prefill": [\(language), \(english), \(asrText)]}}},
              "audio": {"sample_rate": 16000, "mel_bins": 128, "minimum_samples": 8000,
                "window_frames": 800, "window_tokens": 104, "chunk_frames": 100, "tokens_per_chunk": 13}
            }
            """
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try! decoder.decode(QwenModelConfiguration.self, from: Data(json.utf8))
    }

    /// A 440 Hz tone with silence from `silence`, in seconds, at 16 kHz.
    static func tone(seconds: Double, silence: Range<Double>? = nil) -> [Float] {
        (0..<Int(seconds * 16_000)).map { index in
            let time = Double(index) / 16_000
            if let silence, silence.contains(time) { return 0 }
            return Float(0.3 * sin(2 * Double.pi * 440 * time))
        }
    }
}
