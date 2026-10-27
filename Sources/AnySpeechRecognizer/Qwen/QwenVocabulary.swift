import Foundation

/// The prompt, special tokens, and model settings of a converted Qwen3-ASR model,
/// as `Tools/QwenConversion/convert.py` writes them to `qwen3_asr_config.json`.
struct QwenModelConfiguration: Decodable, Sendable, Equatable {
    struct SpecialTokens: Decodable, Sendable, Equatable {
        let audioPad: Int
        let asrText: Int
        let endOfText: Int
    }

    struct Language: Decodable, Sendable, Equatable {
        /// The name in the prefill, such as `Chinese`.
        let name: String
        /// The tokens of `language <name><asr_text>`.
        let prefill: [Int]
    }

    struct Prompt: Decodable, Sendable, Equatable {
        /// The chat template before the audio placeholders.
        let prefix: [Int]
        /// The chat template after the audio placeholders, ending with the assistant turn.
        let suffix: [Int]
        /// Forced-language prefills by language code, such as `zh` or `yue`.
        let languages: [String: Language]
    }

    struct Audio: Decodable, Sendable, Equatable {
        let sampleRate: Int
        let melBins: Int
        let minimumSamples: Int
        let windowFrames: Int
        let windowTokens: Int
        let chunkFrames: Int
        let tokensPerChunk: Int
    }

    let contextLength: Int
    let blockLength: Int
    let hiddenSize: Int
    let vocabularySize: Int
    let specialTokens: SpecialTokens
    /// Tokens that decoding drops, as transformers' `skip_special_tokens` does.
    let skippedTokenIds: [Int]
    let endTokens: [Int]
    let prompt: Prompt
    let audio: Audio

    init(contentsOf url: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        do {
            self = try decoder.decode(Self.self, from: Data(contentsOf: url))
        } catch {
            throw SpeechRecognitionError.missingAssets(
                "The Qwen3-ASR configuration \(url.path) is invalid: \(error.localizedDescription)")
        }
    }

    /// The number of audio embeddings for a number of mel frames: 13 for each whole
    /// 100-frame chunk and `ceil(frames / 8)` for a partial one.
    func audioTokenCount(melFrames: Int) -> Int {
        let (whole, rest) = melFrames.quotientAndRemainder(dividingBy: audio.chunkFrames)
        return whole * audio.tokensPerChunk + (rest + 7) / 8
    }
}

/// Decodes Qwen's byte-level BPE tokens into text.
///
/// `vocab.json` lists each token's string in GPT-2's byte-to-character encoding, indexed by ID.
/// A token can hold part of a UTF-8 sequence, so text is decoded from the joined bytes.
struct QwenVocabulary: Sendable {
    /// The bytes of each token; empty for skipped and unused tokens.
    private let pieces: [[UInt8]]

    init(contentsOf url: URL, skipping skipped: Set<Int>) throws {
        let strings: [String]
        do {
            strings = try JSONDecoder().decode([String].self, from: Data(contentsOf: url))
        } catch {
            throw SpeechRecognitionError.missingAssets(
                "The Qwen3-ASR vocabulary \(url.path) is invalid: \(error.localizedDescription)")
        }
        self.init(strings: strings, skipping: skipped)
    }

    init(strings: [String], skipping skipped: Set<Int>) {
        let decoder = Self.byteDecoder
        pieces = strings.enumerated().map { index, string in
            guard !skipped.contains(index) else { return [] }
            return string.unicodeScalars.flatMap { scalar in
                decoder[scalar].map { [$0] } ?? Array(String(scalar).utf8)
            }
        }
    }

    var count: Int { pieces.count }

    /// The text of tokens, with invalid UTF-8 replaced by U+FFFD.
    func text(for tokens: [Int]) -> String {
        var bytes: [UInt8] = []
        for token in tokens where pieces.indices.contains(token) { bytes += pieces[token] }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// GPT-2's `bytes_to_unicode` mapping, inverted: printable Latin-1 bytes map to themselves,
    /// and the other 68 bytes to U+0100 and later, in byte order.
    static let byteDecoder: [Unicode.Scalar: UInt8] = {
        let printable = Array(33...126) + Array(161...172) + Array(174...255)
        var mapping: [Unicode.Scalar: UInt8] = [:]
        for byte in printable { mapping[Unicode.Scalar(UInt8(byte))] = UInt8(byte) }
        var next = 256
        for byte in 0...255 where !printable.contains(byte) {
            mapping[Unicode.Scalar(UInt32(next))!] = UInt8(byte)
            next += 1
        }
        return mapping
    }()
}

/// Parses Qwen3-ASR output, `language <name><asr_text><transcript>`, as transformers'
/// `Qwen3ASRProcessor.parse_output` and the original Qwen3-ASR library do.
enum QwenOutput {
    static let marker = "<asr_text>"

    /// The detected language name, or `nil` when the output names none or "None",
    /// and the transcript after removing long repetitions.
    static func parse(_ raw: String) -> (language: String?, transcript: String) {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return (nil, "") }
        if let range = text.range(of: "assistant\n") { text = String(text[range.upperBound...]) }
        text = fixRepetitions(text)
        guard let range = text.range(of: marker) else {
            return (nil, text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let prefix = text[..<range.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
        let transcript = text[range.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
        if prefix.lowercased() == "language none" { return (nil, transcript) }
        let line = prefix.split(whereSeparator: \.isNewline).lazy
            .map { $0.trimmingCharacters(in: .whitespaces) }.first { !$0.isEmpty }
        guard let line else { return (nil, transcript) }
        let language =
            line.lowercased().hasPrefix("language ")
            ? String(line.dropFirst("language ".count)).trimmingCharacters(in: .whitespaces) : line
        return (language.isEmpty ? nil : language, transcript)
    }

    /// Collapses a character repeated more than `threshold` times in a row to one character,
    /// and the first pattern of up to 20 characters repeated `threshold` or more times in a row
    /// to one instance, then repeats the search after it, as the original library's
    /// `detect_and_fix_repetitions` does. Characters are Unicode scalars, as in Python.
    static func fixRepetitions(_ text: String, threshold: Int = 20) -> String {
        let scalars = Array(text.unicodeScalars)
        var characters: [Unicode.Scalar] = []
        var index = 0
        while index < scalars.count {
            var count = 1
            while index + count < scalars.count, scalars[index + count] == scalars[index] {
                count += 1
            }
            characters += count > threshold ? [scalars[index]] : scalars[index..<(index + count)]
            index += count
        }
        var result = String.UnicodeScalarView()
        result.append(contentsOf: fixPatternRepeats(characters[...], threshold: threshold))
        return String(result)
    }

    private static func fixPatternRepeats(
        _ text: ArraySlice<Unicode.Scalar>, threshold: Int, maximumLength: Int = 20
    ) -> [Unicode.Scalar] {
        let scalars = Array(text)
        let count = scalars.count
        guard count >= threshold * 2 else { return scalars }
        var result: [Unicode.Scalar] = []
        var index = 0
        while index <= count - threshold * 2 {
            for length in 1...maximumLength {
                if index + length * threshold > count { break }
                let pattern = scalars[index..<(index + length)]
                let repeats = (1..<threshold).allSatisfy { repetition in
                    let start = index + repetition * length
                    return scalars[start..<(start + length)] == pattern
                }
                guard repeats else { continue }
                var end = index + threshold * length
                while end + length <= count, scalars[end..<(end + length)] == pattern {
                    end += length
                }
                result += pattern
                result += fixPatternRepeats(
                    scalars[end...], threshold: threshold, maximumLength: maximumLength)
                return result
            }
            result.append(scalars[index])
            index += 1
        }
        return result + scalars[index...]
    }
}
