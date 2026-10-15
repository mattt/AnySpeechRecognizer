import AVFoundation
import Foundation

@testable import AnySpeechRecognizer

enum ParakeetFixtures {
    static var environment: [String: String] { ProcessInfo.processInfo.environment }
    static var modelDirectory: URL? {
        environment["ANYSPEECH_PARAKEET_MODELS"].map { URL(fileURLWithPath: $0) }
    }
    static var shortAudio: URL? {
        environment["ANYSPEECH_PARAKEET_SHORT_AUDIO"].map { URL(fileURLWithPath: $0) }
    }
    static var longAudio: URL? {
        environment["ANYSPEECH_PARAKEET_LONG_AUDIO"].map { URL(fileURLWithPath: $0) }
    }

    /// Reads a file as 16 kHz mono Float32 samples.
    static func samples(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
        let input = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: input)
        if file.processingFormat == target {
            return Array(
                UnsafeBufferPointer(
                    start: input.floatChannelData![0], count: Int(input.frameLength)))
        }
        let converter = AVAudioConverter(from: file.processingFormat, to: target)!
        let capacity = AVAudioFrameCount(
            Double(input.frameLength) * 16_000 / file.processingFormat.sampleRate + 1024)
        let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity)!
        var supplied = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if supplied {
                status.pointee = .endOfStream
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return input
        }
        if let error { throw error }
        return Array(
            UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
    }

    static func text(_ tokens: [ParakeetToken], vocabulary: ParakeetVocabulary) -> String {
        vocabulary.text(for: tokens.map(\.id))
    }

    struct WordEdits: CustomStringConvertible {
        var substitutions = 0
        var deletions = 0
        var insertions = 0
        var words = 0
        /// Each edit as `-deleted`, `+inserted`, or `old→new`.
        var edits: [String] = []
        var description: String {
            "substitutions \(substitutions), deletions \(deletions), insertions \(insertions) of \(words) words: \(edits.joined(separator: " "))"
        }
    }

    /// Word-level edit operations between two texts, split on whitespace.
    static func wordDifferences(_ reference: String, _ hypothesis: String) -> WordEdits {
        let a = reference.split(whereSeparator: \.isWhitespace).map(String.init)
        let b = hypothesis.split(whereSeparator: \.isWhitespace).map(String.init)
        var cost = [[Int]](repeating: [Int](repeating: 0, count: b.count + 1), count: a.count + 1)
        for i in 0...a.count { cost[i][0] = i }
        for j in 0...b.count { cost[0][j] = j }
        for i in stride(from: 1, through: a.count, by: 1) {
            for j in stride(from: 1, through: b.count, by: 1) {
                cost[i][j] = min(
                    cost[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1), cost[i - 1][j] + 1,
                    cost[i][j - 1] + 1)
            }
        }
        var result = WordEdits(words: a.count)
        var (i, j) = (a.count, b.count)
        while i > 0 || j > 0 {
            if i > 0, j > 0, cost[i][j] == cost[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1) {
                if a[i - 1] != b[j - 1] {
                    result.substitutions += 1
                    result.edits.append("\(a[i - 1])→\(b[j - 1])")
                }
                (i, j) = (i - 1, j - 1)
            } else if i > 0, cost[i][j] == cost[i - 1][j] + 1 {
                result.deletions += 1
                result.edits.append("-\(a[i - 1])")
                i -= 1
            } else {
                result.insertions += 1
                result.edits.append("+\(b[j - 1])")
                j -= 1
            }
        }
        result.edits.reverse()
        return result
    }
}
