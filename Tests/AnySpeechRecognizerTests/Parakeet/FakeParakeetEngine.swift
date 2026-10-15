import Foundation

@testable import AnySpeechRecognizer

/// A small Parakeet model whose "speech" is pure tones.
///
/// Token `k` is a tone at `ToneSpeech.frequency(k)`. The encoder reports, for each encoder
/// frame, the tone in that frame and in the previous frame, found from the mel features.
/// The joint network emits a token where a tone starts, with a fixed duration, and blank
/// otherwise. The prediction network output holds the last token and the number of tokens
/// it consumed, so tests can inspect the context that the recognizer carries.
final class FakeParakeetEngine: ParakeetInferenceEngine {
    let model: ParakeetModelDescription
    let forcedCommitModel: ParakeetModelDescription?
    let tokenDuration: Int
    let innerBlankDuration: Int?
    let blanksAfterSentenceEnd: Bool
    let blanksAfterSpeechStart: Bool
    let silenceToken: Int?
    private let bins: [Int: Int]
    private(set) var encodeCount = 0
    private(set) var forcedCommitEncodeCount = 0
    private(set) var predictCount = 0

    static let description = ParakeetModelDescription(
        windowSamples: 24 * 1280 + 640, melBinCount: 128, melFrameCount: 197,
        encoderFrameCount: 25, samplesPerEncoderFrame: 1280, encoderHiddenSize: 2,
        decoderHiddenSize: 2, decoderLayerCount: 1, vocabularySize: 10, durations: [0, 1, 2, 3, 4])

    /// A window twice as long, with 48 whole encoder frames.
    static let longDescription = description.withEncoderWindow(
        melFrameCount: 389, encoderFrameCount: 49)

    /// - Parameters:
    ///   - forcedCommitWindow: Whether the engine has the longer forced-commit window.
    ///   - innerBlankDuration: The blank duration at a frame where the tone of the last
    ///     emitted token continues, or `nil` for one frame. Greedy decoding skips these frames
    ///     when `tokenDuration` covers the tone; Parakeet Ultra predicted blanks of four
    ///     frames at such a frame.
    ///   - blanksAfterSentenceEnd: Whether a window in which the period's tone starts after
    ///     the first frame encodes everything after that tone as silence. Some window
    ///     placements made Parakeet emit only blanks after a sentence end on natural speech.
    ///   - blanksAfterSpeechStart: Whether a window that starts inside a tone encodes all its
    ///     audio as silence. Windows that started inside speech lost words far more often
    ///     than windows that started in a pause on natural speech.
    ///   - silenceToken: A token that the joint network emits with high probability at a
    ///     silent frame when the decoder context is the initial one, or `nil` for blank.
    ///     In a pause, Parakeet decoded text that the audio does not contain from the
    ///     initial context, such as "Продолжение следует" in German speech.
    init(
        tokenDuration: Int = 1, forcedCommitWindow: Bool = false, innerBlankDuration: Int? = nil,
        blanksAfterSentenceEnd: Bool = false, blanksAfterSpeechStart: Bool = false,
        silenceToken: Int? = nil
    ) {
        model = Self.description
        forcedCommitModel = forcedCommitWindow ? Self.longDescription : nil
        self.tokenDuration = tokenDuration
        self.innerBlankDuration = innerBlankDuration
        self.blanksAfterSentenceEnd = blanksAfterSentenceEnd
        self.blanksAfterSpeechStart = blanksAfterSpeechStart
        self.silenceToken = silenceToken
        // Learn the strongest mel bin of each tone from the real frontend.
        let frontend = ParakeetMelFrontend(windowSamples: model.windowSamples)
        var bins: [Int: Int] = [:]
        for token in ToneSpeech.tokens {
            var samples = [Float](repeating: 0, count: model.windowSamples)
            ToneSpeech.write(token: token, frames: 10..<13, into: &samples, offset: 0)
            var mel = [Float](repeating: 0, count: model.melBinCount * model.melFrameCount)
            samples.withUnsafeBufferPointer { samples in
                mel.withUnsafeMutableBufferPointer {
                    frontend.compute(samples, validSampleCount: samples.count, into: $0)
                }
            }
            bins[Self.strongestBin(mel, frame: 11 * 8 + 4, frames: model.melFrameCount).bin] = token
        }
        self.bins = bins
    }

    static func strongestBin(_ mel: [Float], frame: Int, frames: Int) -> (bin: Int, value: Float) {
        var best = (bin: 0, value: -Float.infinity)
        for bin in 0..<128 where mel[bin * frames + frame] > best.value {
            best = (bin, mel[bin * frames + frame])
        }
        return best
    }

    private func tone(_ mel: [Float], frame: Int, validMelFrames: Int, melFrames: Int) -> Int {
        let melFrame = frame * 8 + 4
        guard frame >= 0, melFrame < validMelFrames else { return 0 }
        let strongest = Self.strongestBin(mel, frame: melFrame, frames: melFrames)
        guard strongest.value > 0.5 else { return 0 }
        // A tone can peak one bin away from its calibration bin.
        return bins.filter { abs($0.key - strongest.bin) <= 1 }
            .min { abs($0.key - strongest.bin) < abs($1.key - strongest.bin) }?.value ?? 0
    }

    func encode(
        mel: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: inout ParakeetEncoderOutput
    ) throws {
        encodeCount += 1
        encode(mel, validFrameCount: validFrameCount, model: model, into: &output)
    }

    func encodeForcedCommit(
        mel: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: inout ParakeetEncoderOutput
    ) throws {
        forcedCommitEncodeCount += 1
        encode(mel, validFrameCount: validFrameCount, model: forcedCommitModel!, into: &output)
    }

    private func encode(
        _ mel: UnsafeBufferPointer<Float>, validFrameCount: Int,
        model: ParakeetModelDescription, into output: inout ParakeetEncoderOutput
    ) {
        precondition(mel.count == model.melBinCount * model.melFrameCount)
        var frames = validFrameCount
        for _ in 0..<3 { frames = frames == 0 ? 0 : (frames - 1) / 2 + 1 }
        output.validFrameCount = frames
        let mel = Array(mel)
        for frame in 0..<model.encoderFrameCount {
            output.features[frame * 2] = Float(
                tone(
                    mel, frame: frame, validMelFrames: validFrameCount,
                    melFrames: model.melFrameCount))
            output.features[frame * 2 + 1] = Float(
                tone(
                    mel, frame: frame - 1, validMelFrames: validFrameCount,
                    melFrames: model.melFrameCount))
        }
        if blanksAfterSpeechStart, output.features[0] != 0 {
            for index in output.features.indices { output.features[index] = 0 }
        }
        let period = Float(ToneSpeech.sentenceEnd)
        if blanksAfterSentenceEnd,
            let onset = (1..<model.encoderFrameCount).first(where: {
                output.features[$0 * 2] == period && output.features[$0 * 2 + 1] != period
            })
        {
            for frame in min(onset + 3, model.encoderFrameCount)..<model.encoderFrameCount {
                output.features[frame * 2] = 0
                output.features[frame * 2 + 1] = 0
            }
        }
    }

    func predict(token: Int, state: inout ParakeetDecoderState, output: inout [Float]) throws {
        predictCount += 1
        state.hidden[0] = token == model.blankID ? 0 : state.hidden[0] + 1
        state.cell[0] = Float(token)
        output = [Float(token), state.hidden[0]]
    }

    func joint(
        encoderFrame: UnsafeBufferPointer<Float>, decoderOutput: UnsafeBufferPointer<Float>
    ) throws -> ParakeetJointDecision {
        let current = Int(encoderFrame[0])
        let previous = Int(encoderFrame[1])
        if current != 0 && current != previous {
            return ParakeetJointDecision(token: current, probability: 0.9, duration: tokenDuration)
        }
        if let silenceToken, current == 0, Int(decoderOutput[0]) == model.blankID {
            return ParakeetJointDecision(token: silenceToken, probability: 0.95, duration: 1)
        }
        if let innerBlankDuration, current != 0, Int(decoderOutput[0]) == current {
            // A frame inside the duration of the token just emitted.
            return ParakeetJointDecision(
                token: model.blankID, probability: 0.8, duration: innerBlankDuration)
        }
        return ParakeetJointDecision(token: model.blankID, probability: 0.8, duration: 1)
    }
}

/// Builds tone audio from a token script.
enum ToneSpeech {
    static let tokens = Array(1...9)
    static let vocabulary = ParakeetVocabulary(pieces: [
        "<unk>", "\u{2581}one", "\u{2581}two", "\u{2581}th", "ree", ",", ".", "\u{2581}four",
        "\u{2581}five", "\u{2581}six",
    ])

    /// The token of the period.
    static let sentenceEnd = 6

    static func frequency(_ token: Int) -> Double { 300 + Double(token) * 650 }

    /// Writes a tone for `token` over encoder `frames` of the absolute timeline.
    static func write(token: Int, frames: Range<Int>, into samples: inout [Float], offset: Int) {
        for index in (frames.lowerBound * 1280)..<(frames.upperBound * 1280) {
            let local = index - offset
            guard samples.indices.contains(local) else { continue }
            samples[local] = Float(
                0.3 * sin(2 * Double.pi * frequency(token) * Double(index) / 16_000))
        }
    }

    /// Audio with one token every `spacing` frames, each three frames long, and the expected
    /// tokens with their onset frames.
    static func script(count: Int, spacing: Int = 5, lead: Int = 2, trailing: Int = 3) -> (
        samples: [Float], tokens: [(id: Int, frame: Int)]
    ) {
        let frames = lead + count * spacing + trailing
        var samples = [Float](repeating: 0, count: frames * 1280)
        var expected: [(Int, Int)] = []
        for index in 0..<count {
            // Neighbors always differ, so each segment has a clear onset.
            let token = tokens[(index * 4) % tokens.count]
            let onset = lead + index * spacing
            write(token: token, frames: onset..<(onset + 3), into: &samples, offset: 0)
            expected.append((token, onset))
        }
        return (samples, expected)
    }
}
