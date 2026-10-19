import Foundation

@testable import AnySpeechRecognizer

/// A small cache-aware Nemotron model whose "speech" is the pure tones of ``ToneSpeech``.
///
/// Each encoder frame reports the tone in its audio and the tone of the frame before it.
/// The tone before the first frame of a chunk comes from the engine's cache, the last frame
/// of the previous step that updated the caches, so a chunk that is encoded twice, or a step
/// that updates the caches when it should not, changes the frames. The joint network emits a
/// token at the first frame of a tone, once, and blank otherwise, as an RNN-T model does with
/// duration 0.
final class FakeNemotronEngine: NemotronInferenceEngine {
    let model = NemotronModel.transducerForTests
    let chunkFrames: Int
    private let bins: [Int: Int]
    /// The tone of the last frame of the last step that updated the caches.
    private var cache = 0
    private(set) var steps: [(validFrameCount: Int, prompt: Int, updatesCaches: Bool)] = []
    private(set) var resets = 0

    init(chunkFrames: Int) {
        self.chunkFrames = chunkFrames
        // Learn the strongest mel bin of each tone from the real frontend.
        var bins: [Int: Int] = [:]
        for token in ToneSpeech.tokens {
            var samples = [Float](repeating: 0, count: 16 * 1280)
            ToneSpeech.write(token: token, frames: 4..<8, into: &samples, offset: 0)
            let frontend = NemotronMelFrontend()
            frontend.append(samples)
            let frame = 5 * 8 + 4
            let row = frontend.frames[(frame * 128)..<((frame + 1) * 128)]
            bins[row.indices.max { row[$0] < row[$1] }! - frame * 128] = token
        }
        self.bins = bins
    }

    func resetCaches() {
        resets += 1
        cache = 0
    }

    private func tone(_ mel: UnsafeBufferPointer<Float>, column: Int, width: Int) -> Int {
        var best = (bin: 0, value: -Float.infinity)
        for bin in 0..<128 where mel[bin * width + column] > best.value {
            best = (bin, mel[bin * width + column])
        }
        // Silence is the log guard, about -16.6.
        guard best.value > -5 else { return 0 }
        return bins.filter { abs($0.key - best.bin) <= 1 }
            .min { abs($0.key - best.bin) < abs($1.key - best.bin) }?.value ?? 0
    }

    func encode(
        mel: UnsafeBufferPointer<Float>, validFrameCount: Int, prompt: Int, updatesCaches: Bool,
        into output: inout ParakeetEncoderOutput
    ) throws {
        precondition(mel.count == 128 * inputFrames)
        steps.append((validFrameCount, prompt, updatesCaches))
        let melFrames = validFrameCount - NemotronModel.preEncodeFrames
        let valid = min(chunkFrames, (melFrames + 7) / 8)
        if output.frameCount != chunkFrames || output.hiddenSize != model.encoderHiddenSize {
            output = ParakeetEncoderOutput(
                hiddenSize: model.encoderHiddenSize, frameCount: chunkFrames)
        }
        var previous = cache
        for frame in 0..<chunkFrames {
            // The last mel frame of the encoder frame, which reads 12.5 ms past its audio.
            let column = NemotronModel.preEncodeFrames + frame * 8 + 7
            let current =
                frame < valid && column < validFrameCount
                ? tone(mel, column: column, width: inputFrames) : 0
            output.features[frame * 2] = Float(current)
            output.features[frame * 2 + 1] = Float(previous)
            previous = current
        }
        output.validFrameCount = valid
        if updatesCaches { cache = Int(output.features[(valid - 1) * 2]) }
    }

    func predict(token: Int, state: inout ParakeetDecoderState, output: inout [Float]) throws {
        state.hidden[0] = token == model.blankID ? 0 : state.hidden[0] + 1
        output = [Float(token), state.hidden[0]]
    }

    func joint(
        encoderFrame: UnsafeBufferPointer<Float>, decoderOutput: UnsafeBufferPointer<Float>
    ) throws -> ParakeetJointDecision {
        let current = Int(encoderFrame[0])
        let previous = Int(encoderFrame[1])
        // A token stays at its frame, so the next decision there sees it as the last token.
        if current != 0, current != previous, Int(decoderOutput[0]) != current {
            return ParakeetJointDecision(token: current, probability: 0.9, duration: 0)
        }
        return ParakeetJointDecision(token: model.blankID, probability: 0.8, duration: 0)
    }
}

extension NemotronModel {
    /// The tone vocabulary of ``ToneSpeech`` with two-value encoder frames.
    static let transducerForTests = ParakeetModelDescription(
        windowSamples: 0, melBinCount: 128, melFrameCount: 0, encoderFrameCount: 0,
        samplesPerEncoderFrame: 1280, encoderHiddenSize: 2, decoderHiddenSize: 2,
        decoderLayerCount: 1, vocabularySize: 10, durations: [0])
}
