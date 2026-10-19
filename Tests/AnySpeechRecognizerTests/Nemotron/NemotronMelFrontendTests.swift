import Foundation
import Testing

@testable import AnySpeechRecognizer

struct NemotronMelFrontendTests {
    /// The generated signal of `Tools/NemotronConversion/mel_reference.py`.
    static func referenceSignal() -> [Float] {
        var state: UInt32 = 1
        return (0..<NemotronMelReference.sampleCount).map { index in
            state = state &* 1_664_525 &+ 1_013_904_223
            let noise = Double(state >> 8) / 16_777_216 - 0.5
            let time = Double(index) / 16_000
            let envelope = 0.5 + 0.5 * sin(2 * Double.pi * 1.5 * time)
            let voiced =
                sin(2 * Double.pi * 220 * time) + 0.5 * sin(2 * Double.pi * 1_300 * time)
                + 0.25 * sin(2 * Double.pi * 3_100 * time)
            return Float(0.2 * envelope * voiced + 0.02 * noise)
        }
    }

    /// All frames of a frontend, frame-major.
    static func features(_ samples: [Float], blocks: [Int], finish: Bool = true) -> (
        frames: [Float], count: Int
    ) {
        let frontend = NemotronMelFrontend()
        var offset = 0
        var index = 0
        while offset < samples.count {
            let count = min(blocks[index % blocks.count], samples.count - offset)
            samples[offset..<(offset + count)].withUnsafeBufferPointer { frontend.append($0) }
            offset += count
            index += 1
        }
        if finish { frontend.finish() }
        return (frontend.frames, frontend.frameCount)
    }

    /// NeMo's preprocessor does not normalize Nemotron's features, so the values match directly,
    /// including the last frames, which read zeros after the audio.
    @Test func matchesNeMo() {
        let (frames, count) = Self.features(Self.referenceSignal(), blocks: [1600])
        #expect(count == NemotronMelReference.frameCount)
        var maximum: Float = 0
        for (frameIndex, frame) in NemotronMelReference.frames.enumerated() {
            for (binIndex, bin) in NemotronMelReference.bins.enumerated() {
                let difference = abs(
                    frames[frame * 128 + bin] - NemotronMelReference.values[frameIndex][binIndex])
                maximum = max(maximum, difference)
            }
        }
        #expect(maximum < 1e-3, "maximum difference \(maximum)")
    }

    /// The largest difference between two sets of frames.
    ///
    /// Accelerate can round the last bit differently for the same values at different buffer
    /// offsets; the differences are about 1e-6.
    static func maximumDifference(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { max($0, abs($1.0 - $1.1)) }
    }

    /// A frame depends only on its own audio, so the block size does not change its values.
    @Test func blockSizeDoesNotChangeFrames() {
        let signal = Self.referenceSignal()
        let whole = Self.features(signal, blocks: [signal.count])
        for blocks in [[1600], [1], [37, 400, 3], [161]] {
            let streamed = Self.features(signal, blocks: blocks)
            #expect(streamed.count == whole.count)
            #expect(
                Self.maximumDifference(streamed.frames, whole.frames) < 1e-4, "blocks \(blocks)")
        }
    }

    /// Frame `k` is computed when samples through `160 k + 199` have arrived; at the end of
    /// input, `N` samples have `N / 160` frames.
    @Test func computesFramesAsTheirAudioArrives() {
        let frontend = NemotronMelFrontend()
        frontend.append([Float](repeating: 0.1, count: 199))
        #expect(frontend.frameCount == 0)
        frontend.append([0.1])
        #expect(frontend.frameCount == 1)
        frontend.append([Float](repeating: 0.1, count: 159))
        #expect(frontend.frameCount == 1)
        frontend.append([0.1])
        #expect(frontend.frameCount == 2)
        frontend.append([Float](repeating: 0.1, count: 1_000))
        #expect(frontend.frameCount == (1_360 - 200) / 160 + 1)
        frontend.finish()
        #expect(frontend.frameCount == 1_360 / 160)
    }

    /// Provisional end frames equal the frames of a finished recording, and later audio
    /// replaces them.
    @Test func provisionalEndFramesAreRemoved() {
        let signal = Self.referenceSignal()
        let half = Array(signal[..<8_000])
        let finished = Self.features(half, blocks: [half.count])
        let frontend = NemotronMelFrontend()
        frontend.append(half)
        let count = frontend.frameCount
        let provisional = frontend.withProvisionalEnd { (frontend.frameCount, frontend.frames) }
        #expect(provisional.0 == finished.count)
        #expect(Self.maximumDifference(provisional.1, finished.frames) < 1e-4)
        #expect(frontend.frameCount == count)
        frontend.append(Array(signal[8_000...]))
        frontend.finish()
        let whole = Self.features(signal, blocks: [signal.count])
        #expect(frontend.frameCount == whole.count)
        #expect(Self.maximumDifference(frontend.frames, whole.frames) < 1e-4)
    }

    @Test func silenceProducesTheLogGuard() {
        let frontend = NemotronMelFrontend()
        frontend.append([Float](repeating: 0, count: 3_200))
        frontend.finish()
        #expect(frontend.frameCount == 20)
        #expect(frontend.frames.allSatisfy { abs($0 - log(Float(0x1p-24))) < 1e-4 })
    }

    /// Discarding keeps later frames and the audio they read.
    @Test func discardingKeepsLaterFrames() {
        let signal = Self.referenceSignal()
        let whole = Self.features(signal, blocks: [signal.count])
        let frontend = NemotronMelFrontend()
        var offset = 0
        var copied = [Float](repeating: .nan, count: 128 * whole.count)
        var next = 0
        while offset < signal.count {
            let count = min(500, signal.count - offset)
            signal[offset..<(offset + count)].withUnsafeBufferPointer { frontend.append($0) }
            offset += count
            if offset >= signal.count { frontend.finish() }
            // Copy new frames into a mel-major array, then discard them.
            copied.withUnsafeMutableBufferPointer { copied in
                frontend.copy(
                    next..<frontend.frameCount, into: copied, width: whole.count, column: next)
            }
            next = frontend.frameCount
            frontend.discard(before: next)
        }
        #expect(frontend.firstRetainedFrame == whole.count)
        var transposed = [Float](repeating: 0, count: copied.count)
        for frame in 0..<whole.count {
            for bin in 0..<128 {
                transposed[frame * 128 + bin] = copied[bin * whole.count + frame]
            }
        }
        #expect(Self.maximumDifference(transposed, whole.frames) < 1e-4)
        #expect(!transposed.contains { $0.isNaN })
    }

    /// Frames before the recording are zero in an encoder input.
    @Test func copiesZerosBeforeTheRecording() {
        let frontend = NemotronMelFrontend()
        frontend.append(Self.referenceSignal())
        var input = [Float](repeating: .nan, count: 128 * 4)
        input.withUnsafeMutableBufferPointer {
            frontend.copy(-2..<2, into: $0, width: 4, column: 0)
        }
        for bin in 0..<128 {
            #expect(input[bin * 4] == 0 && input[bin * 4 + 1] == 0)
            #expect(input[bin * 4 + 2] == frontend.frames[bin])
            #expect(input[bin * 4 + 3] == frontend.frames[128 + bin])
        }
    }
}
