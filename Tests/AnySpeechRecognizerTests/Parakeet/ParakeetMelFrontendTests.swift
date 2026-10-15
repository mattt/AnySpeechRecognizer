import Foundation
import Testing

@testable import AnySpeechRecognizer

struct ParakeetMelFrontendTests {
    let frontend = ParakeetMelFrontend()

    func features(_ samples: [Float], valid: Int) -> (mel: [Float], frames: Int) {
        var mel = [Float](repeating: .nan, count: 128 * 1501)
        let frames = samples.withUnsafeBufferPointer { samples in
            mel.withUnsafeMutableBufferPointer {
                frontend.compute(samples, validSampleCount: valid, into: $0)
            }
        }
        return (mel, frames)
    }

    func tone(_ frequency: Double, count: Int, amplitude: Float = 0.3) -> [Float] {
        (0..<count).map { amplitude * Float(sin(2 * Double.pi * frequency * Double($0) / 16_000)) }
    }

    func strongestBin(_ mel: [Float], frame: Int) -> Int {
        (0..<128).max { mel[$0 * 1501 + frame] < mel[$1 * 1501 + frame] }!
    }

    /// The encoder input keeps `N / 160 + 1` frames; NeMo 2.5 and later count `N / 160` as valid.
    @Test func reportsNeMoFrameCounts() {
        let frontend = self.frontend
        #expect(frontend.frameCount == 1501)
        #expect(frontend.validFrameCount(sampleCount: 240_000) == 1500)
        #expect(frontend.validFrameCount(sampleCount: 100_003) == 625)
        #expect(frontend.validFrameCount(sampleCount: 160) == 1)
        #expect(frontend.validFrameCount(sampleCount: 159) == 0)
        #expect(frontend.validFrameCount(sampleCount: 0) == 0)
        #expect(ParakeetMelFrontend(windowSamples: 128_000).frameCount == 801)
    }

    /// The generated signal of `Tools/ParakeetConversion/mel_reference.py`.
    func referenceSignal() -> [Float] {
        var state: UInt32 = 1
        return (0..<ParakeetMelReference.sampleCount).map { index in
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

    /// Edge frames depend on the padding and on the pre-emphasis after the valid samples.
    @Test(arguments: [ParakeetMelReference.roundedUp, ParakeetMelReference.exact])
    func matchesNeMo(reference: ParakeetMelReference.Case) {
        let frontend = ParakeetMelFrontend(windowSamples: ParakeetMelReference.windowSamples)
        var mel = [Float](repeating: .nan, count: 128 * frontend.frameCount)
        let frames = referenceSignal().withUnsafeBufferPointer { samples in
            mel.withUnsafeMutableBufferPointer {
                frontend.compute(samples, validSampleCount: reference.validSampleCount, into: $0)
            }
        }
        #expect(frames == reference.validFrameCount)
        var maximum: Float = 0
        for (frameIndex, frame) in reference.frames.enumerated() {
            for (binIndex, bin) in ParakeetMelReference.bins.enumerated() {
                let difference = abs(
                    mel[bin * frontend.frameCount + frame] - reference.values[frameIndex][binIndex])
                maximum = max(maximum, difference)
            }
        }
        #expect(maximum < 1e-3, "maximum difference \(maximum)")
        for bin in 0..<128 {
            let row = (bin * frontend.frameCount)..<((bin + 1) * frontend.frameCount)
            #expect(mel[row].dropFirst(frames).allSatisfy { $0 == 0 })
        }
    }

    @Test func silenceProducesZeroFeatures() {
        let (mel, frames) = features([Float](repeating: 0, count: 50_000), valid: 50_000)
        #expect(frames == 312)
        #expect(mel.allSatisfy { $0 == 0 })
        #expect(features([], valid: 0).mel.allSatisfy { $0 == 0 })
    }

    @Test func normalizesValidFramesAndZeroesTheRest() {
        // A tone for one second, then silence.
        var samples = tone(1_000, count: 16_000)
        samples += [Float](repeating: 0, count: 84_003)
        let (mel, frames) = features(samples, valid: samples.count)
        #expect(frames == 625)
        #expect(mel.allSatisfy { $0.isFinite })
        for bin in [10, 40, 90] {
            let row = mel[(bin * 1501)..<(bin * 1501 + frames)]
            let mean = row.reduce(0, +) / Float(frames)
            let variance = row.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(frames - 1)
            #expect(abs(mean) < 1e-3)
            #expect(abs(variance.squareRoot() - 1) < 1e-3)
            #expect(mel[(bin * 1501 + frames)..<((bin + 1) * 1501)].allSatisfy { $0 == 0 })
        }
    }

    @Test func placesTonesInIncreasingMelBins() {
        var bins: [Int] = []
        for frequency in [250.0, 1_000, 4_000, 7_000] {
            var samples = tone(frequency, count: 32_000)
            samples += [Float](repeating: 0, count: 32_000)
            let (mel, _) = features(samples, valid: samples.count)
            let bin = strongestBin(mel, frame: 100)
            #expect(strongestBin(mel, frame: 50) == bin)
            bins.append(bin)
        }
        #expect(bins == bins.sorted() && Set(bins).count == bins.count)
        // Slaney bins are linear below 1 kHz, about 23 Hz wide at this resolution.
        #expect((9...12).contains(bins[0]))
    }

    /// Live windows reuse cached frames; the result must equal computing every frame.
    @Test(arguments: [240_000, 128_000])
    func reusedFramesMatchTheFullComputation(windowSamples: Int) {
        // Noisy speech-like audio: a few tones with a slow envelope and pseudo-random noise.
        var generator = SystemRandomNumberGenerator()
        let audio = (0..<(windowSamples * 3)).map { index -> Float in
            let time = Double(index) / 16_000
            let envelope = Float(0.5 + 0.5 * sin(2 * Double.pi * 0.7 * time))
            let voiced = Float(
                sin(2 * Double.pi * 220 * time) + 0.5 * sin(2 * Double.pi * 1_300 * time))
            return envelope * 0.2 * voiced + Float.random(in: -0.02...0.02, using: &generator)
        }
        let reusing = ParakeetMelFrontend(windowSamples: windowSamples)
        let full = ParakeetMelFrontend(windowSamples: windowSamples)
        let size = 128 * reusing.frameCount
        // Sliding windows that end inside the audio, a window that reaches the end, windows
        // that start earlier than the cache, and a jump that leaves a gap.
        var windows: [(start: Int, count: Int)] = []
        for step in 0..<12 {
            windows.append((step * 15_360, min(windowSamples, 37 * 1_280 + step * 15_360 + 1_000)))
        }
        windows += [(11 * 15_360, windowSamples), (5 * 15_360, windowSamples - 3_000)]
        windows += [
            (windowSamples + 40 * 1_280, windowSamples / 2 + 17),
            (windowSamples * 2, windowSamples),
        ]
        for (start, count) in windows {
            var expected = [Float](repeating: .nan, count: size)
            var actual = [Float](repeating: .nan, count: size)
            let valid = min(windowSamples, (count + 1_279) / 1_280 * 1_280)
            audio.withUnsafeBufferPointer { audio in
                let slice = UnsafeBufferPointer(rebasing: audio[start..<(start + count)])
                _ = expected.withUnsafeMutableBufferPointer {
                    full.compute(slice, validSampleCount: valid, into: $0)
                }
                _ = actual.withUnsafeMutableBufferPointer {
                    reusing.compute(slice, validSampleCount: valid, startHop: start / 160, into: $0)
                }
            }
            #expect(actual == expected, "window at \(start) with \(count) samples")
        }
    }

    @Test func treatsMissingSamplesAsZeroPadding() {
        let samples = tone(500, count: 20_000)
        let padded = samples + [Float](repeating: 0, count: 240_000 - samples.count)
        #expect(features(samples, valid: 21_120).mel == features(padded, valid: 21_120).mel)
        #expect(features(samples, valid: 21_120).mel != features(samples, valid: 20_000).mel)
    }
}
