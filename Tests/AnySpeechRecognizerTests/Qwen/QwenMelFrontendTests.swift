import Foundation
import Testing

@testable import AnySpeechRecognizer

struct QwenMelFrontendTests {
    let frontend = QwenMelFrontend()

    /// The extractor drops the last centered frame: `N` samples give `floor(N / 160)` frames.
    @Test func reportsExtractorFrameCounts() {
        #expect(QwenMelFrontend.frameCount(sampleCount: 16_517) == 103)
        #expect(QwenMelFrontend.frameCount(sampleCount: 16_000) == 100)
        #expect(QwenMelFrontend.frameCount(sampleCount: 159) == 0)
        let samples = [Float](repeating: 0.1, count: 8_000)
        #expect(samples.withUnsafeBufferPointer { frontend.compute($0) }.count == 128 * 50)
    }

    /// The generated signal of `Tools/QwenConversion/mel_reference.py`.
    func referenceSignal(count: Int) -> [Float] {
        var state: UInt32 = 1
        return (0..<count).map { index in
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

    /// The short case is padded with zeros to the 8,000-sample minimum, as the recognizer
    /// and the feature extractor pad it. Edge frames depend on the reflection padding.
    @Test(arguments: [QwenMelReference.long, QwenMelReference.short])
    func matchesTransformers(reference: QwenMelReference.Case) {
        var samples = referenceSignal(count: reference.sampleCount)
        samples += [Float](repeating: 0, count: max(0, 8_000 - samples.count))
        let mel = samples.withUnsafeBufferPointer { frontend.compute($0) }
        let frames = QwenMelFrontend.frameCount(sampleCount: samples.count)
        #expect(frames == reference.frameCount)
        var largest: Float = 0
        for (row, frame) in reference.frames.enumerated() {
            for (column, bin) in QwenMelReference.bins.enumerated() {
                largest = max(
                    largest, abs(mel[bin * frames + frame] - reference.values[row][column]))
            }
        }
        #expect(largest < 1e-3, "largest difference \(largest)")
    }

    /// With 128 mel bins over 201 frequency bins, every triangular filter still covers at least
    /// one frequency bin, and no weight is negative.
    @Test func filterbankCoversEveryMelBin() {
        let bank = QwenMelFrontend.slaneyFilterbank()
        for bin in 0..<QwenMelFrontend.melBinCount {
            let row = bank[
                (bin * QwenMelFrontend.binCount)..<((bin + 1) * QwenMelFrontend.binCount)]
            #expect(row.contains { $0 > 0 })
            #expect(row.allSatisfy { $0 >= 0 })
        }
    }
}
