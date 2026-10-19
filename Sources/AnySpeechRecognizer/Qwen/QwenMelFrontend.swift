import Accelerate
import Foundation

/// Computes Qwen3-ASR log-mel features with Accelerate.
///
/// The features follow transformers' `Qwen3ASRFeatureExtractor` with the processor's settings,
/// which are Whisper's: centered frames with the signal reflected at its ends,
/// a periodic 400-sample Hann window, a 10 ms hop, a power spectrum, 128 Slaney mel bins
/// from 0 to 8 kHz, `log10(max(x, 1e-10))`, values below the maximum minus 8 raised to it,
/// and `(x + 4) / 4`. The maximum is over the whole input, so a segment's features depend
/// on all of its audio. `N` samples give `floor(N / 160)` frames, because the extractor
/// drops the last centered frame. Dither is off.
///
/// vDSP's DFT does not support 400 points, so the frontend computes the 201 frequency bins
/// as one matrix product with a windowed cosine and sine basis.
/// One instance serves one owner.
final class QwenMelFrontend {
    static let sampleRate = 16_000
    static let hopLength = 160
    static let fftLength = 400
    static let melBinCount = 128
    static let binCount = fftLength / 2 + 1

    /// The windowed DFT basis, `[fftLength × 2 binCount]`: cosines, then negated sines.
    private let basis: [Float]
    /// The Slaney filterbank, transposed for a frame-major product, `[binCount × melBinCount]`.
    private let filterbank: [Float]

    init() {
        let length = Self.fftLength
        let bins = Self.binCount
        var basis = [Float](repeating: 0, count: length * 2 * bins)
        for sample in 0..<length {
            // torch.hann_window is periodic by default.
            let window = 0.5 - 0.5 * cos(2 * Double.pi * Double(sample) / Double(length))
            for bin in 0..<bins {
                let angle = 2 * Double.pi * Double(sample * bin % length) / Double(length)
                basis[sample * 2 * bins + bin] = Float(window * cos(angle))
                basis[sample * 2 * bins + bins + bin] = Float(-window * sin(angle))
            }
        }
        self.basis = basis
        let bank = Self.slaneyFilterbank()
        var transposed = [Float](repeating: 0, count: bank.count)
        vDSP_mtrans(bank, 1, &transposed, 1, vDSP_Length(bins), vDSP_Length(Self.melBinCount))
        filterbank = transposed
    }

    /// The number of feature frames for a sample count.
    static func frameCount(sampleCount: Int) -> Int { max(0, sampleCount / hopLength) }

    /// Returns features in mel-major order, `[melBinCount × frameCount(sampleCount:)]`.
    /// - Parameter samples: 16 kHz audio of more than 200 samples, for the reflection padding.
    func compute(_ samples: UnsafeBufferPointer<Float>) -> [Float] {
        let count = samples.count
        let frames = Self.frameCount(sampleCount: count)
        let pad = Self.fftLength / 2
        guard frames > 0, count > pad, let source = samples.baseAddress else { return [] }
        // Reflect `pad` samples at each end without repeating the edge sample, as
        // `torch.stft(center=True)` does.
        var padded = [Float](repeating: 0, count: count + 2 * pad)
        padded.withUnsafeMutableBufferPointer { padded in
            let base = padded.baseAddress!
            (base + pad).update(from: source, count: count)
            for offset in 1...pad {
                base[pad - offset] = source[offset]
                base[pad + count - 1 + offset] = source[count - 1 - offset]
            }
        }
        let length = Self.fftLength
        let bins = Self.binCount
        var framed = [Float](repeating: 0, count: frames * length)
        padded.withUnsafeBufferPointer { padded in
            framed.withUnsafeMutableBufferPointer { framed in
                for frame in 0..<frames {
                    (framed.baseAddress! + frame * length).update(
                        from: padded.baseAddress! + frame * Self.hopLength, count: length)
                }
            }
        }
        var spectrum = [Float](repeating: 0, count: frames * 2 * bins)
        vDSP_mmul(
            framed, 1, basis, 1, &spectrum, 1, vDSP_Length(frames), vDSP_Length(2 * bins),
            vDSP_Length(length))
        var power = [Float](repeating: 0, count: frames * bins)
        spectrum.withUnsafeBufferPointer { spectrum in
            power.withUnsafeMutableBufferPointer { power in
                for frame in 0..<frames {
                    let row = spectrum.baseAddress! + frame * 2 * bins
                    let real = row
                    let imaginary = row + bins
                    let output = power.baseAddress! + frame * bins
                    vDSP_vmma(
                        real, 1, real, 1, imaginary, 1, imaginary, 1, output, 1, vDSP_Length(bins))
                }
            }
        }
        var mel = [Float](repeating: 0, count: frames * Self.melBinCount)
        vDSP_mmul(
            power, 1, filterbank, 1, &mel, 1, vDSP_Length(frames), vDSP_Length(Self.melBinCount),
            vDSP_Length(bins))
        var floor: Float = 1e-10
        var ceiling = Float.greatestFiniteMagnitude
        vDSP_vclip(mel, 1, &floor, &ceiling, &mel, 1, vDSP_Length(mel.count))
        var total = Int32(mel.count)
        mel.withUnsafeMutableBufferPointer { vvlog10f($0.baseAddress!, $0.baseAddress!, &total) }
        var maximum: Float = 0
        vDSP_maxv(mel, 1, &maximum, vDSP_Length(mel.count))
        var lower = maximum - 8
        vDSP_vclip(mel, 1, &lower, &ceiling, &mel, 1, vDSP_Length(mel.count))
        var scale: Float = 0.25
        var offset: Float = 1
        vDSP_vsmsa(mel, 1, &scale, &offset, &mel, 1, vDSP_Length(mel.count))
        var features = [Float](repeating: 0, count: mel.count)
        vDSP_mtrans(mel, 1, &features, 1, vDSP_Length(Self.melBinCount), vDSP_Length(frames))
        return features
    }

    /// Builds the Slaney-scale, Slaney-normalized filterbank of transformers' `mel_filter_bank`
    /// for 201 bins from 0 to 8 kHz, `[melBinCount × binCount]`.
    static func slaneyFilterbank() -> [Float] {
        func mel(_ hertz: Double) -> Double {
            hertz < 1000 ? hertz * 3 / 200 : 15 + log(hertz / 1000) * 27 / log(6.4)
        }
        func hertz(_ mel: Double) -> Double {
            mel < 15 ? mel * 200 / 3 : 1000 * exp(log(6.4) * (mel - 15) / 27)
        }
        let maximum = mel(Double(sampleRate / 2))
        let edges = (0..<(melBinCount + 2)).map {
            hertz(maximum * Double($0) / Double(melBinCount + 1))
        }
        var weights = [Float](repeating: 0, count: melBinCount * binCount)
        for bin in 0..<melBinCount {
            let (lower, center, upper) = (edges[bin], edges[bin + 1], edges[bin + 2])
            let scale = 2 / (upper - lower)
            for frequencyBin in 0..<binCount {
                let frequency = Double(frequencyBin) * Double(sampleRate / 2) / Double(binCount - 1)
                let rising = (frequency - lower) / (center - lower)
                let falling = (upper - frequency) / (upper - center)
                weights[bin * binCount + frequencyBin] = Float(max(0, min(rising, falling)) * scale)
            }
        }
        return weights
    }
}
