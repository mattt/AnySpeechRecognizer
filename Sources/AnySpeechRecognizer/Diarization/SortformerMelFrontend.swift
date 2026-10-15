import Accelerate
import Foundation

/// Computes Sortformer log-mel features incrementally from a 16 kHz mono stream.
///
/// The features follow NeMo's `AudioToMelSpectrogramPreprocessor` with Sortformer's
/// settings: pre-emphasis 0.97, a centered STFT with zero padding, a symmetric 400-sample
/// Hann window in a 512-point DFT, a 10 ms hop, a power spectrum, 128 Slaney mel bins,
/// and `log(x + 2^-24)`, without normalization or dither.
/// Frame `k` is centered on sample `k * 160` and reads samples `k * 160 - 200 ..< k * 160 + 200`,
/// so it is computed once those samples arrive.
/// At the end of a stream, NeMo's frame count is `sampleCount / 160`, and samples after
/// the end are zero.
/// One instance owns scratch buffers and serves one owner.
final class SortformerMelFrontend {
    static let hopLength = 160
    static let windowLength = 400
    static let fftLength = 512
    static let melBinCount = 128
    static let preemphasis: Float = 0.97
    static let logGuard: Float = 0x1p-24

    private static let binCount = fftLength / 2 + 1
    private static let halfWindow = windowLength / 2
    private let setup: vDSP_DFT_Setup
    private let window: [Float]
    private let filterbank: [Float]
    /// Pre-emphasized samples from absolute index `sampleOffset`.
    private var emphasized: [Float] = []
    private var sampleOffset = 0
    private var previousSample: Float?
    /// Log-mel frames from absolute frame `featureOffset`, time-major.
    private var features: [Float] = []
    private(set) var featureOffset = 0
    private(set) var sampleCount = 0
    private(set) var finished = false
    private var frame: [Float]
    private var even: [Float]
    private var odd: [Float]
    private var real: [Float]
    private var imaginary: [Float]
    private var power: [Float]
    private var mel: [Float]

    init() {
        guard let setup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(Self.fftLength), .FORWARD)
        else { fatalError("Accelerate cannot create a \(Self.fftLength)-point real DFT.") }
        self.setup = setup
        window = (0..<Self.windowLength).map { index in
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double(index) / Double(Self.windowLength - 1)))
        }
        filterbank = ParakeetMelFrontend.slaneyFilterbank()
        frame = [Float](repeating: 0, count: Self.fftLength)
        even = [Float](repeating: 0, count: Self.fftLength / 2)
        odd = [Float](repeating: 0, count: Self.fftLength / 2)
        real = [Float](repeating: 0, count: Self.fftLength / 2)
        imaginary = [Float](repeating: 0, count: Self.fftLength / 2)
        power = [Float](repeating: 0, count: Self.binCount)
        mel = [Float](repeating: 0, count: Self.melBinCount)
    }

    deinit { vDSP_DFT_DestroySetup(setup) }

    /// The number of frames computed so far, including discarded frames.
    var frameCount: Int { featureOffset + features.count / Self.melBinCount }

    /// The total number of frames of a finished stream of `sampleCount` samples.
    static func finalFrameCount(sampleCount: Int) -> Int { sampleCount / hopLength }

    /// Adds samples and computes every frame whose samples have arrived.
    func append(_ samples: UnsafeBufferPointer<Float>) {
        precondition(!finished, "Append samples only before the stream finishes.")
        guard !samples.isEmpty else { return }
        emphasized.reserveCapacity(emphasized.count + samples.count)
        var previous = previousSample
        for sample in samples {
            emphasized.append(previous.map { sample - Self.preemphasis * $0 } ?? sample)
            previous = sample
        }
        previousSample = previous
        sampleCount += samples.count
        while (frameCount * Self.hopLength) + Self.halfWindow <= sampleCount {
            computeFrame(frameCount)
        }
    }

    /// Computes the remaining frames of the stream with zeros after its last sample.
    func finish() {
        guard !finished else { return }
        finished = true
        let total = Self.finalFrameCount(sampleCount: sampleCount)
        while frameCount < total { computeFrame(frameCount) }
    }

    /// Copies frames into `output`, time-major, `melBinCount` values per frame.
    func copyFrames(_ frames: Range<Int>, into output: UnsafeMutablePointer<Float>) {
        precondition(frames.lowerBound >= featureOffset && frames.upperBound <= frameCount)
        features.withUnsafeBufferPointer { features in
            output.update(
                from: features.baseAddress! + (frames.lowerBound - featureOffset)
                    * Self.melBinCount,
                count: frames.count * Self.melBinCount)
        }
    }

    /// Releases frames before `frame` and the samples that no later frame reads.
    func discard(before frame: Int) {
        let frames = min(frame, frameCount) - featureOffset
        if frames > 0 {
            features.removeFirst(frames * Self.melBinCount)
            featureOffset += frames
        }
        let firstNeeded = max(0, frameCount * Self.hopLength - Self.halfWindow)
        let samples = min(firstNeeded - sampleOffset, emphasized.count)
        if samples > 4096 {
            emphasized.removeFirst(samples)
            sampleOffset += samples
        }
    }

    private func computeFrame(_ index: Int) {
        let start = index * Self.hopLength - Self.halfWindow
        let offset = (Self.fftLength - Self.windowLength) / 2
        frame.withUnsafeMutableBufferPointer { frame in
            frame.update(repeating: 0)
            for position in 0..<Self.windowLength {
                let sample = start + position
                guard sample >= 0, sample < sampleCount else { continue }
                frame[offset + position] = emphasized[sample - sampleOffset] * window[position]
            }
        }
        let half = Self.fftLength / 2
        frame.withUnsafeBufferPointer { frame in
            even.withUnsafeMutableBufferPointer { even in
                odd.withUnsafeMutableBufferPointer { odd in
                    var split = DSPSplitComplex(realp: even.baseAddress!, imagp: odd.baseAddress!)
                    frame.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
                        vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
                    }
                }
            }
        }
        vDSP_DFT_Execute(setup, even, odd, &real, &imaginary)
        // The real DFT packs the Nyquist term in the first imaginary value
        // and scales every term by 2.
        power[0] = real[0] * real[0] / 4
        power[half] = imaginary[0] * imaginary[0] / 4
        var bin = 1
        while bin < half {
            power[bin] = (real[bin] * real[bin] + imaginary[bin] * imaginary[bin]) / 4
            bin += 1
        }
        filterbank.withUnsafeBufferPointer { bank in
            power.withUnsafeBufferPointer { power in
                mel.withUnsafeMutableBufferPointer { mel in
                    vDSP_mmul(
                        bank.baseAddress!, 1, power.baseAddress!, 1, mel.baseAddress!, 1,
                        vDSP_Length(Self.melBinCount), 1, vDSP_Length(Self.binCount))
                }
            }
        }
        var guardValue = Self.logGuard
        var count = Int32(Self.melBinCount)
        mel.withUnsafeMutableBufferPointer { mel in
            vDSP_vsadd(mel.baseAddress!, 1, &guardValue, mel.baseAddress!, 1, vDSP_Length(count))
            vvlogf(mel.baseAddress!, mel.baseAddress!, &count)
        }
        features += mel
    }
}
