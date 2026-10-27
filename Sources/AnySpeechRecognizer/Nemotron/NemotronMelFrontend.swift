import Accelerate
import Foundation

/// Computes Nemotron's log-mel features incrementally from 16 kHz mono audio with Accelerate.
///
/// The features are NeMo's `AudioToMelSpectrogramPreprocessor` with the Parakeet settings of
/// ``ParakeetMelFrontend`` (NeMo 2.5 and later: pre-emphasis 0.97, zero padding of `n_fft / 2`
/// at the ends, a symmetric 400-sample Hann window in a 512-point DFT, a 10 ms hop, 128 Slaney
/// mel bins, and `log(x + 2^-24)`), but without per-feature normalization: Nemotron's
/// preprocessor sets `normalize: NA`. A frame therefore depends only on its own 25 ms of audio,
/// and the frontend computes each frame once, as soon as the audio after its center arrives.
/// The frames equal those of NeMo's preprocessor over the whole recording.
///
/// Frame `k` reads samples `160 k - 200 ..< 160 k + 200`, with zeros before the first sample.
/// At the end of input, a recording of `N` samples has `floor(N / 160)` frames, and frames
/// that read past the last sample read zeros, as NeMo's preprocessor does.
/// One instance owns scratch buffers and serves one owner.
final class NemotronMelFrontend {
    static let hopLength = ParakeetMelFrontend.hopLength
    static let windowLength = ParakeetMelFrontend.windowLength
    static let fftLength = ParakeetMelFrontend.fftLength
    static let melBinCount = ParakeetMelFrontend.melBinCount
    private static let binCount = fftLength / 2 + 1
    private static let halfWindow = windowLength / 2

    /// Frames computed so far, including discarded ones.
    private(set) var frameCount = 0
    /// The first frame that ``frames`` holds.
    private(set) var firstRetainedFrame = 0
    /// Retained log-mel frames in frame-major order, `[frames × melBinCount]`.
    private(set) var frames: [Float] = []
    /// Samples received, including discarded ones.
    private(set) var sampleCount = 0
    private(set) var ended = false

    private let setup: vDSP_DFT_Setup
    private let window: [Float]
    private let filterbank: [Float]
    /// Pre-emphasized samples from absolute sample `sampleBase` on.
    private var emphasized: [Float] = []
    private var sampleBase = 0
    private var lastSample: Float = 0
    private var frame = [Float](repeating: 0, count: fftLength)
    private var even = [Float](repeating: 0, count: fftLength / 2)
    private var odd = [Float](repeating: 0, count: fftLength / 2)
    private var real = [Float](repeating: 0, count: fftLength / 2)
    private var imaginary = [Float](repeating: 0, count: fftLength / 2)
    private var power: [Float] = []
    private var mel: [Float] = []

    init() {
        guard
            let setup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(Self.fftLength), .FORWARD)
        else { fatalError("Accelerate cannot create a \(Self.fftLength)-point real DFT.") }
        self.setup = setup
        window = (0..<Self.windowLength).map { index in
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double(index) / Double(Self.windowLength - 1)))
        }
        filterbank = ParakeetMelFrontend.slaneyFilterbank()
    }

    deinit { vDSP_DFT_DestroySetup(setup) }

    /// Appends samples and computes the frames whose audio they complete.
    /// Appending after the end of input is a programming error.
    func append(_ samples: UnsafeBufferPointer<Float>) {
        precondition(!ended, "Cannot append audio after the end of input.")
        guard let source = samples.baseAddress, !samples.isEmpty else { return }
        let start = emphasized.count
        emphasized.append(contentsOf: repeatElement(0, count: samples.count))
        emphasized.withUnsafeMutableBufferPointer { target in
            let output = target.baseAddress! + start
            // The first sample of the recording has no predecessor, as in NeMo.
            output[0] =
                sampleCount == 0
                ? source[0] : source[0] - ParakeetMelFrontend.preemphasis * lastSample
            if samples.count > 1 {
                var coefficient = -ParakeetMelFrontend.preemphasis
                vDSP_vsma(
                    source, 1, &coefficient, source + 1, 1, output + 1, 1,
                    vDSP_Length(samples.count - 1))
            }
        }
        lastSample = source[samples.count - 1]
        sampleCount += samples.count
        // Frame `k` is complete when samples through `160 k + 199` have arrived.
        let complete =
            sampleCount >= Self.halfWindow
            ? (sampleCount - Self.halfWindow) / Self.hopLength + 1 : 0
        compute(through: complete)
    }

    func append(_ samples: [Float]) {
        samples.withUnsafeBufferPointer { append($0) }
    }

    /// Computes the last frames of the recording, which read zeros after its last sample.
    func finish() {
        guard !ended else { return }
        ended = true
        compute(through: sampleCount / Self.hopLength)
    }

    /// Runs `body` with the frames that the end of input would add, then removes them.
    ///
    /// A finalization encodes the received audio as if it ended; later audio replaces
    /// the provisional frames, which read zeros after the last received sample.
    func withProvisionalEnd<Result>(_ body: () throws -> Result) rethrows -> Result {
        let (count, stored) = (frameCount, frames.count)
        compute(through: sampleCount / Self.hopLength)
        defer {
            frames.removeLast(frames.count - stored)
            frameCount = count
        }
        return try body()
    }

    /// Copies frames into a mel-major encoder input, `[melBinCount × width]`.
    ///
    /// - Parameters:
    ///   - range: Absolute frames; frames before the first frame of the recording are zero.
    ///   - input: The input, whose column `column` receives the first frame of `range`.
    func copy(
        _ range: Range<Int>, into input: UnsafeMutableBufferPointer<Float>, width: Int, column: Int
    ) {
        precondition(range.upperBound <= frameCount && column + range.count <= width)
        frames.withUnsafeBufferPointer { frames in
            for (offset, index) in range.enumerated() {
                let target = column + offset
                guard index >= 0 else {
                    for bin in 0..<Self.melBinCount { input[bin * width + target] = 0 }
                    continue
                }
                precondition(index >= firstRetainedFrame, "The frame was discarded.")
                let source = (index - firstRetainedFrame) * Self.melBinCount
                for bin in 0..<Self.melBinCount {
                    input[bin * width + target] = frames[source + bin]
                }
            }
        }
    }

    /// Discards frames before `frame` and the audio that later frames do not read.
    func discard(before frame: Int) {
        let count = min(max(0, frame - firstRetainedFrame), frameCount - firstRetainedFrame)
        if count > 0 {
            frames.removeFirst(count * Self.melBinCount)
            firstRetainedFrame += count
        }
        // The next frame reads from `160 frameCount - 200`.
        let keep = max(0, frameCount * Self.hopLength - Self.halfWindow)
        let drop = min(keep - sampleBase, emphasized.count)
        if drop > 0 {
            emphasized.removeFirst(drop)
            sampleBase += drop
        }
    }

    /// Computes frames up to `end`, exclusive.
    private func compute(through end: Int) {
        let count = end - frameCount
        guard count > 0 else { return }
        if power.count < Self.binCount * count {
            power = [Float](repeating: 0, count: Self.binCount * count)
            mel = [Float](repeating: 0, count: Self.melBinCount * count)
        }
        for column in 0..<count {
            spectrum(of: frameCount + column, column: column, columns: count)
        }
        filterbank.withUnsafeBufferPointer { bank in
            power.withUnsafeBufferPointer { power in
                mel.withUnsafeMutableBufferPointer { mel in
                    vDSP_mmul(
                        bank.baseAddress!, 1, power.baseAddress!, 1, mel.baseAddress!, 1,
                        vDSP_Length(Self.melBinCount), vDSP_Length(count),
                        vDSP_Length(Self.binCount))
                    var guardValue = ParakeetMelFrontend.logGuard
                    vDSP_vsadd(
                        mel.baseAddress!, 1, &guardValue, mel.baseAddress!, 1,
                        vDSP_Length(Self.melBinCount * count))
                    var length = Int32(Self.melBinCount * count)
                    vvlogf(mel.baseAddress!, mel.baseAddress!, &length)
                }
            }
        }
        // Store the new frames in frame-major order.
        let start = frames.count
        frames.append(contentsOf: repeatElement(0, count: Self.melBinCount * count))
        frames.withUnsafeMutableBufferPointer { frames in
            mel.withUnsafeBufferPointer { mel in
                vDSP_mtrans(
                    mel.baseAddress!, 1, frames.baseAddress! + start, 1, vDSP_Length(count),
                    vDSP_Length(Self.melBinCount))
            }
        }
        frameCount = end
    }

    /// Writes the power spectrum of frame `index` to column `column` of ``power``,
    /// `[binCount × columns]`.
    private func spectrum(of index: Int, column: Int, columns: Int) {
        let half = Self.fftLength / 2
        let offset = (Self.fftLength - Self.windowLength) / 2
        let first = index * Self.hopLength - Self.halfWindow
        frame.withUnsafeMutableBufferPointer { frame in
            frame.update(repeating: 0)
            emphasized.withUnsafeBufferPointer { emphasized in
                window.withUnsafeBufferPointer { window in
                    // Samples before the recording and after the end of input are zero.
                    let lower = max(first, 0, sampleBase)
                    let upper = min(first + Self.windowLength, sampleBase + emphasized.count)
                    guard lower < upper else { return }
                    vDSP_vmul(
                        emphasized.baseAddress! + (lower - sampleBase), 1,
                        window.baseAddress! + (lower - first), 1,
                        frame.baseAddress! + offset + (lower - first), 1,
                        vDSP_Length(upper - lower))
                }
            }
        }
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
        // The real DFT packs the Nyquist term in the first imaginary value and scales every
        // term by 2.
        power.withUnsafeMutableBufferPointer { power in
            let target = power.baseAddress! + column
            target[0] = real[0] * real[0] / 4
            target[half * columns] = imaginary[0] * imaginary[0] / 4
            for bin in 1..<half {
                target[bin * columns] =
                    (real[bin] * real[bin] + imaginary[bin] * imaginary[bin]) / 4
            }
        }
    }
}
