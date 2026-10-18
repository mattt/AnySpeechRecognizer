import Accelerate
import Foundation

/// Computes Parakeet log-mel features for one encoder window with Accelerate.
///
/// The features follow NeMo's `AudioToMelSpectrogramPreprocessor` from NeMo 2.5 on
/// (checked against NeMo 3.0.0): pre-emphasis 0.97 over the valid samples, with zeros
/// after them; `n_fft / 2` zeros of padding at both window ends;
/// a symmetric 400-sample Hann window centered in a 512-point DFT, a 10 ms hop,
/// a power spectrum, 128 Slaney mel bins, `log(x + 2^-24)`, and per-feature normalization
/// over the valid frames. A window of `N` valid samples has `floor(N / 160)` valid frames,
/// and frames at or after that count are zero. Dither is off.
/// One instance owns scratch buffers and serves one owner.
///
/// NeMo 2.4 and earlier, and FluidInference's Core ML `Preprocessor`, reflected the window
/// at its ends and counted `floor(N / 160) + 1` valid frames.
///
/// Overlapping windows can reuse log-mel frames; see
/// ``compute(_:validSampleCount:startHop:into:)``.
final class ParakeetMelFrontend {
    static let sampleRate = 16_000
    static let hopLength = 160
    static let windowLength = 400
    static let fftLength = 512
    static let melBinCount = 128
    static let preemphasis: Float = 0.97
    static let logGuard: Float = 0x1p-24

    /// The number of samples in one encoder window.
    let windowSamples: Int
    /// The number of mel frames in one encoder window.
    let frameCount: Int

    private static let binCount = fftLength / 2 + 1
    private let setup: vDSP_DFT_Setup
    private let window: [Float]
    private let filterbank: [Float]
    private var padded: [Float]
    /// Power spectra in frequency-major order, `[binCount × frameCount]`.
    private var power: [Float]
    /// Log-mel values of the frames that a reusing window computes, `[melBinCount × count]`.
    private var melColumns: [Float]
    private var frame: [Float]
    private var even: [Float]
    private var odd: [Float]
    private var real: [Float]
    private var imaginary: [Float]
    /// Absolute hops whose log-mel values are in `cache`, at most `frameCount` of them.
    private var cachedHops = 0..<0
    /// Log-mel values by hop, `[melBinCount × frameCount]`; hop `h` is in column
    /// `h % frameCount`. Allocated by the first reusing window.
    private var cache: [Float] = []

    init(windowSamples: Int = 240_000) {
        precondition(windowSamples > Self.fftLength)
        self.windowSamples = windowSamples
        frameCount = windowSamples / Self.hopLength + 1
        guard let setup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(Self.fftLength), .FORWARD)
        else { fatalError("Accelerate cannot create a \(Self.fftLength)-point real DFT.") }
        self.setup = setup
        window = (0..<Self.windowLength).map { index in
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double(index) / Double(Self.windowLength - 1)))
        }
        filterbank = Self.slaneyFilterbank()
        padded = [Float](repeating: 0, count: windowSamples + Self.fftLength)
        power = [Float](repeating: 0, count: Self.binCount * frameCount)
        melColumns = [Float](repeating: 0, count: Self.melBinCount * frameCount)
        frame = [Float](repeating: 0, count: Self.fftLength)
        even = [Float](repeating: 0, count: Self.fftLength / 2)
        odd = [Float](repeating: 0, count: Self.fftLength / 2)
        real = [Float](repeating: 0, count: Self.fftLength / 2)
        imaginary = [Float](repeating: 0, count: Self.fftLength / 2)
    }

    deinit { vDSP_DFT_DestroySetup(setup) }

    /// The number of valid mel frames for a valid sample count, as NeMo's preprocessor reports it.
    func validFrameCount(sampleCount: Int) -> Int {
        max(0, min(frameCount, sampleCount / Self.hopLength))
    }

    /// Writes features for one window into `output` in mel-major order, `[melBinCount × frameCount]`.
    ///
    /// When `startHop` is set, the frontend keeps the log-mel values of interior frames by
    /// absolute hop and reuses them in later windows, so a window that overlaps earlier ones
    /// computes only its new frames and its edge frames. An interior frame reads only window
    /// audio after the first sample, which has no pre-emphasis, and before the valid sample
    /// count, so its values do not depend on where the window starts or ends.
    /// The normalization still covers every valid frame of the window.
    ///
    /// - Parameters:
    ///   - samples: The audio at the start of the window. Later window samples are zero.
    ///   - validSampleCount: The number of samples the encoder treats as audio.
    ///   - startHop: The window's first sample divided by ``hopLength``, counted from an origin
    ///     that every reusing call shares, or `nil` to compute every frame without the cache.
    ///   - output: A buffer of `melBinCount × frameCount` values.
    /// - Returns: The number of valid mel frames.
    @discardableResult
    func compute(
        _ samples: UnsafeBufferPointer<Float>, validSampleCount: Int, startHop: Int? = nil,
        into output: UnsafeMutableBufferPointer<Float>
    ) -> Int {
        precondition(samples.count <= windowSamples && validSampleCount <= windowSamples)
        precondition(output.count == Self.melBinCount * frameCount)
        let validFrames = validFrameCount(sampleCount: validSampleCount)
        output.update(repeating: 0)
        guard validFrames > 0 else { return 0 }
        preemphasize(samples, validSampleCount: validSampleCount)
        let frames = frameCount
        if let startHop {
            computeReusingFrames(
                sampleCount: min(samples.count, validSampleCount), validFrames: validFrames,
                startHop: startHop,
                into: output.baseAddress!)
        } else {
            computePower(Array(0..<validFrames), columns: frames)
            filterbank.withUnsafeBufferPointer { bank in
                power.withUnsafeBufferPointer { power in
                    vDSP_mmul(
                        bank.baseAddress!, 1, power.baseAddress!, 1, output.baseAddress!, 1,
                        vDSP_Length(Self.melBinCount), vDSP_Length(frames),
                        vDSP_Length(Self.binCount))
                }
            }
            for bin in 0..<Self.melBinCount {
                Self.logarithm(output.baseAddress! + bin * frames, count: validFrames)
            }
        }
        for bin in 0..<Self.melBinCount {
            normalize(
                UnsafeMutableBufferPointer(
                    rebasing: output[(bin * frames)..<(bin * frames + validFrames)]))
        }
        return validFrames
    }

    /// Writes log-mel values for the valid frames, reusing cached interior frames.
    private func computeReusingFrames(
        sampleCount: Int, validFrames: Int, startHop: Int, into output: UnsafeMutablePointer<Float>
    ) {
        let frames = frameCount
        // Frame `k` reads window samples `k * hop - 200 ..< k * hop + 200`.
        // Frames from 2 on start after the first sample, and frames before `interiorEnd`
        // end before the zero padding.
        let halfWindow = Self.windowLength / 2
        let interior =
            2..<max(2, min(validFrames, (sampleCount - halfWindow) / Self.hopLength + 1))
        let lower = max(interior.lowerBound, cachedHops.lowerBound - startHop)
        let upper = min(interior.upperBound, cachedHops.upperBound - startHop)
        let reused = lower < upper ? lower..<upper : validFrames..<validFrames
        let missing = (0..<validFrames).filter { !reused.contains($0) }
        let computed = missing.count
        computePower(missing, columns: computed)
        filterbank.withUnsafeBufferPointer { bank in
            power.withUnsafeBufferPointer { power in
                melColumns.withUnsafeMutableBufferPointer { mel in
                    vDSP_mmul(
                        bank.baseAddress!, 1, power.baseAddress!, 1, mel.baseAddress!, 1,
                        vDSP_Length(Self.melBinCount), vDSP_Length(computed),
                        vDSP_Length(Self.binCount))
                }
            }
        }
        if cache.isEmpty { cache = [Float](repeating: 0, count: Self.melBinCount * frames) }
        melColumns.withUnsafeMutableBufferPointer { mel in
            let mel = mel.baseAddress!
            Self.logarithm(mel, count: Self.melBinCount * computed)
            // The missing frames form at most two runs: before and after the reused frames.
            var column = 0
            for run in [0..<min(validFrames, reused.lowerBound), reused.upperBound..<validFrames]
            where !run.isEmpty {
                for bin in 0..<Self.melBinCount {
                    (output + bin * frames + run.lowerBound).update(
                        from: mel + bin * computed + column, count: run.count)
                }
                store(
                    run.clamped(to: interior), startHop: startHop, from: mel + column,
                    firstFrame: run.lowerBound, stride: computed)
                column += run.count
            }
        }
        guard !reused.isEmpty else { return }
        cache.withUnsafeBufferPointer { cache in
            Self.forEachSlot(of: reused, startHop: startHop, frames: frames) { index, slot, count in
                for bin in 0..<Self.melBinCount {
                    (output + bin * frames + index).update(
                        from: cache.baseAddress! + bin * frames + slot, count: count)
                }
            }
        }
    }

    /// Caches computed interior frames when they continue the cached hops or follow a gap.
    ///
    /// - Parameters:
    ///   - frames: Window frame indices, from `firstFrame` on.
    ///   - values: The log-mel values, where frame `k` of bin `b` is at
    ///     `b * stride + k - firstFrame`.
    private func store(
        _ frames: Range<Int>, startHop: Int, from values: UnsafePointer<Float>, firstFrame: Int,
        stride: Int
    ) {
        guard !frames.isEmpty else { return }
        let hops = (startHop + frames.lowerBound)..<(startHop + frames.upperBound)
        if hops.lowerBound > cachedHops.upperBound || cachedHops.isEmpty {
            cachedHops = hops.lowerBound..<hops.lowerBound
        } else if hops.lowerBound < cachedHops.upperBound {
            // Earlier frames than the cache holds; a later window does not need them.
            return
        }
        let capacity = frameCount
        cache.withUnsafeMutableBufferPointer { cache in
            Self.forEachSlot(of: frames, startHop: startHop, frames: capacity) {
                index, slot, count in
                for bin in 0..<Self.melBinCount {
                    (cache.baseAddress! + bin * capacity + slot).update(
                        from: values + bin * stride + index - firstFrame, count: count)
                }
            }
        }
        cachedHops =
            max(cachedHops.lowerBound, hops.upperBound - capacity)..<hops.upperBound
    }

    /// Calls `body` with contiguous runs of window frames and their cache slots.
    private static func forEachSlot(
        of indices: Range<Int>, startHop: Int, frames: Int,
        _ body: (_ index: Int, _ slot: Int, _ count: Int) -> Void
    ) {
        var index = indices.lowerBound
        while index < indices.upperBound {
            let slot = (startHop + index) % frames
            let count = min(indices.upperBound - index, frames - slot)
            body(index, slot, count)
            index += count
        }
    }

    /// Stores the pre-emphasized window with zero padding in `padded`.
    ///
    /// Pre-emphasized samples from `validSampleCount` on are zero. Before that, samples
    /// missing from `samples` are zero, so the first of them is `-0.97` times the last sample.
    private func preemphasize(_ samples: UnsafeBufferPointer<Float>, validSampleCount: Int) {
        let pad = Self.fftLength / 2
        let count = windowSamples
        let end = min(count, max(0, validSampleCount))
        let audio = min(samples.count, end)
        padded.withUnsafeMutableBufferPointer { padded in
            let signal = padded.baseAddress! + pad
            signal.update(repeating: 0, count: count)
            if let source = samples.baseAddress, audio > 0 {
                signal[0] = source[0]
                if audio > 1 {
                    var coefficient = -Self.preemphasis
                    vDSP_vsma(
                        source, 1, &coefficient, source + 1, 1, signal + 1, 1,
                        vDSP_Length(audio - 1))
                }
                if audio < end {
                    signal[audio] = -Self.preemphasis * source[audio - 1]
                }
            }
            padded.baseAddress!.update(repeating: 0, count: pad)
            (signal + count).update(repeating: 0, count: pad)
        }
    }

    /// Stores the power spectra of window frames in frequency-major order in `power`,
    /// `[binCount × columns]`, one column per frame in order. Later columns are zero.
    private func computePower(_ indices: [Int], columns: Int) {
        let half = Self.fftLength / 2
        let offset = (Self.fftLength - Self.windowLength) / 2
        power.withUnsafeMutableBufferPointer { power in
            power.baseAddress!.update(repeating: 0, count: Self.binCount * columns)
            for (columnIndex, index) in indices.enumerated() {
                padded.withUnsafeBufferPointer { padded in
                    window.withUnsafeBufferPointer { window in
                        frame.withUnsafeMutableBufferPointer { frame in
                            vDSP_vmul(
                                padded.baseAddress! + index * Self.hopLength + offset, 1,
                                window.baseAddress!, 1, frame.baseAddress! + offset, 1,
                                vDSP_Length(Self.windowLength))
                        }
                    }
                }
                frame.withUnsafeBufferPointer { frame in
                    even.withUnsafeMutableBufferPointer { even in
                        odd.withUnsafeMutableBufferPointer { odd in
                            var split = DSPSplitComplex(
                                realp: even.baseAddress!, imagp: odd.baseAddress!)
                            frame.baseAddress!.withMemoryRebound(
                                to: DSPComplex.self, capacity: half
                            ) {
                                vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
                            }
                        }
                    }
                }
                vDSP_DFT_Execute(setup, even, odd, &real, &imaginary)
                // The real DFT packs the Nyquist term in the first imaginary value
                // and scales every term by 2.
                let column = power.baseAddress! + columnIndex
                real.withUnsafeBufferPointer { real in
                    imaginary.withUnsafeBufferPointer { imaginary in
                        let (real, imaginary) = (real.baseAddress!, imaginary.baseAddress!)
                        column[0] = real[0] * real[0] / 4
                        column[half * columns] = imaginary[0] * imaginary[0] / 4
                        var bin = 1
                        while bin < half {
                            column[bin * columns] =
                                (real[bin] * real[bin] + imaginary[bin] * imaginary[bin]) / 4
                            bin += 1
                        }
                    }
                }
            }
        }
    }

    /// Applies the log guard and the logarithm in place.
    private static func logarithm(_ values: UnsafeMutablePointer<Float>, count: Int) {
        var guardValue = logGuard
        vDSP_vsadd(values, 1, &guardValue, values, 1, vDSP_Length(count))
        var length = Int32(count)
        vvlogf(values, values, &length)
    }

    /// Applies per-feature normalization to one mel bin's valid log-mel values.
    private func normalize(_ row: UnsafeMutableBufferPointer<Float>) {
        let count = row.count
        let values = row.baseAddress!
        // Double accumulation keeps a constant row, such as digital silence, exactly zero.
        // The loops use `while` because range iteration is slow in unoptimized builds.
        var sum = 0.0
        var index = 0
        while index < count {
            sum += Double(values[index])
            index += 1
        }
        let mean = sum / Double(count)
        var squares = 0.0
        index = 0
        while index < count {
            let difference = Double(values[index]) - mean
            squares += difference * difference
            index += 1
        }
        // NeMo uses the unbiased variance and adds 1e-5 to the standard deviation.
        let deviation = (squares / Double(max(count - 1, 1))).squareRoot() + 1e-5
        index = 0
        while index < count {
            values[index] = Float((Double(values[index]) - mean) / deviation)
            index += 1
        }
    }

    /// Builds librosa's default Slaney-scale, Slaney-normalized filterbank, `[melBinCount × binCount]`.
    static func slaneyFilterbank() -> [Float] {
        func mel(_ hertz: Double) -> Double {
            hertz < 1000 ? hertz * 3 / 200 : 15 + log(hertz / 1000) / (log(6.4) / 27)
        }
        func hertz(_ mel: Double) -> Double {
            mel < 15 ? mel * 200 / 3 : 1000 * exp((log(6.4) / 27) * (mel - 15))
        }
        let maximum = mel(Double(sampleRate) / 2)
        let edges = (0..<(melBinCount + 2)).map {
            hertz(maximum * Double($0) / Double(melBinCount + 1))
        }
        var weights = [Float](repeating: 0, count: melBinCount * binCount)
        for bin in 0..<melBinCount {
            let (lower, center, upper) = (edges[bin], edges[bin + 1], edges[bin + 2])
            let scale = 2 / (upper - lower)
            for frequencyBin in 0..<binCount {
                let frequency = Double(frequencyBin * sampleRate) / Double(fftLength)
                let rising = (frequency - lower) / (center - lower)
                let falling = (upper - frequency) / (upper - center)
                weights[bin * binCount + frequencyBin] = Float(
                    max(0, min(rising, falling)) * scale)
            }
        }
        return weights
    }
}
