import AVFoundation
import CoreMedia
import Foundation

/// Converts timed source buffers to one module's format, keeping source time.
///
/// Mono targets average all source channels explicitly.
/// Sample rate conversion is stateful; a format change or a gap between buffers flushes
/// the converter and starts a new segment at the next buffer's start time.
/// One input operation uses a converter at a time.
final class StreamConverter {
    let output: AVAudioFormat
    private var input: AVAudioFormat?
    private var floatConverter: AVAudioConverter?
    private var rateConverter: AVAudioConverter?
    private var intermediate: AVAudioFormat?
    private var segmentStart = CMTime.zero
    private var emittedFrames: Int64 = 0
    private var expectedStart: CMTime?

    init(output: AVAudioFormat) { self.output = output }

    /// Converts one source buffer, flushing first if the format or timeline changes.
    func convert(_ source: TimedAudioBuffer) throws -> [TimedAudioBuffer] {
        var converted: [TimedAudioBuffer] = []
        let buffer = source.buffer
        if let input, let expectedStart,
            !input.isEqual(buffer.format) || source.startTime != expectedStart
        {
            guard source.startTime >= expectedStart else {
                throw SpeechRecognitionError.invalidAudio(
                    "Input buffers overlap in source time.")
            }
            converted += try flush()
        }
        if input == nil { try configure(for: buffer.format, start: source.startTime) }
        expectedStart = source.endTime
        guard buffer.frameLength > 0 else { return converted }
        if Self.matches(buffer.format, output) && rateConverter == nil && floatConverter == nil
            && intermediate == nil
        {
            emittedFrames += Int64(buffer.frameLength)
            return converted + [source]
        }
        let mixed = try mix(buffer)
        if let rateConverter {
            converted += try resample(mixed, with: rateConverter, endOfStream: false)
        } else {
            converted.append(stamp(mixed))
        }
        return converted
    }

    /// Emits frames held by the rate converter and resets the segment.
    func flush() throws -> [TimedAudioBuffer] {
        defer {
            input = nil
            floatConverter = nil
            rateConverter = nil
            intermediate = nil
            expectedStart = nil
        }
        guard let rateConverter else { return [] }
        return try resample(nil, with: rateConverter, endOfStream: true)
    }

    private static func matches(_ lhs: AVAudioFormat, _ rhs: AVAudioFormat) -> Bool {
        lhs.sampleRate == rhs.sampleRate && lhs.channelCount == rhs.channelCount
            && lhs.commonFormat == rhs.commonFormat && lhs.isInterleaved == rhs.isInterleaved
    }

    private func configure(for format: AVAudioFormat, start: CMTime) throws {
        guard format.sampleRate.isFinite, format.sampleRate > 0, format.channelCount > 0,
            format.sampleRate.rounded() == format.sampleRate
        else {
            throw SpeechRecognitionError.invalidAudio(
                "Input audio must have channels and a positive integral sample rate.")
        }
        input = format
        segmentStart = start
        emittedFrames = 0
        if Self.matches(format, output) { return }
        let channels = output.channelCount == 1 ? 1 : format.channelCount
        guard
            let mixed = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: format.sampleRate,
                channels: channels, interleaved: false)
        else {
            throw SpeechRecognitionError.invalidAudio("Cannot describe the intermediate format.")
        }
        intermediate = mixed
        if format.commonFormat != .pcmFormatFloat32 || format.isInterleaved {
            guard
                let float = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32, sampleRate: format.sampleRate,
                    channels: format.channelCount, interleaved: false),
                let converter = AVAudioConverter(from: format, to: float)
            else {
                throw SpeechRecognitionError.invalidAudio(
                    "Cannot convert the input sample format.")
            }
            floatConverter = converter
        }
        if mixed.sampleRate != output.sampleRate || mixed.channelCount != output.channelCount
            || output.commonFormat != .pcmFormatFloat32 || output.isInterleaved
        {
            guard let converter = AVAudioConverter(from: mixed, to: output) else {
                throw SpeechRecognitionError.invalidAudio(
                    "Cannot convert input audio to the module format.")
            }
            rateConverter = converter
        }
    }

    /// Converts to non-interleaved Float32 and averages channels for a mono target.
    private func mix(_ buffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        guard let intermediate else { return buffer }
        var source = buffer
        if let floatConverter {
            guard
                let float = AVAudioPCMBuffer(
                    pcmFormat: floatConverter.outputFormat, frameCapacity: buffer.frameLength)
            else { throw SpeechRecognitionError.invalidAudio("Cannot allocate audio.") }
            try floatConverter.convert(to: float, from: buffer)
            source = float
        }
        if source.format.channelCount == intermediate.channelCount { return source }
        guard let channels = source.floatChannelData,
            let mono = AVAudioPCMBuffer(pcmFormat: intermediate, frameCapacity: source.frameLength),
            let destination = mono.floatChannelData?[0]
        else {
            throw SpeechRecognitionError.invalidAudio(
                "The input does not contain floating-point samples.")
        }
        mono.frameLength = source.frameLength
        let count = Int(source.format.channelCount)
        for frame in 0..<Int(source.frameLength) {
            var sample: Double = 0
            for channel in 0..<count { sample += Double(channels[channel][frame]) }
            let average = Float(sample / Double(count))
            guard average.isFinite else {
                throw SpeechRecognitionError.invalidAudio(
                    "The input contains nonfinite audio samples.")
            }
            destination[frame] = average
        }
        return mono
    }

    private func resample(
        _ buffer: AVAudioPCMBuffer?, with converter: AVAudioConverter, endOfStream: Bool
    ) throws -> [TimedAudioBuffer] {
        let ratio = output.sampleRate / converter.inputFormat.sampleRate
        let frames = Double(buffer?.frameLength ?? 0) * ratio
        let capacity = AVAudioFrameCount(max(1024, frames.rounded(.up) + 64))
        var pending = buffer
        var results: [TimedAudioBuffer] = []
        while true {
            guard let chunk = AVAudioPCMBuffer(pcmFormat: output, frameCapacity: capacity) else {
                throw SpeechRecognitionError.invalidAudio("Cannot allocate audio.")
            }
            var error: NSError?
            let status = converter.convert(to: chunk, error: &error) { _, status in
                if let next = pending {
                    pending = nil
                    status.pointee = .haveData
                    return next
                }
                status.pointee = endOfStream ? .endOfStream : .noDataNow
                return nil
            }
            if let error { throw SpeechRecognitionError.backendFailure(underlying: error) }
            if chunk.frameLength > 0 { results.append(stamp(chunk)) }
            switch status {
            case .haveData: continue
            case .inputRanDry, .endOfStream: return results
            case .error: throw SpeechRecognitionError.invalidAudio("Audio conversion failed.")
            @unknown default: throw SpeechRecognitionError.invalidAudio("Audio conversion failed.")
            }
        }
    }

    private func stamp(_ buffer: AVAudioPCMBuffer) -> TimedAudioBuffer {
        let start =
            segmentStart
            + CMTime(value: emittedFrames, timescale: CMTimeScale(buffer.format.sampleRate))
        emittedFrames += Int64(buffer.frameLength)
        return TimedAudioBuffer(buffer: buffer, startTime: start)
    }
}

/// Assigns source times to caller input and records consumed input.
struct InputTimeline {
    /// The end of the last stamped buffer.
    var lastEnd: CMTime?

    /// Stamps an analyzer input, using its explicit start time or the previous end.
    mutating func stamp(_ buffer: AVAudioPCMBuffer, startTime: CMTime?) throws -> TimedAudioBuffer {
        let rate = buffer.format.sampleRate
        guard rate.isFinite, rate > 0, rate.rounded() == rate, rate <= Double(Int32.max),
            buffer.format.channelCount > 0
        else {
            throw SpeechRecognitionError.invalidAudio(
                "Input audio must have channels and a positive integral sample rate.")
        }
        var start = startTime ?? lastEnd ?? .zero
        guard start.isNumeric, start >= .zero else {
            throw SpeechRecognitionError.invalidAudio("Input start times must be finite.")
        }
        if let lastEnd {
            // Treat a difference within one frame as rounding in caller timestamps.
            let frame = CMTime(value: 1, timescale: CMTimeScale(rate))
            if start < lastEnd {
                guard lastEnd - start <= frame else {
                    throw SpeechRecognitionError.invalidAudio(
                        "Input buffers overlap in source time.")
                }
                start = lastEnd
            } else if start - lastEnd <= frame {
                start = lastEnd
            }
        }
        let timed = TimedAudioBuffer(buffer: buffer, startTime: start)
        lastEnd = timed.endTime
        return timed
    }
}
