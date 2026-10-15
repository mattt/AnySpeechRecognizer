import CoreMedia
import Foundation
import Speech

/// Converts source frame positions without accumulating floating-point offsets.
enum AudioTimeline {
    static func time(frame: Int64, sampleRate: Double) throws -> CMTime {
        guard frame >= 0, sampleRate.isFinite, sampleRate > 0,
            sampleRate.rounded() == sampleRate, sampleRate <= Double(Int32.max)
        else {
            throw SpeechRecognitionError.invalidAudio(
                "The source sample rate must be a positive integral rate.")
        }
        return CMTime(value: frame, timescale: Int32(sampleRate))
    }
}

func validateTiming(in text: AttributedString, range: CMTimeRange) throws {
    var lastStart = range.start
    for run in text.runs {
        guard let wordRange = run[AttributeScopes.SpeechAttributes.TimeRangeAttribute.self] else {
            continue
        }
        guard wordRange.isValid, wordRange.start.isNumeric, wordRange.duration.isNumeric,
            wordRange.duration >= .zero, wordRange.start >= lastStart,
            wordRange.start >= range.start, wordRange.end <= range.end
        else {
            throw SpeechRecognitionError.invalidResult(
                "A text attribute has invalid or unordered source timing.")
        }
        lastStart = wordRange.start
    }
}
