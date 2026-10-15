import CoreMedia
import Foundation
import Speech

extension AttributeScopes {
    /// Attributes that AnySpeechRecognizer defines.
    ///
    /// Apple's Speech attribute scope defines no speaker key,
    /// so the package keeps its keys in its own scope.
    public struct AnySpeechRecognizerAttributes: AttributeScope {
        /// The speaker of a word, from ``SpeakerTimeline/attributing(_:maximumGap:)``.
        public let speaker: SpeakerAttribute

        /// The speaker of a run of transcript text.
        public enum SpeakerAttribute: CodableAttributedStringKey {
            public typealias Value = SpeakerDiarizer.Speaker
            public static let name = "AnySpeechRecognizer.speaker"
        }
    }

    /// The scope of the attributes that AnySpeechRecognizer defines.
    public var anySpeechRecognizer: AnySpeechRecognizerAttributes.Type {
        AnySpeechRecognizerAttributes.self
    }
}

extension AttributeDynamicLookup {
    public subscript<T: AttributedStringKey>(
        dynamicMember keyPath: KeyPath<AttributeScopes.AnySpeechRecognizerAttributes, T>
    ) -> T {
        self[T.self]
    }
}

/// The speaker segments of one diarization session, built from a diarizer's results.
///
/// Apply every result in order.
/// Final segments accumulate, and a segment that a finalization split is joined again.
/// Volatile segments are replaced by each newer result.
///
/// ```swift
/// var timeline = SpeakerTimeline()
/// for try await result in diarizer.results { timeline.apply(result) }
/// let attributed = timeline.attributing(transcript)
/// for turn in attributed.speakerTurns { print(turn.speaker, turn.text) }
/// ```
public struct SpeakerTimeline: Sendable, Hashable {
    /// Final segments, ordered by start time, then by speaker.
    public private(set) var segments: [SpeakerDiarizer.Segment] = []
    /// Segments of the newest volatile result after the last finalization.
    public private(set) var volatileSegments: [SpeakerDiarizer.Segment] = []
    /// The boundary before which no applied result can change, or `nil` before any final result.
    public private(set) var finalizationTime: CMTime?

    /// Creates an empty timeline.
    public init() {}

    /// Creates a timeline from final segments.
    /// - Parameter segments: Final segments in any order.
    public init(segments: [SpeakerDiarizer.Segment]) {
        for segment in segments.sorted(by: Self.precedes) { appendFinal(segment) }
    }

    /// Final and volatile segments, ordered by start time, then by speaker.
    public var allSegments: [SpeakerDiarizer.Segment] {
        (segments + volatileSegments).sorted(by: Self.precedes)
    }

    /// The distinct speakers of the final and volatile segments, in identifier order.
    public var speakers: [SpeakerDiarizer.Speaker] {
        Array(Set(allSegments.map(\.speaker))).sorted()
    }

    /// Adds a diarizer result.
    ///
    /// A final result appends its segments and removes volatile segments before its
    /// finalization time. A volatile result replaces the volatile segments.
    /// - Parameter result: The next result in source order.
    public mutating func apply(_ result: SpeakerDiarizer.Result) {
        if result.isFinal {
            for segment in result.segments { appendFinal(segment) }
            let boundary = result.resultsFinalizationTime
            finalizationTime = max(finalizationTime ?? boundary, boundary)
            volatileSegments = volatileSegments.compactMap { segment in
                guard segment.range.end > boundary else { return nil }
                let start = max(segment.range.start, boundary)
                return .init(
                    speaker: segment.speaker,
                    range: CMTimeRange(start: start, end: segment.range.end))
            }
        } else {
            volatileSegments = result.segments
        }
    }

    /// Appends a final segment, joining it with an earlier segment of the same speaker
    /// that ends where it starts.
    private mutating func appendFinal(_ segment: SpeakerDiarizer.Segment) {
        if let index = segments.lastIndex(where: {
            $0.speaker == segment.speaker && $0.range.end == segment.range.start
        }) {
            let joined = segments[index].range.union(segment.range)
            segments[index] = .init(speaker: segment.speaker, range: joined)
            return
        }
        let index =
            segments.lastIndex(where: { !Self.precedes(segment, $0) }).map { $0 + 1 } ?? 0
        segments.insert(segment, at: index)
    }

    private static func precedes(_ lhs: SpeakerDiarizer.Segment, _ rhs: SpeakerDiarizer.Segment)
        -> Bool
    {
        lhs.range.start == rhs.range.start
            ? lhs.speaker < rhs.speaker : lhs.range.start < rhs.range.start
    }

    /// Returns the speaker of a source range, from final and volatile segments.
    ///
    /// The speaker whose segments overlap the range the longest wins.
    /// Ties go to the speaker whose overlapping segment starts first, then to the lower
    /// identifier. When no segment overlaps the range, the speaker of the nearest segment
    /// within `maximumGap` wins.
    /// - Parameters:
    ///   - range: A source range, such as a word's time range.
    ///   - maximumGap: The largest distance to a segment that does not overlap the range.
    /// - Returns: A speaker, or `nil` when no segment is close enough.
    public func speaker(
        for range: CMTimeRange, maximumGap: CMTime = CMTime(value: 1, timescale: 2)
    ) -> SpeakerDiarizer.Speaker? {
        Self.speaker(for: range, in: allSegments, maximumGap: maximumGap)
    }

    static func speaker(
        for range: CMTimeRange, in segments: [SpeakerDiarizer.Segment], maximumGap: CMTime
    ) -> SpeakerDiarizer.Speaker? {
        var overlaps: [SpeakerDiarizer.Speaker: (duration: CMTime, start: CMTime)] = [:]
        var nearest: (distance: CMTime, start: CMTime, speaker: SpeakerDiarizer.Speaker)?
        for segment in segments {
            let start = max(segment.range.start, range.start)
            let end = min(segment.range.end, range.end)
            if end > start {
                let previous = overlaps[segment.speaker]
                overlaps[segment.speaker] = (
                    (previous?.duration ?? .zero) + (end - start),
                    min(previous?.start ?? segment.range.start, segment.range.start)
                )
                continue
            }
            let distance = max(
                .zero, max(segment.range.start - range.end, range.start - segment.range.end))
            guard distance <= maximumGap else { continue }
            if let current = nearest {
                let closer =
                    distance != current.distance
                    ? distance < current.distance
                    : segment.range.start != current.start
                        ? segment.range.start < current.start
                        : segment.speaker < current.speaker
                guard closer else { continue }
            }
            nearest = (distance, segment.range.start, segment.speaker)
        }
        if let best = overlaps.max(by: { lhs, rhs in
            if lhs.value.duration != rhs.value.duration {
                return lhs.value.duration < rhs.value.duration
            }
            if lhs.value.start != rhs.value.start { return lhs.value.start > rhs.value.start }
            return lhs.key > rhs.key
        }) {
            return best.key
        }
        return nearest?.speaker
    }

    /// Returns a copy of transcript text with a speaker attribute on each timed run.
    ///
    /// Each run with a `TimeRangeAttribute` gets the speaker that
    /// ``speaker(for:maximumGap:)`` returns for its time range, or keeps no speaker.
    /// Request `.audioTimeRange` from the transcriber so that words have time ranges.
    /// Runs without timing, such as spaces between words, get no speaker.
    /// The text and its other attributes do not change.
    /// - Parameters:
    ///   - text: Transcript text, such as the appended text of final transcriber results.
    ///   - maximumGap: The largest distance from a word to a segment that does not overlap it.
    /// - Returns: The attributed transcript.
    public func attributing(
        _ text: AttributedString, maximumGap: CMTime = CMTime(value: 1, timescale: 2)
    ) -> AttributedString {
        let segments = allSegments
        var attributed = text
        for run in text.runs {
            guard let range = run[AttributeScopes.SpeechAttributes.TimeRangeAttribute.self] else {
                continue
            }
            attributed[run.range][
                AttributeScopes.AnySpeechRecognizerAttributes.SpeakerAttribute.self] =
                Self.speaker(for: range, in: segments, maximumGap: maximumGap)
        }
        return attributed
    }
}

/// Consecutive transcript words of one speaker.
public struct SpeakerTurn: Sendable, Hashable {
    /// The speaker, or `nil` for words that no segment attributes.
    public let speaker: SpeakerDiarizer.Speaker?
    /// The turn's text, with its original attributes.
    public let text: AttributedString
    /// The union of the turn's word time ranges, or `nil` when no word has timing.
    public let range: CMTimeRange?

    /// Creates a turn.
    /// - Parameters:
    ///   - speaker: The speaker, or `nil`.
    ///   - text: The turn's text.
    ///   - range: The union of the turn's word time ranges.
    public init(speaker: SpeakerDiarizer.Speaker?, text: AttributedString, range: CMTimeRange?) {
        self.speaker = speaker
        self.text = text
        self.range = range
    }
}

extension AttributedString {
    /// Groups speaker-attributed text into turns of consecutive words with the same speaker.
    ///
    /// A run with a time range starts a new turn when its speaker differs from the
    /// current turn's. Runs without a time range, such as spaces, join the current turn.
    /// Use ``SpeakerTimeline/attributing(_:maximumGap:)`` to add speakers first.
    public var speakerTurns: [SpeakerTurn] {
        typealias SpeakerAttribute = AttributeScopes.AnySpeechRecognizerAttributes.SpeakerAttribute
        typealias TimeRangeAttribute = AttributeScopes.SpeechAttributes.TimeRangeAttribute
        var turns: [SpeakerTurn] = []
        var speaker: SpeakerDiarizer.Speaker?
        var text = AttributedString()
        var range: CMTimeRange?
        var started = false
        for run in runs {
            let piece = AttributedString(self[run.range])
            guard let wordRange = run[TimeRangeAttribute.self] else {
                text += piece
                continue
            }
            let wordSpeaker = run[SpeakerAttribute.self]
            if started && wordSpeaker != speaker {
                turns.append(SpeakerTurn(speaker: speaker, text: text, range: range))
                text = AttributedString()
                range = nil
            }
            started = true
            speaker = wordSpeaker
            text += piece
            range = range.map { $0.union(wordRange) } ?? wordRange
        }
        if !text.characters.isEmpty || started {
            turns.append(SpeakerTurn(speaker: speaker, text: text, range: range))
        }
        return turns
    }
}
