import CoreMedia
import Foundation

/// Turns per-frame speaker activity probabilities into final and volatile diarizer results.
///
/// Each speaker's probabilities pass through NeMo's diarization post-processing
/// (`ts_vad_post_processing`): hysteresis thresholds evaluated on 10 ms steps,
/// padding before and after each speech run, merging of runs that the padding overlaps,
/// removal of speech shorter than the minimum, and filling of gaps shorter than the minimum.
///
/// Later frames can change a speaker's activity only back to
/// `frontier - padOnset - minimumDurationOn - minimumDurationOff`,
/// so activity before that time is decided.
/// Final results end at the latest decided speaker change or silence,
/// or at the decided time when undecided audio exceeds ``maximumUndecided``.
/// A forced cut, for finalization, decides activity through a time immediately.
struct SpeakerSegmenter: Sendable {
    /// NeMo's post-processing parameters, with durations in source time.
    struct Parameters: Sendable, Hashable {
        var onset: Float
        var offset: Float
        var padOnset: CMTime
        var padOffset: CMTime
        var minimumDurationOn: CMTime
        var minimumDurationOff: CMTime
    }

    /// A padded speech interval of one speaker.
    struct Interval: Sendable, Hashable {
        var start: CMTime
        var end: CMTime
        /// Whether the interval continues decided speech, so the minimum duration
        /// does not apply.
        var exempt = false
        var duration: CMTime { end - start }
    }

    private struct Track: Sendable {
        var speaking = false
        /// The padded start of the open run.
        var runStart = CMTime.zero
        var runExempt = false
        /// Closed padded runs, merged where they overlap, in order.
        var closed: [Interval] = []
    }

    /// The longest undecided span before a final result splits ongoing speech.
    static let maximumUndecided = CMTime(value: 10, timescale: 1)
    /// The post-processing step, as in NeMo's 10 ms frames.
    static let step = CMTime(value: 160, timescale: 16_000)

    let parameters: Parameters
    /// The finalization boundary: results before it never change.
    private(set) var finalized: CMTime
    /// The end of the last appended frame.
    private(set) var frontier: CMTime
    private var tracks: [Track]

    init(speakerCount: Int, parameters: Parameters, origin: CMTime) {
        self.parameters = parameters
        finalized = origin
        frontier = origin
        tracks = Array(repeating: Track(), count: speakerCount)
    }

    /// The time before which later frames cannot change activity.
    var decidedTime: CMTime {
        let lookback =
            parameters.padOnset + parameters.minimumDurationOn + parameters.minimumDurationOff
        return max(finalized, frontier - lookback)
    }

    /// Adds one frame of speaker probabilities.
    /// - Parameters:
    ///   - start: The frame's source start time.
    ///   - end: The frame's source end time, which can be shorter at the end of a stream.
    ///   - probabilities: One probability for each speaker.
    mutating func append(start: CMTime, end: CMTime, probabilities: ArraySlice<Float>) {
        guard end > start else { return }
        for (speaker, probability) in zip(tracks.indices, probabilities) {
            var time = start
            while time < end {
                step(speaker, probability: probability, at: time)
                time = time + Self.step
            }
        }
        frontier = max(frontier, end)
    }

    private mutating func step(_ speaker: Int, probability: Float, at time: CMTime) {
        if tracks[speaker].speaking {
            if probability < parameters.offset {
                close(speaker, at: time + parameters.padOffset)
            }
        } else if probability > parameters.onset {
            tracks[speaker].speaking = true
            tracks[speaker].runExempt = false
            tracks[speaker].runStart = max(time - parameters.padOnset, finalized)
        }
    }

    private mutating func close(_ speaker: Int, at end: CMTime) {
        let interval = Interval(
            start: tracks[speaker].runStart, end: max(end, tracks[speaker].runStart),
            exempt: tracks[speaker].runExempt)
        tracks[speaker].speaking = false
        tracks[speaker].runExempt = false
        Self.insert(interval, into: &tracks[speaker].closed)
    }

    /// Adds an interval to an ordered list, merging intervals that touch or overlap.
    private static func insert(_ interval: Interval, into intervals: inout [Interval]) {
        if var last = intervals.last, interval.start <= last.end {
            last.end = max(last.end, interval.end)
            last.exempt = last.exempt || interval.exempt
            intervals[intervals.count - 1] = last
        } else {
            intervals.append(interval)
        }
    }

    /// Ends the current stream at `end`: open runs close there, without padding past it.
    mutating func endStream(at end: CMTime) {
        for speaker in tracks.indices where tracks[speaker].speaking {
            close(speaker, at: end)
        }
        for speaker in tracks.indices {
            tracks[speaker].closed = tracks[speaker].closed.compactMap { interval in
                guard interval.start < end else { return nil }
                return Interval(
                    start: interval.start, end: min(interval.end, end), exempt: interval.exempt)
            }
        }
        frontier = max(frontier, end)
    }

    /// A speaker's filtered activity and the raw intervals that the filter dropped.
    /// - Parameter forced: Whether an open run counts as speech regardless of its duration.
    private func activity(of track: Track, forced: Bool) -> (
        active: [Interval], dropped: [Interval]
    ) {
        var candidates = track.closed
        var openIndex: Int?
        if track.speaking {
            Self.insert(
                Interval(start: track.runStart, end: frontier, exempt: track.runExempt || forced),
                into: &candidates)
            openIndex = candidates.count - 1
        }
        var active: [Interval] = []
        var dropped: [Interval] = []
        for (index, candidate) in candidates.enumerated() {
            let survives =
                candidate.exempt || candidate.duration >= parameters.minimumDurationOn
                || (forced && index == openIndex)
            guard survives else {
                dropped.append(candidate)
                continue
            }
            var interval = candidate
            interval.exempt = true
            if let last = active.last, interval.start - last.end < parameters.minimumDurationOff {
                active[active.count - 1].end = max(last.end, interval.end)
            } else {
                active.append(interval)
            }
        }
        return (active, dropped)
    }

    /// Returns final results whose activity is decided, splitting at speaker changes,
    /// silence, or after ``maximumUndecided``.
    mutating func decide() -> SpeakerDiarizer.Result? {
        let decided = decidedTime
        guard decided > finalized else { return nil }
        // Speech that is still open at the frontier continues past the decided time.
        let activities = tracks.map { track in
            var active = activity(of: track, forced: false).active
            if track.speaking, let last = active.last, last.end >= frontier {
                active[active.count - 1].end = .positiveInfinity
            }
            return active
        }
        var cut: CMTime?
        var activeAtDecided = false
        for intervals in activities {
            for interval in intervals {
                for boundary in [interval.start, interval.end]
                where boundary > finalized && boundary <= decided {
                    cut = max(cut ?? boundary, boundary)
                }
                if interval.start < decided && interval.end > decided { activeAtDecided = true }
            }
        }
        if !activeAtDecided { cut = decided }
        if cut == nil, decided - finalized >= Self.maximumUndecided { cut = decided }
        guard let cut, cut > finalized else { return nil }
        return self.cut(at: cut, forced: false)
    }

    /// Publishes activity through `time` and makes it final.
    ///
    /// A forced cut keeps an open run regardless of its duration.
    /// Speech that continues past `time` continues as decided speech.
    /// - Returns: A final result, or `nil` when no speaker speaks in the range.
    mutating func cut(at time: CMTime, forced: Bool) -> SpeakerDiarizer.Result? {
        guard time > finalized else { return nil }
        var segments: [SpeakerDiarizer.Segment] = []
        for speaker in tracks.indices {
            let (active, dropped) = activity(of: tracks[speaker], forced: forced)
            for interval in active {
                let start = max(interval.start, finalized)
                let end = min(interval.end, time)
                if end > start {
                    segments.append(
                        .init(
                            speaker: .init(id: speaker), range: CMTimeRange(start: start, end: end))
                    )
                }
            }
            // Keep what follows the cut: continuations of decided speech, open runs,
            // and raw intervals that can still be judged.
            var track = tracks[speaker]
            var closed: [Interval] = []
            let openEnd = track.speaking ? frontier : nil
            for interval in active where interval.end > time {
                if let openEnd, interval.end == openEnd, interval.start <= track.runStart {
                    track.runStart = max(interval.start, time)
                    track.runExempt = true
                } else {
                    closed.append(
                        Interval(start: max(interval.start, time), end: interval.end, exempt: true))
                }
            }
            for interval in dropped where interval.end > time {
                if let openEnd, interval.end == openEnd {
                    track.runStart = max(track.runStart, time)
                } else {
                    closed.append(
                        Interval(start: max(interval.start, time), end: interval.end, exempt: false)
                    )
                }
            }
            if track.speaking { track.runStart = max(track.runStart, time) }
            track.closed = []
            for interval in closed.sorted(by: { $0.start < $1.start }) {
                Self.insert(interval, into: &track.closed)
            }
            tracks[speaker] = track
        }
        let previous = finalized
        finalized = time
        guard !segments.isEmpty else { return nil }
        segments.sort { lhs, rhs in
            lhs.range.start == rhs.range.start
                ? lhs.speaker < rhs.speaker : lhs.range.start < rhs.range.start
        }
        let start = segments.map(\.range.start).min() ?? previous
        let end = segments.map(\.range.end).max() ?? time
        return SpeakerDiarizer.Result(
            segments: segments, range: CMTimeRange(start: start, end: end),
            resultsFinalizationTime: time)
    }

    /// A volatile result from the finalization boundary through `end`,
    /// with preview frames that are not committed.
    /// - Parameters:
    ///   - preview: Frames after the frontier, such as the model's right context.
    ///   - end: The end of the preview, at most the received audio.
    /// - Returns: The volatile result, or `nil` when no audio follows the boundary.
    func volatileResult(
        preview: [(start: CMTime, end: CMTime, probabilities: ArraySlice<Float>)], end: CMTime
    ) -> SpeakerDiarizer.Result? {
        var copy = self
        for frame in preview {
            copy.append(start: frame.start, end: frame.end, probabilities: frame.probabilities)
        }
        let rangeEnd = max(copy.frontier, end)
        guard rangeEnd > finalized else { return nil }
        var segments: [SpeakerDiarizer.Segment] = []
        for speaker in copy.tracks.indices {
            for interval in copy.activity(of: copy.tracks[speaker], forced: true).active {
                let start = max(interval.start, finalized)
                let stop = min(interval.end, rangeEnd)
                if stop > start {
                    segments.append(
                        .init(
                            speaker: .init(id: speaker), range: CMTimeRange(start: start, end: stop)
                        ))
                }
            }
        }
        segments.sort { lhs, rhs in
            lhs.range.start == rhs.range.start
                ? lhs.speaker < rhs.speaker : lhs.range.start < rhs.range.start
        }
        return SpeakerDiarizer.Result(
            segments: segments, range: CMTimeRange(start: finalized, end: rangeEnd),
            resultsFinalizationTime: finalized)
    }
}
