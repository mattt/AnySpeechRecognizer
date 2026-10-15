import AVFoundation
import CoreMedia
import Foundation

/// A module that identifies who speaks when, with an explicitly selected model.
///
/// Speaker diarization is a package extension; Apple's Speech framework has no counterpart.
/// A diarizer runs beside a ``SpeechTranscriber`` in the same ``SpeechAnalyzer``
/// and receives the same input on the same source timeline.
/// It does not change which audio is transcribed or the transcriber's results.
/// Use ``SpeakerTimeline`` to attribute transcript words to speakers.
///
/// Each instance belongs to one analyzer and permits one result iterator.
public final class SpeakerDiarizer: SpeechModule, AnalyzerModule {
    /// Options that affect result delivery.
    public enum ReportingOption: String, Sendable, Hashable, Codable, CaseIterable {
        /// Publishes volatile results for audio whose speakers can still change.
        case volatileResults
    }

    /// The results of one diarizer.
    public typealias Results = SpeechModuleResults<Result>

    /// The immutable model configuration.
    public let model: any SpeakerDiarizationModel
    /// The requested result delivery options.
    public let reportingOptions: Set<ReportingOption>
    let channel = ResultChannel<Result>()

    /// Creates a diarizer without loading or downloading assets.
    /// Options the model does not support fail before audio is consumed.
    /// - Parameters:
    ///   - model: The immutable model configuration.
    ///   - reportingOptions: Options that affect result delivery.
    public init(model: any SpeakerDiarizationModel, reportingOptions: Set<ReportingOption> = []) {
        self.model = model
        self.reportingOptions = reportingOptions
    }

    var request: SpeakerDiarizationRequest { .init(reportingOptions: reportingOptions) }

    /// The audio formats this diarizer can analyze.
    public var availableCompatibleAudioFormats: [AVAudioFormat] {
        get async { await model.compatibleAudioFormats(for: request) }
    }

    /// Results in source order.
    ///
    /// Consume results concurrently with analysis.
    /// The buffer holds at most 256 results.
    /// An unread volatile result is replaced when a newer result supersedes it.
    /// Final results are never dropped; overflow fails the session instead.
    /// If iteration stops early, explicitly cancel the analyzer.
    public var results: Results { Results(channel: channel) }

    var binding: ModuleBinding {
        let channel = channel
        let member = DiarizationMember(
            model: model, request: request, deliver: { try channel.publish($0) })
        return ModuleBinding(
            id: ObjectIdentifier(self), backend: .diarization(member),
            attach: { channel.attach() },
            deliver: { _ in
                throw SpeechRecognitionError.invalidSessionState(
                    "A speaker diarizer does not accept transcription results.")
            },
            finish: { channel.finish(throwing: $0, discardBuffered: $1) })
    }

    /// A speaker identity that is stable within one analyzer session.
    ///
    /// Identifiers come from the model; they are unrelated across sessions.
    /// ``SortformerDiarizationModel`` numbers speakers from 0 in order of arrival.
    public struct Speaker: Sendable, Hashable, Comparable, Codable, CustomStringConvertible {
        /// The speaker's identifier within the session.
        public let id: Int

        /// Creates a speaker identity.
        /// - Parameter id: The speaker's identifier within the session.
        public init(id: Int) { self.id = id }

        public static func < (lhs: Self, rhs: Self) -> Bool { lhs.id < rhs.id }
        public var description: String { "Speaker \(id + 1)" }
    }

    /// A range of source audio in which one speaker speaks.
    public struct Segment: Sendable, Hashable, CustomStringConvertible {
        /// The speaker.
        public let speaker: Speaker
        /// The range in the source recording.
        public let range: CMTimeRange

        /// Creates a segment.
        /// - Parameters:
        ///   - speaker: The speaker.
        ///   - range: The range in the source recording.
        public init(speaker: Speaker, range: CMTimeRange) {
            self.speaker = speaker
            self.range = range
        }

        public var description: String {
            String(
                format: "%@ %.2f–%.2f", speaker.description, range.start.seconds, range.end.seconds)
        }
    }

    /// The speaker segments in a range of source audio.
    ///
    /// Overlapping speech appears as overlapping segments of different speakers.
    /// Final results cover consecutive parts of the timeline and never change.
    /// A segment that continues past a final result's finalization time is split there,
    /// and the rest appears in a later result.
    /// A volatile result replaces earlier volatile results;
    /// one without segments clears them.
    public struct Result: SpeechModuleResult, ChannelResult, Sendable, Hashable,
        CustomStringConvertible
    {
        /// The range in the source recording.
        public let range: CMTimeRange
        /// The exclusive boundary before which no result can change.
        public let resultsFinalizationTime: CMTime
        /// The speaker segments in the range, ordered by start time, then by speaker.
        public let segments: [Segment]

        /// Creates a result without rewriting segments.
        /// - Parameters:
        ///   - segments: The speaker segments, inside `range`.
        ///   - range: The source audio range.
        ///   - resultsFinalizationTime: The finalized audio boundary after this result.
        public init(segments: [Segment], range: CMTimeRange, resultsFinalizationTime: CMTime) {
            self.segments = segments
            self.range = range
            self.resultsFinalizationTime = resultsFinalizationTime
        }

        /// The distinct speakers in the result, in identifier order.
        public var speakers: [Speaker] { Array(Set(segments.map(\.speaker))).sorted() }

        public var description: String {
            segments.map(\.description).joined(separator: ", ")
        }
        var textByteCount: Int { 0 }
    }
}

/// A diarizer's model, request, and typed result delivery.
struct DiarizationMember: Sendable {
    let model: any SpeakerDiarizationModel
    let request: SpeakerDiarizationRequest
    let deliver: @Sendable (SpeakerDiarizer.Result) throws -> Void
}
