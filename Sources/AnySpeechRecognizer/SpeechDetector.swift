import AVFoundation
import CoreMedia
import Foundation
import Speech

/// An Apple-backed module that detects speech in the input.
/// Each instance belongs to one analyzer and permits one result iterator.
public final class SpeechDetector: SpeechModule, AnalyzerModule {
    /// Native detection sensitivity.
    public typealias SensitivityLevel = Speech.SpeechDetector.SensitivityLevel
    /// Native detection options.
    public typealias DetectionOptions = Speech.SpeechDetector.DetectionOptions
    /// The results of one detector.
    public typealias Results = SpeechModuleResults<Result>

    let channel = ResultChannel<Result>()
    let native: Speech.SpeechDetector

    /// Creates a detector.
    /// - Parameters:
    ///   - detectionOptions: The detection sensitivity.
    ///   - reportResults: Whether to publish detection results.
    public init(detectionOptions: DetectionOptions, reportResults: Bool) {
        native = Speech.SpeechDetector(
            detectionOptions: detectionOptions, reportResults: reportResults)
    }

    /// Creates a detector with Apple's default options.
    public init() { native = Speech.SpeechDetector() }

    /// The audio formats this detector can analyze.
    public var availableCompatibleAudioFormats: [AVAudioFormat] {
        native.availableCompatibleAudioFormats
    }

    /// Detection results in source order.
    public var results: Results { Results(channel: channel) }

    var binding: ModuleBinding {
        let channel = channel
        let native = native
        let member = NativeMember(module: native) {
            for try await result in native.results {
                try channel.publish(
                    Result(
                        range: result.range,
                        resultsFinalizationTime: result.resultsFinalizationTime,
                        speechDetected: result.speechDetected))
            }
        }
        return ModuleBinding(
            id: ObjectIdentifier(self), backend: .native(member), attach: { channel.attach() },
            deliver: { _ in
                throw SpeechRecognitionError.invalidSessionState(
                    "A speech detector does not accept executor results.")
            },
            finish: { channel.finish(throwing: $0, discardBuffered: $1) })
    }

    /// Whether a range of audio contains speech.
    public struct Result: SpeechModuleResult, ChannelResult, Sendable, CustomStringConvertible,
        Hashable
    {
        /// The range in the source recording.
        public let range: CMTimeRange
        /// The exclusive boundary before which no result can change.
        public let resultsFinalizationTime: CMTime
        /// Whether the range contains speech.
        public let speechDetected: Bool

        /// Creates a detection result.
        /// - Parameters:
        ///   - range: The source audio range.
        ///   - resultsFinalizationTime: The finalized audio boundary after this result.
        ///   - speechDetected: Whether the range contains speech.
        public init(range: CMTimeRange, resultsFinalizationTime: CMTime, speechDetected: Bool) {
            self.range = range
            self.resultsFinalizationTime = resultsFinalizationTime
            self.speechDetected = speechDetected
        }

        public var description: String {
            "\(speechDetected ? "speech" : "no speech") \(range.start.seconds)–\(range.end.seconds)"
        }
        var textByteCount: Int { 0 }
    }
}
