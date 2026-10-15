import AVFoundation
import CoreMedia
import Foundation
import Speech

/// A transcription module with Apple's system backend or an explicitly selected model.
/// Each instance belongs to one analyzer and permits one result iterator.
public final class SpeechTranscriber: LocaleDependentSpeechModule, AnalyzerModule {
    /// Native options that affect transcription text.
    public typealias TranscriptionOption = Speech.SpeechTranscriber.TranscriptionOption
    /// Native options that affect result delivery.
    public typealias ReportingOption = Speech.SpeechTranscriber.ReportingOption
    /// Native options that select result attributes.
    public typealias ResultAttributeOption = Speech.SpeechTranscriber.ResultAttributeOption
    /// Native preset values.
    public typealias Preset = Speech.SpeechTranscriber.Preset
    /// The results of one transcriber.
    public typealias Results = SpeechModuleResults<Result>

    /// The immutable backend configuration.
    public let model: any SpeechRecognitionModel
    /// The requested transcription options.
    public let configuration: SpeechRecognitionConfiguration
    let channel = ResultChannel<Result>()
    let native: Speech.SpeechTranscriber?

    /// Creates an Apple-backed transcriber with a native preset.
    /// - Parameters:
    ///   - locale: The recognition locale.
    ///   - preset: The native preset.
    public convenience init(locale: Locale, preset: Preset) {
        self.init(
            locale: locale, transcriptionOptions: preset.transcriptionOptions,
            reportingOptions: preset.reportingOptions, attributeOptions: preset.attributeOptions
        )
    }

    /// Creates an Apple-backed transcriber.
    /// - Parameters:
    ///   - locale: The recognition locale.
    ///   - transcriptionOptions: Options that affect text.
    ///   - reportingOptions: Options that affect result delivery.
    ///   - attributeOptions: Attributes to include in result text.
    public convenience init(
        locale: Locale,
        transcriptionOptions: Set<TranscriptionOption>,
        reportingOptions: Set<ReportingOption>,
        attributeOptions: Set<ResultAttributeOption>
    ) {
        self.init(
            model: SystemSpeechModel(locale: locale), transcriptionOptions: transcriptionOptions,
            reportingOptions: reportingOptions, attributeOptions: attributeOptions
        )
    }

    /// Creates a transcriber with an explicit recognition provider.
    /// Construction does not load or download assets.
    /// Options the provider does not support fail before audio is consumed.
    /// - Parameters:
    ///   - model: The immutable provider configuration.
    ///   - transcriptionOptions: Options that affect text.
    ///   - reportingOptions: Options that affect result delivery.
    ///   - attributeOptions: Attributes to include in result text.
    public init(
        model: any SpeechRecognitionModel,
        transcriptionOptions: Set<TranscriptionOption> = [],
        reportingOptions: Set<ReportingOption> = [],
        attributeOptions: Set<ResultAttributeOption> = []
    ) {
        self.model = model
        self.configuration = .init(
            transcriptionOptions: transcriptionOptions, reportingOptions: reportingOptions,
            attributeOptions: attributeOptions
        )
        self.native = (model as? SystemSpeechModel).map {
            Speech.SpeechTranscriber(
                locale: $0.locale, transcriptionOptions: transcriptionOptions,
                reportingOptions: reportingOptions, attributeOptions: attributeOptions)
        }
    }

    /// Creates a transcriber with an explicit provider and a native preset.
    /// - Parameters:
    ///   - model: The immutable provider configuration.
    ///   - preset: The native preset.
    public convenience init(model: any SpeechRecognitionModel, preset: Preset) {
        self.init(
            model: model, transcriptionOptions: preset.transcriptionOptions,
            reportingOptions: preset.reportingOptions, attributeOptions: preset.attributeOptions)
    }

    /// Whether Apple's transcriber supports this device.
    public static var isAvailable: Bool { Speech.SpeechTranscriber.isAvailable }
    /// The locales supported by Apple's transcriber, without downloading assets.
    public static var supportedLocales: [Locale] {
        get async { await Speech.SpeechTranscriber.supportedLocales }
    }
    /// The installed locales for Apple's transcriber.
    public static var installedLocales: [Locale] {
        get async { await Speech.SpeechTranscriber.installedLocales }
    }

    /// Resolves a requested locale through Apple's native locale matching.
    /// - Parameter locale: The requested locale.
    /// - Returns: A supported equivalent locale, or `nil`.
    public static func supportedLocale(equivalentTo locale: Locale) async -> Locale? {
        await Speech.SpeechTranscriber.supportedLocale(equivalentTo: locale)
    }

    /// The recognition locales. A model with automatic language selection reports none.
    public var selectedLocales: [Locale] { native?.selectedLocales ?? [] }

    /// The audio formats this transcriber can analyze.
    public var availableCompatibleAudioFormats: [AVAudioFormat] {
        get async {
            if let native { return await native.availableCompatibleAudioFormats }
            return await model.compatibleAudioFormats(for: .transcription(configuration))
        }
    }

    /// Results in source order.
    ///
    /// Consume results concurrently with analysis.
    /// The buffer holds at most 256 results and 4 MiB of UTF-8 text.
    /// An unread volatile result is replaced when a newer result supersedes it.
    /// Final results are never dropped; overflow fails the session instead.
    /// If iteration stops early, explicitly cancel the analyzer.
    public var results: Results { Results(channel: channel) }

    var binding: ModuleBinding {
        let channel = channel
        let native = native
        return .transcription(
            owner: self, channel: channel, native: native, model: model,
            request: .transcription(configuration), convert: Result.init(_:)
        ) {
            guard let native else { return }
            for try await result in native.results {
                try channel.publish(
                    Result(
                        text: result.text, alternatives: result.alternatives, range: result.range,
                        resultsFinalizationTime: result.resultsFinalizationTime))
            }
        }
    }

    /// A phrase snapshot that preserves attributed text and source timing.
    public struct Result: SpeechModuleResult, ChannelResult, Sendable, CustomStringConvertible,
        Hashable
    {
        /// The range in the source recording.
        public let range: CMTimeRange
        /// The exclusive boundary before which no result can change.
        public let resultsFinalizationTime: CMTime
        /// The most likely text, with the backend's spacing and attributes.
        public let text: AttributedString
        /// Alternative interpretations, in backend order.
        public let alternatives: [AttributedString]

        /// Creates a result without rewriting text or manufacturing timing.
        /// - Parameters:
        ///   - text: The attributed phrase snapshot.
        ///   - alternatives: The backend's alternative interpretations.
        ///   - range: The source audio range.
        ///   - resultsFinalizationTime: The finalized audio boundary after this result.
        public init(
            text: AttributedString, alternatives: [AttributedString] = [], range: CMTimeRange,
            resultsFinalizationTime: CMTime
        ) {
            self.text = text
            self.alternatives = alternatives
            self.range = range
            self.resultsFinalizationTime = resultsFinalizationTime
        }

        init(_ result: SpeechRecognitionResult) {
            self.init(
                text: result.text, alternatives: result.alternatives, range: result.range,
                resultsFinalizationTime: result.resultsFinalizationTime)
        }

        public var description: String { String(text.characters) }
        var textByteCount: Int { AnySpeechRecognizer.textByteCount(text, alternatives) }
    }
}
