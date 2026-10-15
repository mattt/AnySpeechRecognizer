import AVFoundation
import CoreMedia
import Foundation
import Speech

/// A dictation module with Apple's system backend or an explicitly selected model.
/// Each instance belongs to one analyzer and permits one result iterator.
public final class DictationTranscriber: LocaleDependentSpeechModule, AnalyzerModule {
    /// Native hints about the expected speech.
    public typealias ContentHint = Speech.DictationTranscriber.ContentHint
    /// Native options that affect dictation text.
    public typealias TranscriptionOption = Speech.DictationTranscriber.TranscriptionOption
    /// Native options that affect result delivery.
    public typealias ReportingOption = Speech.DictationTranscriber.ReportingOption
    /// Native options that select result attributes.
    public typealias ResultAttributeOption = Speech.DictationTranscriber.ResultAttributeOption
    /// Native preset values.
    public typealias Preset = Speech.DictationTranscriber.Preset
    /// The results of one dictation transcriber.
    public typealias Results = SpeechModuleResults<Result>

    /// The immutable backend configuration.
    public let model: any SpeechRecognitionModel
    /// The requested dictation options.
    public let configuration: DictationRecognitionConfiguration
    let channel = ResultChannel<Result>()
    let native: Speech.DictationTranscriber?

    /// Creates an Apple-backed dictation transcriber with a native preset.
    /// - Parameters:
    ///   - locale: The recognition locale.
    ///   - preset: The native preset.
    public convenience init(locale: Locale, preset: Preset) {
        self.init(
            locale: locale, contentHints: preset.contentHints,
            transcriptionOptions: preset.transcriptionOptions,
            reportingOptions: preset.reportingOptions, attributeOptions: preset.attributeOptions)
    }

    /// Creates an Apple-backed dictation transcriber.
    /// - Parameters:
    ///   - locale: The recognition locale.
    ///   - contentHints: Hints about the expected speech.
    ///   - transcriptionOptions: Options that affect text.
    ///   - reportingOptions: Options that affect result delivery.
    ///   - attributeOptions: Attributes to include in result text.
    public convenience init(
        locale: Locale, contentHints: Set<ContentHint>,
        transcriptionOptions: Set<TranscriptionOption>, reportingOptions: Set<ReportingOption>,
        attributeOptions: Set<ResultAttributeOption>
    ) {
        self.init(
            model: SystemSpeechModel(locale: locale), contentHints: contentHints,
            transcriptionOptions: transcriptionOptions, reportingOptions: reportingOptions,
            attributeOptions: attributeOptions)
    }

    /// Creates a dictation transcriber with an explicit recognition provider.
    /// Construction does not load or download assets.
    /// Options the provider does not support fail before audio is consumed.
    /// - Parameters:
    ///   - model: The immutable provider configuration.
    ///   - contentHints: Hints about the expected speech.
    ///   - transcriptionOptions: Options that affect text.
    ///   - reportingOptions: Options that affect result delivery.
    ///   - attributeOptions: Attributes to include in result text.
    public init(
        model: any SpeechRecognitionModel, contentHints: Set<ContentHint> = [],
        transcriptionOptions: Set<TranscriptionOption> = [],
        reportingOptions: Set<ReportingOption> = [],
        attributeOptions: Set<ResultAttributeOption> = []
    ) {
        self.model = model
        self.configuration = .init(
            contentHints: contentHints, transcriptionOptions: transcriptionOptions,
            reportingOptions: reportingOptions, attributeOptions: attributeOptions)
        self.native = (model as? SystemSpeechModel).map {
            Speech.DictationTranscriber(
                locale: $0.locale, contentHints: contentHints,
                transcriptionOptions: transcriptionOptions, reportingOptions: reportingOptions,
                attributeOptions: attributeOptions)
        }
    }

    /// The locales supported by Apple's dictation transcriber.
    public static var supportedLocales: [Locale] {
        get async { await Speech.DictationTranscriber.supportedLocales }
    }
    /// The installed locales for Apple's dictation transcriber.
    public static var installedLocales: [Locale] {
        get async { await Speech.DictationTranscriber.installedLocales }
    }

    /// Resolves a requested locale through Apple's native locale matching.
    /// - Parameter locale: The requested locale.
    /// - Returns: A supported equivalent locale, or `nil`.
    public static func supportedLocale(equivalentTo locale: Locale) async -> Locale? {
        await Speech.DictationTranscriber.supportedLocale(equivalentTo: locale)
    }

    /// The recognition locales. A model with automatic language selection reports none.
    public var selectedLocales: [Locale] { native?.selectedLocales ?? [] }

    /// The audio formats this transcriber can analyze.
    public var availableCompatibleAudioFormats: [AVAudioFormat] {
        get async {
            if let native { return await native.availableCompatibleAudioFormats }
            return await model.compatibleAudioFormats(for: .dictation(configuration))
        }
    }

    /// Results in source order, with the same buffering rules as ``SpeechTranscriber/results``.
    public var results: Results { Results(channel: channel) }

    var binding: ModuleBinding {
        let channel = channel
        let native = native
        return .transcription(
            owner: self, channel: channel, native: native, model: model,
            request: .dictation(configuration), convert: Result.init(_:)
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

    /// A dictation snapshot that preserves attributed text and source timing.
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
        ///   - text: The attributed dictation snapshot.
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
