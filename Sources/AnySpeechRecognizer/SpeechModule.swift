import AVFoundation
import CoreMedia
import Foundation
import Speech

/// An analysis module that a ``SpeechAnalyzer`` can drive.
///
/// The package provides ``SpeechTranscriber``, ``DictationTranscriber``, and ``SpeechDetector``.
/// The analyzer accepts only these package modules.
/// Package modules do not conform to `Speech.SpeechModule`.
public protocol SpeechModule: AnyObject, Sendable {
    /// The asynchronous sequence of results.
    associatedtype Results: Sendable, AsyncSequence where Results.Failure == any Error
    /// The type of each result.
    associatedtype Result: SpeechModuleResult, Sendable where Result == Results.Element

    /// Results in source order, as they are created.
    var results: Results { get }
    /// The audio formats this module can analyze with its configuration.
    var availableCompatibleAudioFormats: [AVAudioFormat] { get async }
}

/// A module whose behavior depends on a recognition locale.
public protocol LocaleDependentSpeechModule: SpeechModule {
    /// The locales the module's backend supports.
    static var supportedLocales: [Locale] { get async }
    /// Resolves a requested locale to an equivalent supported locale.
    /// - Parameter locale: The requested locale.
    /// - Returns: A supported equivalent locale, or `nil`.
    static func supportedLocale(equivalentTo locale: Locale) async -> Locale?
    /// The locales this instance recognizes. Automatic language selection reports none.
    var selectedLocales: [Locale] { get }
}

/// A result that covers a range of the source audio.
public protocol SpeechModuleResult {
    /// The range in the source recording.
    var range: CMTimeRange { get }
    /// The exclusive boundary before which no result can change.
    var resultsFinalizationTime: CMTime { get }
}

extension SpeechModuleResult {
    /// Whether the result is final.
    /// A volatile result is replaced by later results for the same audio.
    public var isFinal: Bool { range.end <= resultsFinalizationTime }
}

/// Process-wide model retention control.
public enum SpeechModels {
    /// Releases models retained by ``SpeechAnalyzer/Options/ModelRetention/processLifetime``
    /// or ``SpeechAnalyzer/Options/ModelRetention/lingering`` once they are no longer in use.
    public static func endRetention() async {
        await Speech.SpeechModels.endRetention()
        await ModelRetention.shared.endRetention()
    }
}

/// Retention hooks for package-owned model caches.
actor ModelRetention {
    static let shared = ModelRetention()
    private var handlers: [String: @Sendable () async -> Void] = [:]

    func register(_ key: String, handler: @escaping @Sendable () async -> Void) {
        handlers[key] = handler
    }

    func endRetention() async {
        for handler in handlers.values { await handler() }
    }
}

/// Native audio input with an optional explicit start time.
public typealias AnalyzerInput = Speech.AnalyzerInput
/// Native contextual strings and user data for analysis modules.
public typealias AnalysisContext = Speech.AnalysisContext
