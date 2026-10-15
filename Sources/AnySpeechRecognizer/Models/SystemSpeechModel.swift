import AVFoundation
import CoreMedia
import Foundation
import Speech

/// Immutable configuration for Apple's OS-managed speech models.
///
/// Modules with this model run on Apple's native analyzer.
/// All Apple-backed modules in one ``SpeechAnalyzer`` share one native analyzer,
/// so native behavior across modules is preserved.
public struct SystemSpeechModel: SpeechRecognitionModel {
    /// The requested recognition locale.
    public let locale: Locale
    public var modelID: String { "apple/speech/\(locale.identifier)" }
    public let capabilities = SpeechRecognitionCapabilities(
        transcription: .all, dictation: .all, liveInput: true, languageControl: true)

    /// Creates an Apple model configuration without loading or downloading assets.
    /// - Parameter locale: The recognition locale.
    public init(locale: Locale) { self.locale = locale }

    /// Returns an executor that rejects use outside an analyzer.
    /// The analyzer drives Apple-backed modules through Apple's native analyzer instead.
    public func makeExecutor() -> any SpeechRecognitionExecutor { SystemSpeechExecutor() }

    func nativeModule(for request: SpeechRecognitionRequest) -> any Speech.SpeechModule {
        switch request {
        case .transcription(let configuration):
            Speech.SpeechTranscriber(
                locale: locale, transcriptionOptions: configuration.transcriptionOptions,
                reportingOptions: configuration.reportingOptions,
                attributeOptions: configuration.attributeOptions)
        case .dictation(let configuration):
            Speech.DictationTranscriber(
                locale: locale, contentHints: configuration.contentHints,
                transcriptionOptions: configuration.transcriptionOptions,
                reportingOptions: configuration.reportingOptions,
                attributeOptions: configuration.attributeOptions)
        }
    }

    public func compatibleAudioFormats(for request: SpeechRecognitionRequest) async
        -> [AVAudioFormat]
    {
        await nativeModule(for: request).availableCompatibleAudioFormats
    }

    public func availability() async -> SpeechRecognitionAvailability {
        guard Speech.SpeechTranscriber.isAvailable else { return .unsupportedHardware }
        guard await Speech.SpeechTranscriber.supportedLocale(equivalentTo: locale) != nil else {
            return .unsupportedLanguage
        }
        let module = Speech.SpeechTranscriber(locale: locale, preset: .transcription)
        return Self.availability(await Speech.AssetInventory.status(forModules: [module]))
    }

    static func availability(_ status: Speech.AssetInventory.Status)
        -> SpeechRecognitionAvailability
    {
        switch status {
        case .installed: .ready
        case .downloading: .preparing
        case .supported: .missingAssets
        case .unsupported: .unsupportedLanguage
        @unknown default: .unsupportedHardware
        }
    }

    public func assetInstallationRequest() async throws -> AssetInstallationRequest? {
        switch await availability() {
        case .unsupportedHardware:
            throw SpeechRecognitionError.unavailableProvider(
                "Apple Speech is unavailable on this device.")
        case .unsupportedLanguage:
            throw SpeechRecognitionError.unsupportedLanguage(
                "Apple Speech does not support \(locale.identifier).")
        default: break
        }
        let module = Speech.SpeechTranscriber(locale: locale, preset: .transcription)
        guard
            let request = try await Speech.AssetInventory.assetInstallationRequest(supporting: [
                module
            ])
        else { return nil }
        return AssetInstallationRequest(native: request)
    }

    public func diagnostics() async -> SpeechRecognitionDiagnostics {
        let effective = await Speech.SpeechTranscriber.supportedLocale(equivalentTo: locale)
        return .init(
            provider: "Apple Speech", modelID: modelID, computeConfiguration: "OS-managed",
            requestedLocale: locale.identifier, effectiveLocale: effective?.identifier,
            capabilities: capabilities)
    }
}

/// Rejects direct use; Apple-backed modules run on the analyzer's native engine.
private struct SystemSpeechExecutor: SpeechRecognitionExecutor {
    func prepare(
        request: SpeechRecognitionRequest, format: AVAudioFormat, context: AnalysisContext
    ) async throws {
        throw SpeechRecognitionError.unsupportedConfiguration(
            "Apple Speech modules run only inside a SpeechAnalyzer.")
    }
    func analyze(_ input: SpeechAnalysisInput, results: SpeechRecognitionResultSink) async throws {
        throw CancellationError()
    }
    func finalize(through time: CMTime) async throws { throw CancellationError() }
    func cancel() async {}
}
