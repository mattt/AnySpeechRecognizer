import Foundation

/// An error in configuration, audio, session control, or a recognition backend.
/// Task cancellation uses `CancellationError` instead.
public enum SpeechRecognitionError: Error, Sendable {
    /// The selected provider does not support a requested module, option, or operation.
    case unsupportedConfiguration(String)
    /// The selected backend cannot run on this device.
    case unavailableProvider(String)
    /// The backend cannot recognize the requested locale.
    case unsupportedLanguage(String)
    /// Required assets are absent; install them before analysis.
    case missingAssets(String)
    /// A downloaded model file does not match its expected size or digest.
    /// The message names the file.
    case assetVerificationFailed(String)
    /// The audio format, duration, or sample data is invalid.
    case invalidAudio(String)
    /// An operation conflicts with the current session state.
    case invalidSessionState(String)
    /// The consumer did not drain the finite result buffer.
    case bufferOverflow
    /// A backend returned inconsistent text or timing.
    case invalidResult(String)
    /// An upstream operation failed; its original error is retained.
    case backendFailure(underlying: any Error)
}

extension SpeechRecognitionError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration(let reason), .unavailableProvider(let reason),
            .unsupportedLanguage(let reason), .missingAssets(let reason),
            .assetVerificationFailed(let reason),
            .invalidAudio(let reason), .invalidSessionState(let reason), .invalidResult(let reason):
            return reason
        case .bufferOverflow:
            return
                "The result buffer is full. Consume results during analysis, or cancel the analyzer."
        case .backendFailure(let error):
            return "Speech recognition failed: \(error.localizedDescription)"
        }
    }
}

func recognitionError(_ error: any Error) -> any Error {
    if error is CancellationError || error is SpeechRecognitionError { return error }
    return SpeechRecognitionError.backendFailure(underlying: error)
}
