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

extension SpeechRecognitionError {
    /// What went wrong. More kinds can be added, so a `switch` over them needs a `default`.
    ///
    /// - Note: This API is exclusive to AnySpeechRecognizer
    ///   and using it means your code is no longer drop-in compatible with the Speech framework.
    public struct Kind: Hashable, Sendable, CustomStringConvertible {
        public let description: String

        private init(_ description: String) {
            self.description = description
        }

        /// The provider does not support a requested module, option, or operation.
        public static let unsupportedConfiguration = Kind("Unsupported configuration")
        /// The selected backend cannot run on this device.
        public static let unavailableProvider = Kind("Unavailable provider")
        /// The backend cannot recognize the requested locale.
        public static let unsupportedLanguage = Kind("Unsupported language")
        /// Required assets are absent.
        public static let missingAssets = Kind("Missing assets")
        /// A downloaded model file does not match its expected size or digest.
        public static let assetVerificationFailed = Kind("Asset verification failed")
        /// The audio format, duration, or sample data is invalid.
        public static let invalidAudio = Kind("Invalid audio")
        /// An operation conflicts with the current session state.
        public static let invalidSessionState = Kind("Invalid session state")
        /// The consumer did not drain the finite result buffer.
        public static let bufferOverflow = Kind("Buffer overflow")
        /// A backend returned inconsistent text or timing.
        public static let invalidResult = Kind("Invalid result")
        /// An upstream operation failed.
        public static let backendFailure = Kind("Backend failure")
    }

    /// What went wrong.
    ///
    /// - Note: This API is exclusive to AnySpeechRecognizer
    ///   and using it means your code is no longer drop-in compatible with the Speech framework.
    public var kind: Kind {
        switch self {
        case .unsupportedConfiguration: .unsupportedConfiguration
        case .unavailableProvider: .unavailableProvider
        case .unsupportedLanguage: .unsupportedLanguage
        case .missingAssets: .missingAssets
        case .assetVerificationFailed: .assetVerificationFailed
        case .invalidAudio: .invalidAudio
        case .invalidSessionState: .invalidSessionState
        case .bufferOverflow: .bufferOverflow
        case .invalidResult: .invalidResult
        case .backendFailure: .backendFailure
        }
    }

    /// A description of the error, which ``errorDescription`` also returns.
    ///
    /// - Note: This API is exclusive to AnySpeechRecognizer
    ///   and using it means your code is no longer drop-in compatible with the Speech framework.
    public var message: String {
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

    /// The error that caused this one: the wrapped error of a backend failure, and `nil` otherwise.
    ///
    /// - Note: This API is exclusive to AnySpeechRecognizer
    ///   and using it means your code is no longer drop-in compatible with the Speech framework.
    public var underlyingError: (any Error)? {
        if case .backendFailure(let error) = self { return error }
        return nil
    }
}

extension SpeechRecognitionError: LocalizedError {
    public var errorDescription: String? { message }
}

func recognitionError(_ error: any Error) -> any Error {
    if error is CancellationError || error is SpeechRecognitionError { return error }
    if let error = error as? ModelAssetError, case .verificationFailed = error {
        return SpeechRecognitionError.assetVerificationFailed(error.localizedDescription)
    }
    return SpeechRecognitionError.backendFailure(underlying: error)
}
