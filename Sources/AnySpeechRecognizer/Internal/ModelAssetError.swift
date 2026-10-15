import Foundation

/// A failure to install or verify Parakeet model files.
/// `verificationFailed` becomes `SpeechRecognitionError.assetVerificationFailed`.
enum ModelAssetError: Error, Sendable, Equatable {
    /// A manifest has an unsafe path, a malformed revision, or a malformed digest.
    case invalidManifest(String)
    /// A downloaded file's size or SHA-256 digest does not match its manifest entry,
    /// or the file is not in the cache at the pinned revision.
    case verificationFailed(path: String, reason: String)
}

extension ModelAssetError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .invalidManifest(let reason):
            return reason
        case .verificationFailed(let path, let reason):
            return "The model file \(path) failed verification: \(reason)"
        }
    }
}
