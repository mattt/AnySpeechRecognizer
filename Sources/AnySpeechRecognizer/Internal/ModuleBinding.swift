import CoreMedia
import Foundation
import Speech

/// How an analyzer drives one package module.
struct ModuleBinding: Sendable {
    enum Backend: Sendable {
        /// Apple's native module, sharing one native analyzer with other native modules.
        case native(NativeMember)
        /// A model executor that receives converted input.
        case model(any SpeechRecognitionModel, SpeechRecognitionRequest)
    }

    let id: ObjectIdentifier
    let backend: Backend
    /// Claims the module for one analyzer.
    let attach: @Sendable () -> Bool
    /// Publishes an executor result to the module's results.
    let deliver: @Sendable (SpeechRecognitionResult) throws -> Void
    /// Closes the module's results, with an error or by discarding buffered results.
    let finish: @Sendable (_ error: (any Error)?, _ discardBuffered: Bool) -> Void
}

/// A native Speech module and the task body that forwards its results.
struct NativeMember: Sendable {
    let module: any Speech.SpeechModule
    /// Forwards native results to the package module until the native stream ends.
    let forward: @Sendable () async throws -> Void
}

/// A package module that the analyzer can drive.
protocol AnalyzerModule: AnyObject, Sendable {
    var binding: ModuleBinding { get }
}

extension ModuleBinding {
    /// Builds a binding for a transcription-style module with a typed result channel.
    static func transcription<Result: ChannelResult>(
        owner: AnyObject, channel: ResultChannel<Result>, native: (any Speech.SpeechModule)?,
        model: any SpeechRecognitionModel, request: SpeechRecognitionRequest,
        convert: @escaping @Sendable (SpeechRecognitionResult) -> Result,
        forwardNative: @escaping @Sendable () async throws -> Void
    ) -> ModuleBinding {
        let backend: Backend =
            if let native {
                .native(NativeMember(module: native, forward: forwardNative))
            } else {
                .model(model, request)
            }
        return ModuleBinding(
            id: ObjectIdentifier(owner), backend: backend, attach: { channel.attach() },
            deliver: { try channel.publish(convert($0)) },
            finish: { channel.finish(throwing: $0, discardBuffered: $1) })
    }
}

/// Text size used for the result buffer limit.
func textByteCount(_ text: AttributedString, _ alternatives: [AttributedString]) -> Int {
    String(text.characters).utf8.count
        + alternatives.reduce(0) { $0 + String($1.characters).utf8.count }
}
