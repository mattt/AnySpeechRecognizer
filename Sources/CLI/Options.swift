import AnySpeechRecognizer
import ArgumentParser
import Foundation

enum Backend: String, CaseIterable, ExpressibleByArgument {
    case apple, parakeet, qwen
}

/// Options that select a speech recognition model.
struct ModelOptions: ParsableArguments {
    /// The names that `--version` accepts for each backend, such as `v3, ultra, unified`.
    static let parakeetVariants = ParakeetSpeechModel.Variant.allCases.map(\.rawValue)
        .joined(separator: ", ")
    static let qwenSizes = QwenSpeechModel.Variant.allCases.map(\.rawValue)
        .joined(separator: ", ")

    @Option(help: "The speech recognition backend.")
    var backend: Backend

    @Option(
        help: ArgumentHelp(
            "Apple recognition locale (default: en-US), or a language that Qwen3-ASR must use "
                + "(default: automatic).",
            valueName: "identifier"))
    var locale: String?

    @Option(
        help: ArgumentHelp(
            "Parakeet variant (\(Self.parakeetVariants); default: v3) "
                + "or Qwen3-ASR size (\(Self.qwenSizes); default: 1.7b).",
            valueName: "version"))
    var version: String?

    @Option(
        help: ArgumentHelp(
            "A staged Parakeet or Qwen3-ASR directory, used without download.",
            valueName: "path"))
    var modelDirectory: String?

    func validate() throws {
        switch backend {
        case .apple:
            if modelDirectory != nil || version != nil {
                throw ValidationError("Model directories and versions do not apply to Apple.")
            }
        case .parakeet:
            if locale != nil {
                throw ValidationError(
                    "Parakeet selects language automatically and does not accept --locale.")
            }
            if let version, ParakeetSpeechModel.Variant(rawValue: version) == nil {
                throw ValidationError("Parakeet variants are \(Self.parakeetVariants).")
            }
        case .qwen:
            if let version, QwenSpeechModel.Variant(rawValue: version) == nil {
                throw ValidationError("Qwen3-ASR sizes are \(Self.qwenSizes).")
            }
        }
    }

    func model() -> any SpeechRecognitionModel {
        switch backend {
        case .apple:
            SystemSpeechModel(locale: Locale(identifier: locale ?? "en-US"))
        case .parakeet:
            ParakeetSpeechModel(
                variant: version.flatMap(ParakeetSpeechModel.Variant.init(rawValue:)) ?? .v3,
                assets: modelDirectory.map { .directory(URL(fileURLWithPath: $0)) } ?? .builtIn)
        case .qwen:
            QwenSpeechModel(
                variant: version.flatMap(QwenSpeechModel.Variant.init(rawValue:)) ?? .large,
                assets: modelDirectory.map { .directory(URL(fileURLWithPath: $0)) } ?? .builtIn,
                language: locale.map { Locale.Language(identifier: $0) })
        }
    }
}

/// Options that select the transcriber's reporting options.
struct ResultOptions: ParsableArguments {
    @Flag(
        help: """
            Request volatile results, as for live audio; transcribe and evaluate read the file \
            in 100 ms blocks.
            """)
    var live = false

    @Flag(help: "Request fast results, which trade accuracy for latency.")
    var fast = false

    var reportingOptions: Set<SpeechTranscriber.ReportingOption> {
        var options: Set<SpeechTranscriber.ReportingOption> = live ? [.volatileResults] : []
        if fast { options.insert(.fastResults) }
        return options
    }
}

/// Options that add and configure a Sortformer speaker diarizer.
struct DiarizerOptions: ParsableArguments {
    @Flag(help: "Add a Sortformer speaker diarizer.")
    var diarize = false

    @Option(
        help: ArgumentHelp(
            "Sortformer model version (default: v2). v2.1 is under the NVIDIA Open Model License.",
            valueName: "version"))
    var diarizerVersion: SortformerDiarizationModel.Version?

    @Option(
        help: ArgumentHelp(
            "A staged compiled Sortformer model (.mlmodelc) of the selected version.",
            valueName: "path"))
    var diarizerModel: String?

    @Option(help: ArgumentHelp("Core ML compute units for the diarizer.", valueName: "units"))
    var diarizerComputeUnits: SortformerDiarizationModel.ComputeUnits = .cpuAndNeuralEngine

    @Option(
        help: ArgumentHelp(
            "Diarizer post-processing (default: callHome); none uses 0.5 thresholds.",
            valueName: "preset"))
    var diarizerPostProcessing: PostProcessingPreset?

    func validate() throws {
        guard
            diarize
                || (diarizerModel == nil && diarizerVersion == nil
                    && diarizerPostProcessing == nil)
        else {
            throw ValidationError(
                "--diarizer-model, --diarizer-version, and --diarizer-post-processing "
                    + "require --diarize.")
        }
    }

    /// A speaker diarizer with a built-in Sortformer model or a staged compiled model,
    /// or `nil` without `--diarize`.
    func diarizer(live: Bool) -> SpeakerDiarizer? {
        guard diarize else { return nil }
        let model = SortformerDiarizationModel(
            version: diarizerVersion ?? .v2,
            assets: diarizerModel.map { .compiledModel(URL(fileURLWithPath: $0)) } ?? .builtIn,
            computeUnits: diarizerComputeUnits,
            postProcessing: (diarizerPostProcessing ?? .callHome).postProcessing)
        return SpeakerDiarizer(model: model, reportingOptions: live ? [.volatileResults] : [])
    }
}

enum PostProcessingPreset: String, CaseIterable, ExpressibleByArgument {
    case callHome, dihard3, none

    var postProcessing: SortformerDiarizationModel.PostProcessing {
        switch self {
        case .callHome: .callHome
        case .dihard3: .dihard3
        case .none: .none
        }
    }
}

extension SortformerDiarizationModel.Version: ExpressibleByArgument {}
extension SortformerDiarizationModel.ComputeUnits: ExpressibleByArgument {
    public static var allValueStrings: [String] {
        [Self.cpuOnly, .cpuAndNeuralEngine, .cpuAndGPU, .all].map(\.rawValue)
    }
}
