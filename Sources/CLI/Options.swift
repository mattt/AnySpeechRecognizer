import AnySpeechRecognizer
import ArgumentParser
import Foundation

enum Backend: String, CaseIterable, ExpressibleByArgument {
    case apple, parakeet, nemotron
}

/// Options that select a speech recognition model.
struct ModelOptions: ParsableArguments {
    @Option(help: "The speech recognition backend.")
    var backend: Backend

    @Option(
        help: ArgumentHelp(
            "Apple recognition locale (default: en-US), or Nemotron's language "
                + "(default: automatic detection).",
            valueName: "identifier"))
    var locale: String?

    @Option(help: "Parakeet model variant (default: v3).")
    var version: ParakeetSpeechModel.Variant?

    @Option(
        help: ArgumentHelp(
            "A staged Parakeet or Nemotron directory, used without download. "
                + "Nemotron requires one.",
            valueName: "path"))
    var modelDirectory: String?

    @Option(
        help: ArgumentHelp("Nemotron chunk size in milliseconds (default: 560).", valueName: "ms"))
    var chunkSize: NemotronSpeechModel.ChunkSize?

    func validate() throws {
        if backend == .parakeet, locale != nil {
            throw ValidationError(
                "Parakeet selects language automatically and does not accept --locale.")
        }
        if backend == .apple, modelDirectory != nil || version != nil {
            throw ValidationError("Model directories and versions apply to Parakeet only.")
        }
        if backend == .nemotron, modelDirectory == nil {
            throw ValidationError("Nemotron requires --model-directory.")
        }
        if backend == .nemotron, version != nil {
            throw ValidationError("Versions apply to Parakeet only.")
        }
        if backend != .nemotron, chunkSize != nil {
            throw ValidationError("Chunk sizes apply to Nemotron only.")
        }
    }

    func model() -> any SpeechRecognitionModel {
        switch backend {
        case .apple:
            SystemSpeechModel(locale: Locale(identifier: locale ?? "en-US"))
        case .parakeet:
            ParakeetSpeechModel(
                variant: version ?? .v3,
                assets: modelDirectory.map { .directory(URL(fileURLWithPath: $0)) } ?? .builtIn)
        case .nemotron:
            NemotronSpeechModel(
                language: locale.map { .locale(Locale(identifier: $0)) } ?? .automatic,
                chunkSize: chunkSize ?? .ms560,
                assets: .directory(URL(fileURLWithPath: modelDirectory ?? ".")))
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

    @Flag(
        help: """
            With --live, read each block when its audio would have been captured, \
            so result times measure latency.
            """)
    var realTime = false

    func validate() throws {
        if realTime, !live { throw ValidationError("--real-time requires --live.") }
    }

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

extension ParakeetSpeechModel.Variant: ExpressibleByArgument {}
extension NemotronSpeechModel.ChunkSize: ExpressibleByArgument {}
extension SortformerDiarizationModel.Version: ExpressibleByArgument {}
extension SortformerDiarizationModel.ComputeUnits: ExpressibleByArgument {
    public static var allValueStrings: [String] {
        [Self.cpuOnly, .cpuAndNeuralEngine, .cpuAndGPU, .all].map(\.rawValue)
    }
}
