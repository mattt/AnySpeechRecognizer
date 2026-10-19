import AVFoundation
import CoreML
import CoreMedia
import Foundation
import Speech

/// NVIDIA's Nemotron 3.5 ASR Streaming model, run by the package's own runtime on Core ML.
///
/// Nemotron is a cache-aware streaming transducer: its encoder keeps the attention and
/// convolution caches of every layer, so it encodes each chunk of audio once, and every token
/// is final when it is decoded. The chunk size sets the latency, from 80 ms to 1.12 s.
/// A language prompt conditions recognition on one of 40 locales, or the model detects the
/// language. It supports file and live input, word timing, and word confidence derived from
/// token probabilities. Final results are phrases, as for Parakeet; with volatile results,
/// the decoded words of the current phrase are volatile until the phrase ends. They do not
/// change, but later tokens can continue the last word. It does not support fast results,
/// alternatives, etiquette replacement, or dictation.
///
/// No published repository has the converted models yet, so the model loads a staged
/// directory that `Tools/NemotronConversion` writes.
public struct NemotronSpeechModel: SpeechRecognitionModel {
    /// The language prompt.
    public enum Language: Sendable, Hashable {
        /// The model detects the language. It writes a language tag after each sentence,
        /// which the results omit.
        case automatic
        /// The model transcribes one language, such as `de-DE`.
        /// The locale must match a prompt of the model, by language and region or by language.
        case locale(Locale)
    }

    /// The streaming chunk size, which is also the encoder's latency.
    public enum ChunkSize: Int, Sendable, Codable, CaseIterable {
        case ms80 = 80
        case ms160 = 160
        case ms320 = 320
        case ms560 = 560
        case ms1120 = 1120

        /// The name of the encoder function for this chunk size.
        var function: String { "chunk_\(rawValue)ms" }
    }

    /// The source of model files.
    public enum Assets: Sendable, Hashable {
        /// A staged directory with `Encoder.mlmodelc`, `Decoder.mlmodelc`,
        /// `JointDecision.mlmodelc`, `vocab.json`, and `prompts.json`.
        /// The directory is used as is, without download or digest verification.
        case directory(URL)
    }

    /// The Core ML compute units for the encoder.
    public typealias ComputeUnits = ParakeetSpeechModel.ComputeUnits

    static let vocabularyName = "vocab.json"
    static let promptsName = "prompts.json"
    static let requiredFiles = [
        CoreMLNemotronEngine.encoderName, CoreMLNemotronEngine.decoderName,
        CoreMLNemotronEngine.jointName, vocabularyName, promptsName,
    ]

    /// The language prompt.
    public let language: Language
    /// The streaming chunk size.
    public let chunkSize: ChunkSize
    /// The source of model files.
    public let assets: Assets
    /// The compute units for the encoder.
    public let computeUnits: ComputeUnits

    public let modelID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
    public let capabilities = SpeechRecognitionCapabilities(
        transcription: .init(
            reportingOptions: [.volatileResults],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]),
        liveInput: true, languageControl: true)

    /// Creates a configuration without loading, inference, or downloads.
    /// - Parameters:
    ///   - language: The language prompt.
    ///   - chunkSize: The streaming chunk size. The staged encoder must have its function.
    ///   - assets: The source of model files.
    ///   - computeUnits: The compute units for the encoder.
    public init(
        language: Language = .automatic, chunkSize: ChunkSize = .ms560, assets: Assets,
        computeUnits: ComputeUnits = .cpuAndNeuralEngine
    ) {
        self.language = language
        self.chunkSize = chunkSize
        if case .directory(let url) = assets {
            self.assets = .directory(url.standardizedFileURL)
        } else {
            self.assets = assets
        }
        self.computeUnits = computeUnits
    }

    /// The license and attribution text that applications show.
    public var attribution: String {
        "Nemotron 3.5 ASR Streaming 0.6B by NVIDIA "
            + "(https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), "
            + "licensed under the OpenMDW License 1.1 (https://openmdw.ai/license/1-1/)."
    }

    var directory: URL {
        switch assets {
        case .directory(let url): url
        }
    }

    public func makeExecutor() -> any SpeechRecognitionExecutor { NemotronExecutor(model: self) }

    /// Reports whether the staged directory has every file.
    public func availability() async -> SpeechRecognitionAvailability {
        #if !arch(arm64)
            return .unsupportedHardware
        #else
            let directory = directory
            return Self.requiredFiles.allSatisfy {
                FileManager.default.fileExists(atPath: directory.appending(path: $0).path)
            } ? .ready : .missingAssets
        #endif
    }

    /// Returns `nil` when the staged directory is complete; the model has no installable assets.
    public func assetInstallationRequest() async throws -> AssetInstallationRequest? {
        switch await availability() {
        case .ready: return nil
        case .unsupportedHardware:
            throw SpeechRecognitionError.unavailableProvider(
                "The Nemotron runtime requires Apple silicon.")
        default:
            throw SpeechRecognitionError.missingAssets(
                "Stage the Nemotron encoder, decoder, joint, vocabulary, and prompts in "
                    + "\(directory.path).")
        }
    }

    public func assetInstallationRequest(supporting requests: [SpeechRecognitionRequest])
        async throws -> AssetInstallationRequest?
    {
        try await assetInstallationRequest()
    }

    /// The prompt slot for the language, from the staged `prompts.json`.
    func promptSlot(in prompts: [String: Int]) throws -> Int {
        switch language {
        case .automatic:
            guard let slot = prompts["auto"] else {
                throw SpeechRecognitionError.missingAssets(
                    "The Nemotron prompts have no auto slot.")
            }
            return slot
        case .locale(let locale):
            let language = locale.language.languageCode?.identifier
            let region = locale.region?.identifier
            let candidates = [language.flatMap { l in region.map { "\(l)-\($0)" } }, language]
            for case let key? in candidates {
                if let slot = prompts[key] { return slot }
            }
            throw SpeechRecognitionError.unsupportedLanguage(
                "Nemotron has no prompt for \(locale.identifier).")
        }
    }

    static func prompts(contentsOf url: URL) throws -> [String: Int] {
        guard let data = try? Data(contentsOf: url),
            let prompts = try? JSONDecoder().decode([String: Int].self, from: data)
        else {
            throw SpeechRecognitionError.missingAssets("Missing Nemotron prompts \(url.path).")
        }
        return prompts
    }

    /// Reports the staged directory, chunk size, language, and compute units.
    public func diagnostics() async -> SpeechRecognitionDiagnostics {
        let languageDescription: String
        switch language {
        case .automatic: languageDescription = "automatic language detection"
        case .locale(let locale): languageDescription = "language \(locale.identifier)"
        }
        var requestedLocale: String?
        if case .locale(let locale) = language { requestedLocale = locale.identifier }
        return .init(
            provider: "AnySpeechRecognizer Nemotron", modelID: modelID, runtimeVersion: "Core ML",
            modelRepository: directory.path, modelRevision: nil,
            computeConfiguration:
                "encoder \(computeUnits.rawValue); decoder and joint cpuAndNeuralEngine; "
                + "\(chunkSize.rawValue) ms chunks (\(chunkSize.function)); \(languageDescription)",
            requestedLocale: requestedLocale, effectiveLocale: requestedLocale,
            capabilities: capabilities)
    }
}

/// Runs one Nemotron recognizer for one module, with source-time mapping and phrase results.
actor NemotronExecutor: SpeechRecognitionExecutor {
    let model: NemotronSpeechModel
    private var engine: (any NemotronInferenceEngine)?
    private var vocabulary: ParakeetVocabulary?
    private var prompt = 0
    private var reportsVolatileResults = false
    private var recognizer: NemotronRecognizer?
    private var builder: ParakeetResultBuilder?
    private var request: SpeechRecognitionRequest?
    private var results: SpeechRecognitionResultSink?
    /// The next expected buffer start in the current segment.
    private var expected: CMTime?
    /// The emission of the latest publication's results, which keeps results in source
    /// order when a finalization runs while ``append(_:)`` waits to emit; see
    /// ``ParakeetExecutor``.
    private var emission: Task<Void, any Error>?
    private var cancelled = false

    init(model: NemotronSpeechModel) { self.model = model }

    func prepare(
        request: SpeechRecognitionRequest, format: AVAudioFormat, context: AnalysisContext
    ) async throws {
        try request.validate(for: model.capabilities)
        guard format.sampleRate == 16_000, format.channelCount == 1,
            format.commonFormat == .pcmFormatFloat32, !format.isInterleaved
        else {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "Nemotron requires 16 kHz mono Float32 input.")
        }
        guard case .transcription = request else {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "Nemotron does not support DictationTranscriber.")
        }
        let directory = model.directory
        let prompt = try model.promptSlot(
            in: NemotronSpeechModel.prompts(
                contentsOf: directory.appending(path: NemotronSpeechModel.promptsName)))
        var configuration = CoreMLNemotronEngine.Configuration(function: model.chunkSize.function)
        configuration.encoderComputeUnits = model.computeUnits.coreML
        let (engine, vocabulary) = try await Task.detached { [configuration] in
            let engine = try CoreMLNemotronEngine(
                directory: directory, configuration: configuration)
            let vocabulary = try ParakeetVocabulary(
                contentsOf: directory.appending(path: NemotronSpeechModel.vocabularyName))
            return (UncheckedNemotronEngine(engine), vocabulary)
        }.value
        try Task.checkCancellation()
        guard !cancelled else { throw CancellationError() }
        use(engine.value, vocabulary: vocabulary, request: request, prompt: prompt)
    }

    /// Serves a validated request with a loaded engine. Tests call this method directly
    /// with a fake engine.
    func use(
        _ engine: sending any NemotronInferenceEngine, vocabulary: ParakeetVocabulary,
        request: SpeechRecognitionRequest, prompt: Int
    ) {
        self.engine = engine
        self.vocabulary = vocabulary
        self.request = request
        self.prompt = prompt
        reportsVolatileResults = request.includesVolatileResults
    }

    func analyze(_ input: SpeechAnalysisInput, results: SpeechRecognitionResultSink) async throws {
        self.results = results
        for try await buffer in input {
            guard !cancelled else { throw CancellationError() }
            try await append(buffer)
        }
    }

    private func append(_ buffer: TimedAudioBuffer) async throws {
        guard let samples = buffer.buffer.floatChannelData?[0] else {
            throw SpeechRecognitionError.invalidAudio("Nemotron requires Float32 samples.")
        }
        // A gap in the input finalizes the current segment and starts a new one.
        if let expected, buffer.startTime > expected + CMTime(value: 1, timescale: 16_000) {
            try await flushSegment()
        }
        if recognizer == nil { try startSegment(at: buffer.startTime) }
        let count = Int(buffer.buffer.frameLength)
        recognizer?.append(UnsafeBufferPointer(start: samples, count: count))
        builder?.receivedSamples += count
        expected = buffer.endTime
        try await publish(recognizer!.process(), flush: false)
    }

    private func startSegment(at origin: CMTime) throws {
        guard let engine, let vocabulary, let request else { throw CancellationError() }
        recognizer = try NemotronRecognizer(engine: engine, vocabulary: vocabulary, prompt: prompt)
        if builder == nil {
            builder = ParakeetResultBuilder(
                vocabulary: vocabulary, origin: origin,
                includeTiming: request.includesAudioTimeRange,
                includeConfidence: request.includesConfidence)
        } else {
            builder?.restart(at: origin)
        }
    }

    private func flushSegment() async throws {
        guard let recognizer else { return }
        try await publish(recognizer.process(endOfInput: true), flush: true, segmentEnded: true)
        self.recognizer = nil
        expected = nil
    }

    private func publish(_ update: ParakeetUpdate, flush: Bool, segmentEnded: Bool = false)
        async throws
    {
        guard let results, var builder else { return }
        // Decoded tokens never change, so an update without tokens changes no result.
        guard flush || !update.committed.isEmpty else { return }
        let finals = builder.commit(update.committed, flush: flush)
        // Committed words that later tokens can continue are the only uncommitted text.
        let volatile =
            reportsVolatileResults && !segmentEnded ? builder.volatileResult([]) : nil
        self.builder = builder
        let reportsVolatileResults = reportsVolatileResults
        let previous = emission
        let task = Task {
            _ = await previous?.result
            for result in finals { try await results.emit(result) }
            if let volatile {
                try await results.emit(volatile)
                results.setVolatileRange(volatile.range)
            } else if reportsVolatileResults {
                results.setVolatileRange(nil)
            }
        }
        emission = task
        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    func finalize(through time: CMTime) async throws {
        guard let recognizer, let builder, !cancelled else { return }
        let offset = time - builder.origin
        guard offset >= .zero else { return }
        // Finalization is inclusive, so commit through the sample after `time`.
        let sample =
            Int(
                CMTimeConvertScale(offset, timescale: 16_000, method: .roundTowardPositiveInfinity)
                    .value) + 1
        try await publish(recognizer.process(finalizeThrough: sample), flush: true)
    }

    func cancel() async {
        cancelled = true
        recognizer = nil
        engine = nil
        results = nil
    }
}

/// Moves a newly loaded engine to its owning executor.
/// The engine is created on a background task and used only by that executor afterward.
private struct UncheckedNemotronEngine: @unchecked Sendable {
    let value: CoreMLNemotronEngine
    init(_ value: CoreMLNemotronEngine) { self.value = value }
}
