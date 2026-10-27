import AVFoundation
import CoreML
import CoreMedia
import Foundation
import Speech

/// A Parakeet TDT or RNN-T model run by the package's own runtime on Core ML.
///
/// Parakeet selects the language automatically and reports no detected language,
/// so the model has no locale parameter.
/// It supports file and live input, volatile results, word timing,
/// and word confidence derived from token probabilities.
/// It does not support alternatives, etiquette replacement, or dictation.
///
/// Built-in assets install on demand.
/// Every request needs the 15-second encoder, decoder, joint, and vocabulary.
/// Requests for volatile or fast results also use a shorter-window live encoder,
/// which ``AssetInventory`` installs for modules with those options.
/// Until the live encoder is installed and verified, those requests run with the
/// 15-second encoder instead.
public struct ParakeetSpeechModel: SpeechRecognitionModel {
    /// A Parakeet model variant with built-in, pinned assets.
    public struct Variant: RawRepresentable, CaseIterable, Sendable, Hashable, Codable {
        /// NVIDIA Parakeet TDT 0.6B v3, which recognizes 25 European languages.
        public static let v3 = Self(rawValue: "v3", manifest: .v3, modelDescription: .v3)
        /// Moondream's Parakeet Ultra, a post-trained Parakeet TDT 0.6B v3.
        public static let ultra = Self(rawValue: "ultra", manifest: .ultra, modelDescription: .v3)
        /// NVIDIA Parakeet Unified EN 0.6B, an RNN-T model that recognizes English only,
        /// licensed under the NVIDIA Open Model License.
        public static let unified = Self(
            rawValue: "unified", manifest: .unified, modelDescription: .unified)

        /// Every variant with built-in assets.
        public static let allCases: [Self] = [.v3, .ultra, .unified]

        /// The variant's name, such as `v3`, which is also its coded form.
        public let rawValue: String
        /// The pinned files of the variant's built-in assets.
        let manifest: ParakeetModelManifest
        /// The tensor contract of the variant's models.
        let modelDescription: ParakeetModelDescription

        /// Returns the variant with a name, or `nil` if no built-in variant has that name.
        public init?(rawValue: String) {
            guard let variant = Self.allCases.first(where: { $0.rawValue == rawValue }) else {
                return nil
            }
            self = variant
        }

        private init(
            rawValue: String, manifest: ParakeetModelManifest,
            modelDescription: ParakeetModelDescription
        ) {
            self.rawValue = rawValue
            self.manifest = manifest
            self.modelDescription = modelDescription
        }

        public static func == (lhs: Self, rhs: Self) -> Bool { lhs.rawValue == rhs.rawValue }

        public func hash(into hasher: inout Hasher) { hasher.combine(rawValue) }
    }

    /// The source of model files.
    public enum Assets: Sendable, Hashable {
        /// The variant's pinned files, installed with ``AssetInventory``
        /// into the Hugging Face cache and loaded from there.
        /// The live encoder installs only for modules that request volatile or fast results.
        case builtIn
        /// A staged directory with compiled encoder, decoder, and joint models and a vocabulary.
        /// The directory is used as is, without download or digest verification.
        /// The runtime loads `Encoder_v2.mlmodelc` when present, and `Encoder.mlmodelc`
        /// otherwise. For volatile or fast results, it loads the optional shorter-window
        /// `Encoder8s.mlmodelc` instead when present.
        case directory(URL)
    }

    /// The Core ML compute units for the encoder.
    public enum ComputeUnits: String, Sendable, Codable {
        case cpuOnly, cpuAndNeuralEngine, cpuAndGPU, all

        var coreML: MLComputeUnits {
            switch self {
            case .cpuOnly: .cpuOnly
            case .cpuAndNeuralEngine: .cpuAndNeuralEngine
            case .cpuAndGPU: .cpuAndGPU
            case .all: .all
            }
        }
    }

    /// The encoders that one configuration loads.
    struct EncoderSelection: Sendable, Hashable {
        /// The encoder for every window.
        let encoder: String
        /// The 15-second encoder for windows that commit without right context,
        /// when ``encoder`` is the live encoder; otherwise `nil`.
        let forcedCommitEncoder: String?

        /// Whether the selection uses the shorter-window live encoder.
        var usesLiveEncoder: Bool { forcedCommitEncoder != nil }
    }

    typealias Part = ParakeetModelManifest.Part

    /// The selected variant.
    public let variant: Variant
    /// The source of model files.
    public let assets: Assets
    /// The compute units for the encoder.
    public let computeUnits: ComputeUnits
    /// The pinned files of built-in assets, and the component names of a staged directory.
    let manifest: ParakeetModelManifest
    /// The store that installs and checks built-in assets.
    let store: ModelAssetStore

    public var modelID: String { manifest.identifier }
    public let capabilities = SpeechRecognitionCapabilities(
        transcription: .init(
            reportingOptions: [.volatileResults, .fastResults],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]),
        liveInput: true)

    /// Creates a configuration without loading, inference, or downloads.
    /// - Parameters:
    ///   - variant: The model variant.
    ///   - assets: The source of model files.
    ///   - computeUnits: The compute units for the encoder.
    public init(
        variant: Variant = .v3, assets: Assets = .builtIn,
        computeUnits: ComputeUnits = .cpuAndNeuralEngine
    ) {
        self.init(
            variant: variant, assets: assets, computeUnits: computeUnits,
            manifest: variant.manifest, store: .shared)
    }

    /// Creates a configuration with a specific manifest and store, for tests.
    init(
        variant: Variant = .v3, assets: Assets = .builtIn,
        computeUnits: ComputeUnits = .cpuAndNeuralEngine, manifest: ParakeetModelManifest,
        store: ModelAssetStore
    ) {
        self.variant = variant
        if case .directory(let url) = assets {
            self.assets = .directory(url.standardizedFileURL)
        } else {
            self.assets = assets
        }
        self.computeUnits = computeUnits
        self.manifest = manifest
        self.store = store
    }

    /// The license and attribution text that applications show for the selected variant.
    public var attribution: String { manifest.attribution }

    var directory: URL {
        switch assets {
        case .builtIn: store.directory(for: manifest)
        case .directory(let url): url
        }
    }

    public func makeExecutor() -> any SpeechRecognitionExecutor { ParakeetExecutor(model: self) }

    /// Whether a request uses the live encoder: requests for volatile or fast results.
    static func usesLiveEncoder(_ request: SpeechRecognitionRequest) -> Bool {
        guard case .transcription(let configuration) = request else { return false }
        return !configuration.reportingOptions.isDisjoint(with: [.volatileResults, .fastResults])
    }

    /// The manifest parts that a set of requests use, in manifest order.
    /// Every request uses the base; volatile and fast results also use the live encoder.
    func parts(for requests: [SpeechRecognitionRequest]) -> [Part] {
        let live = requests.contains(where: Self.usesLiveEncoder)
        return manifest.parts.filter { $0 == .base || live }
    }

    /// Reports the base assets, which every request needs.
    public func availability() async -> SpeechRecognitionAvailability {
        availability(of: [.base])
    }

    /// Reports the base assets, and for volatile or fast results also the live encoder.
    /// A staged directory reports only the base, because its live encoder is optional.
    public func availability(for request: SpeechRecognitionRequest) async
        -> SpeechRecognitionAvailability
    {
        availability(of: parts(for: [request]))
    }

    private func availability(of parts: [Part]) -> SpeechRecognitionAvailability {
        #if !arch(arm64)
            return .unsupportedHardware
        #else
            switch assets {
            case .builtIn:
                if parts.contains(where: { store.isInstalling(manifest, part: $0) }) {
                    return .preparing
                }
                return parts.allSatisfy { store.status(of: manifest, part: $0) == .installed }
                    ? .ready : .missingAssets
            case .directory(let url):
                let components = manifest.components
                let required = [components.decoder, components.joint, components.vocabulary]
                let encoders = ["Encoder_v2.mlmodelc", "Encoder.mlmodelc"]
                func exists(_ name: String) -> Bool {
                    FileManager.default.fileExists(atPath: url.appending(path: name).path)
                }
                return required.allSatisfy(exists) && encoders.contains(where: exists)
                    ? .ready : .missingAssets
            }
        #endif
    }

    /// Creates a request that installs the base assets.
    public func assetInstallationRequest() async throws -> AssetInstallationRequest? {
        try installationRequest(for: [.base])
    }

    /// Creates one request that installs the base assets, and the live encoder when any
    /// request asks for volatile or fast results.
    /// Parts that are already installed are not downloaded or verified again.
    public func assetInstallationRequest(supporting requests: [SpeechRecognitionRequest])
        async throws -> AssetInstallationRequest?
    {
        try installationRequest(for: parts(for: requests))
    }

    private func installationRequest(for parts: [Part]) throws -> AssetInstallationRequest? {
        switch availability(of: parts) {
        case .ready: return nil
        case .unsupportedHardware:
            throw SpeechRecognitionError.unavailableProvider(
                "The Parakeet runtime requires Apple silicon.")
        default: break
        }
        guard case .builtIn = assets else {
            throw SpeechRecognitionError.missingAssets(
                "Stage the Parakeet encoder, decoder, joint, and vocabulary in \(directory.path).")
        }
        let manifest = manifest
        let store = store
        let missing = parts.filter { store.status(of: manifest, part: $0) != .installed }
        // The request owns its progress object, which follows the shared part installs.
        let progress = store.summedProgress(for: manifest, parts: missing)
        return AssetInstallationRequest(progress: progress.progress) {
            try await store.install(manifest, parts: missing)
            withExtendedLifetime(progress) {}
        }
    }

    /// Selects the encoders for file results or for volatile and fast results.
    ///
    /// Live results use the live encoder only when a built-in install has verified it,
    /// or when a staged directory contains it; otherwise they use the 15-second encoder
    /// with the 15-second live settings. The check reads no file contents.
    func encoderSelection(live: Bool) -> EncoderSelection {
        let regular: String
        let liveEncoder: String?
        switch assets {
        case .builtIn:
            regular = manifest.components.encoder
            liveEncoder =
                store.status(of: manifest, part: .liveEncoder) == .installed
                ? manifest.components.liveEncoder : nil
        case .directory(let url):
            func exists(_ name: String) -> Bool {
                FileManager.default.fileExists(atPath: url.appending(path: name).path)
            }
            regular = exists("Encoder_v2.mlmodelc") ? "Encoder_v2.mlmodelc" : "Encoder.mlmodelc"
            let name = manifest.components.liveEncoder ?? CoreMLParakeetEngine.liveEncoderName
            liveEncoder = exists(name) ? name : nil
        }
        if live, let liveEncoder {
            return EncoderSelection(encoder: liveEncoder, forcedCommitEncoder: regular)
        }
        return EncoderSelection(encoder: regular, forcedCommitEncoder: nil)
    }

    /// Reports the repository, revision, compute units, and the encoder that file results
    /// and live results each use, including whether the live encoder is installed.
    public func diagnostics() async -> SpeechRecognitionDiagnostics {
        let builtIn = assets == .builtIn
        let file = encoderSelection(live: false)
        let live = encoderSelection(live: true)
        let liveEncoder = manifest.components.liveEncoder ?? CoreMLParakeetEngine.liveEncoderName
        let liveDescription =
            live.usesLiveEncoder
            ? "volatile and fast results use \(live.encoder), "
                + "with \(live.forcedCommitEncoder ?? file.encoder) for forced commits"
            : "volatile and fast results use \(live.encoder) because the live encoder "
                + "\(liveEncoder) is not installed"
        return .init(
            provider: "AnySpeechRecognizer Parakeet", modelID: modelID,
            runtimeVersion: "Core ML",
            modelRepository: builtIn ? manifest.repository : directory.path,
            modelRevision: builtIn ? manifest.revision : nil,
            computeConfiguration:
                "encoder \(computeUnits.rawValue); decoder and joint cpuAndNeuralEngine; "
                + "no language constraint; file results use \(file.encoder); \(liveDescription)",
            capabilities: capabilities)
    }
}

/// Runs one Parakeet recognizer for one module, with source-time mapping and phrase results.
actor ParakeetExecutor: SpeechRecognitionExecutor {
    let model: ParakeetSpeechModel
    private var engine: (any ParakeetInferenceEngine)?
    private var vocabulary: ParakeetVocabulary?
    private var configuration = ParakeetRecognizer.Configuration.file
    private var recognizer: ParakeetRecognizer?
    private var builder: ParakeetResultBuilder?
    private var request: SpeechRecognitionRequest?
    private var results: SpeechRecognitionResultSink?
    /// The next expected buffer start in the current segment.
    private var expected: CMTime?
    /// The emission of the latest publication's results.
    ///
    /// Emitting a result suspends the executor, so ``finalize(through:)`` can run while
    /// ``append(_:)`` waits to emit, or the reverse. Each publication builds its results
    /// immediately, in the order that the recognizer commits tokens, and emits them after the
    /// previous publication's results, so results arrive in source order. Without this, the
    /// finalization at the end of a file emitted the last phrase between the phrases of the
    /// last buffer, and validation rejected the next of those.
    private var emission: Task<Void, any Error>?
    private var cancelled = false

    init(model: ParakeetSpeechModel) { self.model = model }

    func prepare(
        request: SpeechRecognitionRequest, format: AVAudioFormat, context: AnalysisContext
    ) async throws {
        try request.validate(for: model.capabilities)
        guard format.sampleRate == 16_000, format.channelCount == 1,
            format.commonFormat == .pcmFormatFloat32, !format.isInterleaved
        else {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "Parakeet requires 16 kHz mono Float32 input.")
        }
        guard case .transcription(let transcription) = request else {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "Parakeet does not support DictationTranscriber.")
        }
        let options = transcription.reportingOptions
        let fast = options.contains(.fastResults)
        let live = fast || options.contains(.volatileResults)
        let directory = model.directory
        let manifest = model.manifest
        // Live and fast results use the shorter live encoder when it is installed,
        // and the regular encoder for forced commits, which have no right context.
        // Without it, they use the regular encoder, and the window sets the live settings.
        let selection = model.encoderSelection(live: live)
        let units = model.computeUnits.coreML
        let variant = model.variant
        let description = variant.modelDescription
        let (engine, vocabulary) = try await Task.detached {
            // A staged directory for a different variant fails here, before the encoder loads,
            // instead of decoding with the wrong blank token.
            let vocabulary = try ParakeetVocabulary(
                contentsOf: directory.appending(path: manifest.components.vocabulary))
            guard vocabulary.count == description.vocabularySize else {
                throw SpeechRecognitionError.unsupportedConfiguration(
                    "The Parakeet vocabulary in \(directory.path) has \(vocabulary.count) tokens, "
                        + "but the \(variant.rawValue) variant has "
                        + "\(description.vocabularySize). Select the variant of the model files.")
            }
            var options = CoreMLParakeetEngine.Configuration()
            options.encoderComputeUnits = units
            options.encoderName = selection.encoder
            options.forcedCommitEncoderName = selection.forcedCommitEncoder
            let engine = try CoreMLParakeetEngine(
                directory: directory, configuration: options, model: description)
            return (UncheckedEngine(engine), vocabulary)
        }.value
        try Task.checkCancellation()
        guard !cancelled else { throw CancellationError() }
        let windowFrames = engine.value.model.windowFrames
        use(
            engine.value, vocabulary: vocabulary, request: request,
            configuration: fast
                ? .fast(windowFrames: windowFrames)
                : live ? .live(windowFrames: windowFrames) : .file)
    }

    /// Serves a validated request with a loaded engine and recognizer configuration.
    /// Tests call this method directly with a fake engine.
    func use(
        _ engine: sending any ParakeetInferenceEngine, vocabulary: ParakeetVocabulary,
        request: SpeechRecognitionRequest, configuration: ParakeetRecognizer.Configuration
    ) {
        self.engine = engine
        self.vocabulary = vocabulary
        self.request = request
        self.configuration = configuration
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
            throw SpeechRecognitionError.invalidAudio("Parakeet requires Float32 samples.")
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
        recognizer = try ParakeetRecognizer(
            engine: engine, vocabulary: vocabulary, configuration: configuration)
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
        let finals = builder.commit(update.committed, flush: flush)
        let volatile =
            configuration.reportsVolatileResults && !segmentEnded
            ? builder.volatileResult(update.volatile) : nil
        self.builder = builder
        let reportsVolatileResults = configuration.reportsVolatileResults
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
private struct UncheckedEngine: @unchecked Sendable {
    let value: CoreMLParakeetEngine
    init(_ value: CoreMLParakeetEngine) { self.value = value }
}
