import AVFoundation
import CoreML
import CoreMedia
import Foundation

/// NVIDIA's Streaming Sortformer 4spk speaker diarization model, version v2 or v2.1,
/// run by the package on Core ML.
///
/// Sortformer predicts, for each 80 ms frame, the activity of up to four speakers,
/// so it reports overlapping speech.
/// An arrival-order speaker cache keeps each speaker's identifier for the whole session,
/// and speakers are numbered from 0 in order of arrival.
/// Accuracy degrades with five or more speakers.
/// The model streams with NVIDIA's low-latency setting:
/// a chunk is processed 1.04 s after its audio arrives,
/// and post-processing decides speaker changes about 1 s later.
/// File and live input produce the same results, except after a finalization or a gap,
/// which ends the feature stream early.
///
/// This is a package extension; Apple's Speech framework has no diarization module.
public struct SortformerDiarizationModel: SpeakerDiarizationModel {
    /// A Streaming Sortformer version with built-in, pinned assets.
    ///
    /// Both versions have the same architecture, streaming configuration, and Core ML
    /// interface; they differ in their trained weights and licenses.
    /// Each version has its own license; see ``attribution``.
    ///
    /// `Version` is a structure, not an enumeration,
    /// so a new version does not break source compatibility.
    /// A `switch` over a version needs a `default` case.
    public struct Version: RawRepresentable, CaseIterable, Sendable, Hashable, Codable {
        /// NVIDIA's diar_streaming_sortformer_4spk-v2, licensed under CC BY 4.0.
        public static let v2 = Self(rawValue: "v2", manifest: .builtIn)
        /// NVIDIA's diar_streaming_sortformer_4spk-v2.1,
        /// licensed under the NVIDIA Open Model License Agreement.
        ///
        /// NVIDIA trained v2.1 with additional meeting data and reports lower error than v2
        /// on most of its meeting and telephone benchmarks.
        /// The license requires that copies of the model include the agreement and the notice
        /// "Licensed by NVIDIA Corporation under the NVIDIA Open Model License",
        /// and that use is consistent with NVIDIA's Trustworthy AI terms.
        // The underscore stands for the version's period; `v21` would read as version 21.
        // swift-format-ignore: AlwaysUseLowerCamelCase
        public static let v2_1 = Self(rawValue: "v2.1", manifest: .v2_1)

        /// Every version with built-in assets.
        public static let allCases: [Self] = [.v2, .v2_1]

        /// The version's name, such as `v2.1`, which is also its coded form.
        public let rawValue: String
        /// The pinned files of the version's built-in assets.
        let manifest: SortformerModelManifest

        /// Returns the version with a name, or `nil` if no built-in version has that name.
        public init?(rawValue: String) {
            guard let version = Self.allCases.first(where: { $0.rawValue == rawValue }) else {
                return nil
            }
            self = version
        }

        private init(rawValue: String, manifest: SortformerModelManifest) {
            self.rawValue = rawValue
            self.manifest = manifest
        }

        public static func == (lhs: Self, rhs: Self) -> Bool { lhs.rawValue == rhs.rawValue }

        public func hash(into hasher: inout Hasher) { hasher.combine(rawValue) }
    }

    /// The source of the compiled model.
    public enum Assets: Sendable, Hashable {
        /// The pinned files, installed with ``AssetInventory`` into the Hugging Face cache
        /// and loaded from there.
        case builtIn
        /// A compiled model directory (`.mlmodelc`) used as is,
        /// without download or digest verification.
        case compiledModel(URL)
    }

    /// The Core ML compute units for the model.
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

    /// Thresholds and durations that turn frame probabilities into speaker segments,
    /// as in NeMo's diarization post-processing.
    public struct PostProcessing: Sendable, Hashable {
        /// The probability above which a speaker starts speaking.
        public var onset: Float
        /// The probability below which a speaker stops speaking.
        public var offset: Float
        /// Seconds added before each speech run.
        public var padOnset: Double
        /// Seconds added after each speech run.
        public var padOffset: Double
        /// Speech runs shorter than this many seconds are removed.
        public var minimumSpeechDuration: Double
        /// Gaps shorter than this many seconds within one speaker's speech are filled.
        public var minimumGapDuration: Double

        /// Creates post-processing parameters.
        /// - Parameters:
        ///   - onset: The probability above which a speaker starts speaking.
        ///   - offset: The probability below which a speaker stops speaking.
        ///   - padOnset: Seconds added before each speech run.
        ///   - padOffset: Seconds added after each speech run.
        ///   - minimumSpeechDuration: Seconds below which speech runs are removed.
        ///   - minimumGapDuration: Seconds below which gaps are filled.
        public init(
            onset: Float, offset: Float, padOnset: Double = 0, padOffset: Double = 0,
            minimumSpeechDuration: Double = 0, minimumGapDuration: Double = 0
        ) {
            self.onset = onset
            self.offset = offset
            self.padOnset = padOnset
            self.padOffset = padOffset
            self.minimumSpeechDuration = minimumSpeechDuration
            self.minimumGapDuration = minimumGapDuration
        }

        /// NeMo's parameters for Streaming Sortformer v2, tuned on CALLHOME part 1
        /// telephone conversations.
        ///
        /// NVIDIA publishes no parameters for v2.1, so both versions use these by default.
        public static let callHome = PostProcessing(
            onset: 0.641, offset: 0.561, padOnset: 0.229, padOffset: 0.079,
            minimumSpeechDuration: 0.511, minimumGapDuration: 0.296)
        /// NeMo's parameters for Streaming Sortformer v2, tuned on the DIHARD III
        /// development set.
        public static let dihard3 = PostProcessing(
            onset: 0.56, offset: 1.0, padOnset: 0.063, padOffset: 0.002,
            minimumSpeechDuration: 0.007, minimumGapDuration: 0.151)
        /// Thresholds of 0.5 without padding or filtering.
        public static let none = PostProcessing(onset: 0.5, offset: 0.5)

        var segmenterParameters: SpeakerSegmenter.Parameters {
            func time(_ seconds: Double) -> CMTime {
                CMTime(value: CMTimeValue((max(0, seconds) * 16_000).rounded()), timescale: 16_000)
            }
            return .init(
                onset: onset, offset: offset, padOnset: time(padOnset), padOffset: time(padOffset),
                minimumDurationOn: time(minimumSpeechDuration),
                minimumDurationOff: time(minimumGapDuration))
        }
    }

    /// The model version.
    public let version: Version
    /// The source of the compiled model.
    public let assets: Assets
    /// The compute units for the model.
    public let computeUnits: ComputeUnits
    /// The post-processing parameters.
    public let postProcessing: PostProcessing
    let manifest: SortformerModelManifest
    let store: ModelAssetStore
    /// Creates the inference engine; tests replace the Core ML engine.
    let makeInference: @Sendable (URL, MLComputeUnits) throws -> any SortformerInference

    public var modelID: String { manifest.identifier }
    public let capabilities = SpeakerDiarizationCapabilities(
        maximumSpeakerCount: 4, overlappingSpeech: true, liveInput: true,
        reportingOptions: [.volatileResults])

    /// Creates a configuration without loading, inference, or downloads.
    /// - Parameters:
    ///   - version: The model version. With `.compiledModel`, it selects the identifier
    ///     and attribution, which must match the staged model.
    ///   - assets: The source of the compiled model.
    ///   - computeUnits: The compute units. On an M4 Max, a chunk took 56 ms with
    ///     `cpuAndNeuralEngine` and 19 ms with `cpuAndGPU`, but `cpuAndGPU` raised peak
    ///     resident memory from about 70 MB to about 500 MB.
    ///   - postProcessing: The thresholds and durations for speaker segments.
    public init(
        version: Version = .v2, assets: Assets = .builtIn,
        computeUnits: ComputeUnits = .cpuAndNeuralEngine, postProcessing: PostProcessing = .callHome
    ) {
        self.init(
            version: version, assets: assets, computeUnits: computeUnits,
            postProcessing: postProcessing, manifest: version.manifest, store: .shared)
    }

    /// Creates a configuration with a specific manifest, store, and engine, for tests.
    init(
        version: Version = .v2, assets: Assets = .builtIn,
        computeUnits: ComputeUnits = .cpuAndNeuralEngine,
        postProcessing: PostProcessing = .callHome, manifest: SortformerModelManifest,
        store: ModelAssetStore,
        makeInference: @escaping @Sendable (URL, MLComputeUnits) throws -> any SortformerInference =
            { try CoreMLSortformerEngine(contentsOf: $0, computeUnits: $1) }
    ) {
        if case .compiledModel(let url) = assets {
            self.assets = .compiledModel(url.standardizedFileURL)
        } else {
            self.assets = assets
        }
        self.version = version
        self.computeUnits = computeUnits
        self.postProcessing = postProcessing
        self.manifest = manifest
        self.store = store
        self.makeInference = makeInference
    }

    /// The license and attribution text that applications show for the model.
    public var attribution: String { manifest.attribution }

    /// The compiled model directory.
    var modelURL: URL {
        switch assets {
        case .builtIn: store.directory(for: manifest).appending(path: manifest.model)
        case .compiledModel(let url): url
        }
    }

    public func makeExecutor() -> any SpeakerDiarizationExecutor {
        SortformerExecutor(model: self)
    }

    /// Reports whether the compiled model is installed and verified, or staged.
    public func availability(for request: SpeakerDiarizationRequest) async
        -> SpeechRecognitionAvailability
    {
        availability
    }

    private var availability: SpeechRecognitionAvailability {
        #if !arch(arm64)
            return .unsupportedHardware
        #else
            switch assets {
            case .builtIn:
                if store.isInstalling(manifest) { return .preparing }
                return store.status(of: manifest) == .installed ? .ready : .missingAssets
            case .compiledModel(let url):
                return FileManager.default.fileExists(
                    atPath: url.appending(path: "coremldata.bin").path)
                    ? .ready : .missingAssets
            }
        #endif
    }

    /// Creates a request that installs the pinned files, or returns `nil` when they are installed.
    public func assetInstallationRequest(supporting requests: [SpeakerDiarizationRequest])
        async throws -> AssetInstallationRequest?
    {
        switch availability {
        case .ready: return nil
        case .unsupportedHardware:
            throw SpeechRecognitionError.unavailableProvider(
                "The Sortformer runtime requires Apple silicon.")
        default: break
        }
        guard case .builtIn = assets else {
            throw SpeechRecognitionError.missingAssets(
                "Stage the compiled Sortformer model at \(modelURL.path).")
        }
        let manifest = manifest
        let store = store
        // The request owns its progress object, which follows the shared install.
        let progress = store.summedProgress(for: manifest, parts: [.base])
        return AssetInstallationRequest(progress: progress.progress) {
            try await store.install(manifest, parts: [.base])
            withExtendedLifetime(progress) {}
        }
    }

    /// Reports the repository, revision, compute units, and post-processing.
    public func diagnostics() async -> SpeechRecognitionDiagnostics {
        let builtIn = assets == .builtIn
        let processing = postProcessing
        return .init(
            provider: "AnySpeechRecognizer Sortformer", modelID: modelID,
            runtimeVersion: "Core ML",
            modelRepository: builtIn ? "\(manifest.repository)/\(manifest.model)" : modelURL.path,
            modelRevision: builtIn ? manifest.revision : nil,
            computeConfiguration:
                "\(computeUnits.rawValue); up to 4 speakers; onset \(processing.onset), "
                + "offset \(processing.offset), pad \(processing.padOnset)/\(processing.padOffset) s, "
                + "minimum speech \(processing.minimumSpeechDuration) s, "
                + "minimum gap \(processing.minimumGapDuration) s")
    }
}

/// Runs one Sortformer pipeline for one diarizer, with source-time results.
actor SortformerExecutor: SpeakerDiarizationExecutor {
    let model: SortformerDiarizationModel
    private var pipeline: SortformerPipeline?
    private var request: SpeakerDiarizationRequest?
    private var results: SpeakerDiarizationResultSink?
    /// The emission of the latest publication's results.
    ///
    /// Emitting a result suspends the executor, so ``finalize(through:)`` can run while
    /// ``analyze(_:results:)`` waits to emit, or the reverse. Each publication takes its
    /// results from the pipeline immediately, in pipeline order, and emits them after the
    /// previous publication's results, so results arrive in source order.
    private var emission: Task<Void, any Error>?
    private var cancelled = false

    init(model: SortformerDiarizationModel) { self.model = model }

    func prepare(request: SpeakerDiarizationRequest, format: AVAudioFormat) async throws {
        try request.validate(for: model.capabilities)
        guard format.sampleRate == 16_000, format.channelCount == 1,
            format.commonFormat == .pcmFormatFloat32, !format.isInterleaved
        else {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "Sortformer requires 16 kHz mono Float32 input.")
        }
        let url = model.modelURL
        let units = model.computeUnits.coreML
        let make = model.makeInference
        let inference = try await Task.detached {
            UncheckedInference(try make(url, units))
        }.value
        try Task.checkCancellation()
        guard !cancelled else { throw CancellationError() }
        self.request = request
        pipeline = SortformerPipeline(
            inference: inference.value, parameters: model.postProcessing.segmenterParameters,
            reportsVolatileResults: request.includesVolatileResults)
    }

    func analyze(_ input: SpeechAnalysisInput, results: SpeakerDiarizationResultSink) async throws {
        self.results = results
        for try await buffer in input {
            guard !cancelled, let pipeline else { throw CancellationError() }
            guard let samples = buffer.buffer.floatChannelData?[0] else {
                throw SpeechRecognitionError.invalidAudio("Sortformer requires Float32 samples.")
            }
            let update = try pipeline.append(
                UnsafeBufferPointer(start: samples, count: Int(buffer.buffer.frameLength)),
                at: buffer.startTime)
            try await publish(update)
        }
    }

    func finalize(through time: CMTime) async throws {
        guard let pipeline, !cancelled else { return }
        try await publish(pipeline.finalize(through: time))
    }

    private func publish(_ update: SortformerPipeline.Update) async throws {
        guard let results else { return }
        let reportsVolatileResults = request?.includesVolatileResults == true
        let previous = emission
        let task = Task {
            _ = await previous?.result
            for result in update.results { try await results.emit(result) }
            if reportsVolatileResults { results.setVolatileRange(update.volatileRange) }
        }
        emission = task
        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    func cancel() async {
        cancelled = true
        pipeline = nil
        results = nil
    }
}

/// Moves a newly loaded engine to its owning executor.
/// The engine is created on a background task and used only by that executor afterward.
private struct UncheckedInference: @unchecked Sendable {
    let value: any SortformerInference
    init(_ value: any SortformerInference) { self.value = value }
}
