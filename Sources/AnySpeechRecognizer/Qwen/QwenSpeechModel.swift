import AVFoundation
import CoreML
import CoreMedia
import Foundation
import Speech

/// A Qwen3-ASR model run by the package's own runtime on Core ML, for file transcription.
///
/// Qwen3-ASR recognizes 30 languages and 22 Chinese dialects, including Mandarin, Cantonese,
/// Japanese, and Korean, which Parakeet does not cover. It identifies the language
/// automatically unless ``language`` forces one.
///
/// The model transcribes audio in segments of up to 40 seconds, cut at quiet points,
/// and publishes one final result for each segment, with the segment's source range.
/// Results arrive when a segment is complete: at a cut, at a gap in the input, or at
/// finalization. The model does not support volatile or fast results, word timing,
/// confidence, alternatives, or dictation; requests for them fail before the analyzer reads
/// audio. The separate Qwen3-ForcedAligner model, which predicts word timing, is not integrated.
///
/// Built-in assets install on demand for a size whose published conversion the package pins.
/// For another size, convert the model with `Tools/QwenConversion` and pass the output directory
/// as ``Assets/directory(_:)``.
public struct QwenSpeechModel: SpeechRecognitionModel {
    /// Qwen3-ASR model sizes.
    public enum Variant: String, Sendable, Codable, CaseIterable {
        /// Qwen3-ASR-0.6B: an 18-layer audio encoder and a 28-layer Qwen3 decoder.
        case small = "0.6b"
        /// Qwen3-ASR-1.7B: a 24-layer audio encoder and a 28-layer Qwen3 decoder twice as wide.
        case large = "1.7b"

        var identifier: String { "qwen3-asr-\(rawValue)" }

        var title: String {
            switch self {
            case .small: "Qwen3-ASR 0.6B"
            case .large: "Qwen3-ASR 1.7B"
            }
        }

        var repository: String {
            switch self {
            case .small: "Qwen/Qwen3-ASR-0.6B-hf"
            case .large: "Qwen/Qwen3-ASR-1.7B-hf"
            }
        }
    }

    /// The source of model files.
    public enum Assets: Sendable, Hashable {
        /// The size's pinned files from its published conversion, installed with
        /// ``AssetInventory`` into the Hugging Face cache and loaded from there.
        /// A size that the package does not pin reports missing assets, and its installation fails.
        case builtIn
        /// A directory with `AudioEncoder.mlmodelc`, `TextDecoder.mlmodelc`, `vocab.json`, and
        /// `qwen3_asr_config.json`, used as is, without download or digest verification.
        case directory(URL)
    }

    /// Core ML compute units.
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

    /// The language codes that the model can be forced to, from the model card.
    public static let supportedLanguageCodes: Set<String> = [
        "ar", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fil", "fr", "hi", "hu", "id", "it",
        "ja", "ko", "mk", "ms", "nl", "pl", "pt", "ro", "ru", "sv", "th", "tr", "vi", "yue", "zh",
    ]

    /// The selected size.
    public let variant: Variant
    /// The source of model files.
    public let assets: Assets
    /// The forced language, or `nil` to identify the language of each segment automatically.
    public let language: Locale.Language?
    /// The compute units for the audio encoder.
    public let encoderComputeUnits: ComputeUnits
    /// The compute units for the text decoder.
    /// Core ML cannot build the 28-layer stateful decoder for the Neural Engine (error -14),
    /// so use the GPU or the CPU; `Tools/QwenConversion/README.md` describes the tests.
    public let decoderComputeUnits: ComputeUnits
    /// The pinned files of the size's published conversion, or `nil` when none is pinned.
    let manifest: QwenModelManifest?
    /// The store that installs and checks built-in assets.
    let store: ModelAssetStore

    public var modelID: String { manifest?.identifier ?? variant.identifier }

    /// The license and attribution text that applications show for the selected size:
    /// the pinned manifest's text for built-in assets, and otherwise the text that
    /// `Tools/QwenConversion` writes into each converted directory.
    public var attribution: String {
        if case .builtIn = assets, let manifest { return manifest.attribution }
        return "\(variant.title) by the Qwen team, Alibaba Cloud "
            + "(https://huggingface.co/\(variant.repository)), licensed under the Apache License, "
            + "Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Converted to Core ML for "
            + "AnySpeechRecognizer: the model is split into an audio encoder and a text decoder "
            + "with a KV cache, weights are stored in float16, and weights of large layers are "
            + "quantized to int8."
    }

    public let capabilities = SpeechRecognitionCapabilities(
        transcription: .init(reportingOptions: [], attributeOptions: []), liveInput: false,
        languageControl: true)

    /// Creates a configuration without loading, inference, or downloads.
    /// - Parameters:
    ///   - variant: The model size.
    ///   - assets: The source of model files.
    ///   - language: A language to force, such as `Locale.Language(identifier: "yue")` for
    ///     Cantonese, or `nil` to identify it automatically. Only the language code is used.
    ///   - encoderComputeUnits: The compute units for the audio encoder.
    ///   - decoderComputeUnits: The compute units for the text decoder.
    public init(
        variant: Variant = .large, assets: Assets = .builtIn, language: Locale.Language? = nil,
        encoderComputeUnits: ComputeUnits = .cpuAndGPU,
        decoderComputeUnits: ComputeUnits = .cpuAndGPU
    ) {
        self.init(
            variant: variant, assets: assets, language: language,
            encoderComputeUnits: encoderComputeUnits, decoderComputeUnits: decoderComputeUnits,
            manifest: QwenModelManifest.builtIn[variant], store: .shared)
    }

    /// Creates a configuration with a specific manifest and store, for tests.
    init(
        variant: Variant, assets: Assets, language: Locale.Language? = nil,
        encoderComputeUnits: ComputeUnits = .cpuAndGPU,
        decoderComputeUnits: ComputeUnits = .cpuAndGPU, manifest: QwenModelManifest?,
        store: ModelAssetStore
    ) {
        self.variant = variant
        if case .directory(let url) = assets {
            self.assets = .directory(url.standardizedFileURL)
        } else {
            self.assets = assets
        }
        self.language = language
        self.encoderComputeUnits = encoderComputeUnits
        self.decoderComputeUnits = decoderComputeUnits
        self.manifest = manifest
        self.store = store
    }

    /// The directory of model files: the verified snapshot of built-in assets,
    /// or `nil` for built-in assets of a size that the package does not pin.
    var directory: URL? {
        switch assets {
        case .builtIn: manifest.map { store.directory(for: $0) }
        case .directory(let url): url
        }
    }

    /// The forced language's code, such as `zh`, or `nil` for automatic identification.
    var languageCode: String? { language?.languageCode?.identifier }

    public func makeExecutor() -> any SpeechRecognitionExecutor { QwenExecutor(model: self) }

    /// Reports the forced language, and the installed or staged files without reading
    /// their contents.
    public func availability() async -> SpeechRecognitionAvailability {
        #if !arch(arm64)
            return .unsupportedHardware
        #else
            if let languageCode, !Self.supportedLanguageCodes.contains(languageCode) {
                return .unsupportedLanguage
            }
            switch assets {
            case .builtIn:
                guard let manifest else { return .missingAssets }
                if store.isInstalling(manifest) { return .preparing }
                return store.status(of: manifest) == .installed ? .ready : .missingAssets
            case .directory(let url):
                return QwenModelManifest.components.allSatisfy {
                    FileManager.default.fileExists(atPath: url.appending(path: $0).path)
                } ? .ready : .missingAssets
            }
        #endif
    }

    /// Creates a request that installs the pinned files, or returns `nil` when they are
    /// installed or staged. Throws for a staged directory without the files, and for built-in
    /// assets of a size that the package does not pin.
    public func assetInstallationRequest() async throws -> AssetInstallationRequest? {
        switch await availability() {
        case .ready: return nil
        case .unsupportedHardware:
            throw SpeechRecognitionError.unavailableProvider(
                "The Qwen3-ASR runtime requires Apple silicon.")
        case .unsupportedLanguage:
            throw SpeechRecognitionError.unsupportedLanguage(
                "Qwen3-ASR does not support the language \(languageCode ?? "").")
        default: break
        }
        guard case .builtIn = assets, let manifest else {
            let location = directory.map { " in \($0.path)" } ?? ""
            throw SpeechRecognitionError.missingAssets(
                "No published conversion of \(variant.title) is pinned. Stage the Qwen3-ASR "
                    + "encoder, decoder, vocabulary, and configuration\(location); convert them "
                    + "with Tools/QwenConversion.")
        }
        let store = store
        // The request owns its progress object, which follows the shared install.
        let progress = store.summedProgress(for: manifest, parts: [.base])
        return AssetInstallationRequest(progress: progress.progress) {
            try await store.install(manifest, parts: [.base])
            withExtendedLifetime(progress) {}
        }
    }

    /// Reports the repository and revision of built-in assets or the staged directory,
    /// the compute units, and the language setting.
    public func diagnostics() async -> SpeechRecognitionDiagnostics {
        let builtIn = assets == .builtIn
        return .init(
            provider: "AnySpeechRecognizer Qwen3-ASR", modelID: modelID, runtimeVersion: "Core ML",
            modelRepository: builtIn ? manifest?.repository : directory?.path,
            modelRevision: builtIn ? manifest?.revision : nil,
            computeConfiguration:
                "encoder \(encoderComputeUnits.rawValue); decoder \(decoderComputeUnits.rawValue); "
                + "greedy decoding; segments of up to "
                + "\(QwenSegmenter.maximumSamples / QwenSegmenter.sampleRate) s; "
                + (languageCode.map { "language forced to \($0)" } ?? "automatic language"),
            requestedLocale: language?.minimalIdentifier, effectiveLocale: languageCode,
            capabilities: capabilities)
    }
}

/// Runs one Qwen3-ASR recognizer for one module: buffers input, transcribes segments,
/// and publishes one final result per segment.
actor QwenExecutor: SpeechRecognitionExecutor {
    let model: QwenSpeechModel
    private var recognizer: QwenRecognizer?
    private var results: SpeechRecognitionResultSink?
    /// Samples of the current segment that no result covers yet.
    private var buffered: [Float] = []
    /// The source time of `buffered[0]`.
    private var origin = CMTime.zero
    /// The next expected buffer start.
    private var expected: CMTime?
    /// The last character of the latest published text, for spacing between results.
    private var lastCharacter: Character?
    /// The emission of the latest result; see `ParakeetExecutor.emission`.
    private var emission: Task<Void, any Error>?
    private var cancelled = false

    init(model: QwenSpeechModel) { self.model = model }

    /// The end of the latest appended buffer, for tests.
    var inputEnd: CMTime? { expected }

    func prepare(
        request: SpeechRecognitionRequest, format: AVAudioFormat, context: AnalysisContext
    ) async throws {
        guard case .transcription = request else {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "Qwen3-ASR does not support DictationTranscriber.")
        }
        try request.validate(for: model.capabilities)
        guard format.sampleRate == 16_000, format.channelCount == 1,
            format.commonFormat == .pcmFormatFloat32, !format.isInterleaved
        else {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "Qwen3-ASR requires 16 kHz mono Float32 input.")
        }
        guard let directory = model.directory else {
            throw SpeechRecognitionError.missingAssets(
                "No published conversion of \(model.variant.title) is pinned; stage a directory.")
        }
        let languageCode = model.languageCode
        let encoderUnits = model.encoderComputeUnits.coreML
        let decoderUnits = model.decoderComputeUnits.coreML
        let loaded = try await Task.detached {
            let configuration = try QwenModelConfiguration(
                contentsOf: directory.appending(path: CoreMLQwenEngine.configurationName))
            var prefill: [Int]?
            if let languageCode {
                guard let language = configuration.prompt.languages[languageCode] else {
                    throw SpeechRecognitionError.unsupportedLanguage(
                        "Qwen3-ASR does not support the language \(languageCode).")
                }
                prefill = language.prefill
            }
            let vocabulary = try QwenVocabulary(
                contentsOf: directory.appending(path: CoreMLQwenEngine.vocabularyName),
                skipping: Set(configuration.skippedTokenIds))
            let engine = try CoreMLQwenEngine(
                directory: directory, configuration: configuration,
                encoderComputeUnits: encoderUnits, decoderComputeUnits: decoderUnits)
            return UncheckedRecognizer(
                QwenRecognizer(engine: engine, vocabulary: vocabulary, prefill: prefill))
        }.value
        try Task.checkCancellation()
        guard !cancelled else { throw CancellationError() }
        use(loaded.value)
    }

    /// Serves a loaded recognizer. Tests call this method directly with a fake engine.
    func use(_ recognizer: sending QwenRecognizer) {
        self.recognizer = recognizer
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
            throw SpeechRecognitionError.invalidAudio("Qwen3-ASR requires Float32 samples.")
        }
        // A gap in the input ends the current segment.
        if let expected, buffer.startTime > expected + CMTime(value: 1, timescale: 16_000) {
            try await transcribe(through: buffered.count)
        }
        if buffered.isEmpty { origin = buffer.startTime }
        buffered += UnsafeBufferPointer(start: samples, count: Int(buffer.buffer.frameLength))
        expected = buffer.endTime
        try await transcribe(through: 0)
    }

    func finalize(through time: CMTime) async throws {
        guard !cancelled, !buffered.isEmpty else { return }
        let offset = time - origin
        guard offset >= .zero else { return }
        // Finalization is inclusive, so finalize through the sample after `time`.
        let sample =
            Int(
                CMTimeConvertScale(offset, timescale: 16_000, method: .roundTowardPositiveInfinity)
                    .value) + 1
        try await transcribe(through: min(sample, buffered.count))
    }

    /// Transcribes segments while the buffer is longer than a segment can be,
    /// then the first `count` remaining samples as one segment.
    private func transcribe(through count: Int) async throws {
        var remaining = count
        while buffered.count > QwenSegmenter.maximumSamples {
            let length = buffered.withUnsafeBufferPointer { QwenSegmenter.boundary(in: $0) }
            try await transcribeSegment(length: length)
            remaining = max(0, remaining - length)
        }
        if remaining > 0 { try await transcribeSegment(length: remaining) }
    }

    /// Transcribes and removes the first `length` buffered samples, and publishes their text.
    private func transcribeSegment(length: Int) async throws {
        guard !cancelled else { throw CancellationError() }
        let text = try transcribe(Array(buffered[..<length]))
        let range = CMTimeRange(
            start: origin, duration: CMTime(value: CMTimeValue(length), timescale: 16_000))
        // Update the buffer before publishing suspends the executor, so that a finalization
        // that runs meanwhile does not transcribe the same samples again.
        buffered.removeFirst(length)
        origin = range.end
        try await publish(text, range: range)
    }

    /// Transcribes one segment, and splits it in two near the middle when decoding reaches the
    /// end of the decoder's context, which only unusually dense speech does.
    private func transcribe(_ samples: [Float]) throws -> String {
        guard let recognizer else { throw CancellationError() }
        let transcript = try recognizer.transcribe(samples)
        let minimum = 2 * QwenSegmenter.sampleRate
        guard transcript.reachedContextLimit, samples.count >= 2 * minimum else {
            return transcript.text
        }
        let middle = samples.withUnsafeBufferPointer {
            QwenSegmenter.boundary(in: $0, target: $0.count / 2, search: $0.count / 4)
        }
        let first = try transcribe(Array(samples[..<middle]))
        let second = try transcribe(Array(samples[middle...]))
        return Self.join(first, second)
    }

    private func publish(_ text: String, range: CMTimeRange) async throws {
        guard let results, !text.isEmpty else { return }
        let separated =
            lastCharacter.map { Self.needsSpace(between: $0, and: text.first!) } ?? false
        lastCharacter = text.last
        let result = SpeechRecognitionResult(
            text: AttributedString(separated ? " " + text : text), range: range,
            resultsFinalizationTime: range.end)
        let previous = emission
        let task = Task {
            _ = await previous?.result
            try await results.emit(result)
        }
        emission = task
        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    /// Joins the transcripts of adjacent segments.
    static func join(_ first: String, _ second: String) -> String {
        guard let last = first.last, let next = second.first else { return first + second }
        return needsSpace(between: last, and: next) ? first + " " + second : first + second
    }

    /// Whether text that ends with one character and continues with another needs a space:
    /// not around Chinese characters, kana, or full-width punctuation, which are not spaced.
    static func needsSpace(between last: Character, and next: Character) -> Bool {
        func isUnspaced(_ character: Character) -> Bool {
            character.unicodeScalars.contains { scalar in
                switch scalar.value {
                case 0x3000...0x303F,  // CJK symbols and punctuation
                    0x3040...0x30FF,  // Hiragana and katakana
                    0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF,  // CJK ideographs
                    0x20000...0x2FA1F,
                    0xFF00...0xFFEF:  // Full-width forms
                    true
                default: false
                }
            }
        }
        return !last.isWhitespace && !next.isWhitespace && !isUnspaced(last) && !isUnspaced(next)
    }

    func cancel() async {
        cancelled = true
        recognizer = nil
        results = nil
        buffered = []
    }
}

/// Moves a newly loaded recognizer to its owning executor.
/// The recognizer is created on a background task and used only by that executor afterward.
private struct UncheckedRecognizer: @unchecked Sendable {
    let value: QwenRecognizer
    init(_ value: QwenRecognizer) { self.value = value }
}
