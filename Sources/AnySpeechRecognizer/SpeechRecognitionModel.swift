import AVFoundation
import CoreMedia
import Foundation
import Speech

/// Immutable recognition configuration that creates an executor for each module in a session.
/// Construction and status queries must not load models or start downloads.
public protocol SpeechRecognitionModel: Sendable {
    /// The identifier of the selected model.
    var modelID: String { get }
    /// The modules, options, and input modes the model supports.
    var capabilities: SpeechRecognitionCapabilities { get }
    /// Creates an unloaded executor for one module in one analyzer.
    func makeExecutor() -> any SpeechRecognitionExecutor
    /// The audio formats the executor accepts for a request, in order of preference.
    /// The analyzer converts input to the first format unless the source already matches one.
    func compatibleAudioFormats(for request: SpeechRecognitionRequest) async -> [AVAudioFormat]
    /// Checks hardware, language, and the local assets that every request needs,
    /// without downloading.
    func availability() async -> SpeechRecognitionAvailability
    /// Checks hardware, language, and the local assets that one request uses,
    /// without downloading.
    ///
    /// ``AssetInventory/status(forModules:)`` and analyzer preparation call this method.
    /// A model can report ``SpeechRecognitionAvailability/missingAssets`` for assets that
    /// only improve a request; analysis still proceeds when ``availability()``
    /// reports ``SpeechRecognitionAvailability/ready``, and the executor runs without them.
    /// The default implementation returns ``availability()``.
    /// - Parameter request: A module's request.
    func availability(for request: SpeechRecognitionRequest) async -> SpeechRecognitionAvailability
    /// Creates an explicit installation request for the assets that every request needs,
    /// or returns `nil` when they exist.
    /// The returned request performs network work only when its download method is called.
    func assetInstallationRequest() async throws -> AssetInstallationRequest?
    /// Creates one explicit installation request for the assets that a set of requests use,
    /// or returns `nil` when they exist.
    ///
    /// ``AssetInventory/assetInstallationRequest(supporting:)`` calls this method once
    /// for each model, with the requests of every module that uses the model,
    /// so that assets the requests share install once.
    /// The returned request performs network work only when its download method is called.
    /// The default implementation returns ``assetInstallationRequest()``.
    /// - Parameter requests: The requests of the modules that use this model.
    func assetInstallationRequest(supporting requests: [SpeechRecognitionRequest]) async throws
        -> AssetInstallationRequest?
    /// Describes the runtime and model without including recognized text or audio.
    func diagnostics() async -> SpeechRecognitionDiagnostics
}

/// Performs recognition for one module in one analyzer, with private decoder state.
///
/// The analyzer calls ``prepare(request:format:context:)`` once,
/// then ``analyze(_:results:)`` once for the life of the session.
/// Control methods run concurrently with `analyze`.
/// Each control refers to a time in the source timeline, so its effect does not depend on
/// when it arrives relative to input.
public protocol SpeechRecognitionExecutor: Sendable {
    /// Validates the request and loads local resources without downloading.
    /// - Parameters:
    ///   - request: The module and options to serve.
    ///   - format: The format of every buffer that ``analyze(_:results:)`` receives.
    ///   - context: The initial analysis context.
    /// - Throws: A configuration, availability, or preparation error.
    func prepare(
        request: SpeechRecognitionRequest, format: AVAudioFormat, context: AnalysisContext
    ) async throws

    /// Pulls input until the sequence ends, publishing results as they become available.
    ///
    /// Input ends when the analyzer finishes or the module is removed.
    /// Do not finalize remaining audio when input ends; the analyzer calls
    /// ``finalize(through:)`` first when results must be final.
    /// - Parameters:
    ///   - input: Buffers in the prepared format, with source start times.
    ///   - results: A bounded sink that validates results and can throw on cancellation or overflow.
    /// - Throws: An audio or backend error, or `CancellationError`.
    func analyze(_ input: SpeechAnalysisInput, results: SpeechRecognitionResultSink) async throws

    /// Publishes final results through a time and returns when they are published.
    ///
    /// The analyzer calls this method only after the executor has pulled input through `time`.
    /// Later results must not start before `time`.
    /// - Parameter time: The inclusive source time to finalize through.
    /// - Throws: A backend error or `CancellationError`.
    func finalize(through time: CMTime) async throws

    /// Requests that analysis skip audio before a time. This request is best effort.
    /// - Parameter time: The source time before which results are no longer wanted.
    func cancelAnalysis(before time: CMTime) async

    /// Applies new context as soon as possible.
    /// - Parameter context: The replacement context.
    /// - Throws: An error if the executor cannot apply the context.
    func setContext(_ context: AnalysisContext) async throws

    /// Stops input and inference, discards late output, and releases session resources.
    /// Repeated calls must be safe, including during preparation.
    func cancel() async
}

extension SpeechRecognitionExecutor {
    public func cancelAnalysis(before time: CMTime) async {}
    public func setContext(_ context: AnalysisContext) async throws {}
}

/// The module and options that an executor serves.
public enum SpeechRecognitionRequest: Sendable, Hashable {
    /// A ``SpeechTranscriber`` request.
    case transcription(SpeechRecognitionConfiguration)
    /// A ``DictationTranscriber`` request.
    case dictation(DictationRecognitionConfiguration)

    /// Whether word timing attributes were requested.
    public var includesAudioTimeRange: Bool {
        switch self {
        case .transcription(let configuration):
            configuration.attributeOptions.contains(.audioTimeRange)
        case .dictation(let configuration):
            configuration.attributeOptions.contains(.audioTimeRange)
        }
    }

    /// Whether confidence attributes were requested.
    public var includesConfidence: Bool {
        switch self {
        case .transcription(let configuration):
            configuration.attributeOptions.contains(.transcriptionConfidence)
        case .dictation(let configuration):
            configuration.attributeOptions.contains(.transcriptionConfidence)
        }
    }

    /// Whether volatile results were requested.
    public var includesVolatileResults: Bool {
        switch self {
        case .transcription(let configuration):
            configuration.reportingOptions.contains(.volatileResults)
        case .dictation(let configuration):
            configuration.reportingOptions.contains(.volatileResults)
        }
    }

    /// Checks the request against a model's capabilities.
    /// - Parameter capabilities: The selected model's capabilities.
    /// - Throws: ``SpeechRecognitionError/unsupportedConfiguration(_:)``.
    public func validate(for capabilities: SpeechRecognitionCapabilities) throws {
        switch self {
        case .transcription(let configuration):
            guard let support = capabilities.transcription else {
                throw SpeechRecognitionError.unsupportedConfiguration(
                    "The model does not support SpeechTranscriber.")
            }
            try Self.check(
                configuration.transcriptionOptions, supported: support.transcriptionOptions,
                kind: "transcription option")
            try Self.check(
                configuration.reportingOptions, supported: support.reportingOptions,
                kind: "reporting option")
            try Self.check(
                configuration.attributeOptions, supported: support.attributeOptions,
                kind: "attribute option")
        case .dictation(let configuration):
            guard let support = capabilities.dictation else {
                throw SpeechRecognitionError.unsupportedConfiguration(
                    "The model does not support DictationTranscriber.")
            }
            guard configuration.contentHints.isEmpty || support.contentHints else {
                throw SpeechRecognitionError.unsupportedConfiguration(
                    "The model does not support dictation content hints.")
            }
            try Self.check(
                configuration.transcriptionOptions, supported: support.transcriptionOptions,
                kind: "dictation transcription option")
            try Self.check(
                configuration.reportingOptions, supported: support.reportingOptions,
                kind: "dictation reporting option")
            try Self.check(
                configuration.attributeOptions, supported: support.attributeOptions,
                kind: "dictation attribute option")
        }
    }

    private static func check<Option: Hashable>(
        _ requested: Set<Option>, supported: Set<Option>, kind: String
    ) throws {
        let unsupported = requested.subtracting(supported)
        guard unsupported.isEmpty else {
            let names = unsupported.map { String(describing: $0) }.sorted().joined(separator: ", ")
            throw SpeechRecognitionError.unsupportedConfiguration(
                "The model does not support the \(kind) \(names).")
        }
    }
}

/// The hardware, language, and local asset state of a model.
public enum SpeechRecognitionAvailability: Sendable, Equatable {
    /// The model can prepare local resources.
    case ready
    /// Assets must be installed explicitly.
    case missingAssets
    /// An installation or shared model load is in progress.
    case preparing
    /// The device cannot run this provider.
    case unsupportedHardware
    /// The selected locale is unsupported.
    case unsupportedLanguage
}

/// The modules, options, and input modes a provider supports.
/// Requests for anything else fail before audio is consumed.
public struct SpeechRecognitionCapabilities: Sendable, Hashable {
    /// Options a provider supports for ``SpeechTranscriber``.
    public struct Transcription: Sendable, Hashable {
        /// Supported text options.
        public var transcriptionOptions: Set<SpeechTranscriber.TranscriptionOption>
        /// Supported result delivery options.
        public var reportingOptions: Set<SpeechTranscriber.ReportingOption>
        /// Supported result attributes.
        public var attributeOptions: Set<SpeechTranscriber.ResultAttributeOption>

        /// Creates a transcription capability description.
        /// - Parameters:
        ///   - transcriptionOptions: Supported text options.
        ///   - reportingOptions: Supported result delivery options.
        ///   - attributeOptions: Supported result attributes.
        public init(
            transcriptionOptions: Set<SpeechTranscriber.TranscriptionOption> = [],
            reportingOptions: Set<SpeechTranscriber.ReportingOption> = [],
            attributeOptions: Set<SpeechTranscriber.ResultAttributeOption> = [.audioTimeRange]
        ) {
            self.transcriptionOptions = transcriptionOptions
            self.reportingOptions = reportingOptions
            self.attributeOptions = attributeOptions
        }

        /// Every option Apple's transcriber defines.
        public static var all: Self {
            .init(
                transcriptionOptions: Set(SpeechTranscriber.TranscriptionOption.allCases),
                reportingOptions: Set(SpeechTranscriber.ReportingOption.allCases),
                attributeOptions: Set(SpeechTranscriber.ResultAttributeOption.allCases))
        }
    }

    /// Options a provider supports for ``DictationTranscriber``.
    public struct Dictation: Sendable, Hashable {
        /// Whether content hints are supported.
        public var contentHints: Bool
        /// Supported text options.
        public var transcriptionOptions: Set<DictationTranscriber.TranscriptionOption>
        /// Supported result delivery options.
        public var reportingOptions: Set<DictationTranscriber.ReportingOption>
        /// Supported result attributes.
        public var attributeOptions: Set<DictationTranscriber.ResultAttributeOption>

        /// Creates a dictation capability description.
        /// - Parameters:
        ///   - contentHints: Whether content hints are supported.
        ///   - transcriptionOptions: Supported text options.
        ///   - reportingOptions: Supported result delivery options.
        ///   - attributeOptions: Supported result attributes.
        public init(
            contentHints: Bool = false,
            transcriptionOptions: Set<DictationTranscriber.TranscriptionOption> = [],
            reportingOptions: Set<DictationTranscriber.ReportingOption> = [],
            attributeOptions: Set<DictationTranscriber.ResultAttributeOption> = [.audioTimeRange]
        ) {
            self.contentHints = contentHints
            self.transcriptionOptions = transcriptionOptions
            self.reportingOptions = reportingOptions
            self.attributeOptions = attributeOptions
        }

        /// Every option Apple's dictation transcriber defines.
        public static var all: Self {
            .init(
                contentHints: true,
                transcriptionOptions: Set(DictationTranscriber.TranscriptionOption.allCases),
                reportingOptions: Set(DictationTranscriber.ReportingOption.allCases),
                attributeOptions: Set(DictationTranscriber.ResultAttributeOption.allCases))
        }
    }

    /// ``SpeechTranscriber`` support, or `nil` if the provider cannot serve it.
    public var transcription: Transcription?
    /// ``DictationTranscriber`` support, or `nil` if the provider cannot serve it.
    public var dictation: Dictation?
    /// Whether the provider produces results while input continues,
    /// with latency suitable for live audio.
    public var liveInput: Bool
    /// Whether the caller can constrain the recognition language.
    public var languageControl: Bool

    /// Creates a provider capability description.
    /// - Parameters:
    ///   - transcription: ``SpeechTranscriber`` support, or `nil`.
    ///   - dictation: ``DictationTranscriber`` support, or `nil`.
    ///   - liveInput: Whether the provider supports live input.
    ///   - languageControl: Whether the provider can constrain the language.
    public init(
        transcription: Transcription? = Transcription(), dictation: Dictation? = nil,
        liveInput: Bool = false, languageControl: Bool = false
    ) {
        self.transcription = transcription
        self.dictation = dictation
        self.liveInput = liveInput
        self.languageControl = languageControl
    }
}

/// The requested ``SpeechTranscriber`` options.
public struct SpeechRecognitionConfiguration: Sendable, Hashable {
    /// Options that affect text.
    public let transcriptionOptions: Set<SpeechTranscriber.TranscriptionOption>
    /// Options that affect result delivery.
    public let reportingOptions: Set<SpeechTranscriber.ReportingOption>
    /// Attributes requested in each result's text.
    public let attributeOptions: Set<SpeechTranscriber.ResultAttributeOption>

    /// Creates a request without performing validation or inference.
    /// - Parameters:
    ///   - transcriptionOptions: The requested text behavior.
    ///   - reportingOptions: The requested result delivery behavior.
    ///   - attributeOptions: The requested result attributes.
    public init(
        transcriptionOptions: Set<SpeechTranscriber.TranscriptionOption> = [],
        reportingOptions: Set<SpeechTranscriber.ReportingOption> = [],
        attributeOptions: Set<SpeechTranscriber.ResultAttributeOption> = []
    ) {
        self.transcriptionOptions = transcriptionOptions
        self.reportingOptions = reportingOptions
        self.attributeOptions = attributeOptions
    }
}

/// The requested ``DictationTranscriber`` options.
public struct DictationRecognitionConfiguration: Sendable, Hashable {
    /// Hints about the expected speech.
    public let contentHints: Set<DictationTranscriber.ContentHint>
    /// Options that affect text.
    public let transcriptionOptions: Set<DictationTranscriber.TranscriptionOption>
    /// Options that affect result delivery.
    public let reportingOptions: Set<DictationTranscriber.ReportingOption>
    /// Attributes requested in each result's text.
    public let attributeOptions: Set<DictationTranscriber.ResultAttributeOption>

    /// Creates a request without performing validation or inference.
    /// - Parameters:
    ///   - contentHints: Hints about the expected speech.
    ///   - transcriptionOptions: The requested text behavior.
    ///   - reportingOptions: The requested result delivery behavior.
    ///   - attributeOptions: The requested result attributes.
    public init(
        contentHints: Set<DictationTranscriber.ContentHint> = [],
        transcriptionOptions: Set<DictationTranscriber.TranscriptionOption> = [],
        reportingOptions: Set<DictationTranscriber.ReportingOption> = [],
        attributeOptions: Set<DictationTranscriber.ResultAttributeOption> = []
    ) {
        self.contentHints = contentHints
        self.transcriptionOptions = transcriptionOptions
        self.reportingOptions = reportingOptions
        self.attributeOptions = attributeOptions
    }
}

/// A recognition result that an executor publishes for a transcription or dictation module.
public struct SpeechRecognitionResult: SpeechModuleResult, Sendable, Hashable {
    /// The most likely text, with the backend's spacing and attributes.
    public let text: AttributedString
    /// Alternative interpretations, in backend order.
    public let alternatives: [AttributedString]
    /// The range in the source recording.
    public let range: CMTimeRange
    /// The exclusive boundary before which no result can change.
    /// A result is volatile when this time is before the end of its range.
    public let resultsFinalizationTime: CMTime

    /// Creates a result without rewriting text or manufacturing timing.
    /// - Parameters:
    ///   - text: The attributed phrase snapshot.
    ///   - alternatives: The backend's alternative interpretations.
    ///   - range: The source audio range.
    ///   - resultsFinalizationTime: The finalized audio boundary after this result.
    public init(
        text: AttributedString, alternatives: [AttributedString] = [], range: CMTimeRange,
        resultsFinalizationTime: CMTime
    ) {
        self.text = text
        self.alternatives = alternatives
        self.range = range
        self.resultsFinalizationTime = resultsFinalizationTime
    }
}

/// An audio buffer with its start time in the source recording.
/// The buffer must not be mutated after it is wrapped.
public struct TimedAudioBuffer: @unchecked Sendable {
    /// The audio samples.
    public let buffer: AVAudioPCMBuffer
    /// The source time of the first frame.
    public let startTime: CMTime

    /// Wraps a buffer and its source start time.
    /// - Parameters:
    ///   - buffer: The audio samples. Do not mutate the buffer afterward.
    ///   - startTime: The source time of the first frame.
    public init(buffer: AVAudioPCMBuffer, startTime: CMTime) {
        self.buffer = buffer
        self.startTime = startTime
    }

    /// The source time after the last frame.
    public var endTime: CMTime {
        startTime
            + CMTime(
                value: CMTimeValue(buffer.frameLength),
                timescale: CMTimeScale(buffer.format.sampleRate))
    }
}

/// The input sequence for one executor, in the executor's prepared format.
public struct SpeechAnalysisInput: AsyncSequence, Sendable {
    public typealias Element = TimedAudioBuffer
    public typealias Failure = any Error
    let branch: AudioBranch

    public func makeAsyncIterator() -> AsyncIterator { AsyncIterator(branch: branch) }

    /// Pulls buffers; pulling marks them consumed.
    public struct AsyncIterator: AsyncIteratorProtocol {
        let branch: AudioBranch
        public mutating func next() async throws -> TimedAudioBuffer? { try await branch.next() }
    }
}

/// A bounded sink that validates results before publishing them to one module.
public struct SpeechRecognitionResultSink: Sendable {
    let publish: @Sendable (SpeechRecognitionResult) async throws -> Void
    let volatile: @Sendable (CMTimeRange?) -> Void

    /// Publishes a result. Awaiting the call applies backpressure.
    /// - Parameter result: A final or volatile result in source time.
    /// - Throws: A validation error, ``SpeechRecognitionError/bufferOverflow``,
    ///   or `CancellationError`.
    public func emit(_ result: SpeechRecognitionResult) async throws {
        try await publish(result)
    }

    /// Reports the audio range whose results can still change, or `nil` when none can.
    /// - Parameter range: The volatile source range.
    public func setVolatileRange(_ range: CMTimeRange?) { volatile(range) }
}

/// Runtime metadata for diagnosis and evaluations.
public struct SpeechRecognitionDiagnostics: Codable, Sendable, Equatable {
    /// The provider name.
    public let provider: String
    /// The model identifier.
    public let modelID: String
    /// The upstream runtime version, if known.
    public let runtimeVersion: String?
    /// The model repository, if applicable.
    public let modelRepository: String?
    /// An immutable revision or local content digest, if available.
    public let modelRevision: String?
    /// The compute configuration.
    public let computeConfiguration: String
    /// The requested locale; automatic language selection has none.
    public let requestedLocale: String?
    /// The effective locale; automatic selection leaves this unset.
    public let effectiveLocale: String?
    /// The OS version and build string.
    public let operatingSystem: String
    /// Options and input modes the provider supports.
    public let supportedOptions: [String]

    /// Creates diagnostic metadata without loading resources.
    /// - Parameters:
    ///   - provider: The runtime name.
    ///   - modelID: The selected model identifier.
    ///   - runtimeVersion: The inference library version, if known.
    ///   - modelRepository: The upstream model repository, if applicable.
    ///   - modelRevision: An immutable revision or local content digest.
    ///   - computeConfiguration: The configured compute units and relevant options.
    ///   - requestedLocale: The caller's requested recognition locale.
    ///   - effectiveLocale: The resolved recognition locale, if the backend reports one.
    ///   - capabilities: The capabilities to list as supported options.
    public init(
        provider: String, modelID: String, runtimeVersion: String? = nil,
        modelRepository: String? = nil, modelRevision: String? = nil,
        computeConfiguration: String, requestedLocale: String? = nil,
        effectiveLocale: String? = nil, capabilities: SpeechRecognitionCapabilities? = nil
    ) {
        self.provider = provider
        self.modelID = modelID
        self.runtimeVersion = runtimeVersion
        self.modelRepository = modelRepository
        self.modelRevision = modelRevision
        self.computeConfiguration = computeConfiguration
        self.requestedLocale = requestedLocale
        self.effectiveLocale = effectiveLocale
        self.operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
        self.supportedOptions = capabilities.map(Self.describe) ?? []
    }

    private static func describe(_ capabilities: SpeechRecognitionCapabilities) -> [String] {
        var options: [String] = []
        if capabilities.liveInput { options.append("liveInput") }
        if capabilities.languageControl { options.append("languageControl") }
        if let transcription = capabilities.transcription {
            options.append("SpeechTranscriber")
            options += transcription.transcriptionOptions.map { "\($0)" }
            options += transcription.reportingOptions.map { "\($0)" }
            options += transcription.attributeOptions.map { "\($0)" }
        }
        if let dictation = capabilities.dictation {
            options.append("DictationTranscriber")
            if dictation.contentHints { options.append("contentHints") }
            options += dictation.transcriptionOptions.map { "dictation.\($0)" }
            options += dictation.reportingOptions.map { "dictation.\($0)" }
            options += dictation.attributeOptions.map { "dictation.\($0)" }
        }
        return Array(Set(options)).sorted()
    }
}

extension SpeechRecognitionModel {
    public func compatibleAudioFormats(for request: SpeechRecognitionRequest) async
        -> [AVAudioFormat]
    {
        [AudioFormats.monoFloat16k]
    }
    public func availability() async -> SpeechRecognitionAvailability { .ready }
    public func availability(for request: SpeechRecognitionRequest) async
        -> SpeechRecognitionAvailability
    {
        await availability()
    }
    public func assetInstallationRequest() async throws -> AssetInstallationRequest? { nil }
    public func assetInstallationRequest(supporting requests: [SpeechRecognitionRequest])
        async throws -> AssetInstallationRequest?
    {
        try await assetInstallationRequest()
    }
    public func diagnostics() async -> SpeechRecognitionDiagnostics {
        .init(
            provider: String(describing: Self.self), modelID: modelID,
            computeConfiguration: "provider-defined", capabilities: capabilities)
    }
}

enum AudioFormats {
    /// 16 kHz mono Float32, the default executor format.
    static let monoFloat16k = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)!
}
