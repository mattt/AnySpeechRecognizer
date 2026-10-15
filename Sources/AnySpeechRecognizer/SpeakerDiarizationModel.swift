import AVFoundation
import CoreMedia
import Foundation

/// Immutable diarization configuration that creates an executor for each diarizer in a session.
/// Construction and status queries must not load models or start downloads.
///
/// This is a package extension; Apple's Speech framework has no diarization module.
public protocol SpeakerDiarizationModel: Sendable {
    /// The identifier of the selected model.
    var modelID: String { get }
    /// The speakers, options, and input modes the model supports.
    var capabilities: SpeakerDiarizationCapabilities { get }
    /// Creates an unloaded executor for one diarizer in one analyzer.
    func makeExecutor() -> any SpeakerDiarizationExecutor
    /// The audio formats the executor accepts for a request, in order of preference.
    /// The analyzer converts input to the first format unless the source already matches one.
    func compatibleAudioFormats(for request: SpeakerDiarizationRequest) async -> [AVAudioFormat]
    /// Checks hardware and the local assets that a request uses, without downloading.
    /// The default implementation returns ``SpeechRecognitionAvailability/ready``.
    /// - Parameter request: A diarizer's request.
    func availability(for request: SpeakerDiarizationRequest) async -> SpeechRecognitionAvailability
    /// Creates one explicit installation request for the assets that a set of requests use,
    /// or returns `nil` when they exist.
    /// The returned request performs network work only when its download method is called.
    /// The default implementation returns `nil`.
    /// - Parameter requests: The requests of the diarizers that use this model.
    func assetInstallationRequest(supporting requests: [SpeakerDiarizationRequest]) async throws
        -> AssetInstallationRequest?
    /// Describes the runtime and model without including audio.
    func diagnostics() async -> SpeechRecognitionDiagnostics
}

/// Performs diarization for one diarizer in one analyzer, with private model state.
///
/// The analyzer calls ``prepare(request:format:)`` once,
/// then ``analyze(_:results:)`` once for the life of the session.
/// Control methods run concurrently with `analyze`.
/// Each control refers to a time in the source timeline.
public protocol SpeakerDiarizationExecutor: Sendable {
    /// Validates the request and loads local resources without downloading.
    /// - Parameters:
    ///   - request: The options to serve.
    ///   - format: The format of every buffer that ``analyze(_:results:)`` receives.
    /// - Throws: A configuration, availability, or preparation error.
    func prepare(request: SpeakerDiarizationRequest, format: AVAudioFormat) async throws

    /// Pulls input until the sequence ends, publishing results as they become available.
    ///
    /// Do not finalize remaining audio when input ends; the analyzer calls
    /// ``finalize(through:)`` first when results must be final.
    /// - Parameters:
    ///   - input: Buffers in the prepared format, with source start times.
    ///   - results: A bounded sink that validates results and can throw on cancellation or overflow.
    /// - Throws: An audio or backend error, or `CancellationError`.
    func analyze(_ input: SpeechAnalysisInput, results: SpeakerDiarizationResultSink) async throws

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

    /// Stops input and inference, discards late output, and releases session resources.
    /// Repeated calls must be safe, including during preparation.
    func cancel() async
}

extension SpeakerDiarizationExecutor {
    public func cancelAnalysis(before time: CMTime) async {}
}

extension SpeakerDiarizationModel {
    public func compatibleAudioFormats(for request: SpeakerDiarizationRequest) async
        -> [AVAudioFormat]
    {
        [AudioFormats.monoFloat16k]
    }
    public func availability(for request: SpeakerDiarizationRequest) async
        -> SpeechRecognitionAvailability
    {
        .ready
    }
    public func assetInstallationRequest(supporting requests: [SpeakerDiarizationRequest])
        async throws -> AssetInstallationRequest?
    {
        nil
    }
    public func diagnostics() async -> SpeechRecognitionDiagnostics {
        .init(
            provider: String(describing: Self.self), modelID: modelID,
            computeConfiguration: "provider-defined")
    }
}

/// The options that a diarization executor serves.
public struct SpeakerDiarizationRequest: Sendable, Hashable {
    /// Options that affect result delivery.
    public let reportingOptions: Set<SpeakerDiarizer.ReportingOption>

    /// Creates a request without performing validation or inference.
    /// - Parameter reportingOptions: Options that affect result delivery.
    public init(reportingOptions: Set<SpeakerDiarizer.ReportingOption> = []) {
        self.reportingOptions = reportingOptions
    }

    /// Whether volatile results were requested.
    public var includesVolatileResults: Bool { reportingOptions.contains(.volatileResults) }

    /// Checks the request against a model's capabilities.
    /// - Parameter capabilities: The selected model's capabilities.
    /// - Throws: ``SpeechRecognitionError/unsupportedConfiguration(_:)``.
    public func validate(for capabilities: SpeakerDiarizationCapabilities) throws {
        let unsupported = reportingOptions.subtracting(capabilities.reportingOptions)
        guard unsupported.isEmpty else {
            let names = unsupported.map(\.rawValue).sorted().joined(separator: ", ")
            throw SpeechRecognitionError.unsupportedConfiguration(
                "The diarization model does not support the reporting option \(names).")
        }
    }
}

/// The speakers, options, and input modes a diarization model supports.
public struct SpeakerDiarizationCapabilities: Sendable, Hashable {
    /// The largest number of distinct speakers the model can report in one session.
    public var maximumSpeakerCount: Int
    /// Whether the model reports overlapping speech of several speakers.
    public var overlappingSpeech: Bool
    /// Whether the model produces results while input continues,
    /// with latency suitable for live audio.
    public var liveInput: Bool
    /// Supported result delivery options.
    public var reportingOptions: Set<SpeakerDiarizer.ReportingOption>

    /// Creates a capability description.
    /// - Parameters:
    ///   - maximumSpeakerCount: The largest number of distinct speakers in one session.
    ///   - overlappingSpeech: Whether the model reports overlapping speech.
    ///   - liveInput: Whether the model supports live input.
    ///   - reportingOptions: Supported result delivery options.
    public init(
        maximumSpeakerCount: Int, overlappingSpeech: Bool = false, liveInput: Bool = false,
        reportingOptions: Set<SpeakerDiarizer.ReportingOption> = []
    ) {
        self.maximumSpeakerCount = maximumSpeakerCount
        self.overlappingSpeech = overlappingSpeech
        self.liveInput = liveInput
        self.reportingOptions = reportingOptions
    }
}

/// A bounded sink that validates diarization results before publishing them to one diarizer.
public struct SpeakerDiarizationResultSink: Sendable {
    let publish: @Sendable (SpeakerDiarizer.Result) async throws -> Void
    let volatile: @Sendable (CMTimeRange?) -> Void

    /// Publishes a result. Awaiting the call applies backpressure.
    /// - Parameter result: A final or volatile result in source time.
    /// - Throws: A validation error, ``SpeechRecognitionError/bufferOverflow``,
    ///   or `CancellationError`.
    public func emit(_ result: SpeakerDiarizer.Result) async throws {
        try await publish(result)
    }

    /// Reports the audio range whose results can still change, or `nil` when none can.
    /// - Parameter range: The volatile source range.
    public func setVolatileRange(_ range: CMTimeRange?) { volatile(range) }
}
