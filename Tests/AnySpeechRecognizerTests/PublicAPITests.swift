import AVFoundation
import CoreMedia
import Foundation
import Synchronization
import Testing

import AnySpeechRecognizer

private struct UpstreamError: LocalizedError, Equatable {
    var errorDescription: String? { "The upstream model stopped." }
}

private let errorCases: [(SpeechRecognitionError, SpeechRecognitionError.Kind, String)] = [
    (.unsupportedConfiguration("No such option."), .unsupportedConfiguration, "No such option."),
    (.unavailableProvider("No Neural Engine."), .unavailableProvider, "No Neural Engine."),
    (.unsupportedLanguage("No Klingon."), .unsupportedLanguage, "No Klingon."),
    (.missingAssets("Install the model."), .missingAssets, "Install the model."),
    (.assetVerificationFailed("model.bin"), .assetVerificationFailed, "model.bin"),
    (.invalidAudio("Empty audio."), .invalidAudio, "Empty audio."),
    (.invalidSessionState("Already finished."), .invalidSessionState, "Already finished."),
    (
        .bufferOverflow, .bufferOverflow,
        "The result buffer is full. Consume results during analysis, or cancel the analyzer."
    ),
    (.invalidResult("Bad timing."), .invalidResult, "Bad timing."),
    (
        .backendFailure(underlying: UpstreamError()), .backendFailure,
        "Speech recognition failed: The upstream model stopped."
    ),
]

private let kindDescriptions: [(SpeechRecognitionError.Kind, String)] = [
    (.unsupportedConfiguration, "Unsupported configuration"),
    (.unavailableProvider, "Unavailable provider"),
    (.unsupportedLanguage, "Unsupported language"),
    (.missingAssets, "Missing assets"),
    (.assetVerificationFailed, "Asset verification failed"),
    (.invalidAudio, "Invalid audio"),
    (.invalidSessionState, "Invalid session state"),
    (.bufferOverflow, "Buffer overflow"),
    (.invalidResult, "Invalid result"),
    (.backendFailure, "Backend failure"),
]

@Test func errorAccessorsDescribeEveryCase() {
    for (error, kind, message) in errorCases {
        #expect(error.kind == kind)
        #expect(error.message == message)
        #expect(error.errorDescription == message)
        #expect(error.errorDescription == error.message)
        #expect(error.localizedDescription == message)
        if error.kind == .backendFailure {
            #expect(error.underlyingError as? UpstreamError == UpstreamError())
        } else {
            #expect(error.underlyingError == nil)
        }
    }
}

@Test func errorKindsAreDistinctAndDescribed() {
    #expect(kindDescriptions.count == errorCases.count)
    for (index, (kind, description)) in kindDescriptions.enumerated() {
        #expect(kind.description == description)
        #expect(kind == kind)
        for (otherIndex, (other, _)) in kindDescriptions.enumerated() where otherIndex != index {
            #expect(kind != other)
        }
    }
    #expect(Set(kindDescriptions.map(\.0)).count == kindDescriptions.count)
}

@Test func errorCaseConstructorsAreFunctionValues() {
    let typed: (String) -> SpeechRecognitionError = SpeechRecognitionError.missingAssets
    let untyped = SpeechRecognitionError.missingAssets
    #expect(typed("a").kind == .missingAssets)
    #expect(untyped("b").message == "b")

    let typedFailure: (any Error) -> SpeechRecognitionError =
        SpeechRecognitionError.backendFailure(underlying:)
    let untypedFailure = SpeechRecognitionError.backendFailure(underlying:)
    #expect(typedFailure(UpstreamError()).underlyingError is UpstreamError)
    #expect(untypedFailure(UpstreamError()).kind == .backendFailure)

    let unsupportedConfiguration = SpeechRecognitionError.unsupportedConfiguration
    let unavailableProvider = SpeechRecognitionError.unavailableProvider
    let unsupportedLanguage = SpeechRecognitionError.unsupportedLanguage
    let assetVerificationFailed = SpeechRecognitionError.assetVerificationFailed
    let invalidAudio = SpeechRecognitionError.invalidAudio
    let invalidSessionState = SpeechRecognitionError.invalidSessionState
    let invalidResult = SpeechRecognitionError.invalidResult
    let untypedResults: [(SpeechRecognitionError, SpeechRecognitionError.Kind)] = [
        (unsupportedConfiguration("c"), .unsupportedConfiguration),
        (unavailableProvider("c"), .unavailableProvider),
        (unsupportedLanguage("c"), .unsupportedLanguage),
        (assetVerificationFailed("c"), .assetVerificationFailed),
        (invalidAudio("c"), .invalidAudio),
        (invalidSessionState("c"), .invalidSessionState),
        (invalidResult("c"), .invalidResult),
    ]
    for (error, kind) in untypedResults {
        #expect(error.kind == kind)
        #expect(error.message == "c")
    }

    let constructors: [(String) -> SpeechRecognitionError] = [
        SpeechRecognitionError.unsupportedConfiguration,
        SpeechRecognitionError.unavailableProvider,
        SpeechRecognitionError.unsupportedLanguage,
        SpeechRecognitionError.missingAssets,
        SpeechRecognitionError.assetVerificationFailed,
        SpeechRecognitionError.invalidAudio,
        SpeechRecognitionError.invalidSessionState,
        SpeechRecognitionError.invalidResult,
    ]
    #expect(constructors.map { $0("reason").message } == Array(repeating: "reason", count: 8))
}

@Test func handlersMatchByKind() {
    var matched = false
    do {
        throw SpeechRecognitionError.missingAssets("Install the model.")
    } catch let error as SpeechRecognitionError where error.kind == .missingAssets {
        matched = error.message == "Install the model."
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    #expect(matched)
}

@Test func handlersReadTheUnderlyingError() {
    var underlying: (any Error)?
    do {
        throw SpeechRecognitionError.backendFailure(underlying: UpstreamError())
    } catch let error as SpeechRecognitionError {
        underlying = error.underlyingError
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    #expect(underlying as? UpstreamError == UpstreamError())
}

@Test func caseHandlersFromVersion01StillMatch() {
    var matchedOverflow = false
    do {
        throw SpeechRecognitionError.bufferOverflow
    } catch SpeechRecognitionError.bufferOverflow {
        matchedOverflow = true
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    #expect(matchedOverflow)

    var underlying: (any Error)?
    do {
        throw SpeechRecognitionError.backendFailure(underlying: UpstreamError())
    } catch SpeechRecognitionError.backendFailure(let error) {
        underlying = error
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    #expect(underlying is UpstreamError)
}

private let availabilityCases:
    [(SpeechRecognitionAvailability, SpeechRecognitionAvailability.UnavailableReason?, AssetInventory.Status)] = [
        (.ready, nil, .installed),
        (.missingAssets, .assetsNotInstalled, .supported),
        (.preparing, .assetsInstalling, .downloading),
        (.unsupportedHardware, .unsupportedHardware, .unsupported),
        (.unsupportedLanguage, .unsupportedLanguage, .unsupported),
    ]

@Test func availabilityAccessorsDescribeEveryCase() {
    for (availability, reason, _) in availabilityCases {
        #expect(availability.isAvailable == (availability == .ready))
        #expect(availability.unavailableReason == reason)
    }
}

@Test func availabilityValuesAreHashable() {
    let values = availabilityCases.map(\.0)
    #expect(Set(values).count == values.count)
    for (index, value) in values.enumerated() {
        #expect(value == value)
        #expect(Set([value, value]).count == 1)
        for (otherIndex, other) in values.enumerated() where otherIndex != index {
            #expect(value != other)
        }
    }
    let reasons = availabilityCases.compactMap(\.1)
    #expect(Set(reasons).count == reasons.count)
}

/// Records the asset methods that the package calls on a custom provider.
private final class ProviderCalls: Sendable {
    private let names = Mutex<[String]>([])
    func record(_ name: String) { names.withLock { $0.append(name) } }
    var recorded: [String] { names.withLock { $0 } }
}

private struct IdleExecutor: SpeechRecognitionExecutor {
    func prepare(
        request: SpeechRecognitionRequest,
        format: AVAudioFormat,
        context: AnalysisContext
    ) async throws {}
    func analyze(_ input: SpeechAnalysisInput, results: SpeechRecognitionResultSink) async throws {}
    func finalize(through time: CMTime) async throws {}
    func cancel() async {}
}

private struct IdleDiarizationExecutor: SpeakerDiarizationExecutor {
    func prepare(request: SpeakerDiarizationRequest, format: AVAudioFormat) async throws {}
    func analyze(_ input: SpeechAnalysisInput, results: SpeakerDiarizationResultSink) async throws {}
    func finalize(through time: CMTime) async throws {}
    func cancel() async {}
}

/// A provider written for version 0.1, with only the asset methods without arguments.
private struct Version01Model: SpeechRecognitionModel {
    let modelID = "test/version-0.1"
    let capabilities = SpeechRecognitionCapabilities()
    let calls: ProviderCalls

    func makeExecutor() -> any SpeechRecognitionExecutor { IdleExecutor() }

    func availability() async -> SpeechRecognitionAvailability {
        calls.record("availability()")
        return .missingAssets
    }

    func assetInstallationRequest() async throws -> AssetInstallationRequest? {
        calls.record("assetInstallationRequest()")
        let calls = calls
        return AssetInstallationRequest(progress: Progress(totalUnitCount: 1)) {
            calls.record("install")
        }
    }
}

/// A provider that implements the asset methods that version 0.3.0 keeps.
/// Its live requests use assets that only improve them.
private struct MigratedModel: SpeechRecognitionModel {
    let modelID = "test/migrated"
    let capabilities = SpeechRecognitionCapabilities(
        transcription: .init(reportingOptions: [.volatileResults]),
        liveInput: true
    )
    let calls: ProviderCalls

    func makeExecutor() -> any SpeechRecognitionExecutor { IdleExecutor() }

    func availability(for request: SpeechRecognitionRequest) async
        -> SpeechRecognitionAvailability
    {
        calls.record("availability(for:)")
        return .ready
    }

    func assetStatus(for request: SpeechRecognitionRequest) async -> AssetInventory.Status {
        calls.record("assetStatus(for:)")
        return request.includesVolatileResults ? .supported : .installed
    }

    func assetInstallationRequest(supporting requests: [SpeechRecognitionRequest]) async throws
        -> AssetInstallationRequest?
    {
        calls.record("assetInstallationRequest(supporting:) \(requests.count)")
        let calls = calls
        return AssetInstallationRequest(progress: Progress(totalUnitCount: 1)) {
            calls.record("install")
        }
    }
}

@Test func version01ProvidersStillReportAndInstallAssets() async throws {
    let calls = ProviderCalls()
    let model: any SpeechRecognitionModel = Version01Model(calls: calls)
    let request = SpeechRecognitionRequest.transcription(.init())

    #expect(await model.availability(for: request) == .missingAssets)
    #expect(await model.assetStatus(for: request) == .supported)
    #expect(try await model.assetInstallationRequest(supporting: [request]) != nil)
    #expect(
        calls.recorded == ["availability()", "availability()", "assetInstallationRequest()"]
    )

    let transcriber = SpeechTranscriber(model: Version01Model(calls: calls))
    #expect(await AssetInventory.status(forModules: [transcriber]) == .supported)
    let installation = try #require(
        try await AssetInventory.assetInstallationRequest(supporting: [transcriber])
    )
    try await installation.downloadAndInstall()
    #expect(
        calls.recorded.suffix(3) == ["availability()", "assetInstallationRequest()", "install"]
    )
}

@Test func migratedProvidersReceiveTheNewAssetMethods() async throws {
    let calls = ProviderCalls()
    let model: any SpeechRecognitionModel = MigratedModel(calls: calls)
    let file = SpeechRecognitionRequest.transcription(.init())
    let live = SpeechRecognitionRequest.transcription(.init(reportingOptions: [.volatileResults]))

    #expect(await model.availability(for: live) == .ready)
    #expect(await model.assetStatus(for: live) == .supported)
    #expect(await model.assetStatus(for: file) == .installed)
    #expect(try await model.assetInstallationRequest(supporting: [file, live]) != nil)
    #expect(
        calls.recorded == [
            "availability(for:)", "assetStatus(for:)", "assetStatus(for:)",
            "assetInstallationRequest(supporting:) 2",
        ]
    )

    let modules = [
        SpeechTranscriber(model: MigratedModel(calls: calls)),
        SpeechTranscriber(model: MigratedModel(calls: calls), reportingOptions: [.volatileResults]),
    ]
    let before = calls.recorded.count
    #expect(await AssetInventory.status(forModules: [modules[0]]) == .installed)
    #expect(await AssetInventory.status(forModules: modules) == .supported)
    let installation = try #require(
        try await AssetInventory.assetInstallationRequest(supporting: modules)
    )
    try await installation.downloadAndInstall()
    #expect(
        Array(calls.recorded.dropFirst(before)) == [
            "assetStatus(for:)", "assetStatus(for:)", "assetStatus(for:)",
            "assetInstallationRequest(supporting:) 2", "install",
        ]
    )
}

/// A provider that implements only `availability(for:)`.
private struct RequestAvailabilityModel: SpeechRecognitionModel {
    let modelID = "test/request-availability"
    let capabilities = SpeechRecognitionCapabilities()
    let state: SpeechRecognitionAvailability

    func makeExecutor() -> any SpeechRecognitionExecutor { IdleExecutor() }
    func availability(for request: SpeechRecognitionRequest) async
        -> SpeechRecognitionAvailability
    {
        state
    }
}

/// A diarization provider that implements only `availability(for:)`.
private struct RequestAvailabilityDiarizationModel: SpeakerDiarizationModel {
    let modelID = "test/request-availability-diarization"
    let capabilities = SpeakerDiarizationCapabilities(maximumSpeakerCount: 2)
    let state: SpeechRecognitionAvailability

    func makeExecutor() -> any SpeakerDiarizationExecutor { IdleDiarizationExecutor() }
    func availability(for request: SpeakerDiarizationRequest) async
        -> SpeechRecognitionAvailability
    {
        state
    }
}

@Test func defaultAssetStatusMapsAvailability() async {
    for (availability, _, status) in availabilityCases {
        let model: any SpeechRecognitionModel = RequestAvailabilityModel(state: availability)
        #expect(await model.assetStatus(for: .transcription(.init())) == status)
        let transcriber = SpeechTranscriber(model: RequestAvailabilityModel(state: availability))
        #expect(await AssetInventory.status(forModules: [transcriber]) == status)

        let diarizationModel: any SpeakerDiarizationModel =
            RequestAvailabilityDiarizationModel(state: availability)
        #expect(await diarizationModel.assetStatus(for: .init()) == status)
        let diarizer = SpeakerDiarizer(
            model: RequestAvailabilityDiarizationModel(state: availability)
        )
        #expect(await AssetInventory.status(forModules: [diarizer]) == status)
    }
}
