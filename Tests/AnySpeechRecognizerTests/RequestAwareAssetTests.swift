import Foundation
import Synchronization
import Testing

@testable import AnySpeechRecognizer

/// Counts calls to a custom model's asset methods.
final class AssetCalls: Sendable {
    private let counts = Mutex<[String: Int]>([:])
    func record(_ name: String) { counts.withLock { $0[name, default: 0] += 1 } }
    func count(_ name: String) -> Int { counts.withLock { $0[name, default: 0] } }
}

/// A custom provider that implements only the request-free asset methods.
struct RequestFreeAssetModel: SpeechRecognitionModel {
    let modelID = "test/request-free"
    let capabilities = SpeechRecognitionCapabilities(
        transcription: .init(reportingOptions: [.volatileResults]), liveInput: true)
    let state: SpeechRecognitionAvailability
    let calls: AssetCalls
    let executor = FakeExecutor()

    func makeExecutor() -> any SpeechRecognitionExecutor { executor }

    func availability() async -> SpeechRecognitionAvailability {
        calls.record("availability")
        return state
    }

    func assetInstallationRequest() async throws -> AssetInstallationRequest? {
        calls.record("assetInstallationRequest")
        let calls = calls
        return AssetInstallationRequest(progress: Progress(totalUnitCount: 1)) {
            calls.record("install")
        }
    }
}

/// A custom provider whose volatile results use assets that only improve them.
struct OptionalAssetModel: SpeechRecognitionModel {
    let modelID = "test/optional-assets"
    let capabilities = SpeechRecognitionCapabilities(
        transcription: .init(reportingOptions: [.volatileResults]), liveInput: true)
    let base: SpeechRecognitionAvailability
    let executor = FakeExecutor()

    func makeExecutor() -> any SpeechRecognitionExecutor { executor }
    func availability() async -> SpeechRecognitionAvailability { base }
    func availability(for request: SpeechRecognitionRequest) async
        -> SpeechRecognitionAvailability
    {
        request.includesVolatileResults ? .missingAssets : base
    }
}

@Test func customModelsForwardRequestAwareAssetMethods() async throws {
    let calls = AssetCalls()
    let model = RequestFreeAssetModel(state: .missingAssets, calls: calls)
    let live = SpeechRecognitionRequest.transcription(.init(reportingOptions: [.volatileResults]))
    #expect(await model.availability(for: live) == .missingAssets)
    #expect(calls.count("availability") == 1)
    #expect(try await model.assetInstallationRequest(supporting: [live]) != nil)
    #expect(calls.count("assetInstallationRequest") == 1)

    let modules = [
        SpeechTranscriber(model: model), SpeechTranscriber(model: model),
        SpeechTranscriber(model: model, reportingOptions: [.volatileResults]),
    ]
    #expect(await AssetInventory.status(forModules: modules) == .supported)
    // One model's modules make one installation request.
    let request = try #require(
        try await AssetInventory.assetInstallationRequest(supporting: modules))
    #expect(calls.count("assetInstallationRequest") == 2)
    try await request.downloadAndInstall()
    #expect(calls.count("install") == 1)

    let ready = RequestFreeAssetModel(state: .ready, calls: calls)
    #expect(
        await AssetInventory.status(forModules: [SpeechTranscriber(model: ready)]) == .installed)
}

@Test func optionalAssetsDoNotBlockAnalysis() async throws {
    let model = OptionalAssetModel(base: .ready)
    let live = SpeechTranscriber(model: model, reportingOptions: [.volatileResults])
    #expect(await AssetInventory.status(forModules: [live]) == .supported)
    #expect(
        await AssetInventory.status(forModules: [SpeechTranscriber(model: model)]) == .installed)
    let analyzer = SpeechAnalyzer(modules: [live])
    try await analyzer.prepareToAnalyze(in: nil)
    #expect(await model.executor.prepareCount == 1)
    await analyzer.cancelAndFinishNow()
}

@Test func missingRequiredAssetsBlockAnalysis() async throws {
    let model = OptionalAssetModel(base: .missingAssets)
    let analyzer = SpeechAnalyzer(modules: [
        SpeechTranscriber(model: model, reportingOptions: [.volatileResults])
    ])
    await #expect {
        try await analyzer.prepareToAnalyze(in: nil)
    } throws: { error in
        guard case .missingAssets = error as? SpeechRecognitionError else { return false }
        return true
    }
    #expect(await model.executor.prepareCount == 0)
    await analyzer.cancelAndFinishNow()
}
