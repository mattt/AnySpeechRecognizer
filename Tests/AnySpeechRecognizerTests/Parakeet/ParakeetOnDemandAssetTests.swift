import AVFoundation
import Foundation
import Testing

@testable import AnySpeechRecognizer

/// On-demand installation of the base files and the optional live encoder,
/// through the store and through `AssetInventory`, without network access.
@Suite(.timeLimit(.minutes(1)))
struct ParakeetOnDemandAssetTests {
    typealias Fixture = ModelFixture

    static let basePaths = Fixture.files.keys.sorted()
    static let livePaths = Fixture.liveFiles.keys.sorted()

    func makeModel(_ store: ModelAssetStore) -> ParakeetSpeechModel {
        ParakeetSpeechModel(manifest: Fixture.manifestWithLiveEncoder(), store: store)
    }

    func transcriber(
        _ model: ParakeetSpeechModel, _ options: Set<SpeechTranscriber.ReportingOption> = []
    ) -> SpeechTranscriber {
        SpeechTranscriber(model: model, reportingOptions: options)
    }

    @Test func manifestPartsSplitTheLiveEncoderFiles() throws {
        let manifest = Fixture.manifestWithLiveEncoder()
        try manifest.validate()
        #expect(manifest.parts == [.base, .liveEncoder])
        #expect(manifest.files(in: .base).map(\.path) == Self.basePaths)
        #expect(manifest.files(in: .liveEncoder).map(\.path) == Self.livePaths)
        #expect(
            manifest.byteCount(of: .base) + manifest.byteCount(of: .liveEncoder)
                == manifest.totalByteCount)
    }

    @Test func eachPartHasItsOwnMarker() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.filesWithLiveEncoder)
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifestWithLiveEncoder()
            let base = store.markerURL(for: manifest, part: .base)
            let live = store.markerURL(for: manifest, part: .liveEncoder)
            #expect(base != live)
            #expect(base.lastPathComponent.contains("--\(manifest.revision)--base--"))
            #expect(live.lastPathComponent.contains("--\(manifest.revision)--liveEncoder--"))
            // The base marker covers only the base files, so it is the same with or without
            // a live encoder in the manifest.
            #expect(base == store.markerURL(for: Fixture.manifest(), part: .base))
            // A manifest without a live encoder has no live encoder part.
            #expect(store.status(of: Fixture.manifest(), part: .liveEncoder) == .missing)

            try await store.install(manifest)

            #expect(FileManager.default.fileExists(atPath: base.path))
            #expect(!FileManager.default.fileExists(atPath: live.path))
            let marker = try JSONDecoder().decode(
                ModelAssetStore.VerificationMarker.self, from: Data(contentsOf: base))
            #expect(marker.subject.part == .base)
            #expect(marker.subject.files.map(\.path) == Self.basePaths)
            #expect(marker.fileNumbers.keys.sorted() == Self.basePaths)

            try await store.install(manifest, parts: [.liveEncoder])

            let liveMarker = try JSONDecoder().decode(
                ModelAssetStore.VerificationMarker.self, from: Data(contentsOf: live))
            #expect(liveMarker.subject.part == .liveEncoder)
            #expect(liveMarker.fileNumbers.keys.sorted() == Self.livePaths)

            // Removing the live encoder leaves the base installed.
            let directory = store.directory(for: manifest)
            try FileManager.default.removeItem(
                at: directory.appending(path: Fixture.liveEncoderWeights))
            #expect(store.status(of: manifest, part: .liveEncoder) == .missing)
            #expect(store.status(of: manifest, part: .base) == .installed)
            try FileManager.default.removeItem(at: live)
            #expect(store.status(of: manifest, part: .base) == .installed)
        }
    }

    @Test func liveEncoderInstallsLaterWithoutDownloadingOrHashingTheBase() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.filesWithLiveEncoder)
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifestWithLiveEncoder()
            let directory = try await store.install(manifest, parts: [.base])
            for path in Self.basePaths { #expect(server.requestCount(for: path) == 1) }
            for path in Self.livePaths { #expect(server.requestCount(for: path) == 0) }
            let baseMarker = store.markerURL(for: manifest, part: .base)
            let markerData = try Data(contentsOf: baseMarker)

            // Rewrite a base file in place with the same size: status reads only sizes and
            // file numbers, so the base still reads as installed. A digest check of the base
            // would now fail, so the live install below shows that it hashes no base file.
            let weights = directory.appending(path: Fixture.encoderWeights)
            let original = try Data(contentsOf: weights)
            let handle = try FileHandle(forWritingTo: weights)
            try handle.write(contentsOf: Data(repeating: 0xEE, count: 16))
            try handle.close()
            #expect(store.status(of: manifest, part: .base) == .installed)

            try await store.install(manifest, parts: [.base, .liveEncoder])

            for path in Self.basePaths { #expect(server.requestCount(for: path) == 1) }
            for path in Self.livePaths { #expect(server.requestCount(for: path) == 1) }
            #expect(try Data(contentsOf: baseMarker) == markerData)
            #expect(try Data(contentsOf: weights) != original)
            #expect(store.status(of: manifest, part: .base) == .installed)
            #expect(store.status(of: manifest, part: .liveEncoder) == .installed)
            let progress = store.progress(for: manifest, part: .liveEncoder)
            #expect(progress.totalUnitCount == manifest.byteCount(of: .liveEncoder))
            #expect(progress.completedUnitCount == manifest.byteCount(of: .liveEncoder))
        }
    }

    @Test func requestAwareStatusAndInstallation() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.filesWithLiveEncoder)
            let store = server.makeStore(root: root)
            let model = makeModel(store)
            let manifest = model.manifest
            let file = transcriber(model)
            let live = transcriber(model, [.volatileResults])
            let fast = transcriber(model, [.fastResults])

            #expect(await AssetInventory.status(forModules: [file]) == .supported)
            #expect(await AssetInventory.status(forModules: [live]) == .supported)

            // A module without volatile or fast results installs only the base files.
            let baseRequest = try #require(
                try await AssetInventory.assetInstallationRequest(supporting: [file]))
            #expect(baseRequest.progress.totalUnitCount == manifest.byteCount(of: .base))
            try await baseRequest.downloadAndInstall()
            #expect(baseRequest.progress.completedUnitCount == manifest.byteCount(of: .base))
            for path in Self.basePaths { #expect(server.requestCount(for: path) == 1) }
            for path in Self.livePaths { #expect(server.requestCount(for: path) == 0) }

            let requests = server.requests.count
            #expect(await AssetInventory.status(forModules: [file]) == .installed)
            #expect(await AssetInventory.status(forModules: [live]) == .supported)
            #expect(await AssetInventory.status(forModules: [fast]) == .supported)
            #expect(await AssetInventory.status(forModules: [file, live]) == .supported)
            #expect(await model.availability() == .ready)
            #expect(await model.availability(for: .transcription(.init())) == .ready)
            #expect(
                await model.availability(
                    for: .transcription(.init(reportingOptions: [.fastResults])))
                    == .missingAssets)
            #expect(try await AssetInventory.assetInstallationRequest(supporting: [file]) == nil)
            #expect(try await model.assetInstallationRequest() == nil)
            // Status checks make no requests.
            #expect(server.requests.count == requests)

            // A module with volatile results adds only the live encoder.
            let liveRequest = try #require(
                try await AssetInventory.assetInstallationRequest(supporting: [file, live]))
            #expect(liveRequest.progress.totalUnitCount == manifest.byteCount(of: .liveEncoder))
            try await liveRequest.downloadAndInstall()
            for path in Self.basePaths { #expect(server.requestCount(for: path) == 1) }
            for path in Self.livePaths { #expect(server.requestCount(for: path) == 1) }
            #expect(await AssetInventory.status(forModules: [file, live, fast]) == .installed)
            #expect(
                try await AssetInventory.assetInstallationRequest(supporting: [file, live, fast])
                    == nil)
        }
    }

    @Test func mixedModulesShareOneInstallation() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.filesWithLiveEncoder)
            let store = server.makeStore(root: root)
            let model = makeModel(store)
            let manifest = model.manifest
            let modules = [
                transcriber(model), transcriber(model, [.volatileResults]),
                transcriber(model, [.fastResults]),
            ]

            let request = try #require(
                try await AssetInventory.assetInstallationRequest(supporting: modules))
            #expect(request.progress.totalUnitCount == manifest.totalByteCount)
            let recorder = ProgressRecorder(request.progress)
            try await request.downloadAndInstall()
            recorder.invalidate()

            for path in Self.basePaths + Self.livePaths {
                #expect(server.requestCount(for: path) == 1)
            }
            let values = recorder.recorded
            #expect(values == values.sorted())
            #expect(values.last == manifest.totalByteCount)
            #expect(await AssetInventory.status(forModules: modules) == .installed)
            #expect(store.status(of: manifest, part: .base) == .installed)
            #expect(store.status(of: manifest, part: .liveEncoder) == .installed)
        }
    }

    @Test func requestsHaveTheirOwnProgress() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.filesWithLiveEncoder)
            let model = makeModel(server.makeStore(root: root))
            let live = transcriber(model, [.volatileResults])
            let first = try #require(try await model.assetInstallationRequest())
            let second = try #require(
                try await model.assetInstallationRequest(supporting: [.transcription(.init())]))
            #expect(first.progress !== second.progress)
            // A combined request adds each model request's progress as a child,
            // which fails if another parent already has it.
            let combined = AssetInstallationRequest(combining: [first, second])
            let again = try #require(
                try await AssetInventory.assetInstallationRequest(supporting: [live]))
            _ = AssetInstallationRequest(combining: [again])
            try await combined.downloadAndInstall()
            #expect(combined.progress.fractionCompleted == 1)
            try await again.downloadAndInstall()
            #expect(again.progress.completedUnitCount == again.progress.totalUnitCount)
        }
    }

    @Test func liveResultsFallBackUntilTheLiveEncoderIsVerified() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.filesWithLiveEncoder)
            let store = server.makeStore(root: root)
            let model = makeModel(store)
            let manifest = model.manifest
            let regular = ParakeetSpeechModel.EncoderSelection(
                encoder: "Encoder.mlmodelc", forcedCommitEncoder: nil)
            try await store.install(manifest)
            #expect(model.encoderSelection(live: false) == regular)
            #expect(model.encoderSelection(live: true) == regular)
            #expect(
                await model.diagnostics().computeConfiguration.hasSuffix(
                    "file results use Encoder.mlmodelc; volatile and fast results use "
                        + "Encoder.mlmodelc because the live encoder Encoder8s.mlmodelc "
                        + "is not installed"))

            // Live encoder files without a verification marker are not used.
            for (path, data) in Fixture.liveFiles {
                try await store.cache.storeData(
                    data, repo: "Example/parakeet-test", kind: .model,
                    revision: manifest.revision, filename: path, etag: Fixture.sha256(data))
            }
            #expect(
                FileManager.default.fileExists(
                    atPath: model.directory.appending(path: Fixture.liveEncoderWeights).path))
            #expect(model.encoderSelection(live: true) == regular)

            let requests = server.downloads.count
            try await store.install(manifest, parts: [.liveEncoder])
            // The cached files are verified without downloading them.
            #expect(server.downloads.count == requests)
            #expect(model.encoderSelection(live: false) == regular)
            #expect(
                model.encoderSelection(live: true)
                    == .init(encoder: "Encoder8s.mlmodelc", forcedCommitEncoder: "Encoder.mlmodelc")
            )
            #expect(
                await model.diagnostics().computeConfiguration.hasSuffix(
                    "file results use Encoder.mlmodelc; volatile and fast results use "
                        + "Encoder8s.mlmodelc, with Encoder.mlmodelc for forced commits"))
        }
    }

    @Test func analysisDoesNotWaitForOrDownloadTheLiveEncoder() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.filesWithLiveEncoder)
            let store = server.makeStore(root: root)
            let model = makeModel(store)
            try await store.install(model.manifest)
            let requests = server.requests.count
            let live = transcriber(model, [.volatileResults])
            let analyzer = SpeechAnalyzer(modules: [live])
            // The fixture files are not Core ML models, so loading fails after the
            // availability check; the check itself does not report missing assets.
            let error = await #expect(throws: (any Error).self) {
                try await analyzer.prepareToAnalyze(in: nil)
            }
            if case .missingAssets(let message)? = error as? SpeechRecognitionError {
                Issue.record("Analysis required the live encoder: \(message)")
            }
            await analyzer.cancelAndFinishNow()
            #expect(server.requests.count == requests)
        }
    }

    @Test func stagedDirectoriesUseTheLiveEncoderWhenPresent() throws {
        let directory = URL.temporaryDirectory.appending(path: "anyspeech-staged-\(UUID())")
        defer { try? FileManager.default.removeItem(at: directory) }
        for name in ["Encoder.mlmodelc", "Decoder.mlmodelc", "JointDecisionv3.mlmodelc"] {
            try FileManager.default.createDirectory(
                at: directory.appending(path: name), withIntermediateDirectories: true)
        }
        let model = ParakeetSpeechModel(assets: .directory(directory))
        let regular = ParakeetSpeechModel.EncoderSelection(
            encoder: "Encoder.mlmodelc", forcedCommitEncoder: nil)
        #expect(model.encoderSelection(live: true) == regular)
        try FileManager.default.createDirectory(
            at: directory.appending(path: "Encoder8s.mlmodelc"), withIntermediateDirectories: true)
        #expect(model.encoderSelection(live: false) == regular)
        #expect(
            model.encoderSelection(live: true)
                == .init(encoder: "Encoder8s.mlmodelc", forcedCommitEncoder: "Encoder.mlmodelc"))
    }
}
