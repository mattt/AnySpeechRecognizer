import Foundation
import Testing

@testable import AnySpeechRecognizer

/// Built-in Nemotron assets: the pinned manifest, and installation through `AssetInventory`
/// from a stub server, without network access.
@Suite(.timeLimit(.minutes(1)))
struct NemotronAssetTests {
    /// The files of the published conversion, as `Tools/NemotronConversion/package.py` lays
    /// them out.
    static let files: [String: Data] = [
        "Encoder.mlmodelc/model.mil": ModelFixture.bytes(5_000, seed: 21),
        "Encoder.mlmodelc/weights/weight.bin": ModelFixture.bytes(64 * 1024, seed: 22),
        "Decoder.mlmodelc/model.mil": ModelFixture.bytes(3_000, seed: 23),
        "JointDecision.mlmodelc/model.mil": ModelFixture.bytes(3_000, seed: 24),
        "vocab.json": Data(#"{"0": "<unk>"}"#.utf8),
        "prompts.json": Data(#"{"auto": 101, "en-US": 0}"#.utf8),
        "LICENSE": Data("OpenMDW License Agreement, version 1.1".utf8),
        "NOTICE": Data("Nemotron 3.5 ASR Streaming 0.6B".utf8),
    ]

    static func manifest(_ files: [String: Data] = files) -> NemotronModelManifest {
        NemotronModelManifest(
            identifier: "nemotron-test",
            repository: "Example/nemotron-test",
            revision: "0123456789abcdef0123456789abcdef01234567",
            contractVersion: 1,
            files: files.keys.sorted().map { path in
                .init(
                    path: path, size: Int64(files[path]!.count),
                    sha256: ModelFixture.sha256(files[path]!))
            },
            license: "OpenMDW-1.1",
            attribution: "Test fixture.")
    }

    @Test func manifestsNeedEveryComponent() throws {
        try Self.manifest().validate()
        var incomplete = Self.files
        incomplete["prompts.json"] = nil
        #expect(throws: ModelAssetError.self) { try Self.manifest(incomplete).validate() }
        if let manifest = NemotronModelManifest.builtIn {
            try manifest.validate()
            #expect(manifest.license == "OpenMDW-1.1")
            #expect(manifest.files.contains { $0.path == "LICENSE" })
            #expect(manifest.files.contains { $0.path == "NOTICE" })
        }
    }

    @Test func defaultsToBuiltInAssets() {
        let model = NemotronSpeechModel()
        #expect(model.assets == .builtIn)
        #expect(model.manifest == NemotronModelManifest.builtIn)
    }

    @Test func installsPinnedFilesThroughAssetInventory() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Self.files)
            let store = server.makeStore(root: root)
            let manifest = Self.manifest()
            let model = NemotronSpeechModel(assets: .builtIn, manifest: manifest, store: store)
            let transcriber = SpeechTranscriber(model: model)
            #expect(await model.availability() == .missingAssets)
            #expect(await AssetInventory.status(forModules: [transcriber]) == .supported)

            let request = try #require(
                try await AssetInventory.assetInstallationRequest(supporting: [transcriber]))
            try await request.downloadAndInstall()

            #expect(await model.availability() == .ready)
            #expect(await AssetInventory.status(forModules: [transcriber]) == .installed)
            #expect(try await model.assetInstallationRequest() == nil)
            let directory = try #require(model.directory)
            #expect(directory == store.directory(for: manifest))
            for path in Self.files.keys {
                #expect(
                    FileManager.default.fileExists(atPath: directory.appending(path: path).path))
            }
            #expect(model.attribution == "Test fixture.")
            #expect(model.modelID == "nemotron-test")
            let diagnostics = await model.diagnostics()
            #expect(diagnostics.modelRepository == "Example/nemotron-test")
            #expect(diagnostics.modelRevision == manifest.revision)
        }
    }

    /// Without a pinned conversion, built-in assets report missing assets and cannot install.
    @Test func withoutAManifestNeedsAStagedDirectory() async throws {
        try await withTemporaryModelRoot { root in
            let model = NemotronSpeechModel(
                assets: .builtIn, manifest: nil,
                store: StubModelServer(files: [:]).makeStore(root: root))
            #expect(model.directory == nil)
            #expect(await model.availability() == .missingAssets)
            await #expect(throws: SpeechRecognitionError.self) {
                _ = try await model.assetInstallationRequest()
            }
            let executor = NemotronExecutor(model: model)
            await #expect(throws: SpeechRecognitionError.self) {
                try await executor.prepare(
                    request: .transcription(.init()), format: AudioFormats.monoFloat16k,
                    context: .init())
            }
        }
    }
}
