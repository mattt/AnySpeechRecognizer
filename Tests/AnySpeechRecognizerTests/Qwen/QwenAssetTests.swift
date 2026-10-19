import Foundation
import Testing

@testable import AnySpeechRecognizer

/// Built-in Qwen3-ASR assets: pinned manifests, and installation through `AssetInventory`
/// from a stub server, without network access.
@Suite(.timeLimit(.minutes(1)))
struct QwenAssetTests {
    /// The files of a published conversion, as `Tools/QwenConversion/package.py` lays them out.
    static let files: [String: Data] = [
        "AudioEncoder.mlmodelc/model.mil": ModelFixture.bytes(5_000, seed: 11),
        "AudioEncoder.mlmodelc/weights/weight.bin": ModelFixture.bytes(64 * 1024, seed: 12),
        "TextDecoder.mlmodelc/model.mil": ModelFixture.bytes(7_000, seed: 13),
        "TextDecoder.mlmodelc/weights/weight.bin": ModelFixture.bytes(96 * 1024, seed: 14),
        "vocab.json": Data(#"["!", "Hello"]"#.utf8),
        "qwen3_asr_config.json": Data("{}".utf8),
        "LICENSE": Data("Apache License".utf8),
        "NOTICE": Data("Qwen3-ASR".utf8),
    ]

    static func manifest(_ files: [String: Data] = files) -> QwenModelManifest {
        QwenModelManifest(
            identifier: "qwen3-asr-test",
            repository: "Example/qwen3-asr-test",
            revision: "0123456789abcdef0123456789abcdef01234567",
            contractVersion: 1,
            files: files.keys.sorted().map { path in
                .init(
                    path: path, size: Int64(files[path]!.count),
                    sha256: ModelFixture.sha256(files[path]!))
            },
            license: "Apache-2.0",
            attribution: "Test fixture.")
    }

    @Test func manifestsNeedEveryComponent() throws {
        try Self.manifest().validate()
        var incomplete = Self.files
        incomplete["qwen3_asr_config.json"] = nil
        #expect(throws: ModelAssetError.self) { try Self.manifest(incomplete).validate() }
        for manifest in QwenModelManifest.builtIn.values {
            try manifest.validate()
            #expect(manifest.license == "Apache-2.0")
            #expect(manifest.files.contains { $0.path == "LICENSE" })
        }
        #expect(Set(QwenModelManifest.builtIn.keys) == [.small, .large])
    }

    /// The default model is the larger size with built-in assets.
    @Test func defaultsToBuiltInAssetsOfTheLargerSize() {
        let model = QwenSpeechModel()
        #expect(model.variant == .large)
        #expect(model.assets == .builtIn)
        #expect(model.manifest == QwenModelManifest.builtIn[.large])
    }

    @Test func installsPinnedFilesThroughAssetInventory() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Self.files)
            let store = server.makeStore(root: root)
            let manifest = Self.manifest()
            let model = QwenSpeechModel(
                variant: .large, assets: .builtIn, manifest: manifest, store: store)
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
            #expect(model.modelID == "qwen3-asr-test")
            let diagnostics = await model.diagnostics()
            #expect(diagnostics.modelRepository == "Example/qwen3-asr-test")
            #expect(diagnostics.modelRevision == manifest.revision)
        }
    }

    /// A size without a pinned conversion reports missing assets and cannot install.
    @Test func sizesWithoutAManifestNeedAStagedDirectory() async throws {
        try await withTemporaryModelRoot { root in
            let model = QwenSpeechModel(
                variant: .small, assets: .builtIn, manifest: nil,
                store: StubModelServer(files: [:]).makeStore(root: root))
            #expect(model.directory == nil)
            #expect(await model.availability() == .missingAssets)
            await #expect(throws: SpeechRecognitionError.self) {
                _ = try await model.assetInstallationRequest()
            }
            let executor = QwenExecutor(model: model)
            await #expect(throws: SpeechRecognitionError.self) {
                try await executor.prepare(
                    request: .transcription(.init()), format: AudioFormats.monoFloat16k,
                    context: .init())
            }
        }
    }
}
