import Foundation
import HuggingFace
import Testing

@testable import AnySpeechRecognizer

struct SortformerManifestTests {
    @Test func builtInManifestsPinVerifiedFiles() throws {
        let v2 = SortformerModelManifest.builtIn
        let latest = SortformerModelManifest.v2_1
        for manifest in [v2, latest] {
            try manifest.validate()
            #expect(manifest.contractVersion == 1)
            #expect(manifest.parts == [.base])
            #expect(manifest.files.allSatisfy { $0.path.hasPrefix(manifest.model + "/") })
            #expect(manifest.files.contains { $0.path == manifest.model + "/coremldata.bin" })
        }
        #expect(v2.identifier == "diar-streaming-sortformer-4spk-v2")
        #expect(v2.license == "CC-BY-4.0")
        #expect(v2.files.count == 10)

        #expect(latest.identifier == "diar-streaming-sortformer-4spk-v2.1")
        #expect(latest.repository == "mattt/diar-streaming-sortformer-4spk-v2.1-coreml")
        #expect(latest.model == "SortformerLowLatency.mlmodelc")
        #expect(latest.license == "LicenseRef-NVIDIA-Open-Model-License")
        #expect(latest.files.count == 11)

        let store = ModelAssetStore(
            cache: HubCache(cacheDirectory: URL(filePath: "/cache")),
            markerDirectory: URL(filePath: "/verified"))
        #expect(
            store.directory(for: latest).path
                == "/cache/models--mattt--diar-streaming-sortformer-4spk-v2.1-coreml/snapshots/"
                + latest.revision)
        #expect(store.directory(for: v2) != store.directory(for: latest))
    }

    @Test func versionsSelectTheirManifestsAndAttribution() async throws {
        #expect(SortformerDiarizationModel.Version.allCases == [.v2, .v2_1])
        #expect(SortformerDiarizationModel.Version(rawValue: "v2.1") == .v2_1)
        #expect(SortformerDiarizationModel.Version.v2_1.rawValue == "v2.1")
        #expect(SortformerDiarizationModel.Version(rawValue: "v3") == nil)
        #expect(Set(SortformerDiarizationModel.Version.allCases).count == 2)
        let encoded = try JSONEncoder().encode(SortformerDiarizationModel.Version.v2_1)
        #expect(String(decoding: encoded, as: UTF8.self) == "\"v2.1\"")
        let decoder = JSONDecoder()
        #expect(
            try decoder.decode(SortformerDiarizationModel.Version.self, from: Data("\"v2\"".utf8))
                == .v2)
        #expect(throws: DecodingError.self) {
            try decoder.decode(SortformerDiarizationModel.Version.self, from: Data("\"v3\"".utf8))
        }

        // v2 stays the default.
        let v2 = SortformerDiarizationModel()
        #expect(v2.version == .v2)
        #expect(v2.modelID == "diar-streaming-sortformer-4spk-v2")
        #expect(v2.attribution.contains("licensed under CC BY 4.0"))

        let latest = SortformerDiarizationModel(version: .v2_1)
        #expect(latest.modelID == "diar-streaming-sortformer-4spk-v2.1")
        #expect(latest.postProcessing == .callHome)
        #expect(latest.capabilities == v2.capabilities)
        // Section 3.1 of the NVIDIA Open Model License requires this notice verbatim.
        #expect(
            latest.attribution.contains(
                "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"))
        #expect(latest.attribution.contains("diar_streaming_sortformer_4spk-v2.1"))
        #expect(
            latest.attribution.contains(
                "https://www.nvidia.com/en-us/agreements/enterprise-software/"
                    + "nvidia-open-model-license/"))
        #expect(!latest.attribution.contains("\n"))

        // A staged model of either version reports that version's identifier.
        let staged = SortformerDiarizationModel(
            version: .v2_1, assets: .compiledModel(URL(filePath: "/staged/Model.mlmodelc")))
        #expect(staged.modelID == "diar-streaming-sortformer-4spk-v2.1")
        let diagnostics = await staged.diagnostics()
        #expect(diagnostics.modelID == "diar-streaming-sortformer-4spk-v2.1")
        #expect(diagnostics.modelRepository == "/staged/Model.mlmodelc")
        let builtIn = await latest.diagnostics()
        #expect(
            builtIn.modelRepository
                == "mattt/diar-streaming-sortformer-4spk-v2.1-coreml/SortformerLowLatency.mlmodelc")
        #expect(builtIn.modelRevision == SortformerModelManifest.v2_1.revision)
    }
}
