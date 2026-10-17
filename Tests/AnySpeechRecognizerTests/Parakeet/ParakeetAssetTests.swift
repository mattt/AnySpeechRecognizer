import Foundation
import HuggingFace
import Testing

@testable import AnySpeechRecognizer

@Suite(.timeLimit(.minutes(1)))
struct ParakeetAssetTests {
    typealias Fixture = ModelFixture

    @Test func builtInManifestsPinVerifiedFiles() throws {
        let manifests = [ParakeetModelManifest.v3, .ultra]
        for manifest in manifests {
            try manifest.validate()
            #expect(manifest.license == "CC-BY-4.0")
            #expect(manifest.attribution.contains("NVIDIA"))
            #expect(manifest.contractVersion == 1)
            #expect(!manifest.files.contains { $0.path.hasPrefix("Preprocessor") })
        }
        #expect(ParakeetModelManifest.v3.components.encoder == "Encoder.mlmodelc")
        #expect(ParakeetModelManifest.ultra.components.encoder == "Encoder.mlmodelc")
        #expect(ParakeetModelManifest.v3.files.count == 21)
        #expect(ParakeetModelManifest.ultra.files.count == 21)
        // The base files and digests are the same as at the previous pins,
        // which did not contain the live encoder.
        for manifest in manifests {
            #expect(manifest.parts == [.base, .liveEncoder])
            #expect(manifest.files(in: .base).count == 16)
            #expect(manifest.files(in: .liveEncoder).count == 5)
            #expect(
                manifest.files(in: .liveEncoder).allSatisfy {
                    $0.path.hasPrefix("Encoder8s.mlmodelc/")
                })
        }
        #expect(ParakeetModelManifest.v3.byteCount(of: .base) == 631_653_214)
        #expect(ParakeetModelManifest.ultra.byteCount(of: .base) == 631_654_261)
        #expect(ParakeetModelManifest.v3.byteCount(of: .liveEncoder) == 590_783_171)
        #expect(ParakeetModelManifest.ultra.byteCount(of: .liveEncoder) == 590_783_520)
        #expect(ParakeetModelManifest.v3.totalByteCount == 1_222_436_385)
        #expect(ParakeetModelManifest.ultra.totalByteCount == 1_222_437_781)
        let store = ModelAssetStore(
            cache: HubCache(cacheDirectory: URL(filePath: "/cache")),
            markerDirectory: URL(filePath: "/verified"))
        #expect(
            store.directory(for: ParakeetModelManifest.v3).path
                == "/cache/models--mattt--parakeet-tdt-0.6b-v3-coreml/snapshots/"
                + "f2f43eaa239d1ea45191c2a2d5bd0f980dae1b1f")
        #expect(
            store.directory(for: ParakeetModelManifest.ultra).path
                == "/cache/models--mattt--parakeet-ultra-coreml/snapshots/"
                + "4038b9e5eced45b04703afb1dab0d30e5dc162b4")
    }

    @Test func unifiedManifestPinsFilesUnderTheNVIDIALicense() throws {
        let manifest = ParakeetModelManifest.unified
        try manifest.validate()
        #expect(ParakeetSpeechModel.Variant.unified.manifest == manifest)
        #expect(ParakeetSpeechModel.Variant.unified.modelDescription.blankID == 1024)
        #expect(manifest.repository == "mattt/parakeet-unified-en-0.6b-coreml")
        #expect(manifest.revision == "a2f32b4bdf3062c32d1668b2bba2354ebfe6b06d")
        #expect(manifest.license == "LicenseRef-NVIDIA-Open-Model-License")
        // Section 3.1 of the NVIDIA Open Model License requires this notice.
        #expect(
            manifest.attribution.contains(
                "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"))
        #expect(manifest.contractVersion == 1)
        #expect(manifest.components.encoder == "Encoder.mlmodelc")
        #expect(manifest.components.liveEncoder == "Encoder8s.mlmodelc")
        #expect(manifest.parts == [.base, .liveEncoder])
        #expect(manifest.files.count == 21)
        #expect(manifest.files(in: .liveEncoder).count == 5)
        #expect(manifest.byteCount(of: .base) == 614_094_723)
        #expect(manifest.byteCount(of: .liveEncoder) == 591_725_504)
        #expect(manifest.totalByteCount == 1_205_820_227)
    }

    @Test func manifestValidationRejectsUnsafeEntries() throws {
        let valid = Fixture.manifest()
        try valid.validate()
        func replacing(path: String? = nil, digest: String? = nil, revision: String? = nil)
            -> ParakeetModelManifest
        {
            var files = valid.files
            files[0] = .init(
                path: path ?? files[0].path, size: files[0].size,
                sha256: digest ?? files[0].sha256)
            return ParakeetModelManifest(
                identifier: valid.identifier, repository: valid.repository,
                revision: revision ?? valid.revision, contractVersion: valid.contractVersion,
                components: valid.components, files: files, license: valid.license,
                attribution: valid.attribution)
        }
        for invalid in [
            replacing(path: "../outside.bin"), replacing(path: "/absolute.bin"),
            replacing(path: "Decoder.mlmodelc//model.mil"),
            replacing(digest: String(repeating: "A", count: 64)),
            replacing(digest: "abc"), replacing(revision: "main"),
        ] {
            #expect(throws: ModelAssetError.self) { try invalid.validate() }
        }
    }

    @Test func installsVerifiedFilesIntoCacheSnapshot() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()
            #expect(store.status(of: manifest) == .missing)

            let directory = try await store.install(manifest)

            #expect(
                directory.standardizedFileURL
                    == root.appending(
                        path: "hub/models--Example--parakeet-test/snapshots/\(manifest.revision)/"
                    ).standardizedFileURL)
            for (path, data) in Fixture.files {
                let url = directory.appending(path: path)
                #expect(try Data(contentsOf: url) == data)
                // Each snapshot entry links to the blob that its ETag names.
                let link = try FileManager.default.destinationOfSymbolicLink(atPath: url.path)
                #expect(link.hasSuffix("blobs/\(Fixture.sha256(data))"))
                #expect(server.requestCount(for: path) == 1)
            }
            #expect(server.downloads.allSatisfy { $0.range == nil })
            #expect(store.status(of: manifest) == .installed)
            #expect(FileManager.default.fileExists(atPath: store.markerURL(for: manifest).path))
            let progress = store.progress(for: manifest)
            #expect(progress.totalUnitCount == manifest.totalByteCount)
            #expect(progress.completedUnitCount == manifest.totalByteCount)

            // A second install finds the verified files and makes no requests.
            let count = server.requests.count
            _ = try await store.install(manifest)
            #expect(server.requests.count == count)
        }
    }

    @Test func markersAreSpecificToManifestRevisionAndCache() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()
            var changed = Fixture.files
            changed[Fixture.vocabulary] = Data("{}".utf8)
            let otherFiles = Fixture.manifest(changed)
            let otherRevision = Fixture.manifest(revision: String(repeating: "f", count: 40))
            let otherCache = ModelAssetStore(
                cache: HubCache(cacheDirectory: root.appending(path: "elsewhere")),
                markerDirectory: store.markerDirectory)
            let markers = [
                store.markerURL(for: manifest), store.markerURL(for: otherFiles),
                store.markerURL(for: otherRevision), otherCache.markerURL(for: manifest),
            ]
            #expect(Set(markers).count == markers.count)
            #expect(store.markerURL(for: manifest) == store.markerURL(for: manifest))

            _ = try await store.install(manifest)

            #expect(store.status(of: manifest) == .installed)
            #expect(store.status(of: otherFiles) == .missing)
            #expect(store.status(of: otherRevision) == .missing)
            #expect(otherCache.status(of: manifest) == .missing)
        }
    }

    @Test func statusRequiresMarkerSizesAndFileNumbers() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()
            let directory = try await store.install(manifest)
            let marker = store.markerURL(for: manifest)
            let markerData = try Data(contentsOf: marker)

            try FileManager.default.removeItem(at: marker)
            #expect(store.status(of: manifest) == .missing)
            let recorded = try JSONDecoder().decode(
                ModelAssetStore.VerificationMarker.self, from: markerData)
            let subject = recorded.subject
            let forged = ModelAssetStore.VerificationMarker(
                subject: .init(
                    repository: subject.repository, revision: String(repeating: "f", count: 40),
                    contractVersion: subject.contractVersion, part: subject.part,
                    files: subject.files, snapshot: subject.snapshot),
                fileNumbers: recorded.fileNumbers)
            try JSONEncoder().encode(forged).write(to: marker)
            #expect(store.status(of: manifest) == .missing)
            try markerData.write(to: marker)
            #expect(store.status(of: manifest) == .installed)

            // Writing through the snapshot link changes the blob in place.
            let weights = directory.appending(path: Fixture.encoderWeights)
            let weightData = try Data(contentsOf: weights)
            try Data(weightData.dropLast()).write(to: weights)
            #expect(store.status(of: manifest) == .missing)
            try weightData.write(to: weights)
            #expect(store.status(of: manifest) == .installed)

            // A replaced blob has a new file number, even with the same contents.
            let blob = weights.resolvingSymlinksInPath()
            try FileManager.default.removeItem(at: blob)
            try weightData.write(to: blob)
            #expect(store.status(of: manifest) == .missing)

            // Installing again verifies the cached files without downloading them.
            let count = server.downloads.count
            _ = try await store.install(manifest)
            #expect(server.downloads.count == count)
            #expect(store.status(of: manifest) == .installed)

            try FileManager.default.removeItem(at: weights)
            #expect(store.status(of: manifest) == .missing)
        }
    }

    @Test func stagedDirectoryNeedsEveryComponentName() throws {
        try FileManager.default.createDirectory(
            at: URL.temporaryDirectory, withIntermediateDirectories: true)
        let directory = URL.temporaryDirectory.appending(path: "anyspeech-staged-\(UUID())")
        defer { try? FileManager.default.removeItem(at: directory) }
        let components = Fixture.manifest().components
        for name in ["Encoder.mlmodelc", "Decoder.mlmodelc", "Joint.mlmodelc"] {
            try FileManager.default.createDirectory(
                at: directory.appending(path: name), withIntermediateDirectories: true)
        }
        #expect(
            ModelAssetStore.status(ofStagedDirectory: directory, components: components)
                == .missing)
        try Data("{}".utf8).write(to: directory.appending(path: Fixture.vocabulary))
        #expect(
            ModelAssetStore.status(ofStagedDirectory: directory, components: components)
                == .installed)
        // The live encoder is optional in a staged directory.
        let live = ParakeetModelManifest.Components(
            encoder: components.encoder, decoder: components.decoder, joint: components.joint,
            vocabulary: components.vocabulary, liveEncoder: "Encoder8s.mlmodelc")
        #expect(
            ModelAssetStore.status(ofStagedDirectory: directory, components: live) == .installed)
        #expect(live.missingPaths(in: directory).isEmpty)
    }

    @Test func manifestsListTheOptionalLiveEncoderWithItsFiles() throws {
        #expect(ParakeetModelManifest.v3.components.liveEncoder == "Encoder8s.mlmodelc")
        #expect(ParakeetModelManifest.ultra.components.liveEncoder == "Encoder8s.mlmodelc")
        let valid = Fixture.manifest()
        #expect(valid.parts == [.base])
        #expect(valid.files(in: .base) == valid.files)
        #expect(valid.files(in: .liveEncoder).isEmpty)
        func manifest(liveEncoder: String?, files: [ParakeetModelManifest.File])
            -> ParakeetModelManifest
        {
            let components = valid.components
            return ParakeetModelManifest(
                identifier: valid.identifier, repository: valid.repository,
                revision: valid.revision, contractVersion: valid.contractVersion,
                components: .init(
                    encoder: components.encoder, decoder: components.decoder,
                    joint: components.joint, vocabulary: components.vocabulary,
                    liveEncoder: liveEncoder),
                files: files, license: valid.license, attribution: valid.attribution)
        }
        let liveFile = ParakeetModelManifest.File(
            path: "Encoder8s.mlmodelc/weights/weight.bin", size: 1,
            sha256: String(repeating: "a", count: 64))
        #expect(throws: ModelAssetError.self) {
            try manifest(liveEncoder: "Encoder8s.mlmodelc", files: valid.files).validate()
        }
        try manifest(liveEncoder: "Encoder8s.mlmodelc", files: valid.files + [liveFile]).validate()
        // The live encoder cannot contain or be a required component.
        #expect(throws: ModelAssetError.self) {
            try manifest(liveEncoder: "Encoder.mlmodelc", files: valid.files).validate()
        }
        let nested = ParakeetModelManifest.File(
            path: "Decoder.mlmodelc/Live.mlmodelc/model.mil", size: 1,
            sha256: String(repeating: "b", count: 64))
        #expect(throws: ModelAssetError.self) {
            try manifest(
                liveEncoder: "Decoder.mlmodelc/Live.mlmodelc", files: valid.files + [nested]
            )
            .validate()
        }
        // Manifests decoded from JSON without the component have no live encoder.
        let data = try JSONEncoder().encode(valid)
        let decoded = try JSONDecoder().decode(ParakeetModelManifest.self, from: data)
        #expect(decoded == valid && decoded.components.liveEncoder == nil)
    }

    @Test func resumesIncompleteBlobWithRange() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()
            let data = try #require(Fixture.files[Fixture.encoderWeights])
            let kept = 40_000
            let incomplete = try writeIncompleteBlob(
                Data(data.prefix(kept)), for: data, of: manifest, in: store)

            let directory = try await store.install(manifest)

            #expect(try Data(contentsOf: directory.appending(path: Fixture.encoderWeights)) == data)
            #expect(
                server.downloads.filter { $0.path == Fixture.encoderWeights }
                    == [.init(method: "GET", path: Fixture.encoderWeights, range: "bytes=\(kept)-")]
            )
            #expect(!FileManager.default.fileExists(atPath: incomplete.path))
        }
    }

    @Test func restartsWhenServerIgnoresRange() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            server.setHonorsRange(false)
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()
            let data = try #require(Fixture.files[Fixture.encoderWeights])
            // These bytes are wrong, so appending the full response would fail verification.
            try writeIncompleteBlob(
                Data(repeating: 0xEE, count: 30_000), for: data, of: manifest, in: store)
            let recorder = ProgressRecorder(store.progress(for: manifest))

            let directory = try await store.install(manifest)
            recorder.invalidate()

            #expect(try Data(contentsOf: directory.appending(path: Fixture.encoderWeights)) == data)
            #expect(server.downloads.contains { $0.range == "bytes=30000-" })
            let values = recorder.recorded
            #expect(values == values.sorted())
            #expect(values.last == manifest.totalByteCount)
        }
    }

    @Test func rejectsDigestMismatchAndRemovesCachedFile() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            let original = try #require(Fixture.files[Fixture.decoderProgram])
            let tampered = Fixture.bytes(original.count, seed: 99)
            server.setFile(Fixture.decoderProgram, tampered)
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()

            let error = await #expect(throws: ModelAssetError.self) {
                try await store.install(manifest)
            }

            guard case .verificationFailed(let path, let reason) = error else {
                Issue.record("Unexpected error: \(String(describing: error))")
                return
            }
            #expect(path == Fixture.decoderProgram)
            #expect(reason.contains("SHA-256"))
            let mapped = recognitionError(try #require(error))
            guard case SpeechRecognitionError.assetVerificationFailed(let message) = mapped else {
                Issue.record("Unexpected mapping: \(mapped)")
                return
            }
            #expect(message.contains(Fixture.decoderProgram))
            let directory = store.directory(for: manifest)
            #expect(!FileManager.default.fileExists(atPath: directory.appending(path: path).path))
            #expect(
                store.cache.cachedBlobPath(
                    repo: "Example/parakeet-test", kind: .model, etag: Fixture.sha256(tampered))
                    == nil)
            #expect(store.status(of: manifest) == .missing)

            // A later install downloads the file again.
            server.setFile(Fixture.decoderProgram, original)
            _ = try await store.install(manifest)
            #expect(server.requestCount(for: Fixture.decoderProgram) == 2)
            #expect(store.status(of: manifest) == .installed)
        }
    }

    @Test(arguments: [-1, 1]) func rejectsSizeMismatch(difference: Int) async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            let original = try #require(Fixture.files[Fixture.jointProgram])
            server.setFile(
                Fixture.jointProgram, Fixture.bytes(original.count + difference, seed: 4))
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()

            let error = await #expect(throws: ModelAssetError.self) {
                try await store.install(manifest)
            }

            guard case .verificationFailed(let path, let reason) = error else {
                Issue.record("Unexpected error: \(String(describing: error))")
                return
            }
            #expect(path == Fixture.jointProgram)
            #expect(reason.contains("bytes"))
            let directory = store.directory(for: manifest)
            #expect(!FileManager.default.fileExists(atPath: directory.appending(path: path).path))
            #expect(store.status(of: manifest) == .missing)
        }
    }

    @Test func cancellationKeepsFinishedFiles() async throws {
        try await withTemporaryModelRoot { root in
            let path = Fixture.encoderWeights
            let data = try #require(Fixture.files[path])
            let others = Fixture.files.keys.filter { $0 != path }
            let server = StubModelServer(files: Fixture.files)
            server.setStall(path, after: 20 * 1024)
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()
            let directory = store.directory(for: manifest)
            func cached(_ path: String) -> Bool {
                FileManager.default.fileExists(atPath: directory.appending(path: path).path)
            }

            let install = Task { try await store.install(manifest) }
            try await waitUntil { server.requestCount(for: path) == 1 && others.allSatisfy(cached) }
            install.cancel()
            await #expect(throws: CancellationError.self) { try await install.value }

            #expect(others.allSatisfy(cached))
            #expect(!cached(path))
            #expect(store.status(of: manifest) == .missing)
            #expect(store.isInstalling(manifest) == false)

            server.setStall(path, after: nil)
            _ = try await store.install(manifest)
            #expect(try Data(contentsOf: directory.appending(path: path)) == data)
            for other in others { #expect(server.requestCount(for: other) == 1) }
            // swift-huggingface keeps no partial bytes after cancellation,
            // so the interrupted file starts over.
            #expect(
                server.downloads.filter { $0.path == path }
                    == Array(repeating: .init(method: "GET", path: path, range: nil), count: 2))
            let progress = store.progress(for: manifest)
            #expect(progress.completedUnitCount == manifest.totalByteCount)
        }
    }

    @Test func progressCountsCachedBytesAndNeverMovesBackward() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            let store = server.makeStore(root: root, maximumConcurrentDownloads: 2)
            let manifest = Fixture.manifest()
            let present = try #require(Fixture.files[Fixture.decoderProgram])
            try await store.cache.storeData(
                present, repo: "Example/parakeet-test", kind: .model,
                revision: manifest.revision, filename: Fixture.decoderProgram,
                etag: Fixture.sha256(present))
            let recorder = ProgressRecorder(store.progress(for: manifest))

            _ = try await store.install(manifest)
            recorder.invalidate()

            #expect(server.requestCount(for: Fixture.decoderProgram) == 0)
            let values = recorder.recorded
            #expect(values == values.sorted())
            #expect(values.count > Fixture.files.count)
            #expect(values.dropFirst().first ?? 0 >= Int64(present.count))
            #expect(values.last == manifest.totalByteCount)
        }
    }

    @Test func concurrentInstallsShareOneDownload() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            server.hold()
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()

            let first = Task { try await store.install(manifest) }
            let second = Task { try await store.install(manifest) }
            try await waitUntil { store.waiterCount(for: manifest) == 2 }
            server.release()

            let directories = try await [first.value, second.value]
            #expect(directories[0] == directories[1])
            for path in Fixture.files.keys { #expect(server.requestCount(for: path) == 1) }
            #expect(store.status(of: manifest) == .installed)
        }
    }

    @Test func cancellingOneWaiterKeepsSharedInstall() async throws {
        try await withTemporaryModelRoot { root in
            let server = StubModelServer(files: Fixture.files)
            server.hold()
            let store = server.makeStore(root: root)
            let manifest = Fixture.manifest()

            let cancelled = Task { try await store.install(manifest) }
            let kept = Task { try await store.install(manifest) }
            try await waitUntil {
                store.waiterCount(for: manifest) == 2 && !server.requests.isEmpty
            }
            cancelled.cancel()
            await #expect(throws: CancellationError.self) { try await cancelled.value }
            #expect(store.waiterCount(for: manifest) == 1)
            #expect(store.isInstalling(manifest))
            server.release()

            _ = try await kept.value
            for path in Fixture.files.keys { #expect(server.requestCount(for: path) == 1) }
            #expect(store.status(of: manifest) == .installed)
        }
    }

    /// Writes the bytes that an interrupted download left in the cache's blob store.
    @discardableResult
    private func writeIncompleteBlob(
        _ data: Data, for file: Data, of manifest: ParakeetModelManifest,
        in store: ModelAssetStore
    ) throws -> URL {
        let url = try store.cache.incompleteBlobPath(
            repo: try #require(Repo.ID(rawValue: manifest.repository)), kind: .model,
            etag: Fixture.sha256(file))
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
        return url
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(10)
        while !condition() {
            guard ContinuousClock.now < deadline else {
                Issue.record("The condition did not become true within 10 seconds.")
                throw CancellationError()
            }
            try await Task.sleep(for: .milliseconds(2))
        }
    }
}

/// Installs only small files from the pinned v3 revision to check real URLs, redirects,
/// cache layout, and digests.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["ANYSPEECH_NETWORK_TESTS"] == "1"))
struct ParakeetAssetNetworkTests {
    @Test func installsSmallPinnedFilesFromHuggingFace() async throws {
        try await withTemporaryModelRoot { root in
            let v3 = ParakeetModelManifest.v3
            // One Git file through a redirect to the resolve cache and one LFS file through
            // a redirect to the CDN; neither is a weight file.
            let paths = ["parakeet_vocab.json", "Decoder.mlmodelc/coremldata.bin"]
            let manifest = ParakeetModelManifest(
                identifier: "\(v3.identifier)-network-test", repository: v3.repository,
                revision: v3.revision, contractVersion: v3.contractVersion,
                components: .init(
                    encoder: "Decoder.mlmodelc", decoder: "Decoder.mlmodelc",
                    joint: "Decoder.mlmodelc", vocabulary: "parakeet_vocab.json"),
                files: v3.files.filter { paths.contains($0.path) }, license: v3.license,
                attribution: v3.attribution)
            #expect(manifest.files.count == 2)
            #expect(manifest.totalByteCount < 200_000)
            let cache = HubCache(cacheDirectory: root.appending(path: "hub"))

            let store = ModelAssetStore(
                cache: cache, markerDirectory: root.appending(path: "verified"))
            let directory = try await store.install(manifest)

            #expect(store.status(of: manifest) == .installed)
            let lfs = try #require(manifest.files.first { $0.path.hasPrefix("Decoder") })
            let link = try FileManager.default.destinationOfSymbolicLink(
                atPath: directory.appending(path: lfs.path).path)
            #expect(link.hasSuffix("blobs/\(lfs.sha256)"))

            // Another store with an unreachable host verifies the cached files without network.
            let offline = ModelAssetStore(
                cache: cache, markerDirectory: root.appending(path: "offline"),
                host: URL(string: "https://anyspeech.invalid")!)
            #expect(offline.status(of: manifest) == .missing)
            #expect(try await offline.install(manifest) == directory)
            #expect(offline.status(of: manifest) == .installed)
        }
    }
}
