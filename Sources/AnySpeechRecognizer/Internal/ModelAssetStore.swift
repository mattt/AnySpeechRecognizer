import CryptoKit
import Foundation
import HuggingFace
import Synchronization

/// Locates, checks, and installs Parakeet model files in the Hugging Face cache.
///
/// The store accepts any `PinnedModelManifest`, so the Sortformer diarization model
/// installs the same way, with a single base part.
///
/// swift-huggingface downloads each manifest file at the pinned revision into the cache,
/// `<cache>/models--<owner>--<name>/snapshots/<revision>/`, as a symlink into the cache's
/// blob store; the runtime loads the files in place.
/// The cache location follows swift-huggingface: `HF_HUB_CACHE`, then `HF_HOME/hub`,
/// then `~/.cache/huggingface/hub`, or `Library/Caches/huggingface/hub` in sandboxed apps.
///
/// The store installs and verifies each part of a manifest separately
/// (see `ParakeetModelManifest.Part`): the base files that every configuration needs,
/// and the optional live encoder that only volatile and fast results use.
/// After every file of a part matches its manifest size and SHA-256 digest, the store writes
/// a verification marker for that part in its own directory, so status checks compare file
/// sizes and file numbers instead of digests.
/// A base install therefore has one marker, and a base and live encoder install has two.
/// Installing the live encoder later writes only its own marker; it neither downloads nor
/// hashes the base files again, and a missing or stale live encoder marker never makes
/// the base read as missing.
/// Concurrent installs of one part through the same store share one download,
/// and cancelling one caller does not cancel the download while another caller waits.
/// Use `shared` so that every caller in a process shares installs.
final class ModelAssetStore: Sendable {
    typealias Part = ParakeetModelManifest.Part

    /// The local state of a manifest part or staged directory.
    enum Status: Sendable, Hashable {
        /// Every required file is present and recorded as verified.
        case installed
        /// At least one file is absent, has the wrong size, or is unverified.
        case missing
    }

    /// The record that an install writes after every file of one part passes verification.
    struct VerificationMarker: Sendable, Hashable, Codable {
        /// The facts that a marker vouches for. The marker's file name includes their digest.
        struct Subject: Sendable, Hashable, Codable {
            let repository: String
            let revision: String
            let contractVersion: Int
            /// The part of the manifest that the marker covers.
            let part: Part
            /// The part's files, without the files of other parts.
            let files: [ParakeetModelManifest.File]
            /// The snapshot directory that holds the verified files.
            let snapshot: String
        }

        let subject: Subject
        /// The file system number of each verified file's contents, keyed by path,
        /// so that a replaced blob reads as unverified.
        let fileNumbers: [String: UInt64]
    }

    private final class Installation: Sendable {
        let work = SharedPreparation<URL>(retainValue: false)
        let progress: Progress
        init(byteCount: Int64) { progress = Progress(totalUnitCount: byteCount) }
    }

    static let shared = ModelAssetStore()

    /// The package's Hugging Face cache.
    static var defaultCache: HubCache {
        HubCache(location: .custom { defaultCacheDirectory() })
    }

    /// Resolves the cache directory for model files.
    ///
    /// `HF_HUB_CACHE` and `HF_HOME` keep their usual meaning.
    /// A Mac app outside the sandbox shares `~/.cache/huggingface/hub` with other tools.
    /// Sandboxed apps, including every iOS app, use
    /// `Application Support/AnySpeechRecognizer/HuggingFace/hub` instead of the library's
    /// default in `Library/Caches`, which iOS can purge when storage is low.
    /// The package excludes that directory from backup when it installs files.
    static func defaultCacheDirectory(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        applicationSupport: URL = .applicationSupportDirectory,
        home: URL = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
    ) -> URL {
        if let hubCache = environment["HF_HUB_CACHE"] {
            return URL(fileURLWithPath: NSString(string: hubCache).expandingTildeInPath)
        }
        if let hfHome = environment["HF_HOME"] {
            return URL(fileURLWithPath: NSString(string: hfHome).expandingTildeInPath)
                .appending(path: "hub")
        }
        #if os(macOS)
            if environment["APP_SANDBOX_CONTAINER_ID"] == nil {
                return home.appending(path: ".cache/huggingface/hub", directoryHint: .isDirectory)
            }
        #endif
        return packageCacheDirectory(applicationSupport: applicationSupport)
    }

    /// The package-owned cache directory for sandboxed apps.
    static func packageCacheDirectory(applicationSupport: URL = .applicationSupportDirectory)
        -> URL
    {
        applicationSupport.appending(
            path: "AnySpeechRecognizer/HuggingFace/hub", directoryHint: .isDirectory)
    }
    /// `~/Library/Application Support/AnySpeechRecognizer/Verified`, inside the app container
    /// when the app is sandboxed.
    static var defaultMarkerDirectory: URL {
        URL.applicationSupportDirectory.appending(path: "AnySpeechRecognizer")
            .appending(path: "Verified")
    }

    let cache: HubCache
    let markerDirectory: URL
    /// The largest number of files of one part that download at the same time.
    let maximumConcurrentDownloads: Int
    private let client: HubClient
    private let installations = Mutex<[String: Installation]>([:])

    /// Creates a store without file system or network access.
    /// - Parameters:
    ///   - cache: The Hugging Face cache that holds model files.
    ///   - markerDirectory: The directory for verification markers.
    ///   - session: The session for downloads; tests can add `URLProtocol` stubs.
    ///   - host: The Hub server, or `nil` to use `HF_ENDPOINT` and the environment's token.
    ///   - maximumConcurrentDownloads: The number of files of one part that download
    ///     at the same time.
    init(
        cache: HubCache = defaultCache, markerDirectory: URL = defaultMarkerDirectory,
        session: URLSession = URLSession(configuration: .default), host: URL? = nil,
        maximumConcurrentDownloads: Int = 4
    ) {
        self.cache = cache
        self.markerDirectory = markerDirectory.standardizedFileURL
        self.maximumConcurrentDownloads = max(1, maximumConcurrentDownloads)
        if let host {
            client = HubClient(session: session, host: host, tokenProvider: .none, cache: cache)
        } else {
            client = HubClient(session: session, cache: cache)
        }
    }

    /// The cache snapshot directory for a manifest's revision.
    func directory(for manifest: some PinnedModelManifest) -> URL {
        cache.snapshotsDirectory(repo: Self.repo(manifest), kind: .model)
            .appending(path: manifest.revision, directoryHint: .isDirectory)
    }

    /// The verification marker for one part of a manifest in this store's cache.
    ///
    /// The name includes the repository, the revision, the part, and a digest of the marker's
    /// subject, so manifests that differ in any file of the part, or caches in different
    /// directories, never share a marker.
    /// Each part has its own marker, so the base marker does not change when a manifest
    /// gains or loses a live encoder.
    func markerURL(for manifest: some PinnedModelManifest, part: Part = .base) -> URL {
        let subject = subject(for: manifest, part: part)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let data = (try? encoder.encode(subject)) ?? Data()
        let digest = SHA256.hash(data: data).prefix(8).map { String(format: "%02x", $0) }
        let name = manifest.repository.replacing("/", with: "--")
        return markerDirectory.appending(
            path: "\(name)--\(manifest.revision)--\(part.rawValue)--\(digest.joined()).json")
    }

    /// Checks one part's verification marker, file sizes, and file numbers
    /// without network access or digest computation.
    /// A part that the manifest does not have, such as the live encoder of a manifest
    /// without one, is missing.
    func status(of manifest: some PinnedModelManifest, part: Part = .base) -> Status {
        guard manifest.parts.contains(part), (try? manifest.validate()) != nil,
            let data = try? Data(contentsOf: markerURL(for: manifest, part: part)),
            let marker = try? JSONDecoder().decode(VerificationMarker.self, from: data),
            marker.subject == subject(for: manifest, part: part)
        else { return .missing }
        let directory = directory(for: manifest)
        let intact = manifest.files(in: part).allSatisfy { file in
            guard let info = Self.fileInfo(directory.appending(path: file.path)) else {
                return false
            }
            return info.size == file.size && marker.fileNumbers[file.path] == info.number
        }
        return intact ? .installed : .missing
    }

    /// Checks that a caller-staged directory contains every component by name.
    /// Staged directories have no manifest, so this check does not verify sizes or digests.
    static func status(
        ofStagedDirectory directory: URL, components: ParakeetModelManifest.Components
    ) -> Status {
        components.missingPaths(in: directory).isEmpty ? .installed : .missing
    }

    /// The byte progress of the part's current or most recent install.
    func progress(for manifest: some PinnedModelManifest, part: Part = .base) -> Progress {
        installation(for: manifest, part: part).progress
    }

    /// Creates a progress object for one caller that sums the byte progress of the parts'
    /// installs while the returned object exists.
    ///
    /// Each caller gets its own object because a `Progress` can have only one parent,
    /// and callers such as `AssetInventory` add installation progress to their own.
    func summedProgress(for manifest: some PinnedModelManifest, parts: [Part]) -> SummedProgress {
        SummedProgress(sources: parts.map { progress(for: manifest, part: $0) })
    }

    /// Whether an install of the part is running in this store.
    func isInstalling(_ manifest: some PinnedModelManifest, part: Part = .base) -> Bool {
        installations.withLock { $0[key(for: manifest, part: part)]?.work.isRunning == true }
    }

    /// The number of callers waiting for the part's install.
    func waiterCount(for manifest: some PinnedModelManifest, part: Part = .base) -> Int {
        installations.withLock { $0[key(for: manifest, part: part)]?.work.waiterCount ?? 0 }
    }

    /// Downloads missing files of the parts, verifies them, then records each part as verified.
    ///
    /// Parts install at the same time, and each part that is already installed returns
    /// without network access or digest computation.
    /// Callers that install the same part at the same time share one download.
    /// Cancelling a caller stops a part's download only when no other caller still waits for it.
    /// Files that finished downloading stay in the cache, so a later install skips them;
    /// a file that was still downloading starts over.
    /// - Parameters:
    ///   - manifest: The files to install.
    ///   - parts: The parts to install. Parts that the manifest does not have are ignored.
    /// - Returns: The cache snapshot directory.
    /// - Throws: `ModelAssetError`, a swift-huggingface or `URLError` error,
    ///   a file system error, or `CancellationError`.
    @discardableResult
    func install(_ manifest: some PinnedModelManifest, parts: [Part] = [.base]) async throws -> URL
    {
        try manifest.validate()
        excludePackageCacheFromBackup()
        let requested = manifest.parts.filter { parts.contains($0) }
        try await withThrowingTaskGroup(of: Void.self) { group in
            for part in requested {
                group.addTask { _ = try await self.install(manifest, part: part) }
            }
            try await group.waitForAll()
        }
        return directory(for: manifest)
    }

    /// Keeps models in the package-owned cache out of device backups.
    /// Shared caches that other tools manage are left unchanged.
    private func excludePackageCacheFromBackup() {
        var directory = Self.packageCacheDirectory()
        guard cache.cacheDirectory.standardizedFileURL == directory.standardizedFileURL else {
            return
        }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? directory.setResourceValues(values)
    }

    private func install(_ manifest: some PinnedModelManifest, part: Part) async throws -> URL {
        let installation = installation(for: manifest, part: part)
        return try await installation.work.value { [self] in
            let directory = directory(for: manifest)
            let files = manifest.files(in: part)
            let counter = ByteCounter(
                progress: installation.progress, total: manifest.byteCount(of: part))
            if status(of: manifest, part: part) == .installed {
                for file in files { counter.update(file.path, bytes: file.size) }
                return directory
            }
            let marker = markerURL(for: manifest, part: part)
            try? FileManager.default.removeItem(at: marker)
            // Count files that are already in the cache before any transfer reports.
            for file in files
            where Self.fileInfo(directory.appending(path: file.path))?.size == file.size {
                counter.update(file.path, bytes: file.size)
            }

            let numbers = try await withThrowingTaskGroup(of: (String, UInt64).self) { group in
                var remaining = files.makeIterator()
                var numbers: [String: UInt64] = [:]
                for _ in 0..<maximumConcurrentDownloads {
                    guard let file = remaining.next() else { break }
                    group.addTask { try await self.fetch(file, of: manifest, counter: counter) }
                }
                while let (path, number) = try await group.next() {
                    numbers[path] = number
                    guard let file = remaining.next() else { continue }
                    group.addTask { try await self.fetch(file, of: manifest, counter: counter) }
                }
                return numbers
            }

            try FileManager.default.createDirectory(
                at: markerDirectory, withIntermediateDirectories: true)
            let record = VerificationMarker(
                subject: subject(for: manifest, part: part), fileNumbers: numbers)
            try JSONEncoder().encode(record).write(to: marker, options: .atomic)
            return directory
        }
    }

    /// Downloads one file into the cache unless it is there, then verifies it.
    /// - Returns: The file's path and the file system number of its contents.
    private func fetch(
        _ file: ParakeetModelManifest.File, of manifest: some PinnedModelManifest,
        counter: ByteCounter
    ) async throws -> (String, UInt64) {
        try Task.checkCancellation()
        let progress = Progress(totalUnitCount: file.size)
        let observation = progress.observe(\.completedUnitCount) { progress, _ in
            counter.update(file.path, bytes: min(max(progress.completedUnitCount, 0), file.size))
        }
        defer { observation.invalidate() }
        do {
            _ = try await client.downloadFile(
                at: file.path, from: Self.repo(manifest), revision: manifest.revision,
                progress: progress)
        } catch let error as URLError where error.code == .cancelled && Task.isCancelled {
            throw CancellationError()
        }
        try Task.checkCancellation()
        // Verify the pinned snapshot entry, not the returned path, so that a response for
        // another commit cannot satisfy the manifest.
        let number = try verify(file, of: manifest)
        counter.update(file.path, bytes: file.size)
        return (file.path, number)
    }

    /// Checks a cached file's size and SHA-256 digest, and removes it from the cache
    /// when either does not match, so that a later install downloads it again.
    /// - Returns: The file system number of the file's contents.
    private func verify(
        _ file: ParakeetModelManifest.File, of manifest: some PinnedModelManifest
    ) throws -> UInt64 {
        let url = directory(for: manifest).appending(path: file.path)
        func reject(_ reason: String) -> ModelAssetError {
            discard(url, of: manifest)
            return .verificationFailed(path: file.path, reason: reason)
        }
        guard let info = Self.fileInfo(url) else {
            throw ModelAssetError.verificationFailed(
                path: file.path, reason: "the file is not in the cache at the pinned revision.")
        }
        guard info.size == file.size else {
            throw reject("expected \(file.size) bytes, received \(info.size) bytes.")
        }
        let digest = try Self.sha256(of: url)
        guard digest == file.sha256 else {
            throw reject("expected SHA-256 \(file.sha256), received \(digest).")
        }
        return info.number
    }

    /// Removes a snapshot entry and the blob it links to.
    private func discard(_ url: URL, of manifest: some PinnedModelManifest) {
        let target = url.resolvingSymlinksInPath()
        let blobs = cache.blobsDirectory(repo: Self.repo(manifest), kind: .model)
            .resolvingSymlinksInPath()
        if target.deletingLastPathComponent().path == blobs.path {
            try? FileManager.default.removeItem(at: target)
        }
        try? FileManager.default.removeItem(at: url)
    }

    private func subject(for manifest: some PinnedModelManifest, part: Part)
        -> VerificationMarker.Subject
    {
        .init(
            repository: manifest.repository, revision: manifest.revision,
            contractVersion: manifest.contractVersion, part: part,
            files: manifest.files(in: part),
            snapshot: directory(for: manifest).standardizedFileURL.path)
    }

    private func key(for manifest: some PinnedModelManifest, part: Part) -> String {
        markerURL(for: manifest, part: part).path
    }

    private func installation(for manifest: some PinnedModelManifest, part: Part) -> Installation {
        installations.withLock { installations in
            let key = key(for: manifest, part: part)
            if let existing = installations[key] { return existing }
            let created = Installation(byteCount: manifest.byteCount(of: part))
            installations[key] = created
            return created
        }
    }

    private static func repo(_ manifest: some PinnedModelManifest) -> Repo.ID {
        Repo.ID(rawValue: manifest.repository) ?? Repo.ID(namespace: "", name: manifest.repository)
    }

    /// The size and file system number of a regular file, following symlinks.
    static func fileInfo(_ url: URL) -> (size: Int64, number: UInt64)? {
        var info = stat()
        guard stat(url.path, &info) == 0, info.st_mode & S_IFMT == S_IFREG else { return nil }
        return (Int64(info.st_size), UInt64(info.st_ino))
    }

    /// The lowercase hexadecimal SHA-256 digest of a file's contents.
    static func sha256(of url: URL) throws -> String {
        let reader = try FileHandle(forReadingFrom: url)
        defer { try? reader.close() }
        var hasher = SHA256()
        // Each read returns an autoreleased buffer; without a pool per chunk,
        // hashing the encoder weights keeps about 600 MB resident until the task ends.
        while try autoreleasepool(invoking: {
            try Task.checkCancellation()
            guard let chunk = try reader.read(upToCount: 1 << 20), !chunk.isEmpty else {
                return false
            }
            hasher.update(data: chunk)
            return true
        }) {}
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

extension ModelAssetStore {
    /// Sums bytes across files and reports a total that never decreases.
    fileprivate final class ByteCounter: Sendable {
        private struct State {
            var files: [String: Int64] = [:]
            var current: Int64 = 0
            var reported: Int64
        }
        private let progress: Progress
        private let state: Mutex<State>

        init(progress: Progress, total: Int64) {
            if progress.totalUnitCount != total { progress.totalUnitCount = total }
            self.progress = progress
            self.state = Mutex(State(reported: progress.completedUnitCount))
        }

        func update(_ path: String, bytes: Int64) {
            state.withLock { state in
                state.current += bytes - state.files[path, default: 0]
                state.files[path] = bytes
                guard state.current > state.reported else { return }
                state.reported = state.current
                // Report while locked so that concurrent updates cannot arrive out of order.
                progress.completedUnitCount = state.current
            }
        }
    }

    /// A caller's own progress object that follows the summed byte progress of several
    /// part installs, never moves backward, and follows the parts until it is deallocated.
    final class SummedProgress: Sendable {
        let progress: Progress
        private let sources: [Progress]
        private let reported = Mutex<Int64>(0)
        private let observations = Mutex<[NSKeyValueObservation]>([])

        init(sources: [Progress]) {
            self.sources = sources
            progress = Progress(totalUnitCount: sources.reduce(0) { $0 + $1.totalUnitCount })
            let tokens = sources.map { source in
                source.observe(\.completedUnitCount) { [weak self] _, _ in self?.update() }
            }
            observations.withLock { $0 = tokens }
            update()
        }

        deinit {
            observations.withLock { observations in
                for observation in observations { observation.invalidate() }
            }
        }

        private func update() {
            reported.withLock { reported in
                let current = sources.reduce(0) {
                    $0 + min(max($1.completedUnitCount, 0), $1.totalUnitCount)
                }
                guard current > reported else { return }
                reported = current
                // Report while locked so that concurrent updates cannot arrive out of order.
                progress.completedUnitCount = current
            }
        }
    }
}
