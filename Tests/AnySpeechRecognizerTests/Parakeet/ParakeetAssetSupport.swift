import CryptoKit
import Foundation
import HuggingFace
import Synchronization

@testable import AnySpeechRecognizer

/// An in-memory Hugging Face host reached through `URLProtocol`, one unique host name per server.
/// Like the Hub, it answers `HEAD` with the commit and ETag headers that the cache uses,
/// and serves every file as an LFS file whose ETag is its SHA-256 digest.
final class StubModelServer: Sendable {
    struct Request: Sendable, Equatable {
        let method: String
        let path: String
        let range: String?
    }

    private struct State {
        var files: [String: Data]
        var honorsRange = true
        /// Sends this many body bytes of a file and then waits until the request is cancelled.
        var stalls: [String: Int] = [:]
        /// Holds responses until `release()`.
        var holding = false
        var held: [@Sendable () -> Void] = []
        var requests: [Request] = []
    }

    private static let servers = Mutex<[String: StubModelServer]>([:])
    let host = "stub-\(UUID().uuidString.lowercased()).test"
    private let state: Mutex<State>

    init(files: [String: Data]) {
        state = Mutex(State(files: files))
        Self.servers.withLock { $0[host] = self }
    }

    static func server(for host: String?) -> StubModelServer? {
        servers.withLock { servers in host.flatMap { servers[$0] } }
    }

    var url: URL { URL(string: "https://\(host)")! }
    var requests: [Request] { state.withLock { $0.requests } }

    /// Creates a store that downloads from this server into a cache under `root`.
    func makeStore(root: URL, maximumConcurrentDownloads: Int = 2) -> ModelAssetStore {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubModelProtocol.self]
        return ModelAssetStore(
            cache: HubCache(cacheDirectory: root.appending(path: "hub")),
            markerDirectory: root.appending(path: "verified"),
            session: URLSession(configuration: configuration), host: url,
            maximumConcurrentDownloads: maximumConcurrentDownloads)
    }

    func setFile(_ path: String, _ data: Data) { state.withLock { $0.files[path] = data } }
    func setHonorsRange(_ value: Bool) { state.withLock { $0.honorsRange = value } }
    func setStall(_ path: String, after bytes: Int?) { state.withLock { $0.stalls[path] = bytes } }
    func hold() { state.withLock { $0.holding = true } }

    func release() {
        let held = state.withLock { state in
            state.holding = false
            defer { state.held.removeAll() }
            return state.held
        }
        for respond in held { respond() }
    }

    /// The number of `GET` requests for a file.
    func requestCount(for path: String) -> Int {
        requests.count { $0.method == "GET" && $0.path == path }
    }

    /// The `GET` requests, in order.
    var downloads: [Request] { requests.filter { $0.method == "GET" } }

    fileprivate func handle(_ request: URLRequest, with stub: StubModelProtocol) {
        let method = request.httpMethod ?? "GET"
        let path = Self.filePath(of: request.url)
        let revision = Self.revision(of: request.url)
        let range = request.value(forHTTPHeaderField: "Range")
        let respond: @Sendable () -> Void = { [self] in
            respond(method: method, path: path, revision: revision, range: range, stub)
        }
        let deferred = state.withLock { state in
            state.requests.append(Request(method: method, path: path, range: range))
            if state.holding { state.held.append(respond) }
            return state.holding
        }
        if !deferred { respond() }
    }

    private func respond(
        method: String, path: String, revision: String, range: String?, _ stub: StubModelProtocol
    ) {
        let (data, honorsRange, stallAfter) = state.withLock {
            ($0.files[path], $0.honorsRange, $0.stalls[path])
        }
        guard let data else {
            stub.send(status: 404, headers: [:], body: Data(), finish: true)
            return
        }
        var status = 200
        var body = data
        var headers = [
            "Content-Type": "application/octet-stream",
            "X-Repo-Commit": revision,
            "X-Linked-Etag": "\"\(ModelFixture.sha256(data))\"",
            "X-Linked-Size": "\(data.count)",
        ]
        if method == "HEAD" {
            stub.send(status: 200, headers: headers, body: Data(), finish: true)
            return
        }
        if honorsRange, let range, let start = Self.rangeStart(range), start < data.count {
            status = 206
            body = data.subdata(in: start..<data.count)
            headers["Content-Range"] = "bytes \(start)-\(data.count - 1)/\(data.count)"
        }
        if let stallAfter {
            stub.send(
                status: status, headers: headers, body: body.prefix(stallAfter), finish: false)
        } else {
            stub.send(status: status, headers: headers, body: body, finish: true)
        }
    }

    private static func rangeStart(_ header: String) -> Int? {
        guard header.hasPrefix("bytes="), header.hasSuffix("-") else { return nil }
        return Int(header.dropFirst("bytes=".count).dropLast())
    }

    /// Extracts `<path>` from `/<owner>/<name>/resolve/<revision>/<path>`.
    private static func filePath(of url: URL?) -> String {
        let components = url?.pathComponents.filter { $0 != "/" } ?? []
        guard let index = components.firstIndex(of: "resolve"), components.count > index + 2
        else { return "" }
        return components[(index + 2)...].joined(separator: "/")
    }

    /// Extracts `<revision>` from `/<owner>/<name>/resolve/<revision>/<path>`.
    private static func revision(of url: URL?) -> String {
        let components = url?.pathComponents.filter { $0 != "/" } ?? []
        guard let index = components.firstIndex(of: "resolve"), components.count > index + 1
        else { return "" }
        return components[index + 1]
    }
}

final class StubModelProtocol: URLProtocol, @unchecked Sendable {
    private let stopped = Mutex(false)

    override class func canInit(with request: URLRequest) -> Bool {
        StubModelServer.server(for: request.url?.host()) != nil
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let server = StubModelServer.server(for: request.url?.host()) else {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotFindHost))
            return
        }
        server.handle(request, with: self)
    }

    override func stopLoading() { stopped.withLock { $0 = true } }

    fileprivate func send(status: Int, headers: [String: String], body: Data, finish: Bool) {
        guard stopped.withLock({ !$0 }), let url = request.url,
            let response = HTTPURLResponse(
                url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)
        else { return }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        // Several chunks exercise incremental hashing and progress.
        var offset = body.startIndex
        while offset < body.endIndex {
            let end = min(offset + 4096, body.endIndex)
            client?.urlProtocol(self, didLoad: body.subdata(in: offset..<end))
            offset = end
        }
        if finish { client?.urlProtocolDidFinishLoading(self) }
    }
}

/// Records every `completedUnitCount` value that a progress object reports.
final class ProgressRecorder: Sendable {
    private let values = Mutex<[Int64]>([])
    private let observation = Mutex<NSKeyValueObservation?>(nil)

    init(_ progress: Progress) {
        let token = progress.observe(\.completedUnitCount, options: [.initial, .new]) {
            [weak self] progress, _ in
            let value = progress.completedUnitCount
            self?.values.withLock { $0.append(value) }
        }
        observation.withLock { $0 = token }
    }

    var recorded: [Int64] { values.withLock { $0 } }

    func invalidate() { observation.withLock { $0?.invalidate() } }
}

/// Runs a test body with a model root directory that is removed afterward.
func withTemporaryModelRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
    let url = FileManager.default.temporaryDirectory.appending(
        path: "anyspeech-models-\(UUID().uuidString)", directoryHint: .isDirectory)
    defer { try? FileManager.default.removeItem(at: url) }
    return try await body(url)
}

enum ModelFixture {
    static let encoderWeights = "Encoder.mlmodelc/weights/weight.bin"
    static let encoderProgram = "Encoder.mlmodelc/model.mil"
    static let decoderProgram = "Decoder.mlmodelc/model.mil"
    static let jointProgram = "Joint.mlmodelc/model.mil"
    static let vocabulary = "vocab.json"

    static func bytes(_ count: Int, seed: UInt8) -> Data {
        Data((0..<count).map { UInt8(truncatingIfNeeded: $0 &* 31 &+ Int(seed)) })
    }

    static let files: [String: Data] = [
        encoderWeights: bytes(96 * 1024, seed: 1),
        encoderProgram: bytes(5_000, seed: 2),
        decoderProgram: bytes(12_345, seed: 3),
        jointProgram: bytes(4_096, seed: 4),
        vocabulary: Data(#"{"0": "<unk>", "1": "▁hello"}"#.utf8),
    ]

    static let liveEncoder = "Encoder8s.mlmodelc"
    static let liveEncoderWeights = "Encoder8s.mlmodelc/weights/weight.bin"
    static let liveEncoderProgram = "Encoder8s.mlmodelc/model.mil"

    /// The live encoder's files, which form the manifest's live encoder part.
    static let liveFiles: [String: Data] = [
        liveEncoderWeights: bytes(64 * 1024, seed: 5),
        liveEncoderProgram: bytes(4_000, seed: 6),
    ]

    /// The base files and the live encoder's files.
    static var filesWithLiveEncoder: [String: Data] { files.merging(liveFiles) { $1 } }

    /// A manifest with the base files and the optional live encoder.
    static func manifestWithLiveEncoder() -> ParakeetModelManifest {
        manifest(filesWithLiveEncoder, liveEncoder: liveEncoder)
    }

    static func manifest(
        _ files: [String: Data] = files,
        revision: String = "0123456789abcdef0123456789abcdef01234567",
        liveEncoder: String? = nil
    ) -> ParakeetModelManifest {
        ParakeetModelManifest(
            identifier: "test",
            repository: "Example/parakeet-test",
            revision: revision,
            contractVersion: 1,
            components: .init(
                encoder: "Encoder.mlmodelc", decoder: "Decoder.mlmodelc",
                joint: "Joint.mlmodelc", vocabulary: vocabulary, liveEncoder: liveEncoder),
            files: files.keys.sorted().map { path in
                .init(path: path, size: Int64(files[path]!.count), sha256: sha256(files[path]!))
            },
            license: "CC-BY-4.0",
            attribution: "Test fixture."
        )
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
