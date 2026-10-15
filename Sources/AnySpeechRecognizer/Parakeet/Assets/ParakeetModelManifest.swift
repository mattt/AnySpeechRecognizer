import Foundation

/// One pinned set of Parakeet model files and the facts needed to verify and credit them.
/// Built-in manifests are Swift literals that `Tools/make-parakeet-manifest.py` generates.
struct ParakeetModelManifest: Sendable, Hashable, Codable {
    /// A file at a path relative to the repository root and the install directory.
    struct File: Sendable, Hashable, Codable {
        let path: String
        let size: Int64
        /// The lowercase hexadecimal SHA-256 digest of the file's contents.
        let sha256: String
    }

    /// The relative locations of the files and compiled model directories the runtime loads.
    struct Components: Sendable, Hashable, Codable {
        let encoder: String
        let decoder: String
        let joint: String
        let vocabulary: String
        /// An optional encoder with a shorter fixed window, such as `Encoder8s.mlmodelc`,
        /// that live and fast results use instead of `encoder` when it is installed.
        /// Its files form the manifest's ``Part/liveEncoder`` part.
        let liveEncoder: String?

        init(
            encoder: String, decoder: String, joint: String, vocabulary: String,
            liveEncoder: String? = nil
        ) {
            self.encoder = encoder
            self.decoder = decoder
            self.joint = joint
            self.vocabulary = vocabulary
            self.liveEncoder = liveEncoder
        }

        /// The components that every model directory needs.
        var requiredPaths: [String] { [encoder, decoder, joint, vocabulary] }
        /// Every component that the manifest lists, including the optional live encoder.
        var paths: [String] { requiredPaths + (liveEncoder.map { [$0] } ?? []) }

        /// Returns the required component paths that do not exist in a directory.
        /// The live encoder is optional, so its absence is not reported.
        func missingPaths(in directory: URL) -> [String] {
            requiredPaths.filter {
                !FileManager.default.fileExists(atPath: directory.appending(path: $0).path)
            }
        }
    }

    /// A group of manifest files that installs, verifies, and records its verification
    /// as a unit, so that a configuration installs only the parts it needs.
    enum Part: String, Sendable, Hashable, Codable, CaseIterable {
        /// The encoder, decoder, joint, and vocabulary, which every configuration needs:
        /// every file outside the live encoder.
        case base
        /// The files under the optional live encoder, which only volatile and fast
        /// results use.
        case liveEncoder
    }

    /// A stable name for logs and diagnostics, such as `parakeet-tdt-0.6b-v3`.
    let identifier: String
    /// The Hugging Face repository, such as `FluidInference/parakeet-ultra-coreml`.
    let repository: String
    /// The immutable 40-character commit hash that pins every file.
    let revision: String
    /// The version of the input and output contract that the runtime expects.
    let contractVersion: Int
    let components: Components
    let files: [File]
    /// The SPDX identifier of the model license.
    let license: String
    /// The notice that applications show to credit the model's authors.
    let attribution: String

    /// The sum of every file's size in bytes.
    var totalByteCount: Int64 { files.reduce(0) { $0 + $1.size } }

    /// The parts that have files: the base, and the live encoder when the manifest names one.
    var parts: [Part] { components.liveEncoder == nil ? [.base] : [.base, .liveEncoder] }

    /// The files of one part, in manifest order.
    func files(in part: Part) -> [File] {
        files.filter { isInLiveEncoder($0.path) == (part == .liveEncoder) }
    }

    /// The sum of the sizes of one part's files in bytes.
    func byteCount(of part: Part) -> Int64 { files(in: part).reduce(0) { $0 + $1.size } }

    private func isInLiveEncoder(_ path: String) -> Bool {
        guard let liveEncoder = components.liveEncoder else { return false }
        return path == liveEncoder || path.hasPrefix(liveEncoder + "/")
    }

    /// Checks the revision, paths, sizes, and digests before any file system or network use.
    /// - Throws: `ModelAssetError.invalidManifest` with the first problem found.
    func validate() throws {
        func fail(_ reason: String) -> ModelAssetError {
            .invalidManifest("The \(identifier) manifest is invalid: \(reason)")
        }
        try Self.validatePinnedFiles(
            identifier: identifier, repository: repository, revision: revision, files: files,
            components: components.paths)
        // The base part must not depend on files of the optional live encoder.
        if let liveEncoder = components.liveEncoder {
            for component in components.requiredPaths
            where isInLiveEncoder(component) || liveEncoder.hasPrefix(component + "/") {
                throw fail("the live encoder \(liveEncoder) overlaps the component \(component).")
            }
        }
    }

    static func isSafeComponent(_ component: Substring) -> Bool {
        !component.isEmpty && component != "." && component != ".."
            && !component.contains("\\") && !component.contains(":")
    }

    static func isSafeRelativePath(_ path: String) -> Bool {
        !path.hasPrefix("/")
            && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy(
                isSafeComponent)
    }
}

extension Character {
    var isLowercaseHexDigit: Bool {
        ("0"..."9").contains(self) || ("a"..."f").contains(self)
    }
}
