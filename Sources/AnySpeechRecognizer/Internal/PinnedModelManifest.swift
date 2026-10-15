import Foundation

/// Pinned model files that `ModelAssetStore` installs, verifies, and locates.
///
/// Parakeet and Sortformer manifests conform, so both share the store's
/// verified installs, shared downloads, and progress.
protocol PinnedModelManifest: Sendable {
    /// A stable name for logs and diagnostics.
    var identifier: String { get }
    /// The Hugging Face repository, such as `owner/name`.
    var repository: String { get }
    /// The immutable 40-character commit hash that pins every file.
    var revision: String { get }
    /// The version of the input and output contract that the runtime expects.
    var contractVersion: Int { get }
    /// The parts that have files.
    var parts: [ParakeetModelManifest.Part] { get }
    /// The files of one part, in manifest order.
    func files(in part: ParakeetModelManifest.Part) -> [ParakeetModelManifest.File]
    /// Checks the revision, paths, sizes, and digests before any file system or network use.
    func validate() throws
}

extension PinnedModelManifest {
    /// The sum of the sizes of one part's files in bytes.
    func byteCount(of part: ParakeetModelManifest.Part) -> Int64 {
        files(in: part).reduce(0) { $0 + $1.size }
    }
}

extension ParakeetModelManifest: PinnedModelManifest {}

extension ParakeetModelManifest {
    /// Checks a repository, revision, and file list, and that files provide each component.
    /// - Throws: `ModelAssetError.invalidManifest` with the first problem found.
    static func validatePinnedFiles(
        identifier: String, repository: String, revision: String, files: [File],
        components: [String]
    ) throws {
        func fail(_ reason: String) -> ModelAssetError {
            .invalidManifest("The \(identifier) manifest is invalid: \(reason)")
        }
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy(isSafeComponent) else {
            throw fail("the repository \"\(repository)\" is not an owner and name.")
        }
        guard revision.count == 40, revision.allSatisfy(\.isLowercaseHexDigit) else {
            throw fail("the revision \"\(revision)\" is not a 40-character commit hash.")
        }
        guard !files.isEmpty else { throw fail("it lists no files.") }
        var seen: Set<String> = []
        for file in files {
            guard isSafeRelativePath(file.path) else {
                throw fail("the path \"\(file.path)\" is not a safe relative path.")
            }
            guard seen.insert(file.path).inserted else {
                throw fail("the path \"\(file.path)\" appears more than once.")
            }
            guard file.size >= 0 else { throw fail("\(file.path) has a negative size.") }
            guard file.sha256.count == 64, file.sha256.allSatisfy(\.isLowercaseHexDigit) else {
                throw fail("\(file.path) does not have a SHA-256 digest.")
            }
        }
        for component in components {
            guard
                seen.contains(component) || seen.contains(where: { $0.hasPrefix(component + "/") })
            else { throw fail("no file provides the component \(component).") }
        }
    }
}
