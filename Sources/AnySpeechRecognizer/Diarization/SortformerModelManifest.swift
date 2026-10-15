import Foundation

/// The pinned files of one compiled Sortformer model and the facts needed to verify
/// and credit them.
/// The built-in manifests are Swift literals that `Tools/make-sortformer-manifest.py` generates.
struct SortformerModelManifest: PinnedModelManifest, Hashable, Codable {
    typealias File = ParakeetModelManifest.File
    typealias Part = ParakeetModelManifest.Part

    let identifier: String
    let repository: String
    let revision: String
    let contractVersion: Int
    /// The compiled model directory, relative to the repository root.
    let model: String
    let files: [File]
    /// The SPDX identifier of the model license,
    /// or a `LicenseRef-` identifier for a license without one.
    let license: String
    /// The notice that applications show to credit the model's authors.
    let attribution: String

    /// A single base part: every file is required.
    var parts: [Part] { [.base] }

    func files(in part: Part) -> [File] { part == .base ? files : [] }

    func validate() throws {
        try ParakeetModelManifest.validatePinnedFiles(
            identifier: identifier, repository: repository, revision: revision, files: files,
            components: [model])
    }
}
