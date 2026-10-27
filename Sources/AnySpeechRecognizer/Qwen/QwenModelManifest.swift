import Foundation

/// The pinned files of one published Qwen3-ASR conversion and the facts needed to verify
/// and credit them.
/// The built-in manifests are Swift literals that `Tools/make-qwen-manifest.py` generates.
struct QwenModelManifest: PinnedModelManifest, Hashable, Codable {
    typealias File = ParakeetModelManifest.File
    typealias Part = ParakeetModelManifest.Part

    let identifier: String
    let repository: String
    let revision: String
    let contractVersion: Int
    let files: [File]
    /// The SPDX identifier of the model license.
    let license: String
    /// The notice that applications show to credit the model's authors.
    let attribution: String

    /// The files and compiled model directories that the runtime loads,
    /// at the repository root as `Tools/QwenConversion/package.py` lays them out.
    static let components = [
        CoreMLQwenEngine.encoderName, CoreMLQwenEngine.decoderName,
        CoreMLQwenEngine.vocabularyName, CoreMLQwenEngine.configurationName,
    ]

    /// A single base part: every file is required.
    var parts: [Part] { [.base] }

    func files(in part: Part) -> [File] { part == .base ? files : [] }

    func validate() throws {
        try ParakeetModelManifest.validatePinnedFiles(
            identifier: identifier, repository: repository, revision: revision, files: files,
            components: Self.components)
    }
}
