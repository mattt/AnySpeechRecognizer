import AnySpeechRecognizer
import ArgumentParser
import Foundation

struct Prepare: CancellableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Install assets, then load and validate the model.",
        discussion: """
            With --live, also install the assets for volatile results, \
            such as Parakeet's live encoder. \
            With --diarize, also install the diarization model.
            """
    )

    @OptionGroup(title: "Model Options") var model: ModelOptions
    @OptionGroup var results: ResultOptions
    @OptionGroup(title: "Diarization Options") var diarizer: DiarizerOptions

    @concurrent func execute() async throws {
        let model = model.model()
        let diarizer = diarizer.diarizer(live: results.live)
        // The transcriber's options select the assets, such as Parakeet's live encoder.
        let transcriber = SpeechTranscriber(
            model: model, reportingOptions: results.reportingOptions)
        let modules: [any SpeechModule] = [transcriber] + (diarizer.map { [$0] } ?? [])
        if let request = try await AssetInventory.assetInstallationRequest(supporting: modules) {
            let download = Task { try await request.downloadAndInstall() }
            do {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    group.addTask {
                        try await withTaskCancellationHandler {
                            try await download.value
                        } onCancel: {
                            download.cancel()
                        }
                    }
                    group.addTask {
                        while !Task.isCancelled {
                            let fraction = request.progress.fractionCompleted
                            FileHandle.standardError.write(
                                Data("Asset progress: \(Int(fraction * 100))%\n".utf8))
                            try await Task.sleep(for: .seconds(1))
                        }
                    }
                    _ = try await group.next()
                    group.cancelAll()
                }
            } catch {
                download.cancel()
                _ = await download.result
                throw error
            }
        }
        let analyzer = SpeechAnalyzer(modules: modules)
        do { try await analyzer.prepareToAnalyze(in: nil) } catch {
            await analyzer.cancelAndFinishNow()
            throw error
        }
        await analyzer.cancelAndFinishNow()
        try writeJSON(await model.diagnostics())
        if let diarizer { try writeJSON(await diarizer.model.diagnostics()) }
    }
}
