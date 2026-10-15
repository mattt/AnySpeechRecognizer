import AnySpeechRecognizer
import ArgumentParser

struct Status: CancellableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Check hardware, locale, and local assets without downloading.",
        discussion: """
            With --live, also check the assets for volatile results. \
            With --diarize, also check the diarization model.
            """
    )

    @OptionGroup(title: "Model Options") var model: ModelOptions
    @OptionGroup var results: ResultOptions
    @OptionGroup(title: "Diarization Options") var diarizer: DiarizerOptions

    @concurrent func execute() async throws {
        let model = model.model()
        let request = SpeechRecognitionRequest.transcription(
            .init(reportingOptions: results.reportingOptions))
        print(String(describing: await model.availability(for: request)))
        if let diarizer = diarizer.diarizer(live: results.live) {
            print(String(describing: await diarizer.model.availability(for: .init())))
        }
    }
}
