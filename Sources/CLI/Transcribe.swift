import AnySpeechRecognizer
import ArgumentParser
import Foundation

struct Transcribe: CancellableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribe one local file. Assets must already be installed.",
        discussion: "With --diarize, print one line per speaker turn."
    )

    @Argument(help: "A local audio file.")
    var file: String

    @OptionGroup(title: "Model Options") var model: ModelOptions
    @OptionGroup var results: ResultOptions
    @OptionGroup(title: "Diarization Options") var diarizer: DiarizerOptions

    @Flag(help: "Write JSON lines with source and word ranges.")
    var json = false

    @Option(help: ArgumentHelp("Record cold/warm conditions in JSON metadata.", valueName: "label"))
    var cacheState = "unspecified"

    @concurrent func execute() async throws {
        let analysis = try await Analysis.run(
            file, model: model, results: results, diarizer: diarizer, json: json,
            cacheState: cacheState)
        guard !json else { return }
        if let speakers = analysis.speakers {
            // A speaker-attributed transcript, one turn per line.
            for turn in speakers.attributing(analysis.transcript).speakerTurns {
                let words = String(turn.text.characters)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !words.isEmpty else { continue }
                let label = turn.speaker.map(\.description) ?? "Unknown speaker"
                let range =
                    turn.range.map {
                        String(format: "%.2f-%.2f", $0.start.seconds, $0.end.seconds)
                    } ?? "untimed"
                print("[\(range)] \(label): \(words)")
            }
        } else {
            FileHandle.standardOutput.write(Data("\n".utf8))
        }
    }
}
