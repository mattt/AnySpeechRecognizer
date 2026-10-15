import AnySpeechRecognizer
import ArgumentParser
import Darwin
import Foundation

struct Evaluate: CancellableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Write JSON lines with results, runtime metadata, time, and peak RSS.",
        discussion: """
            With --reference, also report the word error rate. \
            With --diarize, also report the speaker count, \
            and with --rttm, the diarization error rate.
            """
    )

    @Argument(help: "A local audio file.")
    var file: String

    @OptionGroup(title: "Model Options") var model: ModelOptions
    @OptionGroup var results: ResultOptions
    @OptionGroup(title: "Diarization Options") var diarizer: DiarizerOptions

    @Option(
        help: ArgumentHelp(
            "A reference transcript for case-sensitive whitespace-token WER.",
            valueName: "text-file"))
    var reference: String?

    @Option(
        help: ArgumentHelp("Record cold/warm conditions; no cache is cleared.", valueName: "label"))
    var cacheState = "unspecified"

    @Option(
        help: ArgumentHelp(
            "Reference speaker segments for the diarization error rate.", valueName: "file"))
    var rttm: String?

    @Option(
        help: ArgumentHelp("Unscored time around reference boundaries.", valueName: "seconds"))
    var collar = 0.25

    func validate() throws {
        guard diarizer.diarize || rttm == nil else {
            throw ValidationError("--rttm requires --diarize.")
        }
        guard collar >= 0 else {
            throw ValidationError("--collar requires a nonnegative number of seconds.")
        }
    }

    @concurrent func execute() async throws {
        let analysis = try await Analysis.run(
            file, model: model, results: results, diarizer: diarizer, json: true,
            cacheState: cacheState)
        let text = String(analysis.transcript.characters)
        let reference = try reference.map { try String(contentsOfFile: $0, encoding: .utf8) }
        let wer = reference.map { wordErrorRate(reference: $0, hypothesis: text) }
        let diarization = try rttm.map { path in
            DiarizationScore(
                reference: try RTTM.read(path), hypothesis: analysis.speakers?.segments ?? [],
                collar: collar)
        }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        try writeJSON(
            EvaluationRecord(
                type: "evaluation", elapsedSeconds: elapsed(since: analysis.started),
                peakResidentBytes: Int64(usage.ru_maxrss), wordErrorRate: wer,
                finalSourceSeconds: analysis.audio.durationSeconds,
                speakerCount: analysis.speakers.map(\.speakers.count), diarization: diarization
            ))
    }

    private func wordErrorRate(reference: String, hypothesis: String) -> Double {
        let expected = reference.split(whereSeparator: \.isWhitespace).map(String.init)
        let actual = hypothesis.split(whereSeparator: \.isWhitespace).map(String.init)
        var previous = Array(0...expected.count)
        for (row, word) in actual.enumerated() {
            var current = [row + 1]
            for (column, other) in expected.enumerated() {
                current.append(
                    min(
                        current[column] + 1, previous[column + 1] + 1,
                        previous[column] + (word == other ? 0 : 1)))
            }
            previous = current
        }
        return Double(previous.last!) / Double(max(1, expected.count))
    }
}

private struct EvaluationRecord: Encodable {
    let type: String
    let elapsedSeconds: Double
    let peakResidentBytes: Int64
    let wordErrorRate: Double?
    let finalSourceSeconds: Double
    let speakerCount: Int?
    let diarization: DiarizationScore?
}
