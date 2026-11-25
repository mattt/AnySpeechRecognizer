import AnySpeechRecognizer
import CoreMedia
import Foundation
import Testing

@testable import hear

struct DiarizationScoreTests {
    @Test func rejectsMultipleRecordingIDs() throws {
        let contents = """
            SPEAKER first 1 0 1 <NA> <NA> A <NA> <NA>
            SPEAKER second 1 0 1 <NA> <NA> A <NA> <NA>
            """
        let url = try writeReference(contents)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: CLIError.self) { try RTTM.read(url.path) }
    }

    @Test func acceptsOneRecordingIDRegardlessOfFilename() throws {
        let contents = """
            SPEAKER recording 1 0 1 <NA> <NA> A <NA> <NA>
            SPEAKER recording 1 1 1 <NA> <NA> B <NA> <NA>
            """
        let url = try writeReference(contents)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(try RTTM.read(url.path).map(\.speaker) == ["A", "B"])
    }

    private func writeReference(_ contents: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(UUID().uuidString).rttm")
        try contents.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    @Test func collarChangesTheOptimalSpeakerAssignment() {
        // Speaker 0 covers all of A and B. Only B has frames outside the collar.
        let reference: [RTTM.Segment] = [
            .init(speaker: "A", start: 0, end: 1),
            .init(speaker: "A", start: 1, end: 2),
            .init(speaker: "A", start: 2, end: 3),
            .init(speaker: "B", start: 3, end: 5),
        ]
        let hypothesis = SpeakerDiarizer.Segment(
            speaker: .init(id: 0),
            range: CMTimeRange(start: .zero, duration: CMTime(value: 5, timescale: 1))
        )
        let score = DiarizationScore(reference: reference, hypothesis: [hypothesis], collar: 0.5)
        #expect(score.withCollar.scoredSpeakerSeconds == 1)
        #expect(score.withCollar.confusionSeconds == 0)
        #expect(score.withCollar.diarizationErrorRate == 0)
        #expect(score.withoutCollar.scoredSpeakerSeconds == 5)
        #expect(score.withoutCollar.confusionSeconds == 2)
        #expect(score.withoutCollar.diarizationErrorRate == 0.4)
        #expect(score.mapping == ["0": "A"])
    }
}
