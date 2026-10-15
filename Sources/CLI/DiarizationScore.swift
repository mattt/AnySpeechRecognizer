import AnySpeechRecognizer
import CoreMedia
import Foundation

/// Reference speaker segments from an RTTM file.
enum RTTM {
    struct Segment {
        let speaker: String
        let start: Double
        let end: Double
    }

    /// Reads `SPEAKER` lines: `SPEAKER <file> <channel> <start> <duration> ... <speaker> ...`.
    static func read(_ path: String) throws -> [Segment] {
        try String(contentsOfFile: path, encoding: .utf8).split(whereSeparator: \.isNewline)
            .compactMap { line in
                let fields = line.split(whereSeparator: \.isWhitespace).map(String.init)
                guard fields.first == "SPEAKER" else { return nil }
                guard fields.count >= 8, let start = Double(fields[3]),
                    let duration = Double(fields[4])
                else { throw CLIError("Malformed RTTM line: \(line)") }
                return Segment(speaker: fields[7], start: start, end: start + duration)
            }
    }
}

/// Frame-based diarization error rate with an optimal one-to-one speaker mapping.
///
/// Time is scored in 10 ms frames. Overlapping speech is scored.
/// A collar removes time within `collar` seconds of each reference segment boundary.
/// Missed speech, false alarms, and speaker confusion are summed over frames,
/// as in NIST md-eval, and divided by the scored reference speaker time.
struct DiarizationScore: Encodable {
    struct Components: Encodable {
        let scoredSpeakerSeconds: Double
        let missedSeconds: Double
        let falseAlarmSeconds: Double
        let confusionSeconds: Double
        let diarizationErrorRate: Double
    }

    let referenceSpeakerCount: Int
    let hypothesisSpeakerCount: Int
    let collarSeconds: Double
    /// The reference speaker of each hypothesis speaker, from the optimal mapping.
    let mapping: [String: String]
    let withCollar: Components
    let withoutCollar: Components

    static let frameSeconds = 0.01

    init(reference: [RTTM.Segment], hypothesis: [SpeakerDiarizer.Segment], collar: Double) {
        let referenceSpeakers = Array(Set(reference.map(\.speaker))).sorted()
        let hypothesisSpeakers = Array(Set(hypothesis.map(\.speaker.id))).sorted()
        let end = max(
            reference.map(\.end).max() ?? 0, hypothesis.map(\.range.end.seconds).max() ?? 0)
        let frames = Int((end / Self.frameSeconds).rounded(.up)) + 1
        func activity(_ intervals: [(Int, Double, Double)], speakers: Int) -> [[Bool]] {
            var active = Array(repeating: [Bool](repeating: false, count: frames), count: speakers)
            for (speaker, start, stop) in intervals {
                let first = max(0, Int((start / Self.frameSeconds).rounded()))
                let last = min(frames, Int((stop / Self.frameSeconds).rounded()))
                for frame in first..<max(first, last) { active[speaker][frame] = true }
            }
            return active
        }
        let referenceIndex = Dictionary(
            uniqueKeysWithValues: referenceSpeakers.enumerated().map { ($1, $0) })
        let hypothesisIndex = Dictionary(
            uniqueKeysWithValues: hypothesisSpeakers.enumerated().map { ($1, $0) })
        let referenceActivity = activity(
            reference.map { (referenceIndex[$0.speaker]!, $0.start, $0.end) },
            speakers: referenceSpeakers.count)
        let hypothesisActivity = activity(
            hypothesis.map {
                (hypothesisIndex[$0.speaker.id]!, $0.range.start.seconds, $0.range.end.seconds)
            }, speakers: hypothesisSpeakers.count)

        // Overlap of each hypothesis and reference speaker, for the optimal mapping.
        var overlap = Array(
            repeating: [Int](repeating: 0, count: referenceSpeakers.count),
            count: hypothesisSpeakers.count)
        for h in hypothesisSpeakers.indices {
            for r in referenceSpeakers.indices {
                for frame in 0..<frames
                where hypothesisActivity[h][frame] && referenceActivity[r][frame] {
                    overlap[h][r] += 1
                }
            }
        }
        let assignment = Self.bestAssignment(overlap, references: referenceSpeakers.count)

        var unscored = [Bool](repeating: false, count: frames)
        for segment in reference {
            for boundary in [segment.start, segment.end] {
                let first = max(0, Int(((boundary - collar) / Self.frameSeconds).rounded()))
                let last = min(frames, Int(((boundary + collar) / Self.frameSeconds).rounded()))
                for frame in first..<max(first, last) { unscored[frame] = true }
            }
        }
        func score(skipping skipped: [Bool]) -> Components {
            var total = 0
            var missed = 0
            var falseAlarm = 0
            var confusion = 0
            for frame in 0..<frames where !skipped[frame] {
                let references = referenceSpeakers.indices.filter { referenceActivity[$0][frame] }
                let hypotheses = hypothesisSpeakers.indices.filter { hypothesisActivity[$0][frame] }
                let correct = hypotheses.filter { h in
                    assignment[h].map { references.contains($0) } ?? false
                }.count
                total += references.count
                missed += max(0, references.count - hypotheses.count)
                falseAlarm += max(0, hypotheses.count - references.count)
                confusion += min(references.count, hypotheses.count) - correct
            }
            let seconds = { (frames: Int) -> Double in Double(frames) * Self.frameSeconds }
            return Components(
                scoredSpeakerSeconds: seconds(total), missedSeconds: seconds(missed),
                falseAlarmSeconds: seconds(falseAlarm), confusionSeconds: seconds(confusion),
                diarizationErrorRate: Double(missed + falseAlarm + confusion)
                    / Double(max(total, 1)))
        }
        referenceSpeakerCount = referenceSpeakers.count
        hypothesisSpeakerCount = hypothesisSpeakers.count
        collarSeconds = collar
        withCollar = score(skipping: unscored)
        withoutCollar = score(skipping: [Bool](repeating: false, count: frames))
        var mapping: [String: String] = [:]
        for (h, r) in assignment.enumerated() {
            if let r { mapping["\(hypothesisSpeakers[h])"] = referenceSpeakers[r] }
        }
        self.mapping = mapping
    }

    /// Maps each hypothesis speaker to at most one reference speaker, maximizing overlap.
    /// The search is exhaustive, which suits the few speakers of a diarization session.
    private static func bestAssignment(_ overlap: [[Int]], references: Int) -> [Int?] {
        var best: (score: Int, assignment: [Int?]) = (-1, [])
        var current: [Int?] = []
        var used = Set<Int>()
        func search(_ h: Int, _ score: Int) {
            if h == overlap.count {
                if score > best.score { best = (score, current) }
                return
            }
            current.append(nil)
            search(h + 1, score)
            current.removeLast()
            for r in 0..<references where !used.contains(r) {
                used.insert(r)
                current.append(r)
                search(h + 1, score + overlap[h][r])
                current.removeLast()
                used.remove(r)
            }
        }
        search(0, 0)
        return best.assignment
    }
}
