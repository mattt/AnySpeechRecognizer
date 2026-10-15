import CoreMedia
import Foundation
import Speech
import Testing

@testable import AnySpeechRecognizer

private typealias TimeRangeAttribute = AttributeScopes.SpeechAttributes.TimeRangeAttribute
private typealias SpeakerAttribute = AttributeScopes.AnySpeechRecognizerAttributes.SpeakerAttribute

/// Builds transcript text with one timed run per word, separated by untimed spaces.
private func transcript(_ words: [(String, Double, Double)]) -> AttributedString {
    var text = AttributedString()
    for (index, (word, start, end)) in words.enumerated() {
        if index > 0 { text += AttributedString(" ") }
        var run = AttributedString(word)
        run[TimeRangeAttribute.self] = range(start, end)
        text += run
    }
    return text
}

private func speakers(_ text: AttributedString) -> [String: Int?] {
    var values: [String: Int?] = [:]
    for run in text.runs where run[TimeRangeAttribute.self] != nil {
        let speaker: Int? = run[SpeakerAttribute.self]?.id
        values[String(text[run.range].characters)] = .some(speaker)
    }
    return values
}

@Test func wordsTakeTheSpeakerWithTheLongestOverlap() {
    let timeline = SpeakerTimeline(segments: [segment(1, 2, 4), segment(0, 0, 2)])
    let text = transcript([("Hello", 0.5, 1.0), ("there", 1.8, 2.6), ("friend", 3.0, 3.5)])
    let attributed = timeline.attributing(text)
    // "there" straddles the change: 0.2 s with speaker 0 and 0.6 s with speaker 1.
    #expect(speakers(attributed) == ["Hello": 0, "there": 1, "friend": 1])
    #expect(String(attributed.characters) == String(text.characters))
    #expect(attributed.runs.contains { $0[TimeRangeAttribute.self] == range(1.8, 2.6) })
}

@Test func overlappingSpeechPrefersTheLongerOverlapThenTheEarlierSpeaker() {
    let timeline = SpeakerTimeline(segments: [segment(0, 0, 3), segment(1, 1, 2)])
    let text = transcript([("both", 1.2, 1.8), ("mostly", 1.5, 2.5), ("later", 2.2, 2.8)])
    #expect(
        speakers(timeline.attributing(text)) == ["both": 0, "mostly": 0, "later": 0])
    let reversed = SpeakerTimeline(segments: [segment(1, 0, 3), segment(0, 1, 2)])
    #expect(speakers(reversed.attributing(transcript([("both", 1.2, 1.8)]))) == ["both": 1])
}

@Test func wordsWithoutOverlapTakeTheNearestSpeakerWithinTheGap() {
    let timeline = SpeakerTimeline(segments: [segment(0, 0, 1), segment(1, 2, 4)])
    let text = transcript([("near", 4.2, 4.4), ("far", 6.0, 6.5), ("between", 1.3, 1.4)])
    #expect(speakers(timeline.attributing(text)) == ["near": 1, "far": nil, "between": 0])
    #expect(
        speakers(timeline.attributing(text, maximumGap: .zero))
            == ["near": nil, "far": nil, "between": nil])
    // Untimed runs, such as spaces, get no speaker.
    let attributed = timeline.attributing(text)
    #expect(
        attributed.runs.allSatisfy { run in
            run[TimeRangeAttribute.self] != nil || run[SpeakerAttribute.self] == nil
        })
}

@Test func speakerTurnsGroupConsecutiveWords() throws {
    let timeline = SpeakerTimeline(segments: [segment(0, 0, 2), segment(1, 2, 4)])
    let text = transcript([
        ("Good", 0.1, 0.4), ("morning.", 0.5, 1.2), ("Hi", 2.1, 2.4), ("there.", 2.5, 3.0),
    ])
    let turns = timeline.attributing(text).speakerTurns
    #expect(turns.map(\.speaker?.id) == [0, 1])
    #expect(turns.map { String($0.text.characters) } == ["Good morning. ", "Hi there."])
    #expect(turns.map(\.range) == [range(0.1, 1.2), range(2.1, 3.0)])
    // Turn text keeps the original attributes and gains the speaker.
    let first = try #require(turns.first)
    #expect(first.text.runs.contains { $0[TimeRangeAttribute.self] == range(0.1, 0.4) })
    #expect(first.text.runs.contains { $0.speaker?.id == 0 })
}

@Test func timelineJoinsSplitSegmentsAndReplacesVolatileResults() {
    var timeline = SpeakerTimeline()
    timeline.apply(diarization([segment(0, 0, 1.5)], range: range(0, 1.5), final: 0))
    #expect(timeline.volatileSegments == [segment(0, 0, 1.5)])
    timeline.apply(diarization([segment(0, 0, 1)], final: 1))
    // A final result clips volatile segments at its finalization time.
    #expect(timeline.volatileSegments == [segment(0, 1, 1.5)])
    timeline.apply(
        diarization([segment(0, 1, 2), segment(1, 1.5, 2.5)], range: range(1, 2.5), final: 1))
    #expect(timeline.volatileSegments == [segment(0, 1, 2), segment(1, 1.5, 2.5)])
    timeline.apply(diarization([segment(0, 1, 2), segment(1, 1.5, 2)], final: 2))
    timeline.apply(diarization([segment(1, 2, 3)], final: 3))
    #expect(timeline.segments == [segment(0, 0, 2), segment(1, 1.5, 3)])
    #expect(timeline.volatileSegments.isEmpty)
    #expect(timeline.speakers == [.init(id: 0), .init(id: 1)])
    #expect(timeline.finalizationTime == seconds(3))
    #expect(timeline.speaker(for: range(2.2, 2.4)) == .init(id: 1))
}
