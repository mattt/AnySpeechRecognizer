import CoreMedia
import Foundation
import Speech
import Testing

@testable import AnySpeechRecognizer

private let pieces = ["▁Hello", ",", "▁world", ".", "▁Again", "▁and", "▁again", "!", "▁ex", "tra"]
private let vocabulary = ParakeetVocabulary(pieces: pieces)

/// One token per 80 ms frame, each lasting one frame unless stated.
private func token(_ id: Int, frame: Int, frames: Int = 1, probability: Float = 0.9)
    -> ParakeetToken
{
    ParakeetToken(
        id: id, piece: pieces[id], startSample: frame * 1280,
        endSample: (frame + frames) * 1280, probability: probability)
}

private func builder(origin: CMTime = .zero, received: Int = 1_000_000) -> ParakeetResultBuilder {
    var builder = ParakeetResultBuilder(
        vocabulary: vocabulary, origin: origin, includeTiming: true, includeConfidence: true)
    builder.receivedSamples = received
    return builder
}

private func text(_ results: [SpeechRecognitionResult]) -> String {
    results.map { String($0.text.characters) }.joined()
}

@Test func phrasesEndAtSentencePunctuationAndHoldTheLastWord() throws {
    var builder = builder()
    let finals = builder.commit(
        [
            token(0, frame: 0), token(1, frame: 1), token(2, frame: 2), token(3, frame: 3),
            token(4, frame: 5),
        ],
        flush: false)
    #expect(text(finals) == "Hello, world.")
    #expect(finals.allSatisfy { $0.isFinal })
    #expect(builder.hasPending)
    let rest = builder.commit([token(8, frame: 7), token(9, frame: 8)], flush: true)
    #expect(text(finals + rest) == "Hello, world. Again extra")
    for result in finals + rest { try validateTiming(in: result.text, range: result.range) }
    #expect(rest.first!.range.start >= finals.last!.resultsFinalizationTime)
}

@Test func laterTokensCanContinueAHeldWord() {
    var builder = builder()
    let first = builder.commit([token(4, frame: 0), token(8, frame: 3)], flush: false)
    #expect(text(first) == "")
    let second = builder.commit([token(9, frame: 4)], flush: true)
    #expect(text(second) == "Again extra")
}

@Test func pausesEndPhrases() {
    var builder = builder()
    let finals = builder.commit(
        [token(4, frame: 0), token(5, frame: 30), token(6, frame: 31)], flush: false)
    #expect(text(finals) == "Again")
}

@Test func overlappingWordsJoinThePhrase() {
    var builder = builder()
    // "world." ends at frame 6, after "Again" starts at frame 5.
    let finals = builder.commit(
        [
            token(2, frame: 0), token(3, frame: 1, frames: 5), token(4, frame: 5),
            token(5, frame: 20), token(6, frame: 50),
        ],
        flush: false)
    #expect(text(finals) == "world. Again and")
}

@Test func longPhrasesSplitAfterThirtyTwoWords() {
    var builder = builder()
    let tokens = (0..<70).map { token(5, frame: $0) }
    let finals = builder.commit(tokens, flush: true)
    #expect(finals.count == 3)
    #expect(finals[0].text.characters.split(separator: " ").count == 32)
}

@Test func volatileResultCoversPendingAndUncommittedWords() throws {
    var builder = builder()
    _ = builder.commit([token(0, frame: 0), token(1, frame: 1), token(4, frame: 3)], flush: false)
    let volatile = try #require(builder.volatileResult([token(5, frame: 5), token(6, frame: 6)]))
    #expect(!volatile.isFinal)
    #expect(String(volatile.text.characters) == "Hello, Again and again")
    #expect(volatile.resultsFinalizationTime == .zero)
}

@Test func wordEndsAreClippedToReceivedAudio() throws {
    var builder = builder(received: 1_500)
    let finals = builder.commit([token(4, frame: 0, frames: 4)], flush: true)
    #expect(finals.first?.range.end == CMTime(value: 1_500, timescale: 16_000))
}

@Test func confidenceIsTheGeometricMeanOfTokenProbabilities() throws {
    var builder = builder()
    let finals = builder.commit(
        [token(8, frame: 0, probability: 0.25), token(9, frame: 1, probability: 1)], flush: true)
    let run = try #require(finals.first?.text.runs.first)
    let confidence = try #require(run[AttributeScopes.SpeechAttributes.ConfidenceAttribute.self])
    #expect(abs(confidence - 0.5) < 1e-9)
}

@Test func restartKeepsTheSourceTimeline() {
    var builder = builder(origin: CMTime(value: 2, timescale: 1))
    let first = builder.commit([token(4, frame: 0)], flush: true)
    builder.restart(at: CMTime(value: 10, timescale: 1))
    builder.receivedSamples = 1_000_000
    let second = builder.commit([token(5, frame: 0)], flush: true)
    #expect(first.first?.range.start == CMTime(value: 2, timescale: 1))
    #expect(second.first?.range.start == CMTime(value: 10, timescale: 1))
    #expect(text(first + second) == "Again and")
}
