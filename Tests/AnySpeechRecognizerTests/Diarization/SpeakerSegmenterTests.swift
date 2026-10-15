import CoreMedia
import Foundation
import Testing

@testable import AnySpeechRecognizer

/// Feeds 80 ms frames of probabilities, one row per frame, from `first`.
private func feed(
    _ segmenter: inout SpeakerSegmenter, _ rows: [[Float]], first: Int = 0
) -> [SpeakerDiarizer.Result] {
    var results: [SpeakerDiarizer.Result] = []
    for (offset, row) in rows.enumerated() {
        let index = first + offset
        segmenter.append(
            start: seconds(Double(index) * 0.08), end: seconds(Double(index + 1) * 0.08),
            probabilities: row[...])
        if let result = segmenter.decide() { results.append(result) }
    }
    return results
}

private func rows(_ count: Int, _ row: [Float]) -> [[Float]] {
    Array(repeating: row, count: count)
}

private let plain = SortformerDiarizationModel.PostProcessing.none.segmenterParameters

@Test func segmenterSplitsAtSpeakerChangesAndSilence() {
    var segmenter = SpeakerSegmenter(speakerCount: 2, parameters: plain, origin: .zero)
    var results = feed(
        &segmenter,
        rows(10, [0.9, 0.1]) + rows(10, [0.1, 0.9]) + rows(10, [0.1, 0.1]) + rows(5, [0.9, 0.1]))
    if let last = segmenter.cut(at: seconds(35 * 0.08), forced: true) { results.append(last) }
    expectFinalizationContract(results)
    #expect(results.allSatisfy { $0.isFinal })
    #expect(
        finalSegments(results) == [
            segment(0, 0, 0.8), segment(1, 0.8, 1.6), segment(0, 2.4, 2.8),
        ])
    // Turns end at speaker changes, so the first two results are one turn each.
    #expect(results.first?.segments == [segment(0, 0, 0.8)])
}

@Test func segmenterReportsOverlappingSpeech() {
    var segmenter = SpeakerSegmenter(speakerCount: 2, parameters: plain, origin: .zero)
    var results = feed(
        &segmenter, rows(10, [0.9, 0.1]) + rows(5, [0.9, 0.9]) + rows(10, [0.1, 0.9]))
    if let last = segmenter.cut(at: seconds(25 * 0.08), forced: true) { results.append(last) }
    expectFinalizationContract(results)
    #expect(finalSegments(results) == [segment(0, 0, 1.2), segment(1, 0.8, 2.0)])
}

@Test func segmenterAppliesPaddingAndDurationFilters() {
    let parameters = SpeakerSegmenter.Parameters(
        onset: 0.5, offset: 0.5, padOnset: seconds(0.08), padOffset: seconds(0.08),
        minimumDurationOn: seconds(0.4), minimumDurationOff: seconds(0.3))
    var segmenter = SpeakerSegmenter(speakerCount: 1, parameters: parameters, origin: .zero)
    // Speech 0.8–1.6, a 0.16 s gap that the filter fills, speech 1.76–2.4,
    // silence, then a 0.08 s blip that is too short to keep.
    let pattern: [[Float]] =
        rows(10, [0.1]) + rows(10, [0.9]) + rows(2, [0.1]) + rows(8, [0.9]) + rows(15, [0.1])
        + rows(1, [0.9]) + rows(15, [0.1])
    var results = feed(&segmenter, pattern)
    if let last = segmenter.cut(at: seconds(Double(pattern.count) * 0.08), forced: true) {
        results.append(last)
    }
    expectFinalizationContract(results)
    #expect(finalSegments(results) == [segment(0, 0.72, 2.48)])
}

@Test func segmenterWaitsForTheDecidedTime() {
    let parameters = SpeakerSegmenter.Parameters(
        onset: 0.5, offset: 0.5, padOnset: seconds(0.2), padOffset: .zero,
        minimumDurationOn: seconds(0.5), minimumDurationOff: seconds(0.3))
    var segmenter = SpeakerSegmenter(speakerCount: 1, parameters: parameters, origin: .zero)
    // A segment that ends at 1.6 s is decided only 1 s later.
    var results = feed(&segmenter, rows(20, [0.9]) + rows(5, [0.1]))
    #expect(results.isEmpty)
    #expect(segmenter.decidedTime == seconds(2.0 - 1.0))
    results += feed(&segmenter, rows(10, [0.1]), first: 25)
    #expect(results.count == 1)
    #expect(results.first?.segments == [segment(0, 0, 1.6)])
}

@Test func segmenterSplitsLongSpeechAfterTheUndecidedLimit() {
    var segmenter = SpeakerSegmenter(speakerCount: 1, parameters: plain, origin: .zero)
    let results = feed(&segmenter, rows(200, [0.9]))
    expectFinalizationContract(results)
    #expect(!results.isEmpty)
    let first = try? #require(results.first)
    #expect(first?.resultsFinalizationTime ?? .zero >= SpeakerSegmenter.maximumUndecided)
    // Split parts join again in a timeline.
    var all = results
    if let last = segmenter.cut(at: seconds(16), forced: true) { all.append(last) }
    #expect(finalSegments(all) == [segment(0, 0, 16)])
}

@Test func forcedCutFinalizesOpenSpeechAndContinuesAfterIt() {
    var segmenter = SpeakerSegmenter(speakerCount: 2, parameters: plain, origin: .zero)
    _ = feed(&segmenter, rows(10, [0.9, 0.1]))
    let cut = segmenter.cut(at: seconds(0.5), forced: true)
    #expect(cut?.segments == [segment(0, 0, 0.5)])
    #expect(cut?.resultsFinalizationTime == seconds(0.5))
    var results = [cut].compactMap { $0 }
    results += feed(&segmenter, rows(10, [0.1, 0.1]), first: 10)
    if let last = segmenter.cut(at: seconds(1.6), forced: true) { results.append(last) }
    expectFinalizationContract(results)
    #expect(finalSegments(results) == [segment(0, 0, 0.8)])
}

@Test func volatileResultsPreviewUncommittedFrames() throws {
    var segmenter = SpeakerSegmenter(speakerCount: 2, parameters: plain, origin: .zero)
    _ = feed(&segmenter, rows(5, [0.9, 0.1]))
    let preview: [(start: CMTime, end: CMTime, probabilities: ArraySlice<Float>)] = (5..<8).map {
        (seconds(Double($0) * 0.08), seconds(Double($0 + 1) * 0.08), [0.1, 0.9][...])
    }
    let result = try #require(segmenter.volatileResult(preview: preview, end: seconds(0.64)))
    #expect(!result.isFinal)
    #expect(result.range == range(0, 0.64))
    #expect(result.segments == [segment(0, 0, 0.4), segment(1, 0.4, 0.64)])
    // The preview does not change the committed state.
    #expect(segmenter.frontier == seconds(0.4))
}
