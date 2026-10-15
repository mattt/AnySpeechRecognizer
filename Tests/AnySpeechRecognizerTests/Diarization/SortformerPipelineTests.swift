import CoreMedia
import Foundation
import Testing

@testable import AnySpeechRecognizer

/// Two tone "speakers": low tones are speaker 0 and high tones are speaker 1.
private let conversation = toneSamples(
    duration: 30,
    low: [(1, 6), (12, 15), (20, 26)],
    high: [(6.5, 11), (15, 18), (24, 28)])

/// Checks a segment's boundaries against expected seconds within one 80 ms frame and padding.
private func expect(
    _ segments: [SpeakerDiarizer.Segment], match expected: [(Int, Double, Double)],
    tolerance: Double = 0.17, sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(segments.count == expected.count, sourceLocation: sourceLocation)
    for (segment, (speaker, start, end)) in zip(segments, expected) {
        #expect(segment.speaker.id == speaker, sourceLocation: sourceLocation)
        #expect(
            abs(segment.range.start.seconds - start) <= tolerance, sourceLocation: sourceLocation)
        #expect(abs(segment.range.end.seconds - end) <= tolerance, sourceLocation: sourceLocation)
    }
}

@Test func pipelineKeepsSpeakerIdentitiesAcrossChunks() throws {
    let results = try runPipeline(conversation, block: 4_096)
    expectFinalizationContract(results)
    #expect(results.allSatisfy { $0.isFinal })
    expect(
        finalSegments(results),
        match: [
            (0, 1, 6), (1, 6.5, 11), (0, 12, 15), (1, 15, 18), (0, 20, 26), (1, 24, 28),
        ])
}

@Test func blockSizeDoesNotChangeResults() throws {
    let file = try runPipeline(conversation, block: 4_096)
    let live = try runPipeline(conversation, block: 1_600)
    let odd = try runPipeline(conversation, block: 777)
    #expect(finalSegments(file) == finalSegments(live))
    #expect(finalSegments(file) == finalSegments(odd))
}

@Test func volatileResultsPrecedeFinalResults() throws {
    let results = try runPipeline(conversation, block: 1_600, volatile: true)
    expectFinalizationContract(results)
    let volatile = results.filter { !$0.isFinal }
    #expect(!volatile.isEmpty)
    // Volatile results report speech before the final result for the same audio.
    let firstFinalSpeaker1 = try #require(
        results.firstIndex { $0.isFinal && $0.segments.contains { $0.speaker.id == 1 } })
    let firstVolatileSpeaker1 = try #require(
        results.firstIndex { !$0.isFinal && $0.segments.contains { $0.speaker.id == 1 } })
    #expect(firstVolatileSpeaker1 < firstFinalSpeaker1)
    // Final results are the same with and without volatile results.
    let withoutVolatile = try runPipeline(conversation, block: 1_600)
    #expect(finalSegments(results) == finalSegments(withoutVolatile))
}

@Test func sourceOffsetsCarryToResults() throws {
    let origin = CMTime(value: 90, timescale: 1)
    let shifted = try runPipeline(conversation, block: 4_096, origin: origin)
    let plain = try runPipeline(conversation, block: 4_096)
    let expected = finalSegments(plain).map {
        SpeakerDiarizer.Segment(
            speaker: $0.speaker,
            range: CMTimeRange(start: $0.range.start + origin, end: $0.range.end + origin))
    }
    #expect(finalSegments(shifted) == expected)
}

@Test func finalizationAndGapsKeepSpeakerIdentities() throws {
    let pipeline = SortformerPipeline(
        inference: ToneInference(),
        parameters: SortformerDiarizationModel.PostProcessing.none.segmenterParameters,
        reportsVolatileResults: false)
    var results: [SpeakerDiarizer.Result] = []
    let half = conversation.count / 2
    try conversation.withUnsafeBufferPointer { samples in
        results += try pipeline.append(
            UnsafeBufferPointer(rebasing: samples[0..<half]), at: .zero
        ).results
        // Finalize mid-stream, inside speaker 0's turn at 13 s.
        results += try pipeline.finalize(through: seconds(13)).results
        results += try pipeline.append(
            UnsafeBufferPointer(rebasing: samples[half..<(half + 16_000)]), at: seconds(15)
        ).results
        // Resume after a 10 s gap.
        results += try pipeline.append(
            UnsafeBufferPointer(rebasing: samples[(half + 16_000)...]), at: seconds(26)
        ).results
    }
    results += try pipeline.finalize(through: seconds(39.999)).results
    expectFinalizationContract(results)
    let segments = finalSegments(results)
    // Speaker 0 is the low tone before and after the finalization and the gap.
    #expect(
        segments.filter { $0.speaker.id == 0 }.allSatisfy { segment in
            let start = segment.range.start.seconds
            return [0.8...1.2, 11.8...12.2, 29.8...30.2].contains { $0.contains(start) }
        })
    #expect(Set(segments.map(\.speaker.id)) == [0, 1])
    // Nothing is attributed to the gap.
    #expect(!segments.contains { $0.range.start.seconds > 16.1 && $0.range.end.seconds < 26 })
}

@Test func streamingStateCompressesTheSpeakerCache() throws {
    var state = SortformerStreamingState(configuration: SortformerConfiguration())
    let configuration = state.configuration
    let size = configuration.embeddingSize
    // Each embedding's first value is its frame number, from 1. Speaker 0 speaks first,
    // then speaker 1, then nobody; predictions follow each row's frame.
    func speaker(_ frame: Int) -> Int { frame < 1 ? -1 : frame <= 180 ? 0 : frame <= 360 ? 1 : -1 }
    for chunk in 0..<120 {
        let rows = configuration.chunkLength
        var embeddings = [Float](repeating: 0, count: rows * size)
        for row in 0..<rows { embeddings[row * size] = Float(chunk * rows + row + 1) }
        let packed = (state.speakerCache + state.fifo + embeddings)
        var predictions = [Float](
            repeating: 0.01, count: configuration.predictionRows * configuration.speakerCount)
        for row in 0..<(packed.count / size) {
            let value = packed[row * size]
            // Silence slots hold the mean silence embedding, which is not a frame index.
            guard value == value.rounded(), case let who = speaker(Int(value)), who >= 0 else {
                continue
            }
            predictions[row * configuration.speakerCount + who] = 0.9
        }
        let output = state.update(
            embeddings: embeddings, embeddingCount: rows, predictions: predictions,
            leftContext: 0, rightContext: 0)
        #expect(output.count == rows * configuration.speakerCount)
        #expect(state.speakerCacheFrames <= configuration.speakerCacheLength)
        #expect(state.fifoFrames <= configuration.fifoLength)
    }
    #expect(state.speakerCacheFrames == configuration.speakerCacheLength)
    #expect(state.compressed)
    #expect(state.silenceFrameCount > 0)
    // The cache keeps frames of both speakers, grouped by speaker slot in arrival order.
    let cached = state.speakerCachePredictions
    let speakers = configuration.speakerCount
    let slots = (0..<configuration.speakerCacheLength).map { row -> Int in
        let values = cached[(row * speakers)..<((row + 1) * speakers)]
        guard let maximum = values.max(), maximum > 0.5 else { return -1 }
        return values.firstIndex(of: maximum)! - row * speakers
    }
    let speech = slots.filter { $0 >= 0 }
    #expect(Set(speech) == [0, 1])
    #expect(speech == speech.sorted())
}

@Test func melFrontendStreamsLikeOneBlockAndFollowsNeMoFrameCounts() {
    let samples = toneSamples(duration: 1.234, low: [(0.1, 0.9)], high: [(0.5, 1.2)])
    func features(block: Int) -> (Int, [Float]) {
        let frontend = SortformerMelFrontend()
        samples.withUnsafeBufferPointer { samples in
            var start = 0
            while start < samples.count {
                let end = min(start + block, samples.count)
                frontend.append(UnsafeBufferPointer(rebasing: samples[start..<end]))
                start = end
            }
        }
        let beforeFinish = frontend.frameCount
        frontend.finish()
        var values = [Float](repeating: 0, count: frontend.frameCount * 128)
        values.withUnsafeMutableBufferPointer {
            frontend.copyFrames(0..<frontend.frameCount, into: $0.baseAddress!)
        }
        return (beforeFinish, values)
    }
    let (available, whole) = features(block: samples.count)
    let (_, streamed) = features(block: 333)
    #expect(whole == streamed)
    #expect(whole.count / 128 == samples.count / 160)
    // Before the stream ends, a frame waits for the 200 samples after its center.
    #expect(available == (samples.count - 200) / 160 + 1)
}
