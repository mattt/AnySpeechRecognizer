import AVFoundation
import CoreML
import CoreMedia
import Foundation
import Testing

@testable import AnySpeechRecognizer

/// A diarization model whose executor publishes scripted results.
struct FakeDiarizationModel: SpeakerDiarizationModel {
    var modelID = "test/fake-diarization"
    var capabilities = SpeakerDiarizationCapabilities(
        maximumSpeakerCount: 4, overlappingSpeech: true, liveInput: true,
        reportingOptions: [.volatileResults])
    let executor: FakeDiarizationExecutor
    var availability = SpeechRecognitionAvailability.ready
    var installation: AssetInstallationRequest?

    func makeExecutor() -> any SpeakerDiarizationExecutor { executor }
    func availability(for request: SpeakerDiarizationRequest) async
        -> SpeechRecognitionAvailability
    {
        availability
    }
    func assetInstallationRequest(supporting requests: [SpeakerDiarizationRequest]) async throws
        -> AssetInstallationRequest?
    {
        installation
    }
}

/// Emits scripted results once it has pulled audio through each result's end.
actor FakeDiarizationExecutor: SpeakerDiarizationExecutor {
    private var pending: [SpeakerDiarizer.Result]
    private var deferred: [SpeakerDiarizer.Result]
    private var sink: SpeakerDiarizationResultSink?
    private(set) var prepareCount = 0
    private(set) var pulledFrames = 0
    private(set) var finalizeTimes: [CMTime] = []

    init(results: [SpeakerDiarizer.Result] = [], deferred: [SpeakerDiarizer.Result] = []) {
        pending = results
        self.deferred = deferred
    }

    func prepare(request: SpeakerDiarizationRequest, format: AVAudioFormat) async throws {
        prepareCount += 1
    }

    func analyze(_ input: SpeechAnalysisInput, results: SpeakerDiarizationResultSink) async throws {
        sink = results
        for try await buffer in input {
            pulledFrames += Int(buffer.buffer.frameLength)
            while let next = pending.first, next.range.end <= buffer.endTime {
                pending.removeFirst()
                try await results.emit(next)
                if !next.isFinal { results.setVolatileRange(next.range) }
            }
        }
    }

    func finalize(through time: CMTime) async throws {
        finalizeTimes.append(time)
        let ready = deferred.filter { $0.range.start <= time }
        deferred.removeAll { $0.range.start <= time }
        for result in ready { try await sink?.emit(result) }
    }

    func cancel() async { sink = nil }
}

func seconds(_ value: Double) -> CMTime {
    CMTime(value: CMTimeValue((value * 16_000).rounded()), timescale: 16_000)
}

func range(_ start: Double, _ end: Double) -> CMTimeRange {
    CMTimeRange(start: seconds(start), end: seconds(end))
}

func segment(_ speaker: Int, _ start: Double, _ end: Double) -> SpeakerDiarizer.Segment {
    .init(speaker: .init(id: speaker), range: range(start, end))
}

func diarization(
    _ segments: [SpeakerDiarizer.Segment], range: CMTimeRange? = nil, final: Double? = nil
) -> SpeakerDiarizer.Result {
    let hull =
        range
        ?? CMTimeRange(
            start: segments.map(\.range.start).min() ?? .zero,
            end: segments.map(\.range.end).max() ?? .zero)
    return .init(
        segments: segments, range: hull,
        resultsFinalizationTime: final.map(seconds) ?? hull.end)
}

/// A Sortformer stand-in that labels each 80 ms row by the frequency of a test tone:
/// low tones are speaker 0, high tones are speaker 1, and both together are overlap.
///
/// It follows the model's shapes: rows of the cache and FIFO, then one row for each
/// pre-encoder output frame, whose count follows NeMo's three stride-2 convolutions.
final class ToneInference: SortformerInference {
    let configuration = SortformerConfiguration()
    private(set) var calls = 0
    private(set) var cacheSizes: [Int] = []

    static func embeddingRows(_ frames: Int) -> Int {
        var length = frames
        for _ in 0..<3 { length = (length - 1) / 2 + 1 }
        return frames > 0 ? length : 0
    }

    func predict(chunk: [Float], chunkFrames: Int, speakerCache: [Float], fifo: [Float]) throws
        -> SortformerPrediction
    {
        calls += 1
        let size = configuration.embeddingSize
        let speakers = configuration.speakerCount
        let bins = SortformerMelFrontend.melBinCount
        cacheSizes.append(speakerCache.count / size)
        let offset = speakerCache.count / size + fifo.count / size
        let rows = Self.embeddingRows(chunkFrames)
        var predictions = [Float](repeating: 0.02, count: configuration.predictionRows * speakers)
        for row in 0..<rows {
            var low: Float = -.infinity
            var high: Float = -.infinity
            for frame in (row * 8)..<min(row * 8 + 8, chunkFrames) {
                for bin in 0..<40 { low = max(low, chunk[frame * bins + bin]) }
                for bin in 64..<bins { high = max(high, chunk[frame * bins + bin]) }
            }
            if low > -6 { predictions[(offset + row) * speakers + 0] = 0.95 }
            if high > -6 { predictions[(offset + row) * speakers + 1] = 0.95 }
        }
        return SortformerPrediction(
            predictions: predictions, embeddings: [Float](repeating: 0, count: rows * size),
            embeddingCount: rows)
    }
}

/// 16 kHz mono audio with low and high tones in the given ranges, in seconds.
func toneSamples(duration: Double, low: [(Double, Double)], high: [(Double, Double)]) -> [Float] {
    var samples = [Float](repeating: 0, count: Int(duration * 16_000))
    for (ranges, frequency) in [(low, 300.0), (high, 3_000.0)] {
        for (start, end) in ranges {
            for index in Int(start * 16_000)..<min(samples.count, Int(end * 16_000)) {
                samples[index] += Float(
                    0.3 * sin(2 * Double.pi * frequency * Double(index) / 16_000))
            }
        }
    }
    return samples
}

/// A Sortformer model that runs `ToneInference` from a placeholder compiled model directory.
func toneModel(postProcessing: SortformerDiarizationModel.PostProcessing = .none) throws
    -> SortformerDiarizationModel
{
    let directory = FileManager.default.temporaryDirectory.appending(
        path: "anyspeech-tone-\(UUID().uuidString).mlmodelc")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data().write(to: directory.appending(path: "coremldata.bin"))
    return SortformerDiarizationModel(
        assets: .compiledModel(directory), postProcessing: postProcessing,
        manifest: .builtIn, store: .shared, makeInference: { _, _ in ToneInference() })
}

/// Runs a pipeline over samples in blocks and returns every result.
func runPipeline(
    _ samples: [Float], block: Int, volatile: Bool = false,
    postProcessing: SortformerDiarizationModel.PostProcessing = .none,
    origin: CMTime = .zero
) throws -> [SpeakerDiarizer.Result] {
    let pipeline = SortformerPipeline(
        inference: ToneInference(), parameters: postProcessing.segmenterParameters,
        reportsVolatileResults: volatile)
    var results: [SpeakerDiarizer.Result] = []
    try samples.withUnsafeBufferPointer { samples in
        var start = 0
        while start < samples.count {
            let end = min(start + block, samples.count)
            results += try pipeline.append(
                UnsafeBufferPointer(rebasing: samples[start..<end]),
                at: origin + CMTime(value: CMTimeValue(start), timescale: 16_000)
            ).results
            start = end
        }
    }
    results +=
        try pipeline.finalize(
            through: origin + CMTime(value: CMTimeValue(samples.count - 1), timescale: 16_000)
        ).results
    return results
}

/// The final segments of results, with split parts joined.
func finalSegments(_ results: [SpeakerDiarizer.Result]) -> [SpeakerDiarizer.Segment] {
    var timeline = SpeakerTimeline()
    for result in results where result.isFinal { timeline.apply(result) }
    return timeline.segments
}

/// Checks the finalization contract: final results never overlap or precede an earlier
/// finalization, and volatile results start at the current boundary.
func expectFinalizationContract(
    _ results: [SpeakerDiarizer.Result], sourceLocation: SourceLocation = #_sourceLocation
) {
    var boundary: CMTime?
    for result in results {
        if let boundary {
            #expect(result.range.start >= boundary, sourceLocation: sourceLocation)
            #expect(result.resultsFinalizationTime >= boundary, sourceLocation: sourceLocation)
        }
        for segment in result.segments {
            #expect(segment.range.start >= result.range.start, sourceLocation: sourceLocation)
            #expect(segment.range.end <= result.range.end, sourceLocation: sourceLocation)
        }
        if result.isFinal { boundary = result.resultsFinalizationTime }
    }
}
