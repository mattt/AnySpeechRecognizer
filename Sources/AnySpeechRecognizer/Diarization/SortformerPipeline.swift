import CoreMedia
import Foundation

/// Streams 16 kHz mono audio through log-mel features, Sortformer chunks,
/// the speaker cache, and the segmenter, and returns diarizer results.
///
/// Chunks follow NeMo's `streaming_feat_loader`: each chunk has up to one frame of
/// left context and runs once its full right context has arrived, so file and live input
/// produce the same results.
/// The end of a stream, at a gap in the input or a finalization past the processed audio,
/// processes the remaining audio with the right context that exists.
/// Later audio starts a new stream with new features and the same speaker cache,
/// so speaker identities stay the same.
/// One instance serves one executor.
final class SortformerPipeline {
    /// Results and the volatile range after one operation.
    struct Update: Sendable {
        var results: [SpeakerDiarizer.Result] = []
        /// The audio whose results can still change, or `nil`.
        var volatileRange: CMTimeRange?
    }

    static let sampleRate: CMTimeScale = 16_000

    let inference: any SortformerInference
    let configuration: SortformerConfiguration
    let reportsVolatileResults: Bool
    private var state: SortformerStreamingState
    private var segmenter: SpeakerSegmenter?
    private let parameters: SpeakerSegmenter.Parameters
    private var frontend = SortformerMelFrontend()
    /// The source time of the current stream's first sample, or `nil` before audio.
    private var origin: CMTime?
    /// The next mel frame to process in the current stream.
    private var melStart = 0
    /// Whether the next audio starts a new stream.
    private var streamEnded = false
    /// Predictions for the right context of the latest chunk.
    private var preview: [(start: CMTime, end: CMTime, probabilities: ArraySlice<Float>)] = []
    private var chunkBuffer: [Float]
    /// The last volatile result, which is not published again unchanged.
    private var lastVolatile: SpeakerDiarizer.Result?
    /// Committed frame probabilities, row-major, when recording is on; parity tests set it.
    var recordedProbabilities: [Float]?

    init(
        inference: any SortformerInference, parameters: SpeakerSegmenter.Parameters,
        reportsVolatileResults: Bool
    ) {
        self.inference = inference
        configuration = inference.configuration
        self.parameters = parameters
        self.reportsVolatileResults = reportsVolatileResults
        state = SortformerStreamingState(configuration: inference.configuration)
        chunkBuffer = [Float](
            repeating: 0,
            count: inference.configuration.chunkMelFrames * SortformerMelFrontend.melBinCount)
    }

    private var frameSamples: Int { configuration.subsampling * SortformerMelFrontend.hopLength }

    private func time(sample: Int) -> CMTime {
        (origin ?? .zero) + CMTime(value: CMTimeValue(sample), timescale: Self.sampleRate)
    }

    /// The source time after the last sample of the current stream.
    private var streamEnd: CMTime { time(sample: frontend.sampleCount) }

    /// Adds samples that start at a source time.
    /// - Parameters:
    ///   - samples: 16 kHz mono samples.
    ///   - start: The source time of the first sample.
    /// - Returns: New results and the volatile range.
    func append(_ samples: UnsafeBufferPointer<Float>, at start: CMTime) throws -> Update {
        var update = Update()
        if origin != nil, !streamEnded {
            // A gap in the input ends the stream; overlapping times continue it.
            let expected = streamEnd
            if start > expected + CMTime(value: 1, timescale: Self.sampleRate) {
                update.results += try endStream(cutAt: expected)
            }
        }
        if origin == nil || streamEnded {
            startStream(at: start)
        }
        if segmenter == nil {
            segmenter = SpeakerSegmenter(
                speakerCount: configuration.speakerCount, parameters: parameters, origin: start)
        }
        frontend.append(samples)
        let chunkFrames = configuration.chunkLength * configuration.subsampling
        let rightFrames = configuration.rightContext * configuration.subsampling
        while frontend.frameCount >= melStart + chunkFrames + rightFrames {
            try runChunk(end: melStart + chunkFrames)
        }
        if let result = segmenter?.decide() { update.results.append(result) }
        update.volatileRange = volatile(into: &update)
        return update
    }

    /// Makes results final through a time.
    ///
    /// When the model has not processed audio through `time`, the stream ends:
    /// the remaining audio is processed with the right context that exists.
    /// - Parameter time: The inclusive source time.
    /// - Returns: New results and the volatile range.
    func finalize(through time: CMTime) throws -> Update {
        var update = Update()
        guard let segmenter, origin != nil else { return update }
        let target = CMTime(value: 1, timescale: Self.sampleRate) + time
        if !streamEnded, target > segmenter.frontier {
            update.results += try endStream(cutAt: min(target, streamEnd))
        } else if let result = self.segmenter?.cut(
            at: min(target, segmenter.frontier), forced: true)
        {
            update.results.append(result)
        }
        update.volatileRange = volatile(into: &update)
        return update
    }

    private func startStream(at start: CMTime) {
        origin = start
        streamEnded = false
        frontend = SortformerMelFrontend()
        melStart = 0
        preview = []
    }

    /// Processes the rest of the stream, ends it, and cuts at `cut`.
    private func endStream(cutAt cut: CMTime) throws -> [SpeakerDiarizer.Result] {
        frontend.finish()
        let total = frontend.frameCount
        let chunkFrames = configuration.chunkLength * configuration.subsampling
        while melStart < total {
            try runChunk(end: min(melStart + chunkFrames, total))
        }
        preview = []
        streamEnded = true
        guard var segmenter else { return [] }
        segmenter.endStream(at: streamEnd)
        var results: [SpeakerDiarizer.Result] = []
        if let result = segmenter.cut(at: min(cut, streamEnd), forced: true) {
            results.append(result)
        }
        self.segmenter = segmenter
        return results
    }

    /// Runs the chunk that ends at mel frame `end`, with the context that exists.
    private func runChunk(end: Int) throws {
        let leftFrames = min(configuration.leftContext * configuration.subsampling, melStart)
        let rightFrames = min(
            configuration.rightContext * configuration.subsampling, frontend.frameCount - end)
        let first = melStart - leftFrames
        let count = end + rightFrames - first
        let bins = SortformerMelFrontend.melBinCount
        chunkBuffer.withUnsafeMutableBufferPointer { buffer in
            frontend.copyFrames(first..<(first + count), into: buffer.baseAddress!)
            (buffer.baseAddress! + count * bins).update(
                repeating: 0, count: buffer.count - count * bins)
        }
        let output = try inference.predict(
            chunk: chunkBuffer, chunkFrames: count, speakerCache: state.speakerCache,
            fifo: state.fifo)
        let leftContext = Int((Double(leftFrames) / Double(configuration.subsampling)).rounded())
        let rightContext = Int(
            (Double(rightFrames) / Double(configuration.subsampling)).rounded(.up))
        let cacheRows = state.speakerCacheFrames + state.fifoFrames
        let probabilities = state.update(
            embeddings: output.embeddings, embeddingCount: output.embeddingCount,
            predictions: output.predictions, leftContext: leftContext, rightContext: rightContext)

        let speakers = configuration.speakerCount
        let firstFrame = melStart / configuration.subsampling
        let frames = probabilities.count / speakers
        let limit = frontend.finished ? frontend.sampleCount : Int.max
        for frame in 0..<frames {
            appendFrame(
                firstFrame + frame,
                probabilities: probabilities[(frame * speakers)..<((frame + 1) * speakers)],
                limit: limit)
        }
        // Right-context predictions preview the next frames for volatile results.
        preview = []
        let previewStart = cacheRows + leftContext + frames
        let previewRows = max(0, output.embeddingCount - leftContext - frames)
        for row in 0..<previewRows {
            let index = (previewStart + row) * speakers
            guard index + speakers <= output.predictions.count else { break }
            let frameIndex = firstFrame + frames + row
            let start = frameIndex * frameSamples
            let end = min((frameIndex + 1) * frameSamples, frontend.sampleCount)
            guard end > start else { break }
            preview.append(
                (
                    time(sample: start), time(sample: end),
                    output.predictions[index..<(index + speakers)]
                ))
        }
        melStart = end
        frontend.discard(before: melStart - configuration.leftContext * configuration.subsampling)
    }

    private func appendFrame(_ index: Int, probabilities: ArraySlice<Float>, limit: Int) {
        let start = index * frameSamples
        let end = min((index + 1) * frameSamples, limit)
        guard end > start else { return }
        recordedProbabilities?.append(contentsOf: probabilities)
        segmenter?.append(
            start: time(sample: start), end: time(sample: end), probabilities: probabilities)
    }

    /// Adds a volatile result when requested and it changed, and returns the volatile range.
    private func volatile(into update: inout Update) -> CMTimeRange? {
        guard reportsVolatileResults, let segmenter else { return nil }
        guard
            let result = segmenter.volatileResult(
                preview: preview, end: preview.last?.end ?? segmenter.frontier)
        else {
            lastVolatile = nil
            return nil
        }
        if result != lastVolatile || !update.results.isEmpty {
            update.results.append(result)
            lastVolatile = result
        }
        return result.range
    }
}
