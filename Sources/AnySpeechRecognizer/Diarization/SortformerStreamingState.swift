import Foundation

/// The fixed shapes and streaming settings of a Sortformer Core ML model.
///
/// Lengths are in 80 ms diarization frames unless their names say otherwise.
/// The defaults are NVIDIA's low-latency setting, which the built-in model uses.
struct SortformerConfiguration: Sendable, Hashable {
    var chunkLength = 6
    var leftContext = 1
    var rightContext = 7
    var fifoLength = 188
    var speakerCacheLength = 188
    var updatePeriod = 144
    /// Mel frames per diarization frame.
    var subsampling = 8
    var speakerCount = 4
    var embeddingSize = 512
    var melBinCount = 128

    /// The mel frames of one model input: left context, chunk, and right context.
    var chunkMelFrames: Int { (leftContext + chunkLength + rightContext) * subsampling }
    /// The rows of the model's speaker predictions: cache, FIFO, and chunk.
    var predictionRows: Int {
        speakerCacheLength + fifoLength + leftContext + chunkLength + rightContext
    }
    /// The embedding rows of one model input.
    var chunkEmbeddingRows: Int { leftContext + chunkLength + rightContext }
}

/// The speaker cache and FIFO queue of streaming Sortformer, updated after each chunk.
///
/// This is a port of the synchronous `streaming_update` and the speaker cache compression
/// of NVIDIA NeMo's `SortformerModules` (Apache License 2.0).
/// The speaker cache keeps the most informative frames of each speaker in arrival order,
/// so each speaker keeps its output slot for the whole stream.
/// Embeddings and predictions are row-major: one row per frame.
struct SortformerStreamingState: Sendable {
    let configuration: SortformerConfiguration
    /// Silence frames reserved for each speaker in a compressed cache.
    let silenceFramesPerSpeaker = 3
    let predictionScoreThreshold: Float = 0.25
    let latestScoreBoost: Float = 0.05
    let silenceThreshold: Float = 0.2
    let strongBoostRate = 0.75
    let weakBoostRate = 1.5
    let minimumPositiveScoreRate = 0.5

    private(set) var speakerCache: [Float] = []
    private(set) var speakerCachePredictions: [Float] = []
    private(set) var compressed = false
    private(set) var fifo: [Float] = []
    private(set) var fifoPredictions: [Float] = []
    private(set) var meanSilenceEmbedding: [Float]
    private(set) var silenceFrameCount = 0

    init(configuration: SortformerConfiguration) {
        self.configuration = configuration
        meanSilenceEmbedding = [Float](repeating: 0, count: configuration.embeddingSize)
    }

    var speakerCacheFrames: Int { speakerCache.count / configuration.embeddingSize }
    var fifoFrames: Int { fifo.count / configuration.embeddingSize }

    /// Adds a chunk's embeddings to the FIFO queue, moves the oldest FIFO frames into the
    /// speaker cache when the queue is full, and compresses the cache when it is full.
    /// - Parameters:
    ///   - embeddings: The chunk's pre-encoder embeddings, `embeddingCount` rows.
    ///   - embeddingCount: The valid embedding rows, including left and right context.
    ///   - predictions: The model's predictions for the cache, FIFO, and chunk rows,
    ///     packed in that order.
    ///   - leftContext: Rows of left context at the start of the chunk.
    ///   - rightContext: Rows of right context at the end of the chunk.
    /// - Returns: The predictions of the chunk's rows without context, row-major.
    mutating func update(
        embeddings: [Float], embeddingCount: Int, predictions: [Float], leftContext: Int,
        rightContext: Int
    ) -> [Float] {
        let size = configuration.embeddingSize
        let speakers = configuration.speakerCount
        let cacheFrames = speakerCacheFrames
        let queuedFrames = fifoFrames
        let chunkFrames = max(0, embeddingCount - leftContext - rightContext)

        func rows(_ values: [Float], _ range: Range<Int>, width: Int) -> ArraySlice<Float> {
            values[(range.lowerBound * width)..<(range.upperBound * width)]
        }
        let chunkStart = cacheFrames + queuedFrames + leftContext
        let chunkPredictions = Array(
            rows(predictions, chunkStart..<(chunkStart + chunkFrames), width: speakers))
        fifoPredictions = Array(
            rows(predictions, cacheFrames..<(cacheFrames + queuedFrames), width: speakers))
        fifo += rows(embeddings, leftContext..<(leftContext + chunkFrames), width: size)
        fifoPredictions += chunkPredictions

        guard queuedFrames + chunkFrames > configuration.fifoLength else {
            return chunkPredictions
        }
        var popCount = max(
            configuration.updatePeriod, chunkFrames - configuration.fifoLength + queuedFrames)
        popCount = min(popCount, queuedFrames + chunkFrames)
        let poppedEmbeddings = Array(fifo[0..<(popCount * size)])
        let poppedPredictions = Array(fifoPredictions[0..<(popCount * speakers)])
        updateSilenceProfile(embeddings: poppedEmbeddings, predictions: poppedPredictions)
        fifo.removeFirst(popCount * size)
        fifoPredictions.removeFirst(popCount * speakers)

        speakerCache += poppedEmbeddings
        if compressed {
            speakerCachePredictions += poppedPredictions
        } else {
            // Before the first compression, keep fresh predictions for every cached frame.
            speakerCachePredictions =
                Array(rows(predictions, 0..<cacheFrames, width: speakers)) + poppedPredictions
        }
        if speakerCacheFrames > configuration.speakerCacheLength {
            (speakerCache, speakerCachePredictions) = compress(
                embeddings: speakerCache, predictions: speakerCachePredictions)
            compressed = true
        }
        return chunkPredictions
    }

    /// Updates the running mean of embeddings whose summed speaker activity is low.
    private mutating func updateSilenceProfile(embeddings: [Float], predictions: [Float]) {
        let size = configuration.embeddingSize
        let speakers = configuration.speakerCount
        let frames = predictions.count / speakers
        var sum = [Float](repeating: 0, count: size)
        var count = 0
        for frame in 0..<frames {
            var activity: Float = 0
            for speaker in 0..<speakers { activity += predictions[frame * speakers + speaker] }
            guard activity < silenceThreshold else { continue }
            count += 1
            for index in 0..<size { sum[index] += embeddings[frame * size + index] }
        }
        guard count > 0 else { return }
        let total = silenceFrameCount + count
        for index in 0..<size {
            meanSilenceEmbedding[index] =
                (meanSilenceEmbedding[index] * Float(silenceFrameCount) + sum[index])
                / Float(max(total, 1))
        }
        silenceFrameCount = total
    }

    /// Keeps the `speakerCacheLength` most important frames, grouped by speaker
    /// in arrival order, with silence frames reserved for each speaker.
    func compress(embeddings: [Float], predictions: [Float]) -> ([Float], [Float]) {
        let size = configuration.embeddingSize
        let speakers = configuration.speakerCount
        let cacheLength = configuration.speakerCacheLength
        let frames = predictions.count / speakers
        let perSpeaker = cacheLength / speakers - silenceFramesPerSpeaker
        let strongBoost = Int((Double(perSpeaker) * strongBoostRate).rounded(.down))
        let weakBoost = Int((Double(perSpeaker) * weakBoostRate).rounded(.down))
        let minimumPositive = Int((Double(perSpeaker) * minimumPositiveScoreRate).rounded(.down))

        // Scores are high for confident, non-overlapped speech of one speaker.
        var scores = [Float](repeating: 0, count: frames * speakers)
        let threshold = predictionScoreThreshold
        for frame in 0..<frames {
            var silenceSum: Float = 0
            for speaker in 0..<speakers {
                silenceSum += log(max(1 - predictions[frame * speakers + speaker], threshold))
            }
            for speaker in 0..<speakers {
                let probability = predictions[frame * speakers + speaker]
                scores[frame * speakers + speaker] =
                    log(max(probability, threshold)) - log(max(1 - probability, threshold))
                    + silenceSum - log(0.5)
            }
        }
        // Non-speech scores are disabled. When a speaker has enough positive scores,
        // its non-positive speech scores, usually overlapped speech, are disabled too.
        for speaker in 0..<speakers {
            var positive = 0
            for frame in 0..<frames {
                let index = frame * speakers + speaker
                if predictions[index] > 0.5 && scores[index] > 0 { positive += 1 }
            }
            for frame in 0..<frames {
                let index = frame * speakers + speaker
                if predictions[index] <= 0.5 {
                    scores[index] = -.infinity
                } else if scores[index] <= 0 && positive >= minimumPositive {
                    scores[index] = -.infinity
                }
            }
        }
        if latestScoreBoost > 0, frames > cacheLength {
            for index in (cacheLength * speakers)..<(frames * speakers) {
                scores[index] += latestScoreBoost
            }
        }
        // Strong boosting keeps at least some frames of each speaker,
        // and weak boosting keeps one speaker from filling the cache.
        boostTopScores(&scores, frames: frames, count: strongBoost, amount: -2 * log(0.5))
        boostTopScores(&scores, frames: frames, count: weakBoost, amount: -log(0.5))

        // Flatten speaker-major, with silence slots at the end of each speaker's block.
        let paddedFrames = frames + silenceFramesPerSpeaker
        var flattened: [(score: Float, index: Int)] = []
        flattened.reserveCapacity(paddedFrames * speakers)
        for speaker in 0..<speakers {
            for frame in 0..<paddedFrames {
                let score = frame < frames ? scores[frame * speakers + speaker] : .infinity
                flattened.append((score, speaker * paddedFrames + frame))
            }
        }
        let placeholder = Int.max
        let selected = Self.topIndices(flattened, count: cacheLength).map { entry in
            entry.score == -.infinity ? placeholder : entry.index
        }.sorted()

        var cache = [Float](repeating: 0, count: cacheLength * size)
        var cachePredictions = [Float](repeating: 0, count: cacheLength * speakers)
        for (slot, index) in selected.enumerated() {
            let frame = index == placeholder ? frames : index % paddedFrames
            if frame >= frames {
                for offset in 0..<size {
                    cache[slot * size + offset] = meanSilenceEmbedding[offset]
                }
                continue
            }
            for offset in 0..<size {
                cache[slot * size + offset] = embeddings[frame * size + offset]
            }
            for speaker in 0..<speakers {
                cachePredictions[slot * speakers + speaker] =
                    predictions[frame * speakers + speaker]
            }
        }
        return (cache, cachePredictions)
    }

    /// Adds `amount` to each speaker's `count` highest scores. Disabled scores stay disabled.
    private func boostTopScores(_ scores: inout [Float], frames: Int, count: Int, amount: Float) {
        let speakers = configuration.speakerCount
        for speaker in 0..<speakers {
            let column = (0..<frames).map { (score: scores[$0 * speakers + speaker], index: $0) }
            for entry in Self.topIndices(column, count: count) {
                scores[entry.index * speakers + speaker] += amount
            }
        }
    }

    /// Returns the `count` entries with the highest scores.
    /// Equal scores keep their original order, so earlier entries win ties.
    static func topIndices(_ entries: [(score: Float, index: Int)], count: Int) -> [(
        score: Float, index: Int
    )] {
        let ordered = entries.enumerated().sorted { lhs, rhs in
            lhs.element.score != rhs.element.score
                ? lhs.element.score > rhs.element.score : lhs.offset < rhs.offset
        }
        return ordered.prefix(count).map(\.element)
    }
}
