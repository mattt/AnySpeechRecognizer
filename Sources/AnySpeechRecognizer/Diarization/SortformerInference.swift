import CoreML
import Foundation

/// The outputs of one Sortformer model call.
struct SortformerPrediction: Sendable {
    /// Speaker probabilities, row-major: the valid cache rows, then the valid FIFO rows,
    /// then the chunk's embedding rows, `speakerCount` values per row.
    var predictions: [Float]
    /// The chunk's pre-encoder embeddings, row-major, `embeddingSize` values per row.
    var embeddings: [Float]
    /// The number of valid embedding rows.
    var embeddingCount: Int
}

/// Runs one Sortformer chunk: pre-encoder, encoder, and speaker heads.
/// The Core ML engine implements it; tests supply scripted predictions.
protocol SortformerInference: AnyObject {
    var configuration: SortformerConfiguration { get }
    /// Predicts speaker activity for a chunk with the current cache and FIFO.
    /// - Parameters:
    ///   - chunk: Log-mel features, time-major, `chunkFrames` frames of `melBinCount` values.
    ///   - chunkFrames: The valid mel frames, at most `configuration.chunkMelFrames`.
    ///   - speakerCache: Cache embeddings, row-major.
    ///   - fifo: FIFO embeddings, row-major.
    func predict(chunk: [Float], chunkFrames: Int, speakerCache: [Float], fifo: [Float]) throws
        -> SortformerPrediction
}

/// Runs a compiled Sortformer pipeline model with Core ML.
///
/// The model takes `chunk`, `chunk_lengths`, `spkcache`, `spkcache_lengths`, `fifo`,
/// and `fifo_lengths`, and returns `speaker_preds`, `chunk_pre_encoder_embs_out`,
/// and `chunk_pre_encoder_lengths_out`.
/// Its creator metadata records the streaming configuration.
/// The SDK does not mark `MLModel` as `Sendable`, so each executor loads its own engine.
final class CoreMLSortformerEngine: SortformerInference {
    let configuration: SortformerConfiguration
    private let model: MLModel
    private let chunk: MLMultiArray
    private let chunkLength: MLMultiArray
    private let speakerCache: MLMultiArray
    private let speakerCacheLength: MLMultiArray
    private let fifo: MLMultiArray
    private let fifoLength: MLMultiArray
    private let provider: MLDictionaryFeatureProvider

    /// Loads and checks a compiled model.
    /// - Parameters:
    ///   - url: The `.mlmodelc` directory.
    ///   - computeUnits: The Core ML compute units.
    init(contentsOf url: URL, computeUnits: MLComputeUnits) throws {
        let options = MLModelConfiguration()
        options.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: options)
        let description = model.modelDescription
        var configuration = SortformerConfiguration()
        if let metadata = description.metadata[.creatorDefinedKey] as? [String: String] {
            func value(_ key: String) -> Int? { metadata[key].flatMap { Int($0) } }
            configuration.chunkLength = value("chunk_len") ?? configuration.chunkLength
            configuration.leftContext = value("chunk_left_context") ?? configuration.leftContext
            configuration.rightContext =
                value("chunk_right_context") ?? configuration.rightContext
            configuration.fifoLength = value("fifo_len") ?? configuration.fifoLength
            configuration.speakerCacheLength =
                value("spkcache_len") ?? configuration.speakerCacheLength
            configuration.updatePeriod =
                value("spkcache_update_period") ?? configuration.updatePeriod
            configuration.subsampling = value("subsampling_factor") ?? configuration.subsampling
        }
        func shape(_ name: String, outputs: Bool = false) -> [Int]? {
            let descriptions =
                outputs ? description.outputDescriptionsByName : description.inputDescriptionsByName
            return descriptions[name]?.multiArrayConstraint?.shape.map(\.intValue)
        }
        let expected = [
            ("chunk", [1, configuration.chunkMelFrames, configuration.melBinCount]),
            ("spkcache", [1, configuration.speakerCacheLength, configuration.embeddingSize]),
            ("fifo", [1, configuration.fifoLength, configuration.embeddingSize]),
        ]
        for (name, dimensions) in expected where shape(name) != dimensions {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "The Sortformer model input \(name) does not have the shape \(dimensions).")
        }
        guard
            shape("speaker_preds", outputs: true)
                == [1, configuration.predictionRows, configuration.speakerCount],
            shape("chunk_pre_encoder_embs_out", outputs: true)
                == [1, configuration.chunkEmbeddingRows, configuration.embeddingSize]
        else {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "The Sortformer model outputs do not match its streaming configuration.")
        }
        self.configuration = configuration
        func array(_ shape: [Int], _ type: MLMultiArrayDataType) throws -> MLMultiArray {
            try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
        }
        chunk = try array(expected[0].1, .float32)
        speakerCache = try array(expected[1].1, .float32)
        fifo = try array(expected[2].1, .float32)
        chunkLength = try array([1], .int32)
        speakerCacheLength = try array([1], .int32)
        fifoLength = try array([1], .int32)
        provider = try MLDictionaryFeatureProvider(dictionary: [
            "chunk": MLFeatureValue(multiArray: chunk),
            "chunk_lengths": MLFeatureValue(multiArray: chunkLength),
            "spkcache": MLFeatureValue(multiArray: speakerCache),
            "spkcache_lengths": MLFeatureValue(multiArray: speakerCacheLength),
            "fifo": MLFeatureValue(multiArray: fifo),
            "fifo_lengths": MLFeatureValue(multiArray: fifoLength),
        ])
    }

    func predict(
        chunk features: [Float], chunkFrames: Int, speakerCache cache: [Float], fifo queue: [Float]
    )
        throws -> SortformerPrediction
    {
        let size = configuration.embeddingSize
        Self.fill(chunk, with: features)
        Self.fill(speakerCache, with: cache)
        Self.fill(fifo, with: queue)
        chunkLength[0] = NSNumber(value: Int32(chunkFrames))
        speakerCacheLength[0] = NSNumber(value: Int32(cache.count / size))
        fifoLength[0] = NSNumber(value: Int32(queue.count / size))
        let output = try model.prediction(from: provider)
        guard let predictions = output.featureValue(for: "speaker_preds")?.multiArrayValue,
            let embeddings = output.featureValue(for: "chunk_pre_encoder_embs_out")?
                .multiArrayValue,
            let lengths = output.featureValue(for: "chunk_pre_encoder_lengths_out")?
                .multiArrayValue
        else {
            throw SpeechRecognitionError.invalidResult(
                "The Sortformer model returned no predictions.")
        }
        let count = min(lengths[0].intValue, configuration.chunkEmbeddingRows)
        return SortformerPrediction(
            predictions: Self.floats(predictions), embeddings: Self.floats(embeddings, rows: count),
            embeddingCount: count)
    }

    /// Copies values into the start of a Float32 array and zeros the rest.
    private static func fill(_ array: MLMultiArray, with values: [Float]) {
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in
            let count = min(values.count, buffer.count)
            values.withUnsafeBufferPointer { source in
                buffer.baseAddress!.update(from: source.baseAddress!, count: count)
            }
            (buffer.baseAddress! + count).update(repeating: 0, count: buffer.count - count)
        }
    }

    /// Reads the leading rows of a `[1, rows, columns]` array as Float32, following its strides.
    private static func floats(_ array: MLMultiArray, rows: Int? = nil) -> [Float] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let rowCount = min(rows ?? shape[1], shape[1])
        let columns = shape[2]
        var values = [Float](repeating: 0, count: rowCount * columns)
        switch array.dataType {
        case .float16:
            #if arch(arm64)
                array.withUnsafeBufferPointer(ofType: Float16.self) { buffer in
                    for row in 0..<rowCount {
                        for column in 0..<columns {
                            values[row * columns + column] = Float(
                                buffer[row * strides[1] + column * strides[2]])
                        }
                    }
                }
            #else
                for row in 0..<rowCount {
                    for column in 0..<columns {
                        values[row * columns + column] =
                            array[[0, row, column] as [NSNumber]].floatValue
                    }
                }
            #endif
        default:
            array.withUnsafeBufferPointer(ofType: Float.self) { buffer in
                for row in 0..<rowCount {
                    for column in 0..<columns {
                        values[row * columns + column] =
                            buffer[row * strides[1] + column * strides[2]]
                    }
                }
            }
        }
        return values
    }
}
