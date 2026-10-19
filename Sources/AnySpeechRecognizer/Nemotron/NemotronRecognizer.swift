import Foundation

/// Recognizes 16 kHz mono audio incrementally with Nemotron's cache-aware streaming encoder.
///
/// The recognizer computes mel frames as audio arrives and encodes each chunk of
/// `8 × chunkFrames` mel frames once, with the encoder caches of the previous chunk, as NeMo's
/// cache-aware streaming inference does with `pad_and_drop_preencoded`. Greedy RNN-T decoding
/// continues across chunks with one prediction network context, so every token is final when
/// it is decoded: nothing is decoded twice, and updates report no volatile tokens.
///
/// Encoder frame `k` reads mel frames through `8k + 7`, so a token at frame `k` covers samples
/// `1280 k ..< 1280 (k + 1)`. A chunk is ready when the audio covers the window of its last mel
/// frame, 12.5 ms after the chunk's audio ends.
///
/// A finalization before a chunk is complete encodes the partial chunk without updating the
/// caches and decodes its frames through the finalized time. Its last mel frames read zeros
/// after the received audio, as at the end of input, because the analyzer finalizes a file
/// instead of ending its input. The tokens are final; when the chunk is complete, its encoding
/// updates the caches, and decoding resumes after the frames already decoded. At the end of input, the last partial chunk is encoded with its valid
/// length. The recognizer retains only the mel frames and audio that later chunks read.
/// One owner uses it at a time.
final class NemotronRecognizer {
    let vocabulary: ParakeetVocabulary
    let prompt: Int
    private let engine: any NemotronInferenceEngine
    private let decoder: ParakeetTDTDecoder
    private let frontend = NemotronMelFrontend()
    private let frameSamples: Int
    private let chunkMelFrames: Int
    private var input: [Float]
    private var encoded: ParakeetEncoderOutput
    private var context: ParakeetDecoderContext
    /// The first mel frame of the next chunk that updates the caches.
    private var chunkStart = 0
    /// Encoder frames before this one are decoded.
    private(set) var decodedFrame = 0
    private var ended = false

    /// The samples appended so far.
    var receivedSampleCount: Int { frontend.sampleCount }
    var committedThroughSample: Int { min(decodedFrame * frameSamples, receivedSampleCount) }

    init(engine: any NemotronInferenceEngine, vocabulary: ParakeetVocabulary, prompt: Int) throws {
        self.engine = engine
        self.vocabulary = vocabulary
        self.prompt = prompt
        frameSamples = engine.model.samplesPerEncoderFrame
        chunkMelFrames = NemotronModel.subsampling * engine.chunkFrames
        input = [Float](repeating: 0, count: engine.model.melBinCount * engine.inputFrames)
        encoded = ParakeetEncoderOutput(
            hiddenSize: engine.model.encoderHiddenSize, frameCount: engine.chunkFrames)
        decoder = ParakeetTDTDecoder(maxSymbolsPerFrame: NemotronModel.maxSymbolsPerFrame)
        engine.resetCaches()
        context = try .initial(engine: engine)
    }

    /// Appends 16 kHz mono samples. Appending after the end of input is a programming error.
    func append(_ samples: UnsafeBufferPointer<Float>) {
        precondition(!ended, "Cannot append audio after the end of input.")
        frontend.append(samples)
    }

    func append(_ samples: [Float]) {
        samples.withUnsafeBufferPointer { append($0) }
    }

    /// Encodes and decodes every complete chunk.
    ///
    /// - Parameters:
    ///   - finalizeThrough: A sample through which to decode even before its chunk is complete.
    ///     The recognizer decodes whole encoder frames, through at most the received audio.
    ///   - endOfInput: Whether to decode all received audio. Later appends are not allowed.
    func process(finalizeThrough: Int? = nil, endOfInput: Bool = false) throws -> ParakeetUpdate {
        var update = ParakeetUpdate(committedThroughSample: 0)
        if endOfInput, !ended {
            ended = true
            frontend.finish()
        }
        while frontend.frameCount >= chunkStart + chunkMelFrames {
            update.committed += try runChunk(validMelFrames: chunkMelFrames, updatesCaches: true)
            chunkStart += chunkMelFrames
            frontend.discard(before: chunkStart - NemotronModel.preEncodeFrames)
        }
        let pending = frontend.frameCount - chunkStart
        if ended, pending > 0 {
            update.committed += try runChunk(validMelFrames: pending, updatesCaches: true)
            chunkStart += chunkMelFrames
        } else if let finalizeThrough,
            (max(0, finalizeThrough) + frameSamples - 1) / frameSamples > decodedFrame
        {
            // Encode the received audio as if it ended, as the end of input would.
            let target = (max(0, finalizeThrough) + frameSamples - 1) / frameSamples
            update.committed += try frontend.withProvisionalEnd {
                let pending = min(chunkMelFrames, frontend.frameCount - chunkStart)
                guard pending > 0 else { return [] }
                return try runChunk(validMelFrames: pending, updatesCaches: false, through: target)
            }
        }
        update.committedThroughSample = committedThroughSample
        return update
    }

    /// Encodes the chunk at ``chunkStart`` and decodes its frames after ``decodedFrame``.
    ///
    /// - Parameter through: The encoder frame before which to stop decoding, or `nil` for
    ///   every valid frame.
    private func runChunk(validMelFrames: Int, updatesCaches: Bool, through: Int? = nil) throws
        -> [ParakeetToken]
    {
        let width = engine.inputFrames
        let first = chunkStart - NemotronModel.preEncodeFrames
        input.withUnsafeMutableBufferPointer { input in
            input.update(repeating: 0)
            frontend.copy(
                first..<(chunkStart + validMelFrames), into: input, width: width, column: 0)
        }
        try input.withUnsafeBufferPointer {
            try engine.encode(
                mel: $0, validFrameCount: NemotronModel.preEncodeFrames + validMelFrames,
                prompt: prompt, updatesCaches: updatesCaches, into: &encoded)
        }
        let chunkFrame = chunkStart / NemotronModel.subsampling
        let end = min(encoded.validFrameCount, (through ?? .max) - chunkFrame)
        let start = max(0, decodedFrame - chunkFrame)
        guard end > start else { return [] }
        var decoded: [ParakeetDecodedToken] = []
        _ = try decoder.decode(
            encoded, from: start, to: end, context: &context, engine: engine,
            frameOffset: chunkFrame, into: &decoded)
        decodedFrame = chunkFrame + end
        return decoded.map(token)
    }

    private func token(_ decoded: ParakeetDecodedToken) -> ParakeetToken {
        let start = decoded.frame * frameSamples
        let end = min((decoded.frame + 1) * frameSamples, receivedSampleCount)
        return ParakeetToken(
            id: decoded.id, piece: vocabulary.piece(for: decoded.id), startSample: start,
            endSample: max(end, start + 1), probability: decoded.probability)
    }
}
