import Foundation

/// The tensor contract of Nemotron 3.5 ASR Streaming.
enum NemotronModel {
    /// The transducer: 13,087 tokens and the blank, a two-layer LSTM prediction network, and
    /// RNN-T decoding, which ``ParakeetTDTDecoder`` runs with the single duration 0.
    static let transducer = ParakeetModelDescription(
        windowSamples: 0, melBinCount: 128, melFrameCount: 0, encoderFrameCount: 0,
        samplesPerEncoderFrame: 1280, encoderHiddenSize: 1024, decoderHiddenSize: 640,
        decoderLayerCount: 2, vocabularySize: 13_087, durations: [0])
    /// Mel frames per encoder frame.
    static let subsampling = 8
    /// The mel frames before each chunk that the causal subsampling reads again.
    static let preEncodeFrames = 9
    static let promptCount = 128
    /// Tokens that NeMo's greedy decoding emits at most at one frame.
    static let maxSymbolsPerFrame = 10
}

/// Runs the cache-aware encoder, prediction network, and joint network of Nemotron.
///
/// The encoder keeps the attention and convolution caches of every layer between steps.
/// Each step encodes the mel frames of one chunk with the ``NemotronModel/preEncodeFrames``
/// frames before it and produces ``chunkFrames`` encoder frames, which already include the
/// language prompt projection. An engine keeps reusable buffers, so one owner uses it at a time.
protocol NemotronInferenceEngine: ParakeetTransducer {
    /// Encoder frames per chunk.
    var chunkFrames: Int { get }

    /// Clears the caches to start a new stream.
    func resetCaches()

    /// Encodes one chunk.
    ///
    /// - Parameters:
    ///   - mel: Features in mel-major order, `[128 × (9 + 8 chunkFrames)]`: the 9 frames
    ///     before the chunk, zero before the stream, then the chunk.
    ///   - validFrameCount: The mel frames that contain audio, including the first 9; fewer
    ///     than all only for the last chunk of a stream.
    ///   - prompt: The language prompt slot.
    ///   - updatesCaches: Whether the next step continues from this step. A step that does
    ///     not update the caches encodes audio that a later step encodes again.
    ///   - output: Receives the encoder frames and their valid count.
    func encode(
        mel: UnsafeBufferPointer<Float>, validFrameCount: Int, prompt: Int, updatesCaches: Bool,
        into output: inout ParakeetEncoderOutput
    ) throws
}

extension NemotronInferenceEngine {
    /// Mel frames of one encoder input.
    var inputFrames: Int { NemotronModel.preEncodeFrames + NemotronModel.subsampling * chunkFrames }
}
