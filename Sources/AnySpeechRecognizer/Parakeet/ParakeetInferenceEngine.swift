import Foundation

/// The tensor contract of a Parakeet TDT model.
struct ParakeetModelDescription: Sendable, Equatable {
    /// The audio samples in one encoder window, 15 seconds for the published encoder.
    /// ``CoreMLParakeetEngine`` derives the window from the encoder it loads.
    var windowSamples = 240_000
    /// The mel bins per frame.
    var melBinCount = 128
    /// The mel frames in one encoder window.
    var melFrameCount = 1501
    /// The encoder frames in one encoder window.
    var encoderFrameCount = 188
    /// The audio samples per encoder frame.
    var samplesPerEncoderFrame = 1280
    var encoderHiddenSize = 1024
    var decoderHiddenSize = 640
    var decoderLayerCount = 2
    /// The token count before the blank token.
    var vocabularySize = 8192
    /// Encoder frames to advance for each duration class of the joint network.
    var durations = [0, 1, 2, 3, 4]

    /// Parakeet TDT 0.6B v3 and models with the same contract.
    static let v3 = ParakeetModelDescription()

    /// The blank token, which also starts the prediction network.
    var blankID: Int { vocabularySize }
    /// Encoder frames whose audio fits completely in one window.
    var windowFrames: Int { windowSamples / samplesPerEncoderFrame }

    /// The same model with an encoder for a different fixed window.
    ///
    /// The window follows from the encoder's mel input length: `melFrameCount - 1` hops of
    /// ``ParakeetMelFrontend/hopLength`` samples, as the mel frontend centers its frames.
    func withEncoderWindow(melFrameCount: Int, encoderFrameCount: Int) -> Self {
        var model = self
        model.melFrameCount = melFrameCount
        model.encoderFrameCount = encoderFrameCount
        model.windowSamples = (melFrameCount - 1) * ParakeetMelFrontend.hopLength
        return model
    }
    /// The LSTM values in each of the hidden and cell states.
    var decoderStateSize: Int { decoderLayerCount * decoderHiddenSize }
}

/// The LSTM state of the prediction network.
struct ParakeetDecoderState: Sendable, Equatable {
    /// The hidden state, `[layers × hidden]`.
    var hidden: [Float]
    /// The cell state, `[layers × hidden]`.
    var cell: [Float]

    /// A zero state for a model.
    init(zeroFor model: ParakeetModelDescription) {
        hidden = [Float](repeating: 0, count: model.decoderStateSize)
        cell = hidden
    }
}

/// Encoder features for one window, stored frame by frame.
struct ParakeetEncoderOutput: Sendable {
    /// The features, `[frameCount × hiddenSize]`.
    var features: [Float]
    var hiddenSize: Int
    /// The frames in `features`.
    var frameCount: Int
    /// The frames that the encoder reports as valid.
    var validFrameCount: Int

    init(hiddenSize: Int, frameCount: Int) {
        features = [Float](repeating: 0, count: hiddenSize * frameCount)
        self.hiddenSize = hiddenSize
        self.frameCount = frameCount
        validFrameCount = 0
    }

    func withFrame<Result>(
        _ index: Int, _ body: (UnsafeBufferPointer<Float>) throws -> Result
    ) rethrows -> Result {
        precondition(index >= 0 && index < frameCount)
        return try features.withUnsafeBufferPointer {
            try body(
                UnsafeBufferPointer(rebasing: $0[(index * hiddenSize)..<((index + 1) * hiddenSize)])
            )
        }
    }
}

/// The greedy decision of the joint network for one encoder frame.
struct ParakeetJointDecision: Sendable, Equatable {
    /// The most likely token, or the blank token.
    var token: Int
    /// The probability of `token` over the vocabulary and blank, in `0...1`.
    var probability: Float
    /// The encoder frames to advance.
    var duration: Int
}

/// Runs the neural networks of a Parakeet TDT model.
///
/// Implementations convert their outputs to plain Swift values, so the decoder does not
/// depend on a framework's tensor types. An engine that produces raw logits takes the
/// argmax of the token and duration logits and a softmax probability for the token.
/// An engine can keep reusable buffers, so one owner uses it at a time.
protocol ParakeetInferenceEngine: AnyObject {
    var model: ParakeetModelDescription { get }
    /// The contract of a longer-window encoder for windows that commit without right
    /// context, or `nil` when the engine has only one encoder.
    var forcedCommitModel: ParakeetModelDescription? { get }

    /// Encodes one window of mel features.
    ///
    /// - Parameters:
    ///   - mel: Features in mel-major order, `[melBinCount × melFrameCount]`.
    ///   - validFrameCount: The mel frames that contain audio.
    ///   - output: Receives the features and the valid encoder frame count.
    func encode(
        mel: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: inout ParakeetEncoderOutput
    ) throws

    /// Encodes one window of mel features with the encoder of ``forcedCommitModel``.
    func encodeForcedCommit(
        mel: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: inout ParakeetEncoderOutput
    ) throws

    /// Advances the prediction network by one token.
    ///
    /// - Parameters:
    ///   - token: The previous token, or the blank token to start.
    ///   - state: The LSTM state before `token`; receives the state after it.
    ///   - output: Receives the prediction network output, `decoderHiddenSize` values.
    func predict(token: Int, state: inout ParakeetDecoderState, output: inout [Float]) throws

    /// Evaluates the joint network for one encoder frame and prediction network output.
    func joint(
        encoderFrame: UnsafeBufferPointer<Float>, decoderOutput: UnsafeBufferPointer<Float>
    ) throws -> ParakeetJointDecision
}

extension ParakeetInferenceEngine {
    var forcedCommitModel: ParakeetModelDescription? { nil }

    func encodeForcedCommit(
        mel: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: inout ParakeetEncoderOutput
    ) throws {
        throw SpeechRecognitionError.invalidResult("The Parakeet engine has one encoder.")
    }
}
