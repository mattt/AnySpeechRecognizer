import Foundation

/// The prediction network context that greedy decoding carries between frames and windows.
struct ParakeetDecoderContext: Sendable, Equatable {
    /// The LSTM state after `lastToken`.
    var state: ParakeetDecoderState
    /// The prediction network output for `lastToken`.
    var output: [Float]
    /// The last emitted token, or `nil` before the first token.
    var lastToken: Int?

    /// Starts the prediction network with the blank token from a zero state, as NeMo does.
    static func initial(engine: any ParakeetInferenceEngine) throws -> Self {
        var state = ParakeetDecoderState(zeroFor: engine.model)
        var output: [Float] = []
        try engine.predict(token: engine.model.blankID, state: &state, output: &output)
        return Self(state: state, output: output, lastToken: nil)
    }
}

/// A token from greedy decoding.
struct ParakeetDecodedToken: Sendable, Equatable {
    var id: Int
    /// The encoder frame at which the joint network emitted the token.
    var frame: Int
    /// The predicted duration in encoder frames.
    var duration: Int
    var probability: Float
}

/// The result of decoding one encoder window up to and beyond a commit frame.
struct ParakeetWindowDecoding: Sendable {
    /// Tokens emitted before the commit frame.
    var committed: [ParakeetDecodedToken] = []
    /// Tokens emitted from the commit frame to the end of the decoded range.
    var lookahead: [ParakeetDecodedToken] = []
    /// The context after the committed tokens; lookahead tokens do not affect it.
    var context: ParakeetDecoderContext
    /// The frame where decoding resumes, relative to the commit frame.
    /// A duration that crosses the commit frame makes it positive.
    var overshoot: Int
}

/// Greedy token-and-duration transducer decoding.
///
/// Decoding follows NeMo's `GreedyTDTInfer`: the joint network evaluates the current
/// frame with the prediction network output for the last token. A blank token advances by
/// its duration, at least one frame, without updating the prediction network. Another token
/// is emitted at the current frame, updates the prediction network, and advances by its
/// duration. After `maxSymbolsPerFrame` tokens at one frame, decoding advances one frame,
/// and with ``discardsFramesAtSymbolLimit`` it first discards those tokens.
/// The duration classes map to frames through ``ParakeetModelDescription/durations``,
/// `[0, 1, 2, 3, 4]` for v3, as FluidAudio's `TdtDecoderV3` maps them.
struct ParakeetTDTDecoder {
    var maxSymbolsPerFrame = 10
    /// Whether the tokens of a frame that reaches `maxSymbolsPerFrame` are discarded, and the
    /// context returns to the one before them, instead of being kept.
    ///
    /// Parakeet rarely emits more than one token at a frame of speech. Where it emitted
    /// repeated filler such as "and the other thing and the other thing", it emitted the
    /// tokens with zero duration, up to the limit at each of a few frames.
    var discardsFramesAtSymbolLimit = false
    /// Contexts that replace the context after a sentence-final token, keyed by the token.
    ///
    /// Each holds only the blank start token and the sentence-final token. After a long
    /// sentence and its period, the prediction network can treat the utterance as finished
    /// and the decoder can emit only blanks for many seconds of following speech.
    /// The sentence-final token alone keeps the punctuated style without that history.
    var sentenceEndContexts: [Int: ParakeetDecoderContext] = [:]

    /// Decodes `encoder` frames from `start` while the frame index is before `end`.
    ///
    /// - Parameters:
    ///   - frameOffset: An offset added to the frame of each token.
    ///   - previousTokenEnd: The frame where the duration of the token before `start` ends,
    ///     where greedy decoding would continue after that token. Blanks before this frame
    ///     advance at most to it: greedy decoding never evaluates the frames inside a token's
    ///     duration, and a blank there can predict a duration that skips the next token.
    /// - Returns: The frame index where decoding stops, `end` or later.
    func decode(
        _ encoder: ParakeetEncoderOutput, from start: Int, to end: Int,
        context: inout ParakeetDecoderContext, engine: any ParakeetInferenceEngine,
        frameOffset: Int = 0, previousTokenEnd: Int = 0,
        into tokens: inout [ParakeetDecodedToken]
    ) throws -> Int {
        precondition(start >= 0 && end <= encoder.frameCount)
        let blank = engine.model.blankID
        var frame = start
        var symbols = 0
        // The context before the first token at the current frame.
        var frameContext = context
        while frame < end {
            let decision = try encoder.withFrame(frame) { features in
                try context.output.withUnsafeBufferPointer { output in
                    try engine.joint(encoderFrame: features, decoderOutput: output)
                }
            }
            if decision.token == blank {
                let next = frame + max(1, decision.duration)
                frame = frame < previousTokenEnd ? min(next, previousTokenEnd) : next
                symbols = 0
                continue
            }
            if discardsFramesAtSymbolLimit, symbols == 0 { frameContext = context }
            tokens.append(
                ParakeetDecodedToken(
                    id: decision.token, frame: frame + frameOffset, duration: decision.duration,
                    probability: decision.probability))
            if let reset = sentenceEndContexts[decision.token] {
                context = reset
            } else {
                try engine.predict(
                    token: decision.token, state: &context.state, output: &context.output)
                context.lastToken = decision.token
            }
            symbols += 1
            if decision.duration > 0 {
                frame += decision.duration
                symbols = 0
            } else if symbols >= maxSymbolsPerFrame {
                if discardsFramesAtSymbolLimit {
                    tokens.removeLast(symbols)
                    context = frameContext
                }
                frame += 1
                symbols = 0
            }
        }
        return frame
    }

    /// Decodes a window in two parts and keeps the commit-boundary context separate.
    ///
    /// The decoder runs from `start` to `commit` and records the context there.
    /// When `end` is after the frame where committed decoding stopped, it continues
    /// to `end` with a copy of that context.
    /// Blanks before `previousTokenEnd` advance at most to it, as in `decode`.
    func decodeWindow(
        _ encoder: ParakeetEncoderOutput, from start: Int, commit: Int, end: Int,
        context: ParakeetDecoderContext, engine: any ParakeetInferenceEngine, frameOffset: Int,
        previousTokenEnd: Int = 0
    ) throws -> ParakeetWindowDecoding {
        var committedContext = context
        var result = ParakeetWindowDecoding(context: context, overshoot: 0)
        let resume = try decode(
            encoder, from: start, to: commit, context: &committedContext, engine: engine,
            frameOffset: frameOffset, previousTokenEnd: previousTokenEnd,
            into: &result.committed)
        result.context = committedContext
        result.overshoot = resume - commit
        if end > resume {
            var lookaheadContext = committedContext
            _ = try decode(
                encoder, from: resume, to: end, context: &lookaheadContext,
                engine: engine, frameOffset: frameOffset, previousTokenEnd: previousTokenEnd,
                into: &result.lookahead)
        }
        return result
    }
}
