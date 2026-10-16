import Foundation
import Testing

@testable import AnySpeechRecognizer

/// An engine whose encoder frames carry the joint decision: `[token, duration]`, with -1 for blank.
private final class ScriptedEngine: ParakeetInferenceEngine {
    let model = ParakeetModelDescription(
        encoderHiddenSize: 2, decoderHiddenSize: 1, decoderLayerCount: 1, vocabularySize: 10)

    func encode(
        mel: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: inout ParakeetEncoderOutput
    ) throws {}

    /// The state counts consumed tokens; the output is the last token.
    func predict(token: Int, state: inout ParakeetDecoderState, output: inout [Float]) throws {
        state.hidden[0] += 1
        state.cell[0] = Float(token)
        output = [Float(token)]
    }

    func joint(
        encoderFrame: UnsafeBufferPointer<Float>, decoderOutput: UnsafeBufferPointer<Float>
    ) throws -> ParakeetJointDecision {
        let token = Int(encoderFrame[0])
        return ParakeetJointDecision(
            token: token < 0 ? model.blankID : token, probability: 0.5,
            duration: Int(encoderFrame[1]))
    }

    static func frames(_ script: [(token: Int, duration: Int)]) -> ParakeetEncoderOutput {
        var output = ParakeetEncoderOutput(hiddenSize: 2, frameCount: script.count)
        output.validFrameCount = script.count
        for (index, step) in script.enumerated() {
            output.features[index * 2] = Float(step.token)
            output.features[index * 2 + 1] = Float(step.duration)
        }
        return output
    }
}

struct ParakeetTDTDecoderTests {
    private let engine = ScriptedEngine()

    @Test func startsWithTheBlankToken() throws {
        let context = try ParakeetDecoderContext.initial(engine: engine)
        #expect(context.lastToken == nil)
        #expect(context.output == [10])
        #expect(context.state.hidden == [1])
    }

    @Test func advancesByDurationsAndAtLeastOneFrameForBlanks() throws {
        let encoder = ScriptedEngine.frames([
            (1, 2), (-1, 0), (2, 1), (-1, 3), (-1, 0), (-1, 0), (3, 4), (-1, 1),
        ])
        var context = try ParakeetDecoderContext.initial(engine: engine)
        var tokens: [ParakeetDecodedToken] = []
        let stop = try ParakeetTDTDecoder().decode(
            encoder, from: 0, to: 8, context: &context, engine: engine, frameOffset: 100,
            into: &tokens)
        // Frame 1 is skipped by the first duration; frame 3 skips frames 4 and 5.
        #expect(tokens.map(\.id) == [1, 2, 3])
        #expect(tokens.map(\.frame) == [100, 102, 106])
        #expect(tokens.map(\.duration) == [2, 1, 4])
        #expect(stop == 10)
        #expect(context.lastToken == 3)
        #expect(context.state.hidden == [4])
    }

    @Test func blanksDoNotSkipThePreviousTokenEnd() throws {
        // Decoding starts inside the previous token's duration, which ends at frame 2.
        let encoder = ScriptedEngine.frames([(-1, 4), (-1, 4), (5, 2), (-1, 1), (6, 1), (-1, 1)])
        var context = try ParakeetDecoderContext.initial(engine: engine)
        var tokens: [ParakeetDecodedToken] = []
        _ = try ParakeetTDTDecoder().decode(
            encoder, from: 0, to: 6, context: &context, engine: engine, previousTokenEnd: 2,
            into: &tokens)
        #expect(tokens.map(\.id) == [5, 6])
        #expect(tokens.map(\.frame) == [2, 4])
        // Without the limit, the first blank skips token 5.
        context = try ParakeetDecoderContext.initial(engine: engine)
        tokens = []
        _ = try ParakeetTDTDecoder().decode(
            encoder, from: 0, to: 6, context: &context, engine: engine, into: &tokens)
        #expect(tokens.map(\.id) == [6])
    }

    @Test func limitsSymbolsPerFrame() throws {
        let encoder = ScriptedEngine.frames([(5, 0), (-1, 1), (6, 0)])
        var context = try ParakeetDecoderContext.initial(engine: engine)
        var tokens: [ParakeetDecodedToken] = []
        let stop = try ParakeetTDTDecoder(maxSymbolsPerFrame: 3).decode(
            encoder, from: 0, to: 3, context: &context, engine: engine, into: &tokens)
        #expect(tokens.map(\.id) == [5, 5, 5, 6, 6, 6])
        #expect(tokens.map(\.frame) == [0, 0, 0, 2, 2, 2])
        #expect(stop == 3)
    }

    /// Parakeet emitted repeated filler such as "and the other thing and the other thing"
    /// as zero-duration tokens, up to the symbol limit at each of a few frames.
    @Test func discardsFramesAtTheSymbolLimit() throws {
        let encoder = ScriptedEngine.frames([(1, 1), (5, 0), (7, 0), (2, 1), (3, 1), (-1, 1)])
        var decoder = ParakeetTDTDecoder(maxSymbolsPerFrame: 3)
        decoder.discardsFramesAtSymbolLimit = true
        var context = try ParakeetDecoderContext.initial(engine: engine)
        var tokens: [ParakeetDecodedToken] = []
        let stop = try decoder.decode(
            encoder, from: 0, to: 6, context: &context, engine: engine, into: &tokens)
        // Frames 1 and 2 reach the limit, and the context returns to the one after token 1.
        #expect(tokens.map(\.id) == [1, 2, 3])
        #expect(tokens.map(\.frame) == [0, 3, 4])
        #expect(stop == 6)
        var expected = try ParakeetDecoderContext.initial(engine: engine)
        for token in [1, 2, 3] {
            try engine.predict(token: token, state: &expected.state, output: &expected.output)
            expected.lastToken = token
        }
        #expect(context == expected)
    }

    @Test func keepsTheCommitContextFreeOfLookaheadTokens() throws {
        let encoder = ScriptedEngine.frames([
            (1, 1), (-1, 1), (2, 3), (-1, 1), (-1, 1), (3, 1), (4, 1), (-1, 1),
        ])
        let initial = try ParakeetDecoderContext.initial(engine: engine)
        let decoding = try ParakeetTDTDecoder().decodeWindow(
            encoder, from: 0, commit: 4, end: 8, context: initial, engine: engine, frameOffset: 0)
        #expect(decoding.committed.map(\.id) == [1, 2])
        // The duration of token 2 crosses the commit frame.
        #expect(decoding.overshoot == 1)
        #expect(decoding.lookahead.map(\.id) == [3, 4])
        #expect(decoding.lookahead.map(\.frame) == [5, 6])
        var expected = initial
        var tokens: [ParakeetDecodedToken] = []
        _ = try ParakeetTDTDecoder().decode(
            encoder, from: 0, to: 4, context: &expected, engine: engine, into: &tokens)
        #expect(decoding.context == expected)
        #expect(decoding.context.lastToken == 2)
    }

    @Test func restartsTheContextAfterASentenceEnds() throws {
        var restarted = try ParakeetDecoderContext.initial(engine: engine)
        try engine.predict(token: 6, state: &restarted.state, output: &restarted.output)
        restarted.lastToken = 6
        var decoder = ParakeetTDTDecoder()
        decoder.sentenceEndContexts = [6: restarted]
        let encoder = ScriptedEngine.frames([(1, 1), (2, 1), (6, 1), (3, 1), (-1, 1)])
        var context = try ParakeetDecoderContext.initial(engine: engine)
        var tokens: [ParakeetDecodedToken] = []
        _ = try decoder.decode(
            encoder, from: 0, to: 3, context: &context, engine: engine, into: &tokens)
        // Tokens 1 and 2 no longer affect the context after token 6.
        #expect(context == restarted)
        _ = try decoder.decode(
            encoder, from: 3, to: 5, context: &context, engine: engine, into: &tokens)
        #expect(tokens.map(\.id) == [1, 2, 6, 3])
        #expect(context.state.hidden == [restarted.state.hidden[0] + 1])
        #expect(context.lastToken == 3)
    }

    @Test func reportsOvershootWhenDecodingStartsAfterTheCommitFrame() throws {
        let encoder = ScriptedEngine.frames([(-1, 1), (-1, 1), (-1, 1), (7, 1), (-1, 1)])
        let decoding = try ParakeetTDTDecoder().decodeWindow(
            encoder, from: 3, commit: 2, end: 5,
            context: try .initial(engine: engine), engine: engine, frameOffset: 0)
        #expect(decoding.committed.isEmpty)
        #expect(decoding.overshoot == 1)
        #expect(decoding.lookahead.map(\.id) == [7])
    }
}
