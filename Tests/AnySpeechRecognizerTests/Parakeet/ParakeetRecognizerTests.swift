import Foundation
import Testing

@testable import AnySpeechRecognizer

/// Recognizer bookkeeping with tone "speech"; see ``FakeParakeetEngine``.
/// A fake window holds 24 whole encoder frames.
struct ParakeetRecognizerTests {
    static let file = ParakeetRecognizer.Configuration(
        leftContextFrames: 4, chunkFrames: .max, rightContextFrames: 4,
        reportsVolatileResults: false, boundaryMarginFrames: 2, decoderHistoryFrames: 6)
    static let live = ParakeetRecognizer.Configuration(
        leftContextFrames: 10, chunkFrames: 4, rightContextFrames: 4,
        reportsVolatileResults: true, boundaryMarginFrames: 2, decoderHistoryFrames: 6)

    struct Run {
        var updates: [ParakeetUpdate] = []
        var received: [Int] = []
        var retained: [Int] = []
        var committed: [ParakeetToken] { updates.flatMap(\.committed) }
    }

    func run(
        _ recognizer: ParakeetRecognizer, _ samples: [Float], block: Int, endOfInput: Bool = true
    ) throws -> Run {
        var run = Run()
        var offset = 0
        while offset < samples.count {
            let end = min(samples.count, offset + block)
            recognizer.append(Array(samples[offset..<end]))
            offset = end
            run.updates.append(
                try recognizer.process(endOfInput: endOfInput && offset == samples.count))
            run.received.append(offset)
            run.retained.append(recognizer.retainedSampleCount)
        }
        return run
    }

    func expectTokens(
        _ tokens: [ParakeetToken], _ expected: [(id: Int, frame: Int)],
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(tokens.map(\.id) == expected.map(\.id), sourceLocation: sourceLocation)
        #expect(
            tokens.map(\.startSample) == expected.map { $0.frame * 1280 },
            sourceLocation: sourceLocation)
    }

    @Test(arguments: [(1, 16_000), (2, 1_600), (3, 777), (1, 1_000_000)])
    func fileModeCommitsEachTokenOnce(duration: Int, block: Int) throws {
        let script = ToneSpeech.script(count: 40)
        let engine = FakeParakeetEngine(tokenDuration: duration)
        let recognizer = try ParakeetRecognizer(
            engine: engine, vocabulary: ToneSpeech.vocabulary, configuration: Self.file)
        let run = try run(recognizer, script.samples, block: block)
        expectTokens(run.committed, script.tokens)
        #expect(run.updates.allSatisfy { $0.volatile.isEmpty })
        #expect(run.updates.last?.committedThroughSample == script.samples.count)
        #expect(recognizer.committedThroughSample == script.samples.count)
        // About 207 frames in chunks of at most 16 frames.
        #expect(engine.encodeCount >= 13)
        #expect(run.committed.allSatisfy { $0.endSample > $0.startSample && $0.piece != "" })
    }

    @Test(arguments: [1, 3])
    func liveModeReplacesVolatileTokens(duration: Int) throws {
        let script = ToneSpeech.script(count: 30)
        let recognizer = try ParakeetRecognizer(
            engine: FakeParakeetEngine(tokenDuration: duration),
            vocabulary: ToneSpeech.vocabulary, configuration: Self.live)
        let run = try run(recognizer, script.samples, block: 640)
        expectTokens(run.committed, script.tokens)
        var previousThrough = 0
        var firstVolatile: [Int: Int] = [:]
        for (index, update) in run.updates.enumerated() {
            #expect(update.committedThroughSample >= previousThrough)
            previousThrough = update.committedThroughSample
            #expect(update.committed.allSatisfy { $0.startSample < update.committedThroughSample })
            #expect(update.volatile.allSatisfy { $0.startSample >= update.committedThroughSample })
            #expect(update.volatile.allSatisfy { $0.startSample < run.received[index] })
            for token in update.volatile where firstVolatile[token.startSample] == nil {
                firstVolatile[token.startSample] = index
            }
        }
        #expect(run.updates.last?.volatile.isEmpty == true)
        // Volatile results show most tokens before they are committed.
        var commitIndex: [Int: Int] = [:]
        for (index, update) in run.updates.enumerated() {
            for token in update.committed { commitIndex[token.startSample] = index }
        }
        let early = commitIndex.filter { entry in
            firstVolatile[entry.key].map { $0 < entry.value } ?? false
        }
        #expect(early.count >= script.tokens.count - 2)
        // Volatile tokens agree with the tokens committed later.
        for update in run.updates {
            for token in update.volatile {
                if let committed = run.committed.first(where: {
                    $0.startSample == token.startSample
                }) {
                    #expect(committed.id == token.id)
                }
            }
        }
    }

    @Test func finalizeThroughCommitsWithoutRightContext() throws {
        let script = ToneSpeech.script(count: 12)
        let recognizer = try ParakeetRecognizer(
            engine: FakeParakeetEngine(), vocabulary: ToneSpeech.vocabulary,
            configuration: Self.live)
        let split = 30 * 1280 + 300
        let first = try run(
            recognizer, Array(script.samples[..<split]), block: 640, endOfInput: false)
        #expect(recognizer.committedThroughSample < split)
        let finalized = try recognizer.process(finalizeThrough: split)
        #expect(finalized.committedThroughSample == split)
        #expect(finalized.volatile.isEmpty)
        let early = first.committed + finalized.committed
        expectTokens(early, script.tokens.filter { $0.frame < 31 })
        // A later target is limited to the received audio.
        #expect(
            try recognizer.process(finalizeThrough: split + 10_000).committedThroughSample == split)
        let rest = try run(recognizer, Array(script.samples[split...]), block: 640)
        expectTokens(early + rest.committed, script.tokens)
    }

    @Test func endOfInputCommitsShortAudio() throws {
        let script = ToneSpeech.script(count: 2, trailing: 1)
        let recognizer = try ParakeetRecognizer(
            engine: FakeParakeetEngine(), vocabulary: ToneSpeech.vocabulary,
            configuration: Self.live)
        recognizer.append(Array(script.samples.dropLast(500)))
        let update = try recognizer.process(endOfInput: true)
        expectTokens(update.committed, script.tokens)
        #expect(update.volatile.isEmpty)
        #expect(update.committedThroughSample == script.samples.count - 500)
        #expect(try recognizer.process(endOfInput: true).committed.isEmpty)
    }

    @Test func emptyInputProducesNoTokens() throws {
        let recognizer = try ParakeetRecognizer(
            engine: FakeParakeetEngine(), vocabulary: ToneSpeech.vocabulary,
            configuration: Self.file)
        #expect(
            try recognizer.process(endOfInput: true) == ParakeetUpdate(committedThroughSample: 0))
    }

    @Test(arguments: [false, true])
    func retainsBoundedAudio(live: Bool) throws {
        let script = ToneSpeech.script(count: 200)
        let configuration = live ? Self.live : Self.file
        let recognizer = try ParakeetRecognizer(
            engine: FakeParakeetEngine(), vocabulary: ToneSpeech.vocabulary,
            configuration: configuration)
        let block = 1_600
        let run = try run(recognizer, script.samples, block: block)
        expectTokens(run.committed, script.tokens)
        // One window before the commit frame, plus a chunk, right context, margin, and a block.
        let limit = (24 + 24 + 2) * 1280 + block
        #expect(run.retained.max()! <= limit)
        #expect(script.samples.count > 4 * limit)
    }

    /// A window resumes after the last committed token, inside its duration, at frames that
    /// greedy decoding skips. Parakeet Ultra predicted a four-frame blank at such a frame and
    /// dropped the next three words. Here each frame inside a tone predicts such a blank.
    @Test(arguments: [false, true])
    func blanksDoNotSkipTheEndOfTheLastCommittedToken(live: Bool) throws {
        let script = ToneSpeech.script(count: 40, spacing: 3)
        var configuration = live ? Self.live : Self.file
        // A chunk longer than the blank, so the blank does not end at the commit frame.
        if live { configuration.chunkFrames = 8 }
        let recognizer = try ParakeetRecognizer(
            engine: FakeParakeetEngine(tokenDuration: 3, innerBlankDuration: 4),
            vocabulary: ToneSpeech.vocabulary, configuration: configuration)
        let run = try run(recognizer, script.samples, block: 640)
        expectTokens(run.committed, script.tokens)
    }

    /// On LibriSpeech, some window placements made Parakeet emit only blanks after a sentence
    /// end, for the rest of the window. Here a window in which the period starts after its
    /// first frame encodes the audio after the period as silence, and a window that starts at
    /// the gap after the period encodes it as speech.
    @Test(arguments: [false, true])
    func recoversWindowsDecodedAsBlanksAfterASentenceEnd(live: Bool) throws {
        let script = ToneSpeech.script(count: 40, spacing: 4)
        #expect(script.tokens.filter { $0.id == ToneSpeech.sentenceEnd }.count == 4)
        var configuration = live ? Self.live : Self.file
        // A gap shorter than a live chunk, so the window after the period recovers it before
        // the next token's audio is committed. Tokens three frames long at a spacing of four
        // leave gaps of one frame.
        configuration.blankRecoveryFrames = 3
        let engine = FakeParakeetEngine(tokenDuration: 3, blanksAfterSentenceEnd: true)
        let recognizer = try ParakeetRecognizer(
            engine: engine, vocabulary: ToneSpeech.vocabulary, configuration: configuration)
        expectTokens(try run(recognizer, script.samples, block: 640).committed, script.tokens)

        configuration.blankRecoveryFrames = nil
        let unrecovered = try ParakeetRecognizer(
            engine: FakeParakeetEngine(tokenDuration: 3, blanksAfterSentenceEnd: true),
            vocabulary: ToneSpeech.vocabulary, configuration: configuration)
        let committed = try run(unrecovered, script.samples, block: 640).committed
        #expect(committed.count < script.tokens.count)
    }

    /// On LibriSpeech, windows that started inside speech lost words far more often than
    /// windows that started in a pause. Here a window that starts inside a tone encodes all
    /// its audio as silence, and windows start at the quietest frame of their range.
    @Test func fileWindowsStartAtTheQuietestFrame() throws {
        let script = ToneSpeech.script(count: 40)
        var configuration = Self.file
        configuration.blankRecoveryFrames = nil
        configuration.quietStartFrames = 2...(configuration.leftContextFrames + 4)
        let recognizer = try ParakeetRecognizer(
            engine: FakeParakeetEngine(blanksAfterSpeechStart: true),
            vocabulary: ToneSpeech.vocabulary, configuration: configuration)
        expectTokens(try run(recognizer, script.samples, block: 640).committed, script.tokens)

        configuration.quietStartFrames = nil
        let fixed = try ParakeetRecognizer(
            engine: FakeParakeetEngine(blanksAfterSpeechStart: true),
            vocabulary: ToneSpeech.vocabulary, configuration: configuration)
        let committed = try run(fixed, script.samples, block: 640).committed
        #expect(committed.count < script.tokens.count)
    }

    /// Silence has no speech-level frame, so the recovery does not encode it again.
    @Test func blankRecoveryKeepsSilenceSilent() throws {
        let script = ToneSpeech.script(count: 4, trailing: 120)
        let engine = FakeParakeetEngine()
        let recognizer = try ParakeetRecognizer(
            engine: engine, vocabulary: ToneSpeech.vocabulary, configuration: Self.file)
        expectTokens(try run(recognizer, script.samples, block: 1_600).committed, script.tokens)
        var configuration = Self.file
        configuration.blankRecoveryFrames = nil
        let reference = FakeParakeetEngine()
        _ = try run(
            ParakeetRecognizer(
                engine: reference, vocabulary: ToneSpeech.vocabulary, configuration: configuration),
            script.samples, block: 1_600)
        // Only the first gap, which starts inside the last tone, is encoded again.
        #expect(engine.encodeCount <= reference.encodeCount + 1)
    }

    /// In a pause at the end of a German FLEURS file, a recovery decoded
    /// "Продолжение следует" ("to be continued") with high probability. Here the initial
    /// context decodes a token at a silent frame, which only a recovery decodes from, and
    /// silent gaps are not decoded again.
    @Test(arguments: [false, true])
    func blankRecoverySkipsSilentGaps(live: Bool) throws {
        var configuration = live ? Self.live : Self.file
        // The replayed context covers the pauses, so only a recovery uses the initial one.
        configuration.decoderHistoryFrames = 200
        let tokens = [(1, 0), (5, 5), (2, 10), (4, 15), (8, 20), (3, 60), (7, 65), (2, 70)]
        let samples = toneAudio(tokens, frames: 100)
        let engine = FakeParakeetEngine(silenceToken: 9)
        let recognizer = try ParakeetRecognizer(
            engine: engine, vocabulary: ToneSpeech.vocabulary, configuration: configuration)
        expectTokens(try run(recognizer, samples, block: 640).committed, tokens)
    }

    /// A pause that a second window does not recover can come before a gap after a sentence
    /// end that a second window does recover. Here a pause precedes the period in the same
    /// window.
    @Test func recoversGapsAfterAnUnrecoveredPause() throws {
        let tokens = [(1, 2), (5, 5), (3, 12), (ToneSpeech.sentenceEnd, 15), (9, 18), (2, 21)]
        let samples = toneAudio(tokens, frames: 40)
        var configuration = Self.file
        configuration.blankRecoveryFrames = 2
        configuration.recoversGapsAfterAFailure = true
        let recognizer = try ParakeetRecognizer(
            engine: FakeParakeetEngine(tokenDuration: 3, blanksAfterSentenceEnd: true),
            vocabulary: ToneSpeech.vocabulary, configuration: configuration)
        expectTokens(try run(recognizer, samples, block: 640).committed, tokens)

        configuration.recoversGapsAfterAFailure = false
        let stopping = try ParakeetRecognizer(
            engine: FakeParakeetEngine(tokenDuration: 3, blanksAfterSentenceEnd: true),
            vocabulary: ToneSpeech.vocabulary, configuration: configuration)
        // Otherwise a later window enters the tones after the period inside the first one.
        let stopped = try run(stopping, samples, block: 640).committed
        #expect(stopped.map(\.startSample) != tokens.map { $0.1 * 1280 })
    }

    /// In live and fast modes on long LibriSpeech files, Parakeet inserted filler such as
    /// "and the other thing and the other thing" as zero-duration tokens, up to the symbol
    /// limit at each of a few frames. Here every pause after a tone starts with such a frame.
    @Test(arguments: [false, true])
    func discardsFramesAtTheSymbolLimit(live: Bool) throws {
        let script = ToneSpeech.script(count: 20)
        var configuration = live ? Self.live : Self.file
        configuration.discardsFramesAtSymbolLimit = true
        let recognizer = try ParakeetRecognizer(
            engine: FakeParakeetEngine(fillerToken: 9), vocabulary: ToneSpeech.vocabulary,
            configuration: configuration)
        expectTokens(try run(recognizer, script.samples, block: 640).committed, script.tokens)

        configuration.discardsFramesAtSymbolLimit = false
        let keeping = try ParakeetRecognizer(
            engine: FakeParakeetEngine(fillerToken: 9), vocabulary: ToneSpeech.vocabulary,
            configuration: configuration)
        let committed = try run(keeping, script.samples, block: 640).committed
        #expect(committed.filter { $0.id == 9 }.count >= 10 * (script.tokens.count - 1))
    }

    /// Tone audio of `frames` encoder frames with three-frame tokens at their onset frames.
    func toneAudio(_ tokens: [(id: Int, frame: Int)], frames: Int) -> [Float] {
        var samples = [Float](repeating: 0, count: frames * 1280)
        for token in tokens {
            ToneSpeech.write(
                token: token.id, frames: token.frame..<(token.frame + 3), into: &samples,
                offset: 0)
        }
        return samples
    }

    @Test func carriesRecentTokensIntoTheNextWindow() throws {
        let script = ToneSpeech.script(count: 20)
        let engine = FakeParakeetEngine()
        let recognizer = try ParakeetRecognizer(
            engine: engine, vocabulary: ToneSpeech.vocabulary, configuration: Self.file)
        _ = try run(recognizer, script.samples, block: script.samples.count)
        // Each window replays at most the tokens of `decoderHistoryFrames` frames,
        // two tokens at a spacing of five frames, then one prediction per new token.
        #expect(engine.predictCount <= 1 + 20 + engine.encodeCount * 2)
    }

    @Test(arguments: [false, true])
    func forcedCommitsUseTheLongerWindow(endOfInput: Bool) throws {
        let script = ToneSpeech.script(count: 30)
        let engine = FakeParakeetEngine(forcedCommitWindow: true)
        let recognizer = try ParakeetRecognizer(
            engine: engine, vocabulary: ToneSpeech.vocabulary, configuration: Self.live)
        var run = try run(recognizer, script.samples, block: 640, endOfInput: false)
        #expect(engine.forcedCommitEncodeCount == 0)
        if endOfInput {
            run.updates.append(try recognizer.process(endOfInput: true))
        } else {
            run.updates.append(try recognizer.process(finalizeThrough: script.samples.count))
        }
        #expect(engine.forcedCommitEncodeCount == 1)
        expectTokens(run.committed, script.tokens)
        // Audio stays for the longer window: 48 frames before the commit frame.
        #expect(recognizer.retainedSampleCount >= 48 * 1280)
    }

    @Test func liveSettingsFitTheEncoderWindow() {
        typealias Configuration = ParakeetRecognizer.Configuration
        // The 15 s encoder holds 187 whole frames, so the 15 s settings stay.
        #expect(Configuration.live(windowFrames: 187) == .live)
        #expect(Configuration.fast(windowFrames: 187) == .fast)
        #expect(Configuration.live(windowFrames: 187).decoderHistoryFrames == 250)
        // An 8 s encoder holds 100 frames: 63 frames of left context, the same chunk and
        // right context, and the short-window history.
        let live = Configuration.live(windowFrames: 100)
        #expect(live.leftContextFrames == 63)
        #expect(live.chunkFrames == 12 && live.rightContextFrames == 25)
        #expect(live.decoderHistoryFrames == 75 && !live.sentenceScopedHistory)
        #expect(Configuration.live.sentenceScopedHistory)
        #expect(live.reportsVolatileResults && live.reusesMelFrames)
        let fast = Configuration.fast(windowFrames: 100)
        #expect(fast.leftContextFrames == 63 && fast.rightContextFrames == 12)
        // The window, not a fixed length, sets the context: 10 s keeps 63 frames,
        // and 6 s leaves 38.
        #expect(Configuration.live(windowFrames: 125).leftContextFrames == 63)
        #expect(Configuration.live(windowFrames: 75).leftContextFrames == 38)
        #expect(Configuration.file.reusesMelFrames == false)
        // Live and fast decoding discard frames that reach the symbol limit; file mode keeps
        // NeMo's behavior.
        #expect(live.discardsFramesAtSymbolLimit && fast.discardsFramesAtSymbolLimit)
        #expect(!Configuration.file.discardsFramesAtSymbolLimit)
        // Only file mode keeps recovering gaps after an unrecovered one.
        #expect(Configuration.file.recoversGapsAfterAFailure)
        #expect(!live.recoversGapsAfterAFailure && !fast.recoversGapsAfterAFailure)
    }

    @Test func rejectsWindowsWithoutRoomForAChunk() {
        var configuration = Self.file
        configuration.leftContextFrames = 12
        configuration.rightContextFrames = 12
        #expect(throws: SpeechRecognitionError.self) {
            try ParakeetRecognizer(
                engine: FakeParakeetEngine(), vocabulary: ToneSpeech.vocabulary,
                configuration: configuration)
        }
        configuration = Self.live
        configuration.chunkFrames = 2
        #expect(throws: SpeechRecognitionError.self) {
            try ParakeetRecognizer(
                engine: FakeParakeetEngine(), vocabulary: ToneSpeech.vocabulary,
                configuration: configuration)
        }
    }
}
