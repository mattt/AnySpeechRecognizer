import Foundation
import Testing

@testable import AnySpeechRecognizer

/// The cache-aware recognizer with tone "speech"; see ``FakeNemotronEngine``.
struct NemotronRecognizerTests {
    struct Run {
        var committed: [ParakeetToken] = []
        var updates: [ParakeetUpdate] = []
    }

    /// Appends audio in blocks and processes after each block, then ends the input.
    static func run(
        _ samples: [Float], engine: FakeNemotronEngine, block: Int, prompt: Int = 7,
        finalizations: [Int: Int] = [:]
    ) throws -> Run {
        let recognizer = try NemotronRecognizer(
            engine: engine, vocabulary: ToneSpeech.vocabulary, prompt: prompt)
        var run = Run()
        var offset = 0
        while offset < samples.count {
            let end = min(samples.count, offset + block)
            recognizer.append(Array(samples[offset..<end]))
            offset = end
            let update = try recognizer.process(finalizeThrough: finalizations[offset])
            run.updates.append(update)
            run.committed += update.committed
        }
        let last = try recognizer.process(endOfInput: true)
        run.updates.append(last)
        run.committed += last.committed
        return run
    }

    static func ids(_ tokens: [ParakeetToken]) -> [Int] { tokens.map(\.id) }

    /// Every token is decoded once, at its onset frame, whatever the block size.
    @Test(arguments: [1, 2, 7])
    func decodesEveryTokenOnce(chunkFrames: Int) throws {
        let script = ToneSpeech.script(count: 12, spacing: 4)
        for block in [160, 1600, 4000, script.samples.count] {
            let run = try Self.run(
                script.samples, engine: FakeNemotronEngine(chunkFrames: chunkFrames), block: block)
            #expect(Self.ids(run.committed) == script.tokens.map(\.id), "block \(block)")
            #expect(run.committed.map { $0.startSample / 1280 } == script.tokens.map(\.frame))
            #expect(run.updates.allSatisfy { $0.volatile.isEmpty })
        }
    }

    /// Each chunk is encoded once and updates the caches; only the last chunk is partial.
    @Test func encodesEachChunkOnce() throws {
        let script = ToneSpeech.script(count: 6, spacing: 4)
        let engine = FakeNemotronEngine(chunkFrames: 2)
        _ = try Self.run(script.samples, engine: engine, block: 1600, prompt: 42)
        let melFrames = script.samples.count / 160
        let chunks = (melFrames + 15) / 16
        #expect(engine.steps.count == chunks)
        #expect(engine.steps.allSatisfy { $0.updatesCaches && $0.prompt == 42 })
        #expect(engine.steps.dropLast().allSatisfy { $0.validFrameCount == 9 + 16 })
        #expect(engine.steps.last?.validFrameCount == 9 + melFrames - (chunks - 1) * 16)
        #expect(engine.resets == 1)
    }

    /// A chunk waits for the mel frame at the end of its audio, which reads 12.5 ms past it.
    @Test func waitsForTheLastMelFrameOfAChunk() throws {
        let engine = FakeNemotronEngine(chunkFrames: 2)
        let recognizer = try NemotronRecognizer(
            engine: engine, vocabulary: ToneSpeech.vocabulary, prompt: 0)
        // 16 mel frames need samples through 160 × 15 + 199.
        recognizer.append([Float](repeating: 0, count: 160 * 15 + 199))
        _ = try recognizer.process()
        #expect(engine.steps.isEmpty)
        recognizer.append([0])
        _ = try recognizer.process()
        #expect(engine.steps.count == 1)
        #expect(recognizer.committedThroughSample == 2 * 1280)
    }

    /// A finalization in the middle of a chunk decodes through the finalized time without
    /// updating the caches; the complete chunk then updates them and decodes only later frames.
    @Test func finalizationDecodesAPartialChunkOnce() throws {
        let script = ToneSpeech.script(count: 8, spacing: 4)
        let engine = FakeNemotronEngine(chunkFrames: 7)
        // Finalize after 18 encoder frames, inside the third chunk (frames 14 to 20).
        let finalized = 18 * 1280
        let block = 1280
        let run = try Self.run(
            script.samples, engine: engine, block: block,
            finalizations: [finalized + block: finalized])
        #expect(Self.ids(run.committed) == script.tokens.map(\.id))
        #expect(run.committed.map { $0.startSample / 1280 } == script.tokens.map(\.frame))
        let speculative = engine.steps.filter { !$0.updatesCaches }
        #expect(speculative.count == 1)
        let update = try #require(
            run.updates.first {
                $0.committedThroughSample >= finalized && $0.committedThroughSample < 21 * 1280
            })
        #expect(update.committedThroughSample == finalized)
    }

    /// The end of input decodes the last partial chunk.
    @Test func endOfInputDecodesThePartialChunk() throws {
        // 27 encoder frames: three chunks of 7, then a partial chunk of 6 frames in which
        // the last tone starts, at frame 21.
        let script = ToneSpeech.script(count: 3, spacing: 4, lead: 13, trailing: 2)
        let engine = FakeNemotronEngine(chunkFrames: 7)
        let run = try Self.run(script.samples, engine: engine, block: 1600)
        #expect(Self.ids(run.committed) == script.tokens.map(\.id))
        #expect(script.tokens.last?.frame == 21)
        #expect(engine.steps.count == 4)
        #expect(engine.steps.last?.validFrameCount == 9 + 6 * 8)
        #expect(run.updates.last?.committedThroughSample == script.samples.count)
    }

    /// The analyzer finalizes a file instead of ending its input. Finalizing through the last
    /// sample decodes the same tokens, including a tone in the last 20 ms, whose mel frames
    /// read past the received audio.
    @Test func finalizationAtTheEndMatchesTheEndOfInput() throws {
        // The last tone fills the last 280 samples, which only the last mel frame of the last
        // encoder frame reads; that frame's window ends 40 samples after the audio.
        var samples = ToneSpeech.script(count: 3, spacing: 4).samples
        let frames = samples.count / 1280
        var last = [Float](repeating: 0, count: 1280)
        ToneSpeech.write(
            token: 2, frames: frames..<(frames + 1), into: &last, offset: frames * 1280)
        samples += [Float](repeating: 0, count: 1000) + last[1000...]
        let ended = try Self.run(samples, engine: FakeNemotronEngine(chunkFrames: 7), block: 1600)
        let recognizer = try NemotronRecognizer(
            engine: FakeNemotronEngine(chunkFrames: 7), vocabulary: ToneSpeech.vocabulary,
            prompt: 0)
        var committed: [ParakeetToken] = []
        for start in stride(from: 0, to: samples.count, by: 1600) {
            recognizer.append(Array(samples[start..<min(samples.count, start + 1600)]))
            committed += try recognizer.process().committed
        }
        let update = try recognizer.process(finalizeThrough: samples.count)
        committed += update.committed
        #expect(Self.ids(committed) == Self.ids(ended.committed))
        #expect(committed.last?.id == 2)
        #expect(update.committedThroughSample == samples.count)
    }

    /// Token times cover one encoder frame and end at most at the received audio.
    @Test func tokenTimesCoverOneFrame() throws {
        let script = ToneSpeech.script(count: 4, spacing: 4)
        let run = try Self.run(
            script.samples, engine: FakeNemotronEngine(chunkFrames: 4), block: 1600)
        for token in run.committed {
            #expect(token.endSample - token.startSample == 1280)
            #expect(token.endSample <= script.samples.count)
        }
    }
}
