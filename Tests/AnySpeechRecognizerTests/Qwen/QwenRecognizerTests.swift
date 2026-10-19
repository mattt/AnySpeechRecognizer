import Foundation
import Testing

@testable import AnySpeechRecognizer

struct QwenRecognizerTests {
    typealias T = QwenTestSupport

    /// 13 embeddings for each whole 100-frame chunk and `ceil(frames / 8)` for a partial one.
    @Test func countsAudioEmbeddings() {
        let configuration = T.configuration()
        #expect(configuration.audioTokenCount(melFrames: 800) == 104)
        #expect(configuration.audioTokenCount(melFrames: 103) == 14)
        #expect(configuration.audioTokenCount(melFrames: 50) == 7)
        #expect(configuration.audioTokenCount(melFrames: 1) == 1)
    }

    /// The prompt is the prefix, one placeholder per embedding, and the suffix; embeddings fill
    /// the placeholder rows in window order, and 8-second windows are encoded separately.
    @Test func buildsThePromptAroundAudioEmbeddings() throws {
        let engine = FakeQwenEngine(scripts: [T.englishReply])
        let recognizer = QwenRecognizer(engine: engine, vocabulary: T.vocabulary, prefill: nil)
        // 10.3 seconds: 1,030 frames, a whole window and a window of 230 frames.
        let transcript = try recognizer.transcribe(T.tone(seconds: 10.3))
        #expect(engine.encodedWindows == [800, 230])
        let audioTokens = 104 + 2 * 13 + 4
        let prompt = try #require(engine.prompts.first)
        #expect(
            prompt == [T.prefix] + Array(repeating: T.audioPad, count: audioTokens) + [T.suffix])
        let rows = try #require(engine.promptAudio.first)
        #expect(rows.first == 0 && rows.last == 0)
        #expect(rows[1..<105].allSatisfy { $0 == 800 })
        #expect(rows[105..<(1 + audioTokens)].allSatisfy { $0 == 230 })
        #expect(
            transcript
                == .init(
                    language: "English", text: "Hello world", tokens: T.englishReply,
                    reachedContextLimit: false))
    }

    /// A forced language appends its prefill, and the output is all transcript.
    @Test func forcesTheLanguageWithAPrefill() throws {
        let engine = FakeQwenEngine(scripts: [[T.hello, T.world]])
        let prefill = try #require(T.configuration().prompt.languages["en"]?.prefill)
        let recognizer = QwenRecognizer(engine: engine, vocabulary: T.vocabulary, prefill: prefill)
        let transcript = try recognizer.transcribe(T.tone(seconds: 2))
        #expect(engine.prompts.first?.suffix(4) == [T.suffix] + prefill)
        #expect(transcript.text == "Hello world")
        #expect(transcript.language == nil)
    }

    /// Audio shorter than 0.5 seconds is padded to 50 frames: 7 embeddings.
    @Test func padsShortAudio() throws {
        let engine = FakeQwenEngine(scripts: [[T.language, T.none, T.asrText]])
        let recognizer = QwenRecognizer(engine: engine, vocabulary: T.vocabulary, prefill: nil)
        let transcript = try recognizer.transcribe(T.tone(seconds: 0.1))
        #expect(engine.encodedWindows == [50])
        #expect(engine.prompts.first?.count == 2 + 7)
        #expect(transcript.text.isEmpty && transcript.language == nil)
    }

    /// Decoding stops at the end of the context when no end token arrives.
    @Test func stopsAtTheContextLimit() throws {
        let engine = FakeQwenEngine(configuration: T.configuration(context: 64), scripts: [[]])
        engine.endless = true
        let recognizer = QwenRecognizer(engine: engine, vocabulary: T.vocabulary, prefill: nil)
        // 2 seconds: 26 embeddings and a 28-token prompt, so 36 positions remain.
        let transcript = try recognizer.transcribe(T.tone(seconds: 2))
        #expect(transcript.reachedContextLimit)
        #expect(transcript.tokens.count == 64 - 28 + 1)
        #expect(engine.position == 64)
    }
}

struct QwenSegmenterTests {
    /// The boundary is in the quietest 100 ms within 5 seconds of the 30-second target.
    @Test func cutsAtTheQuietestPoint() {
        let samples = QwenTestSupport.tone(seconds: 45, silence: 27.0..<27.3)
        let boundary = samples.withUnsafeBufferPointer { QwenSegmenter.boundary(in: $0) }
        #expect((27 * 16_000)..<Int(27.3 * 16_000) ~= boundary)
    }

    /// Silence outside the search window does not move the boundary out of it.
    @Test func searchesOnlyNearTheTarget() {
        let samples = QwenTestSupport.tone(seconds: 45, silence: 10..<12)
        let boundary = samples.withUnsafeBufferPointer { QwenSegmenter.boundary(in: $0) }
        #expect((25 * 16_000)...(35 * 16_000) ~= boundary)
    }

    @Test func handlesShortInput() {
        let samples = [Float](repeating: 0.1, count: 100)
        #expect(samples.withUnsafeBufferPointer { QwenSegmenter.boundary(in: $0) } == 100)
    }
}
