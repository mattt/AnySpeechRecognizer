import AVFoundation
import CoreMedia
import Foundation
import Testing

@testable import AnySpeechRecognizer

enum QwenFixtures {
    static var environment: [String: String] { ProcessInfo.processInfo.environment }
    static var modelDirectory: URL? {
        environment["ANYSPEECH_QWEN_MODELS"].map { URL(fileURLWithPath: $0) }
    }
    static var audio: URL? {
        environment["ANYSPEECH_QWEN_AUDIO"].map { URL(fileURLWithPath: $0) }
    }
}

/// Opt-in tests of the Qwen3-ASR backend through the public analyzer.
/// Set `ANYSPEECH_QWEN_MODELS` to a directory that `Tools/QwenConversion/convert.py` wrote
/// and `ANYSPEECH_QWEN_AUDIO` to a speech recording.
@Suite(.enabled(if: QwenFixtures.modelDirectory != nil && QwenFixtures.audio != nil), .serialized)
struct QwenAnalyzer {
    let audio = QwenFixtures.audio!
    var model: QwenSpeechModel { QwenSpeechModel(assets: .directory(QwenFixtures.modelDirectory!)) }

    /// Transcribes the recording as one segment with the recognizer, without the analyzer.
    func directText() throws -> String {
        let directory = QwenFixtures.modelDirectory!
        let configuration = try QwenModelConfiguration(
            contentsOf: directory.appending(path: CoreMLQwenEngine.configurationName))
        let engine = try CoreMLQwenEngine(
            directory: directory, configuration: configuration, encoderComputeUnits: .cpuAndGPU,
            decoderComputeUnits: .cpuAndGPU)
        let vocabulary = try QwenVocabulary(
            contentsOf: directory.appending(path: CoreMLQwenEngine.vocabularyName),
            skipping: Set(configuration.skippedTokenIds))
        let recognizer = QwenRecognizer(engine: engine, vocabulary: vocabulary, prefill: nil)
        return try recognizer.transcribe(ParakeetFixtures.samples(audio)).text
    }

    @Test func fileTextMatchesTheRecognizer() async throws {
        #expect(await model.availability() == .ready)
        let transcriber = SpeechTranscriber(model: model)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        async let results = collect(transcriber)
        let file = try AVAudioFile(forReading: audio)
        let time = try #require(try await analyzer.analyzeSequence(from: file))
        try await analyzer.finalizeAndFinish(through: time)
        let values = try await results
        #expect(!values.isEmpty)
        #expect(values.allSatisfy { $0.isFinal && $0.text.runs.count == 1 })
        #expect(zip(values, values.dropFirst()).allSatisfy { $0.range.end <= $1.range.start })
        // `time` is the start of the last sample; the last segment ends after that sample.
        let end = CMTime(
            value: file.length, timescale: CMTimeScale(file.processingFormat.sampleRate))
        #expect(values.last?.range.end == end)
        // A recording of up to 40 seconds is one segment.
        if Double(file.length) / file.processingFormat.sampleRate <= 40 {
            let direct = try directText()
            #expect(texts(values) == [direct])
        }
    }

    /// Live input without volatile results works; results arrive when a segment is complete,
    /// here at the finalization.
    @Test func liveInputPublishesAtFinalization() async throws {
        let transcriber = SpeechTranscriber(model: model)
        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        let analyzer = SpeechAnalyzer(inputSequence: stream, modules: [transcriber])
        async let results = collect(transcriber)
        for input in try liveBlocks(audio, seconds: 0.1) { continuation.yield(input) }
        continuation.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        let values = try await results
        #expect(!values.isEmpty)
        #expect(values.allSatisfy { $0.isFinal })
    }

    @Test func volatileResultsFailBeforeAnalysis() async throws {
        let transcriber = SpeechTranscriber(model: model, reportingOptions: [.volatileResults])
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        await #expect(throws: SpeechRecognitionError.self) {
            try await analyzer.prepareToAnalyze(in: nil)
        }
        await analyzer.cancelAndFinishNow()
    }
}
