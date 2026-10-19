import AVFoundation
import CoreMedia
import Foundation
import Speech
import Synchronization
import Testing

@testable import AnySpeechRecognizer

struct QwenExecutorTests {
    typealias T = QwenTestSupport

    final class Received: Sendable {
        let results = Mutex<[SpeechRecognitionResult]>([])
        var all: [SpeechRecognitionResult] { results.withLock { $0 } }
        var texts: [String] { all.map { String($0.text.characters) } }
    }

    static let model = QwenSpeechModel(
        assets: .directory(URL(fileURLWithPath: "/nonexistent/qwen")))

    func executor(scripts: [[Int]]) async -> (QwenExecutor, FakeQwenEngine) {
        let engine = FakeQwenEngine(scripts: scripts)
        let executor = QwenExecutor(model: Self.model)
        await executor.use(QwenRecognizer(engine: engine, vocabulary: T.vocabulary, prefill: nil))
        return (executor, engine)
    }

    func buffer(_ samples: [Float], start: Double) throws -> TimedAudioBuffer {
        let buffer = try #require(
            AVAudioPCMBuffer(
                pcmFormat: AudioFormats.monoFloat16k,
                frameCapacity: AVAudioFrameCount(samples.count)))
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for (index, sample) in samples.enumerated() { buffer.floatChannelData![0][index] = sample }
        return TimedAudioBuffer(
            buffer: buffer, startTime: CMTime(seconds: start, preferredTimescale: 16_000))
    }

    /// Runs the executor over buffers, then finalizes through the end of the last one.
    func run(_ executor: QwenExecutor, _ buffers: [TimedAudioBuffer]) async throws -> Received {
        let received = Received()
        let sink = SpeechRecognitionResultSink(
            publish: { result in received.results.withLock { $0.append(result) } },
            volatile: { _ in Issue.record("Qwen3-ASR reported a volatile range.") })
        let branch = AudioBranch(format: AudioFormats.monoFloat16k)
        let analysis = Task {
            try await executor.analyze(SpeechAnalysisInput(branch: branch), results: sink)
        }
        for buffer in buffers {
            try await branch.send(.init(audio: buffer, sourceEnd: buffer.endTime))
        }
        // A pulled buffer is appended afterward; finalize once the executor has appended it.
        let end = try #require(buffers.last).endTime
        await eventually { await executor.inputEnd == end }
        try await executor.finalize(through: end)
        branch.finish()
        try await analysis.value
        return received
    }

    /// Audio longer than 40 seconds is cut at a quiet point near 30 seconds, and each segment
    /// becomes one final result with the segment's source range.
    @Test func publishesOneFinalResultPerSegment() async throws {
        let (executor, engine) = await executor(scripts: [T.englishReply])
        let samples = T.tone(seconds: 50, silence: 28.0..<28.4)
        let received = try await run(
            executor,
            [
                try buffer(Array(samples[..<400_000]), start: 2),
                try buffer(Array(samples[400_000...]), start: 27),
            ])
        #expect(received.texts == ["Hello world", " Hello world"])
        let ranges = received.all.map(\.range)
        #expect(ranges.first?.start == CMTime(value: 32_000, timescale: 16_000))
        #expect((30.0..<30.4).contains(try #require(ranges.first).end.seconds))
        #expect(ranges.first?.end == ranges.last?.start)
        #expect(ranges.last?.end == CMTime(value: 52 * 16_000, timescale: 16_000))
        #expect(
            received.all.allSatisfy { $0.isFinal && $0.resultsFinalizationTime == $0.range.end })
        #expect(engine.prompts.count == 2)
    }

    /// A gap in the input ends the segment before it; results do not cross the gap.
    @Test func endsASegmentAtAGap() async throws {
        let (executor, _) = await executor(scripts: [T.englishReply, T.chineseReply])
        let received = try await run(
            executor,
            [try buffer(T.tone(seconds: 3), start: 0), try buffer(T.tone(seconds: 2), start: 10)])
        #expect(received.texts == ["Hello world", "你好"])
        #expect(received.all.map(\.range.start.seconds) == [0, 10])
        #expect(received.all.map(\.range.end.seconds) == [3, 12])
    }

    /// Segments with an empty transcript publish nothing.
    @Test func publishesNothingForSilence() async throws {
        let (executor, _) = await executor(scripts: [[T.language, T.none, T.asrText]])
        let received = try await run(
            executor, [try buffer([Float](repeating: 0, count: 32_000), start: 0)])
        #expect(received.all.isEmpty)
    }

    /// Results are joined with spaces, except around Chinese characters and kana.
    @Test func spacesTextBetweenSegments() {
        #expect(QwenExecutor.join("Hello.", "World") == "Hello. World")
        #expect(QwenExecutor.join("你好。", "世界") == "你好。世界")
        #expect(QwenExecutor.join("今日は", "Hello") == "今日はHello")
        #expect(QwenExecutor.join("안녕하세요.", "반갑습니다") == "안녕하세요. 반갑습니다")
    }

    /// Volatile and fast results, word timing, confidence, and dictation fail preparation
    /// before the executor loads models.
    @Test(arguments: [
        SpeechRecognitionRequest.transcription(.init(reportingOptions: [.volatileResults])),
        .transcription(.init(reportingOptions: [.fastResults])),
        .transcription(.init(attributeOptions: [.audioTimeRange])),
        .transcription(.init(attributeOptions: [.transcriptionConfidence])),
        .dictation(.init()),
    ])
    func rejectsUnsupportedRequests(request: SpeechRecognitionRequest) async {
        let executor = QwenExecutor(model: Self.model)
        await #expect(throws: SpeechRecognitionError.self) {
            try await executor.prepare(
                request: request, format: AudioFormats.monoFloat16k, context: AnalysisContext())
        }
    }
}

struct QwenSpeechModelTests {
    func stagedDirectory(_ names: [String]) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: "anyspeech-qwen-\(UUID())")
        for name in names {
            let url = directory.appending(path: name)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data().write(to: url)
        }
        return directory
    }

    static let files = [
        "AudioEncoder.mlmodelc/model.mil", "TextDecoder.mlmodelc/model.mil", "vocab.json",
        "qwen3_asr_config.json",
    ]

    @Test func reportsStagedAssets() async throws {
        let complete = try stagedDirectory(Self.files)
        let incomplete = try stagedDirectory(Array(Self.files.dropLast()))
        defer {
            try? FileManager.default.removeItem(at: complete)
            try? FileManager.default.removeItem(at: incomplete)
        }
        #expect(await QwenSpeechModel(assets: .directory(complete)).availability() == .ready)
        #expect(
            try await QwenSpeechModel(assets: .directory(complete)).assetInstallationRequest()
                == nil)
        #expect(
            await QwenSpeechModel(assets: .directory(incomplete)).availability() == .missingAssets)
        await #expect(throws: SpeechRecognitionError.self) {
            _ = try await QwenSpeechModel(assets: .directory(incomplete)).assetInstallationRequest()
        }
    }

    /// Only the language code selects the forced language; unsupported codes are reported.
    @Test func checksTheForcedLanguage() async throws {
        let directory = try stagedDirectory(Self.files)
        defer { try? FileManager.default.removeItem(at: directory) }
        let cantonese = QwenSpeechModel(
            assets: .directory(directory), language: Locale.Language(identifier: "yue-Hant-HK"))
        #expect(cantonese.languageCode == "yue")
        #expect(await cantonese.availability() == .ready)
        let klingon = QwenSpeechModel(
            assets: .directory(directory), language: Locale.Language(identifier: "tlh"))
        #expect(await klingon.availability() == .unsupportedLanguage)
        #expect(QwenSpeechModel(assets: .directory(directory)).languageCode == nil)
    }

    /// The attribution names the checkpoint and its license, as the converter writes it.
    @Test func attributesTheCheckpoint() {
        let model = QwenSpeechModel(variant: .small, assets: .directory(URL(fileURLWithPath: "/")))
        #expect(model.attribution.hasPrefix("Qwen3-ASR 0.6B by the Qwen team, Alibaba Cloud"))
        #expect(model.attribution.contains("https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf"))
        #expect(model.attribution.contains("Apache License, Version 2.0"))
        #expect(model.modelID == "qwen3-asr-0.6b")
    }

    /// The model supports file transcription with automatic or forced language only.
    @Test func declaresFileModeCapabilities() throws {
        let capabilities = QwenSpeechModel(assets: .directory(URL(fileURLWithPath: "/")))
            .capabilities
        let transcription = try #require(capabilities.transcription)
        #expect(transcription.reportingOptions.isEmpty)
        #expect(transcription.attributeOptions.isEmpty)
        #expect(capabilities.dictation == nil)
        #expect(!capabilities.liveInput)
        #expect(capabilities.languageControl)
    }
}
