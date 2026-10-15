import AVFoundation
import CoreMedia
import Foundation
import Speech
import Testing

@testable import AnySpeechRecognizer

@Test func finalSnapshotsDrainAndPreserveAttributes() async throws {
    let (url, file) = try audioFile(frames: 32_000)
    defer { try? FileManager.default.removeItem(at: url) }
    let first = snapshot(start: 0, end: 16_000)
    var attributed = AttributedString(" e\u{301}, 世界!")
    let range = CMTimeRange(
        start: CMTime(value: 16_000, timescale: 16_000),
        duration: CMTime(value: 16_000, timescale: 16_000))
    attributed[AttributeScopes.SpeechAttributes.TimeRangeAttribute.self] = range
    attributed[AttributeScopes.SpeechAttributes.ConfidenceAttribute.self] = 0.75
    let last = SpeechRecognitionResult(
        text: attributed, range: range, resultsFinalizationTime: range.end)
    let executor = FakeExecutor(snapshots: [first], deferred: [last])
    let transcriber = SpeechTranscriber(
        model: FakeModel(executor: executor), attributeOptions: [.audioTimeRange])
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    async let results = collect(transcriber)
    try await analyzer.prepareToAnalyze(in: file.processingFormat)
    let end = try #require(try await analyzer.analyzeSequence(from: file))
    #expect(end == CMTime(value: 31_999, timescale: 16_000))
    #expect(await analyzer.state == .draining)
    try await analyzer.finalizeAndFinish(through: end)
    let values = try await results
    #expect(values.map(\.text) == [first.text, last.text])
    #expect(values.allSatisfy { $0.isFinal })
    #expect(await executor.prepareCount == 1)
    #expect(await analyzer.state == .finished)
}

@Test func emptyFileFinishesWithoutResults() async throws {
    let (url, file) = try audioFile(frames: 0)
    defer { try? FileManager.default.removeItem(at: url) }
    let transcriber = SpeechTranscriber(model: FakeModel(executor: FakeExecutor()))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    async let results = collect(transcriber)
    #expect(try await analyzer.analyzeSequence(from: file) == nil)
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    #expect(try await results.isEmpty)
    #expect(await analyzer.state == .finished)
}

@Test func unsupportedOptionsFailBeforeReading() async throws {
    let (url, file) = try audioFile()
    defer { try? FileManager.default.removeItem(at: url) }
    let configurations =
        SpeechTranscriber.TranscriptionOption.allCases.map {
            SpeechRecognitionConfiguration(transcriptionOptions: [$0])
        }
        + [SpeechTranscriber.ReportingOption.alternativeTranscriptions, .fastResults].map {
            SpeechRecognitionConfiguration(reportingOptions: [$0])
        } + [.init(attributeOptions: [.transcriptionConfidence])]
    for config in configurations {
        let executor = FakeExecutor()
        let transcriber = SpeechTranscriber(
            model: FakeModel(executor: executor), transcriptionOptions: config.transcriptionOptions,
            reportingOptions: config.reportingOptions, attributeOptions: config.attributeOptions
        )
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        await #expect(throws: SpeechRecognitionError.self) {
            try await analyzer.analyzeSequence(from: file)
        }
        #expect(await executor.prepareCount == 0)
        #expect(await executor.pulledFrames == 0)
        await #expect(throws: SpeechRecognitionError.self) { try await collect(transcriber) }
    }
}

@Test func dictationRequiresModelSupport() async throws {
    let executor = FakeExecutor()
    let dictation = DictationTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [dictation])
    await #expect(throws: SpeechRecognitionError.self) {
        try await analyzer.prepareToAnalyze(in: nil)
    }
    #expect(await executor.prepareCount == 0)
}

@Test func backendFailureIsRetainedAndTerminatesResults() async throws {
    let (url, file) = try audioFile()
    defer { try? FileManager.default.removeItem(at: url) }
    let transcriber = SpeechTranscriber(
        model: FakeModel(executor: FakeExecutor(backendError: .backend)))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    do {
        _ = try await analyzer.analyzeSequence(from: file)
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        Issue.record("Expected the backend failure.")
    } catch SpeechRecognitionError.backendFailure(let error) {
        #expect(error as? TestFailure == .backend)
    }
    await analyzer.cancelAndFinishNow()
    #expect(await analyzer.state == .failed)
    do {
        _ = try await collect(transcriber)
        Issue.record("Expected the original result error.")
    } catch SpeechRecognitionError.backendFailure(let error) {
        #expect(error as? TestFailure == .backend)
    }
}

@Test func sequentialInputsContinueTheTimeline() async throws {
    let (firstURL, first) = try audioFile(frames: 16_000)
    let (secondURL, second) = try audioFile(frames: 8_000)
    defer {
        try? FileManager.default.removeItem(at: firstURL)
        try? FileManager.default.removeItem(at: secondURL)
    }
    let executor = FakeExecutor()
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    async let results = collect(transcriber)
    _ = try await analyzer.analyzeSequence(from: first)
    let end = try #require(try await analyzer.analyzeSequence(from: second))
    #expect(end == CMTime(value: 23_999, timescale: 16_000))
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    _ = try await results
    let starts = await executor.pulled.map(\.startTime)
    #expect(starts.first == .zero)
    #expect(await executor.pulled.last?.endTime == CMTime(value: 24_000, timescale: 16_000))
    #expect(zip(starts, starts.dropFirst()).allSatisfy { $0 < $1 })
}

@Test func concurrentInputIsRejected() async throws {
    let gate = Gate()
    let executor = FakeExecutor(analyzeGate: gate)
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    let (stream, continuation) = liveInput()
    try await analyzer.start(inputSequence: stream)
    continuation.yield(input(frames: 1600))
    await executor.analyzeEntered.wait()
    await #expect(throws: SpeechRecognitionError.self) {
        try await analyzer.analyzeSequence(stream)
    }
    continuation.finish()
    await gate.open()
    try await analyzer.finalizeAndFinishThroughEndOfInput()
}

@Test func duplicateFinalRangeFails() async throws {
    let (url, file) = try audioFile()
    defer { try? FileManager.default.removeItem(at: url) }
    let result = snapshot()
    let transcriber = SpeechTranscriber(
        model: FakeModel(executor: FakeExecutor(snapshots: [result, result])))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    await #expect(throws: SpeechRecognitionError.self) {
        _ = try await analyzer.analyzeSequence(from: file)
        try await analyzer.finalizeAndFinishThroughEndOfInput()
    }
    #expect(await analyzer.state == .failed)
    await #expect(throws: SpeechRecognitionError.self) { try await collect(transcriber) }
}

@Test func resultsBeyondConsumedInputFail() async throws {
    let (url, file) = try audioFile(frames: 8_000)
    defer { try? FileManager.default.removeItem(at: url) }
    let executor = FakeExecutor()
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    _ = try await analyzer.analyzeSequence(from: file)
    await #expect(throws: SpeechRecognitionError.self) {
        try await executor.publishLater(snapshot(start: 0, end: 16_000))
    }
    #expect(await analyzer.state == .failed)
}

@Test func undrainedFinalResultsFailWithOverflow() async throws {
    let (url, file) = try audioFile(frames: 16_000)
    defer { try? FileManager.default.removeItem(at: url) }
    let maximum = ResultChannel<SpeechTranscriber.Result>.maximumResults
    let snapshots = (0...maximum).map {
        snapshot("a", start: Int64($0), end: Int64($0 + 1))
    }
    let transcriber = SpeechTranscriber(
        model: FakeModel(executor: FakeExecutor(snapshots: snapshots)))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    do {
        _ = try await analyzer.analyzeSequence(from: file)
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        Issue.record("Expected bounded result overflow.")
    } catch SpeechRecognitionError.bufferOverflow {}
    await #expect(throws: SpeechRecognitionError.self) { try await collect(transcriber) }
    #expect(await analyzer.state == .failed)
}

@Test func oneTranscriberCannotBeAttachedTwice() async throws {
    let transcriber = SpeechTranscriber(model: FakeModel(executor: FakeExecutor()))
    let owner = SpeechAnalyzer(modules: [transcriber])
    let other = SpeechAnalyzer(modules: [transcriber])
    await #expect(throws: SpeechRecognitionError.self) { try await other.prepareToAnalyze(in: nil) }
    await other.cancelAndFinishNow()
    try await owner.prepareToAnalyze(in: nil)
    await owner.cancelAndFinishNow()
}

@Test func lateFailureStartsCleanupAndPreservesError() async throws {
    let (url, file) = try audioFile()
    defer { try? FileManager.default.removeItem(at: url) }
    let executor = FakeExecutor()
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    _ = try await analyzer.analyzeSequence(from: file)
    await #expect(throws: SpeechRecognitionError.self) {
        try await executor.publishLater(snapshot(start: 16_000, end: 32_000))
    }
    await analyzer.cancelAndFinishNow()
    #expect(await executor.cancelCount == 1)
    #expect(await analyzer.state == .failed)
    await #expect(throws: SpeechRecognitionError.self) { try await collect(transcriber) }
}

@Test func emptyModulesFailAtFirstInput() async throws {
    let analyzer = SpeechAnalyzer(modules: [])
    await #expect(throws: SpeechRecognitionError.self) {
        try await analyzer.prepareToAnalyze(in: nil)
    }
}

@Test func multipleModulesReceiveTheSameInput() async throws {
    let (url, file) = try audioFile(sampleRate: 44_100, channels: 2, frames: 44_100)
    defer { try? FileManager.default.removeItem(at: url) }
    let first = FakeExecutor(snapshots: [snapshot("One.", start: 0, end: 8_000)])
    let second = FakeExecutor(snapshots: [snapshot("Two.", start: 0, end: 8_000)])
    let one = SpeechTranscriber(model: FakeModel(executor: first))
    let two = SpeechTranscriber(model: FakeModel(executor: second))
    let analyzer = SpeechAnalyzer(modules: [one, two])
    async let firstResults = collect(one)
    async let secondResults = collect(two)
    _ = try await analyzer.analyzeSequence(from: file)
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    #expect(texts(try await firstResults) == ["One."])
    #expect(texts(try await secondResults) == ["Two."])
    for executor in [first, second] {
        #expect(await executor.format?.sampleRate == 16_000)
        // Resampling keeps the source duration once the converter is flushed.
        #expect(abs(await executor.pulledFrames - 16_000) <= 1)
    }
}

@Test func analyzerReturnsFormatAcceptedByEveryModule() async throws {
    let one = SpeechTranscriber(model: FakeModel(executor: FakeExecutor()))
    let two = SpeechTranscriber(model: FakeModel(executor: FakeExecutor()))
    let natural = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)
    let format = await SpeechAnalyzer.bestAvailableAudioFormat(
        compatibleWith: [one, two], considering: natural)
    #expect(format?.sampleRate == 16_000)
    #expect(format?.channelCount == 1)
}
