import AVFoundation
import CoreMedia
import Foundation
import Testing

@testable import AnySpeechRecognizer

private func time(_ frames: Int64) -> CMTime { CMTime(value: frames, timescale: 16_000) }

@Test func finalizeWaitsUntilInputReachesTheTarget() async throws {
    let executor = FakeExecutor()
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    let (stream, continuation) = liveInput()
    try await analyzer.start(inputSequence: stream)
    continuation.yield(input(frames: 8_000))
    let finalized = Task { try await analyzer.finalize(through: time(12_000)) }
    await eventually { await executor.pulledFrames == 8_000 }
    // The control arrived before its audio; the executor must not finalize yet.
    #expect(await executor.finalizeTimes.isEmpty)
    continuation.yield(input(frames: 8_000))
    try await finalized.value
    #expect(await executor.finalizeTimes == [time(12_000)])
    continuation.finish()
    try await analyzer.finalizeAndFinishThroughEndOfInput()
}

@Test func finalizeAfterConsumptionRunsImmediately() async throws {
    let executor = FakeExecutor(deferred: [snapshot("Late.", start: 0, end: 8_000)])
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    let input = AsyncStream<AnalyzerInput> { continuation in
        continuation.yield(input(frames: 16_000))
        continuation.finish()
    }
    async let results = collect(transcriber)
    _ = try await analyzer.analyzeSequence(input)
    try await analyzer.finalize(through: nil)
    #expect(await executor.finalizeTimes == [time(16_000)])
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    #expect(texts(try await results) == ["Late."])
}

@Test func finalizeWithoutInputDoesNothing() async throws {
    let executor = FakeExecutor()
    let analyzer = SpeechAnalyzer(modules: [SpeechTranscriber(model: FakeModel(executor: executor))]
    )
    try await analyzer.finalize(through: nil)
    #expect(await executor.finalizeTimes.isEmpty)
    await analyzer.cancelAndFinishNow()
}

@Test func pendingFinalizeFailsWhenInputEndsFirst() async throws {
    let executor = FakeExecutor()
    let analyzer = SpeechAnalyzer(modules: [SpeechTranscriber(model: FakeModel(executor: executor))]
    )
    let (stream, continuation) = liveInput()
    try await analyzer.start(inputSequence: stream)
    continuation.yield(input(frames: 1_600))
    let finalized = Task { try await analyzer.finalize(through: time(32_000)) }
    await eventually { await executor.pulledFrames == 1_600 }
    await analyzer.cancelAndFinishNow()
    await #expect(throws: CancellationError.self) { try await finalized.value }
}

@Test func explicitStartTimesKeepGaps() async throws {
    let executor = FakeExecutor()
    let analyzer = SpeechAnalyzer(modules: [SpeechTranscriber(model: FakeModel(executor: executor))]
    )
    let input = AsyncStream<AnalyzerInput> { continuation in
        continuation.yield(input(frames: 1_600, start: time(16_000)))
        continuation.yield(input(frames: 1_600, start: time(32_000)))
        continuation.yield(input(frames: 1_600))
        continuation.finish()
    }
    let last = try await analyzer.analyzeSequence(input)
    #expect(last == time(35_199))
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    #expect(await executor.pulled.map(\.startTime) == [time(16_000), time(32_000), time(33_600)])
}

@Test func overlappingStartTimesAreRejected() async throws {
    let analyzer = SpeechAnalyzer(modules: [
        SpeechTranscriber(model: FakeModel(executor: FakeExecutor()))
    ])
    let input = AsyncStream<AnalyzerInput> { continuation in
        continuation.yield(input(frames: 16_000, start: .zero))
        continuation.yield(input(frames: 1_600, start: time(8_000)))
        continuation.finish()
    }
    await #expect(throws: SpeechRecognitionError.self) { try await analyzer.analyzeSequence(input) }
    #expect(await analyzer.state == .failed)
}

@Test func volatileResultsAreReplacedAndFinalsKept() async throws {
    let channel = ResultChannel<SpeechTranscriber.Result>()
    let volatile1 = SpeechTranscriber.Result(snapshot("Hel", start: 0, end: 8_000, final: false))
    let volatile2 = SpeechTranscriber.Result(
        snapshot("Hello wor", start: 0, end: 12_000, final: false))
    let final = SpeechTranscriber.Result(snapshot("Hello", start: 0, end: 8_000))
    let volatile3 = SpeechTranscriber.Result(
        snapshot("world", start: 8_000, end: 16_000, final: false))
    for result in [volatile1, volatile2, final, volatile3] { try channel.publish(result) }
    channel.finish()
    var values: [SpeechTranscriber.Result] = []
    while let next = try await channel.next() { values.append(next) }
    #expect(texts(values) == ["Hello wor", "Hello", "world"])
}

@Test func slowConsumerDoesNotOverflowOnVolatileResults() throws {
    let channel = ResultChannel<SpeechTranscriber.Result>()
    for index in 0..<(ResultChannel<SpeechTranscriber.Result>.maximumResults * 4) {
        let result = snapshot("partial \(index)", start: 0, end: Int64(index + 1), final: false)
        try channel.publish(SpeechTranscriber.Result(result))
    }
}

@Test func volatileRangeIsReported() async throws {
    let executor = FakeExecutor()
    let transcriber = SpeechTranscriber(
        model: FakeModel(executor: executor), reportingOptions: [.volatileResults])
    let ranges = Ranges()
    let (stream, continuation) = liveInput()
    let analyzer = SpeechAnalyzer(
        inputSequence: stream, modules: [transcriber],
        volatileRangeChangedHandler: { range, _, _ in Task { await ranges.append(range) } })
    continuation.yield(input(frames: 8_000))
    await eventually { await executor.pulledFrames == 8_000 }
    let range = CMTimeRange(start: .zero, end: time(8_000))
    await executor.reportVolatile(range)
    await eventually { await analyzer.volatileRange == range }
    await executor.reportVolatile(nil)
    await eventually { await analyzer.volatileRange == nil }
    await eventually { await ranges.values.count == 2 }
    continuation.finish()
    try await analyzer.finalizeAndFinishThroughEndOfInput()
}

private actor Ranges {
    var values: [CMTimeRange] = []
    func append(_ range: CMTimeRange) { values.append(range) }
}

@Test func cancelAnalysisReachesExecutor() async throws {
    let executor = FakeExecutor()
    let analyzer = SpeechAnalyzer(modules: [SpeechTranscriber(model: FakeModel(executor: executor))]
    )
    try await analyzer.prepareToAnalyze(in: nil)
    await analyzer.cancelAnalysis(before: time(8_000))
    await eventually { await executor.cancelledBefore == [time(8_000)] }
    await analyzer.cancelAndFinishNow()
}

@Test func contextReachesExecutor() async throws {
    let executor = FakeExecutor()
    let analyzer = SpeechAnalyzer(modules: [SpeechTranscriber(model: FakeModel(executor: executor))]
    )
    try await analyzer.prepareToAnalyze(in: nil)
    let context = AnalysisContext()
    context.contextualStrings[.general] = ["AnySpeechRecognizer"]
    try await analyzer.setContext(context)
    #expect(await analyzer.context === context)
    #expect(await executor.contexts.count == 1)
    await analyzer.cancelAndFinishNow()
}

@Test func finishAfterStopsWithoutFinalizing() async throws {
    let executor = FakeExecutor(deferred: [snapshot("Never.", start: 0, end: 8_000)])
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    async let results = collect(transcriber)
    let input = AsyncStream<AnalyzerInput> { continuation in
        continuation.yield(input(frames: 16_000))
        continuation.finish()
    }
    _ = try await analyzer.analyzeSequence(input)
    try await analyzer.finish(after: time(8_000))
    #expect(try await results.isEmpty)
    #expect(await executor.finalizeTimes.isEmpty)
    #expect(await analyzer.state == .finished)
}

@Test func addedModuleReceivesLaterInput() async throws {
    let firstExecutor = FakeExecutor()
    let secondExecutor = FakeExecutor()
    let first = SpeechTranscriber(model: FakeModel(executor: firstExecutor))
    let second = SpeechTranscriber(model: FakeModel(executor: secondExecutor))
    let analyzer = SpeechAnalyzer(modules: [first])
    let (stream, continuation) = liveInput()
    try await analyzer.start(inputSequence: stream)
    continuation.yield(input(frames: 1_600))
    await eventually { await firstExecutor.pulledFrames == 1_600 }
    try await analyzer.setModules([first, second])
    continuation.yield(input(frames: 1_600))
    await eventually { await secondExecutor.pulledFrames == 1_600 }
    #expect(await secondExecutor.pulled.first?.startTime == time(1_600))
    try await analyzer.setModules([second])
    await #expect(throws: Never.self) { try await collect(first) }
    continuation.finish()
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    #expect(await analyzer.modules.count == 1)
}

@Test func autonomousFileFinishesAfterFile() async throws {
    let (url, file) = try audioFile(frames: 16_000)
    defer { try? FileManager.default.removeItem(at: url) }
    let executor = FakeExecutor(snapshots: [snapshot("File.", start: 0, end: 16_000)])
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    async let results = collect(transcriber)
    let analyzer = try await SpeechAnalyzer(
        inputAudioFile: file, modules: [transcriber], finishAfterFile: true)
    #expect(texts(try await results) == ["File."])
    await eventually { await analyzer.state == .finished }
}

@Test func releasingAnAnalyzerEndsResults() async throws {
    let executor = FakeExecutor()
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    var analyzer: SpeechAnalyzer? = SpeechAnalyzer(modules: [transcriber])
    try await analyzer?.prepareToAnalyze(in: nil)
    let results = Task { try await collect(transcriber) }
    analyzer = nil
    #expect(try await results.value.isEmpty)
    await eventually { await executor.cancelCount == 1 }
}
