import Foundation
import Testing

@testable import AnySpeechRecognizer

@Test(arguments: [true, false]) func cancellationSettlesActiveWork(duringPreparation: Bool)
    async throws
{
    let (url, file) = try audioFile()
    defer { try? FileManager.default.removeItem(at: url) }
    let gate = Gate()
    let executor = FakeExecutor(
        snapshots: [snapshot()], prepareGate: duringPreparation ? gate : nil,
        analyzeGate: duringPreparation ? nil : gate
    )
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    let input = Task { try await analyzer.analyzeSequence(from: file) }
    if duringPreparation {
        await executor.prepareEntered.wait()
    } else {
        await executor.analyzeEntered.wait()
    }
    await analyzer.cancelAndFinishNow()
    await analyzer.cancelAndFinishNow()
    await #expect(throws: CancellationError.self) { try await input.value }
    #expect(try await collect(transcriber).isEmpty)
    #expect(await executor.cancelCount == 1)
    #expect(await analyzer.state == .cancelled)
}

@Test func consumerCancellationDoesNotCancelAnotherSession() async throws {
    let (firstURL, firstFile) = try audioFile()
    let (secondURL, secondFile) = try audioFile()
    defer {
        try? FileManager.default.removeItem(at: firstURL)
        try? FileManager.default.removeItem(at: secondURL)
    }
    let gate = Gate()
    let firstExecutor = FakeExecutor(analyzeGate: gate)
    let firstTranscriber = SpeechTranscriber(model: FakeModel(executor: firstExecutor))
    let first = SpeechAnalyzer(modules: [firstTranscriber])
    let secondTranscriber = SpeechTranscriber(
        model: FakeModel(executor: FakeExecutor(snapshots: [snapshot("Second.")])))
    let second = SpeechAnalyzer(modules: [secondTranscriber])
    let firstInput = Task { try await first.analyzeSequence(from: firstFile) }
    await firstExecutor.analyzeEntered.wait()
    let firstConsumer = Task { try await collect(firstTranscriber) }
    firstConsumer.cancel()
    await #expect(throws: CancellationError.self) { try await firstConsumer.value }
    await first.cancelAndFinishNow()
    await #expect(throws: CancellationError.self) { try await firstInput.value }
    async let otherResults = collect(secondTranscriber)
    _ = try await second.analyzeSequence(from: secondFile)
    try await second.finalizeAndFinishThroughEndOfInput()
    #expect(try await otherResults.map { String($0.text.characters) } == ["Second."])
    #expect(await second.state == .finished)
}

@Test func sharedPreparationKeepsOtherWaiterAlive() async throws {
    let work = SharedPreparation<Int>(retainValue: true)
    let entered = Gate()
    let finish = Gate()
    let operation: @Sendable () async throws -> Int = {
        await entered.open()
        await finish.wait()
        try Task.checkCancellation()
        return 42
    }
    let first = Task { try await work.value(operation: operation) }
    await entered.wait()
    // Register the second waiter before cancelling the first.
    let second = Task {
        return try await work.value(operation: operation)
    }
    while work.waiterCount < 2 { await Task.yield() }
    first.cancel()
    await #expect(throws: CancellationError.self) { try await first.value }
    #expect(work.isRunning)
    await finish.open()
    #expect(try await second.value == 42)
    #expect(try await work.value { throw TestFailure.backend } == 42)
}

@Test(arguments: [true, false]) func taskCancellationRequestsBackendCleanup(duringPreparation: Bool)
    async throws
{
    let (url, file) = try audioFile()
    defer { try? FileManager.default.removeItem(at: url) }
    let gate = Gate()
    let executor = FakeExecutor(
        prepareGate: duringPreparation ? gate : nil,
        analyzeGate: duringPreparation ? nil : gate
    )
    let transcriber = SpeechTranscriber(model: FakeModel(executor: executor))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    let input = Task { try await analyzer.analyzeSequence(from: file) }
    if duringPreparation {
        await executor.prepareEntered.wait()
    } else {
        await executor.analyzeEntered.wait()
    }
    input.cancel()
    await #expect(throws: CancellationError.self) { try await input.value }
    #expect(await executor.cancelCount == 1)
    #expect(await analyzer.state == .cancelled)
    #expect(try await collect(transcriber).isEmpty)
}
