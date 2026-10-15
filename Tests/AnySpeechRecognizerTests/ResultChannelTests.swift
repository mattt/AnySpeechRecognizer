import Foundation
import Testing

@testable import AnySpeechRecognizer

private typealias Channel = ResultChannel<SpeechTranscriber.Result>

@Test func secondIteratorFailsAndFirstDrains() async throws {
    let transcriber = SpeechTranscriber(model: FakeModel(executor: FakeExecutor()))
    var first = transcriber.results.makeAsyncIterator()
    var second = transcriber.results.makeAsyncIterator()
    let result = SpeechTranscriber.Result(snapshot())
    try transcriber.channel.publish(result)
    transcriber.channel.finish()
    await #expect(throws: SpeechRecognitionError.self) { try await second.next() }
    #expect(try await first.next() == result)
    #expect(try await first.next() == nil)
}

@Test func cancellationClosesAnAwaitingIterator() async throws {
    let channel = Channel()
    let next = Task { try await channel.next() }
    next.cancel()
    await #expect(throws: CancellationError.self) { try await next.value }
    channel.finish()
    #expect(try await channel.next() == nil)
}

@Test func oversizedResultIsRejected() throws {
    let channel = Channel()
    let result = snapshot(String(repeating: "a", count: Channel.maximumTextBytes + 1))
    #expect(throws: SpeechRecognitionError.self) {
        try channel.publish(SpeechTranscriber.Result(result))
    }
}
