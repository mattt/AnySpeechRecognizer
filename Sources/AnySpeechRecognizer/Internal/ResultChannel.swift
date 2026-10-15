import CoreMedia
import Foundation
import Synchronization

/// A result that the channel can size and supersede.
protocol ChannelResult: SpeechModuleResult, Sendable {
    /// The UTF-8 size of the result's text and alternatives.
    var textByteCount: Int { get }
}

/// A finite queue with synchronous cancellation and one waiting consumer.
///
/// Final results are never dropped.
/// An unread volatile result is removed when a newer result supersedes its range.
final class ResultChannel<Element: ChannelResult>: Sendable {
    static var maximumResults: Int { 256 }
    static var maximumTextBytes: Int { 4 * 1024 * 1024 }

    private struct State {
        var claimed = false
        var attached = false
        var queue: [Element] = []
        var textBytes = 0
        var finished = false
        var error: (any Error)?
        var waiter: (UUID, CheckedContinuation<Element?, any Error>)?
    }
    private let state = Mutex(State())

    func attach() -> Bool {
        state.withLock { state in
            guard !state.attached else { return false }
            state.attached = true
            return true
        }
    }

    func detach() {
        state.withLock { $0.attached = false }
    }

    func claimIterator() -> Bool {
        state.withLock { state in
            guard !state.claimed else { return false }
            state.claimed = true
            return true
        }
    }

    var isFinished: Bool { state.withLock { $0.finished } }

    func publish(_ result: Element) throws {
        let bytes = result.textByteCount
        try state.withLock { state in
            if let error = state.error { throw error }
            guard !state.finished else { throw CancellationError() }
            guard bytes <= Self.maximumTextBytes else {
                throw SpeechRecognitionError.bufferOverflow
            }
            if let (_, waiter) = state.waiter {
                state.waiter = nil
                waiter.resume(returning: result)
                return
            }
            state.queue.removeAll { queued in
                guard !queued.isFinal, Self.supersedes(result, queued) else { return false }
                state.textBytes -= queued.textByteCount
                return true
            }
            guard state.queue.count < Self.maximumResults,
                state.textBytes + bytes <= Self.maximumTextBytes
            else {
                // A later result supersedes a skipped volatile result.
                if !result.isFinal { return }
                throw SpeechRecognitionError.bufferOverflow
            }
            state.queue.append(result)
            state.textBytes += bytes
        }
    }

    /// Whether `newer` replaces the unread volatile result `older`.
    private static func supersedes(_ newer: Element, _ older: Element) -> Bool {
        if !newer.isFinal { return newer.range.start <= older.range.start }
        return newer.range.start <= older.range.start && newer.range.end >= older.range.end
    }

    func finish(throwing error: (any Error)? = nil, discardBuffered: Bool = false) {
        state.withLock { state in
            guard !state.finished else { return }
            state.finished = true
            state.error = error
            if error != nil || discardBuffered {
                state.queue.removeAll()
                state.textBytes = 0
            }
            if let (_, waiter) = state.waiter {
                state.waiter = nil
                if let error {
                    waiter.resume(throwing: error)
                } else {
                    waiter.resume(returning: nil)
                }
            }
        }
    }

    func next() async throws -> Element? {
        let id = UUID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                state.withLock { state in
                    if Task.isCancelled {
                        continuation.resume(throwing: CancellationError())
                    } else if let error = state.error {
                        continuation.resume(throwing: error)
                    } else if !state.queue.isEmpty {
                        let result = state.queue.removeFirst()
                        state.textBytes -= result.textByteCount
                        continuation.resume(returning: result)
                    } else if state.finished {
                        continuation.resume(returning: nil)
                    } else if state.waiter != nil {
                        continuation.resume(
                            throwing: SpeechRecognitionError.invalidSessionState(
                                "Concurrent next() calls are unsupported."))
                    } else {
                        state.waiter = (id, continuation)
                    }
                }
            }
        } onCancel: {
            self.state.withLock { state in
                guard let (waitingID, waiter) = state.waiter, waitingID == id else { return }
                state.waiter = nil
                waiter.resume(throwing: CancellationError())
            }
        }
    }
}

/// Results from one module, with one permitted iterator.
public struct SpeechModuleResults<Element: Sendable>: AsyncSequence, Sendable {
    public typealias Failure = any Error
    private let pull: @Sendable () async throws -> Element?
    private let claim: @Sendable () -> Bool

    init(channel: ResultChannel<Element>) where Element: ChannelResult {
        self.pull = { try await channel.next() }
        self.claim = { channel.claimIterator() }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(pull: pull, permitted: claim())
    }

    /// An iterator that fails if another iterator already owns the sequence.
    public struct AsyncIterator: AsyncIteratorProtocol {
        fileprivate let pull: @Sendable () async throws -> Element?
        fileprivate let permitted: Bool

        public mutating func next() async throws -> Element? {
            guard permitted else {
                throw SpeechRecognitionError.invalidSessionState(
                    "A module permits only one result iterator.")
            }
            return try await pull()
        }
    }
}
