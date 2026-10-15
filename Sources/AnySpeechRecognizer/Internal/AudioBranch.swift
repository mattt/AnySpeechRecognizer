import AVFoundation
import CoreMedia
import Foundation
import Synchronization

/// One module's bounded input queue.
///
/// The producer waits for capacity, so the slowest module paces input.
/// Pulling a buffer marks its source range consumed for that module.
/// Each input operation ends with a boundary, which only segment-aware consumers observe.
final class AudioBranch: Sendable {
    struct Item: @unchecked Sendable {
        let audio: TimedAudioBuffer
        /// The end of the source input that produced this buffer.
        /// Conversion latency can make it later than `audio.endTime`.
        var sourceEnd: CMTime
    }

    /// The result of one pull.
    enum Pull: @unchecked Sendable {
        case audio(TimedAudioBuffer)
        /// The end of one input operation.
        case boundary
        /// The end of all input.
        case end
    }

    private enum Entry {
        case item(Item)
        case boundary
    }

    private struct Consumer {
        let continuation: CheckedContinuation<Pull, any Error>
        let observesBoundaries: Bool
    }

    private struct State {
        var queue: [Entry] = []
        var queuedItems = 0
        var finished = false
        var error: (any Error)?
        var consumer: Consumer?
        var producers: [CheckedContinuation<Void, any Error>] = []
        var pulledThrough: CMTime?
        var pullWaiters: [(UUID, CMTime, CheckedContinuation<Void, any Error>)] = []
    }

    let format: AVAudioFormat
    let capacity: Int
    private let state = Mutex(State())

    init(format: AVAudioFormat, capacity: Int = 8) {
        self.format = format
        self.capacity = capacity
    }

    /// The end of the source input the consumer has pulled.
    var pulledThrough: CMTime? { state.withLock { $0.pulledThrough } }
    /// Whether input has ended.
    var isFinished: Bool { state.withLock { $0.finished } }

    /// Sends a buffer, waiting while the queue is full.
    func send(_ item: Item) async throws {
        let capacity = capacity
        while true {
            let sent: Bool = try state.withLock { state in
                if let error = state.error { throw error }
                guard !state.finished else { throw CancellationError() }
                if state.queue.isEmpty, let consumer = state.consumer {
                    state.consumer = nil
                    Self.markPulled(item.sourceEnd, in: &state)
                    consumer.continuation.resume(returning: .audio(item.audio))
                    return true
                }
                guard state.queuedItems < capacity else { return false }
                state.queue.append(.item(item))
                state.queuedItems += 1
                return true
            }
            if sent { return }
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, any Error>) in
                state.withLock { state in
                    if let error = state.error {
                        continuation.resume(throwing: error)
                    } else if state.finished {
                        continuation.resume(throwing: CancellationError())
                    } else if state.queuedItems < capacity {
                        continuation.resume()
                    } else {
                        state.producers.append(continuation)
                    }
                }
            }
        }
    }

    /// Marks the end of one input operation.
    func sendBoundary() {
        state.withLock { state in
            guard !state.finished else { return }
            if state.queue.isEmpty, let consumer = state.consumer {
                guard consumer.observesBoundaries else { return }
                state.consumer = nil
                consumer.continuation.resume(returning: .boundary)
                return
            }
            if case .boundary = state.queue.last { return }
            state.queue.append(.boundary)
        }
    }

    /// Records source input that produced no output yet, such as conversion latency.
    /// The range counts as pulled with the last queued buffer, or now if none is queued.
    func advance(sourceEnd: CMTime) {
        state.withLock { state in
            for index in state.queue.indices.reversed() {
                guard case .item(var item) = state.queue[index] else { continue }
                item.sourceEnd = max(item.sourceEnd, sourceEnd)
                state.queue[index] = .item(item)
                return
            }
            Self.markPulled(sourceEnd, in: &state)
        }
    }

    /// Pulls the next buffer, skipping boundaries, or returns `nil` when input ends.
    func next() async throws -> TimedAudioBuffer? {
        switch try await pull(observingBoundaries: false) {
        case .audio(let audio): audio
        case .boundary, .end: nil
        }
    }

    /// Pulls the next buffer, boundary, or end of input.
    func pull(observingBoundaries: Bool) async throws -> Pull {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                state.withLock { state in
                    if Task.isCancelled {
                        continuation.resume(throwing: CancellationError())
                        return
                    }
                    while let entry = state.queue.first {
                        state.queue.removeFirst()
                        switch entry {
                        case .boundary:
                            guard observingBoundaries else { continue }
                            continuation.resume(returning: .boundary)
                        case .item(let item):
                            state.queuedItems -= 1
                            Self.markPulled(item.sourceEnd, in: &state)
                            if !state.producers.isEmpty {
                                state.producers.removeFirst().resume()
                            }
                            continuation.resume(returning: .audio(item.audio))
                        }
                        return
                    }
                    if let error = state.error {
                        continuation.resume(throwing: error)
                    } else if state.finished {
                        continuation.resume(returning: .end)
                    } else if state.consumer != nil {
                        continuation.resume(
                            throwing: SpeechRecognitionError.invalidSessionState(
                                "Concurrent input iteration is unsupported."))
                    } else {
                        state.consumer = Consumer(
                            continuation: continuation, observesBoundaries: observingBoundaries)
                    }
                }
            }
        } onCancel: {
            self.state.withLock { state in
                state.consumer?.continuation.resume(throwing: CancellationError())
                state.consumer = nil
            }
        }
    }

    private static func markPulled(_ sourceEnd: CMTime, in state: inout State) {
        if let pulled = state.pulledThrough, pulled >= sourceEnd { return }
        state.pulledThrough = sourceEnd
        let ready = state.pullWaiters.filter { $0.1 <= sourceEnd }
        state.pullWaiters.removeAll { $0.1 <= sourceEnd }
        for (_, _, waiter) in ready { waiter.resume() }
    }

    /// Waits until the consumer has pulled source input through a time.
    /// - Throws: `CancellationError` if input ends first.
    func waitUntilPulled(through time: CMTime) async throws {
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, any Error>) in
                state.withLock { state in
                    if let pulled = state.pulledThrough, pulled >= time {
                        continuation.resume()
                    } else if let error = state.error {
                        continuation.resume(throwing: error)
                    } else if state.finished && state.queuedItems == 0 {
                        continuation.resume(throwing: CancellationError())
                    } else if Task.isCancelled {
                        continuation.resume(throwing: CancellationError())
                    } else {
                        state.pullWaiters.append((id, time, continuation))
                    }
                }
            }
        } onCancel: {
            self.state.withLock { state in
                guard let index = state.pullWaiters.firstIndex(where: { $0.0 == id }) else {
                    return
                }
                state.pullWaiters.remove(at: index).2.resume(throwing: CancellationError())
            }
        }
    }

    /// Ends input. Queued buffers remain available unless an error is supplied.
    func finish(throwing error: (any Error)? = nil) {
        state.withLock { state in
            guard !state.finished else { return }
            state.finished = true
            if let error {
                state.error = error
                state.queue.removeAll()
                state.queuedItems = 0
            }
            let producers = state.producers
            state.producers.removeAll()
            for producer in producers { producer.resume(throwing: error ?? CancellationError()) }
            if state.queue.isEmpty, let consumer = state.consumer {
                state.consumer = nil
                if let error {
                    consumer.continuation.resume(throwing: error)
                } else {
                    consumer.continuation.resume(returning: .end)
                }
            }
            // Pending pulls can complete only from queued input.
            let pulled = state.pulledThrough
            var queuedEnd: CMTime?
            for entry in state.queue.reversed() {
                if case .item(let item) = entry {
                    queuedEnd = item.sourceEnd
                    break
                }
            }
            let waiters = state.pullWaiters
            state.pullWaiters.removeAll()
            for (id, time, waiter) in waiters {
                if error == nil, let queuedEnd, queuedEnd >= time {
                    state.pullWaiters.append((id, time, waiter))
                } else if let pulled, pulled >= time {
                    waiter.resume()
                } else {
                    waiter.resume(throwing: error ?? CancellationError())
                }
            }
        }
    }
}
