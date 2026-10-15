import Foundation
import Synchronization

/// Owns shared work until it settles, with cancellation for each waiter.
final class SharedPreparation<Value: Sendable>: Sendable {
    private struct State {
        var task: Task<Void, Never>?
        var cached: Value?
        var waiters: [UUID: CheckedContinuation<Value, any Error>] = [:]
    }
    private let state = Mutex(State())
    private let retainValue: Bool

    init(retainValue: Bool) { self.retainValue = retainValue }
    var isRunning: Bool { state.withLock { $0.task != nil } }
    var waiterCount: Int { state.withLock { $0.waiters.count } }

    func value(operation: @escaping @Sendable () async throws -> Value) async throws -> Value {
        do {
            let value = try await wait(operation: operation)
            try Task.checkCancellation()
            return value
        } catch {
            // The last cancelled waiter waits for noninterruptible loading or
            // installation to release resources. Other waiters keep it alive.
            let unused = state.withLock { state in
                state.waiters.isEmpty ? state.task : nil
            }
            if Task.isCancelled { await unused?.value }
            throw error
        }
    }

    private func wait(operation: @escaping @Sendable () async throws -> Value) async throws -> Value
    {
        let id = UUID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                state.withLock { state in
                    if Task.isCancelled {
                        continuation.resume(throwing: CancellationError())
                    } else if let cached = state.cached {
                        continuation.resume(returning: cached)
                    } else {
                        state.waiters[id] = continuation
                        if state.task == nil {
                            state.task = Task {
                                let result: Swift.Result<Value, any Error>
                                do {
                                    let value = try await operation()
                                    try Task.checkCancellation()
                                    result = .success(value)
                                } catch { result = .failure(error) }
                                self.complete(result)
                            }
                        }
                    }
                }
            }
        } onCancel: {
            self.state.withLock { state in
                if let waiter = state.waiters.removeValue(forKey: id) {
                    waiter.resume(throwing: CancellationError())
                }
                if state.waiters.isEmpty { state.task?.cancel() }
            }
        }
    }

    private func complete(_ result: Swift.Result<Value, any Error>) {
        state.withLock { state in
            state.task = nil
            if retainValue, case .success(let value) = result { state.cached = value }
            let waiters = state.waiters.values
            state.waiters.removeAll()
            for waiter in waiters { waiter.resume(with: result) }
        }
    }
}
