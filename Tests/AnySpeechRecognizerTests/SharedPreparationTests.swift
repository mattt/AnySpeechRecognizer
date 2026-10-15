import Foundation
import Testing

@testable import AnySpeechRecognizer

@Test func lastPreparationWaiterWaitsForResourceRelease() async throws {
    let work = SharedPreparation<Int>(retainValue: true)
    let entered = Gate()
    let release = Gate()
    let settled = Gate()
    let waiter = Task {
        do {
            return try await work.value {
                await entered.open()
                await release.wait()
                return 42
            }
        } catch {
            await settled.open()
            throw error
        }
    }
    await entered.wait()
    waiter.cancel()
    while work.waiterCount != 0 { await Task.yield() }
    #expect(work.isRunning)
    #expect(await settled.isOpen == false)
    await release.open()
    await #expect(throws: CancellationError.self) { try await waiter.value }
    #expect(work.isRunning == false)
}
