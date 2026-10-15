import AVFoundation
import CoreMedia
import Foundation
import Speech
import Synchronization

/// Callbacks from an engine to its analyzer.
struct EngineEvents: Sendable {
    /// Fails the session. Must not wait for the engine's run task.
    let fail: @Sendable (any Error) async -> Void
    /// Reports an engine's volatile range, or `nil` when none remains.
    let volatileRangeChanged: @Sendable (ObjectIdentifier, CMTimeRange?) async -> Void
}

/// Runs one or more modules on one input branch.
protocol AnalysisEngine: AnyObject, Sendable {
    nonisolated var id: ObjectIdentifier { get }
    /// Chooses the branch format, preferring the source format when the backend accepts it.
    func inputFormat(considering natural: AVAudioFormat?) async throws -> AVAudioFormat
    /// Validates configuration and loads local resources.
    func prepare(format: AVAudioFormat, context: AnalysisContext) async throws
    /// Consumes the branch until it ends.
    func run(_ branch: AudioBranch) async throws
    /// Returns when the engine has consumed input through `time`.
    func waitUntilConsumed(through time: CMTime, branch: AudioBranch) async throws
    /// Returns when results through `time` are final.
    func finalize(through time: CMTime, branch: AudioBranch) async throws
    /// Requests that analysis skip audio before a time.
    func cancelAnalysis(before time: CMTime) async
    /// Applies new context.
    func setContext(_ context: AnalysisContext) async throws
    /// Finalizes through `end`, ends input, waits for analysis, and closes results.
    func finishThroughEndOfInput(end: CMTime?, branch: AudioBranch, run: Task<Void, Never>)
        async throws
    /// Stops after `time` without finalizing, and closes results.
    func finish(after time: CMTime, branch: AudioBranch, run: Task<Void, Never>) async throws
    /// Stops work and closes results with an error, or discards them when `error` is `nil`.
    func cancel(error: (any Error)?) async
}

/// Serializes finalization requests so each waits for the previous one.
struct FinalizationQueue {
    private var tail: Task<Void, any Error>?

    mutating func enqueue(_ body: @escaping @Sendable () async throws -> Void) -> Task<
        Void, any Error
    > {
        let previous = tail
        let task = Task {
            _ = await previous?.result
            try await body()
        }
        tail = task
        return task
    }
}

/// Drives one model-backed module through its public executor.
actor ExecutorEngine: AnalysisEngine {
    nonisolated var id: ObjectIdentifier { binding.id }
    nonisolated let binding: ModuleBinding
    let model: any SpeechRecognitionModel
    let request: SpeechRecognitionRequest
    let executor: any SpeechRecognitionExecutor
    let events: EngineEvents
    private var branch: AudioBranch?
    private var finalization: CMTime?
    private var closed = false
    private var queue = FinalizationQueue()

    init(
        binding: ModuleBinding, model: any SpeechRecognitionModel,
        request: SpeechRecognitionRequest, events: EngineEvents
    ) {
        self.binding = binding
        self.model = model
        self.request = request
        self.executor = model.makeExecutor()
        self.events = events
    }

    func inputFormat(considering natural: AVAudioFormat?) async -> AVAudioFormat {
        let formats = await model.compatibleAudioFormats(for: request)
        if let natural, formats.contains(where: { Self.matches($0, natural) }) { return natural }
        return formats.first ?? AudioFormats.monoFloat16k
    }

    private static func matches(_ lhs: AVAudioFormat, _ rhs: AVAudioFormat) -> Bool {
        lhs.sampleRate == rhs.sampleRate && lhs.channelCount == rhs.channelCount
            && lhs.commonFormat == rhs.commonFormat && lhs.isInterleaved == rhs.isInterleaved
    }

    func prepare(format: AVAudioFormat, context: AnalysisContext) async throws {
        try request.validate(for: model.capabilities)
        var availability = await model.availability(for: request)
        // Assets that only improve a request, such as Parakeet's live encoder, do not block
        // analysis: the request proceeds when the assets that every request needs are ready.
        if availability == .missingAssets, await model.availability() == .ready {
            availability = .ready
        }
        switch availability {
        case .ready, .preparing: break
        case .missingAssets:
            throw SpeechRecognitionError.missingAssets(
                "Install assets for \(model.modelID) with AssetInventory before analysis.")
        case .unsupportedHardware:
            throw SpeechRecognitionError.unavailableProvider(
                "The device does not support \(model.modelID).")
        case .unsupportedLanguage:
            throw SpeechRecognitionError.unsupportedLanguage(
                "The language configuration is unsupported by \(model.modelID).")
        }
        try Task.checkCancellation()
        try await executor.prepare(request: request, format: format, context: context)
    }

    func run(_ branch: AudioBranch) async throws {
        self.branch = branch
        let id = id
        let events = events
        let (ranges, continuation) = AsyncStream<CMTimeRange?>.makeStream(
            bufferingPolicy: .bufferingNewest(1))
        let forwarding = Task {
            for await range in ranges { await events.volatileRangeChanged(id, range) }
        }
        defer {
            continuation.finish()
            forwarding.cancel()
        }
        let sink = SpeechRecognitionResultSink(
            publish: { [weak self] result in
                guard let self else { throw CancellationError() }
                try await self.publish(result)
            },
            volatile: { continuation.yield($0) }
        )
        try await executor.analyze(SpeechAnalysisInput(branch: branch), results: sink)
    }

    private func publish(_ result: SpeechRecognitionResult) async throws {
        guard !closed else { throw CancellationError() }
        do {
            try validate(result)
            try binding.deliver(result)
            finalization = max(
                finalization ?? result.resultsFinalizationTime, result.resultsFinalizationTime)
        } catch {
            let error = recognitionError(error)
            await events.fail(error)
            throw error
        }
    }

    private func validate(_ result: SpeechRecognitionResult) throws {
        let range = result.range
        // Converted timestamps can round past the source input by part of a frame.
        let tolerance = CMTime(value: 1, timescale: 1000)
        guard range.isValid, range.start.isNumeric, range.duration.isNumeric,
            range.duration >= .zero, range.start >= .zero,
            result.resultsFinalizationTime.isNumeric,
            let pulled = branch?.pulledThrough, range.end <= pulled + tolerance,
            finalization.map({ range.start >= $0 && result.resultsFinalizationTime >= $0 })
                ?? true,
            !result.isFinal || result.resultsFinalizationTime >= range.end
        else {
            throw SpeechRecognitionError.invalidResult(
                "A result has invalid, overlapping, or unfinalized source timing.")
        }
        try validateTiming(in: result.text, range: range)
        for alternative in result.alternatives { try validateTiming(in: alternative, range: range) }
    }

    func waitUntilConsumed(through time: CMTime, branch: AudioBranch) async throws {
        try await branch.waitUntilPulled(through: time)
    }

    func finalize(through time: CMTime, branch: AudioBranch) async throws {
        let executor = executor
        let task = queue.enqueue { [weak self] in
            try await branch.waitUntilPulled(through: time)
            guard let self, await !self.isFinal(through: time) else { return }
            try await executor.finalize(through: time)
        }
        try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    private func isFinal(through time: CMTime) -> Bool {
        finalization.map { $0 > time } ?? false
    }

    func cancelAnalysis(before time: CMTime) async { await executor.cancelAnalysis(before: time) }

    func setContext(_ context: AnalysisContext) async throws {
        try await executor.setContext(context)
    }

    func finishThroughEndOfInput(end: CMTime?, branch: AudioBranch, run: Task<Void, Never>)
        async throws
    {
        if let end { try await finalize(through: end, branch: branch) }
        branch.finish()
        await run.value
        guard !closed else { return }
        closed = true
        binding.finish(nil, false)
    }

    func finish(after time: CMTime, branch: AudioBranch, run: Task<Void, Never>) async throws {
        try await branch.waitUntilPulled(through: time)
        closed = true
        binding.finish(nil, false)
        await executor.cancel()
        branch.finish()
        run.cancel()
        await run.value
    }

    func cancel(error: (any Error)?) async {
        if !closed {
            closed = true
            binding.finish(error, error == nil)
        }
        await executor.cancel()
    }
}

/// Drives every Apple-backed module of one analyzer with one native analyzer.
actor NativeEngine: AnalysisEngine {
    nonisolated var id: ObjectIdentifier { ObjectIdentifier(self) }
    private var members: [(binding: ModuleBinding, member: NativeMember)]
    private let options: Speech.SpeechAnalyzer.Options?
    private let events: EngineEvents
    private var analyzer: Speech.SpeechAnalyzer?
    private var forwarders: [ObjectIdentifier: Task<Void, Never>] = [:]
    private var volatileTask: Task<Void, Never>?
    private var closed = false
    private var consumedThrough: CMTime?
    private var consumptionWaiters: [(CMTime, CheckedContinuation<Void, any Error>)] = []

    init(
        members: [(binding: ModuleBinding, member: NativeMember)],
        options: Speech.SpeechAnalyzer.Options?, events: EngineEvents
    ) {
        self.members = members
        self.options = options
        self.events = events
    }

    private var natives: [any Speech.SpeechModule] { members.map(\.member.module) }

    func inputFormat(considering natural: AVAudioFormat?) async throws -> AVAudioFormat {
        guard
            let format = await Speech.SpeechAnalyzer.bestAvailableAudioFormat(
                compatibleWith: natives, considering: natural)
        else {
            throw SpeechRecognitionError.missingAssets(
                "Apple Speech has no installed format for these modules.")
        }
        return format
    }

    func prepare(format: AVAudioFormat, context: AnalysisContext) async throws {
        switch await Speech.AssetInventory.status(forModules: natives) {
        case .installed: break
        case .unsupported:
            throw SpeechRecognitionError.unsupportedLanguage(
                "Apple Speech does not support this module configuration.")
        default:
            throw SpeechRecognitionError.missingAssets(
                "Install Apple Speech assets with AssetInventory before analysis.")
        }
        try Task.checkCancellation()
        guard !closed else { throw CancellationError() }
        let analyzer = Speech.SpeechAnalyzer(modules: natives, options: options)
        let (stream, continuation) = AsyncStream<CMTimeRange?>.makeStream(
            bufferingPolicy: .bufferingNewest(1))
        await analyzer.setVolatileRangeChangedHandler { range, _, _ in
            continuation.yield(range.duration > .zero ? range : nil)
        }
        let events = events
        let id = id
        volatileTask = Task {
            for await range in stream { await events.volatileRangeChanged(id, range) }
        }
        try await analyzer.setContext(context)
        try await analyzer.prepareToAnalyze(in: format)
        self.analyzer = analyzer
        for (binding, member) in members { startForwarding(binding, member) }
    }

    private func startForwarding(_ binding: ModuleBinding, _ member: NativeMember) {
        let events = events
        forwarders[binding.id] = Task { [weak self] in
            do {
                try await member.forward()
                binding.finish(nil, false)
            } catch {
                guard let self, await !self.closed else { return }
                let error = recognitionError(error)
                binding.finish(error, false)
                await events.fail(error)
            }
        }
    }

    /// Runs one native analysis per input operation.
    ///
    /// Apple's analyzer can hold the end of an open input sequence until more input arrives,
    /// so each package input operation ends its native sequence.
    /// A new native sequence starts only when audio arrives.
    func run(_ branch: AudioBranch) async throws {
        guard let analyzer else { throw CancellationError() }
        while true {
            let first: TimedAudioBuffer
            switch try await branch.pull(observingBoundaries: true) {
            case .audio(let audio): first = audio
            case .boundary: continue
            case .end: return
            }
            let segment = NativeInput(branch: branch, first: first)
            _ = try await analyzer.analyzeSequence(segment)
            markConsumed(through: segment.end)
            if segment.reachedEnd { return }
        }
    }

    private func markConsumed(through end: CMTime) {
        consumedThrough = max(consumedThrough ?? end, end)
        let ready = consumptionWaiters.filter { $0.0 <= end }
        consumptionWaiters.removeAll { $0.0 <= end }
        for (_, waiter) in ready { waiter.resume() }
    }

    /// Waits until a native analysis that read input through `time` has returned.
    func waitUntilConsumed(through time: CMTime, branch: AudioBranch) async throws {
        if let consumedThrough, consumedThrough >= time { return }
        guard !closed else { throw CancellationError() }
        try await withCheckedThrowingContinuation { consumptionWaiters.append((time, $0)) }
    }

    private func failConsumptionWaiters() {
        let waiters = consumptionWaiters
        consumptionWaiters.removeAll()
        for (_, waiter) in waiters { waiter.resume(throwing: CancellationError()) }
    }

    func finalize(through time: CMTime, branch: AudioBranch) async throws {
        guard let analyzer else { throw CancellationError() }
        try await analyzer.finalize(through: time)
    }

    func cancelAnalysis(before time: CMTime) async {
        await analyzer?.cancelAnalysis(before: time)
    }

    func setContext(_ context: AnalysisContext) async throws {
        try await analyzer?.setContext(context)
    }

    /// Replaces the native modules, keeping the native analyzer's timeline.
    func setMembers(_ newMembers: [(binding: ModuleBinding, member: NativeMember)]) async throws {
        let removed = members.filter { old in
            !newMembers.contains { $0.binding.id == old.binding.id }
        }
        let added = newMembers.filter { new in !members.contains { $0.binding.id == new.binding.id }
        }
        members = newMembers
        if let analyzer {
            try await analyzer.setModules(natives)
            for (binding, member) in added { startForwarding(binding, member) }
        }
        for (binding, _) in removed {
            await forwarders.removeValue(forKey: binding.id)?.value
            binding.finish(nil, false)
        }
    }

    var isEmpty: Bool { members.isEmpty }

    func finishThroughEndOfInput(end: CMTime?, branch: AudioBranch, run: Task<Void, Never>)
        async throws
    {
        branch.finish()
        await run.value
        try await analyzer?.finalizeAndFinishThroughEndOfInput()
        await settle()
    }

    func finish(after time: CMTime, branch: AudioBranch, run: Task<Void, Never>) async throws {
        try await analyzer?.finish(after: time)
        branch.finish()
        await run.value
        await settle()
    }

    private func settle() async {
        for forwarder in forwarders.values { await forwarder.value }
        forwarders.removeAll()
        closed = true
        failConsumptionWaiters()
        for (binding, _) in members { binding.finish(nil, false) }
        volatileTask?.cancel()
    }

    func cancel(error: (any Error)?) async {
        guard !closed else { return }
        closed = true
        failConsumptionWaiters()
        for (binding, _) in members { binding.finish(error, error == nil) }
        await analyzer?.cancelAndFinishNow()
        for forwarder in forwarders.values { forwarder.cancel() }
        for forwarder in forwarders.values { await forwarder.value }
        forwarders.removeAll()
        volatileTask?.cancel()
    }
}

/// Presents one input segment to Apple's analyzer as native input with explicit source times.
struct NativeInput: AsyncSequence, Sendable {
    typealias Element = AnalyzerInput
    typealias Failure = any Error
    let branch: AudioBranch
    let first: TimedAudioBuffer
    private let ended = EndFlag()
    private let last = LastEnd()

    init(branch: AudioBranch, first: TimedAudioBuffer) {
        self.branch = branch
        self.first = first
    }

    /// Whether the segment ended because all input ended.
    var reachedEnd: Bool { ended.value.withLock { $0 } }
    /// The source end of the last buffer delivered in this segment.
    var end: CMTime { last.value.withLock { $0 } ?? first.endTime }

    func makeAsyncIterator() -> Iterator { Iterator(input: self) }

    struct Iterator: AsyncIteratorProtocol {
        let input: NativeInput
        var pending: TimedAudioBuffer?

        init(input: NativeInput) {
            self.input = input
            self.pending = input.first
        }

        mutating func next() async throws -> AnalyzerInput? {
            if let audio = pending {
                pending = nil
                input.last.value.withLock { $0 = audio.endTime }
                return AnalyzerInput(buffer: audio.buffer, bufferStartTime: audio.startTime)
            }
            switch try await input.branch.pull(observingBoundaries: true) {
            case .audio(let audio):
                input.last.value.withLock { $0 = audio.endTime }
                return AnalyzerInput(buffer: audio.buffer, bufferStartTime: audio.startTime)
            case .boundary:
                return nil
            case .end:
                input.ended.value.withLock { $0 = true }
                return nil
            }
        }
    }
}

/// A shared flag that records whether a native segment reached the end of input.
private final class EndFlag: Sendable {
    let value = Mutex(false)
}

/// The end time of the last buffer a native segment delivered.
private final class LastEnd: Sendable {
    let value = Mutex<CMTime?>(nil)
}
