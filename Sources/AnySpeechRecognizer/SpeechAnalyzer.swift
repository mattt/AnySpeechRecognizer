import AVFoundation
import CoreMedia
import Foundation
import Speech
import Synchronization

/// Coordinates audio input, analysis modules, and session lifecycle.
///
/// The analyzer reads input once, converts it to each module's format,
/// and keeps every result on the source timeline.
/// Controls such as ``finalize(through:)`` refer to source times,
/// so they can run while another task supplies input.
public final actor SpeechAnalyzer {
    /// Native analysis options: task priority, model retention, and resource limits.
    public typealias Options = Speech.SpeechAnalyzer.Options

    /// The session's lifecycle state. This is a package extension.
    public enum State: String, Sendable, Codable {
        /// No input has been analyzed.
        case configured
        /// Modules are loading local resources.
        case preparing
        /// An input sequence is being read.
        case analyzing
        /// No input is being read; more input, finalization, or finishing can follow.
        case draining
        /// Finishing completed and results are closed.
        case finished
        /// An error terminated the session.
        case failed
        /// Cancellation closed result delivery.
        case cancelled
    }

    /// The modules that receive input.
    public private(set) var modules: [any SpeechModule]
    /// The audio range whose results can still change, or `nil` when none can.
    public private(set) var volatileRange: CMTimeRange?
    /// The session's lifecycle state. This is a package extension.
    public private(set) var state: State = .configured

    private let options: Options?
    private var contextValue: AnalysisContext
    private var volatileHandler: ((CMTimeRange, Bool, Bool) -> Void)?
    private var bindings: [ModuleBinding] = []
    private var configurationError: (any Error)?
    private var slots: [Slot] = []
    private var native: NativeEngine?
    private var prepared = false
    private var preparation: Task<Void, any Error>?
    private var timeline = InputTimeline(lastEnd: nil)
    private var lastInputFormat: AVAudioFormat?
    private var engineRanges: [ObjectIdentifier: CMTimeRange] = [:]
    private var failure: (any Error)?
    private var cleanup: Task<Void, Never>?
    private var finishing = false
    private var controls: Task<Void, Never>?
    private var preparingEngines: [any AnalysisEngine] = []
    private var inputActive = false
    private var preparationWaiters: [CheckedContinuation<Void, any Error>] = []
    private let input = Mutex<Task<CMTime?, any Error>?>(nil)

    /// An engine with its input branch, converter, and analysis task.
    private struct Slot {
        let engine: any AnalysisEngine
        let input: BranchInput
        let run: Task<Void, Never>
        var moduleIDs: Set<ObjectIdentifier>
    }

    /// Creates an analyzer without loading assets or performing inference.
    /// - Parameters:
    ///   - modules: Package modules. Native Speech modules are not accepted.
    ///   - options: Analysis options, or `nil` for defaults.
    public init(modules: [any SpeechModule], options: Options? = nil) {
        self.modules = modules
        self.options = options
        self.contextValue = AnalysisContext()
        (self.bindings, self.configurationError) = Self.bind(modules)
    }

    /// Creates an analyzer and starts analyzing an input sequence autonomously.
    /// - Parameters:
    ///   - inputSequence: Audio with optional explicit start times.
    ///   - modules: Package modules.
    ///   - options: Analysis options, or `nil` for defaults.
    ///   - analysisContext: Context for the modules.
    ///   - volatileRangeChangedHandler: Called when the volatile range changes.
    public init<InputSequence: AsyncSequence & Sendable>(
        inputSequence: InputSequence, modules: [any SpeechModule], options: Options? = nil,
        analysisContext: AnalysisContext = .init(),
        volatileRangeChangedHandler:
            sending (
                (
                    _ range: CMTimeRange, _ changedStart: Bool, _ changedEnd: Bool
                ) -> Void
            )? = nil
    ) where InputSequence.Element == AnalyzerInput {
        self.modules = modules
        self.options = options
        self.contextValue = analysisContext
        self.volatileHandler = volatileRangeChangedHandler
        (self.bindings, self.configurationError) = Self.bind(modules)
        self.inputActive = true
        let task = Task<CMTime?, any Error> { [self] in
            try await self.consume(.sequence(AnyInputSequence(inputSequence)))
        }
        input.withLock { $0 = task }
        Task { [self] in await self.watchAutonomousInput(task, finishAfter: false) }
    }

    /// Creates an analyzer and starts analyzing a file autonomously.
    /// - Parameters:
    ///   - inputAudioFile: The file, read from its current position.
    ///   - modules: Package modules.
    ///   - options: Analysis options, or `nil` for defaults.
    ///   - analysisContext: Context for the modules.
    ///   - finishAfterFile: Whether to finalize and finish after the file.
    ///   - volatileRangeChangedHandler: Called when the volatile range changes.
    public init(
        inputAudioFile: AVAudioFile, modules: [any SpeechModule], options: Options? = nil,
        analysisContext: AnalysisContext = .init(), finishAfterFile: Bool = false,
        volatileRangeChangedHandler:
            sending (
                (
                    _ range: CMTimeRange, _ changedStart: Bool, _ changedEnd: Bool
                ) -> Void
            )? = nil
    ) async throws {
        self.modules = modules
        self.options = options
        self.contextValue = analysisContext
        self.volatileHandler = volatileRangeChangedHandler
        (self.bindings, self.configurationError) = Self.bind(modules)
        try await start(inputAudioFile: inputAudioFile, finishAfterFile: finishAfterFile)
    }

    /// Releasing an analyzer without finishing it ends input and closes its modules' results.
    deinit {
        input.withLock { $0?.cancel() }
        for slot in slots {
            slot.input.branch.finish(throwing: CancellationError())
            slot.run.cancel()
            let engine = slot.engine
            Task { await engine.cancel(error: nil) }
        }
        for binding in bindings { binding.finish(nil, false) }
    }

    /// Claims each module for this analyzer.
    private static func bind(_ modules: [any SpeechModule]) -> ([ModuleBinding], (any Error)?) {
        var bindings: [ModuleBinding] = []
        var error: (any Error)?
        for module in modules {
            guard let module = module as? any AnalyzerModule else {
                error = SpeechRecognitionError.unsupportedConfiguration(
                    "\(type(of: module)) is not a package speech module.")
                continue
            }
            let binding = module.binding
            guard binding.attach() else {
                error =
                    error
                    ?? SpeechRecognitionError.invalidSessionState(
                        "A module already belongs to another analyzer.")
                continue
            }
            bindings.append(binding)
        }
        if error != nil {
            for binding in bindings { binding.finish(error, false) }
        }
        return (bindings, error)
    }

    // MARK: Preparation

    /// Loads local resources for the modules without downloading assets.
    /// Analysis prepares automatically if this method has not been called.
    /// - Parameter audioFormat: The expected input format, or `nil` if unknown.
    /// - Throws: A configuration, asset, or backend error.
    public func prepareToAnalyze(in audioFormat: AVAudioFormat?) async throws {
        try await prepareToAnalyze(in: audioFormat, withProgressReadyHandler: nil)
    }

    /// Loads local resources and reports preparation progress.
    /// - Parameters:
    ///   - audioFormat: The expected input format, or `nil` if unknown.
    ///   - progressReadyHandler: Receives a progress object for preparation.
    /// - Throws: A configuration, asset, or backend error.
    public func prepareToAnalyze(
        in audioFormat: AVAudioFormat?,
        withProgressReadyHandler progressReadyHandler: sending ((Progress) -> Void)?
    ) async throws {
        try checkUsable()
        let progress = Progress(totalUnitCount: 1)
        progressReadyHandler?(progress)
        try await ensurePrepared(natural: audioFormat)
        progress.completedUnitCount = 1
    }

    private func ensurePrepared(natural: AVAudioFormat?) async throws {
        if prepared { return }
        if let preparation { return try await settle(preparation) }
        try checkUsable()
        let task = Task<Void, any Error> { try await self.prepare(natural: natural) }
        preparation = task
        try await settle(task)
    }

    /// Waits for shared work, cancelling the session if the waiting task is cancelled.
    private func settle(_ task: Task<Void, any Error>) async throws {
        do {
            try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                Task { await self.cancelAndFinishNow() }
            }
        } catch {
            if let failure { throw failure }
            if state == .cancelled || Task.isCancelled { throw CancellationError() }
            throw error
        }
        try checkUsable()
    }

    private func prepare(natural: AVAudioFormat?) async throws {
        if let configurationError { throw await fail(configurationError) }
        guard !bindings.isEmpty else {
            throw await fail(
                SpeechRecognitionError.unsupportedConfiguration(
                    "An analyzer requires at least one module."))
        }
        if state == .configured { state = .preparing }
        do {
            for binding in bindings {
                switch binding.backend {
                case .model(let model, let request): try request.validate(for: model.capabilities)
                case .native: break
                }
            }
            try await addEngines(for: bindings, natural: natural)
            try checkUsable()
            prepared = true
            if state == .preparing { state = .configured }
            resumePreparationWaiters(throwing: nil)
        } catch {
            throw await fail(error)
        }
    }

    private var events: EngineEvents {
        EngineEvents(
            fail: { [weak self] error in _ = await self?.fail(error) },
            volatileRangeChanged: { [weak self] id, range in
                await self?.updateVolatileRange(id, range)
            })
    }

    /// Creates, prepares, and starts engines for new bindings.
    private func addEngines(for bindings: [ModuleBinding], natural: AVAudioFormat?) async throws {
        var engines: [(any AnalysisEngine, Set<ObjectIdentifier>)] = []
        let natives = bindings.compactMap {
            binding -> (binding: ModuleBinding, member: NativeMember)? in
            if case .native(let member) = binding.backend { return (binding, member) }
            return nil
        }
        if !natives.isEmpty {
            if let native {
                let current = slots.first { $0.engine.id == native.id }
                let all = (current?.moduleIDs ?? []).union(natives.map(\.binding.id))
                try await native.setMembers(
                    self.bindings.filter { all.contains($0.id) }.compactMap { binding in
                        if case .native(let member) = binding.backend { return (binding, member) }
                        return nil
                    })
                if let index = slots.firstIndex(where: { $0.engine.id == native.id }) {
                    slots[index].moduleIDs = all
                }
            } else {
                let engine = NativeEngine(members: natives, options: options, events: events)
                native = engine
                engines.append((engine, Set(natives.map(\.binding.id))))
            }
        }
        for binding in bindings {
            switch binding.backend {
            case .model(let model, let request):
                let engine = ExecutorEngine(
                    binding: binding, model: model, request: request, events: events)
                engines.append((engine, [binding.id]))
            case .native: break
            }
        }
        let context = contextValue
        preparingEngines = engines.map(\.0)
        defer { preparingEngines.removeAll() }
        let prepared = try await withThrowingTaskGroup(
            of: (Int, AVAudioFormat).self, returning: [(Int, AVAudioFormat)].self
        ) { group in
            for (index, (engine, _)) in engines.enumerated() {
                group.addTask {
                    let format = try await engine.inputFormat(considering: natural)
                    try await engine.prepare(format: format, context: context)
                    return (index, format)
                }
            }
            return try await group.reduce(into: []) { $0.append($1) }
        }
        try checkUsable()
        for (index, format) in prepared.sorted(by: { $0.0 < $1.0 }) {
            let (engine, ids) = engines[index]
            let input = BranchInput(format: format)
            let branch = input.branch
            let run = Task(priority: options?.priority) { [weak self] in
                do {
                    try await engine.run(branch)
                    if !branch.isFinished {
                        _ = await self?.fail(
                            SpeechRecognitionError.invalidSessionState(
                                "A module stopped reading input before input ended."))
                    }
                } catch {
                    _ = await self?.fail(error)
                }
            }
            slots.append(Slot(engine: engine, input: input, run: run, moduleIDs: ids))
        }
    }

    // MARK: Input

    /// Analyzes an input sequence and returns after it ends.
    ///
    /// Results continue after this method returns; finalize or finish to close them.
    /// Buffers without a start time continue from the previous buffer's end.
    /// - Parameter inputSequence: Audio with optional explicit start times.
    /// - Returns: The time of the last input sample, or `nil` if the sequence was empty.
    /// - Throws: A configuration, audio, or backend error, or `CancellationError`.
    public func analyzeSequence<InputSequence: AsyncSequence & Sendable>(
        _ inputSequence: InputSequence
    ) async throws -> CMTime? where InputSequence.Element == AnalyzerInput {
        try await runInput(.sequence(AnyInputSequence(inputSequence)))
    }

    /// Analyzes a file from its current frame position through its end.
    ///
    /// The first input of a session keeps the file's frame position on the timeline.
    /// Later input continues from the previous input's end.
    /// - Parameter audioFile: The file. Do not use it until analysis returns.
    /// - Returns: The time of the last input sample, or `nil` for an empty file.
    /// - Throws: A configuration, audio, or backend error, or `CancellationError`.
    public func analyzeSequence(from audioFile: AVAudioFile) async throws -> CMTime? {
        try await runInput(.file(FileSource(file: audioFile)))
    }

    /// Starts analyzing an input sequence autonomously and returns immediately.
    /// - Parameter inputSequence: Audio with optional explicit start times.
    /// - Throws: An error if the session cannot accept input.
    public func start<InputSequence: AsyncSequence & Sendable>(inputSequence: InputSequence)
        async throws where InputSequence.Element == AnalyzerInput
    {
        try startInput(.sequence(AnyInputSequence(inputSequence)), finishAfter: false)
    }

    /// Starts analyzing a file autonomously and returns immediately.
    /// - Parameters:
    ///   - audioFile: The file, read from its current position.
    ///   - finishAfterFile: Whether to finalize and finish after the file.
    /// - Throws: An error if the session cannot accept input.
    public func start(inputAudioFile audioFile: AVAudioFile, finishAfterFile: Bool = false)
        async throws
    {
        try startInput(.file(FileSource(file: audioFile)), finishAfter: finishAfterFile)
    }

    private func startInput(_ source: InputSource, finishAfter: Bool) throws {
        try checkUsable()
        try reserveInput()
        let task = Task<CMTime?, any Error>(priority: options?.priority) {
            try await self.consume(source)
        }
        input.withLock { $0 = task }
        Task { await self.watchAutonomousInput(task, finishAfter: finishAfter) }
    }

    /// Reports autonomous input failures and optionally finishes after the input ends.
    private func watchAutonomousInput(_ task: Task<CMTime?, any Error>, finishAfter: Bool) async {
        switch await task.result {
        case .success:
            if finishAfter {
                do { try await finalizeAndFinishThroughEndOfInput() } catch {
                    _ = await fail(error)
                }
            }
        case .failure(let error):
            if !(error is CancellationError) { _ = await fail(error) }
        }
    }

    /// Claims the single input slot.
    private func reserveInput() throws {
        guard !inputActive, !finishing else {
            throw SpeechRecognitionError.invalidSessionState(
                "The analyzer is already reading input or finishing.")
        }
        inputActive = true
    }

    private func runInput(_ source: InputSource) async throws -> CMTime? {
        try checkUsable()
        try reserveInput()
        let task = Task<CMTime?, any Error>(priority: options?.priority) {
            try await self.consume(source)
        }
        input.withLock { $0 = task }
        do {
            return try await withTaskCancellationHandler {
                try await task.value
            } onCancel: {
                task.cancel()
                Task { await self.cancelAndFinishNow() }
            }
        } catch {
            if let failure { throw failure }
            if state == .cancelled || Task.isCancelled { throw CancellationError() }
            throw await fail(error)
        }
    }

    /// Reads one source, sending each buffer to every module branch.
    private func consume(_ source: InputSource) async throws -> CMTime? {
        defer { inputActive = false }
        try checkUsable()
        var last: CMTime?
        var iterator = source.makeIterator(continuingAt: timeline.lastEnd)
        while let (buffer, start) = try await iterator.next() {
            try Task.checkCancellation()
            let (timed, inputs) = try await admit(buffer, startTime: start)
            for input in inputs { try await input.send(timed) }
            last = timed.endTime
        }
        // Each input operation delivers all of its audio, then ends its native segment.
        // Input is consumed when every module has pulled it.
        if let last {
            for slot in slots {
                try await slot.input.flush(sourceEnd: last)
                slot.input.branch.sendBoundary()
            }
            for slot in slots {
                // Converted buffers can end a fraction of a frame before the source.
                let converted = last - CMTime(value: 1, timescale: 1000)
                try await slot.engine.waitUntilConsumed(
                    through: converted, branch: slot.input.branch)
            }
        }
        try checkUsable()
        if state == .analyzing { state = .draining }
        guard let last else { return nil }
        let rate = CMTimeScale(lastInputFormat?.sampleRate ?? 16_000)
        return last - CMTime(value: 1, timescale: rate)
    }

    /// Stamps a buffer, prepares modules on first input, and returns the branches to feed.
    private func admit(_ buffer: AVAudioPCMBuffer, startTime: CMTime?) async throws -> (
        TimedAudioBuffer, [BranchInput]
    ) {
        try checkUsable()
        var stamped = timeline
        let timed = try stamped.stamp(buffer, startTime: startTime)
        if !prepared { try await ensurePrepared(natural: buffer.format) }
        try checkUsable()
        timeline = stamped
        lastInputFormat = buffer.format
        state = .analyzing
        return (timed, slots.map(\.input))
    }

    // MARK: Finalization and finishing

    /// Finalizes results through a time, waiting until input reaches it.
    /// - Parameter time: The inclusive source time, or `nil` for the last consumed input.
    /// - Throws: `CancellationError` if the session ends before input reaches the time.
    public func finalize(through time: CMTime?) async throws {
        try checkUsable()
        if !prepared {
            // Without input there is nothing consumed to finalize.
            guard time != nil else { return }
            try await waitForPreparation()
        }
        guard let target = time ?? timeline.lastEnd else { return }
        try await finalizeEngines(through: target)
    }

    /// Waits until the first input prepares the modules, or the session ends.
    private func waitForPreparation() async throws {
        if let preparation { return try await settle(preparation) }
        try await withCheckedThrowingContinuation { preparationWaiters.append($0) }
        try checkUsable()
    }

    private func resumePreparationWaiters(throwing error: (any Error)?) {
        let waiters = preparationWaiters
        preparationWaiters.removeAll()
        for waiter in waiters {
            if let error { waiter.resume(throwing: error) } else { waiter.resume() }
        }
    }

    private func finalizeEngines(through time: CMTime) async throws {
        let slots = slots
        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                for slot in slots {
                    group.addTask {
                        try await slot.engine.finalize(through: time, branch: slot.input.branch)
                    }
                }
                try await group.waitForAll()
            }
        } catch {
            if let failure { throw failure }
            throw error
        }
        try checkUsable()
    }

    /// Finishes after the current input ends, finalizing all results.
    /// - Throws: An invalid-state or backend error.
    public func finalizeAndFinishThroughEndOfInput() async throws {
        try await finish(finalizing: true, after: nil)
    }

    /// Finalizes through a time and then finishes after it.
    /// - Parameter time: The inclusive source time.
    /// - Throws: An invalid-state or backend error, or `CancellationError`.
    public func finalizeAndFinish(through time: CMTime) async throws {
        try await finalize(through: time)
        try await finish(after: time)
    }

    /// Finishes after a time without finalizing remaining results.
    /// - Parameter time: The source time after which no input is analyzed.
    /// - Throws: `CancellationError` if the session ends before input reaches the time.
    public func finish(after time: CMTime) async throws {
        try await finish(finalizing: false, after: time)
    }

    private func finish(finalizing: Bool, after time: CMTime?) async throws {
        if state == .finished { return }
        try checkUsable()
        guard !finishing else {
            throw SpeechRecognitionError.invalidSessionState("The analyzer is already finishing.")
        }
        finishing = true
        defer { finishing = false }
        if finalizing, let active = input.withLock({ $0 }) {
            _ = await active.result
        }
        try checkUsable()
        if !prepared {
            if let preparation { try await settle(preparation) }
        }
        if !prepared {
            for binding in bindings { binding.finish(nil, false) }
            state = .finished
            return
        }
        do {
            if finalizing { try await flushConverters() }
            let end = timeline.lastEnd
            let slots = slots
            try await withThrowingTaskGroup(of: Void.self) { group in
                for slot in slots {
                    group.addTask {
                        if let time {
                            try await slot.engine.finish(
                                after: time, branch: slot.input.branch, run: slot.run)
                        } else {
                            try await slot.engine.finishThroughEndOfInput(
                                end: end, branch: slot.input.branch, run: slot.run)
                        }
                    }
                }
                try await group.waitForAll()
            }
            try checkUsable()
            input.withLock { $0?.cancel() }
            state = .finished
        } catch {
            if let failure { throw failure }
            if state == .cancelled { throw CancellationError() }
            throw await fail(error)
        }
    }

    /// Emits frames held by sample-rate converters before final analysis.
    private func flushConverters() async throws {
        guard let end = timeline.lastEnd else { return }
        for slot in slots { try await slot.input.flush(sourceEnd: end) }
    }

    /// Requests that modules skip analysis before a time. This request is best effort.
    /// - Parameter time: The source time before which results are no longer wanted.
    public func cancelAnalysis(before time: CMTime) {
        let slots = slots
        let previous = controls
        controls = Task {
            await previous?.value
            for slot in slots { await slot.engine.cancelAnalysis(before: time) }
        }
    }

    /// Cancels input and analysis, closes results, and waits for module cleanup.
    /// Buffered results are discarded. Repeated calls are safe.
    public func cancelAndFinishNow() async {
        if state == .finished { return }
        if state != .failed { state = .cancelled }
        await startCleanup(error: nil).value
    }

    @discardableResult
    private func startCleanup(error: (any Error)?) -> Task<Void, Never> {
        if let cleanup { return cleanup }
        resumePreparationWaiters(throwing: error ?? CancellationError())
        let active = input.withLock { $0 }
        active?.cancel()
        preparation?.cancel()
        let slots = slots
        let bindings = bindings
        let preparing = preparingEngines
        let task = Task<Void, Never> {
            for slot in slots { slot.input.branch.finish(throwing: error ?? CancellationError()) }
            for binding in bindings { binding.finish(error, error == nil) }
            for engine in preparing { await engine.cancel(error: error) }
            for slot in slots { await slot.engine.cancel(error: error) }
            for slot in slots { await slot.run.value }
            _ = await active?.result
        }
        cleanup = task
        return task
    }

    /// Records a session failure, closes every module's results, and starts cleanup.
    /// - Returns: The recorded failure.
    @discardableResult
    private func fail(_ error: any Error) async -> any Error {
        if let failure { return failure }
        if state == .cancelled || state == .finished { return CancellationError() }
        let error = recognitionError(error)
        failure = error
        state = error is CancellationError ? .cancelled : .failed
        startCleanup(error: error)
        return error
    }

    private func checkUsable() throws {
        if let failure { throw failure }
        switch state {
        case .cancelled: throw CancellationError()
        case .finished:
            throw SpeechRecognitionError.invalidSessionState("The analyzer has finished.")
        default: break
        }
        if let configurationError { throw configurationError }
    }

    // MARK: Modules and context

    /// Replaces the analyzer's modules.
    ///
    /// Added modules receive input after the current buffer.
    /// Removed modules finalize their consumed input and close their results.
    /// - Parameter newModules: Package modules.
    /// - Throws: A configuration error, or an error from preparing added modules.
    public func setModules(_ newModules: [any SpeechModule]) async throws {
        try checkUsable()
        let newIDs = Set(newModules.map { ObjectIdentifier($0) })
        let kept = bindings.filter { newIDs.contains($0.id) }
        let removed = bindings.filter { !newIDs.contains($0.id) }
        let (added, error) = Self.bind(
            newModules.filter { module in !kept.contains { $0.id == ObjectIdentifier(module) } })
        if let error { throw error }
        bindings = kept + added
        modules = newModules
        guard prepared else {
            for binding in removed { binding.finish(nil, false) }
            return
        }
        do {
            try await removeEngines(for: removed)
            if !added.isEmpty { try await addEngines(for: added, natural: lastInputFormat) }
        } catch {
            throw await fail(error)
        }
    }

    private func removeEngines(for removed: [ModuleBinding]) async throws {
        let removedIDs = Set(removed.map(\.id))
        if let native, let index = slots.firstIndex(where: { $0.engine.id == native.id }) {
            let remaining = slots[index].moduleIDs.subtracting(removedIDs)
            if remaining.count != slots[index].moduleIDs.count {
                try await native.setMembers(
                    bindings.filter { remaining.contains($0.id) }.compactMap { binding in
                        if case .native(let member) = binding.backend { return (binding, member) }
                        return nil
                    })
                slots[index].moduleIDs = remaining
            }
        }
        let end = timeline.lastEnd
        for slot in slots
        where slot.engine.id != native?.id && !slot.moduleIDs.isDisjoint(with: removedIDs) {
            try await slot.engine.finishThroughEndOfInput(
                end: end, branch: slot.input.branch, run: slot.run)
        }
        slots.removeAll { $0.engine.id != native?.id && !$0.moduleIDs.isDisjoint(with: removedIDs) }
    }

    /// The analysis context for the modules.
    public var context: AnalysisContext { contextValue }

    /// Replaces the analysis context. Modules apply it as soon as possible.
    /// - Parameter newContext: The replacement context.
    /// - Throws: An error if a module cannot apply the context.
    public func setContext(_ newContext: AnalysisContext) async throws {
        try checkUsable()
        contextValue = newContext
        for slot in slots { try await slot.engine.setContext(newContext) }
    }

    /// Sets or clears the volatile range handler.
    /// - Parameter handler: Called with the new range and whether its start and end changed.
    public func setVolatileRangeChangedHandler(
        _ handler:
            sending ((_ range: CMTimeRange, _ changedStart: Bool, _ changedEnd: Bool) -> Void)?
    ) {
        volatileHandler = handler
    }

    private func updateVolatileRange(_ id: ObjectIdentifier, _ range: CMTimeRange?) {
        engineRanges[id] = range
        let ranges = engineRanges.values
        let union = ranges.reduce(nil as CMTimeRange?) { union, range in
            guard let union else { return range }
            return CMTimeRange(start: min(union.start, range.start), end: max(union.end, range.end))
        }
        guard union != volatileRange else { return }
        let previous = volatileRange
        volatileRange = union
        let reported = union ?? CMTimeRange(start: previous?.end ?? .zero, duration: .zero)
        volatileHandler?(
            reported, reported.start != previous?.start, reported.end != previous?.end)
    }

    // MARK: Formats

    /// Returns the best format for the modules from installed assets.
    /// - Parameter modules: Package modules.
    /// - Returns: A format, or `nil` if no format is available.
    public static func bestAvailableAudioFormat(compatibleWith modules: [any SpeechModule]) async
        -> AVAudioFormat?
    {
        await bestAvailableAudioFormat(compatibleWith: modules, considering: nil)
    }

    /// Returns the best format for the modules, preferring the natural format of the input.
    ///
    /// The analyzer converts input for each module, so any input format works.
    /// This method returns the natural format when every module accepts it.
    /// - Parameters:
    ///   - modules: Package modules.
    ///   - naturalFormat: The format of the input, if known.
    /// - Returns: A format, or `nil` if no format is available.
    public static func bestAvailableAudioFormat(
        compatibleWith modules: [any SpeechModule], considering naturalFormat: AVAudioFormat?
    ) async -> AVAudioFormat? {
        guard let groups = try? ModuleGroups(modules) else { return nil }
        if groups.models.isEmpty {
            return await Speech.SpeechAnalyzer.bestAvailableAudioFormat(
                compatibleWith: groups.native, considering: naturalFormat)
        }
        var lists: [[AVAudioFormat]] = []
        for module in modules { lists.append(await module.availableCompatibleAudioFormats) }
        if let naturalFormat,
            lists.allSatisfy({ $0.contains { $0.isEqual(naturalFormat) } })
        {
            return naturalFormat
        }
        return lists.first?.first
    }
}

/// One module branch and the converter that feeds it.
///
/// Only one input operation runs at a time, and finishing runs after input stops,
/// so the converter is never used concurrently.
final class BranchInput: @unchecked Sendable {
    let branch: AudioBranch
    private let converter: StreamConverter

    init(format: AVAudioFormat) {
        branch = AudioBranch(format: format)
        converter = StreamConverter(output: format)
    }

    func send(_ timed: TimedAudioBuffer) async throws {
        try await send(converter.convert(timed), sourceEnd: timed.endTime)
    }

    func flush(sourceEnd: CMTime) async throws {
        try await send(converter.flush(), sourceEnd: sourceEnd)
    }

    private func send(_ converted: [TimedAudioBuffer], sourceEnd: CMTime) async throws {
        for (index, audio) in converted.enumerated() {
            let end = index == converted.count - 1 ? max(sourceEnd, audio.endTime) : audio.endTime
            try await branch.send(.init(audio: audio, sourceEnd: end))
        }
        if converted.isEmpty { branch.advance(sourceEnd: sourceEnd) }
    }
}

/// A caller input sequence or file, read in bounded blocks.
enum InputSource: Sendable {
    case sequence(AnyInputSequence)
    case file(FileSource)

    func makeIterator(continuingAt lastEnd: CMTime?) -> InputIterator {
        switch self {
        case .sequence(let sequence): .sequence(sequence.makeIterator())
        case .file(let file): .file(file, base: lastEnd, start: file.file.framePosition)
        }
    }
}

enum InputIterator {
    case sequence(AnyInputSequence.Iterator)
    case file(FileSource, base: CMTime?, start: AVAudioFramePosition)

    /// Returns the next buffer and its explicit start time, if any.
    mutating func next() async throws -> (AVAudioPCMBuffer, CMTime?)? {
        switch self {
        case .sequence(let iterator):
            let input = try await iterator.next()
            guard let input else { return nil }
            return (input.buffer, input.bufferStartTime)
        case .file(let source, let base, let start):
            return try source.read(base: base, start: start)
        }
    }
}

/// A type-erased native input sequence.
struct AnyInputSequence: Sendable {
    private let make: @Sendable () -> Iterator

    init<S: AsyncSequence & Sendable>(_ sequence: S) where S.Element == AnalyzerInput {
        make = {
            var iterator = sequence.makeAsyncIterator()
            return Iterator { try await iterator.next() }
        }
    }

    func makeIterator() -> Iterator { make() }

    struct Iterator {
        let next: () async throws -> AnalyzerInput?
    }
}

/// A file read from its current position in bounded blocks.
///
/// The analyzer owns the file for the duration of one input operation.
struct FileSource: @unchecked Sendable {
    let file: AVAudioFile
    static let blockFrames: AVAudioFrameCount = 4096

    /// Reads the next block and computes its source start time.
    func read(base: CMTime?, start: AVAudioFramePosition) throws -> (AVAudioPCMBuffer, CMTime?)? {
        guard file.framePosition < file.length else { return nil }
        let format = file.processingFormat
        let position = file.framePosition
        let count = AVAudioFrameCount(min(Int64(Self.blockFrames), file.length - position))
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count) else {
            throw SpeechRecognitionError.invalidAudio("Cannot allocate a file input block.")
        }
        try file.read(into: buffer, frameCount: count)
        guard buffer.frameLength > 0 else { return nil }
        let rate = CMTimeScale(format.sampleRate)
        let origin = base ?? CMTime(value: start, timescale: rate)
        return (buffer, origin + CMTime(value: position - start, timescale: rate))
    }
}
