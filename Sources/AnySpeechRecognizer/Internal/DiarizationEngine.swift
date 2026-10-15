import AVFoundation
import CoreMedia
import Foundation

/// Drives one speaker diarizer through its public executor.
///
/// The engine follows ``ExecutorEngine``: it owns one input branch,
/// validates every result before publishing it, and serializes finalization.
actor DiarizationEngine: AnalysisEngine {
    nonisolated var id: ObjectIdentifier { binding.id }
    nonisolated let binding: ModuleBinding
    let member: DiarizationMember
    let executor: any SpeakerDiarizationExecutor
    let events: EngineEvents
    private var branch: AudioBranch?
    private var finalization: CMTime?
    private var closed = false
    private var queue = FinalizationQueue()

    init(binding: ModuleBinding, member: DiarizationMember, events: EngineEvents) {
        self.binding = binding
        self.member = member
        self.executor = member.model.makeExecutor()
        self.events = events
    }

    func inputFormat(considering natural: AVAudioFormat?) async -> AVAudioFormat {
        let formats = await member.model.compatibleAudioFormats(for: member.request)
        if let natural,
            formats.contains(where: {
                $0.sampleRate == natural.sampleRate && $0.channelCount == natural.channelCount
                    && $0.commonFormat == natural.commonFormat
                    && $0.isInterleaved == natural.isInterleaved
            })
        {
            return natural
        }
        return formats.first ?? AudioFormats.monoFloat16k
    }

    func prepare(format: AVAudioFormat, context: AnalysisContext) async throws {
        let model = member.model
        try member.request.validate(for: model.capabilities)
        switch await model.availability(for: member.request) {
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
        try await executor.prepare(request: member.request, format: format)
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
        let sink = SpeakerDiarizationResultSink(
            publish: { [weak self] result in
                guard let self else { throw CancellationError() }
                try await self.publish(result)
            },
            volatile: { continuation.yield($0) }
        )
        try await executor.analyze(SpeechAnalysisInput(branch: branch), results: sink)
    }

    private func publish(_ result: SpeakerDiarizer.Result) async throws {
        guard !closed else { throw CancellationError() }
        do {
            try validate(result)
            try member.deliver(result)
            finalization = max(
                finalization ?? result.resultsFinalizationTime, result.resultsFinalizationTime)
        } catch {
            let error = recognitionError(error)
            await events.fail(error)
            throw error
        }
    }

    /// Checks result timing, segment containment and order, and speaker identifiers.
    private func validate(_ result: SpeakerDiarizer.Result) throws {
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
                "A diarization result has invalid, overlapping, or unfinalized source timing.")
        }
        let maximum = member.model.capabilities.maximumSpeakerCount
        var previous: CMTime?
        for segment in result.segments {
            let segmentRange = segment.range
            guard segmentRange.isValid, segmentRange.start.isNumeric,
                segmentRange.duration.isNumeric, segmentRange.duration >= .zero,
                segmentRange.start >= range.start, segmentRange.end <= range.end,
                previous.map({ segmentRange.start >= $0 }) ?? true,
                (0..<maximum).contains(segment.speaker.id)
            else {
                throw SpeechRecognitionError.invalidResult(
                    "A diarization segment is outside its result, out of order, "
                        + "or has an unknown speaker.")
            }
            previous = segmentRange.start
        }
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

    /// Diarization ignores contextual strings.
    func setContext(_ context: AnalysisContext) async throws {}

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
