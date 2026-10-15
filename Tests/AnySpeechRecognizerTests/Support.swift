import AVFoundation
import CoreMedia
import Foundation
import Testing

@testable import AnySpeechRecognizer

typealias SpeechTranscriber = AnySpeechRecognizer.SpeechTranscriber
typealias SpeechAnalyzer = AnySpeechRecognizer.SpeechAnalyzer
typealias AnalyzerInput = AnySpeechRecognizer.AnalyzerInput
typealias DictationTranscriber = AnySpeechRecognizer.DictationTranscriber
typealias SpeechDetector = AnySpeechRecognizer.SpeechDetector
typealias AssetInventory = AnySpeechRecognizer.AssetInventory

enum TestFailure: Error, Equatable { case backend }

actor Gate {
    private var opened = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    var isOpen: Bool { opened }
    func wait() async {
        if opened { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func open() {
        opened = true
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }
}

struct FakeModel: SpeechRecognitionModel {
    let modelID = "test/fake"
    var capabilities = SpeechRecognitionCapabilities(
        transcription: .init(reportingOptions: [.volatileResults]), liveInput: true)
    let executor: FakeExecutor
    func makeExecutor() -> any SpeechRecognitionExecutor { executor }
}

/// Emits scripted results once it has pulled audio through each result's end.
actor FakeExecutor: SpeechRecognitionExecutor {
    let prepareEntered = Gate()
    let analyzeEntered = Gate()
    let prepareGate: Gate?
    let analyzeGate: Gate?
    let backendError: TestFailure?
    private var pending: [SpeechRecognitionResult]
    private var deferred: [SpeechRecognitionResult]
    private(set) var prepareCount = 0
    private(set) var analyzeCount = 0
    private(set) var cancelCount = 0
    private(set) var finalizeTimes: [CMTime] = []
    private(set) var cancelledBefore: [CMTime] = []
    private(set) var pulled: [TimedAudioBuffer] = []
    private(set) var format: AVAudioFormat?
    private(set) var contexts: [AnalysisContext] = []
    private var sink: SpeechRecognitionResultSink?

    init(
        snapshots: [SpeechRecognitionResult] = [], deferred: [SpeechRecognitionResult] = [],
        prepareGate: Gate? = nil, analyzeGate: Gate? = nil, backendError: TestFailure? = nil
    ) {
        self.pending = snapshots
        self.deferred = deferred
        self.prepareGate = prepareGate
        self.analyzeGate = analyzeGate
        self.backendError = backendError
    }

    func prepare(
        request: SpeechRecognitionRequest, format: AVAudioFormat, context: AnalysisContext
    ) async throws {
        prepareCount += 1
        self.format = format
        await prepareEntered.open()
        await prepareGate?.wait()
        try Task.checkCancellation()
    }

    func analyze(_ input: SpeechAnalysisInput, results: SpeechRecognitionResultSink) async throws {
        analyzeCount += 1
        sink = results
        await analyzeEntered.open()
        await analyzeGate?.wait()
        try Task.checkCancellation()
        for try await buffer in input {
            pulled.append(buffer)
            if let backendError { throw backendError }
            while let next = pending.first, next.range.end <= buffer.endTime {
                pending.removeFirst()
                try await results.emit(next)
            }
        }
    }

    func finalize(through time: CMTime) async throws {
        finalizeTimes.append(time)
        let ready = deferred.filter { $0.range.start <= time }
        deferred.removeAll { $0.range.start <= time }
        for result in ready { try await sink?.emit(result) }
    }

    func cancelAnalysis(before time: CMTime) async { cancelledBefore.append(time) }

    func setContext(_ context: AnalysisContext) async throws { contexts.append(context) }

    func publishLater(_ result: SpeechRecognitionResult) async throws {
        try await sink?.emit(result)
    }

    func reportVolatile(_ range: CMTimeRange?) { sink?.setVolatileRange(range) }

    func cancel() async {
        cancelCount += 1
        await prepareGate?.open()
        await analyzeGate?.open()
        sink = nil
    }

    var pulledFrames: Int { pulled.reduce(0) { $0 + Int($1.buffer.frameLength) } }
}

func snapshot(
    _ text: String = "Hello.", start: Int64 = 0, end: Int64 = 16_000, final: Bool = true
) -> SpeechRecognitionResult {
    let range = CMTimeRange(
        start: CMTime(value: start, timescale: 16_000), end: CMTime(value: end, timescale: 16_000))
    return .init(
        text: AttributedString(text), range: range,
        resultsFinalizationTime: final ? range.end : range.start)
}

func audioFile(
    sampleRate: Double = 16_000, channels: AVAudioChannelCount = 1, frames: Int = 16_000,
    samples: [Float] = [0.25]
) throws -> (URL, AVAudioFile) {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "anyspeech-test-\(UUID()).wav")
    let format = try #require(
        AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: channels,
            interleaved: false
        ))
    do {
        var settings = format.settings
        settings[AVLinearPCMIsNonInterleaved] = false
        let output = try AVAudioFile(forWriting: url, settings: settings)
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 2048))
        let data = try #require(buffer.floatChannelData)
        var remaining = frames
        while remaining > 0 {
            buffer.frameLength = AVAudioFrameCount(min(remaining, 2048))
            for channel in 0..<Int(channels) {
                for frame in 0..<Int(buffer.frameLength) {
                    data[channel][frame] = samples[channel % samples.count]
                }
            }
            try output.write(from: buffer)
            remaining -= Int(buffer.frameLength)
        }
    }
    return (url, try AVAudioFile(forReading: url))
}

/// A mono Float32 buffer with a constant value.
func pcmBuffer(
    frames: Int, sampleRate: Double = 16_000, channels: AVAudioChannelCount = 1,
    value: Float = 0.25
) -> AVAudioPCMBuffer {
    let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: channels,
        interleaved: false)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))!
    buffer.frameLength = AVAudioFrameCount(frames)
    for channel in 0..<Int(channels) {
        for frame in 0..<frames { buffer.floatChannelData![channel][frame] = value }
    }
    return buffer
}

/// A 16 kHz mono Int16 input. Apple's `AnalyzerInput` initializer traps on Float32
/// buffers on macOS 27.0.1, and Apple's native best format is Int16.
func input(frames: Int, start: CMTime? = nil) -> AnalyzerInput {
    let format = AVAudioFormat(
        commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))!
    buffer.frameLength = AVAudioFrameCount(frames)
    for frame in 0..<frames { buffer.int16ChannelData![0][frame] = 8_192 }
    return AnalyzerInput(buffer: buffer, bufferStartTime: start)
}

/// A live input stream that the test feeds explicitly.
func liveInput() -> (AsyncStream<AnalyzerInput>, AsyncStream<AnalyzerInput>.Continuation) {
    AsyncStream<AnalyzerInput>.makeStream()
}

func collect<Module: SpeechModule>(_ module: Module) async throws -> [Module.Result] {
    var results: [Module.Result] = []
    for try await result in module.results { results.append(result) }
    return results
}

func texts(_ results: [SpeechTranscriber.Result]) -> [String] {
    results.map { String($0.text.characters) }
}

/// Polls an actor-isolated condition until it holds.
func eventually(
    _ condition: @Sendable () async -> Bool, sourceLocation: SourceLocation = #_sourceLocation
) async {
    for _ in 0..<2000 {
        if await condition() { return }
        try? await Task.sleep(for: .milliseconds(1))
    }
    Issue.record("Condition was not met in time.", sourceLocation: sourceLocation)
}
