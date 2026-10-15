import AVFoundation
import CoreMedia
import Foundation
import Speech
import Testing

@testable import AnySpeechRecognizer

/// Opt-in tests against installed Apple assets and a local speech recording.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["ANYSPEECH_INTEGRATION_AUDIO"] != nil))
struct AppleIntegration {
    let url = URL(
        fileURLWithPath: ProcessInfo.processInfo.environment["ANYSPEECH_INTEGRATION_AUDIO"] ?? "")
    let locale = Locale(identifier: "en-US")

    init() async throws {
        if ProcessInfo.processInfo.environment["ANYSPEECH_INSTALL_ASSETS"] == "1",
            let request = try await SystemSpeechModel(locale: locale).assetInstallationRequest()
        {
            try await request.downloadAndInstall()
        }
    }

    @Test func fileTextMatchesDirectNativeAnalysis() async throws {
        let native = Speech.SpeechTranscriber(
            locale: locale, transcriptionOptions: [], reportingOptions: [],
            attributeOptions: [.audioTimeRange])
        let nativeAnalyzer = Speech.SpeechAnalyzer(modules: [native])
        async let nativeText: String = {
            var text = ""
            for try await result in native.results { text += String(result.text.characters) }
            return text
        }()
        if let time = try await nativeAnalyzer.analyzeSequence(from: AVAudioFile(forReading: url)) {
            try await nativeAnalyzer.finalizeAndFinish(through: time)
        } else {
            await nativeAnalyzer.cancelAndFinishNow()
        }
        let direct = try await nativeText

        let transcriber = SpeechTranscriber(
            locale: locale, transcriptionOptions: [], reportingOptions: [],
            attributeOptions: [.audioTimeRange])
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        async let results = collect(transcriber)
        let time = try #require(
            try await analyzer.analyzeSequence(from: AVAudioFile(forReading: url)))
        try await analyzer.finalizeAndFinish(through: time)
        let values = try await results
        #expect(texts(values).joined() == direct)
        for result in values { try validateTiming(in: result.text, range: result.range) }
        #expect(await analyzer.state == .finished)
    }

    @Test func sourceOffsetStaysOnRecordingTimeline() async throws {
        let file = try AVAudioFile(forReading: url)
        file.framePosition = min(file.length / 3, Int64(file.processingFormat.sampleRate))
        let offset = try AudioTimeline.time(
            frame: file.framePosition, sampleRate: file.processingFormat.sampleRate)
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        async let results = collect(transcriber)
        let last = try #require(try await analyzer.analyzeSequence(from: file))
        try await analyzer.finalizeAndFinish(through: last)
        let values = try await results
        #expect(!values.isEmpty)
        #expect(values.allSatisfy { $0.range.start >= offset })
    }

    @Test func liveInputProducesVolatileThenFinalResults() async throws {
        let transcriber = SpeechTranscriber(
            locale: locale, preset: .timeIndexedProgressiveTranscription)
        let ranges = RangeLog()
        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        let analyzer = SpeechAnalyzer(
            inputSequence: stream, modules: [transcriber],
            volatileRangeChangedHandler: { range, _, _ in Task { await ranges.append(range) } })
        async let results = collect(transcriber)
        var end = CMTime.zero
        for input in try liveBlocks(url, seconds: 0.1) {
            continuation.yield(input)
            end =
                input.bufferStartTime!
                + CMTime(
                    value: CMTimeValue(input.buffer.frameLength), timescale: 16_000)
        }
        try await analyzer.finalize(through: end - CMTime(value: 1, timescale: 16_000))
        continuation.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        let values = try await results
        let finals = values.filter { $0.isFinal }
        #expect(values.contains { !$0.isFinal })
        #expect(!finals.isEmpty)
        #expect(zip(finals, finals.dropFirst()).allSatisfy { $0.range.end <= $1.range.start })
        #expect(await !ranges.values.isEmpty)
    }

    @Test func dictationAndDetectorShareOneNativeAnalyzer() async throws {
        let dictation = DictationTranscriber(locale: locale, preset: .longDictation)
        let detector = SpeechDetector(
            detectionOptions: .init(sensitivityLevel: .medium), reportResults: true)
        let analyzer = SpeechAnalyzer(modules: [dictation, detector])
        async let dictated = collect(dictation)
        async let detected = collect(detector)
        let time = try #require(
            try await analyzer.analyzeSequence(from: AVAudioFile(forReading: url)))
        try await analyzer.finalizeAndFinish(through: time)
        let text = try await dictated.map { String($0.text.characters) }.joined()
        #expect(!text.isEmpty)
        _ = try await detected
        #expect(await analyzer.state == .finished)
    }

    @Test func assetStatusCoversNativeModules() async throws {
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        let detector = SpeechDetector()
        #expect(await AssetInventory.status(forModules: [transcriber, detector]) == .installed)
        #expect(try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) == nil)
    }
}

private actor RangeLog {
    var values: [CMTimeRange] = []
    func append(_ range: CMTimeRange) { values.append(range) }
}

/// Reads a file as 16 kHz mono Int16 blocks with explicit start times, as live input.
func liveBlocks(_ url: URL, seconds: Double) throws -> [AnalyzerInput] {
    let file = try AVAudioFile(forReading: url)
    let target = AVAudioFormat(
        commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true)!
    let converter = StreamConverter(output: target)
    var timeline = InputTimeline(lastEnd: nil)
    var converted: [TimedAudioBuffer] = []
    let block = AVAudioFrameCount(file.processingFormat.sampleRate * seconds)
    while file.framePosition < file.length {
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: block)!
        try file.read(into: buffer, frameCount: block)
        converted += try converter.convert(timeline.stamp(buffer, startTime: nil))
    }
    converted += try converter.flush()
    return converted.map { AnalyzerInput(buffer: $0.buffer, bufferStartTime: $0.startTime) }
}
