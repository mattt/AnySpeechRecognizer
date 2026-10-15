import AVFoundation
import CoreMedia
import Foundation
import Speech
import Testing

@testable import AnySpeechRecognizer

/// Opt-in tests of the Parakeet backend through the public analyzer.
/// Set `ANYSPEECH_PARAKEET_MODELS` to a staged model directory and
/// `ANYSPEECH_PARAKEET_LONG_AUDIO` to a speech recording longer than one window.
@Suite(
    .enabled(
        if: ParakeetFixtures.modelDirectory != nil && ParakeetFixtures.longAudio != nil))
struct ParakeetAnalyzer {
    let audio = ParakeetFixtures.longAudio!
    var model: ParakeetSpeechModel {
        ParakeetSpeechModel(assets: .directory(ParakeetFixtures.modelDirectory!))
    }

    /// Transcribes directly with the recognizer, without the analyzer.
    func directText() throws -> String {
        let directory = ParakeetFixtures.modelDirectory!
        let engine = try CoreMLParakeetEngine(directory: directory)
        let vocabulary = try ParakeetVocabulary(
            contentsOf: directory.appending(path: "parakeet_vocab.json"))
        let recognizer = try ParakeetRecognizer(
            engine: engine, vocabulary: vocabulary, configuration: .file)
        recognizer.append(try ParakeetFixtures.samples(audio))
        var tokens = try recognizer.process().committed
        tokens += try recognizer.process(endOfInput: true).committed
        return vocabulary.text(for: tokens.map(\.id))
    }

    @Test func fileTextMatchesTheRecognizer() async throws {
        #expect(await model.availability() == .ready)
        let transcriber = SpeechTranscriber(
            model: model, attributeOptions: [.audioTimeRange, .transcriptionConfidence])
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        async let results = collect(transcriber)
        let time = try #require(
            try await analyzer.analyzeSequence(from: AVAudioFile(forReading: audio)))
        try await analyzer.finalizeAndFinish(through: time)
        let values = try await results
        #expect(values.count > 1)
        #expect(values.allSatisfy { $0.isFinal })
        #expect(zip(values, values.dropFirst()).allSatisfy { $0.range.end <= $1.range.start })
        for result in values {
            try validateTiming(in: result.text, range: result.range)
            for run in result.text.runs
            where !String(result.text[run.range].characters).allSatisfy(\.isWhitespace) {
                let confidence = run[AttributeScopes.SpeechAttributes.ConfidenceAttribute.self]
                #expect(confidence.map { (0...1).contains($0) } == true)
            }
        }
        #expect(texts(values).joined() == (try directText()))
    }

    @Test func liveInputPublishesVolatileThenFinalText() async throws {
        let transcriber = SpeechTranscriber(
            model: model, reportingOptions: [.volatileResults], attributeOptions: [.audioTimeRange])
        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        let analyzer = SpeechAnalyzer(inputSequence: stream, modules: [transcriber])
        async let results = collect(transcriber)
        for input in try liveBlocks(audio, seconds: 0.1) { continuation.yield(input) }
        continuation.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        let values = try await results
        let finals = values.filter { $0.isFinal }
        #expect(values.contains { !$0.isFinal })
        #expect(zip(finals, finals.dropFirst()).allSatisfy { $0.range.end <= $1.range.start })
        for result in values { try validateTiming(in: result.text, range: result.range) }
        let edits = ParakeetFixtures.wordDifferences(try directText(), texts(finals).joined())
        // Live windows differ from file windows; differences are mostly punctuation and case.
        #expect(edits.insertions + edits.deletions <= 2, "\(edits)")
        #expect(edits.substitutions * 10 <= edits.words, "\(edits)")
    }

    @Test func midStreamFinalizationCommitsThroughTheTime() async throws {
        let transcriber = SpeechTranscriber(
            model: model, reportingOptions: [.volatileResults], attributeOptions: [.audioTimeRange])
        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        let analyzer = SpeechAnalyzer(inputSequence: stream, modules: [transcriber])
        let collected = Task { try await collect(transcriber) }
        let blocks = try liveBlocks(audio, seconds: 0.1)
        for input in blocks.prefix(100) { continuation.yield(input) }
        let target = CMTime(value: 9, timescale: 1)
        try await analyzer.finalize(through: target)
        #expect(await analyzer.volatileRange.map { $0.start >= target } ?? true)
        for input in blocks.dropFirst(100) { continuation.yield(input) }
        continuation.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        let finals = try await collected.value.filter { $0.isFinal }
        #expect(finals.contains { $0.resultsFinalizationTime >= target })
    }

    @Test func offsetAndResampledInputStayOnTheSourceTimeline() async throws {
        let source = try AVAudioFile(forReading: audio)
        let url = FileManager.default.temporaryDirectory.appending(path: "parakeet-\(UUID()).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try resample(source, to: url, sampleRate: 44_100, channels: 2)
        let file = try AVAudioFile(forReading: url)
        file.framePosition = 44_100 * 5
        let transcriber = SpeechTranscriber(model: model, attributeOptions: [.audioTimeRange])
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        async let results = collect(transcriber)
        let time = try #require(try await analyzer.analyzeSequence(from: file))
        try await analyzer.finalizeAndFinish(through: time)
        let values = try await results
        #expect(!values.isEmpty)
        #expect(values.allSatisfy { $0.range.start >= CMTime(value: 5, timescale: 1) })
        #expect(values.last!.range.end <= time + CMTime(value: 1, timescale: 1000))
    }

    private func resample(
        _ source: AVAudioFile, to url: URL, sampleRate: Double, channels: AVAudioChannelCount
    ) throws {
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: channels,
            interleaved: false)!
        var settings = format.settings
        settings[AVLinearPCMIsNonInterleaved] = false
        let output = try AVAudioFile(forWriting: url, settings: settings)
        let converter = StreamConverter(output: format)
        var timeline = InputTimeline(lastEnd: nil)
        while source.framePosition < source.length {
            let buffer = AVAudioPCMBuffer(pcmFormat: source.processingFormat, frameCapacity: 4096)!
            try source.read(into: buffer, frameCount: 4096)
            for converted in try converter.convert(timeline.stamp(buffer, startTime: nil)) {
                try output.write(from: converted.buffer)
            }
        }
        for converted in try converter.flush() { try output.write(from: converted.buffer) }
    }
}
