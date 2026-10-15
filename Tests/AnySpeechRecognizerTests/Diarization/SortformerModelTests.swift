import AVFoundation
import CoreMedia
import Foundation
import Testing

@testable import AnySpeechRecognizer

/// Opt-in tests with a local Sortformer model.
///
/// Set `ANYSPEECH_SORTFORMER_MODEL` to a compiled model directory
/// (`SortformerLowLatency.mlmodelc`)
/// and `ANYSPEECH_DIARIZATION_AUDIO` to a 16 kHz mono recording with two or more speakers.
/// Set `ANYSPEECH_SORTFORMER_VERSION` to `v2.1` for a v2.1 model.
/// Set `ANYSPEECH_SORTFORMER_REFERENCE` to the output prefix of
/// `Tools/sortformer-reference.py` for that recording and version to compare with NeMo.
enum SortformerFixtures {
    static var environment: [String: String] { ProcessInfo.processInfo.environment }
    static var model: URL? {
        environment["ANYSPEECH_SORTFORMER_MODEL"].map { URL(fileURLWithPath: $0) }
    }
    static var audio: URL? {
        environment["ANYSPEECH_DIARIZATION_AUDIO"].map { URL(fileURLWithPath: $0) }
    }
    static var reference: String? { environment["ANYSPEECH_SORTFORMER_REFERENCE"] }
    static var version: SortformerDiarizationModel.Version {
        environment["ANYSPEECH_SORTFORMER_VERSION"].flatMap { .init(rawValue: $0) } ?? .v2
    }
    static var enabled: Bool { model != nil && audio != nil }

    static func samples(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(
            forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        guard file.processingFormat.sampleRate == 16_000, file.processingFormat.channelCount == 1
        else { throw SpeechRecognitionError.invalidAudio("Use 16 kHz mono audio.") }
        let buffer = try #require(
            AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        return Array(
            UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    }

    static func floats(_ path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func model(volatile: Bool = false) throws -> SortformerDiarizationModel {
        SortformerDiarizationModel(version: version, assets: .compiledModel(try #require(model)))
    }
}

@Suite(.enabled(if: SortformerFixtures.enabled), .serialized)
struct SortformerModelTests {
    @Test(.enabled(if: SortformerFixtures.reference != nil))
    func melFeaturesMatchNeMo() throws {
        let samples = try SortformerFixtures.samples(try #require(SortformerFixtures.audio))
        let reference = try SortformerFixtures.floats(
            try #require(SortformerFixtures.reference) + ".mel.f32")
        let frontend = SortformerMelFrontend()
        // Feed 100 ms blocks so the incremental path is exercised.
        samples.withUnsafeBufferPointer { samples in
            var start = 0
            while start < samples.count {
                let end = min(start + 1_600, samples.count)
                frontend.append(UnsafeBufferPointer(rebasing: samples[start..<end]))
                start = end
            }
        }
        frontend.finish()
        let bins = SortformerMelFrontend.melBinCount
        #expect(frontend.frameCount * bins == reference.count)
        var computed = [Float](repeating: 0, count: frontend.frameCount * bins)
        computed.withUnsafeMutableBufferPointer {
            frontend.copyFrames(0..<frontend.frameCount, into: $0.baseAddress!)
        }
        var maximum: Float = 0
        var total: Double = 0
        for (lhs, rhs) in zip(computed, reference) {
            maximum = max(maximum, abs(lhs - rhs))
            total += Double(abs(lhs - rhs))
        }
        let mean = total / Double(max(1, reference.count))
        print("Mel difference from NeMo: mean \(mean), maximum \(maximum)")
        #expect(mean < 1e-3)
    }

    @Test(.enabled(if: SortformerFixtures.reference != nil))
    func predictionsMatchNeMo() throws {
        let samples = try SortformerFixtures.samples(try #require(SortformerFixtures.audio))
        let reference = try SortformerFixtures.floats(
            try #require(SortformerFixtures.reference) + ".preds.f32")
        let model = try SortformerFixtures.model()
        let pipeline = SortformerPipeline(
            inference: try model.makeInference(model.modelURL, model.computeUnits.coreML),
            parameters: model.postProcessing.segmenterParameters, reportsVolatileResults: false)
        pipeline.recordedProbabilities = []
        try samples.withUnsafeBufferPointer { samples in
            var start = 0
            while start < samples.count {
                let end = min(start + 4_096, samples.count)
                _ = try pipeline.append(
                    UnsafeBufferPointer(rebasing: samples[start..<end]),
                    at: CMTime(value: CMTimeValue(start), timescale: 16_000))
                start = end
            }
        }
        _ = try pipeline.finalize(
            through: CMTime(value: CMTimeValue(samples.count - 1), timescale: 16_000))
        let computed = try #require(pipeline.recordedProbabilities)
        #expect(computed.count == reference.count)
        var maximum: Float = 0
        var total: Double = 0
        var agreements = 0
        for (lhs, rhs) in zip(computed, reference) {
            maximum = max(maximum, abs(lhs - rhs))
            total += Double(abs(lhs - rhs))
            if (lhs > 0.5) == (rhs > 0.5) { agreements += 1 }
        }
        let count = min(computed.count, reference.count)
        let mean = total / Double(max(1, count))
        let agreement = Double(agreements) / Double(max(1, count))
        print(
            "Prediction difference from NeMo: mean \(mean), maximum \(maximum), "
                + "agreement at 0.5 \(agreement)")
        #expect(mean < 0.02)
        #expect(agreement > 0.98)
    }

    @Test func fileAndLiveInputFindTheSameSpeakers() async throws {
        let url = try #require(SortformerFixtures.audio)
        let file = try await diarize(url, live: false)
        let live = try await diarize(url, live: true)
        #expect(file.speakers.count >= 2)
        #expect(file.speakers == live.speakers)
        #expect(file.segments == live.segments)
        print("Speakers: \(file.speakers), segments: \(file.segments.count)")
    }

    @Test func finalizationAndGapsKeepSpeakerIdentities() async throws {
        let url = try #require(SortformerFixtures.audio)
        let samples = try SortformerFixtures.samples(url)
        let reference = try await diarize(url, live: false)
        // Split the recording at its midpoint with a finalization, then resume after a gap.
        let diarizer = SpeakerDiarizer(model: try SortformerFixtures.model())
        let analyzer = SpeechAnalyzer(modules: [diarizer])
        async let collected = collect(diarizer)
        let half = samples.count / 2
        let first = AsyncStream<AnalyzerInput> { continuation in
            continuation.yield(int16Input(samples[0..<half], start: .zero))
            continuation.finish()
        }
        _ = try await analyzer.analyzeSequence(first)
        try await analyzer.finalize(
            through: CMTime(value: CMTimeValue(half - 1), timescale: 16_000))
        let gap = CMTime(value: CMTimeValue(half + 16_000), timescale: 16_000)
        let second = AsyncStream<AnalyzerInput> { continuation in
            continuation.yield(int16Input(samples[half...], start: gap))
            continuation.finish()
        }
        _ = try await analyzer.analyzeSequence(second)
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        var timeline = SpeakerTimeline()
        for result in try await collected { timeline.apply(result) }
        #expect(timeline.speakers == reference.speakers)
    }

    private func diarize(_ url: URL, live: Bool) async throws -> SpeakerTimeline {
        let diarizer = SpeakerDiarizer(
            model: try SortformerFixtures.model(), reportingOptions: live ? [.volatileResults] : [])
        let analyzer = SpeechAnalyzer(modules: [diarizer])
        async let collected = collect(diarizer)
        if live {
            let samples = try SortformerFixtures.samples(url)
            let stream = AsyncStream<AnalyzerInput> { continuation in
                var start = 0
                while start < samples.count {
                    let end = min(start + 1_600, samples.count)
                    continuation.yield(int16Input(samples[start..<end], start: nil))
                    start = end
                }
                continuation.finish()
            }
            _ = try await analyzer.analyzeSequence(stream)
            try await analyzer.finalizeAndFinishThroughEndOfInput()
        } else {
            let file = try AVAudioFile(forReading: url)
            let end = try #require(try await analyzer.analyzeSequence(from: file))
            try await analyzer.finalizeAndFinish(through: end)
        }
        var timeline = SpeakerTimeline()
        for result in try await collected { timeline.apply(result) }
        return timeline
    }
}

/// A 16 kHz mono Int16 input from Float32 samples.
func int16Input(_ samples: ArraySlice<Float>, start: CMTime?) -> AnalyzerInput {
    let format = AVAudioFormat(
        commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true)!
    let buffer = AVAudioPCMBuffer(
        pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
    buffer.frameLength = AVAudioFrameCount(samples.count)
    for (index, sample) in samples.enumerated() {
        buffer.int16ChannelData![0][index] = Int16(clamping: Int((sample * 32_768).rounded()))
    }
    return AnalyzerInput(buffer: buffer, bufferStartTime: start)
}
