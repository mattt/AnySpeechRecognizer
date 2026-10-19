import AVFoundation
import AnySpeechRecognizer
import CoreMedia
import Darwin
import Foundation
import Speech
import Synchronization

/// The transcript and speaker timeline from one local audio file.
struct Analysis {
    let transcript: AttributedString
    let speakers: SpeakerTimeline?
    let audio: AudioProperties
    /// When the analysis started, before the analyzer prepared its modules.
    let started: ContinuousClock.Instant

    /// Analyzes a local audio file.
    ///
    /// With `json`, this writes JSON lines with metadata, results, and speaker segments;
    /// otherwise it writes final text, unless a diarizer attributes it to speakers later.
    @concurrent static func run(
        _ path: String, model: ModelOptions, results: ResultOptions, diarizer: DiarizerOptions,
        json: Bool, cacheState: String
    ) async throws -> Analysis {
        let model = model.model()
        let diarizer = diarizer.diarizer(live: results.live)
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: URL(fileURLWithPath: path)) } catch {
            throw SpeechRecognitionError.invalidAudio(
                "Cannot open audio file: \(error.localizedDescription)")
        }
        let properties = AudioProperties(
            sampleRate: file.processingFormat.sampleRate,
            channels: Int(file.processingFormat.channelCount),
            frames: file.length,
            durationSeconds: Double(file.length) / file.processingFormat.sampleRate
        )
        let transcriber = SpeechTranscriber(
            model: model, reportingOptions: results.reportingOptions,
            attributeOptions: [.audioTimeRange])
        let modules: [any SpeechModule] = [transcriber] + (diarizer.map { [$0] } ?? [])
        let analyzer = SpeechAnalyzer(modules: modules)
        let started = ContinuousClock.now
        let preparationStart = ContinuousClock.now
        do {
            try await analyzer.prepareToAnalyze(in: file.processingFormat)
            let preparationSeconds = elapsed(since: preparationStart)
            if json {
                try writeJSON(
                    MetadataRecord(
                        type: "metadata", diagnostics: await model.diagnostics(), audio: properties,
                        hardware: hardware(), cacheState: cacheState,
                        preparationSeconds: preparationSeconds,
                        diarizer: await diarizer?.model.diagnostics()
                    ))
            }
            let clock = ResultClock(
                started: .now, live: results.live, sampleRate: properties.sampleRate)
            async let transcript = consume(
                transcriber, json: json, printsText: diarizer == nil, clock: clock)
            async let timeline = consume(diarizer, json: json)
            if results.live {
                _ = try await analyzer.analyzeSequence(
                    try liveBlocks(path, pacedFrom: results.realTime ? clock.started : nil))
                try await analyzer.finalizeAndFinishThroughEndOfInput()
            } else if let time = try await analyzer.analyzeSequence(from: file) {
                try await analyzer.finalizeAndFinish(through: time)
            } else {
                try await analyzer.finalizeAndFinishThroughEndOfInput()
            }
            return Analysis(
                transcript: try await transcript, speakers: try await timeline, audio: properties,
                started: started)
        } catch {
            await analyzer.cancelAndFinishNow()
            throw error
        }
    }

    @concurrent private static func consume(
        _ transcriber: SpeechTranscriber, json: Bool, printsText: Bool, clock: ResultClock
    ) async throws -> AttributedString {
        var text = AttributedString()
        for try await result in transcriber.results {
            try Task.checkCancellation()
            if json {
                try writeJSON(ResultRecord(result, clock: clock))
            } else if result.isFinal, printsText {
                FileHandle.standardOutput.write(Data(String(result.text.characters).utf8))
            }
            if result.isFinal { text += result.text }
        }
        return text
    }

    @concurrent private static func consume(_ diarizer: SpeakerDiarizer?, json: Bool)
        async throws -> SpeakerTimeline?
    {
        guard let diarizer else { return nil }
        var timeline = SpeakerTimeline()
        for try await result in diarizer.results {
            try Task.checkCancellation()
            if json {
                try writeJSON(DiarizationRecord(result))
            }
            timeline.apply(result)
        }
        return timeline
    }

    /// Reads a file as 100 ms Int16 blocks on demand, as a live source would deliver them.
    /// Apple's `AnalyzerInput` accepts Int16 buffers; it traps on Float32 on macOS 27.0.1.
    ///
    /// - Parameter pacedFrom: When set, each block waits until this instant plus the end time
    ///   of its audio, as a capture source would deliver it.
    private static func liveBlocks(_ path: String, pacedFrom: ContinuousClock.Instant?) throws
        -> LiveBlocks
    {
        LiveBlocks(
            file: try AVAudioFile(
                forReading: URL(fileURLWithPath: path), commonFormat: .pcmFormatInt16,
                interleaved: true),
            pacedFrom: pacedFrom)
    }

    private static func hardware() -> String {
        var size = 0
        guard sysctlbyname("hw.model", nil, &size, nil, 0) == 0 else { return "unknown" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 else { return "unknown" }
        return bytes.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
    }
}

func elapsed(since start: ContinuousClock.Instant) -> Double {
    let duration = start.duration(to: .now).components
    return Double(duration.seconds) + Double(duration.attoseconds) / 1e18
}

struct AudioProperties: Codable {
    let sampleRate: Double
    let channels: Int
    let frames: Int64
    let durationSeconds: Double
}

private struct MetadataRecord: Encodable {
    let type: String
    let diagnostics: SpeechRecognitionDiagnostics
    let audio: AudioProperties
    let hardware: String
    let cacheState: String
    let preparationSeconds: Double
    let diarizer: SpeechRecognitionDiagnostics?
}

private struct DiarizationRecord: Encodable {
    struct Segment: Encodable {
        let speaker: Int
        let startSeconds: Double
        let endSeconds: Double
    }
    let type = "diarization"
    let isFinal: Bool
    let startSeconds: Double
    let endSeconds: Double
    let finalizationSeconds: Double
    let segments: [Segment]
    init(_ result: SpeakerDiarizer.Result) {
        isFinal = result.isFinal
        startSeconds = result.range.start.seconds
        endSeconds = result.range.end.seconds
        finalizationSeconds = result.resultsFinalizationTime.seconds
        segments = result.segments.map {
            Segment(
                speaker: $0.speaker.id, startSeconds: $0.range.start.seconds,
                endSeconds: $0.range.end.seconds)
        }
    }
}

private struct ResultRecord: Encodable {
    struct Word: Encodable {
        let text: String
        let startSeconds: Double
        let endSeconds: Double
    }
    let type = "result"
    let text: String
    let isFinal: Bool
    let startSeconds: Double
    let endSeconds: Double
    let finalizationSeconds: Double
    /// Seconds since the analyzer started reading input.
    let elapsedSeconds: Double
    /// With `--live`, the seconds of audio that the input had supplied when the result arrived.
    let inputSeconds: Double?
    let words: [Word]
    init(_ result: SpeechTranscriber.Result, clock: ResultClock) {
        text = String(result.text.characters)
        isFinal = result.isFinal
        startSeconds = result.range.start.seconds
        endSeconds = result.range.end.seconds
        finalizationSeconds = result.resultsFinalizationTime.seconds
        elapsedSeconds = elapsed(since: clock.started)
        inputSeconds =
            clock.live
            ? Double(liveInputFrames.load(ordering: .relaxed)) / clock.sampleRate : nil
        words = result.text.runs.compactMap { run in
            guard let range = run[AttributeScopes.SpeechAttributes.TimeRangeAttribute.self] else {
                return nil
            }
            return Word(
                text: String(result.text[run.range].characters), startSeconds: range.start.seconds,
                endSeconds: range.end.seconds)
        }
    }
}

/// The audio frames that the live input has read, for the latency of results.
private let liveInputFrames = Atomic<Int64>(0)

/// When the analyzer started reading input, and how to report the live input position.
private struct ResultClock: Sendable {
    let started: ContinuousClock.Instant
    let live: Bool
    let sampleRate: Double
}

/// File blocks read only when the analyzer asks for more input.
/// The analyzer reads one input sequence at a time, so the file has one reader.
private struct LiveBlocks: AsyncSequence, @unchecked Sendable {
    typealias Element = AnalyzerInput
    let file: AVAudioFile
    let pacedFrom: ContinuousClock.Instant?

    func makeAsyncIterator() -> Iterator { Iterator(file: file, pacedFrom: pacedFrom) }

    struct Iterator: AsyncIteratorProtocol {
        let file: AVAudioFile
        let pacedFrom: ContinuousClock.Instant?
        mutating func next() async throws -> AnalyzerInput? {
            guard file.framePosition < file.length else { return nil }
            let rate = file.processingFormat.sampleRate
            let frames = AVAudioFrameCount(rate / 10)
            guard
                let buffer = AVAudioPCMBuffer(
                    pcmFormat: file.processingFormat, frameCapacity: frames)
            else { throw CLIError("Cannot allocate an audio block.") }
            try file.read(into: buffer, frameCount: frames)
            if let pacedFrom {
                try await Task.sleep(
                    until: pacedFrom + .seconds(Double(file.framePosition) / rate),
                    clock: .continuous)
            }
            liveInputFrames.store(file.framePosition, ordering: .relaxed)
            return AnalyzerInput(buffer: buffer)
        }
    }
}
