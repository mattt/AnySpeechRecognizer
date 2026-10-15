import AVFoundation
import CoreMedia
import Foundation
import Synchronization
import Testing

@testable import AnySpeechRecognizer

/// The Parakeet executor with tone "speech"; see ``FakeParakeetEngine``.
struct ParakeetExecutorTests {
    /// Results that a sink received, in order.
    final class Received: Sendable {
        let results = Mutex<[SpeechRecognitionResult]>([])
        var finals: [SpeechRecognitionResult] { results.withLock { $0.filter(\.isFinal) } }
    }

    /// A finalization can run while the executor waits to emit the results of an append.
    /// The results of both must still arrive in source order.
    @Test func finalizationDuringEmissionKeepsResultsInOrder() async throws {
        // Each word is its own phrase, because the pauses between words are 11 frames.
        let script = ToneSpeech.script(count: 9, spacing: 12)
        let executor = ParakeetExecutor(model: ParakeetSpeechModel())
        await executor.use(
            FakeParakeetEngine(), vocabulary: ToneSpeech.vocabulary,
            request: .transcription(.init(attributeOptions: [.audioTimeRange])),
            configuration: ParakeetRecognizerTests.file)
        let received = Received()
        let firstEmitted = Gate()
        let release = Gate()
        let sink = SpeechRecognitionResultSink(
            publish: { result in
                let count = received.results.withLock {
                    $0.append(result)
                    return $0.count
                }
                if count == 1 {
                    await firstEmitted.open()
                    await release.wait()
                }
            },
            volatile: { _ in })

        let format = AudioFormats.monoFloat16k
        let buffer = try #require(
            AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(script.samples.count)))
        buffer.frameLength = AVAudioFrameCount(script.samples.count)
        for (index, sample) in script.samples.enumerated() {
            buffer.floatChannelData![0][index] = sample
        }
        let audio = TimedAudioBuffer(buffer: buffer, startTime: .zero)
        let branch = AudioBranch(format: format)
        let analysis = Task {
            try await executor.analyze(SpeechAnalysisInput(branch: branch), results: sink)
        }
        try await branch.send(.init(audio: audio, sourceEnd: audio.endTime))

        // The append commits every token and emits all but the last word, which waits.
        // While the first emission waits, finalize through the end of the audio.
        await firstEmitted.wait()
        let finalization = Task { try await executor.finalize(through: audio.endTime) }
        // Give a finalization that does not wait for the emission time to emit its result.
        for _ in 0..<100 where received.finals.count == 1 {
            try await Task.sleep(for: .milliseconds(1))
        }
        await release.open()
        try await finalization.value
        branch.finish()
        try await analysis.value

        let finals = received.finals
        #expect(finals.count == 6)
        var finalized = CMTime.zero
        for result in finals {
            #expect(result.range.start >= finalized)
            finalized = result.resultsFinalizationTime
        }
        #expect(finals.last.map { String($0.text.characters) } == " two.")
    }
}
