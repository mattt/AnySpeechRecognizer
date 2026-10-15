import AVFoundation
import CoreMedia
import Foundation
import Synchronization
import Testing

@testable import AnySpeechRecognizer

/// The Sortformer executor with tone "speakers"; see ``ToneInference``.
struct SortformerExecutorTests {
    /// Results that a sink received, in order.
    final class Received: Sendable {
        let results = Mutex<[SpeakerDiarizer.Result]>([])
        var all: [SpeakerDiarizer.Result] { results.withLock { $0 } }
    }

    /// A finalization can run while the executor waits to emit the results of an append.
    /// The results of both must still arrive in source order.
    @Test func finalizationDuringEmissionKeepsResultsInOrder() async throws {
        let samples = toneSamples(
            duration: 20, low: [(1, 6), (12, 15)], high: [(6.5, 11), (16, 20)])
        let executor = SortformerExecutor(model: try toneModel())
        let format = AudioFormats.monoFloat16k
        try await executor.prepare(
            request: .init(reportingOptions: [.volatileResults]), format: format)
        let received = Received()
        let firstEmitted = Gate()
        let release = Gate()
        let sink = SpeakerDiarizationResultSink(
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

        let buffer = try #require(
            AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)))
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for (index, sample) in samples.enumerated() {
            buffer.floatChannelData![0][index] = sample
        }
        let audio = TimedAudioBuffer(buffer: buffer, startTime: .zero)
        let branch = AudioBranch(format: format)
        let analysis = Task {
            try await executor.analyze(SpeechAnalysisInput(branch: branch), results: sink)
        }
        try await branch.send(.init(audio: audio, sourceEnd: audio.endTime))

        // The append decides a final result and a volatile preview; the final result waits.
        // While it waits, finalize through the end of the audio.
        await firstEmitted.wait()
        let finalization = Task { try await executor.finalize(through: audio.endTime) }
        // Give a finalization that does not wait for the earlier emission time to emit.
        for _ in 0..<100 where received.all.count == 1 {
            try await Task.sleep(for: .milliseconds(1))
        }
        await release.open()
        try await finalization.value
        branch.finish()
        try await analysis.value

        let results = received.all
        // The append's final and volatile results, then the finalization's final result.
        // Without ordered emission, the finalization's result arrived before the volatile
        // result, which then started before the finalization boundary.
        #expect(results.count == 3)
        #expect(results.first?.isFinal == true)
        #expect(results.dropFirst().first?.isFinal == false)
        #expect(results.last?.isFinal == true)
        expectFinalizationContract(results)
        #expect(
            finalSegments(results).map(\.speaker.id) == [0, 1, 0, 1],
            "Final segments: \(finalSegments(results))")
    }
}
