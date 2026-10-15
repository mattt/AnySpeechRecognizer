import AVFoundation
import CoreMedia
import Foundation
import Testing

@testable import AnySpeechRecognizer

/// Writes 16 kHz mono samples to a temporary WAV file.
private func writeSamples(_ samples: [Float]) throws -> (URL, AVAudioFile) {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "anyspeech-diarization-\(UUID()).wav")
    let format = try #require(
        AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false))
    do {
        var settings = format.settings
        settings[AVLinearPCMIsNonInterleaved] = false
        let output = try AVAudioFile(forWriting: url, settings: settings)
        let buffer = try #require(
            AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)))
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer {
            buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count)
        }
        try output.write(from: buffer)
    }
    return (url, try AVAudioFile(forReading: url))
}

private let tones = toneSamples(duration: 12, low: [(0.5, 4), (8, 11)], high: [(4.5, 7.5)])

@Test func diarizerPublishesValidatedResultsThroughTheAnalyzer() async throws {
    let (url, file) = try audioFile(frames: 48_000)
    defer { try? FileManager.default.removeItem(at: url) }
    let volatile = diarization([segment(0, 0, 0.9)], range: range(0, 0.9), final: 0)
    let first = diarization([segment(0, 0, 1)], final: 1.2)
    let second = diarization([segment(0, 1.2, 1.8), segment(1, 1.5, 2.5)], final: 2.5)
    let last = diarization([segment(1, 2.5, 2.9)], final: 2.9)
    let executor = FakeDiarizationExecutor(results: [volatile, first, second], deferred: [last])
    let diarizer = SpeakerDiarizer(
        model: FakeDiarizationModel(executor: executor), reportingOptions: [.volatileResults])
    let analyzer = SpeechAnalyzer(modules: [diarizer])
    async let results = collect(diarizer)
    let end = try #require(try await analyzer.analyzeSequence(from: file))
    try await analyzer.finalizeAndFinish(through: end)
    let values = try await results
    // The final result that covers the volatile result can replace it while it is unread.
    #expect(values.filter { $0.isFinal } == [first, second, last])
    #expect(values.filter { !$0.isFinal }.allSatisfy { $0 == volatile })
    expectFinalizationContract(values)
    #expect(await executor.finalizeTimes == [end])
    #expect(await analyzer.state == .finished)
}

@Test func invalidDiarizationResultsFailTheSession() async throws {
    let invalid: [SpeakerDiarizer.Result] = [
        // A segment outside its result.
        .init(
            segments: [segment(0, 0, 2)], range: range(0, 1), resultsFinalizationTime: seconds(1)),
        // A speaker the model cannot report.
        diarization([segment(7, 0, 1)]),
        // Segments out of order.
        diarization([segment(0, 0.5, 1), segment(1, 0, 1)], range: range(0, 1)),
    ]
    for result in invalid {
        let (url, file) = try audioFile(frames: 32_000)
        defer { try? FileManager.default.removeItem(at: url) }
        let diarizer = SpeakerDiarizer(
            model: FakeDiarizationModel(executor: FakeDiarizationExecutor(results: [result])))
        let analyzer = SpeechAnalyzer(modules: [diarizer])
        await #expect(throws: SpeechRecognitionError.self) {
            try await analyzer.analyzeSequence(from: file)
            try await analyzer.finalizeAndFinishThroughEndOfInput()
        }
        await #expect(throws: SpeechRecognitionError.self) { try await collect(diarizer) }
    }
}

@Test func unsupportedOptionsAndMissingAssetsFailBeforeReading() async throws {
    let (url, file) = try audioFile()
    defer { try? FileManager.default.removeItem(at: url) }
    var models: [(FakeDiarizationModel, Set<SpeakerDiarizer.ReportingOption>)] = []
    var noVolatile = FakeDiarizationModel(executor: FakeDiarizationExecutor())
    noVolatile.capabilities.reportingOptions = []
    models.append((noVolatile, [.volatileResults]))
    var missing = FakeDiarizationModel(executor: FakeDiarizationExecutor())
    missing.availability = .missingAssets
    models.append((missing, []))
    for (model, options) in models {
        let diarizer = SpeakerDiarizer(model: model, reportingOptions: options)
        let analyzer = SpeechAnalyzer(modules: [diarizer])
        await #expect(throws: SpeechRecognitionError.self) {
            try await analyzer.analyzeSequence(from: file)
        }
        #expect(await model.executor.prepareCount == 0)
        #expect(await model.executor.pulledFrames == 0)
    }
}

@Test func assetInventoryCoversDiarizers() async throws {
    var missing = FakeDiarizationModel(executor: FakeDiarizationExecutor())
    missing.availability = .missingAssets
    let progress = Progress(totalUnitCount: 1)
    missing.installation = AssetInstallationRequest(progress: progress) {
        progress.completedUnitCount = 1
    }
    let transcriber = SpeechTranscriber(model: FakeModel(executor: FakeExecutor()))
    let diarizer = SpeakerDiarizer(model: missing)
    #expect(await AssetInventory.status(forModules: [transcriber, diarizer]) == .supported)
    let request = try #require(
        try await AssetInventory.assetInstallationRequest(supporting: [transcriber, diarizer]))
    try await request.downloadAndInstall()
    #expect(progress.completedUnitCount == 1)

    let ready = SpeakerDiarizer(model: FakeDiarizationModel(executor: FakeDiarizationExecutor()))
    #expect(await AssetInventory.status(forModules: [transcriber, ready]) == .installed)
    #expect(try await AssetInventory.assetInstallationRequest(supporting: [ready]) == nil)

    var noVolatile = FakeDiarizationModel(executor: FakeDiarizationExecutor())
    noVolatile.capabilities.reportingOptions = []
    let unsupported = SpeakerDiarizer(model: noVolatile, reportingOptions: [.volatileResults])
    #expect(await AssetInventory.status(forModules: [unsupported]) == .unsupported)
}

@Test func addingADiarizerDoesNotChangeTranscriptionResults() async throws {
    let (url, _) = try writeSamples(tones)
    defer { try? FileManager.default.removeItem(at: url) }
    func transcribe(withDiarizer: Bool) async throws -> (
        [SpeechTranscriber.Result], [SpeakerDiarizer.Result]
    ) {
        let executor = FakeExecutor(
            snapshots: [
                snapshot("Hello.", start: 8_000, end: 64_000),
                snapshot("Hi.", start: 72_000, end: 120_000),
            ],
            deferred: [snapshot("Bye.", start: 128_000, end: 176_000)])
        let transcriber = SpeechTranscriber(
            model: FakeModel(executor: executor), attributeOptions: [.audioTimeRange])
        let diarizer = SpeakerDiarizer(model: try toneModel())
        let modules: [any SpeechModule] = withDiarizer ? [transcriber, diarizer] : [transcriber]
        let analyzer = SpeechAnalyzer(modules: modules)
        async let transcripts = collect(transcriber)
        async let speakers: [SpeakerDiarizer.Result] =
            withDiarizer ? collect(diarizer) : []
        let file = try AVAudioFile(forReading: url)
        let end = try #require(try await analyzer.analyzeSequence(from: file))
        try await analyzer.finalizeAndFinish(through: end)
        return (try await transcripts, try await speakers)
    }
    let (alone, _) = try await transcribe(withDiarizer: false)
    let (together, speakers) = try await transcribe(withDiarizer: true)
    #expect(alone == together)
    #expect(alone.count == 3)
    #expect(Set(finalSegments(speakers).map(\.speaker.id)) == [0, 1])
}

@Test func sortformerDiarizerMatchesFileAndLiveInput() async throws {
    let (url, file) = try writeSamples(tones)
    defer { try? FileManager.default.removeItem(at: url) }
    let fileDiarizer = SpeakerDiarizer(model: try toneModel())
    let fileAnalyzer = SpeechAnalyzer(modules: [fileDiarizer])
    async let fileResults = collect(fileDiarizer)
    let end = try #require(try await fileAnalyzer.analyzeSequence(from: file))
    try await fileAnalyzer.finalizeAndFinish(through: end)

    let liveDiarizer = SpeakerDiarizer(model: try toneModel(), reportingOptions: [.volatileResults])
    let (stream, continuation) = liveInput()
    let liveAnalyzer = SpeechAnalyzer(inputSequence: stream, modules: [liveDiarizer])
    async let liveResults = collect(liveDiarizer)
    var start = 0
    while start < tones.count {
        let stop = min(start + 1_600, tones.count)
        continuation.yield(int16Input(tones[start..<stop], start: nil))
        start = stop
    }
    continuation.finish()
    try await liveAnalyzer.finalizeAndFinishThroughEndOfInput()

    let fileValues = try await fileResults
    let liveValues = try await liveResults
    expectFinalizationContract(fileValues)
    expectFinalizationContract(liveValues)
    #expect(liveValues.contains { !$0.isFinal })
    #expect(!finalSegments(fileValues).isEmpty)
    // Int16 live input quantizes the tones, so compare speakers and boundaries by frame.
    let fileSegments = finalSegments(fileValues)
    let liveSegments = finalSegments(liveValues)
    #expect(fileSegments.map(\.speaker) == liveSegments.map(\.speaker))
    for (lhs, rhs) in zip(fileSegments, liveSegments) {
        #expect(abs(lhs.range.start.seconds - rhs.range.start.seconds) <= 0.081)
        #expect(abs(lhs.range.end.seconds - rhs.range.end.seconds) <= 0.081)
    }
    #expect(fileSegments.map(\.speaker.id) == [0, 1, 0])
}
