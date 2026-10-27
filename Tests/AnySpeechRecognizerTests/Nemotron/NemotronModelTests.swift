import AVFoundation
import CoreMedia
import Foundation
import Synchronization
import Testing

@testable import AnySpeechRecognizer

struct NemotronModelTests {
    static let prompts = [
        "auto": 101, "en": 0, "en-US": 0, "en-GB": 1, "de": 9, "de-DE": 9, "ja-JP": 10, "zh-CN": 4,
    ]

    func model(_ language: NemotronSpeechModel.Language) -> NemotronSpeechModel {
        NemotronSpeechModel(
            language: language, assets: .directory(URL(fileURLWithPath: "/nonexistent")))
    }

    /// A locale matches a prompt by language and region, or by language alone.
    @Test func selectsPromptSlots() throws {
        #expect(try model(.automatic).promptSlot(in: Self.prompts) == 101)
        #expect(try model(.locale(Locale(identifier: "de_DE"))).promptSlot(in: Self.prompts) == 9)
        #expect(try model(.locale(Locale(identifier: "de-AT"))).promptSlot(in: Self.prompts) == 9)
        #expect(try model(.locale(Locale(identifier: "en-GB"))).promptSlot(in: Self.prompts) == 1)
        #expect(try model(.locale(Locale(identifier: "en-AU"))).promptSlot(in: Self.prompts) == 0)
        #expect(try model(.locale(Locale(identifier: "ja-JP"))).promptSlot(in: Self.prompts) == 10)
        #expect(throws: SpeechRecognitionError.self) {
            try model(.locale(Locale(identifier: "ko-KR"))).promptSlot(in: Self.prompts)
        }
    }

    /// Volatile results are supported; fast results and dictation are not.
    @Test func validatesRequests() throws {
        let capabilities = model(.automatic).capabilities
        try SpeechRecognitionRequest.transcription(
            .init(reportingOptions: [.volatileResults], attributeOptions: [.audioTimeRange])
        ).validate(for: capabilities)
        #expect(throws: SpeechRecognitionError.self) {
            try SpeechRecognitionRequest.transcription(.init(reportingOptions: [.fastResults]))
                .validate(for: capabilities)
        }
        #expect(capabilities.languageControl)
        #expect(capabilities.liveInput)
    }

    @Test func namesEncoderFunctions() {
        #expect(NemotronSpeechModel.ChunkSize.ms160.function == "chunk_160ms")
        #expect(
            NemotronSpeechModel.ChunkSize.allCases.map(\.rawValue) == [80, 160, 320, 560, 1120])
    }

    /// A staged directory is ready when it has every file; there is nothing to install.
    @Test func reportsStagedDirectoryAvailability() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(
            path: "nemotron-assets-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = NemotronSpeechModel(assets: .directory(directory))
        #expect(await model.availability() == .missingAssets)
        await #expect(throws: SpeechRecognitionError.self) {
            try await model.assetInstallationRequest()
        }
        for name in NemotronSpeechModel.requiredFiles {
            try Data().write(to: directory.appending(path: name))
        }
        #expect(await model.availability() == .ready)
        #expect(try await model.assetInstallationRequest() == nil)
        let diagnostics = await model.diagnostics()
        #expect(diagnostics.computeConfiguration.contains("chunk_560ms"))
        #expect(diagnostics.modelRepository == directory.standardizedFileURL.path)
    }

    /// Results that a sink received, in order.
    final class Received: Sendable {
        let results = Mutex<[SpeechRecognitionResult]>([])
        var finals: [SpeechRecognitionResult] { results.withLock { $0.filter(\.isFinal) } }
        var volatile: [SpeechRecognitionResult] { results.withLock { $0.filter { !$0.isFinal } } }
    }

    /// The executor publishes phrases of committed tokens and the words that can still
    /// continue as volatile results.
    @Test func executorPublishesPhrases() async throws {
        let script = ToneSpeech.script(count: 6, spacing: 4)
        let executor = NemotronExecutor(model: model(.automatic))
        await executor.use(
            FakeNemotronEngine(chunkFrames: 2), vocabulary: ToneSpeech.vocabulary,
            request: .transcription(
                .init(reportingOptions: [.volatileResults], attributeOptions: [.audioTimeRange])),
            prompt: 101)
        let received = Received()
        let sink = SpeechRecognitionResultSink(
            publish: { result in received.results.withLock { $0.append(result) } },
            volatile: { _ in })
        let format = AudioFormats.monoFloat16k
        let branch = AudioBranch(format: format)
        let analysis = Task {
            try await executor.analyze(SpeechAnalysisInput(branch: branch), results: sink)
        }
        var end = CMTime.zero
        for start in stride(from: 0, to: script.samples.count, by: 1600) {
            let count = min(1600, script.samples.count - start)
            let buffer = try #require(
                AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)))
            buffer.frameLength = AVAudioFrameCount(count)
            for index in 0..<count {
                buffer.floatChannelData![0][index] = script.samples[start + index]
            }
            let audio = TimedAudioBuffer(
                buffer: buffer, startTime: CMTime(value: CMTimeValue(start), timescale: 16_000))
            try await branch.send(.init(audio: audio, sourceEnd: audio.endTime))
            end = audio.endTime
        }
        // As the analyzer does at the end of input, finalize after the executor reads it all.
        branch.finish()
        try await analysis.value
        try await executor.finalize(through: end)

        let text = received.finals.map { String($0.text.characters) }.joined()
        let expected = ToneSpeech.vocabulary.text(for: script.tokens.map(\.id))
        #expect(text.trimmingCharacters(in: .whitespaces) == expected, "\(text) vs \(expected)")
        #expect(!received.volatile.isEmpty)
        var finalized = CMTime.zero
        for result in received.finals {
            #expect(result.range.start >= finalized)
            finalized = result.resultsFinalizationTime
        }
    }
}

/// Opt-in tests with local Nemotron models; set `ANYSPEECH_NEMOTRON_MODELS` to a directory
/// that `Tools/NemotronConversion/convert.py` wrote, and `ANYSPEECH_NEMOTRON_AUDIO` to speech.
@Suite(.enabled(if: NemotronFixtures.modelDirectory != nil), .serialized)
struct NemotronCoreMLTests {
    let directory = NemotronFixtures.modelDirectory ?? URL(fileURLWithPath: "/")

    func functions() throws -> [String] {
        let record =
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appending(path: "conversion.json")))
            as? [String: Any]
        let conversion = record?["conversion"] as? [String: Any]
        let components = conversion?["components"] as? [String: Any]
        let encoder = components?["encoder"] as? [String: Any]
        let functions = encoder?["functions"] as? [[String: Any]] ?? []
        return functions.compactMap { $0["function"] as? String }
    }

    func recognize(
        _ samples: [Float], engine: CoreMLNemotronEngine, block: Int, prompt: Int
    ) throws -> [ParakeetToken] {
        let vocabulary = try ParakeetVocabulary(contentsOf: directory.appending(path: "vocab.json"))
        let recognizer = try NemotronRecognizer(
            engine: engine, vocabulary: vocabulary, prompt: prompt)
        var tokens: [ParakeetToken] = []
        var offset = 0
        while offset < samples.count {
            let end = min(samples.count, offset + block)
            recognizer.append(Array(samples[offset..<end]))
            offset = end
            tokens += try recognizer.process().committed
        }
        tokens += try recognizer.process(endOfInput: true).committed
        return tokens
    }

    /// Each function loads with the expected shapes, decodes silence as nothing, and gives the
    /// same tokens whatever the block size, because every chunk receives the same input.
    @Test func everyFunctionStreamsDeterministically() throws {
        let prompts = try NemotronSpeechModel.prompts(
            contentsOf: directory.appending(path: "prompts.json"))
        let prompt = try #require(prompts["auto"])
        let speech = try NemotronFixtures.audio.map(ParakeetFixtures.samples)
        for function in try functions() {
            let engine = try CoreMLNemotronEngine(
                directory: directory, configuration: .init(function: function))
            #expect(engine.chunkFrames > 0)
            let silence = try recognize(
                [Float](repeating: 0, count: 48_000), engine: engine, block: 1600, prompt: prompt)
            #expect(silence.allSatisfy { ParakeetVocabulary.isControl($0.piece) })
            guard let speech else { continue }
            let small = try recognize(speech, engine: engine, block: 1600, prompt: prompt)
            let large = try recognize(speech, engine: engine, block: 16_000, prompt: prompt)
            let vocabulary = try ParakeetVocabulary(
                contentsOf: directory.appending(path: "vocab.json"))
            print("\(function): \(vocabulary.text(for: small.map(\.id)))")
            #expect(!small.isEmpty)
            #expect(small.map(\.id) == large.map(\.id))
        }
    }
}

enum NemotronFixtures {
    static var environment: [String: String] { ProcessInfo.processInfo.environment }
    static var modelDirectory: URL? {
        environment["ANYSPEECH_NEMOTRON_MODELS"].map { URL(fileURLWithPath: $0) }
    }
    static var audio: URL? {
        environment["ANYSPEECH_NEMOTRON_AUDIO"].map { URL(fileURLWithPath: $0) }
    }
}
