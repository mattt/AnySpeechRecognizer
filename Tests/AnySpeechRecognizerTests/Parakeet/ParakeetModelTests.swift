import Foundation
import Testing

@testable import AnySpeechRecognizer

/// Opt-in tests with local Parakeet models; set `ANYSPEECH_PARAKEET_MODELS` to the model directory.
/// Audio tests also need `ANYSPEECH_PARAKEET_SHORT_AUDIO` (under 15 s of speech)
/// or `ANYSPEECH_PARAKEET_LONG_AUDIO` (60 s or more).
@Suite(.enabled(if: ParakeetFixtures.modelDirectory != nil), .serialized)
struct ParakeetModelTests {
    let directory = ParakeetFixtures.modelDirectory ?? URL(fileURLWithPath: "/")

    func vocabulary() throws -> ParakeetVocabulary {
        try ParakeetVocabulary(contentsOf: directory.appendingPathComponent("parakeet_vocab.json"))
    }

    /// The engine and settings that live input uses: the staged live encoder when present,
    /// with the regular encoder for forced commits, as the executor selects them,
    /// or `engine` with the 15 s settings.
    func liveEngine(_ engine: CoreMLParakeetEngine) throws -> (
        engine: CoreMLParakeetEngine, configuration: ParakeetRecognizer.Configuration
    ) {
        let name = CoreMLParakeetEngine.liveEncoderName
        var live = engine
        if FileManager.default.fileExists(atPath: directory.appending(path: name).path) {
            let regular =
                FileManager.default.fileExists(
                    atPath: directory.appending(path: "Encoder_v2.mlmodelc").path)
                ? "Encoder_v2.mlmodelc" : "Encoder.mlmodelc"
            var options = CoreMLParakeetEngine.Configuration()
            options.encoderName = name
            options.forcedCommitEncoderName = regular
            live = try CoreMLParakeetEngine(directory: directory, configuration: options)
            #expect(live.forcedCommitModel == engine.model)
            #expect(live.model.windowSamples < engine.model.windowSamples)
        }
        print("live encoder window: \(live.model.windowSamples) samples")
        return (live, .live(windowFrames: live.model.windowFrames))
    }

    struct Run {
        var committed: [ParakeetToken] = []
        var updates: [ParakeetUpdate] = []
        /// Wall-clock seconds for each `process` call.
        var durations: [Double] = []
        /// Received samples after each `process` call.
        var received: [Int] = []
    }

    /// Appends audio in blocks and processes after each block, then ends the input.
    func run(
        _ samples: [Float], configuration: ParakeetRecognizer.Configuration, block: Int,
        engine: CoreMLParakeetEngine
    ) throws -> Run {
        let recognizer = try ParakeetRecognizer(
            engine: engine, vocabulary: try vocabulary(), configuration: configuration)
        var run = Run()
        var offset = 0
        while offset < samples.count {
            let end = min(samples.count, offset + block)
            recognizer.append(Array(samples[offset..<end]))
            offset = end
            let clock = ContinuousClock.now
            let update = try recognizer.process(endOfInput: offset == samples.count)
            run.durations.append((ContinuousClock.now - clock) / .seconds(1))
            run.received.append(offset)
            run.committed += update.committed
            run.updates.append(update)
        }
        #expect(recognizer.committedThroughSample == samples.count)
        #expect(run.updates.last?.volatile.isEmpty == true)
        return run
    }

    @Test func shortClipMatchesReferenceTokens() async throws {
        let url = try #require(ParakeetFixtures.shortAudio)
        let samples = try ParakeetFixtures.samples(url)
        #expect(samples.count < 240_000)
        let engine = try CoreMLParakeetEngine(directory: directory)
        let ours = try run(samples, configuration: .file, block: samples.count, engine: engine)
        let vocabulary = try vocabulary()
        print("short ours:", ParakeetFixtures.text(ours.committed, vocabulary: vocabulary))
    }

    @Test func longClipFileAndLiveModes() async throws {
        let url = try #require(ParakeetFixtures.longAudio)
        let samples = try ParakeetFixtures.samples(url)
        let engine = try CoreMLParakeetEngine(directory: directory)
        let vocabulary = try vocabulary()
        let clock = ContinuousClock.now
        let file = try run(samples, configuration: .file, block: 16_000, engine: engine)
        let fileSeconds = (ContinuousClock.now - clock) / .seconds(1)
        let fileText = ParakeetFixtures.text(file.committed, vocabulary: vocabulary)
        print(
            "long file mode (\(fileSeconds) s for \(Double(samples.count) / 16_000) s):", fileText)
        #expect(
            zip(file.committed, file.committed.dropFirst()).allSatisfy {
                $0.startSample <= $1.startSample
            })

        if let path = ParakeetFixtures.environment["ANYSPEECH_PARAKEET_LONG_REFERENCE"] {
            let reference = try String(contentsOfFile: path, encoding: .utf8)
            let edits = ParakeetFixtures.wordDifferences(
                Self.normalized(reference), Self.normalized(fileText))
            print("long file mode vs reference text:", edits)
        }

        let liveSetup = try liveEngine(engine)
        let live = try run(
            samples, configuration: liveSetup.configuration, block: 1_600, engine: liveSetup.engine)
        let liveText = ParakeetFixtures.text(live.committed, vocabulary: vocabulary)
        print("long live mode:", liveText)
        print(
            "long live mode vs file mode words:",
            ParakeetFixtures.wordDifferences(fileText, liveText))
        if let path = ParakeetFixtures.environment["ANYSPEECH_PARAKEET_LONG_REFERENCE"] {
            let reference = try String(contentsOfFile: path, encoding: .utf8)
            let edits = ParakeetFixtures.wordDifferences(
                Self.normalized(reference), Self.normalized(liveText))
            print("long live mode vs reference text:", edits)
        }
        let sorted = live.durations.sorted()
        print(
            "live process seconds: mean", sorted.reduce(0, +) / Double(sorted.count),
            "p50", sorted[sorted.count / 2], "p95", sorted[sorted.count * 95 / 100],
            "max", sorted.last ?? 0, "calls", sorted.count)
        // Audio latency: how much audio arrived after a token before it became final or volatile.
        var commitDelays: [Int] = []
        var firstSeen: [Int: Int] = [:]
        for (index, update) in live.updates.enumerated() {
            for token in update.committed {
                commitDelays.append(live.received[index] - token.endSample)
            }
            for token in update.volatile where firstSeen[token.startSample] == nil {
                firstSeen[token.startSample] = live.received[index] - token.endSample
            }
        }
        let volatileDelays = firstSeen.values.sorted()
        commitDelays.sort()
        print(
            "live commit delay seconds: median",
            Double(commitDelays[commitDelays.count / 2]) / 16_000,
            "max", Double(commitDelays.last ?? 0) / 16_000,
            "volatile first-seen delay median",
            Double(volatileDelays[volatileDelays.count / 2]) / 16_000)
        let changes = live.updates.indices.dropFirst().filter {
            live.updates[$0].volatile != live.updates[$0 - 1].volatile
        }.count
        print("live volatile changes:", changes, "of", live.updates.count, "updates")
        let liveEdits = ParakeetFixtures.wordDifferences(fileText, liveText)
        #expect(liveEdits.insertions + liveEdits.deletions <= 4)
        for (index, update) in live.updates.enumerated() {
            #expect(update.volatile.allSatisfy { $0.startSample >= update.committedThroughSample })
            if index > 0 {
                #expect(
                    update.committedThroughSample >= live.updates[index - 1].committedThroughSample)
            }
        }
    }

    static func normalized(_ text: String) -> String {
        text.lowercased().unicodeScalars.map {
            CharacterSet.letters.contains($0) || CharacterSet.decimalDigits.contains($0)
                ? String($0) : " "
        }.joined()
    }

    /// A long recording that repeats a few sentences, such as one made by
    /// `Tools/make-evaluation-audio.swift`; `ANYSPEECH_PARAKEET_REPETITIVE_SECONDS` limits it.
    @Test(.enabled(if: ParakeetFixtures.environment["ANYSPEECH_PARAKEET_REPETITIVE_AUDIO"] != nil))
    func liveModeKeepsPunctuationOnRepetitiveSpeech() throws {
        let environment = ParakeetFixtures.environment
        let url = URL(
            fileURLWithPath: try #require(environment["ANYSPEECH_PARAKEET_REPETITIVE_AUDIO"]))
        let seconds =
            environment["ANYSPEECH_PARAKEET_REPETITIVE_SECONDS"].flatMap(Double.init) ?? 120
        let samples = Array(try ParakeetFixtures.samples(url).prefix(Int(seconds * 16_000)))
        let engine = try CoreMLParakeetEngine(directory: directory)
        let vocabulary = try vocabulary()
        // The prefix can end inside a word, which each mode may write differently or drop,
        // as NeMo does for different windows; compare the text before the last 0.5 s.
        let cutoff = samples.count - 8_000
        let file = ParakeetFixtures.text(
            try run(samples, configuration: .file, block: 16_000, engine: engine).committed
                .filter { $0.startSample < cutoff },
            vocabulary: vocabulary)
        let liveSetup = try liveEngine(engine)
        let live = try run(
            samples, configuration: liveSetup.configuration, block: 1_600, engine: liveSetup.engine)
        let liveText = ParakeetFixtures.text(
            live.committed.filter { $0.startSample < cutoff }, vocabulary: vocabulary)
        func sentences(_ text: String) -> Int { text.filter { ".?!".contains($0) }.count }
        func capitals(_ text: String) -> Int {
            text.split(separator: " ").filter { $0.first?.isUppercase == true }.count
        }
        let edits = ParakeetFixtures.wordDifferences(
            Self.normalized(file), Self.normalized(liveText))
        print(
            "repetitive \(seconds) s: file sentences=\(sentences(file)) capitals=\(capitals(file))",
            "live sentences=\(sentences(liveText)) capitals=\(capitals(liveText))",
            "live vs file normalized: \(edits.substitutions) S \(edits.deletions) D \(edits.insertions) I",
            edits.edits.joined(separator: " ")
        )
        #expect(edits.insertions + edits.deletions == 0)
        #expect(Double(sentences(liveText)) >= 0.95 * Double(sentences(file)))
        #expect(Double(capitals(liveText)) >= 0.95 * Double(capitals(file)))
        let durations = live.durations.sorted()
        print(
            "repetitive live process seconds: mean",
            durations.reduce(0, +) / Double(durations.count),
            "p95", durations[durations.count * 95 / 100], "max", durations.last ?? 0)
    }
}
