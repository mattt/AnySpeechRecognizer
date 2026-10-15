import AVFoundation
import Speech

/// Transcribes caller-supplied live audio and replaces volatile text as results arrive.
func transcribeLive(
    _ audio: AsyncStream<AnalyzerInput>, locale: Locale
) async throws -> String {
    let transcriber = SpeechTranscriber(locale: locale, preset: .progressiveTranscription)
    if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        try await request.downloadAndInstall()
    }
    guard
        let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
    else { return "" }
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    try await analyzer.prepareToAnalyze(in: format)
    await analyzer.setVolatileRangeChangedHandler { range, changedStart, changedEnd in
        print(
            "Volatile range \(range.start.seconds)–\(range.end.seconds)", changedStart, changedEnd)
    }
    async let text: String = {
        var finalized = ""
        var volatile = ""
        for try await result in transcriber.results {
            if result.isFinal {
                finalized += String(result.text.characters)
                volatile = ""
            } else {
                volatile = String(result.text.characters)
            }
        }
        return finalized + volatile
    }()
    do {
        let lastSampleTime = try await analyzer.analyzeSequence(audio)
        if let range = await analyzer.volatileRange { print("Pending", range.duration.seconds) }
        try await analyzer.finalize(through: lastSampleTime)
        await analyzer.cancelAnalysis(before: .zero)
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        return try await text
    } catch {
        await analyzer.cancelAndFinishNow()
        throw error
    }
}

@main enum LiveTranscriptionExample {
    static func main() async throws {
        let (audio, input) = AsyncStream<AnalyzerInput>.makeStream()
        guard CommandLine.arguments.count == 2 else {
            print("Usage: LiveTranscriptionExample <local-audio-file>")
            input.finish()
            return
        }
        let file = try AVAudioFile(
            forReading: URL(fileURLWithPath: CommandLine.arguments[1]),
            commonFormat: .pcmFormatInt16, interleaved: true)
        while file.framePosition < file.length {
            let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1600)!
            try file.read(into: buffer, frameCount: 1600)
            input.yield(AnalyzerInput(buffer: buffer))
        }
        input.finish()
        print(try await transcribeLive(audio, locale: Locale(identifier: "en-US")))
    }
}
