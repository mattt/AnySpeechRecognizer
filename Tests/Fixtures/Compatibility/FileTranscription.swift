import AVFoundation
import Speech

enum ExampleError: Error {
    case unsupportedLocale
}

func transcribeFile(_ url: URL, locale requestedLocale: Locale) async throws -> String {
    guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: requestedLocale) else {
        throw ExampleError.unsupportedLocale
    }
    let transcriber = SpeechTranscriber(
        locale: locale,
        transcriptionOptions: [],
        reportingOptions: [],
        attributeOptions: [.audioTimeRange]
    )
    if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        try await request.downloadAndInstall()
    }
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    async let text: String = {
        var text = ""
        for try await result in transcriber.results where result.isFinal {
            text += String(result.text.characters)
        }
        return text
    }()
    do {
        let file = try AVAudioFile(forReading: url)
        if let lastSampleTime = try await analyzer.analyzeSequence(from: file) {
            try await analyzer.finalizeAndFinish(through: lastSampleTime)
        } else {
            await analyzer.cancelAndFinishNow()
        }
        return try await text
    } catch {
        await analyzer.cancelAndFinishNow()
        throw error
    }
}

@main enum FileTranscriptionExample {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            print("Usage: FileTranscriptionExample <local-audio-file>")
            return
        }
        let url = URL(fileURLWithPath: CommandLine.arguments[1])
        print(try await transcribeFile(url, locale: Locale(identifier: "en-US")))
    }
}
