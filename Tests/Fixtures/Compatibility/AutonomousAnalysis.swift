import AVFoundation
import Speech

/// Analyzes a file autonomously, swaps modules, and updates context during analysis.
func analyzeAutonomously(_ url: URL, locale: Locale) async throws -> [String] {
    let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
    let detector = SpeechDetector(
        detectionOptions: .init(sensitivityLevel: .medium), reportResults: true)
    let context = AnalysisContext()
    context.contextualStrings[.general] = ["SpeechAnalyzer"]
    let options = SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .whileInUse)
    let file = try AVAudioFile(forReading: url)
    async let lines: [String] = {
        var lines: [String] = []
        for try await result in transcriber.results {
            lines.append(String(result.text.characters))
        }
        return lines
    }()
    let analyzer = try await SpeechAnalyzer(
        inputAudioFile: file, modules: [transcriber, detector], options: options,
        analysisContext: context, finishAfterFile: true)
    try await analyzer.setContext(context)
    print(await analyzer.modules.count, await analyzer.context.contextualStrings.count)
    return try await lines
}

/// Starts autonomous analysis of an input sequence, then finishes after a time.
func startSequence(_ audio: AsyncStream<AnalyzerInput>, locale: Locale) async throws {
    let dictation = DictationTranscriber(locale: locale, preset: .progressiveShortDictation)
    let analyzer = SpeechAnalyzer(
        inputSequence: audio, modules: [dictation], analysisContext: AnalysisContext(),
        volatileRangeChangedHandler: nil)
    try await analyzer.prepareToAnalyze(
        in: nil,
        withProgressReadyHandler: { progress in
            print(progress.fractionCompleted)
        })
    try await analyzer.setModules([dictation])
    try await analyzer.finish(after: CMTime(value: 1, timescale: 1))
    for try await result in dictation.results { print(result.text) }
}

@main enum AutonomousAnalysisExample {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            print("Usage: AutonomousAnalysisExample <local-audio-file>")
            return
        }
        let lines = try await analyzeAutonomously(
            URL(fileURLWithPath: CommandLine.arguments[1]), locale: Locale(identifier: "en-US"))
        print(lines.joined())
    }
}
