import AVFoundation
import CoreMedia
import Speech

/// Uses members of Speech value types that the package exposes through typealiases.
func configure(buffer: AVAudioPCMBuffer, locale: Locale) -> (AnalyzerInput, SpeechAnalyzer) {
    let input = AnalyzerInput(buffer: buffer)
    let context = AnalysisContext()
    context.contextualStrings[.general] = ["AnySpeechRecognizer"]
    let transcriber = SpeechTranscriber(
        locale: locale,
        transcriptionOptions: [],
        reportingOptions: [.volatileResults],
        attributeOptions: [.audioTimeRange, .transcriptionConfidence]
    )
    let progressive = SpeechTranscriber(locale: locale, preset: .progressiveTranscription)
    let detector = SpeechDetector(
        detectionOptions: SpeechDetector.DetectionOptions(sensitivityLevel: .medium),
        reportResults: true
    )
    let analyzer = SpeechAnalyzer(
        modules: [transcriber, progressive, detector],
        options: SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .whileInUse)
    )
    return (input, analyzer)
}

/// Reads word timing and confidence from a transcription result's text.
func describeWords(in text: AttributedString, during timeRange: CMTimeRange) -> [String] {
    var words: [String] = []
    for run in text.runs {
        guard let range = run.audioTimeRange else { continue }
        let confidence = run.transcriptionConfidence ?? 0
        words.append("\(text[run.range].characters) \(range.start.seconds) \(confidence)")
    }
    if let range = text.rangeOfAudioTimeRangeAttributes(intersecting: timeRange) {
        words.append(String(text[range].characters))
    }
    return words
}

@main enum MemberAccessExample {
    static func main() {
        print(describeWords(in: AttributedString("Hello"), during: .zero))
    }
}
