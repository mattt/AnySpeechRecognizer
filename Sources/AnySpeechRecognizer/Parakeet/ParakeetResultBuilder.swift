import CoreMedia
import Foundation
import Speech

/// Groups committed Parakeet tokens into final phrases and uncommitted tokens into one
/// volatile result.
///
/// A phrase ends at sentence punctuation, at a pause, or after 32 words.
/// The last committed word waits for the next token, because later tokens can continue it,
/// unless a finalization flushes it.
/// Token times are quantized to 80 ms encoder frames, so a phrase can end after the next
/// word starts. A known overlapping word joins the earlier phrase. After a flush, a later
/// word start that falls before the finalized boundary is moved to the boundary.
/// A token's duration can extend past the received audio; word ends are clipped to it.
struct ParakeetResultBuilder {
    static let sampleRate: CMTimeScale = 16_000
    static let maximumWords = 32
    /// A pause of 0.8 s between words ends a phrase.
    static let pauseSamples = 12_800
    static let sentenceEnds: Set<Character> = [".", "?", "!", "。", "？", "！", "…"]

    let vocabulary: ParakeetVocabulary
    let includeTiming: Bool
    let includeConfidence: Bool
    /// The source time of sample zero in the current segment.
    private(set) var origin: CMTime
    /// The finalized boundary, in source time.
    private(set) var finalized: CMTime
    /// Committed tokens not yet published.
    private var pending: [ParakeetToken] = []
    private var publishedText = false
    /// The samples received in the current segment, which bound word ends.
    var receivedSamples = 0

    init(
        vocabulary: ParakeetVocabulary, origin: CMTime, includeTiming: Bool,
        includeConfidence: Bool
    ) {
        self.vocabulary = vocabulary
        self.origin = origin
        self.finalized = origin
        self.includeTiming = includeTiming
        self.includeConfidence = includeConfidence
    }

    /// Whether committed tokens wait for publication.
    var hasPending: Bool { !pending.isEmpty }

    /// Starts a new segment after a gap in the input. Flush pending tokens first.
    mutating func restart(at origin: CMTime) {
        pending.removeAll()
        receivedSamples = 0
        self.origin = origin
        finalized = max(finalized, origin)
    }

    /// Adds committed tokens and returns final results for complete phrases.
    /// - Parameters:
    ///   - tokens: Newly committed tokens, in order.
    ///   - flush: Whether to publish every pending token, such as for a finalization.
    mutating func commit(_ tokens: [ParakeetToken], flush: Bool) -> [SpeechRecognitionResult] {
        pending += tokens
        let words = vocabulary.words(for: pending.map(\.id))
        func start(_ index: Int) -> Int { pending[words[index].tokens.lowerBound].startSample }
        func end(_ index: Int) -> Int { pending[words[index].tokens.upperBound - 1].endSample }

        var results: [SpeechRecognitionResult] = []
        var first = 0
        var index = 0
        while index < words.count {
            let isLast = index == words.count - 1
            if isLast && !flush { break }
            let endsPhrase =
                isLast || index - first + 1 >= Self.maximumWords
                || words[index].text.last.map(Self.sentenceEnds.contains) == true
                || start(index + 1) - end(index) >= Self.pauseSamples
            guard endsPhrase else {
                index += 1
                continue
            }
            var next = index + 1
            var audioEnd = end(index)
            while next < words.count, start(next) < audioEnd {
                audioEnd = max(audioEnd, end(next))
                next += 1
            }
            // The extension reached the last word, which can still continue.
            if next == words.count && !flush { break }
            results.append(publish(words[first..<next]))
            first = next
            index = next
        }
        if first < words.count {
            pending.removeFirst(words[first].tokens.lowerBound)
        } else {
            pending.removeAll()
        }
        return results
    }

    /// A volatile result for pending committed words and uncommitted tokens,
    /// or `nil` when there are none.
    func volatileResult(_ tokens: [ParakeetToken]) -> SpeechRecognitionResult? {
        let all = pending + tokens
        let words = vocabulary.words(for: all.map(\.id))
        guard !words.isEmpty else { return nil }
        return result(words[...], tokens: all, final: false)
    }

    private mutating func publish(_ words: ArraySlice<ParakeetVocabulary.Word>)
        -> SpeechRecognitionResult
    {
        let result = result(words, tokens: pending, final: true)
        finalized = result.resultsFinalizationTime
        publishedText = true
        return result
    }

    private func time(_ sample: Int) -> CMTime {
        origin + CMTime(value: CMTimeValue(sample), timescale: Self.sampleRate)
    }

    /// Builds attributed text with one run per word, separated by single spaces.
    private func result(
        _ words: ArraySlice<ParakeetVocabulary.Word>, tokens: [ParakeetToken], final: Bool
    ) -> SpeechRecognitionResult {
        var text = AttributedString()
        var rangeStart: CMTime?
        var rangeEnd = finalized
        var previousStart = finalized
        for (offset, word) in words.enumerated() {
            let wordTokens = tokens[word.tokens]
            let wordStart = max(time(wordTokens.first!.startSample), previousStart)
            let wordEnd = max(
                time(min(wordTokens.last!.endSample, receivedSamples)), wordStart)
            if offset > 0 || publishedText { text += AttributedString(" ") }
            var run = AttributedString(word.text)
            if includeTiming {
                run[AttributeScopes.SpeechAttributes.TimeRangeAttribute.self] = CMTimeRange(
                    start: wordStart, end: wordEnd)
            }
            if includeConfidence {
                // The geometric mean of token probabilities. These values are not
                // calibrated against Apple's confidence values.
                let logSum = wordTokens.reduce(0.0) {
                    $0 + log(Double(max($1.probability, .leastNormalMagnitude)))
                }
                run[AttributeScopes.SpeechAttributes.ConfidenceAttribute.self] = exp(
                    logSum / Double(wordTokens.count))
            }
            text += run
            rangeStart = rangeStart ?? wordStart
            rangeEnd = max(rangeEnd, wordEnd)
            previousStart = wordStart
        }
        let range = CMTimeRange(start: rangeStart ?? finalized, end: rangeEnd)
        return SpeechRecognitionResult(
            text: text, range: range, resultsFinalizationTime: final ? range.end : finalized)
    }
}
