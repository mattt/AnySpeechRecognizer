import Accelerate
import Foundation

/// A recognized token with absolute sample times since the first appended sample.
struct ParakeetToken: Sendable, Equatable {
    var id: Int
    var piece: String
    var startSample: Int
    var endSample: Int
    var probability: Float
}

/// Tokens that one call to ``ParakeetRecognizer/process(finalizeThrough:endOfInput:)`` produced.
struct ParakeetUpdate: Sendable, Equatable {
    /// Tokens committed by this call. Committed tokens are final and never repeat.
    var committed: [ParakeetToken] = []
    /// The complete uncommitted hypothesis. It replaces the volatile tokens of earlier updates.
    var volatile: [ParakeetToken] = []
    /// Results before this sample are final.
    var committedThroughSample: Int
}

/// Recognizes 16 kHz mono audio incrementally with overlapping encoder windows.
///
/// The window is the engine's fixed encoder window: 15 seconds for the published encoder,
/// or a shorter live encoder's window, such as 8 seconds.
/// Each window has left context, a chunk, and right context. Tokens that the decoder emits
/// in the chunk are committed, and the decoder context after the last committed token carries
/// into the next window, so committed tokens do not repeat across windows.
/// With 15-second windows, each window rebuilds that context from the current sentence:
/// the last committed sentence-final token and the tokens after it. A context that starts in
/// the middle of a sentence can switch Parakeet to unpunctuated, lowercase output, which then
/// feeds every later context. A context with a long finished sentence can make the decoder
/// emit only blanks after the sentence's period, so the decoder also restarts its context
/// after each sentence-final token; see ``ParakeetTDTDecoder/sentenceEndContexts``.
/// Shorter live windows instead replay the committed tokens of the last 75 frames (6 s);
/// see ``Configuration/sentenceScopedHistory``.
/// When the configuration reports volatile results, tokens decoded after the commit frame
/// become volatile and the next update replaces them.
///
/// A forced commit, at the end of input or for a finalization, has no right context.
/// When the engine has a longer forced-commit encoder, such as the 15-second encoder beside
/// an 8-second live encoder, forced commits use it and its window fills with earlier audio.
/// With 8-second windows alone, the last words before an abrupt end of input were often
/// decoded as blanks.
///
/// Encoder features differ between windows, so a token's emission frame can move by a frame
/// or two. The next window therefore resumes after the last committed token or a margin before
/// the commit frame, whichever is later, instead of where the last window's durations and
/// blanks ended. Without this, a token near the boundary can fall between two windows, and
/// entering a word in the middle can make the decoder emit only blanks for many seconds.
/// Resuming after the last token can start inside its duration, at frames that greedy
/// decoding skips. Parakeet Ultra predicted a four-frame blank at such a frame after "at",
/// skipped the start of "a quarter past", and emitted only blanks until "nine". Until the
/// frame where the last token's duration ends, blanks therefore advance at most to that frame.
///
/// On natural speech, some window placements make Parakeet emit only blanks for the rest of
/// the window, usually after a sentence-final token, whatever the decoder context. In one
/// LibriSpeech file, the audio after 10.64 s decoded to one sentence in a 15-second window and
/// to three in a 13-second window, with the encoder on the CPU, GPU, or Neural Engine.
/// When a window commits no token for its last ``Configuration/blankRecoveryFrames`` frames,
/// or for twice as many frames before a later token, the recognizer encodes another window
/// that starts where the gap starts and decodes the gap again from the initial context.
/// A gap without an 80 ms frame at speech level, −20 dB relative to the mean power of the
/// received audio, is not decoded again; see ``speechLevel``. Otherwise the recognizer
/// keeps the new tokens when they outnumber the tokens that the first decoding committed
/// after the gap and their mean probability is at least 0.8, and then looks for a later gap
/// in the window.
///
/// NeMo's PyTorch model decodes the same blanks from the same window samples, and NeMo's
/// buffered streaming inference loses even more words on such audio, so the behavior belongs
/// to the model under windowing. It depends on where a window starts: windows that started
/// inside speech lost words far more often than windows that started in a pause. File-mode
/// windows therefore start at the quietest frame within ``Configuration/quietStartFrames``.
///
/// In live and fast modes on long recordings, Parakeet inserted repeated filler, such as
/// "and the other thing and the other thing", mostly in pauses. It emitted the filler as
/// zero-duration tokens, often up to ``Configuration/maxSymbolsPerFrame`` at each of a few
/// frames, so these modes discard the tokens of a frame that reaches the limit; see
/// ``Configuration/discardsFramesAtSymbolLimit``.
///
/// The recognizer retains at most one window of audio before the commit point, plus the
/// uncommitted audio. One owner uses it at a time.
final class ParakeetRecognizer {
    struct Configuration: Sendable, Equatable {
        /// Encoder frames of audio before the chunk, when available.
        var leftContextFrames: Int
        /// The maximum encoder frames committed by one window.
        var chunkFrames: Int
        /// Encoder frames of audio after the chunk that a regular window waits for.
        var rightContextFrames: Int
        /// Whether updates report tokens decoded after the commit frame.
        var reportsVolatileResults: Bool
        /// Frames before the commit frame that the next window decodes again when this
        /// window emitted no token there. A forced commit uses no margin.
        var boundaryMarginFrames = 4
        /// The longest history, in encoder frames, of committed tokens that the decoder
        /// context replays; with a sentence-scoped history, the limit applies only when no
        /// sentence ends within it.
        var decoderHistoryFrames = 250
        /// Whether the decoder context covers the current sentence and restarts after each
        /// sentence-final token. Otherwise it replays every committed token of the last
        /// `decoderHistoryFrames` frames from the start token, across sentence ends.
        ///
        /// With 8-second windows and 63 frames of left context, the sentence-scoped context
        /// emitted only blanks for up to two sentences after a sentence end (41 of 415 words
        /// on one clip), and a shorter history limit did not prevent it.
        /// A 75-frame history without the sentence scope deleted no words in the same tests
        /// and kept every sentence end of a 10-minute repetitive clip.
        var sentenceScopedHistory = true
        var maxSymbolsPerFrame = 10
        /// Whether the decoder discards the tokens of a frame that reaches
        /// `maxSymbolsPerFrame`; see ``ParakeetTDTDecoder/discardsFramesAtSymbolLimit``.
        var discardsFramesAtSymbolLimit = false
        /// Whether windows reuse the mel features of frames that earlier windows computed.
        /// Live windows overlap almost completely, so they compute only new frames.
        var reusesMelFrames = false
        /// Frames without a committed token at the end of a window after which the recognizer
        /// decodes the gap again from a window that starts at the gap, or `nil` to decode each
        /// window once. A gap that ends with a token must be twice as long.
        ///
        /// On six long LibriSpeech files, 8 frames (0.64 s) recovered live windows that
        /// 12 frames did not: a live window commits 12 frames, so with a longer gap the first
        /// words after a pause were committed as blank before a recovery, and committed audio
        /// is not decoded again.
        var blankRecoveryFrames: Int? = 8
        /// Encoder frames before the commit frame within which a regular window starts at the
        /// quietest 80 ms frame, or `nil` to start `leftContextFrames` before the commit frame.
        ///
        /// On long LibriSpeech files, windows that started inside speech lost words far more
        /// often than windows that started in a pause, with the Core ML encoder and with
        /// NeMo's PyTorch model alike, and decoded repeated filler such as "and the other
        /// thing". In live mode, the same search moved the filler to other windows.
        var quietStartFrames: ClosedRange<Int>? = nil

        /// The largest chunk the window allows: 2 s of right context, and 0.64 to 4 s of left
        /// context that starts at the quietest frame, for chunks of 8.96 to 12.32 s.
        static let file = Configuration(
            leftContextFrames: 25, chunkFrames: .max, rightContextFrames: 25,
            reportsVolatileResults: false, quietStartFrames: 8...50)
        /// Commits 0.96 s chunks after 2 s of right context, with 12 s of left context.
        static let live = Configuration(
            leftContextFrames: 150, chunkFrames: 12, rightContextFrames: 25,
            reportsVolatileResults: true, discardsFramesAtSymbolLimit: true,
            reusesMelFrames: true)
        /// Commits 0.96 s chunks after 0.96 s of right context, for lower latency.
        /// In testing, 1 s of right context caused occasional inserted words.
        static let fast = Configuration(
            leftContextFrames: 150, chunkFrames: 12, rightContextFrames: 12,
            reportsVolatileResults: true, discardsFramesAtSymbolLimit: true,
            reusesMelFrames: true)

        /// The left context of a window shorter than ``live`` needs: 63 frames (5.04 s).
        /// With the 8 s encoder, 63 frames had the lowest error in the conversion tool's
        /// live study; 88 to 112 frames deleted words.
        static let shortWindowLeftContextFrames = 63
        /// The decoder history of short windows: 75 frames (6 s) of committed tokens,
        /// without the sentence scope.
        static let shortWindowDecoderHistoryFrames = 75

        /// ``live`` for an encoder window of `windowFrames` encoder frames.
        static func live(windowFrames: Int) -> Configuration {
            live.fitted(to: windowFrames)
        }

        /// ``fast`` for an encoder window of `windowFrames` encoder frames.
        static func fast(windowFrames: Int) -> Configuration {
            fast.fitted(to: windowFrames)
        }

        /// This configuration when its window fits, or short-window settings otherwise.
        ///
        /// A short window keeps the chunk and right context, takes
        /// ``shortWindowLeftContextFrames`` of left context, or what remains of a window
        /// shorter than 8 s, and replays ``shortWindowDecoderHistoryFrames`` of tokens.
        /// These settings were tested with 8 s windows only.
        ///
        /// Fast windows therefore hold 87 frames of audio in the 100 frames of the 8 s encoder.
        /// Filling them with 76 frames of left context lowered the WER of fast mode in 21 of 30
        /// evaluation results and raised it in 8, five of them on German and French FLEURS.
        private func fitted(to windowFrames: Int) -> Configuration {
            guard leftContextFrames + chunkFrames + rightContextFrames > windowFrames else {
                return self
            }
            var configuration = self
            configuration.leftContextFrames = max(
                0,
                min(
                    Self.shortWindowLeftContextFrames,
                    windowFrames - chunkFrames - rightContextFrames))
            configuration.decoderHistoryFrames = Self.shortWindowDecoderHistoryFrames
            configuration.sentenceScopedHistory = false
            return configuration
        }
    }

    let vocabulary: ParakeetVocabulary
    let configuration: Configuration
    private let engine: any ParakeetInferenceEngine
    /// The engine's regular window, which every window but a forced commit uses.
    private let primary: EncoderWindow
    /// The longer window of the engine's forced-commit encoder, if it has one.
    private let forcedCommitWindow: EncoderWindow?
    private var decoder: ParakeetTDTDecoder
    private let frameSamples: Int
    private let windowSamples: Int
    /// Encoder frames whose audio fits completely in one window.
    private let windowFrames: Int

    private var samples: [Float] = []
    private var retainedStart = 0
    private let initialContext: ParakeetDecoderContext
    /// The committed tokens that the decoder context replays: with a sentence-scoped history,
    /// the last committed sentence-final token and later tokens, within the history limit.
    private var history: [(id: Int, frame: Int)] = []
    /// The last context that ``recentContext()`` built: the sentence-final token it started
    /// from, if any, the tokens it predicted after that, and the resulting context.
    private var replayed:
        (base: Int?, tokens: [(id: Int, frame: Int)], context: ParakeetDecoderContext)?
    private var volatile: [ParakeetToken] = []
    private var volatileCoverage = 0
    /// The frame where the duration of the last committed token ends.
    private var lastTokenEnd = 0
    /// The commit frame of the last window whose blank gap another window did not recover.
    /// The next gap is measured from here or from the last token, whichever is later.
    private var recoveryFrame = 0
    /// The most gaps that the recognizer decodes again in one window.
    private static let maximumRecoveries = 4
    /// The lowest mean token probability of a recovered gap.
    ///
    /// In pauses, the initial context sometimes decoded filler that the audio does not
    /// contain, such as "It was a little bit." in German speech, with mean probabilities
    /// of 0.24 to 0.69. Most recovered speech on LibriSpeech and FLEURS had 0.9 or more.
    private static let minimumRecoveryProbability: Float = 0.8
    /// The lowest mean power of a speech-level frame, relative to the mean power of all
    /// received audio: −20 dB. A gap without such a frame is not decoded again.
    ///
    /// The probability threshold does not separate text decoded in a pause from speech:
    /// in a pause at the end of a German FLEURS file, a recovery decoded
    /// "Продолжение следует" ("to be continued") with high probability. Pauses in read and
    /// close-talk recordings are usually 30 dB or more below speech, most 80 ms frames of
    /// speech are within about 15 dB of the speech level, and the mean power of audio that is
    /// at least a tenth speech is within 10 dB of that level. The level comes from all
    /// received audio, because live mode retains only one window. In audio whose noise is
    /// within 20 dB of the mean, every frame has speech-level power.
    ///
    /// The test applies to the whole gap, not to each recovered word. In an AMI meeting,
    /// Parakeet emitted "I" 400 ms before its audio, and the mean power of the first 30 s was
    /// 8 dB above that of the whole meeting. Dropping recovered words without a speech-level
    /// frame within 240 ms dropped nine words in that meeting and added 11 deletions.
    static let speechLevel: Float = 0.01
    /// Frames after the commit frame that the speech-level test also covers, beyond the
    /// longest token duration: 240 ms, more than the 95th percentile of word start errors
    /// against forced alignment, for a word emitted before the commit frame.
    static let speechMarginFrames = 3
    private var ended = false
    /// The sum of squared samples of all received audio.
    private var receivedEnergy: Double = 0

    /// The samples appended so far.
    private(set) var receivedSampleCount = 0
    /// Frames before this encoder frame are committed; the next window resumes decoding here.
    private(set) var committedFrame = 0
    /// The samples the recognizer currently retains.
    var retainedSampleCount: Int { samples.count }
    var committedThroughSample: Int { min(committedFrame * frameSamples, receivedSampleCount) }
    /// New audio that refreshes volatile results before a regular window is ready.
    private var previewSamples: Int {
        min(configuration.chunkFrames, windowFrames) * frameSamples
    }

    init(
        engine: any ParakeetInferenceEngine, vocabulary: ParakeetVocabulary,
        configuration: Configuration
    ) throws {
        let model = engine.model
        frameSamples = model.samplesPerEncoderFrame
        windowSamples = model.windowSamples
        windowFrames = model.windowSamples / model.samplesPerEncoderFrame
        guard configuration.leftContextFrames >= 0, configuration.rightContextFrames >= 0,
            configuration.chunkFrames > configuration.boundaryMarginFrames,
            configuration.boundaryMarginFrames >= 0, configuration.maxSymbolsPerFrame > 0,
            configuration.decoderHistoryFrames >= 0,
            configuration.leftContextFrames + configuration.rightContextFrames < windowFrames
        else {
            throw SpeechRecognitionError.unsupportedConfiguration(
                "Parakeet window context must leave room for a chunk.")
        }
        self.engine = engine
        self.vocabulary = vocabulary
        self.configuration = configuration
        primary = EncoderWindow(model)
        forcedCommitWindow = engine.forcedCommitModel.map(EncoderWindow.init)
        decoder = ParakeetTDTDecoder(
            maxSymbolsPerFrame: configuration.maxSymbolsPerFrame,
            discardsFramesAtSymbolLimit: configuration.discardsFramesAtSymbolLimit)
        initialContext = try .initial(engine: engine)
        for id in 0..<vocabulary.count
        where configuration.sentenceScopedHistory && vocabulary.isSentenceFinal(id) {
            var context = initialContext
            try engine.predict(token: id, state: &context.state, output: &context.output)
            context.lastToken = id
            decoder.sentenceEndContexts[id] = context
        }
    }

    /// Appends 16 kHz mono samples. Appending after the end of input is a programming error.
    func append(_ newSamples: UnsafeBufferPointer<Float>) {
        precondition(!ended, "Cannot append audio after the end of input.")
        samples.append(contentsOf: newSamples)
        receivedSampleCount += newSamples.count
        if let base = newSamples.baseAddress, !newSamples.isEmpty {
            var energy: Float = 0
            vDSP_svesq(base, 1, &energy, vDSP_Length(newSamples.count))
            receivedEnergy += Double(energy)
        }
    }

    func append(_ newSamples: [Float]) {
        newSamples.withUnsafeBufferPointer { append($0) }
    }

    /// Processes every chunk whose right context is available.
    ///
    /// - Parameters:
    ///   - finalizeThrough: A sample through which to commit even without right context.
    ///     The recognizer commits whole encoder frames, through at most the received audio.
    ///   - endOfInput: Whether to commit all received audio. Later appends are not allowed.
    func process(finalizeThrough: Int? = nil, endOfInput: Bool = false) throws -> ParakeetUpdate {
        if endOfInput { ended = true }
        let available = receivedSampleCount
        let availableFrames = (available + frameSamples - 1) / frameSamples
        let target =
            ended
            ? availableFrames
            : finalizeThrough.map {
                min((max(0, $0) + frameSamples - 1) / frameSamples, availableFrames)
            }
        var update = ParakeetUpdate(committedThroughSample: 0)
        var lookahead: (tokens: [ParakeetToken], coverage: Int)?
        while committedFrame < availableFrames {
            let preferred = quietestStart(
                default: max(retainedFrame, committedFrame - configuration.leftContextFrames))
            let regular = min(
                preferred + windowFrames - configuration.rightContextFrames,
                committedFrame + min(configuration.chunkFrames, windowFrames))
            let result: (committed: [ParakeetToken], lookahead: [ParakeetToken], coverage: Int)
            if (regular + configuration.rightContextFrames) * frameSamples <= available {
                result = try runWindow(
                    start: preferred, commit: regular,
                    margin: configuration.boundaryMarginFrames)
            } else if let target, target > committedFrame {
                result = try runWindow(
                    start: windowStart(
                        availableFrames: availableFrames,
                        frames: forcedCommitWindow?.frames ?? windowFrames),
                    commit: target, margin: 0, forced: forcedCommitWindow != nil)
            } else {
                break
            }
            update.committed += result.committed
            lookahead = (result.lookahead, result.coverage)
        }
        if !configuration.reportsVolatileResults || ended
            || committedFrame * frameSamples >= available
        {
            volatile = []
            volatileCoverage = available
        } else if let lookahead, available - lookahead.coverage < previewSamples {
            // The window ends before the newest audio when its end falls inside an input
            // buffer. A preview for that much audio costs another encoder call.
            volatile = lookahead.tokens
            volatileCoverage = lookahead.coverage
        } else if lookahead != nil || available - volatileCoverage >= previewSamples {
            // A commit makes earlier volatile tokens stale, and new audio deserves a preview.
            let start = windowStart(availableFrames: availableFrames)
            let preview = try runWindow(start: start, commit: committedFrame, margin: nil)
            volatile = preview.lookahead
            volatileCoverage = preview.coverage
        }
        update.volatile = volatile
        update.committedThroughSample = committedThroughSample
        return update
    }

    /// The first retained encoder frame.
    private var retainedFrame: Int { (retainedStart + frameSamples - 1) / frameSamples }

    /// The start frame of a regular window: the frame whose first 80 ms have the least energy
    /// within ``Configuration/quietStartFrames`` before the commit frame, or `default`.
    private func quietestStart(default preferred: Int) -> Int {
        guard let range = configuration.quietStartFrames else { return preferred }
        // The window must still commit frames after the boundary margin.
        let lower = max(
            retainedFrame, committedFrame - range.upperBound,
            committedFrame + configuration.rightContextFrames
                + configuration.boundaryMarginFrames + 1 - windowFrames)
        let upper = committedFrame - range.lowerBound
        guard lower < upper else { return preferred }
        var best = preferred
        var bestEnergy = Float.infinity
        samples.withUnsafeBufferPointer { samples in
            for frame in lower...upper {
                let offset = frame * frameSamples - retainedStart
                guard offset >= 0, offset + frameSamples <= samples.count else { continue }
                var energy: Float = 0
                vDSP_svesq(
                    samples.baseAddress! + offset, 1, &energy, vDSP_Length(frameSamples))
                // Prefer the default start when frames are equally quiet.
                if energy < bestEnergy
                    || (energy == bestEnergy && abs(frame - preferred) < abs(best - preferred))
                {
                    best = frame
                    bestEnergy = energy
                }
            }
        }
        return best
    }

    /// The first frame of a window that ends with the audio: left context before the commit
    /// frame, or more when the audio ends before a window placed there would.
    ///
    /// - Parameter frames: The window's whole encoder frames; the regular window by default.
    private func windowStart(availableFrames: Int, frames: Int? = nil) -> Int {
        let preferred = committedFrame - configuration.leftContextFrames
        return max(0, retainedFrame, min(preferred, availableFrames - (frames ?? windowFrames)))
    }

    /// Encodes and decodes one window.
    ///
    /// - Parameters:
    ///   - commit: The frame before which tokens are committed.
    ///   - margin: The boundary margin, or `nil` for a preview that commits nothing.
    ///   - forced: Whether to use the forced-commit window.
    private func runWindow(start: Int, commit: Int, margin: Int?, forced: Bool = false) throws
        -> (committed: [ParakeetToken], lookahead: [ParakeetToken], coverage: Int)
    {
        let (encoded, count, decodable) = try encode(
            start: start, forced: forced, reusesMelFrames: configuration.reusesMelFrames)
        let startSample = start * frameSamples
        let localCommit = min(commit - start, decodable)
        let commits = margin != nil
        guard !commits || start + localCommit > committedFrame else {
            throw SpeechRecognitionError.invalidResult(
                "The Parakeet encoder reported too few valid frames.")
        }
        var decoding = try decoder.decodeWindow(
            encoded, from: committedFrame - start, commit: localCommit,
            end: configuration.reportsVolatileResults || !commits ? decodable : localCommit,
            context: try recentContext(), engine: engine, frameOffset: start,
            previousTokenEnd: lastTokenEnd - start)
        if commits, let gapFrames = configuration.blankRecoveryFrames {
            var after = max(lastTokenEnd, recoveryFrame)
            for _ in 0..<Self.maximumRecoveries {
                guard
                    let recovered = try recoverBlankGap(
                        in: decoding, commit: start + localCommit, forced: forced, after: after,
                        gapFrames: gapFrames)
                else { break }
                decoding = recovered.decoding
                after = recovered.next
            }
        }
        let committed = decoding.committed.map(token)
        if let margin {
            let afterLastToken = decoding.committed.last.map { $0.frame + 1 } ?? committedFrame
            committedFrame = max(afterLastToken, start + localCommit - margin)
            if let last = decoding.committed.last {
                lastTokenEnd = last.frame + max(1, last.duration)
            }
            history += decoding.committed.map { ($0.id, $0.frame) }
            if configuration.sentenceScopedHistory,
                let end = history.lastIndex(where: { vocabulary.isSentenceFinal($0.id) })
            {
                history.removeFirst(end)
            }
            history.removeFirst(
                history.prefix { $0.frame < committedFrame - configuration.decoderHistoryFrames }
                    .count)
            discardAudio()
        }
        return (committed, decoding.lookahead.map(token), startSample + count)
    }

    /// Computes the features of the window that starts at encoder frame `start` and encodes it.
    ///
    /// - Returns: The encoder output, the window's audio samples, and its decodable frames.
    private func encode(start: Int, forced: Bool, reusesMelFrames: Bool) throws
        -> (encoded: ParakeetEncoderOutput, count: Int, decodable: Int)
    {
        let window = forced ? forcedCommitWindow! : primary
        let startSample = start * frameSamples
        let count = min(window.samples, receivedSampleCount - startSample)
        precondition(startSample >= retainedStart && count > 0)
        // Zero padding to a whole encoder frame counts as audio, as FluidAudio declares it.
        let validSamples = min(
            window.samples, (count + frameSamples - 1) / frameSamples * frameSamples)
        let offset = startSample - retainedStart
        let melFrames = samples.withUnsafeBufferPointer { samples in
            window.mel.withUnsafeMutableBufferPointer { mel in
                window.frontend.compute(
                    UnsafeBufferPointer(rebasing: samples[offset..<(offset + count)]),
                    validSampleCount: validSamples,
                    startHop: reusesMelFrames && !forced
                        ? startSample / ParakeetMelFrontend.hopLength : nil,
                    into: mel)
            }
        }
        try window.mel.withUnsafeBufferPointer {
            if forced {
                try engine.encodeForcedCommit(
                    mel: $0, validFrameCount: melFrames, into: &window.encoded)
            } else {
                try engine.encode(mel: $0, validFrameCount: melFrames, into: &window.encoded)
            }
        }
        let encoded = window.encoded
        let decodable = min(encoded.validFrameCount, (count + frameSamples - 1) / frameSamples)
        return (encoded, count, decodable)
    }

    /// Decodes the first blank gap of a window's committed tokens again from a window that
    /// starts at the gap.
    ///
    /// - Parameters:
    ///   - decoding: The window's decoding.
    ///   - commit: The window's commit frame.
    ///   - after: The frame from which gaps are measured; earlier tokens do not end a gap.
    ///   - gapFrames: The shortest gap at the end of the window to decode again.
    /// - Returns: The decoding with the tokens of the new window from the gap on, and the frame
    ///   from which to look for the next gap; or `nil` when the window has no gap, the gap
    ///   has no speech-level audio, or the new window does not find more tokens with enough
    ///   probability.
    private func recoverBlankGap(
        in decoding: ParakeetWindowDecoding, commit: Int, forced: Bool, after: Int,
        gapFrames: Int
    ) throws -> (decoding: ParakeetWindowDecoding, next: Int)? {
        var anchor = after
        var previousEnd = lastTokenEnd
        var previous = history.last
        var gap: (start: Int, kept: Int)?
        for (index, token) in decoding.committed.enumerated() {
            // A gap that the first decoding ended with a token is usually a pause, so it must
            // be twice as long. On German FLEURS, recovering a shorter one split a compound.
            if token.frame >= after, token.frame - anchor >= 2 * gapFrames {
                gap = (anchor, index)
                break
            }
            let end = token.frame + max(1, token.duration)
            if token.frame >= after { anchor = end }
            previousEnd = end
            previous = (token.id, token.frame)
        }
        if gap == nil, commit - anchor >= gapFrames {
            gap = (anchor, decoding.committed.count)
        }
        guard let gap else { return nil }
        let from = max(gap.start, committedFrame)
        let longestDuration = max(1, engine.model.durations.max() ?? 4)
        // A pause has nothing to recover; the initial context can decode text there.
        guard
            hasSpeech(
                in: from..<(commit - 1 + longestDuration + Self.speechMarginFrames))
        else {
            recoveryFrame = commit
            return nil
        }
        // The window starts at the gap. It still reaches its right context past the commit
        // frame, and it holds at least that much audio, because a window of a fraction of a
        // second decoded "Yeah." at the end of a file.
        let window = forced ? forcedCommitWindow! : primary
        let availableFrames = (receivedSampleCount + frameSamples - 1) / frameSamples
        let reach = forced ? 0 : configuration.rightContextFrames
        let start = min(
            from,
            max(
                retainedFrame, commit + reach - window.frames,
                min(gap.start, availableFrames - configuration.rightContextFrames)))
        let (encoded, _, decodable) = try encode(
            start: start, forced: forced, reusesMelFrames: false)
        let localCommit = min(commit - start, decodable)
        var recovered =
            localCommit > from - start
            ? try decoder.decodeWindow(
                encoded, from: from - start, commit: localCommit,
                end: configuration.reportsVolatileResults ? decodable : localCommit,
                context: initialContext, engine: engine, frameOffset: start,
                previousTokenEnd: previousEnd - start) : nil
        // A window that starts inside the audio of the token before the gap can emit it again.
        if let first = recovered?.committed.first, let previous, first.id == previous.id,
            first.frame - previous.frame <= engine.model.durations.max() ?? 4
        {
            recovered?.committed.removeFirst()
        }
        guard var recovered, recovered.committed.count > decoding.committed.count - gap.kept,
            recovered.committed.map(\.probability).reduce(0, +)
                >= Self.minimumRecoveryProbability * Float(recovered.committed.count)
        else {
            recoveryFrame = commit
            return nil
        }
        let next = recovered.committed[0].frame
        recovered.committed = Array(decoding.committed.prefix(gap.kept)) + recovered.committed
        return (recovered, next)
    }

    /// Whether an encoder frame in `frames` has speech-level power: more than
    /// ``speechLevel`` times the mean power of all received audio.
    /// Frames that the recognizer does not retain do not count.
    private func hasSpeech(in frames: Range<Int>) -> Bool {
        guard receivedSampleCount > 0, !frames.isEmpty else { return false }
        let threshold = Self.speechLevel * Float(receivedEnergy / Double(receivedSampleCount))
        return samples.withUnsafeBufferPointer { samples in
            frames.contains { frame in
                let offset = frame * frameSamples - retainedStart
                guard offset >= 0, offset < samples.count else { return false }
                let count = min(frameSamples, samples.count - offset)
                var energy: Float = 0
                vDSP_svesq(samples.baseAddress! + offset, 1, &energy, vDSP_Length(count))
                return energy > threshold * Float(count)
            }
        }
    }

    /// The prediction network context after the committed tokens of ``history``.
    ///
    /// Consecutive windows usually replay the same tokens plus newly committed ones,
    /// so the context after the last replay is kept and only the new tokens are predicted.
    /// The predictions run in the same order from the same state, so the context is the one
    /// that a full replay produces.
    private func recentContext() throws -> ParakeetDecoderContext {
        var tokens = history[...]
        var base: Int?
        var context = initialContext
        if let first = tokens.first, let sentenceEnd = decoder.sentenceEndContexts[first.id] {
            base = first.id
            context = sentenceEnd
            tokens = tokens.dropFirst()
        } else if configuration.sentenceScopedHistory {
            // Without a sentence end in range, start at a word rather than inside one.
            tokens = tokens.drop {
                !ParakeetVocabulary.startsWord(vocabulary.piece(for: $0.id))
            }
        }
        if let replayed, replayed.base == base, replayed.tokens.count <= tokens.count,
            zip(replayed.tokens, tokens).allSatisfy({ $0.id == $1.id && $0.frame == $1.frame })
        {
            context = replayed.context
            tokens = tokens.dropFirst(replayed.tokens.count)
        } else {
            replayed = (base, [], context)
        }
        for token in tokens {
            try engine.predict(token: token.id, state: &context.state, output: &context.output)
            context.lastToken = token.id
        }
        replayed?.tokens += tokens
        replayed?.context = context
        return context
    }

    private func token(_ decoded: ParakeetDecodedToken) -> ParakeetToken {
        let start = decoded.frame * frameSamples
        let end = min(
            (decoded.frame + max(1, decoded.duration)) * frameSamples, receivedSampleCount)
        return ParakeetToken(
            id: decoded.id, piece: vocabulary.piece(for: decoded.id), startSample: start,
            endSample: max(end, start + 1), probability: decoded.probability)
    }

    /// Discards audio more than one window, the longest one, before the commit frame.
    private func discardAudio() {
        let frames = max(windowFrames, forcedCommitWindow?.frames ?? 0)
        let keep = max(0, (committedFrame - frames) * frameSamples)
        let count = min(keep - retainedStart, samples.count)
        guard count > 0 else { return }
        samples.removeFirst(count)
        retainedStart += count
    }
}

/// The buffers for one encoder window size.
private final class EncoderWindow {
    /// The audio samples in the window.
    let samples: Int
    /// Encoder frames whose audio fits completely in the window.
    let frames: Int
    let frontend: ParakeetMelFrontend
    var mel: [Float]
    var encoded: ParakeetEncoderOutput

    init(_ model: ParakeetModelDescription) {
        samples = model.windowSamples
        frames = model.windowFrames
        frontend = ParakeetMelFrontend(windowSamples: model.windowSamples)
        mel = [Float](repeating: 0, count: model.melBinCount * model.melFrameCount)
        encoded = ParakeetEncoderOutput(
            hiddenSize: model.encoderHiddenSize, frameCount: model.encoderFrameCount)
    }
}
