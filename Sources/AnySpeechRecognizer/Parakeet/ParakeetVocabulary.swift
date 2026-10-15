import Foundation

/// Maps Parakeet token IDs to SentencePiece pieces and joins them into text.
///
/// A piece that starts with `▁` (U+2581) or a space begins a word.
/// Text keeps the model's spelling, punctuation, and spacing; a piece without a
/// boundary, such as `.` or a word continuation, joins the preceding text.
/// Control pieces, such as `<unk>` and `<|nospeech|>`, produce no text.
struct ParakeetVocabulary: Sendable {
    /// A word and the indices of its tokens in the sequence that produced it.
    struct Word: Sendable, Equatable {
        var text: String
        var tokens: Range<Int>
    }

    static let wordBoundary: Character = "\u{2581}"

    /// The pieces, indexed by token ID.
    let pieces: [String]

    init(pieces: [String]) { self.pieces = pieces }

    /// Loads `parakeet_vocab.json`, an object that maps decimal IDs to pieces, or an array of pieces.
    init(contentsOf url: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw SpeechRecognitionError.missingAssets("Missing Parakeet vocabulary \(url.path).")
        }
        let json = try? JSONSerialization.jsonObject(with: data)
        if let array = json as? [String] {
            pieces = array
            return
        }
        guard let object = json as? [String: String] else {
            throw SpeechRecognitionError.missingAssets("The Parakeet vocabulary is not valid JSON.")
        }
        var pieces = [String](repeating: "", count: object.count)
        for (key, piece) in object {
            guard let id = Int(key), pieces.indices.contains(id) else {
                throw SpeechRecognitionError.missingAssets(
                    "The Parakeet vocabulary IDs are not contiguous.")
            }
            pieces[id] = piece
        }
        self.pieces = pieces
    }

    var count: Int { pieces.count }

    /// The piece for a token, or an empty string for the blank token and unknown IDs.
    func piece(for id: Int) -> String {
        pieces.indices.contains(id) ? pieces[id] : ""
    }

    /// The text of a token sequence, without leading or trailing whitespace.
    func text(for ids: some Sequence<Int>) -> String {
        Self.text(pieces: ids.map(piece(for:)))
    }

    /// Groups tokens into words; each word starts at a token with a word boundary.
    func words(for ids: [Int]) -> [Word] {
        Self.words(pieces: ids.map(piece(for:)))
    }

    static func text(pieces: some Sequence<String>) -> String {
        var text = ""
        for piece in pieces where !isControl(piece) {
            text += piece.replacingOccurrences(of: String(wordBoundary), with: " ")
        }
        return text.trimmingCharacters(in: .whitespaces)
    }

    static func words(pieces: [String]) -> [Word] {
        var words: [Word] = []
        var start: Int?
        func close(at end: Int) {
            guard let first = start else { return }
            let text = Self.text(pieces: pieces[first..<end])
            if !text.isEmpty { words.append(Word(text: text, tokens: first..<end)) }
            start = nil
        }
        for (index, piece) in pieces.enumerated() where !isControl(piece) {
            if startsWord(piece) { close(at: index) }
            if start == nil { start = index }
        }
        close(at: pieces.count)
        return words
    }

    /// Whether a token ends a sentence: `.`, `?`, `!`, or a full-width form, with or without
    /// a word boundary.
    func isSentenceFinal(_ id: Int) -> Bool {
        var piece = Substring(piece(for: id))
        if piece.first == Self.wordBoundary || piece.first == " " { piece = piece.dropFirst() }
        return [".", "?", "!", "\u{3002}", "\u{FF1F}", "\u{FF01}"].contains(piece)
    }

    static func startsWord(_ piece: String) -> Bool {
        piece.first == wordBoundary || piece.first == " "
    }

    static func isControl(_ piece: String) -> Bool {
        piece.isEmpty || (piece.count > 2 && piece.hasPrefix("<") && piece.hasSuffix(">"))
    }
}
