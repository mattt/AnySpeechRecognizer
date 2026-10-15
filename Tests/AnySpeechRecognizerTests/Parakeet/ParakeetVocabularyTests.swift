import Foundation
import Testing

@testable import AnySpeechRecognizer

struct ParakeetVocabularyTests {
    let vocabulary = ParakeetVocabulary(pieces: [
        "<unk>", "\u{2581}The", "\u{2581}qu", "ick", "\u{2581}fox", ".", ",", " jumps",
        "<|nospeech|>",
        "\u{2581}42", "%",
    ])

    @Test func joinsPiecesAtWordBoundaries() {
        #expect(vocabulary.text(for: [1, 2, 3, 4, 5]) == "The quick fox.")
        #expect(vocabulary.text(for: [4, 6, 7, 5]) == "fox, jumps.")
        #expect(vocabulary.text(for: [9, 10]) == "42%")
        #expect(vocabulary.text(for: []) == "")
    }

    @Test func omitsControlPiecesAndUnknownIDs() {
        #expect(vocabulary.text(for: [0, 1, 8, 4, 11, 8192]) == "The fox")
        #expect(vocabulary.piece(for: 8192) == "")
        #expect(vocabulary.piece(for: -1) == "")
    }

    @Test func groupsMultiTokenWordsWithTheirTokenIndices() {
        let words = vocabulary.words(for: [1, 2, 3, 6, 4, 5])
        #expect(
            words == [
                .init(text: "The", tokens: 0..<1), .init(text: "quick,", tokens: 1..<4),
                .init(text: "fox.", tokens: 4..<6),
            ])
        // A sequence can start inside a word; its first piece starts a word.
        #expect(vocabulary.words(for: [3, 4]).map(\.text) == ["ick", "fox"])
        // Control pieces belong to no word and do not start one.
        #expect(vocabulary.words(for: [8, 1, 8, 3]).map(\.text) == ["Theick"])
        #expect(vocabulary.words(for: [8]).isEmpty)
    }

    @Test func recognizesSentenceFinalTokens() {
        let vocabulary = ParakeetVocabulary(pieces: [
            ".", "\u{2581}?", " !", "\u{3002}", ",", "a.", "\u{2581}Mr", "...",
        ])
        #expect((0..<9).filter(vocabulary.isSentenceFinal) == [0, 1, 2, 3])
    }

    @Test func keepsCombiningMarksAndScriptsInsideWords() {
        let pieces = [" e", "\u{301}", "lan", ",", " 世界", "。"]
        #expect(ParakeetVocabulary.text(pieces: pieces) == "e\u{301}lan, 世界。")
        #expect(ParakeetVocabulary.words(pieces: pieces).map(\.text) == ["e\u{301}lan,", "世界。"])
    }

    @Test func loadsObjectAndArrayFiles() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "parakeet-vocabulary-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let object = directory.appendingPathComponent("object.json")
        try Data(#"{"1": " world", "0": "▁hello"}"#.utf8).write(to: object)
        #expect(try ParakeetVocabulary(contentsOf: object).text(for: [0, 1]) == "hello world")
        let array = directory.appendingPathComponent("array.json")
        try Data(#"["▁a", "b"]"#.utf8).write(to: array)
        #expect(try ParakeetVocabulary(contentsOf: array).text(for: [0, 1]) == "ab")
        let gap = directory.appendingPathComponent("gap.json")
        try Data(#"{"0": "a", "2": "b"}"#.utf8).write(to: gap)
        #expect(throws: SpeechRecognitionError.self) { try ParakeetVocabulary(contentsOf: gap) }
        #expect(throws: SpeechRecognitionError.self) {
            try ParakeetVocabulary(contentsOf: directory.appendingPathComponent("missing.json"))
        }
    }
}
