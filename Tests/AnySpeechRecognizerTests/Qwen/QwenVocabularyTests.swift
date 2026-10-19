import Foundation
import Testing

@testable import AnySpeechRecognizer

struct QwenVocabularyTests {
    let vocabulary = QwenTestSupport.vocabulary

    /// `Ġ` is a space, and skipped special tokens have no text.
    @Test func decodesByteLevelTokens() {
        typealias T = QwenTestSupport
        #expect(vocabulary.text(for: [T.hello, T.world]) == "Hello world")
        #expect(vocabulary.text(for: T.englishReply) == "language English<asr_text>Hello world")
        #expect(vocabulary.text(for: [T.prefix, T.hello, T.endOfText, T.imEnd]) == "Hello")
        #expect(vocabulary.text(for: [T.ni, T.hao]) == "你好")
    }

    /// A character's UTF-8 bytes can span tokens; text decodes from the joined bytes.
    @Test func joinsBytesAcrossTokens() {
        let split = QwenVocabulary(strings: ["ä½", "ł", "å", "¥½"], skipping: [])
        #expect(split.text(for: [0, 1, 2, 3]) == "你好")
        #expect(split.text(for: [0]) == "\u{FFFD}")
    }

    /// GPT-2's mapping covers all 256 bytes with distinct characters.
    @Test func byteDecoderCoversEveryByte() {
        let decoder = QwenVocabulary.byteDecoder
        #expect(decoder.count == 256)
        #expect(Set(decoder.values).count == 256)
        #expect(decoder["Ġ"] == 0x20)
        #expect(decoder["Ċ"] == 0x0A)
    }

    @Test func parsesLanguageAndTranscript() {
        let english = QwenOutput.parse("language English<asr_text>Hello world.")
        #expect(english == ("English", "Hello world."))
        #expect(QwenOutput.parse("language None<asr_text>") == (nil, ""))
        #expect(QwenOutput.parse("  Hello world.  ") == (nil, "Hello world."))
        let korean = QwenOutput.parse("system\nassistant\nlanguage Korean<asr_text>안녕")
        #expect(korean == ("Korean", "안녕"))
        #expect(QwenOutput.parse("") == (nil, ""))
    }

    /// The original library's repetition fix: a character repeated more than 20 times
    /// becomes one, and a pattern repeated 20 or more times becomes one instance.
    @Test func fixesRepetitions() {
        func repeated(_ text: String, _ count: Int) -> String {
            String(repeating: text, count: count)
        }
        #expect(QwenOutput.fixRepetitions(repeated("a", 25) + "b") == "ab")
        #expect(QwenOutput.fixRepetitions(repeated("a", 20) + "b") == repeated("a", 20) + "b")
        #expect(QwenOutput.fixRepetitions(repeated("ab", 25) + "c") == "abc")
        #expect(QwenOutput.fixRepetitions("x" + repeated("哈哈，", 21) + "好") == "x哈哈，好")
        let sentence = "The quick brown fox jumps over the lazy dog, and the dog sleeps on."
        #expect(QwenOutput.fixRepetitions(sentence) == sentence)
    }
}
