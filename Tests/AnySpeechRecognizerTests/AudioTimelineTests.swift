import CoreMedia
import Testing

@testable import AnySpeechRecognizer

@Test func sourceFrameTimeIsExact() throws {
    let time = try AudioTimeline.time(frame: 441_001, sampleRate: 44_100)
    #expect(time.value == 441_001)
    #expect(time.timescale == 44_100)
}
