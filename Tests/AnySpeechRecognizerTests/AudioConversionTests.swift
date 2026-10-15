import AVFoundation
import CoreMedia
import Foundation
import Testing

@testable import AnySpeechRecognizer

private func convert(_ buffers: [TimedAudioBuffer], to format: AVAudioFormat) throws
    -> [TimedAudioBuffer]
{
    let converter = StreamConverter(output: format)
    var output: [TimedAudioBuffer] = []
    for buffer in buffers { output += try converter.convert(buffer) }
    return output + (try converter.flush())
}

private func blocks(
    frames: Int, sampleRate: Double, channels: AVAudioChannelCount, block: Int = 4096,
    value: Float = 0.25, start: Int64 = 0
) -> [TimedAudioBuffer] {
    stride(from: 0, to: frames, by: block).map { offset in
        TimedAudioBuffer(
            buffer: pcmBuffer(
                frames: min(block, frames - offset), sampleRate: sampleRate, channels: channels,
                value: value),
            startTime: CMTime(value: start + Int64(offset), timescale: CMTimeScale(sampleRate)))
    }
}

@Test(arguments: [16_000.0, 44_100.0, 48_000.0]) func conversionPreservesDuration(
    sampleRate: Double
) throws {
    let frames = Int(sampleRate) * 2 + 123
    let channels: AVAudioChannelCount = sampleRate == 16_000 ? 1 : 2
    let output = try convert(
        blocks(frames: frames, sampleRate: sampleRate, channels: channels),
        to: AudioFormats.monoFloat16k)
    let total = output.reduce(0) { $0 + Int($1.buffer.frameLength) }
    #expect(output.allSatisfy { $0.buffer.format.sampleRate == 16_000 })
    #expect(output.allSatisfy { $0.buffer.format.channelCount == 1 })
    #expect(abs(Double(total) - Double(frames) * 16_000 / sampleRate) <= 1)
    #expect(zip(output, output.dropFirst()).allSatisfy { $0.endTime == $1.startTime })
}

@Test func stereoDownmixAveragesChannels() throws {
    let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 2, interleaved: false)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1024)!
    buffer.frameLength = 1024
    for frame in 0..<1024 {
        buffer.floatChannelData![0][frame] = 0.75
        buffer.floatChannelData![1][frame] = -0.25
    }
    let output = try convert(
        [.init(buffer: buffer, startTime: .zero)], to: AudioFormats.monoFloat16k)
    let samples = try #require(output.first?.buffer.floatChannelData?[0])
    #expect(output.count == 1)
    #expect((0..<1024).allSatisfy { abs(samples[$0] - 0.25) < 0.00001 })
}

@Test func conversionKeepsSourceOffset() throws {
    let output = try convert(
        blocks(frames: 44_100, sampleRate: 44_100, channels: 2, start: 44_100),
        to: AudioFormats.monoFloat16k)
    #expect(output.first?.startTime == CMTime(value: 1, timescale: 1))
}

@Test func gapsStartNewSegments() throws {
    let first = blocks(frames: 4_410, sampleRate: 44_100, channels: 1)
    let second = blocks(frames: 4_410, sampleRate: 44_100, channels: 1, start: 44_100)
    let output = try convert(first + second, to: AudioFormats.monoFloat16k)
    let laterStart = try #require(output.first { $0.startTime >= CMTime(value: 1, timescale: 1) })
    #expect(laterStart.startTime == CMTime(value: 1, timescale: 1))
}

@Test func matchingFormatPassesThrough() throws {
    let input = blocks(frames: 8_000, sampleRate: 16_000, channels: 1)
    let output = try convert(input, to: AudioFormats.monoFloat16k)
    #expect(output.map(\.buffer) == input.map(\.buffer))
}

@Test func malformedAudioCannotBeOpened() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "anyspeech-bad-\(UUID()).wav")
    try Data("invalid audio".utf8).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(throws: (any Error).self) { try AVAudioFile(forReading: url) }
}
