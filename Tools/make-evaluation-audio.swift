import AVFoundation
import Foundation

// Make a local stress fixture from short audio that the caller may use.
// This script does not download audio or redistribute the caller's recordings.
guard CommandLine.arguments.count == 5,
    let seconds = Double(CommandLine.arguments[3]), seconds.isFinite, seconds > 0
else {
    fatalError(
        "Usage: xcrun swift make-evaluation-audio.swift <seed-audio> <output.wav> <seconds> <ending-audio>"
    )
}
let seed = try AVAudioFile(
    forReading: URL(fileURLWithPath: CommandLine.arguments[1]), commonFormat: .pcmFormatFloat32,
    interleaved: false)
let ending = try AVAudioFile(
    forReading: URL(fileURLWithPath: CommandLine.arguments[4]), commonFormat: .pcmFormatFloat32,
    interleaved: false)
let format = seed.processingFormat
guard seed.length > 0, format == ending.processingFormat,
    seconds * format.sampleRate < Double(Int64.max)
else {
    fatalError("Use nonempty seed audio and ending audio with the same decoded format.")
}
let totalFrames = Int64((seconds * format.sampleRate).rounded())
let leadingFrames = Int64(format.sampleRate * 2)
guard totalFrames > leadingFrames + ending.length else {
    fatalError("The duration must contain the leading silence and ending clip.")
}
var settings = format.settings
settings[AVLinearPCMIsNonInterleaved] = false
var output: AVAudioFile? = try AVAudioFile(
    forWriting: URL(fileURLWithPath: CommandLine.arguments[2]), settings: settings)
let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 8192)!
var written: Int64 = 0
while written < leadingFrames {
    buffer.frameLength = AVAudioFrameCount(min(8192, leadingFrames - written))
    for channel in 0..<Int(format.channelCount) {
        for frame in 0..<Int(buffer.frameLength) { buffer.floatChannelData![channel][frame] = 0 }
    }
    try output!.write(from: buffer)
    written += Int64(buffer.frameLength)
}
let bodyEnd = totalFrames - ending.length
while written < bodyEnd {
    if seed.framePosition == seed.length { seed.framePosition = 0 }
    let count = AVAudioFrameCount(min(8192, bodyEnd - written))
    try seed.read(into: buffer, frameCount: count)
    try output!.write(from: buffer)
    written += Int64(buffer.frameLength)
}
while ending.framePosition < ending.length {
    try ending.read(into: buffer, frameCount: 8192)
    try output!.write(from: buffer)
    written += Int64(buffer.frameLength)
}
// Release the writer before exit so AVAudioFile writes the final WAV header.
output = nil
print(
    "Wrote \(written) frames at \(format.sampleRate) Hz (\(Double(written) / format.sampleRate) seconds)."
)
