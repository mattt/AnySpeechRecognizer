import AVFoundation
import Foundation

// Make a synthetic multi-speaker recording with macOS `say` voices from a plan,
// with reference speaker segments (RTTM) and a reference transcript.
//
// Each plan line is `voice<TAB>offset<TAB>text`, or `silence<TAB>seconds`.
// The offset is the time in seconds from the end of the previous utterance's speech
// to the start of this one; a negative offset overlaps the previous utterance.
// Lines that start with `#` are comments.
// Each utterance is synthesized alone and trimmed to its speech, so speaker identity in the
// reference is exact. Speech regions come from a 10 ms energy detector that bridges pauses
// shorter than 0.3 s, so reference boundaries are estimates within a few frames.
// Synthetic voices are an easy case for diarization; do not treat results as real-world accuracy.
guard CommandLine.arguments.count == 3 else {
    fatalError("Usage: xcrun swift make-diarization-audio.swift <plan.tsv> <output-prefix>")
}
let planURL = URL(fileURLWithPath: CommandLine.arguments[1])
let prefix = CommandLine.arguments[2]
let sampleRate = 16_000.0
let format = AVAudioFormat(
    commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
let temporary = FileManager.default.temporaryDirectory.appending(
    path: "diarization-\(UUID().uuidString)", directoryHint: .isDirectory)
try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: temporary) }

/// Synthesizes text with a voice as 16 kHz mono samples.
func synthesize(_ text: String, voice: String) throws -> [Float] {
    let url = temporary.appending(path: "\(UUID().uuidString).wav")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
    process.arguments = ["-v", voice, "-o", url.path, "--data-format=LEF32@16000", text]
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { fatalError("say failed for voice \(voice).") }
    let file = try AVAudioFile(
        forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
    let buffer = AVAudioPCMBuffer(
        pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: buffer)
    precondition(file.processingFormat.sampleRate == sampleRate)
    return Array(
        UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
}

/// Speech regions in samples: 10 ms frames above -40 dBFS RMS, bridging gaps under 0.3 s.
func speechRegions(_ samples: [Float]) -> [Range<Int>] {
    let frame = 160
    var regions: [Range<Int>] = []
    var index = 0
    while index < samples.count {
        let end = min(index + frame, samples.count)
        var energy: Float = 0
        for sample in samples[index..<end] { energy += sample * sample }
        let rms = (energy / Float(end - index)).squareRoot()
        if rms > 0.01 {
            if let last = regions.last, index - last.upperBound < Int(0.3 * sampleRate) {
                regions[regions.count - 1] = last.lowerBound..<end
            } else {
                regions.append(index..<end)
            }
        }
        index = end
    }
    return regions
}

struct Utterance {
    let voice: String
    let text: String
    let start: Int
    let samples: [Float]
    let regions: [Range<Int>]
}

var utterances: [Utterance] = []
var cursor = Int(0.5 * sampleRate)
for line in try String(contentsOf: planURL, encoding: .utf8).split(whereSeparator: \.isNewline) {
    let fields = line.split(separator: "\t", maxSplits: 2).map(String.init)
    guard let first = fields.first, !first.hasPrefix("#") else { continue }
    if first == "silence" {
        cursor += Int(Double(fields[1])! * sampleRate)
        continue
    }
    precondition(fields.count == 3, "Malformed plan line: \(line)")
    let raw = try synthesize(fields[2], voice: first)
    let regions = speechRegions(raw)
    guard let firstRegion = regions.first, let lastRegion = regions.last else { continue }
    let trimmed = Array(raw[firstRegion.lowerBound..<lastRegion.upperBound])
    let start = max(0, cursor + Int(Double(fields[1])! * sampleRate))
    let shifted = regions.map {
        ($0.lowerBound - firstRegion.lowerBound)..<($0.upperBound - firstRegion.lowerBound)
    }
    utterances.append(
        Utterance(voice: first, text: fields[2], start: start, samples: trimmed, regions: shifted))
    cursor = max(cursor, start + trimmed.count)
}
let total = cursor + Int(0.5 * sampleRate)
var mix = [Float](repeating: 0, count: total)
for utterance in utterances {
    for (offset, sample) in utterance.samples.enumerated() {
        mix[utterance.start + offset] += sample
    }
}
let peak = mix.map(abs).max() ?? 1
if peak > 0.99 { mix = mix.map { $0 * 0.99 / peak } }

var settings = format.settings
settings[AVFormatIDKey] = kAudioFormatLinearPCM
settings[AVLinearPCMBitDepthKey] = 16
settings[AVLinearPCMIsFloatKey] = false
settings[AVLinearPCMIsNonInterleaved] = false
var output: AVAudioFile? = try AVAudioFile(
    forWriting: URL(fileURLWithPath: prefix + ".wav"), settings: settings,
    commonFormat: .pcmFormatFloat32, interleaved: false)
let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(total))!
buffer.frameLength = AVAudioFrameCount(total)
mix.withUnsafeBufferPointer {
    buffer.floatChannelData![0].update(from: $0.baseAddress!, count: total)
}
try output!.write(from: buffer)
output = nil

let name = URL(fileURLWithPath: prefix).lastPathComponent
var rttm = ""
for utterance in utterances {
    for region in utterance.regions {
        let start = Double(utterance.start + region.lowerBound) / sampleRate
        let duration = Double(region.count) / sampleRate
        rttm += String(
            format: "SPEAKER %@ 1 %.3f %.3f <NA> <NA> %@ <NA> <NA>\n", name, start, duration,
            utterance.voice)
    }
}
try rttm.write(toFile: prefix + ".rttm", atomically: true, encoding: .utf8)
let transcript = utterances.sorted { $0.start < $1.start }.map(\.text).joined(separator: " ")
try (transcript + "\n").write(toFile: prefix + ".txt", atomically: true, encoding: .utf8)
let speakers = Set(utterances.map(\.voice)).sorted()
print(
    "Wrote \(name).wav: \(String(format: "%.1f", Double(total) / sampleRate)) s, "
        + "\(utterances.count) utterances, speakers \(speakers.joined(separator: ", "))")
