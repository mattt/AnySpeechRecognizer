// Times Nemotron encoder steps, the prediction network, and the joint decision with Core ML.
//
//     xcrun swiftc -O latency.swift -o build/latency
//     build/latency build/nemotron cpuAndNeuralEngine 200
//
// Each encoder function runs with zero caches and silent features after five warm-up calls.
// The caches pass from one call to the next through output backings, as the runtime does.

import CoreML
import Foundation

let arguments = CommandLine.arguments
guard arguments.count >= 3 else {
    print("usage: latency <model directory> <compute units> [calls]")
    exit(1)
}
let directory = URL(fileURLWithPath: arguments[1])
let units: MLComputeUnits =
    switch arguments[2] {
    case "cpuOnly": .cpuOnly
    case "cpuAndGPU": .cpuAndGPU
    case "all": .all
    default: .cpuAndNeuralEngine
    }
let calls = arguments.count > 3 ? Int(arguments[3])! : 100

func load(_ name: String, function: String? = nil) throws -> (MLModel, Double) {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = units
    configuration.functionName = function
    let started = Date()
    let model = try MLModel(
        contentsOf: directory.appending(path: name), configuration: configuration)
    return (model, Date().timeIntervalSince(started))
}

func array(_ shape: [Int], _ type: MLMultiArrayDataType = .float32) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
    array.withUnsafeMutableBytes { pointer, _ in
        _ = pointer.initializeMemory(as: UInt8.self, repeating: 0)
    }
    return array
}

func summary(_ times: [Double]) -> String {
    let sorted = times.sorted()
    let median = sorted[sorted.count / 2] * 1000
    let p95 = sorted[min(sorted.count - 1, sorted.count * 95 / 100)] * 1000
    return String(format: "median %.2f ms, p95 %.2f ms", median, p95)
}

let record =
    try JSONSerialization.jsonObject(
        with: Data(contentsOf: directory.appending(path: "conversion.json"))) as! [String: Any]
let conversion = record["conversion"] as! [String: Any]
let encoderRecord = (conversion["components"] as! [String: Any])["encoder"] as! [String: Any]
let functions = (encoderRecord["functions"] as! [[String: Any]]).map {
    ($0["function"] as! String, $0["chunk_frames"] as! Int, $0["mel_input_frames"] as! Int)
}

for (function, frames, melFrames) in functions {
    let (encoder, loadSeconds) = try load("Encoder.mlmodelc", function: function)
    let mel = try array([1, 128, melFrames])
    let melLength = try array([1], .int32)
    melLength[0] = NSNumber(value: melFrames)
    let prompt = try array([1, 128])
    prompt[0] = 1
    let cacheType =
        encoder.modelDescription.inputDescriptionsByName["cache_channel"]?.multiArrayConstraint?
        .dataType ?? .float32
    let sets = try (0..<2).map { _ in
        (
            try array([24, 1, 56, 1024], cacheType), try array([24, 1, 1024, 8], cacheType),
            try array([1], .int32)
        )
    }
    let encoded = try array([1, 1024, frames])
    let providers = try sets.map { set in
        try MLDictionaryFeatureProvider(dictionary: [
            "mel": mel, "mel_length": melLength, "prompt": prompt, "cache_channel": set.0,
            "cache_time": set.1, "cache_length": set.2,
        ])
    }
    let options = sets.indices.map { index in
        let next = sets[1 - index]
        let options = MLPredictionOptions()
        options.outputBackings = [
            "encoder": encoded, "cache_channel_out": next.0, "cache_time_out": next.1,
            "cache_length_out": next.2,
        ]
        return options
    }
    var current = 0
    var times: [Double] = []
    for call in 0..<(calls + 5) {
        let started = Date()
        _ = try encoder.prediction(from: providers[current], options: options[current])
        if call >= 5 { times.append(Date().timeIntervalSince(started)) }
        current = 1 - current
    }
    let perSecond = 1000 / Double(frames * 80)
    let sorted = times.sorted()
    print(
        String(
            format: "%@: load %.1f s, %@, %.1f ms of encoder per second of audio", function,
            loadSeconds, summary(times), sorted[sorted.count / 2] * 1000 * perSecond))
}

let (decoder, _) = try load("Decoder.mlmodelc")
let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
    "targets": try array([1, 1], .int32), "target_length": try array([1], .int32),
    "h_in": try array([2, 1, 640]), "c_in": try array([2, 1, 640]),
])
var times: [Double] = []
for call in 0..<(calls * 5 + 5) {
    let started = Date()
    _ = try decoder.prediction(from: decoderInput)
    if call >= 5 { times.append(Date().timeIntervalSince(started)) }
}
print("Decoder: \(summary(times))")

let (joint, _) = try load("JointDecision.mlmodelc")
let jointInput = try MLDictionaryFeatureProvider(dictionary: [
    "encoder_step": try array([1, 1024, 1]), "decoder_step": try array([1, 640, 1]),
])
times = []
for call in 0..<(calls * 5 + 5) {
    let started = Date()
    _ = try joint.prediction(from: jointInput)
    if call >= 5 { times.append(Date().timeIntervalSince(started)) }
}
print("JointDecision: \(summary(times))")
