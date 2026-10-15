// SPDX-License-Identifier: Apache-2.0
//
// Measures Core ML load and prediction latency of a compiled model, as the Swift runtime calls it.
// Build and run:
//   xcrun swiftc -O latency.swift -o build/latency
//   build/latency build/v3-live/Encoder8s.mlmodelc cpuAndNeuralEngine 100 fresh
//
// Units: cpuAndNeuralEngine, cpuOnly, cpuAndGPU, or all.
// With "fresh", the model is first copied to a new temporary directory, and the load time
// of that copy is reported as the first load; a second load of the same copy follows.
// Inputs are random features, full valid lengths, and token 100. After five warm-up
// predictions, the tool prints one JSON line with the median, 95th percentile, mean,
// minimum, and maximum prediction time in milliseconds.
import CoreML
import Foundation

func units(_ name: String) -> MLComputeUnits {
    switch name {
    case "cpuOnly": return .cpuOnly
    case "cpuAndGPU": return .cpuAndGPU
    case "all": return .all
    default: return .cpuAndNeuralEngine
    }
}

func seconds(_ block: () throws -> Void) rethrows -> Double {
    let start = DispatchTime.now().uptimeNanoseconds
    try block()
    return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
}

let arguments = CommandLine.arguments
guard arguments.count >= 4 else {
    FileHandle.standardError.write("usage: latency <model.mlmodelc> <units> <iterations> [fresh]\n".data(using: .utf8)!)
    exit(2)
}
var url = URL(fileURLWithPath: arguments[1])
let unitName = arguments[2]
let iterations = Int(arguments[3])!
let fresh = arguments.count > 4 && arguments[4] == "fresh"

let configuration = MLModelConfiguration()
configuration.computeUnits = units(unitName)

if fresh {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        "bench-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let copy = directory.appendingPathComponent(url.lastPathComponent)
    try FileManager.default.copyItem(at: url, to: copy)
    url = copy
}

var model: MLModel!
let firstLoad = try seconds { model = try MLModel(contentsOf: url, configuration: configuration) }
let secondLoad = try seconds { model = try MLModel(contentsOf: url, configuration: configuration) }

// Inputs: random features, full valid lengths, and a non-blank token.
var features: [String: MLFeatureValue] = [:]
for (name, description) in model.modelDescription.inputDescriptionsByName {
    guard let constraint = description.multiArrayConstraint else { continue }
    let array = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
    switch constraint.dataType {
    case .int32:
        let value: Int32
        if name == "mel_length" {
            value = Int32(model.modelDescription.inputDescriptionsByName["mel"]!.multiArrayConstraint!.shape[2].intValue)
        } else if name.hasSuffix("length") {
            value = 1
        } else {
            value = 100
        }
        for index in 0..<array.count { array[index] = NSNumber(value: value) }
    default:
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, _ in
            for index in pointer.indices { pointer[index] = Float.random(in: -1...1) }
        }
    }
    features[name] = MLFeatureValue(multiArray: array)
}
let provider = try MLDictionaryFeatureProvider(dictionary: features)

for _ in 0..<5 { _ = try model.prediction(from: provider) }
var samples: [Double] = []
samples.reserveCapacity(iterations)
for _ in 0..<iterations {
    samples.append(try seconds { _ = try model.prediction(from: provider) })
}
samples.sort()
func percentile(_ p: Double) -> Double {
    samples[min(samples.count - 1, Int((p * Double(samples.count - 1)).rounded()))]
}
let mean = samples.reduce(0, +) / Double(samples.count)
let result: [String: Any] = [
    "model": URL(fileURLWithPath: arguments[1]).lastPathComponent,
    "units": unitName,
    "iterations": iterations,
    "fresh_copy": fresh,
    "first_load_ms": firstLoad * 1000,
    "second_load_ms": secondLoad * 1000,
    "median_ms": percentile(0.5) * 1000,
    "p95_ms": percentile(0.95) * 1000,
    "mean_ms": mean * 1000,
    "min_ms": samples.first! * 1000,
    "max_ms": samples.last! * 1000,
]
let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(data: data, encoding: .utf8)!)
if fresh { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
