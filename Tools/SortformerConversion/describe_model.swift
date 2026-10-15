// SPDX-License-Identifier: Apache-2.0
//
// Prints the interface of a compiled Core ML model as JSON, as Core ML reports it to the
// Swift runtime: input and output names, data types, and shapes, and the model metadata.
//
// Usage:
//     xcrun swiftc -O describe_model.swift -o build/describe_model
//     build/describe_model <model.mlmodelc>

import CoreML
import Foundation

func dataTypeName(_ type: MLMultiArrayDataType) -> String {
    switch type {
    case .float16: "float16"
    case .float32: "float32"
    case .float64: "float64"
    case .int32: "int32"
    default: "other(\(type.rawValue))"
    }
}

func describe(_ features: [String: MLFeatureDescription]) -> [String: Any] {
    var result: [String: Any] = [:]
    for (name, feature) in features {
        guard let constraint = feature.multiArrayConstraint else {
            result[name] = ["type": "non-array(\(feature.type.rawValue))"]
            continue
        }
        result[name] = [
            "dtype": dataTypeName(constraint.dataType),
            "shape": constraint.shape.map(\.intValue),
        ]
    }
    return result
}

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: describe_model <model.mlmodelc>\n".utf8))
    exit(2)
}
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuOnly
let model = try MLModel(
    contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]), configuration: configuration)
let description = model.modelDescription
var metadata: [String: Any] = [:]
for (key, value) in description.metadata {
    metadata[key.rawValue] = value
}
let output: [String: Any] = [
    "inputs": describe(description.inputDescriptionsByName),
    "outputs": describe(description.outputDescriptionsByName),
    "metadata": metadata,
]
let data = try JSONSerialization.data(
    withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
print(String(decoding: data, as: UTF8.self))
