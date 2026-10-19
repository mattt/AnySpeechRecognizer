import Accelerate
import CoreML
import Foundation

/// Runs the encoder, prediction network, and joint network of a Parakeet TDT model with Core ML.
///
/// The engine loads `Encoder_v2.mlmodelc` when it exists and `Encoder.mlmodelc` otherwise,
/// or the encoder that the configuration names, with `Decoder.mlmodelc` and
/// `JointDecisionv3.mlmodelc`. The encoder's input and output shapes set the window of
/// ``model``, so a shorter fixed-window encoder, such as `Encoder8s.mlmodelc`, works with the
/// same prediction and joint networks. A second encoder for forced commits can be loaded
/// beside it; see ``forcedCommitModel``. The engine does not use the Core ML `Preprocessor`;
/// ``ParakeetMelFrontend`` computes the encoder input.
///
/// Each engine owns its `MLModel` instances and reuses its input and output arrays,
/// so one owner uses it at a time. The SDK headers neither mark `MLModel` as `Sendable`
/// nor document concurrent synchronous predictions on one instance, so engines do not
/// share models across threads. Core ML caches compiled models, so later loads of the
/// same directory are faster than the first.
final class CoreMLParakeetEngine: ParakeetInferenceEngine {
    struct Configuration: Sendable {
        /// Compute units for the encoder.
        var encoderComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
        /// Compute units for the prediction and joint networks.
        var predictionComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
        /// An encoder file name that overrides the default selection.
        var encoderName: String?
        /// An encoder with a longer window to load beside the selected one for windows that
        /// commit without right context, or `nil`.
        var forcedCommitEncoderName: String?
    }

    /// The live encoder that a staged directory can contain: the 8-second window that
    /// the conversion tool's live study recommends.
    static let liveEncoderName = "Encoder8s.mlmodelc"

    let model: ParakeetModelDescription
    var forcedCommitModel: ParakeetModelDescription? { forcedCommitEncoder?.model }
    private let encoder: WindowEncoder
    private let forcedCommitEncoder: WindowEncoder?
    private let transducer: CoreMLTransducer

    /// Loads the models from a directory without downloading.
    init(
        directory: URL, configuration: Configuration = Configuration(),
        model: ParakeetModelDescription = .v3
    ) throws {
        func load(_ name: String, units: MLComputeUnits) throws -> MLModel {
            let url = directory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SpeechRecognitionError.missingAssets("Missing Parakeet model \(url.path).")
            }
            let options = MLModelConfiguration()
            options.computeUnits = units
            return try MLModel(contentsOf: url, configuration: options)
        }
        let encoderName =
            configuration.encoderName
            ?? (FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("Encoder_v2.mlmodelc").path)
                ? "Encoder_v2.mlmodelc" : "Encoder.mlmodelc")
        encoder = try WindowEncoder(
            name: encoderName, model: load(encoderName, units: configuration.encoderComputeUnits),
            base: model)
        forcedCommitEncoder = try configuration.forcedCommitEncoderName.map {
            try WindowEncoder(
                name: $0, model: load($0, units: configuration.encoderComputeUnits), base: model)
        }
        let model = encoder.model
        self.model = model
        transducer = try CoreMLTransducer(
            decoder: load("Decoder.mlmodelc", units: configuration.predictionComputeUnits),
            joint: load("JointDecisionv3.mlmodelc", units: configuration.predictionComputeUnits),
            model: model, name: "Parakeet")
        let encoders = [encoder] + (forcedCommitEncoder.map { [$0] } ?? [])
        guard
            encoders.allSatisfy({
                $0.model.windowFrames > 0 && $0.model.windowFrames <= $0.model.encoderFrameCount
            })
        else {
            throw SpeechRecognitionError.missingAssets(
                "The Parakeet models do not match the expected tensor shapes.")
        }
    }

    func encode(
        mel features: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: inout ParakeetEncoderOutput
    ) throws {
        try encoder.encode(features, validFrameCount: validFrameCount, into: &output)
    }

    func encodeForcedCommit(
        mel features: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: inout ParakeetEncoderOutput
    ) throws {
        guard let forcedCommitEncoder else {
            throw SpeechRecognitionError.invalidResult(
                "No Parakeet forced-commit encoder is loaded.")
        }
        try forcedCommitEncoder.encode(features, validFrameCount: validFrameCount, into: &output)
    }

    func predict(token: Int, state: inout ParakeetDecoderState, output: inout [Float]) throws {
        try transducer.predict(token: token, state: &state, output: &output)
    }

    func joint(
        encoderFrame: UnsafeBufferPointer<Float>, decoderOutput: UnsafeBufferPointer<Float>
    ) throws -> ParakeetJointDecision {
        try transducer.joint(encoderFrame: encoderFrame, decoderOutput: decoderOutput)
    }
}

/// Runs a transducer's prediction network and joint decision with Core ML.
///
/// The prediction network takes `targets`, `target_length`, `h_in`, and `c_in` and returns
/// `decoder`, `h_out`, and `c_out`. The joint decision takes `encoder_step` and `decoder_step`
/// and returns `token_id`, `token_prob`, and, for a TDT model, `duration`. A joint decision
/// without a `duration` output reports duration class 0, as an RNN-T model needs.
/// Parakeet and Nemotron share this contract. The transducer reuses its input and output
/// arrays, so one owner uses it at a time.
final class CoreMLTransducer: ParakeetTransducer {
    let model: ParakeetModelDescription
    private let name: String
    private let decoder: MLModel
    private let jointNetwork: MLModel
    private let targets: MLMultiArray
    private let hiddenInput: MLMultiArray
    private let cellInput: MLMultiArray
    private let decoderInput: FeatureProvider
    private let projectionOutput: MLMultiArray
    private let hiddenOutput: MLMultiArray
    private let cellOutput: MLMultiArray
    private let decoderOptions = MLPredictionOptions()
    private let encoderStep: MLMultiArray
    private let decoderStep: MLMultiArray
    private let jointInput: FeatureProvider
    private let tokenOutput: MLMultiArray
    private let probabilityOutput: MLMultiArray
    private let durationOutput: MLMultiArray?
    private let jointOptions = MLPredictionOptions()

    /// - Parameter name: The model name for error messages, such as "Parakeet".
    init(decoder: MLModel, joint: MLModel, model: ParakeetModelDescription, name: String) throws {
        self.model = model
        self.name = name
        self.decoder = decoder
        jointNetwork = joint
        let layers = NSNumber(value: model.decoderLayerCount)
        let decoderHidden = NSNumber(value: model.decoderHiddenSize)
        targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let targetLength = try MLMultiArray(shape: [1], dataType: .int32)
        targetLength[0] = 1
        hiddenInput = try MLMultiArray(shape: [layers, 1, decoderHidden], dataType: .float32)
        cellInput = try MLMultiArray(shape: [layers, 1, decoderHidden], dataType: .float32)
        decoderInput = FeatureProvider([
            "targets": targets, "target_length": targetLength, "h_in": hiddenInput,
            "c_in": cellInput,
        ])
        // Output backings let Core ML write into fixed arrays instead of allocating and
        // binding new outputs for each of the many small predictions.
        projectionOutput = try MLMultiArray(shape: [1, decoderHidden, 1], dataType: .float32)
        hiddenOutput = try MLMultiArray(shape: [layers, 1, decoderHidden], dataType: .float32)
        cellOutput = try MLMultiArray(shape: [layers, 1, decoderHidden], dataType: .float32)
        decoderOptions.outputBackings = [
            "decoder": projectionOutput, "h_out": hiddenOutput, "c_out": cellOutput,
        ]
        encoderStep = try MLMultiArray(
            shape: [1, NSNumber(value: model.encoderHiddenSize), 1], dataType: .float32)
        decoderStep = try MLMultiArray(shape: [1, decoderHidden, 1], dataType: .float32)
        jointInput = FeatureProvider(["encoder_step": encoderStep, "decoder_step": decoderStep])
        tokenOutput = try MLMultiArray(shape: [1, 1, 1], dataType: .int32)
        probabilityOutput = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
        var backings: [String: Any] = ["token_id": tokenOutput, "token_prob": probabilityOutput]
        if joint.modelDescription.outputDescriptionsByName["duration"] != nil {
            let duration = try MLMultiArray(shape: [1, 1, 1], dataType: .int32)
            backings["duration"] = duration
            durationOutput = duration
        } else {
            durationOutput = nil
        }
        jointOptions.outputBackings = backings
        try validateShapes()
    }

    func predict(token: Int, state: inout ParakeetDecoderState, output: inout [Float]) throws {
        targets.withUnsafeMutableBufferPointer(ofType: Int32.self) { pointer, _ in
            pointer[0] = Int32(token)
        }
        try state.hidden.withUnsafeBufferPointer { try hiddenInput.writeLogical($0) }
        try state.cell.withUnsafeBufferPointer { try cellInput.writeLogical($0) }
        let result = try decoder.prediction(from: decoderInput, options: decoderOptions)
        guard let projection = result.featureValue(for: "decoder")?.multiArrayValue,
            let hidden = result.featureValue(for: "h_out")?.multiArrayValue,
            let cell = result.featureValue(for: "c_out")?.multiArrayValue,
            projection.count == model.decoderHiddenSize,
            hidden.count == model.decoderStateSize, cell.count == model.decoderStateSize
        else {
            throw SpeechRecognitionError.invalidResult("The \(name) decoder output is invalid.")
        }
        if output.count != model.decoderHiddenSize {
            output = [Float](repeating: 0, count: model.decoderHiddenSize)
        }
        try output.withUnsafeMutableBufferPointer { try projection.readLogical(into: $0) }
        try state.hidden.withUnsafeMutableBufferPointer { try hidden.readLogical(into: $0) }
        try state.cell.withUnsafeMutableBufferPointer { try cell.readLogical(into: $0) }
    }

    func joint(
        encoderFrame: UnsafeBufferPointer<Float>, decoderOutput: UnsafeBufferPointer<Float>
    ) throws -> ParakeetJointDecision {
        try encoderStep.writeLogical(encoderFrame)
        try decoderStep.writeLogical(decoderOutput)
        _ = try jointNetwork.prediction(from: jointInput, options: jointOptions)
        let token = tokenOutput.withUnsafeBufferPointer(ofType: Int32.self) { Int($0[0]) }
        let probability = probabilityOutput.withUnsafeBufferPointer(ofType: Float.self) { $0[0] }
        let durationClass =
            durationOutput?.withUnsafeBufferPointer(ofType: Int32.self) { Int($0[0]) } ?? 0
        guard (0...model.blankID).contains(token), model.durations.indices.contains(durationClass)
        else {
            throw SpeechRecognitionError.invalidResult("The \(name) joint output is invalid.")
        }
        return ParakeetJointDecision(
            token: token, probability: probability.isFinite ? min(1, max(0, probability)) : 0,
            duration: model.durations[durationClass])
    }

    private func validateShapes() throws {
        func shape(_ model: MLModel, input name: String) -> [Int]? {
            model.modelDescription.inputDescriptionsByName[name]?.multiArrayConstraint?.shape
                .map(\.intValue)
        }
        let expected = model
        guard
            shape(decoder, input: "h_in")
                == [expected.decoderLayerCount, 1, expected.decoderHiddenSize],
            shape(jointNetwork, input: "encoder_step") == [1, expected.encoderHiddenSize, 1],
            shape(jointNetwork, input: "decoder_step") == [1, expected.decoderHiddenSize, 1]
        else {
            throw SpeechRecognitionError.missingAssets(
                "The \(name) models do not match the expected tensor shapes.")
        }
    }
}

/// One fixed-window encoder and its reusable input arrays.
private final class WindowEncoder {
    /// The model contract with this encoder's window.
    let model: ParakeetModelDescription
    private let encoder: MLModel
    private let mel: MLMultiArray
    private let melLength: MLMultiArray
    private let input: FeatureProvider

    /// Reads the window from the encoder's `mel` input and `encoder` output shapes.
    init(name: String, model encoder: MLModel, base: ParakeetModelDescription) throws {
        let description = encoder.modelDescription
        guard
            let input = description.inputDescriptionsByName["mel"]?.multiArrayConstraint?.shape
                .map(\.intValue),
            let output = description.outputDescriptionsByName["encoder"]?.multiArrayConstraint?
                .shape.map(\.intValue),
            input.count == 3, output.count == 3, input[0] == 1, output[0] == 1,
            input[1] == base.melBinCount, output[1] == base.encoderHiddenSize,
            input[2] > 1, output[2] > 0
        else {
            throw SpeechRecognitionError.missingAssets(
                "The Parakeet encoder \(name) does not have the expected fixed-window shapes.")
        }
        model = base.withEncoderWindow(melFrameCount: input[2], encoderFrameCount: output[2])
        self.encoder = encoder
        mel = try MLMultiArray(
            shape: [1, NSNumber(value: model.melBinCount), NSNumber(value: model.melFrameCount)],
            dataType: .float32)
        melLength = try MLMultiArray(shape: [1], dataType: .int32)
        self.input = FeatureProvider(["mel": mel, "mel_length": melLength])
    }

    func encode(
        _ features: UnsafeBufferPointer<Float>, validFrameCount: Int,
        into output: inout ParakeetEncoderOutput
    ) throws {
        precondition(features.count == model.melBinCount * model.melFrameCount)
        try mel.writeLogical(features)
        melLength[0] = NSNumber(value: Int32(validFrameCount))
        let result = try encoder.prediction(from: input)
        guard let encoded = result.featureValue(for: "encoder")?.multiArrayValue,
            let length = result.featureValue(for: "encoder_length")?.multiArrayValue,
            encoded.shape.map(\.intValue)
                == [1, model.encoderHiddenSize, model.encoderFrameCount]
        else {
            throw SpeechRecognitionError.invalidResult("The Parakeet encoder output is invalid.")
        }
        if output.hiddenSize != model.encoderHiddenSize
            || output.frameCount != model.encoderFrameCount
        {
            output = ParakeetEncoderOutput(
                hiddenSize: model.encoderHiddenSize, frameCount: model.encoderFrameCount)
        }
        try encoded.transposeChannels(into: &output.features)
        output.validFrameCount = min(max(0, length[0].intValue), model.encoderFrameCount)
    }
}

/// Serves fixed input arrays, so predictions reuse them without new providers.
final class FeatureProvider: NSObject, MLFeatureProvider {
    private let values: [String: MLFeatureValue]
    init(_ arrays: [String: MLMultiArray]) {
        values = arrays.mapValues { MLFeatureValue(multiArray: $0) }
    }
    var featureNames: Set<String> { Set(values.keys) }
    func featureValue(for featureName: String) -> MLFeatureValue? { values[featureName] }
}

extension MLMultiArray {
    /// The memory offset of each logical row-major element, or `nil` when the layout is dense.
    private func offsets() -> [Int]? {
        let shape = shape.map(\.intValue)
        let strides = strides.map(\.intValue)
        var dense = 1
        var isDense = true
        for axis in shape.indices.reversed() {
            if shape[axis] > 1, strides[axis] != dense { isDense = false }
            dense *= shape[axis]
        }
        if isDense { return nil }
        var offsets = [0]
        for axis in shape.indices {
            offsets = offsets.flatMap { base in
                (0..<shape[axis]).map { base + $0 * strides[axis] }
            }
        }
        return offsets
    }

    /// Copies values in logical row-major order into a float array.
    func writeLogical(_ values: UnsafeBufferPointer<Float>) throws {
        guard dataType == .float32, values.count == count else {
            throw SpeechRecognitionError.invalidResult("A Core ML model input has the wrong size.")
        }
        let offsets = offsets()
        withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, _ in
            if let offsets {
                for (index, offset) in offsets.enumerated() { pointer[offset] = values[index] }
            } else {
                pointer.baseAddress!.update(from: values.baseAddress!, count: values.count)
            }
        }
    }

    /// Copies a float array's values in logical row-major order.
    func readLogical(into values: UnsafeMutableBufferPointer<Float>) throws {
        guard values.count == count else {
            throw SpeechRecognitionError.invalidResult(
                "A Core ML model output has the wrong size.")
        }
        let offsets = offsets()
        if offsets == nil, dataType == .float32 {
            withUnsafeBufferPointer(ofType: Float.self) { pointer in
                values.baseAddress!.update(from: pointer.baseAddress!, count: values.count)
            }
            return
        }
        func read<Scalar: MLShapedArrayScalar>(_ type: Scalar.Type, _ convert: (Scalar) -> Float) {
            withUnsafeBufferPointer(ofType: type) { pointer in
                for index in 0..<values.count {
                    values[index] = convert(pointer[offsets?[index] ?? index])
                }
            }
        }
        switch dataType {
        case .float32: read(Float.self) { $0 }
        #if arch(arm64)
            // Float16 is unavailable on Intel Macs, which do not run Parakeet.
            case .float16: read(Float16.self) { Float($0) }
        #endif
        default:
            throw SpeechRecognitionError.invalidResult(
                "A Core ML model output is not a float array.")
        }
    }

    /// Copies a `[1, channels, frames]` array into frame-major order, `[frames × channels]`.
    func transposeChannels(into values: inout [Float]) throws {
        let shape = shape.map(\.intValue)
        let strides = strides.map(\.intValue)
        let (channels, frames) = (shape[1], shape[2])
        guard values.count == channels * frames else {
            throw SpeechRecognitionError.invalidResult(
                "A Core ML model output has the wrong size.")
        }
        if dataType == .float32, strides[2] == 1, strides[1] >= frames {
            // Transpose whole rows, including any padding after the last frame, and keep the
            // first `frames` rows of the result.
            let rowLength = strides[1]
            withUnsafeBufferPointer(ofType: Float.self) { source in
                let transposed = [Float](unsafeUninitializedCapacity: rowLength * channels) {
                    buffer, count in
                    vDSP_mtrans(
                        source.baseAddress!, 1, buffer.baseAddress!, 1, vDSP_Length(rowLength),
                        vDSP_Length(channels))
                    count = rowLength * channels
                }
                values.withUnsafeMutableBufferPointer { target in
                    transposed.withUnsafeBufferPointer {
                        target.baseAddress!.update(from: $0.baseAddress!, count: target.count)
                    }
                }
            }
            return
        }
        func read<Scalar: MLShapedArrayScalar>(_ type: Scalar.Type, _ convert: (Scalar) -> Float) {
            withUnsafeBufferPointer(ofType: type) { source in
                values.withUnsafeMutableBufferPointer { target in
                    for channel in 0..<channels {
                        let row = channel * strides[1]
                        for frame in 0..<frames {
                            target[frame * channels + channel] = convert(
                                source[row + frame * strides[2]])
                        }
                    }
                }
            }
        }
        switch dataType {
        case .float32: read(Float.self) { $0 }
        #if arch(arm64)
            // Float16 is unavailable on Intel Macs, which do not run Parakeet.
            case .float16: read(Float16.self) { Float($0) }
        #endif
        default:
            throw SpeechRecognitionError.invalidResult(
                "A Core ML model output is not a float array.")
        }
    }
}
