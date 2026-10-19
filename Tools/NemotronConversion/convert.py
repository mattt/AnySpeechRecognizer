# SPDX-License-Identifier: Apache-2.0

"""Convert Nemotron 3.5 ASR Streaming into AnySpeechRecognizer's cache-aware Core ML artifacts.

Usage:
    uv run python convert.py
    uv run python convert.py --chunks 160 560 --output-dir build/nemotron

The encoder is one multifunction model with a function for each chunk size, such as
`chunk_560ms`. The functions share the int8 weights. The prediction network and the joint
decision do not depend on the chunk size.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch

import nemotron as nt

# Multifunction models need the iOS 18 and macOS 15 specification.
DEPLOYMENT_TARGET = ct.target.iOS18
ENCODER_QUANTIZATION = {
    "mode": "linear_symmetric",
    "dtype": "int8",
    "granularity": "per_channel",
    "weight_threshold": 2048,
}
AUTHOR = "AnySpeechRecognizer"


def _computes_in_float16(op: Any) -> bool:
    """Select every operation for float16 except length arithmetic and comparisons.

    The cache lengths, positions, and masks stay exact in float32.
    """
    outputs = list(op.outputs)
    if any(str(output.dtype).endswith("bool") for output in outputs):
        return False
    return not (outputs and all(output.rank <= 1 for output in outputs))


def convert_traced(traced, inputs, outputs, float32_lengths: bool = False) -> ct.models.MLModel:
    return ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=DEPLOYMENT_TARGET,
        compute_precision=ct.transform.FP16ComputePrecision(op_selector=_computes_in_float16)
        if float32_lengths
        else ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )


_NUMPY_TYPES = {"float32": np.float32, "float16": np.float16, "int32": np.int32}


def tensor_types(contract: dict[str, tuple[str, tuple[int, ...]]]) -> list[ct.TensorType]:
    return [
        ct.TensorType(name=name, shape=shape, dtype=_NUMPY_TYPES[dtype]) for name, (dtype, shape) in contract.items()
    ]


def output_types(contract: dict[str, tuple[str, tuple[int, ...]]]) -> list[ct.TensorType]:
    return [ct.TensorType(name=name, dtype=_NUMPY_TYPES[dtype]) for name, (dtype, _) in contract.items()]


def describe(model: ct.models.MLModel, description: str) -> None:
    model.short_description = f"{nt.TITLE} {description}"
    model.author = AUTHOR
    model.license = f"{nt.LICENSE}. {nt.ATTRIBUTION}"
    model.version = "1"
    model.user_defined_metadata["source_checkpoint"] = f"{nt.REPOSITORY}@{nt.REVISION}"


_DTYPE_NAMES = {
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32: "float32",
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16: "float16",
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32: "int32",
}


def check_interface(model: ct.models.MLModel, component: str, contract: dict) -> None:
    """Fail unless the model's inputs and outputs match the contract exactly."""
    spec = model.get_spec()
    for kind, features in (("inputs", spec.description.input), ("outputs", spec.description.output)):
        actual = {
            f.name: (_DTYPE_NAMES.get(f.type.multiArrayType.dataType, "other"), tuple(f.type.multiArrayType.shape))
            for f in features
        }
        if actual != contract[kind]:
            raise RuntimeError(f"{component} {kind} {actual} do not match the contract {contract[kind]}")


def compile_model(package: Path, destination: Path) -> None:
    """Compile an .mlpackage to destination (an .mlmodelc directory) with coremlcompiler."""
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(package), scratch], check=True, stdout=subprocess.DEVNULL
        )
        compiled = Path(scratch) / f"{package.stem}.mlmodelc"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(compiled), str(destination))


def trace_inputs(model: Any, chunk: nt.Chunk) -> tuple[torch.Tensor, ...]:
    """A chunk of deterministic noise after one step, so the caches hold values."""
    generator = torch.Generator().manual_seed(0)
    audio = (0.05 * torch.randn(chunk.mel_frames * nt.MEL_HOP_SAMPLES * 3, generator=generator)).numpy()
    mel = nt.nemo_mel(model, audio.astype(np.float32))
    steps = nt.TorchSteps(model)
    caches = nt.initial_caches()
    window, valid = nt.chunk_input(mel, 0, chunk)
    outputs = steps.encode(
        {**caches, "mel": window, "mel_length": np.array([valid], np.int32), "prompt": nt.prompt_vector(0).numpy()}
    )
    window, valid = nt.chunk_input(mel, chunk.mel_frames, chunk)
    return (
        torch.from_numpy(window),
        torch.tensor([valid], dtype=torch.int32),
        torch.from_numpy(outputs["cache_channel_out"]),
        torch.from_numpy(outputs["cache_time_out"]),
        torch.from_numpy(outputs["cache_length_out"].astype(np.int32)),
        nt.prompt_vector(0),
    )


def export_encoder(model: Any, chunk: nt.Chunk, work_dir: Path, cache_dtype: str) -> Path:
    """Trace, convert, and quantize the encoder step for one chunk size."""
    nt.set_chunk(model, chunk)
    step = nt.EncoderStep(model).eval()
    inputs = trace_inputs(model, chunk)
    with torch.inference_mode():
        reference = step(*inputs)
    if tuple(reference[0].shape) != (1, nt.ENCODER_DIM, chunk.frames):
        raise RuntimeError(f"unexpected encoder shape {tuple(reference[0].shape)}")
    traced = torch.jit.trace(step, inputs, strict=False).eval()
    contract = chunk.contract(cache_dtype)
    converted = convert_traced(
        traced, tensor_types(contract["inputs"]), output_types(contract["outputs"]), float32_lengths=True
    )
    quantized = cto.linear_quantize_weights(
        converted,
        config=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(**ENCODER_QUANTIZATION)),
    )
    del converted
    describe(quantized, f"cache-aware encoder step, {chunk.milliseconds} ms chunks, int8 per-channel weights")
    check_interface(quantized, f"encoder {chunk.function}", contract)
    package = work_dir / chunk.package
    if package.exists():
        shutil.rmtree(package)
    quantized.save(str(package))
    return package


def combine_encoders(packages: dict[nt.Chunk, Path], default: nt.Chunk, destination: Path) -> None:
    """Combine the encoder packages into one multifunction package with shared weights."""
    descriptor = ct.utils.MultiFunctionDescriptor()
    for chunk, package in packages.items():
        descriptor.add_function(str(package), src_function_name="main", target_function_name=chunk.function)
    descriptor.default_function_name = default.function
    if destination.exists():
        shutil.rmtree(destination)
    ct.utils.save_multifunction(descriptor, str(destination))


def model_card(record: dict[str, Any]) -> str:
    tools = record["tools"]
    source = record["source"]
    chunks = ", ".join(f"`{c['function']}`" for c in record["conversion"]["components"]["encoder"]["functions"])
    return f"""---
license: other
license_name: openmdw-1.1
license_link: {nt.LICENSE_URL}
base_model: {nt.REPOSITORY}
base_model_relation: quantized
pipeline_tag: automatic-speech-recognition
tags: [coreml, nemotron, rnnt, streaming-asr, speech-recognition, apple-silicon]
---

# {nt.TITLE} for Core ML

Cache-aware streaming Core ML components of
[{nt.REPOSITORY}](https://huggingface.co/{nt.REPOSITORY})
for [AnySpeechRecognizer](https://github.com/mattt/AnySpeechRecognizer).

## Attribution and license

{nt.ATTRIBUTION}

## Files

| File | Contents |
| --- | --- |
| `{nt.ENCODER_FILE}` | Multifunction cache-aware encoder step with the language prompt projection; functions {chunks}; int8 per-channel weights shared by the functions. |
| `{nt.DECODER_FILE}` | Prediction network (two-layer LSTM), one step with explicit state. |
| `{nt.JOINT_FILE}` | Joint network and greedy decision for one encoder frame. |
| `{nt.VOCAB_FILE}` | Token ID to SentencePiece piece, with the word-boundary marker written as a space. |
| `{nt.PROMPTS_FILE}` | Locale and `auto` to language prompt slot. |
| `{nt.CONVERSION_FILE}` | Source, tool versions, settings, and SHA-256 digests of every file. |

Each encoder function takes the 9 mel frames before a chunk and the chunk's mel frames
(NeMo log-mel features without normalization), the attention and convolution caches
of the previous step, and a one-hot language prompt, and returns the chunk's projected
encoder frames and the next caches.

## Conversion

- Source: `{source["repo"]}` at revision `{source["revision"]}`: `{source["file"]}`
  (SHA-256 `{source["sha256"]}`).
- Conversion: TorchScript trace of NeMo's `cache_aware_stream_step` to ML programs with an
  iOS 18 and macOS 15 deployment target and float16 compute precision.
- Encoder weights: int8 linear symmetric quantization, one scale per output channel.
- Tools: Python {tools["python"]}, PyTorch {tools["torch"]}, NeMo {tools["nemo_toolkit"]},
  coremltools {tools["coremltools"]}, coremlcompiler {tools["coremlcompiler"]}, macOS {tools["macos"]}.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=nt.BUILD_DIR / "nemotron")
    parser.add_argument("--work-dir", type=Path, default=nt.BUILD_DIR / "work" / "nemotron")
    parser.add_argument("--cache-dir", type=Path, default=nt.DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=sorted(nt.CHUNKS),
        choices=sorted(nt.CHUNKS),
        metavar="MILLISECONDS",
        help=f"Chunk sizes to export, from {sorted(nt.CHUNKS)} (default: all)",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="The data type of the encoder's cache inputs and outputs (default: float32)",
    )
    parser.add_argument(
        "--reuse-packages",
        action="store_true",
        help="Use encoder packages that an earlier run left in the work directory",
    )
    args = parser.parse_args()
    chunks = [nt.CHUNKS[ms] for ms in sorted(set(args.chunks))]
    output_dir: Path = args.output_dir.resolve()
    work_dir: Path = args.work_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    timings: dict[str, float] = {}

    def mark(step: str, since: float) -> float:
        now = time.time()
        timings[step] = round(now - since, 1)
        print(f"{step}: {timings[step]} s", flush=True)
        return now

    t = time.time()
    model, source = nt.load_model(args.cache_dir.resolve())
    t = mark("load checkpoint", t)

    packages: dict[nt.Chunk, Path] = {}
    for chunk in chunks:
        package = work_dir / chunk.package
        if args.reuse_packages and package.exists():
            packages[chunk] = package
            continue
        packages[chunk] = export_encoder(model, chunk, work_dir, args.cache_dtype)
        t = mark(f"convert and quantize the {chunk.milliseconds} ms encoder", t)
    default = chunks[-1]
    combined = work_dir / "Encoder.mlpackage"
    combine_encoders(packages, default, combined)
    t = mark("combine encoders", t)

    decoder = nt.DecoderWrapper(model.decoder).eval()
    targets = torch.full((1, 1), nt.BLANK_ID, dtype=torch.int32)
    length = torch.tensor([1], dtype=torch.int32)
    state = torch.zeros(nt.DECODER_LAYERS, 1, nt.DECODER_DIM)
    with torch.inference_mode():
        decoder_reference, _, _ = decoder(targets, length, state, state)
    decoder_reference = decoder_reference.clone()
    traced = torch.jit.trace(decoder, (targets, length, state, state), strict=False).eval()
    contract = nt.CONTRACT["decoder"]
    decoder_model = convert_traced(traced, tensor_types(contract["inputs"]), output_types(contract["outputs"]))
    describe(decoder_model, "prediction network (RNN-T decoder), one step with explicit LSTM state")
    check_interface(decoder_model, "decoder", contract)
    decoder_package = work_dir / "Decoder.mlpackage"
    decoder_model.save(str(decoder_package))
    t = mark("convert decoder", t)

    joint = nt.JointDecision(model.joint).eval()
    encoder_step = 0.1 * torch.randn(1, nt.ENCODER_DIM, 1, generator=torch.Generator().manual_seed(1))
    traced = torch.jit.trace(joint, (encoder_step, decoder_reference), strict=False).eval()
    contract = nt.CONTRACT["joint"]
    joint_model = convert_traced(traced, tensor_types(contract["inputs"]), output_types(contract["outputs"]))
    describe(joint_model, "joint network and greedy RNN-T decision for one frame")
    check_interface(joint_model, "joint", contract)
    joint_package = work_dir / "JointDecision.mlpackage"
    joint_model.save(str(joint_package))
    t = mark("convert joint decision", t)

    compile_model(combined, output_dir / nt.ENCODER_FILE)
    compile_model(decoder_package, output_dir / nt.DECODER_FILE)
    compile_model(joint_package, output_dir / nt.JOINT_FILE)
    t = mark("compile", t)

    (output_dir / nt.VOCAB_FILE).write_text(nt.vocabulary_json(model), encoding="utf-8")
    prompts = dict(sorted(nt.prompt_dictionary(model).items()))
    (output_dir / nt.PROMPTS_FILE).write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")

    record: dict[str, Any] = {
        "format_version": 1,
        "title": nt.TITLE,
        "created": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "license": nt.LICENSE,
        "attribution": nt.ATTRIBUTION,
        "source": source,
        "tools": nt.tool_versions(),
        "conversion": {
            "source_dialect": "TorchScript (torch.jit.trace)",
            "model_type": "mlprogram",
            "minimum_deployment_target": "iOS18 (macOS 15)",
            "compute_precision": "FLOAT16, with float32 lengths and masks in the encoder",
            "mel_features": {
                "preprocessor": "NeMo AudioToMelSpectrogramPreprocessor",
                "normalize": "none",
                "stft_padding": "zeros",
                "valid_frames": "floor(valid samples / 160)",
            },
            "streaming": {
                "attention_cache_frames": nt.ATTENTION_CACHE_FRAMES,
                "convolution_cache_frames": nt.CONVOLUTION_CACHE_FRAMES,
                "pre_encode_cache_frames": nt.PRE_ENCODE_CACHE_FRAMES,
                "dropped_pre_encoded_frames": nt.DROPPED_PRE_ENCODED_FRAMES,
                "first_step": "pad_and_drop_preencoded",
                "cache_dtype": args.cache_dtype,
            },
            "components": {
                "encoder": {
                    "file": nt.ENCODER_FILE,
                    "default_function": default.function,
                    "functions": [
                        {
                            "function": chunk.function,
                            "att_context_size": [nt.ATTENTION_CACHE_FRAMES, chunk.right],
                            "chunk_frames": chunk.frames,
                            "mel_input_frames": chunk.input_frames,
                            "contract": {
                                kind: {name: {"dtype": d, "shape": list(s)} for name, (d, s) in specs.items()}
                                for kind, specs in chunk.contract(args.cache_dtype).items()
                            },
                        }
                        for chunk in chunks
                    ],
                    "weights": "int8 linear quantization",
                    "quantization": {
                        "api": "coremltools.optimize.coreml.linear_quantize_weights",
                        **ENCODER_QUANTIZATION,
                    },
                },
                "decoder": {"file": nt.DECODER_FILE, "weights": "float16"},
                "joint": {"file": nt.JOINT_FILE, "weights": "float16"},
                "vocabulary": {"file": nt.VOCAB_FILE, "pieces": nt.VOCAB_SIZE, "blank_id": nt.BLANK_ID},
                "prompts": {"file": nt.PROMPTS_FILE, "slots": nt.NUM_PROMPTS, "auto": prompts[nt.AUTO_PROMPT]},
            },
            "contract": {
                component: {
                    kind: {name: {"dtype": d, "shape": list(s)} for name, (d, s) in specs.items()}
                    for kind, specs in parts.items()
                }
                for component, parts in nt.CONTRACT.items()
            },
            "max_symbols_per_frame": nt.MAX_SYMBOLS_PER_FRAME,
            "timings_seconds": timings,
        },
    }
    record["files"] = nt.file_inventory(output_dir, exclude=(nt.CONVERSION_FILE, nt.MODEL_CARD_FILE))
    record["total_size"] = sum(entry["size"] for entry in record["files"])
    (output_dir / nt.CONVERSION_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    (output_dir / nt.MODEL_CARD_FILE).write_text(model_card(record), encoding="utf-8")
    print(f"wrote {output_dir} ({record['total_size'] / 1e6:.1f} MB) in {time.time() - started:.0f} s")
    for name in (nt.ENCODER_FILE, nt.DECODER_FILE, nt.JOINT_FILE):
        print(f"  {name}: {nt.directory_size(output_dir / name)} bytes")


if __name__ == "__main__":
    main()
