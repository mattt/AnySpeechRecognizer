# SPDX-License-Identifier: Apache-2.0
"""Convert a Qwen3-ASR checkpoint into AnySpeechRecognizer's Core ML artifacts.

Usage:
    uv run python convert.py --variant 0.6b
    uv run python convert.py --variant 0.6b --weights float16 --output-dir build/0.6b-float16

The output directory holds the compiled audio encoder and text decoder, the vocabulary,
the prompt and model configuration, a conversion record, and a model card.
"""

from __future__ import annotations

import argparse
import gc
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

import qwen

# Stateful models need the iOS 18 and macOS 15 specification.
DEPLOYMENT_TARGET = ct.target.iOS18
QUANTIZATION = {
    "mode": "linear_symmetric",
    "dtype": "int8",
    "granularity": "per_channel",
    "weight_threshold": 2048,
}
AUTHOR = "AnySpeechRecognizer"


def describe(model: ct.models.MLModel, variant: qwen.Variant, description: str) -> None:
    model.short_description = f"{variant.title} {description}"
    model.author = AUTHOR
    model.license = f"{qwen.LICENSE}. {qwen.attribution(variant)}"
    model.version = "1"
    model.user_defined_metadata["source_checkpoint"] = f"{variant.repo}@{variant.revision}"


def quantize(model: ct.models.MLModel, weights: str, keep: tuple[str, ...] = ()) -> ct.models.MLModel:
    """Quantizes weights to int8, except constants whose names start with a prefix in `keep`."""
    if weights == "float16":
        return model
    names = [
        op.name for op in model._mil_program.functions["main"].find_ops(op_type="const") if op.name.startswith(keep)
    ]
    if keep and not names:
        raise RuntimeError(f"No constants start with {keep}.")
    config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(**QUANTIZATION), op_name_configs={name: None for name in names}
    )
    return cto.linear_quantize_weights(model, config=config)


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


def export_encoder(model, variant: qwen.Variant, weights: str, work: Path) -> Path:
    wrapper = qwen.EncoderWindow(model).eval()
    mel = torch.zeros(1, qwen.MEL_BINS, qwen.WINDOW_FRAMES)
    length = torch.tensor([qwen.WINDOW_FRAMES], dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (mel, length))
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="mel", shape=mel.shape, dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="audio_embeddings", dtype=np.float32)],
        minimum_deployment_target=DEPLOYMENT_TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    converted = quantize(converted, weights)
    describe(converted, variant, "audio encoder and projector for one 8-second window")
    package = work / "AudioEncoder.mlpackage"
    converted.save(str(package))
    return package


def export_decoder(model, variant: qwen.Variant, weights: str, context: int, block: int, work: Path) -> Path:
    wrapper = qwen.TextDecoder(model, context, block).eval()
    hidden = wrapper.hidden_size
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, wrapper.example_inputs())
    cache = wrapper.cache_shape
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, block), dtype=np.int32),
            ct.TensorType(name="audio_embeddings", shape=(1, block, hidden), dtype=np.float32),
            ct.TensorType(name="position_ids", shape=(block,), dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=(1, 1, block, context), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="token_ids", dtype=np.int32),
            ct.TensorType(name="token_log_probs", dtype=np.float32),
        ],
        states=[
            ct.StateType(wrapped_type=ct.TensorType(shape=cache, dtype=np.float16), name=name)
            for name in wrapper.cache_names()
        ],
        minimum_deployment_target=DEPLOYMENT_TARGET,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    # The token embedding table, which the output head shares, stays float16: with int8 weights
    # the Core ML decoder's first token after the prompt was wrong for every fixture.
    # The rotary tables are positions, not weights, and stay float16 as well.
    converted = quantize(converted, weights, keep=("embed_tokens", "rope_"))
    describe(converted, variant, f"text decoder with a {context}-token KV cache")
    converted.user_defined_metadata["context_length"] = str(context)
    converted.user_defined_metadata["block_length"] = str(block)
    package = work / "TextDecoder.mlpackage"
    converted.save(str(package))
    return package


def write_vocabulary(tokenizer, size: int, path: Path) -> None:
    """Every token's byte-level BPE string, indexed by ID; special tokens keep their text."""
    vocabulary = [""] * size
    for piece, index in tokenizer.get_vocab().items():
        if index < size:
            vocabulary[index] = piece
    qwen.write_json(path, vocabulary)


def model_config(model, tokenizer, context: int, block: int) -> dict[str, Any]:
    config = model.config
    special = {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in [
            ("audio_start", "<|audio_start|>"),
            ("audio_pad", "<|audio_pad|>"),
            ("audio_end", "<|audio_end|>"),
            ("asr_text", "<asr_text>"),
            ("im_start", "<|im_start|>"),
            ("im_end", "<|im_end|>"),
            ("end_of_text", "<|endoftext|>"),
        ]
    }
    assert special["audio_pad"] == config.audio_token_id
    # The added tokens that `decode(..., skip_special_tokens=True)` drops; `<asr_text>` stays,
    # because the output parser splits the language from the transcript at it.
    added = sorted(index for piece, index in tokenizer.get_vocab().items() if index >= 151_643)
    skipped = [index for index in added if tokenizer.decode([index], skip_special_tokens=True) == ""]
    assert special["asr_text"] not in skipped
    return {
        "context_length": context,
        "block_length": block,
        "hidden_size": config.text_config.hidden_size,
        "vocabulary_size": config.text_config.vocab_size,
        "special_tokens": special,
        "skipped_token_ids": skipped,
        "end_tokens": sorted(set(config.eos_token_id)),
        "prompt": qwen.prompt_ids(tokenizer),
        "audio": {
            "sample_rate": qwen.SAMPLE_RATE,
            "mel_bins": qwen.MEL_BINS,
            "fft_size": 400,
            "hop_length": qwen.MEL_HOP_SAMPLES,
            "minimum_samples": 8_000,
            "window_frames": qwen.WINDOW_FRAMES,
            "window_tokens": qwen.WINDOW_TOKENS,
            "chunk_frames": qwen.CHUNK_FRAMES,
            "tokens_per_chunk": qwen.TOKENS_PER_CHUNK,
        },
    }


def model_card(variant: qwen.Variant, record: dict[str, Any]) -> str:
    tools = record["tools"]
    context = record["conversion"]["context_length"]
    weights = record["conversion"]["weights"]
    return f"""---
license: apache-2.0
base_model: {variant.repo}
pipeline_tag: automatic-speech-recognition
library_name: coreml
tags:
- coreml
- speech-recognition
---

# {variant.title} for Core ML

This repository contains a Core ML conversion of
[{variant.repo}](https://huggingface.co/{variant.repo}) at revision `{variant.revision}`
for [AnySpeechRecognizer](https://github.com/mattt/AnySpeechRecognizer).

| File | Contents |
| --- | --- |
| `{qwen.ENCODER_FILE}` | Audio encoder and projector for one 8-second window of 128-bin log-mel features. |
| `{qwen.DECODER_FILE}` | Text decoder for blocks of {record["conversion"]["block_length"]} tokens with a {context}-token KV cache, one key and one value state per layer; returns the greedy next token after each position. |
| `{qwen.VOCAB_FILE}` | Byte-level BPE string of each token ID. |
| `{qwen.CONFIG_FILE}` | Prompt token IDs, special tokens, language prefills, and audio settings. |
| `{qwen.CONVERSION_FILE}` | Sources, tool versions, settings, and the size and SHA-256 digest of every file. |

Weights: {weights}.
Converted with Python {tools["python"]}, PyTorch {tools["torch"]}, transformers {tools["transformers"]},
coremltools {tools["coremltools"]}, and NumPy {tools["numpy"]} on macOS {tools["macos"]}.

## License

{qwen.attribution(variant)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", choices=sorted(qwen.VARIANTS), default="0.6b")
    parser.add_argument("--weights", choices=["int8", "float16"], default="int8")
    parser.add_argument("--context", type=int, default=1024, help="KV cache length in tokens.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="The type to load and trace the checkpoint in; float16 halves the conversion's memory.",
    )
    parser.add_argument("--block", type=int, default=16, help="Tokens that the decoder takes in each call.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=qwen.DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    variant = qwen.VARIANTS[args.variant]
    output = (args.output_dir or qwen.BUILD_DIR / variant.name).resolve()
    work = (args.work_dir or qwen.BUILD_DIR / "work" / output.name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    path = qwen.snapshot(variant, args.cache_dir)
    model, processor = qwen.load_reference(path, dtype=getattr(torch, args.dtype))
    print(f"Loaded {variant.repo}@{variant.revision} ({time.monotonic() - started:.0f} s)")

    # The encoder always traces in float32: from float16, coremltools 9.0 gave its convolutions
    # float16 biases with float32 weights and failed. It is small, and is freed afterward.
    model.model.audio_tower.float()
    model.model.multi_modal_projector.float()
    encoder = export_encoder(model, variant, args.weights, work)
    compile_model(encoder, output / qwen.ENCODER_FILE)
    print(f"Exported {qwen.ENCODER_FILE} ({time.monotonic() - started:.0f} s)")
    del model.model.audio_tower, model.model.multi_modal_projector
    gc.collect()
    decoder = export_decoder(model, variant, args.weights, args.context, args.block, work)
    compile_model(decoder, output / qwen.DECODER_FILE)
    print(f"Exported {qwen.DECODER_FILE} ({time.monotonic() - started:.0f} s)")

    tokenizer = processor.tokenizer
    write_vocabulary(tokenizer, model.config.text_config.vocab_size, output / qwen.VOCAB_FILE)
    qwen.write_json(output / qwen.CONFIG_FILE, model_config(model, tokenizer, args.context, args.block))

    record = {
        "source": {"repository": variant.repo, "revision": variant.revision},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "tools": qwen.environment(),
        "conversion": {
            "weights": "float16"
            if args.weights == "float16"
            else "int8 linear symmetric, one scale per output channel, for tensors of 2,048 or more "
            "elements, except the decoder's token embedding table (shared with the output head) "
            "and rotary tables; float16 otherwise",
            "quantization": None if args.weights == "float16" else QUANTIZATION,
            "compute_precision": "float16",
            "trace_dtype": args.dtype,
            "context_length": args.context,
            "block_length": args.block,
            "minimum_deployment_target": "iOS18 (macOS 15)",
        },
        "license": qwen.LICENSE,
        "attribution": qwen.attribution(variant),
    }
    (output / qwen.MODEL_CARD_FILE).write_text(model_card(variant, record))
    record["files"] = qwen.tree_files(output)
    qwen.write_json(output / qwen.CONVERSION_FILE, record)
    print(f"Wrote {output} ({time.monotonic() - started:.0f} s)")


if __name__ == "__main__":
    main()
