# SPDX-License-Identifier: Apache-2.0
#
# The conversion steps follow FluidInference/mobius
# (models/speaker-diarization/sortformer-streaming/conversion/convert_to_coreml.py and
# build_all_variants.py at commit 26f5fa8059d5ab3d34449adce6ff6ed86734e522),
# Copyright Fluid Inference, licensed under the Apache License, Version 2.0. See NOTICE in this directory.

"""Convert NVIDIA's Streaming Sortformer 4spk v2 into AnySpeechRecognizer's Core ML model.

Usage:
    uv run python convert.py
    uv run python convert.py --output-dir build/sortformer --work-dir build/work

The output directory holds the compiled pipeline model, conversion.json, and a Hugging Face
model card (README.md).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch

import sortformer as sf

DEPLOYMENT_TARGET = ct.target.iOS16
PRE_ENCODER_PRECISION = ct.precision.FLOAT32
HEAD_PRECISION = ct.precision.FLOAT16
AUTHOR = "NVIDIA (Streaming Sortformer 4spk v2); Core ML conversion by AnySpeechRecognizer"

_DTYPE_NAMES = {
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32: "float32",
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16: "float16",
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32: "int32",
}


def interface(spec: Any) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    """Return the inputs and outputs of a model specification as (dtype, shape) pairs."""
    return {
        kind: {
            f.name: (_DTYPE_NAMES.get(f.type.multiArrayType.dataType, "other"), tuple(f.type.multiArrayType.shape))
            for f in features
        }
        for kind, features in (("inputs", spec.description.input), ("outputs", spec.description.output))
    }


def check_interface(model: ct.models.MLModel) -> None:
    """Fail unless the pipeline's inputs and outputs match the runtime contract exactly."""
    actual = interface(model.get_spec())
    for kind in ("inputs", "outputs"):
        if actual[kind] != sf.CONTRACT[kind]:
            raise RuntimeError(f"pipeline {kind} {actual[kind]} do not match the contract {sf.CONTRACT[kind]}")


def trace_inputs() -> dict[str, torch.Tensor]:
    """Deterministic trace inputs with every length at its maximum, as in mobius."""
    generator = torch.Generator().manual_seed(0)
    cache, fifo = sf.STREAMING["spkcache_len"], sf.STREAMING["fifo_len"]
    return {
        "chunk": torch.randn(1, sf.CHUNK_MEL_FRAMES, sf.MEL_BINS, generator=generator),
        "chunk_lengths": torch.tensor([sf.CHUNK_MEL_FRAMES], dtype=torch.long),
        "spkcache": torch.randn(1, cache, sf.EMBEDDING_SIZE, generator=generator),
        "spkcache_lengths": torch.tensor([cache], dtype=torch.long),
        "fifo": torch.randn(1, fifo, sf.EMBEDDING_SIZE, generator=generator),
        "fifo_lengths": torch.tensor([fifo], dtype=torch.long),
    }


def convert_pre_encoder(model: Any, inputs: dict[str, torch.Tensor]) -> ct.models.MLModel:
    wrapper = sf.PreEncoderWrapper(model).eval()
    traced = torch.jit.trace(wrapper, tuple(inputs.values()), strict=False).eval()
    return ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name=name, shape=tensor.shape, dtype=np.int32 if name.endswith("_lengths") else np.float32)
            for name, tensor in inputs.items()
        ],
        outputs=[
            ct.TensorType(name="pre_encoder_embs", dtype=np.float32),
            ct.TensorType(name="pre_encoder_lengths", dtype=np.int32),
            ct.TensorType(name="chunk_pre_encoder_embs", dtype=np.float32),
            ct.TensorType(name="chunk_pre_encoder_lengths", dtype=np.int32),
        ],
        minimum_deployment_target=DEPLOYMENT_TARGET,
        compute_precision=PRE_ENCODER_PRECISION,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )


def convert_head(model: Any) -> ct.models.MLModel:
    generator = torch.Generator().manual_seed(1)
    packed = sf.PREDICTION_ROWS
    example = (
        torch.randn(1, packed, sf.EMBEDDING_SIZE, generator=generator),
        torch.tensor([packed], dtype=torch.long),
        torch.randn(1, sf.CHUNK_EMBEDDING_ROWS, sf.EMBEDDING_SIZE, generator=generator),
        torch.tensor([sf.CHUNK_EMBEDDING_ROWS], dtype=torch.long),
    )
    wrapper = sf.HeadWrapper(model).eval()
    traced = torch.jit.trace(wrapper, example, strict=False).eval()
    return ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="pre_encoder_embs", shape=example[0].shape, dtype=np.float32),
            ct.TensorType(name="pre_encoder_lengths", shape=example[1].shape, dtype=np.int32),
            ct.TensorType(name="chunk_pre_encoder_embs", shape=example[2].shape, dtype=np.float32),
            ct.TensorType(name="chunk_pre_encoder_lengths", shape=example[3].shape, dtype=np.int32),
        ],
        # Without a dtype, the embeddings stay float16, as in FluidInference's model.
        outputs=[
            ct.TensorType(name="speaker_preds", dtype=np.float32),
            ct.TensorType(name="chunk_pre_encoder_embs_out"),
            ct.TensorType(name="chunk_pre_encoder_lengths_out"),
        ],
        minimum_deployment_target=DEPLOYMENT_TARGET,
        compute_precision=HEAD_PRECISION,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )


def describe(pipeline: ct.models.MLModel, source: dict[str, Any]) -> None:
    pipeline.short_description = (
        f"{sf.TITLE} speaker diarization, one streaming step with NVIDIA's low-latency configuration"
    )
    pipeline.author = AUTHOR
    pipeline.license = f"{sf.LICENSE}. {sf.ATTRIBUTION}"
    pipeline.version = "2"
    for key, value in sf.STREAMING_METADATA.items():
        pipeline.user_defined_metadata[key] = value
    pipeline.user_defined_metadata["source_checkpoint"] = f"{source['repo']}@{source['revision']}/{source['file']}"
    pipeline.user_defined_metadata["source_checkpoint_sha256"] = source["sha256"]
    for name, text in sf.INPUT_DESCRIPTIONS.items():
        pipeline.input_description[name] = text
    for name, text in sf.OUTPUT_DESCRIPTIONS.items():
        pipeline.output_description[name] = text


def compile_model(package: Path, destination: Path) -> None:
    """Compile an .mlpackage into destination (an .mlmodelc directory) with Core ML on this Mac.

    coremltools.utils.compile_model calls MLModel.compileModel, as mobius's build_all_variants.py does.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        compiled = Path(ct.utils.compile_model(str(package), str(Path(scratch) / destination.name)))
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(compiled), str(destination))


def model_card(record: dict[str, Any]) -> str:
    """Return a Hugging Face model card for the output directory."""
    source = record["source"]
    tools = record["tools"]
    rows = "\n".join(
        f"| `{name}` | {dtype} | {list(shape)} | {sf.INPUT_DESCRIPTIONS.get(name) or sf.OUTPUT_DESCRIPTIONS[name]} |"
        for kind in ("inputs", "outputs")
        for name, (dtype, shape) in sf.CONTRACT[kind].items()
    )
    metadata = "\n".join(f"| `{key}` | {value} |" for key, value in sf.STREAMING_METADATA.items())
    return f"""---
license: cc-by-4.0
base_model: {sf.CHECKPOINT.repo}
base_model_relation: quantized
library_name: coreml
pipeline_tag: automatic-speech-recognition
tags: [coreml, speaker-diarization, sortformer, streaming, apple-silicon]
---

# {sf.TITLE} for Core ML

A Core ML conversion of NVIDIA's
[{sf.CHECKPOINT.repo}](https://huggingface.co/{sf.CHECKPOINT.repo})
streaming speaker diarization model, for
[AnySpeechRecognizer](https://github.com/mattt/AnySpeechRecognizer).
It predicts, for each 80 ms frame, an independent activity probability for each of
up to four speakers.

## Attribution and license

{sf.ATTRIBUTION}

This repository is licensed under
[CC BY 4.0]({sf.LICENSE_URL}), the license of the base model.
Redistributions must keep this attribution.
This conversion is not the v2.1 checkpoint, which has a different license.

## Files

| File | Contents |
| --- | --- |
| `{sf.MODEL_FILE}` | Compiled Core ML pipeline: pre-encoder (float32) and FastConformer encoder, Transformer, and speaker sigmoids (float16). |
| `{sf.CONVERSION_FILE}` | Source revision and digest, tool versions, settings, and the size and SHA-256 digest of every file. |

## Usage

The model runs one streaming step.
The caller computes NeMo-compatible 128-bin log-mel features
(16 kHz mono, 25 ms window, 10 ms hop, no normalization),
keeps the speaker cache and FIFO queue between steps,
and updates them with NeMo's `SortformerModules.streaming_update`
from `speaker_preds` and `chunk_pre_encoder_embs_out`.
AnySpeechRecognizer's `SortformerDiarizationModel` does all of this:

```swift
let model = SortformerDiarizationModel(assets: .compiledModel(url))
let diarizer = SpeakerDiarizer(model: model)
```

Pass the URL of the `{sf.MODEL_FILE}` directory.
Applications that use the model must show the attribution above.

### Interface

| Name | Type | Shape | Contents |
| --- | --- | --- | --- |
{rows}

Each step takes {sf.CHUNK_MEL_FRAMES} mel frames: {sf.STREAMING["chunk_left_context"]} frame of left context,
{sf.STREAMING["chunk_len"]} chunk frames, and {sf.STREAMING["chunk_right_context"]} frames of right context,
each 8 mel frames (80 ms).
Unused rows of `chunk`, `spkcache`, and `fifo` are ignored after their lengths.
`speaker_preds` has the valid rows of the speaker cache, then of the FIFO, then of the chunk,
followed by zeros.

### Streaming configuration

NVIDIA's low-latency setting, with 1.04 s of algorithmic latency (chunk and right context).
The values are also stored in the model's creator-defined metadata:

| Key | Value |
| --- | --- |
{metadata}

## Conversion

- Source: `{source["repo"]}` at revision `{source["revision"]}`,
  `{source["file"]}` (SHA-256 `{source["sha256"]}`).
- The pre-encoder and the packing of cache, FIFO, and chunk frames are an ML program with
  float32 compute precision.
  The FastConformer encoder, Transformer, and speaker sigmoids are an ML program with
  float16 compute precision.
  Both use an iOS 16 and macOS 13 deployment target and form one Core ML pipeline.
- Tools: Python {tools["python"]}, PyTorch {tools["torch"]}, NeMo {tools["nemo_toolkit"]},
  coremltools {tools["coremltools"]}, macOS {tools["macos"]} ({tools["macos_build"]}).
- The conversion follows [FluidInference/mobius]({sf.MOBIUS_REPOSITORY}/tree/{sf.MOBIUS_COMMIT}/{sf.MOBIUS_PATH})
  (Apache-2.0) at commit `{sf.MOBIUS_COMMIT}`.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=sf.DEFAULT_OUTPUT_DIR, help="Destination for artifacts")
    parser.add_argument("--work-dir", type=Path, default=sf.DEFAULT_WORK_DIR, help="Destination for .mlpackage files")
    parser.add_argument("--cache-dir", type=Path, default=sf.DEFAULT_CACHE_DIR, help="Checkpoint download directory")
    args = parser.parse_args()
    output_dir: Path = args.output_dir.resolve()
    work_dir: Path = args.work_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    started = t = time.time()
    timings: dict[str, float] = {}

    def mark(step: str, since: float) -> float:
        now = time.time()
        timings[step] = round(now - since, 1)
        print(f"{step}: {timings[step]} s", flush=True)
        return now

    checkpoint, source = sf.download_checkpoint(args.cache_dir.resolve())
    t = mark("download and check checkpoint", t)
    model = sf.load_model(checkpoint)
    t = mark("load checkpoint", t)

    pre_encoder = convert_pre_encoder(model, trace_inputs())
    t = mark("convert pre-encoder", t)
    head = convert_head(model)
    t = mark("convert head", t)

    pipeline = ct.utils.make_pipeline(pre_encoder, head, compute_units=ct.ComputeUnit.CPU_ONLY)
    describe(pipeline, source)
    check_interface(pipeline)
    package = work_dir / sf.PACKAGE_FILE
    if package.exists():
        shutil.rmtree(package)
    pipeline.save(str(package))
    t = mark("build pipeline", t)
    compile_model(package, output_dir / sf.MODEL_FILE)
    t = mark("compile", t)

    record: dict[str, Any] = {
        "format_version": 1,
        "title": sf.TITLE,
        "created": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "license": sf.LICENSE,
        "attribution": sf.ATTRIBUTION,
        "source": source,
        "derived_from": {
            "repository": sf.MOBIUS_REPOSITORY,
            "commit": sf.MOBIUS_COMMIT,
            "path": sf.MOBIUS_PATH,
            "license": "Apache-2.0",
        },
        "reference_artifacts": {
            "repo": sf.REFERENCE.repo,
            "revision": sf.REFERENCE.revision,
            "file": sf.REFERENCE.file,
        },
        "tools": sf.tool_versions(),
        "conversion": {
            "model": sf.MODEL_FILE,
            "source_dialect": "TorchScript (torch.jit.trace)",
            "model_type": "pipeline of two mlprogram models",
            "minimum_deployment_target": "iOS16 (macOS 13)",
            "components": {
                "pre_encoder": {"compute_precision": "FLOAT32", "weights": "float32"},
                "head": {"compute_precision": "FLOAT16", "weights": "float16"},
            },
            "compiler": "coremltools.utils.compile_model (MLModel.compileModel)",
            "streaming": sf.STREAMING,
            "metadata": dict(pipeline.user_defined_metadata),
            "contract": {
                kind: {name: {"dtype": d, "shape": list(s)} for name, (d, s) in specs.items()}
                for kind, specs in sf.CONTRACT.items()
            },
            "timings_seconds": timings,
        },
    }
    record["files"] = sf.file_inventory(output_dir, exclude=(sf.CONVERSION_FILE, sf.MODEL_CARD_FILE))
    record["total_size"] = sum(entry["size"] for entry in record["files"])
    (output_dir / sf.CONVERSION_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    (output_dir / sf.MODEL_CARD_FILE).write_text(model_card(record), encoding="utf-8")
    print(
        f"wrote {output_dir} ({record['total_size']:,} bytes in {len(record['files'])} files) "
        f"in {time.time() - started:.0f} s"
    )


if __name__ == "__main__":
    main()
