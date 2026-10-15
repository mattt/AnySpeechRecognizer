# SPDX-License-Identifier: Apache-2.0

"""Convert NVIDIA's Streaming Sortformer 4spk v2.1 checkpoint to AnySpeechRecognizer's Core ML model.

Usage:
    uv run python convert.py
    uv run python convert.py --checkpoint /path/to/diar_streaming_sortformer_4spk-v2.1.nemo

The output directory (default: build/artifacts) holds the compiled pipeline, the license,
the notice, the model card, and conversion.json, in the layout of the Hugging Face repository.
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
import numpy as np
import sortformer as sf
import torch

DEPLOYMENT_TARGET = ct.target.iOS17
AUTHOR = "AnySpeechRecognizer"

_DTYPE_NAMES = {
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32: "float32",
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16: "float16",
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32: "int32",
}

INPUT_DESCRIPTIONS = {
    "chunk": "Log-mel features of the chunk with its left and right context, time-major, zero-padded",
    "chunk_lengths": "Number of valid mel frames in chunk",
    "spkcache": "Arrival-order speaker cache embeddings, valid rows first",
    "spkcache_lengths": "Number of valid speaker cache rows",
    "fifo": "FIFO queue embeddings, valid rows first",
    "fifo_lengths": "Number of valid FIFO rows",
}
OUTPUT_DESCRIPTIONS = {
    "speaker_preds": "Speaker probabilities for the valid cache rows, FIFO rows, and chunk rows, then zeros",
    "chunk_pre_encoder_embs_out": "Pre-encoder embeddings of the chunk",
    "chunk_pre_encoder_lengths_out": "Number of valid chunk embedding rows",
}


def tensor(name: str, dtype: str, shape: tuple[int, ...] | None = None) -> ct.TensorType:
    return ct.TensorType(name=name, shape=shape, dtype=np.float32 if dtype == "float32" else np.int32)


def convert(traced, inputs, outputs, precision) -> ct.models.MLModel:
    return ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=DEPLOYMENT_TARGET,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )


def interface(spec) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
    def features(items):
        return {
            f.name: (_DTYPE_NAMES.get(f.type.multiArrayType.dataType, "other"), tuple(f.type.multiArrayType.shape))
            for f in items
        }

    return {"inputs": features(spec.description.input), "outputs": features(spec.description.output)}


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


def example_inputs() -> dict[str, torch.Tensor]:
    """Inputs with partly filled cache and FIFO, used for tracing and the conversion check."""
    generator = torch.Generator().manual_seed(0)
    cache_rows, fifo_rows = 120, 57

    def rows(total: int, valid: int) -> torch.Tensor:
        values = torch.randn(1, total, sf.EMBEDDING_SIZE, generator=generator)
        values[:, valid:] = 0
        return values

    return {
        "chunk": torch.randn(1, sf.CHUNK_MEL_FRAMES, sf.MEL_BINS, generator=generator) * 2 - 6,
        "chunk_lengths": torch.tensor([sf.CHUNK_MEL_FRAMES], dtype=torch.int64),
        "spkcache": rows(sf.STREAMING["spkcache_len"], cache_rows),
        "spkcache_lengths": torch.tensor([cache_rows], dtype=torch.int64),
        "fifo": rows(sf.STREAMING["fifo_len"], fifo_rows),
        "fifo_lengths": torch.tensor([fifo_rows], dtype=torch.int64),
    }


def reference_outputs(model, inputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """NeMo's own export path (`forward_for_export`) on the same inputs, for the conversion check."""
    with torch.inference_mode():
        preds, embeddings, lengths = model.forward_for_export(**inputs)
    rows = sf.PACKED_ROWS
    padded = torch.zeros(1, rows, sf.SPEAKERS)
    padded[:, : preds.shape[1]] = preds
    return {
        "speaker_preds": padded.numpy(),
        "chunk_pre_encoder_embs_out": embeddings.numpy(),
        "chunk_pre_encoder_lengths_out": lengths.numpy(),
    }


def describe(pipeline: ct.models.MLModel) -> None:
    pipeline.short_description = (
        f"{sf.TITLE} speaker diarization, one streaming step with NVIDIA's low-latency configuration"
    )
    pipeline.author = AUTHOR
    pipeline.license = f"{sf.REQUIRED_NOTICE} ({sf.LICENSE_URL})"
    pipeline.version = "2.1"
    metadata = pipeline.user_defined_metadata
    for key, value in sf.STREAMING_METADATA.items():
        metadata[key] = value
    metadata["source_checkpoint"] = f"{sf.SOURCE.repo}@{sf.SOURCE.revision}"
    spec = pipeline._spec
    for feature in spec.description.input:
        feature.shortDescription = INPUT_DESCRIPTIONS[feature.name]
    for feature in spec.description.output:
        feature.shortDescription = OUTPUT_DESCRIPTIONS[feature.name]


def notice_text() -> str:
    return f"""{sf.TITLE} for Core ML

{sf.REQUIRED_NOTICE}.

This repository contains a Derivative Model of {sf.TITLE} by NVIDIA
(https://huggingface.co/{sf.SOURCE.repo}, revision {sf.SOURCE.revision}).
Use, reproduction, and distribution are governed by the {sf.LICENSE_NAME}
in the LICENSE file ({sf.LICENSE_URL}, version released {sf.LICENSE_VERSION}).
Use must be consistent with NVIDIA's Trustworthy AI terms ({sf.TRUSTWORTHY_AI_URL}).

Modifications: {sf.MODIFICATIONS}

The conversion was made with AnySpeechRecognizer's conversion tool
(https://github.com/mattt/AnySpeechRecognizer, Tools/SortformerConversion/v2.1),
which is licensed under the Apache License, Version 2.0.
If you distribute these files, give recipients a copy of the LICENSE file
and include this NOTICE file.
"""


def verification_section(report: dict[str, Any] | None) -> str:
    """Summarize verify.py results for the converted model, or explain how to verify."""
    rows = [r for r in (report or {}).get("results", []) if Path(r["model"]).name == sf.MODEL_FILE]
    if not rows:
        return (
            "`verify.py` in AnySpeechRecognizer's `Tools/SortformerConversion/v2.1` compares this model\n"
            "with NeMo's streaming inference.\n"
        )
    table = "\n".join(
        f"| `{r['audio']}`, {r['seconds']} s | `{r['compute_units']}` | {r['mean_abs_difference']:.1e} / "
        f"{r['max_abs_difference']:.3f} | {100 * r['agreement_at_0.5']:.2f}% | {100 * r['callhome_agreement_10ms']:.2f}% |"
        for r in rows
    )
    return f"""`verify.py` in AnySpeechRecognizer's `Tools/SortformerConversion/v2.1` ran NeMo's
streaming inference of the source checkpoint (float32, CPU) and the same streaming loop
with each model call replaced by this Core ML model, on synthetic multi-speaker recordings.
Agreement compares speaker activity per frame and speaker at a 0.5 threshold, and per 10 ms
after NeMo's post-processing with its CALLHOME parameters for v2.

| Audio | Compute units | Probability difference, mean / max | Agreement at 0.5 | Agreement after post-processing |
| --- | --- | --- | --- | --- |
{table}

The recordings use macOS synthetic voices, so these numbers show conversion fidelity,
not diarization accuracy on real recordings.
"""


def model_card(record: dict[str, Any], report: dict[str, Any] | None = None) -> str:
    tools = record["tools"]
    files = "\n".join(
        f"| `{path}` | {description} |"
        for path, description in (
            (sf.MODEL_FILE, "Compiled Core ML pipeline: float32 pre-encoder and packing, float16 encoder and heads."),
            (sf.LICENSE_FILE, f"The {sf.LICENSE_NAME}."),
            (sf.NOTICE_FILE, "The required notice, the source, and the modifications."),
            (
                sf.CONVERSION_FILE,
                "Source revision and digest, tool versions, settings, and SHA-256 digests of every file.",
            ),
        )
    )
    inputs = "\n".join(
        f"| `{name}` | {dtype} {list(shape)} | {INPUT_DESCRIPTIONS[name]} |"
        for name, (dtype, shape) in sf.CONTRACT["inputs"].items()
    )
    outputs = "\n".join(
        f"| `{name}` | {dtype} {list(shape)} | {OUTPUT_DESCRIPTIONS[name]} |"
        for name, (dtype, shape) in sf.CONTRACT["outputs"].items()
    )
    streaming = ", ".join(f"`{k}` {v}" for k, v in sf.STREAMING.items())
    return f"""---
license: other
license_name: {sf.LICENSE_ID}
license_link: {sf.LICENSE_URL}
base_model: {sf.SOURCE.repo}
base_model_relation: quantized
library_name: coreml
tags: [coreml, speaker-diarization, sortformer, streaming, apple-silicon]
---

# {sf.TITLE} for Core ML

A Core ML conversion of NVIDIA's
[{sf.TITLE}](https://huggingface.co/{sf.SOURCE.repo})
speaker diarization model, made for
[AnySpeechRecognizer](https://github.com/mattt/AnySpeechRecognizer).

## License and attribution

{sf.REQUIRED_NOTICE}.

This model is a Derivative Model of {sf.TITLE} by NVIDIA and is governed by the
[{sf.LICENSE_NAME}]({sf.LICENSE_URL}).
The `LICENSE` file is a copy of the agreement (version released {sf.LICENSE_VERSION}),
and the `NOTICE` file contains the notice that Section 3.1 of the agreement requires.
If you redistribute these files, give recipients a copy of the agreement and include the notice.
Use must be consistent with [NVIDIA's Trustworthy AI terms]({sf.TRUSTWORTHY_AI_URL}),
which prohibit use in violation of applicable law, such as illegal surveillance or illegal
collection or processing of biometric information.

Applications that use this model should show this attribution:

> {sf.ATTRIBUTION}

Modifications: {sf.MODIFICATIONS}

## Intended use

Streaming speaker diarization of 16 kHz mono English speech with up to four speakers,
on Apple silicon.
Each 80 ms frame gets an independent activity probability for each of four speakers,
so overlapping speech is reported.
Speakers are numbered in order of arrival, and a speaker cache keeps the numbering stable
within a stream.

NVIDIA reports these limits for the base model:
accuracy degrades with five or more speakers, on very long recordings,
on non-English speech, and on out-of-domain audio such as noisy recordings.
The conversion does not change them.
Speaker labels are arrival-order slots, not identities;
the model does not recognize who a speaker is.

## Usage from AnySpeechRecognizer

```swift
import AnySpeechRecognizer

let transcriber = SpeechTranscriber(
    model: ParakeetSpeechModel(variant: .v3), attributeOptions: [.audioTimeRange])
let diarizer = SpeakerDiarizer(model: SortformerDiarizationModel(version: .v2_1))
if let request = try await AssetInventory.assetInstallationRequest(
    supporting: [transcriber, diarizer]
) {{
    try await request.downloadAndInstall()
}}
let analyzer = SpeechAnalyzer(modules: [transcriber, diarizer])
```

`AssetInventory` downloads the pinned files of this repository
and verifies their SHA-256 digests; analysis never downloads.
A staged copy of `{sf.MODEL_FILE}` can be used directly:

```swift
let model = SortformerDiarizationModel(
    version: .v2_1, assets: .compiledModel(URL(filePath: "/path/to/{sf.MODEL_FILE}")))
```

The package's command-line tool installs and runs it with:

```sh
hear prepare --backend parakeet --diarize --diarizer-version v2.1
hear transcribe meeting.wav --backend parakeet --diarize --diarizer-version v2.1
```

## Files

| File | Contents |
| --- | --- |
{files}

## Interface

One prediction runs one streaming step.
The caller computes NeMo-compatible 128-bin log-mel features (16 kHz, 25 ms Hann window,
10 ms hop, 512-point FFT, natural log, no normalization) and keeps the speaker cache and
FIFO with NeMo's streaming update.

| Input | Type | Contents |
| --- | --- | --- |
{inputs}

| Output | Type | Contents |
| --- | --- | --- |
{outputs}

The model's creator-defined metadata records the streaming configuration
({streaming}, `subsampling_factor` {sf.SUBSAMPLING}).
The interface is the same as the Streaming Sortformer v2 model that AnySpeechRecognizer
also supports, so the same runtime runs both.

## Conversion

- Source: `{sf.SOURCE.repo}` at revision `{sf.SOURCE.revision}`,
  `{sf.SOURCE.file}` (SHA-256 `{sf.SOURCE.sha256}`).
- Conversion: TorchScript traces of the pre-encoder and of the encoder, Transformer, and
  speaker heads, converted to ML programs with an iOS 17 and macOS 14 deployment target
  and combined into a Core ML pipeline.
  The pre-encoder uses float32; the rest uses float16 weights and compute precision.
- Tools: Python {tools["python"]}, PyTorch {tools["torch"]}, NeMo {tools["nemo_toolkit"]},
  coremltools {tools["coremltools"]}, coremlcompiler {tools["coremlcompiler"]},
  macOS {tools["macos"]}.

## Verification

{verification_section(report)}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="A local copy of the pinned .nemo checkpoint")
    parser.add_argument("--cache-dir", type=Path, default=sf.DEFAULT_CACHE_DIR, help="Checkpoint download directory")
    parser.add_argument("--output-dir", type=Path, default=sf.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=sf.DEFAULT_WORK_DIR)
    parser.add_argument(
        "--update-card",
        action="store_true",
        help="Only rewrite README.md in the output directory from its conversion.json and --verify-report",
    )
    parser.add_argument("--verify-report", type=Path, help="A verify.py report to summarize in the model card")
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    report = json.loads(args.verify_report.read_text()) if args.verify_report else None
    if args.update_card:
        record = json.loads((output_dir / sf.CONVERSION_FILE).read_text())
        (output_dir / sf.MODEL_CARD_FILE).write_text(model_card(record, report), encoding="utf-8")
        print(f"wrote {output_dir / sf.MODEL_CARD_FILE}")
        return
    work_dir: Path = args.work_dir.resolve()
    started = time.time()
    timings: dict[str, float] = {}

    def mark(step: str, since: float) -> float:
        now = time.time()
        timings[step] = round(now - since, 1)
        print(f"{step}: {timings[step]} s", flush=True)
        return now

    t = time.time()
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        if sf.sha256_file(checkpoint) != sf.SOURCE.sha256:
            raise SystemExit(f"error: {checkpoint} is not {sf.SOURCE.repo}@{sf.SOURCE.revision}")
    else:
        checkpoint = sf.download_checkpoint(args.cache_dir.resolve())
    model = sf.load_model(checkpoint)
    t = mark("load checkpoint", t)

    inputs = example_inputs()
    pre_encoder = sf.PreEncoder(model).eval()
    head = sf.Head(model).eval()
    with torch.inference_mode():
        packed, total, _, _ = pre_encoder(**inputs)
    names = list(sf.CONTRACT["inputs"])
    traced_pre_encoder = torch.jit.trace(pre_encoder, tuple(inputs[name] for name in names), strict=False).eval()
    traced_head = torch.jit.trace(head, (packed, total), strict=False).eval()
    t = mark("trace", t)

    pre_encoder_model = convert(
        traced_pre_encoder,
        [tensor(name, *sf.CONTRACT["inputs"][name]) for name in names],
        [
            tensor("pre_encoder_embs", "float32"),
            tensor("pre_encoder_lengths", "int32"),
            tensor("chunk_pre_encoder_embs_out", "float32"),
            tensor("chunk_pre_encoder_lengths_out", "int32"),
        ],
        ct.precision.FLOAT32,
    )
    head_model = convert(
        traced_head,
        [
            tensor("pre_encoder_embs", "float32", (1, sf.PACKED_ROWS, sf.EMBEDDING_SIZE)),
            tensor("pre_encoder_lengths", "int32", (1,)),
        ],
        [tensor("speaker_preds", "float32")],
        ct.precision.FLOAT16,
    )
    t = mark("convert", t)

    pipeline = ct.utils.make_pipeline(pre_encoder_model, head_model, compute_units=ct.ComputeUnit.CPU_ONLY)
    describe(pipeline)
    actual = interface(pipeline.get_spec())
    if actual != sf.CONTRACT:
        raise RuntimeError(f"the pipeline interface {actual} does not match the contract {sf.CONTRACT}")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    package = work_dir / sf.MODEL_FILE.replace(".mlmodelc", ".mlpackage")
    pipeline.save(str(package))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    compile_model(package, output_dir / sf.MODEL_FILE)
    t = mark("save and compile", t)

    # Compare the compiled model with NeMo's export path on the tracing inputs.
    compiled = ct.models.CompiledMLModel(str(output_dir / sf.MODEL_FILE), compute_units=ct.ComputeUnit.CPU_ONLY)
    feed = {name: inputs[name].numpy().astype(np.int32 if name.endswith("_lengths") else np.float32) for name in names}
    produced = compiled.predict(feed)
    expected = reference_outputs(model, inputs)
    check = {
        name: float(np.max(np.abs(produced[name].astype(np.float64) - expected[name].astype(np.float64))))
        for name in expected
    }
    print("conversion check, maximum absolute difference (CPU_ONLY):", check)
    if check["chunk_pre_encoder_lengths_out"] != 0 or check["speaker_preds"] > 0.05:
        raise RuntimeError(f"the compiled model differs from NeMo's export path: {check}")
    t = mark("check", t)

    shutil.copyfile(sf.TOOL_DIR / "NVIDIA-Open-Model-License.txt", output_dir / sf.LICENSE_FILE)
    (output_dir / sf.NOTICE_FILE).write_text(notice_text(), encoding="utf-8")

    record: dict[str, Any] = {
        "format_version": 1,
        "title": sf.TITLE,
        "created": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "license": {
            "name": sf.LICENSE_NAME,
            "id": sf.LICENSE_ID,
            "url": sf.LICENSE_URL,
            "pdf": sf.LICENSE_PDF_URL,
            "version": sf.LICENSE_VERSION,
            "required_notice": sf.REQUIRED_NOTICE,
            "trustworthy_ai_terms": sf.TRUSTWORTHY_AI_URL,
        },
        "attribution": sf.ATTRIBUTION,
        "modifications": sf.MODIFICATIONS,
        "source": {
            "repo": sf.SOURCE.repo,
            "revision": sf.SOURCE.revision,
            "file": sf.SOURCE.file,
            "size": sf.SOURCE.size,
            "sha256": sf.SOURCE.sha256,
        },
        "tools": sf.tool_versions(),
        "conversion": {
            "source_dialect": "TorchScript (torch.jit.trace)",
            "model_type": "pipeline of two mlprogram models",
            "minimum_deployment_target": "iOS17 (macOS 14)",
            "stages": [
                {
                    "name": "pre-encoder",
                    "modules": "encoder.pre_encode and packing of cache, FIFO, and chunk rows",
                    "compute_precision": "FLOAT32",
                },
                {
                    "name": "head",
                    "modules": "Fast Conformer encoder (bypass_pre_encode), encoder_proj, Transformer encoder, speaker sigmoids",
                    "compute_precision": "FLOAT16",
                },
            ],
            "streaming": sf.STREAMING,
            "metadata": sf.STREAMING_METADATA,
            "preprocessor": {
                "included": False,
                "sample_rate": 16_000,
                "features": sf.MEL_BINS,
                "window_size": 0.025,
                "window_stride": 0.01,
                "n_fft": 512,
                "normalize": "NA",
                "dither": 0.0,
            },
            "contract": {
                kind: {name: {"dtype": dtype, "shape": list(shape)} for name, (dtype, shape) in specs.items()}
                for kind, specs in sf.CONTRACT.items()
            },
            "check_against_forward_for_export": {"compute_units": "CPU_ONLY", "max_abs_difference": check},
            "timings_seconds": timings,
        },
    }
    record["files"] = sf.file_inventory(output_dir, exclude=(sf.CONVERSION_FILE, sf.MODEL_CARD_FILE))
    record["total_size"] = sum(entry["size"] for entry in record["files"])
    (output_dir / sf.CONVERSION_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    (output_dir / sf.MODEL_CARD_FILE).write_text(model_card(record, report), encoding="utf-8")
    print(f"wrote {output_dir} ({record['total_size'] / 1e6:.1f} MB) in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
