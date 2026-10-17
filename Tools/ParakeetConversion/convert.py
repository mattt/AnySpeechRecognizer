# SPDX-License-Identifier: Apache-2.0
#
# The conversion steps follow FluidInference/mobius
# (models/stt/parakeet-tdt-v3-0.6b/coreml/convert-parakeet.py,
# models/stt/parakeet-redux/coreml/export_decoder_joint.py, and
# models/stt/parakeet-ultra/coreml/quantize_encoder_int8.py at commit
# 26f5fa8059d5ab3d34449adce6ff6ed86734e522), Copyright Fluid Inference,
# licensed under the Apache License, Version 2.0. See NOTICE in this directory.

"""Convert a Parakeet TDT or RNN-T checkpoint into AnySpeechRecognizer's Core ML artifacts.

Usage:
    uv run python convert.py --variant v3
    uv run python convert.py --variant ultra --output-dir build/ultra
    uv run python convert.py --variant v3 --encoder-windows 4 6 8 --output-dir build/v3-live

`--encoder-windows` adds encoders for other fixed windows, named `Encoder<seconds>s.mlmodelc`,
next to the 15-second `Encoder.mlmodelc`. The decoder and joint decision are shared.
Windows longer than 15 seconds, an experiment for file transcription, use NeMo's local attention.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch

import parakeet as pk

DEPLOYMENT_TARGET = ct.target.iOS17
ENCODER_QUANTIZATION = {
    "mode": "linear_symmetric",
    "dtype": "int8",
    "granularity": "per_channel",
    "weight_threshold": 2048,
}
AUTHOR = "AnySpeechRecognizer"


def _computes_in_float16(op: Any) -> bool:
    """Select every operation for float16 except length arithmetic and comparisons.

    Float16 represents integers exactly only up to 2,048. In windows longer than 20 seconds,
    the mel frame counts and positions exceed that, and float16 length arithmetic gave
    `encoder_length` and the padding masks one or two frames too many or too few.
    """
    outputs = list(op.outputs)
    if any(str(output.dtype).endswith("bool") for output in outputs):
        return False
    return not (outputs and all(output.rank <= 1 for output in outputs))


def convert_traced(traced: torch.jit.ScriptModule, inputs, outputs, float32_lengths: bool = False) -> ct.models.MLModel:
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


def tensor_type(name: str, contract: tuple[str, tuple[int, ...]]) -> ct.TensorType:
    dtype, shape = contract
    return ct.TensorType(name=name, shape=shape, dtype=np.float32 if dtype == "float32" else np.int32)


def output_types(component: str) -> list[ct.TensorType]:
    return [
        ct.TensorType(name=name, dtype=np.float32 if dtype == "float32" else np.int32)
        for name, (dtype, _) in pk.CONTRACT[component]["outputs"].items()
    ]


def input_types(component: str, contract: dict | None = None) -> list[ct.TensorType]:
    contract = contract or pk.CONTRACT[component]
    return [tensor_type(name, spec) for name, spec in contract["inputs"].items()]


def describe(model: ct.models.MLModel, variant: pk.Variant, description: str) -> None:
    model.short_description = f"{variant.title} {description}"
    model.author = AUTHOR
    model.license = f"{variant.license}. {variant.attribution}"
    model.version = "1"
    sources = [variant.base] + ([variant.weights] if variant.weights else [])
    model.user_defined_metadata["source_checkpoints"] = ", ".join(f"{s.repo}@{s.revision}" for s in sources)


_DTYPE_NAMES = {
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32: "float32",
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32: "int32",
}


def check_interface(model: ct.models.MLModel, component: str, contract: dict | None = None) -> None:
    """Fail unless the model's inputs and outputs match the contract exactly."""
    contract = contract or pk.CONTRACT[component]
    spec = model.get_spec()
    for kind, features in (("inputs", spec.description.input), ("outputs", spec.description.output)):
        actual = {
            f.name: (_DTYPE_NAMES.get(f.type.multiArrayType.dataType, "other"), tuple(f.type.multiArrayType.shape))
            for f in features
        }
        expected = contract[kind]
        if actual != expected:
            raise RuntimeError(f"{component} {kind} {actual} do not match the contract {expected}")


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


def set_encoder_attention(model: Any, window: pk.EncoderWindow) -> None:
    """Select full attention for windows up to 15 s and local attention for longer windows.

    The model keeps its weights. Windows of up to 256 encoder frames (20.48 s) give the same
    result either way, so the attention changes only when a longer window is exported.
    A unified model's chunked attention style describes full attention as [-1, -1, -1].
    """
    current = list(model.encoder.att_context_size)
    if len(current) == 3 and window.uses_local_attention:
        raise SystemExit("windows longer than 15 seconds need two-sided local attention, which this model lacks")
    full = [-1] * len(current)
    context = list(pk.LOCAL_ATTENTION_CONTEXT) if window.uses_local_attention else full
    if current != context:
        model.change_attention_model(self_attention_model="rel_pos", att_context_size=context)
        model.eval()
    # NeMo extends its positional encoding when the input has more than 5,000 mel frames.
    # Extend it here, outside torch.inference_mode, so that the trace can use it.
    if window.mel_frames > model.encoder.max_audio_length:
        model.encoder.set_max_audio_length(window.mel_frames)


def export_encoder(
    model: Any,
    encoder: torch.nn.Module,
    window: pk.EncoderWindow,
    variant: pk.Variant,
    work_dir: Path,
    on_converted: Callable[[], object] | None = None,
) -> tuple[Path, torch.Tensor]:
    """Trace, convert, and quantize the encoder for one fixed window.

    Returns the saved .mlpackage and the float32 PyTorch encoder output for the trace input.
    """
    set_encoder_attention(model, window)
    # Trace inputs: deterministic noise through NeMo's own preprocessor.
    generator = torch.Generator().manual_seed(0)
    audio = (0.05 * torch.randn(window.samples, generator=generator)).numpy().astype(np.float32)
    mel, mel_length = pk.nemo_mel(model, audio, window.samples, mel_frames=window.mel_frames)
    mel_t, mel_length_t = torch.from_numpy(mel), torch.from_numpy(mel_length)
    contract = window.contract()

    # Encoder: float16 conversion, then int8 per-channel weight quantization (Encoder_v2 encoding).
    with torch.inference_mode():
        encoder_ref, _ = encoder(mel_t, mel_length_t)
    if tuple(encoder_ref.shape) != (1, pk.ENCODER_DIM, window.encoder_frames):
        raise RuntimeError(f"unexpected encoder shape {tuple(encoder_ref.shape)}")
    traced = torch.jit.trace(encoder, (mel_t, mel_length_t), strict=False).eval()
    encoder_fp16 = convert_traced(
        traced,
        input_types("encoder", contract),
        output_types("encoder"),
        float32_lengths=window.uses_local_attention,
    )
    if on_converted is not None:
        on_converted()
    encoder_int8 = cto.linear_quantize_weights(
        encoder_fp16,
        config=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(**ENCODER_QUANTIZATION)),
    )
    del encoder_fp16
    describe(encoder_int8, variant, f"encoder, {window.seconds} s window, int8 per-channel weights")
    check_interface(encoder_int8, "encoder", contract)
    package = work_dir / window.package
    encoder_int8.save(str(package))
    return package, encoder_ref


def window_record(window: pk.EncoderWindow) -> dict[str, Any]:
    return {
        "file": window.file,
        "window_seconds": window.seconds,
        "window_samples": window.samples,
        "mel_frames": window.mel_frames,
        "encoder_frames": window.encoder_frames,
        "whole_encoder_frames": window.whole_frames,
        "weights": "int8 linear quantization",
        "attention_context_frames": list(pk.LOCAL_ATTENTION_CONTEXT) if window.uses_local_attention else None,
        "contract": {
            kind: {name: {"dtype": d, "shape": list(s)} for name, (d, s) in specs.items()}
            for kind, specs in window.contract().items()
        },
    }


def notice_text(variant: pk.Variant) -> str:
    """Return the NOTICE file of a variant whose license requires a notice with every copy.

    The only such license is the NVIDIA Open Model License; Section 3.1 requires the notice.
    """
    return f"""{variant.title} for Core ML

{variant.required_notice}.

This repository contains a Derivative Model of {variant.title} by NVIDIA
(https://huggingface.co/{variant.base.repo}, revision {variant.base.revision}).
Use, reproduction, and distribution are governed by the {variant.license_name}
in the LICENSE file ({variant.license_url}, version released {pk.NVIDIA_OPEN_MODEL_LICENSE_VERSION}).
Use must be consistent with NVIDIA's Trustworthy AI terms ({pk.NVIDIA_TRUSTWORTHY_AI_URL}).

Modifications: {variant.modifications}

The conversion was made with AnySpeechRecognizer's conversion tool
(https://github.com/mattt/AnySpeechRecognizer, Tools/ParakeetConversion),
which is licensed under the Apache License, Version 2.0.
If you distribute these files, give recipients a copy of the LICENSE file
and include this NOTICE file.
"""


def license_section(variant: pk.Variant) -> str:
    """Return the model card's license and attribution text."""
    if not variant.required_notice:
        return f"""{variant.attribution}

This repository is licensed under
[{variant.license_name}]({variant.license_url}), the license of the
base model. Redistributions must keep this attribution."""
    return f"""{variant.required_notice}.

This model is a Derivative Model of {variant.title} by NVIDIA and is governed by the
[{variant.license_name}]({variant.license_url}).
The `LICENSE` file is a copy of the agreement (version released {pk.NVIDIA_OPEN_MODEL_LICENSE_VERSION}),
and the `NOTICE` file contains the notice that Section 3.1 of the agreement requires.
If you redistribute these files, give recipients a copy of the agreement and include the notice.
Use must be consistent with [NVIDIA's Trustworthy AI terms]({pk.NVIDIA_TRUSTWORTHY_AI_URL}),
which prohibit use in violation of applicable law, such as illegal surveillance or illegal
collection or processing of biometric information.

Applications that use this model should show this attribution:

> {variant.attribution}

Modifications: {variant.modifications}"""


def model_card(variant: pk.Variant, record: dict[str, Any]) -> str:
    """Return a Hugging Face model card for the output directory."""
    base_model = variant.weights.repo if variant.weights else variant.base.repo
    decision = "TDT" if variant.is_tdt else "RNN-T"
    tools = record["tools"]
    sources = "\n".join(
        f"  - `{s['repo']}` at revision `{s['revision']}`: `{s['file']}` (SHA-256 `{s['sha256']}`), {s['role']}."
        for s in record["sources"]
    )
    files = "\n".join(
        f"| `{name}` | {description} |"
        for name, description in (
            (pk.ENCODER_FILE, "FastConformer encoder for one 15-second window; int8 per-channel weights."),
            *(
                (
                    window["file"],
                    (
                        f"FastConformer encoder for one {window['window_seconds']}-second window "
                        f"({window['mel_frames']} mel frames, {window['encoder_frames']} encoder frames); "
                        "int8 per-channel weights."
                    ),
                )
                for window in record["conversion"]["components"].get("encoder_windows", [])
            ),
            (pk.DECODER_FILE, "Prediction network (two-layer LSTM), one step with explicit state."),
            (
                pk.JOINT_FILE,
                f"Joint network and greedy {decision} decision for one encoder frame, with the top 64 tokens.",
            ),
            (pk.VOCAB_FILE, "Token ID to SentencePiece piece, with the word-boundary marker written as a space."),
            (pk.CONVERSION_FILE, "Sources, tool versions, settings, and SHA-256 digests of every file."),
            *(((pk.LICENSE_FILE, f"The {variant.license_name}."),) if variant.license_text else ()),
            *(
                ((pk.NOTICE_FILE, "The required notice, the source, and the modifications."),)
                if variant.required_notice
                else ()
            ),
        )
    )
    notes = "".join(f"- {note}\n" for note in variant.notes)
    notes = f"\n## Notes\n\n{notes}" if notes else ""
    shorter_windows = ""
    if record["conversion"]["components"].get("encoder_windows"):
        shorter_windows = (
            "\nEach `Encoder<seconds>s.mlmodelc` is the same encoder traced for another fixed window,\n"
            "with the same trained weights, input and output names, and 80 ms encoder frames.\n"
            "It shares the decoder and joint decision.\n"
        )
        if any(w.get("attention_context_frames") for w in record["conversion"]["components"]["encoder_windows"]):
            shorter_windows += (
                "Encoders for windows longer than 15 seconds limit self-attention to 256 encoder frames\n"
                "(20.48 s) on each side, as NeMo's `rel_pos_local_attn` attention with context [256, 256] does.\n"
            )
    return f"""---
{variant.card_license}
base_model: {base_model}
base_model_relation: quantized
language: [{variant.languages}]
pipeline_tag: automatic-speech-recognition
tags: [coreml, parakeet, {decision.lower()}, speech-recognition, apple-silicon]
---

# {variant.title} for Core ML

Core ML components of [{base_model}](https://huggingface.co/{base_model})
for [AnySpeechRecognizer](https://github.com/mattt/AnySpeechRecognizer).

## Attribution and license

{license_section(variant)}
{notes}
## Files

| File | Contents |
| --- | --- |
{files}

The models take NeMo-compatible 128-bin log-mel features for a 15-second window
(1,501 frames) and produce 188 encoder frames of 80 ms each.
The features follow NeMo 2.5 and later: zero padding at the window ends and
`floor(samples / 160)` valid frames in `mel_length`.
The encoder masks frames after `mel_length` between its subsampling convolutions.
The vocabulary has {variant.vocab_size:,} pieces, and the blank token is {variant.blank_id:,}.
The tensor interface matches FluidInference's `Encoder_v2.mlmodelc`,
`Decoder.mlmodelc`, and `JointDecisionv3.mlmodelc` artifacts.
The preprocessor is not included.
{shorter_windows}
## Conversion

- Sources:
{sources}
- Conversion: TorchScript trace to an ML program with an iOS 17 and macOS 14
  deployment target and float16 compute precision.
- Encoder weights: int8 linear symmetric quantization, one scale per output
  channel, for weights with at least 2,048 elements.
- Tools: Python {tools["python"]}, PyTorch {tools["torch"]},
  NeMo {tools["nemo_toolkit"]}, coremltools {tools["coremltools"]},
  coremlcompiler {tools["coremlcompiler"]}, macOS {tools["macos"]}.
- The conversion follows [FluidInference/mobius]({pk.MOBIUS_REPOSITORY})
  (Apache-2.0) at commit `{pk.MOBIUS_COMMIT}`.
"""


def add_windows(
    variant: pk.Variant,
    windows: list[pk.EncoderWindow],
    base: Path,
    output_dir: Path,
    work_dir: Path,
    cache_dir: Path,
) -> None:
    """Write `base` plus encoders for additional windows to `output_dir`.

    The files of `base` are copied unchanged, so a repository that already publishes
    them can add the new encoders without changing existing files. Converting again
    would not reproduce them byte for byte: each `coremldata.bin` records the conversion
    date and stores user-defined metadata in an unspecified order.
    """
    if output_dir == base:
        raise SystemExit("--output-dir must differ from --reuse-from")
    record = json.loads((base / pk.CONVERSION_FILE).read_text())
    if record.get("variant") != variant.name:
        raise SystemExit(f"{base} contains variant {record.get('variant')!r}, not {variant.name!r}")
    listed = {entry["path"]: (entry["size"], entry["sha256"]) for entry in record["files"]}
    actual = {
        entry["path"]: (entry["size"], entry["sha256"])
        for entry in pk.file_inventory(base, exclude=(pk.CONVERSION_FILE, pk.MODEL_CARD_FILE))
    }
    if listed != actual:
        raise SystemExit(f"{base}: files do not match {pk.CONVERSION_FILE}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(base, output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    started = t = time.time()
    model, _, _ = pk.load_model(variant, cache_dir)
    timings["load checkpoint"] = round(time.time() - t, 1)
    encoder = pk.EncoderWrapper(model.encoder.eval())
    for window in windows:
        t = time.time()
        package, _ = export_encoder(model, encoder, window, variant, work_dir)
        compile_model(package, output_dir / window.file)
        name = window.file.removesuffix(".mlmodelc")
        timings[f"convert, quantize, and compile {name}"] = round(time.time() - t, 1)
        print(f"[{variant.name}] {name}: {timings[f'convert, quantize, and compile {name}']} s", flush=True)

    components = record["conversion"]["components"]
    existing = {w["file"]: w for w in components.get("encoder_windows", [])}
    existing.update({w.file: window_record(w) for w in windows})
    components["encoder_windows"] = sorted(existing.values(), key=lambda w: w["window_seconds"])
    record["encoder_windows_added"] = {
        "created": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "files": [w.file for w in windows],
        # A path relative to this directory, because conversion.json is published.
        "other_files_copied_from": base.relative_to(pk.TOOL_DIR).as_posix()
        if base.is_relative_to(pk.TOOL_DIR)
        else base.name,
        "tools": pk.tool_versions(),
        "timings_seconds": timings,
    }
    record["files"] = pk.file_inventory(output_dir, exclude=(pk.CONVERSION_FILE, pk.MODEL_CARD_FILE))
    record["total_size"] = sum(entry["size"] for entry in record["files"])
    (output_dir / pk.CONVERSION_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    (output_dir / pk.MODEL_CARD_FILE).write_text(model_card(variant, record), encoding="utf-8")
    print(f"[{variant.name}] wrote {output_dir} ({record['total_size'] / 1e6:.1f} MB) in {time.time() - started:.0f} s")
    for window in windows:
        print(f"  {window.file}: {pk.directory_size(output_dir / window.file)} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(pk.VARIANTS), required=True)
    parser.add_argument("--output-dir", type=Path, help="Destination for compiled artifacts (default: build/<variant>)")
    parser.add_argument(
        "--work-dir", type=Path, help="Destination for .mlpackage files (default: build/work/<variant>)"
    )
    parser.add_argument("--cache-dir", type=Path, default=pk.DEFAULT_CACHE_DIR, help="Checkpoint download directory")
    parser.add_argument(
        "--encoder-windows",
        type=int,
        nargs="+",
        default=[],
        metavar="SECONDS",
        help=(
            "Also export encoders for these fixed windows (Encoder<seconds>s.mlmodelc), in whole seconds "
            "from 2 to 14, or experimentally from 16 to 120 with local attention"
        ),
    )
    parser.add_argument(
        "--reuse-from",
        type=Path,
        metavar="DIRECTORY",
        help=(
            "Copy the 15-second encoder, decoder, joint decision, and vocabulary from an earlier output "
            "directory after checking their digests, and export only --encoder-windows"
        ),
    )
    args = parser.parse_args()
    if any(not 2 <= seconds <= 120 or seconds == pk.DEFAULT_WINDOW.seconds for seconds in args.encoder_windows):
        parser.error("--encoder-windows takes whole seconds from 2 to 14 or from 16 to 120")
    windows = [pk.EncoderWindow(seconds) for seconds in sorted(set(args.encoder_windows))]
    if args.reuse_from and not windows:
        parser.error("--reuse-from requires --encoder-windows")

    variant = pk.VARIANTS[args.variant]
    output_dir: Path = (args.output_dir or pk.BUILD_DIR / variant.name).resolve()
    work_dir: Path = (args.work_dir or pk.BUILD_DIR / "work" / variant.name).resolve()
    if args.reuse_from:
        add_windows(variant, windows, args.reuse_from.resolve(), output_dir, work_dir, args.cache_dir.resolve())
        return
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    timings: dict[str, float] = {}

    def mark(step: str, since: float) -> float:
        now = time.time()
        timings[step] = round(now - since, 1)
        print(f"[{variant.name}] {step}: {timings[step]} s", flush=True)
        return now

    t = time.time()
    model, sources, weight_details = pk.load_model(variant, args.cache_dir.resolve())
    t = mark("load checkpoint", t)

    vocab_size = int(model.tokenizer.vocab_size)
    num_durations = int(model.joint.num_extra_outputs)
    durations = list(model.cfg.model_defaults.get("tdt_durations", []))
    expected = (variant.vocab_size, variant.blank_id, variant.durations, len(variant.durations))
    if (vocab_size, int(model.decoder.blank_idx), tuple(durations), num_durations) != expected:
        raise RuntimeError(
            f"unexpected model: vocab {vocab_size}, blank {model.decoder.blank_idx}, durations {durations}"
        )
    if (int(model.decoder.pred_hidden), int(model.decoder.pred_rnn_layers)) != (pk.DECODER_DIM, pk.DECODER_LAYERS):
        raise RuntimeError("unexpected prediction network size")

    encoder = pk.EncoderWrapper(model.encoder.eval())
    decoder = pk.DecoderWrapper(model.decoder.eval())
    joint = pk.JointWrapper(model.joint.eval())
    joint_decision = pk.JointDecisionSingleStep(joint, vocab_size=vocab_size, num_durations=num_durations)

    packages: dict[str, Path] = {}

    converted_at: list[float] = []
    packages["encoder"], encoder_ref = export_encoder(
        model,
        encoder,
        pk.DEFAULT_WINDOW,
        variant,
        work_dir,
        on_converted=lambda: converted_at.append(mark("convert encoder", t)),
    )
    t = mark("quantize encoder", converted_at[0])

    # Additional fixed windows share the decoder and joint decision.
    for window in windows:
        name = window.file.removesuffix(".mlmodelc")
        packages[name], _ = export_encoder(model, encoder, window, variant, work_dir)
        t = mark(f"convert and quantize {name}", t)

    # Prediction network with explicit LSTM state.
    export_flag = getattr(model.decoder, "_rnnt_export", False)
    model.decoder._rnnt_export = True
    try:
        targets = torch.full((1, 1), variant.blank_id, dtype=torch.int32)
        target_length = torch.tensor([1], dtype=torch.int32)
        zero_state = torch.zeros(pk.DECODER_LAYERS, 1, pk.DECODER_DIM)
        with torch.inference_mode():
            decoder_ref, _, _ = decoder(targets, target_length, zero_state, zero_state)
        decoder_ref = decoder_ref.clone()
        traced = torch.jit.trace(decoder, (targets, target_length, zero_state, zero_state), strict=False).eval()
        decoder_model = convert_traced(traced, input_types("decoder"), output_types("decoder"))
        describe(decoder_model, variant, "prediction network (RNN-T decoder), one step with explicit LSTM state")
        check_interface(decoder_model, "decoder")
        packages["decoder"] = work_dir / "Decoder.mlpackage"
        decoder_model.save(str(packages["decoder"]))
    finally:
        model.decoder._rnnt_export = export_flag
    t = mark("convert decoder", t)

    # Joint network and greedy decision for one frame.
    encoder_step = encoder_ref[:, :, :1].clone().contiguous()
    decoder_step = decoder_ref[:, :, :1].contiguous()
    traced = torch.jit.trace(joint_decision, (encoder_step, decoder_step), strict=False).eval()
    joint_model = convert_traced(traced, input_types("joint"), output_types("joint"))
    decision = "TDT" if variant.is_tdt else "RNN-T"
    describe(joint_model, variant, f"joint network and greedy {decision} decision for one frame (top-K 64)")
    check_interface(joint_model, "joint")
    packages["joint"] = work_dir / "JointDecisionv3.mlpackage"
    joint_model.save(str(packages["joint"]))
    t = mark("convert joint decision", t)

    compiled_names = {"encoder": pk.ENCODER_FILE, "decoder": pk.DECODER_FILE, "joint": pk.JOINT_FILE}
    for component, package in packages.items():
        compile_model(package, output_dir / compiled_names.get(component, f"{component}.mlmodelc"))
    t = mark("compile", t)

    (output_dir / pk.VOCAB_FILE).write_text(pk.vocabulary_json(model), encoding="utf-8")
    if variant.license_text:
        shutil.copyfile(variant.license_text, output_dir / pk.LICENSE_FILE)
    if variant.required_notice:
        (output_dir / pk.NOTICE_FILE).write_text(notice_text(variant), encoding="utf-8")

    record: dict[str, Any] = {
        "format_version": 1,
        "variant": variant.name,
        "title": variant.title,
        "created": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "license": variant.license,
        "attribution": variant.attribution,
        "license_files": {
            "license": pk.LICENSE_FILE,
            "name": variant.license_name,
            "url": variant.license_url,
            "version": pk.NVIDIA_OPEN_MODEL_LICENSE_VERSION,
            "notice": pk.NOTICE_FILE,
            "required_notice": variant.required_notice,
            "modifications": variant.modifications,
        }
        if variant.license_text
        else None,
        "notes": list(variant.notes),
        "sources": sources,
        "weight_loading": weight_details or None,
        "derived_from": {
            "repository": pk.MOBIUS_REPOSITORY,
            "commit": pk.MOBIUS_COMMIT,
            "license": "Apache-2.0",
        },
        "reference_artifacts": {
            "repo": variant.reference.repo,
            "revision": variant.reference.revision,
            "components": variant.reference_components,
        }
        if variant.reference
        else None,
        "tools": pk.tool_versions(),
        "conversion": {
            "source_dialect": "TorchScript (torch.jit.trace)",
            "model_type": "mlprogram",
            "minimum_deployment_target": "iOS17 (macOS 14)",
            "compute_precision": "FLOAT16",
            "window_seconds": pk.WINDOW_SAMPLES / pk.SAMPLE_RATE,
            "mel_features": {
                "preprocessor": "NeMo AudioToMelSpectrogramPreprocessor",
                "stft_padding": "zeros",
                "valid_frames": "floor(valid samples / 160)",
                "after_valid_frames": "zeros",
            },
            "components": {
                "encoder": {
                    "file": pk.ENCODER_FILE,
                    "weights": "int8 linear quantization",
                    "quantization": {
                        "api": "coremltools.optimize.coreml.linear_quantize_weights",
                        **ENCODER_QUANTIZATION,
                    },
                },
                "decoder": {"file": pk.DECODER_FILE, "weights": "float16"},
                "joint": {"file": pk.JOINT_FILE, "weights": "float16", "top_k": pk.TOP_K},
                "vocabulary": {"file": pk.VOCAB_FILE, "pieces": vocab_size, "blank_id": variant.blank_id},
            },
            "contract": {
                component: {
                    kind: {name: {"dtype": d, "shape": list(s)} for name, (d, s) in specs.items()}
                    for kind, specs in parts.items()
                }
                for component, parts in pk.CONTRACT.items()
            },
            "decoder_type": "TDT" if variant.is_tdt else "RNN-T",
            "tdt_durations": list(variant.durations),
            "timings_seconds": timings,
        },
    }
    if windows:
        record["conversion"]["components"]["encoder_windows"] = [window_record(w) for w in windows]
    record["files"] = pk.file_inventory(output_dir, exclude=(pk.CONVERSION_FILE, pk.MODEL_CARD_FILE))
    record["total_size"] = sum(entry["size"] for entry in record["files"])
    (output_dir / pk.CONVERSION_FILE).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    (output_dir / pk.MODEL_CARD_FILE).write_text(model_card(variant, record), encoding="utf-8")

    print(f"[{variant.name}] wrote {output_dir} ({record['total_size'] / 1e6:.1f} MB) in {time.time() - started:.0f} s")
    for name in (pk.ENCODER_FILE, *(w.file for w in windows), pk.DECODER_FILE, pk.JOINT_FILE, pk.VOCAB_FILE):
        print(
            f"  {name}: {pk.directory_size(output_dir / name) if (output_dir / name).is_dir() else (output_dir / name).stat().st_size} bytes"
        )


if __name__ == "__main__":
    main()
