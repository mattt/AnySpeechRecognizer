# SPDX-License-Identifier: Apache-2.0
#
# coreml_streaming follows SortformerEncLabelModel.forward_streaming_step in NVIDIA NeMo
# (https://github.com/NVIDIA/NeMo), Copyright (c) 2025 NVIDIA CORPORATION,
# licensed under the Apache License, Version 2.0. See NOTICE in this directory.

"""Verify the converted Sortformer model against NeMo and FluidInference's Core ML model.

Usage:
    uv run python verify.py --audio two-speakers.wav four-speakers.wav overlap.wav long.wav
    uv run python verify.py --audio a.wav --reference-model <SortformerNvidiaLow_v2.mlmodelc> \
        --compute-units CPU_AND_NE CPU_ONLY --nemo-reference-dir /tmp/diar-reference

Checks:

1. Interface: the inputs, outputs, and metadata that Core ML reports for both compiled models,
   with build/describe_model (describe_model.swift).
2. Weights: every weight tensor of both models, with compare_weights.py.
3. Parity: NeMo's float32 streaming inference (forward_streaming) on each file, and the same
   streaming loop with each Core ML model in place of the PyTorch modules. Both use NeMo's mel
   features and NeMo's speaker cache and FIFO update (streaming_update), so the differences come
   from the models alone. Each Core ML step is also compared with the PyTorch step on the same
   inputs.

The NeMo features and predictions are written to build/reference/<name>.mel.f32 and
<name>.preds.f32, in the format of Tools/sortformer-reference.py, for the Swift opt-in tests.
The report is build/reports/verify.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import soundfile
import torch

import compare_weights
import sortformer as sf

# NeMo's post-processing thresholds tuned on CALLHOME for this model (PostProcessing.callHome).
ONSET, OFFSET = 0.641, 0.561
COMPUTE_UNITS = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "ALL": ct.ComputeUnit.ALL,
}
REQUIRED_METADATA = (
    "chunk_len",
    "chunk_left_context",
    "chunk_right_context",
    "fifo_len",
    "spkcache_len",
    "spkcache_update_period",
    "subsampling_factor",
)


# MARK: - Interface


def describe_tool() -> Path:
    tool = sf.BUILD_DIR / "describe_model"
    source = sf.TOOL_DIR / "describe_model.swift"
    if not tool.exists() or tool.stat().st_mtime < source.stat().st_mtime:
        tool.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["xcrun", "swiftc", "-O", str(source), "-o", str(tool)], check=True)
    return tool


def describe(model: Path) -> dict[str, Any]:
    output = subprocess.run([str(describe_tool()), str(model)], check=True, capture_output=True, text=True).stdout
    return json.loads(output)


def check_interface(ours: Path, reference: Path) -> dict[str, Any]:
    a, b = describe(ours), describe(reference)
    contract = {
        kind: {name: {"dtype": d, "shape": list(s)} for name, (d, s) in specs.items()}
        for kind, specs in sf.CONTRACT.items()
    }
    ours_metadata = a["metadata"].get("MLModelCreatorDefinedKey", {})
    reference_metadata = b["metadata"].get("MLModelCreatorDefinedKey", {})
    return {
        "ours": a,
        "reference": b,
        "inputs_match_reference": a["inputs"] == b["inputs"],
        "outputs_match_reference": a["outputs"] == b["outputs"],
        "matches_contract": a["inputs"] == contract["inputs"] and a["outputs"] == contract["outputs"],
        "runtime_metadata_matches_reference": all(
            ours_metadata.get(key) == reference_metadata.get(key) for key in REQUIRED_METADATA
        ),
        "metadata_keys_only_in_reference": sorted(set(reference_metadata) - set(ours_metadata)),
    }


# MARK: - Streaming


def nemo_features(model: Any, audio: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    signal = torch.from_numpy(audio).unsqueeze(0)
    length = torch.tensor([signal.shape[1]])
    with torch.inference_mode():
        features, feature_length = model.process_signal(audio_signal=signal, audio_signal_length=length)
    return features[:, :, : int(feature_length.max())], feature_length


def nemo_streaming(model: Any, features: torch.Tensor, feature_length: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        return model.forward_streaming(features, feature_length)[0].numpy()


def padded(tensor: torch.Tensor, rows: int) -> np.ndarray:
    """Copy a [1, n, d] tensor into the start of a zero [1, rows, d] float32 array."""
    out = np.zeros((1, rows, tensor.shape[2]), dtype=np.float32)
    out[:, : tensor.shape[1]] = tensor.numpy()
    return out


def coreml_streaming(
    model: Any, coreml: ct.models.CompiledMLModel, features: torch.Tensor, feature_length: torch.Tensor
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run NeMo's synchronous streaming loop with a Core ML model for each step.

    Returns the chunk predictions and per-step statistics: the Core ML step against NeMo's
    PyTorch step (forward_for_export) on identical inputs, and the time per Core ML call.
    """
    modules = model.sortformer_modules
    state = modules.init_streaming_state(batch_size=1, async_streaming=False, device=torch.device("cpu"))
    total = []
    step_max, step_sum, step_count, seconds = 0.0, 0.0, 0, []
    loader = modules.streaming_feat_loader(
        feat_seq=features, feat_seq_length=feature_length, feat_seq_offset=torch.zeros(1, dtype=torch.long)
    )
    for _, chunk, chunk_length, left_offset, right_offset in loader:
        cache_rows, fifo_rows = state.spkcache.shape[1], state.fifo.shape[1]
        inputs = {
            "chunk": padded(chunk, sf.CHUNK_MEL_FRAMES),
            "chunk_lengths": chunk_length.numpy().astype(np.int32),
            "spkcache": padded(state.spkcache, sf.STREAMING["spkcache_len"]),
            "spkcache_lengths": np.array([cache_rows], dtype=np.int32),
            "fifo": padded(state.fifo, sf.STREAMING["fifo_len"]),
            "fifo_lengths": np.array([fifo_rows], dtype=np.int32),
        }
        started = time.perf_counter()
        output = coreml.predict(inputs)
        seconds.append(time.perf_counter() - started)
        embedding_rows = int(output["chunk_pre_encoder_lengths_out"].reshape(-1)[0])
        valid = cache_rows + fifo_rows + embedding_rows
        preds = torch.from_numpy(output["speaker_preds"][:, :valid].astype(np.float32))
        embeddings = torch.from_numpy(output["chunk_pre_encoder_embs_out"][:, :embedding_rows].astype(np.float32))

        reference, _, reference_lengths = sf.step_reference(
            model,
            *(
                torch.from_numpy(inputs[name]).long() if name.endswith("_lengths") else torch.from_numpy(inputs[name])
                for name in ("chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths")
            ),
        )
        if int(reference_lengths[0]) != embedding_rows:
            raise RuntimeError(f"embedding rows {embedding_rows} != NeMo's {int(reference_lengths[0])}")
        difference = np.abs(preds.numpy() - reference[:, :valid].numpy())
        step_max = max(step_max, float(difference.max()))
        step_sum += float(difference.sum())
        step_count += difference.size

        state, chunk_preds = modules.streaming_update(
            streaming_state=state,
            chunk=embeddings,
            preds=preds,
            lc=round(left_offset / sf.SUBSAMPLING_FACTOR),
            rc=int(np.ceil(right_offset / sf.SUBSAMPLING_FACTOR)),
        )
        total.append(chunk_preds)
    predictions = torch.cat(total, dim=1)[0].numpy()
    stats = {
        "steps": len(seconds),
        "step_vs_pytorch_max_abs": step_max,
        "step_vs_pytorch_mean_abs": step_sum / max(1, step_count),
        "median_step_ms": 1000 * float(np.median(seconds)),
    }
    return predictions, stats


# MARK: - Metrics


def hysteresis(probabilities: np.ndarray, onset: float = ONSET, offset: float = OFFSET) -> np.ndarray:
    """Per-speaker on/off decisions with onset and offset thresholds, frame by frame."""
    decisions = np.zeros(probabilities.shape, dtype=bool)
    for speaker in range(probabilities.shape[1]):
        active = False
        for frame, value in enumerate(probabilities[:, speaker]):
            active = value >= offset if active else value >= onset
            decisions[frame, speaker] = active
    return decisions


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    frames = min(len(a), len(b))
    a, b = a[:frames], b[:frames]
    difference = np.abs(a - b)
    return {
        "frames": frames,
        "frame_count_difference": int(abs(len(a) - len(b))),
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
        "agreement_0_5": float(np.mean((a >= 0.5) == (b >= 0.5))),
        "agreement_hysteresis": float(np.mean(hysteresis(a) == hysteresis(b))),
        "frames_with_any_decision_difference_0_5": int(np.any((a >= 0.5) != (b >= 0.5), axis=1).sum()),
    }


# MARK: - Main


def reference_model(path: Path | None, cache_dir: Path) -> Path:
    if path:
        return path.resolve()
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / sf.REFERENCE.repo.replace("/", "--") / sf.REFERENCE.revision
    snapshot_download(
        repo_id=sf.REFERENCE.repo,
        revision=sf.REFERENCE.revision,
        allow_patterns=[f"{sf.REFERENCE.file}/**"],
        local_dir=local_dir,
    )
    model = local_dir / sf.REFERENCE.file
    if sf.directory_size(model) != sf.REFERENCE.size:
        raise RuntimeError(f"{model}: {sf.directory_size(model)} bytes; expected {sf.REFERENCE.size}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, nargs="+", required=True, help="16 kHz mono WAV files")
    parser.add_argument("--artifacts", type=Path, default=sf.DEFAULT_OUTPUT_DIR, help="Converted output directory")
    parser.add_argument(
        "--reference-model",
        type=Path,
        help=f"FluidInference's compiled model (default: download {sf.REFERENCE.file} at the pinned revision)",
    )
    parser.add_argument("--cache-dir", type=Path, default=sf.DEFAULT_CACHE_DIR)
    parser.add_argument("--compute-units", nargs="+", choices=sorted(COMPUTE_UNITS), default=["CPU_AND_NE"])
    parser.add_argument(
        "--nemo-reference-dir",
        type=Path,
        help="Directory with <name>.preds.f32 from Tools/sortformer-reference.py to compare as well",
    )
    parser.add_argument("--report", type=Path, default=sf.REPORTS_DIR / "verify.json")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    ours = (args.artifacts / sf.MODEL_FILE).resolve()
    reference = reference_model(args.reference_model, args.cache_dir.resolve())
    report: dict[str, Any] = {"ours": str(ours), "reference": str(reference), "tools": sf.tool_versions()}

    interface = check_interface(ours, reference)
    report["interface"] = interface
    print(
        "Interface: inputs match FluidInference: {inputs_match_reference}; outputs: {outputs_match_reference}; "
        "contract: {matches_contract}; runtime metadata: {runtime_metadata_matches_reference}".format(**interface)
    )

    weights = compare_weights.compare(ours, reference)
    report["weights"] = weights
    print(f"Weights: {weights['identical']} of {weights['tensors']} tensors byte-identical")
    for entry in weights["different"]:
        print(
            f"  {entry['tensor']} {entry['shape']}: {entry['values_different']} of {entry['values']} values differ, "
            f"max |difference| {entry['max_abs_difference']:.4g} (max |value| {entry['max_abs_value']:.4g})"
        )
    for submodel, program in weights["programs"].items():
        print(f"  {submodel}/model.mil without buildInfo identical: {program['identical_without_build_info']}")

    checkpoint, _ = sf.download_checkpoint(args.cache_dir.resolve())
    model = sf.load_model(checkpoint)
    reference_dir = sf.BUILD_DIR / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    loaded = {
        (name, units): ct.models.CompiledMLModel(str(path), compute_units=COMPUTE_UNITS[units])
        for name, path in (("ours", ours), ("fluidinference", reference))
        for units in args.compute_units
    }

    report["files"] = {}
    for audio_path in args.audio:
        audio, rate = soundfile.read(audio_path, dtype="float32")
        if rate != sf.SAMPLE_RATE or audio.ndim != 1:
            raise SystemExit(f"{audio_path}: the audio must be 16 kHz mono")
        stem = audio_path.stem
        features, feature_length = nemo_features(model, audio)
        started = time.perf_counter()
        nemo = nemo_streaming(model, features, feature_length)
        nemo_seconds = time.perf_counter() - started
        features[0].transpose(0, 1).contiguous().numpy().astype("<f4").tofile(reference_dir / f"{stem}.mel.f32")
        nemo.astype("<f4").tofile(reference_dir / f"{stem}.preds.f32")
        entry: dict[str, Any] = {
            "seconds": len(audio) / sf.SAMPLE_RATE,
            "frames": len(nemo),
            "nemo_seconds": round(nemo_seconds, 2),
        }
        if args.nemo_reference_dir and (args.nemo_reference_dir / f"{stem}.preds.f32").exists():
            earlier = np.fromfile(args.nemo_reference_dir / f"{stem}.preds.f32", dtype="<f4").reshape(
                -1, sf.SPEAKER_COUNT
            )
            entry["nemo_vs_earlier_reference"] = compare(nemo, earlier)
        predictions = {}
        for (name, units), coreml in loaded.items():
            preds, stats = coreml_streaming(model, coreml, features, feature_length)
            predictions[(name, units)] = preds
            result = {"vs_nemo": compare(preds, nemo), **stats}
            if "nemo_vs_earlier_reference" in entry:
                result["vs_earlier_reference"] = compare(preds, earlier)
            entry[f"{name}/{units}"] = result
            print(
                f"{stem} {name} {units}: vs NeMo mean {result['vs_nemo']['mean_abs']:.2e} "
                f"max {result['vs_nemo']['max_abs']:.3g}, agreement {100 * result['vs_nemo']['agreement_0_5']:.2f}% "
                f"(hysteresis {100 * result['vs_nemo']['agreement_hysteresis']:.2f}%); "
                f"step vs PyTorch max {stats['step_vs_pytorch_max_abs']:.3g}; {stats['median_step_ms']:.1f} ms/step"
            )
        for units in args.compute_units:
            result = compare(predictions[("ours", units)], predictions[("fluidinference", units)])
            entry[f"ours_vs_fluidinference/{units}"] = result
            print(
                f"{stem} ours vs FluidInference {units}: mean {result['mean_abs']:.2e} max {result['max_abs']:.3g}, "
                f"agreement {100 * result['agreement_0_5']:.2f}% (hysteresis {100 * result['agreement_hysteresis']:.2f}%)"
            )
        report["files"][stem] = entry

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.report}")
    ok = interface["inputs_match_reference"] and interface["outputs_match_reference"] and interface["matches_contract"]
    return 0 if ok and interface["runtime_metadata_matches_reference"] else 1


if __name__ == "__main__":
    sys.exit(main())
