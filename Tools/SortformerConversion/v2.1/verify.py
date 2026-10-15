# SPDX-License-Identifier: Apache-2.0

"""Compare the converted model with NeMo's streaming inference of the same checkpoint.

Usage:
    uv run python verify.py --audio two-speakers.wav overlap.wav
    uv run python verify.py --audio long.wav --compute-units CPU_AND_NE CPU_ONLY --model other.mlmodelc

For each 16 kHz mono file, NeMo computes the log-mel features and runs `forward_streaming`
with NVIDIA's low-latency configuration in float32 on the CPU. The same features then go
through the same chunks and NeMo's own speaker cache and FIFO update (`streaming_update`),
with each model call replaced by the Core ML model. The report compares the frame
probabilities, the decisions at a 0.5 threshold, and the speech regions after NeMo's
post-processing with the CALLHOME parameters, at 10 ms resolution.

With --write-reference, it also writes `<name>.mel.f32` and `<name>.preds.f32` files in the
format of AnySpeechRecognizer's `Tools/sortformer-reference.py`, for the Swift parity tests.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import sortformer as sf
import soundfile
import torch

# NeMo's diar_streaming_sortformer_4spk-v2_callhome-part1.yaml; NVIDIA publishes no v2.1 parameters.
CALLHOME = {
    "onset": 0.641,
    "offset": 0.561,
    "pad_onset": 0.229,
    "pad_offset": 0.079,
    "min_duration_on": 0.511,
    "min_duration_off": 0.296,
}


def features(model, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    audio, rate = soundfile.read(str(path), dtype="float32")
    if rate != 16_000 or audio.ndim != 1:
        raise SystemExit(f"error: {path} must be 16 kHz mono")
    signal = torch.from_numpy(audio).unsqueeze(0)
    with torch.inference_mode():
        mel, length = model.process_signal(audio_signal=signal, audio_signal_length=torch.tensor([signal.shape[1]]))
    return mel[:, :, : int(length.max())], length


def nemo_predictions(model, mel: torch.Tensor, length: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        return model.forward_streaming(mel, length)[0].numpy()


def coreml_predictions(
    model, coreml: ct.models.CompiledMLModel, mel: torch.Tensor, length: torch.Tensor
) -> tuple[np.ndarray, list[float]]:
    """Run NeMo's synchronous streaming loop with Core ML in place of each model call."""
    modules = model.sortformer_modules
    state = modules.init_streaming_state(batch_size=1, async_streaming=False, device="cpu")
    total = torch.zeros((1, 0, sf.SPEAKERS))
    chunk = np.zeros((1, sf.CHUNK_MEL_FRAMES, sf.MEL_BINS), dtype=np.float32)
    cache = np.zeros((1, sf.STREAMING["spkcache_len"], sf.EMBEDDING_SIZE), dtype=np.float32)
    fifo = np.zeros((1, sf.STREAMING["fifo_len"], sf.EMBEDDING_SIZE), dtype=np.float32)
    seconds: list[float] = []
    loader = modules.streaming_feat_loader(
        feat_seq=mel, feat_seq_length=length, feat_seq_offset=torch.zeros((1,), dtype=torch.long)
    )
    with torch.inference_mode():
        for _, chunk_features, chunk_lengths, left, right in loader:
            frames = chunk_features.shape[1]
            chunk[:] = 0
            chunk[0, :frames] = chunk_features[0].numpy()
            cache_rows, fifo_rows = state.spkcache.shape[1], state.fifo.shape[1]
            cache[:] = 0
            cache[0, :cache_rows] = state.spkcache[0].numpy()
            fifo[:] = 0
            fifo[0, :fifo_rows] = state.fifo[0].numpy()
            started = time.perf_counter()
            output = coreml.predict(
                {
                    "chunk": chunk,
                    "chunk_lengths": np.array([int(chunk_lengths[0])], dtype=np.int32),
                    "spkcache": cache,
                    "spkcache_lengths": np.array([cache_rows], dtype=np.int32),
                    "fifo": fifo,
                    "fifo_lengths": np.array([fifo_rows], dtype=np.int32),
                }
            )
            seconds.append(time.perf_counter() - started)
            rows = int(output["chunk_pre_encoder_lengths_out"][0])
            embeddings = torch.from_numpy(output["chunk_pre_encoder_embs_out"][:, :rows].astype(np.float32))
            preds = torch.from_numpy(output["speaker_preds"][:, : cache_rows + fifo_rows + rows].astype(np.float32))
            state, chunk_preds = modules.streaming_update(
                streaming_state=state,
                chunk=embeddings,
                preds=preds,
                lc=round(left / sf.SUBSAMPLING),
                rc=math.ceil(right / sf.SUBSAMPLING),
            )
            total = torch.cat([total, chunk_preds], dim=1)
    return total[0].numpy(), seconds


def speech_mask(probabilities: np.ndarray, parameters: dict[str, float]) -> np.ndarray:
    """Speech regions after NeMo's post-processing, one row per 10 ms and one column per speaker."""
    from nemo.collections.asr.parts.utils.vad_utils import ts_vad_post_processing

    steps = probabilities.shape[0] * 8
    mask = np.zeros((steps, probabilities.shape[1]), dtype=bool)
    for speaker in range(probabilities.shape[1]):
        segments = ts_vad_post_processing(
            torch.from_numpy(probabilities[:, speaker].copy()), dict(parameters), unit_10ms_frame_count=8
        )
        for start, end in segments.tolist():
            mask[max(0, round(start * 100)) : min(steps, round(end * 100)), speaker] = True
    return mask


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"prediction shapes differ: {reference.shape} and {candidate.shape}")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    reference_mask, candidate_mask = speech_mask(reference, CALLHOME), speech_mask(candidate, CALLHOME)
    return {
        "frames": int(reference.shape[0]),
        "mean_abs_difference": float(difference.mean()),
        "max_abs_difference": float(difference.max()),
        "p99_abs_difference": float(np.quantile(difference, 0.99)),
        "agreement_at_0.5": float(np.mean((reference > 0.5) == (candidate > 0.5))),
        "disagreements_at_0.5": int(np.sum((reference > 0.5) != (candidate > 0.5))),
        "callhome_agreement_10ms": float(np.mean(reference_mask == candidate_mask)),
        "callhome_disagreements_10ms": int(np.sum(reference_mask != candidate_mask)),
        "reference_speech_seconds": float(reference_mask.sum() / 100),
        "candidate_speech_seconds": float(candidate_mask.sum() / 100),
        "reference_mean_probability_by_speaker": [round(float(v), 4) for v in reference.mean(axis=0)],
        "candidate_mean_probability_by_speaker": [round(float(v), 4) for v in candidate.mean(axis=0)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, nargs="+", required=True, help="16 kHz mono WAV files")
    parser.add_argument("--checkpoint", type=Path, help="A local copy of the pinned .nemo checkpoint")
    parser.add_argument("--cache-dir", type=Path, default=sf.DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--model",
        type=Path,
        nargs="+",
        default=[sf.DEFAULT_OUTPUT_DIR / sf.MODEL_FILE],
        help="Compiled models to compare (default: the converted model)",
    )
    parser.add_argument(
        "--compute-units", nargs="+", default=["CPU_AND_NE"], choices=["CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU", "ALL"]
    )
    parser.add_argument("--write-reference", type=Path, metavar="DIRECTORY", help="Write NeMo features and predictions")
    parser.add_argument("--report", type=Path, default=sf.REPORTS_DIR / "verify.json")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve() if args.checkpoint else sf.download_checkpoint(args.cache_dir.resolve())
    model = sf.load_model(checkpoint)
    report: dict[str, Any] = {
        "source": f"{sf.SOURCE.repo}@{sf.SOURCE.revision}",
        "streaming": sf.STREAMING,
        "post_processing": CALLHOME,
        "tools": sf.tool_versions(),
        "results": [],
    }
    references = {}
    for path in args.audio:
        mel, length = features(model, path)
        references[path] = (mel, length, nemo_predictions(model, mel, length))
        if args.write_reference:
            args.write_reference.mkdir(parents=True, exist_ok=True)
            prefix = args.write_reference / path.stem
            mel[0].transpose(0, 1).contiguous().numpy().astype("<f4").tofile(f"{prefix}.mel.f32")
            references[path][2].astype("<f4").tofile(f"{prefix}.preds.f32")
    for model_path in args.model:
        for units in args.compute_units:
            coreml = ct.models.CompiledMLModel(str(model_path.resolve()), compute_units=getattr(ct.ComputeUnit, units))
            for path, (mel, length, reference) in references.items():
                candidate, seconds = coreml_predictions(model, coreml, mel, length)
                result = {
                    "model": str(model_path),
                    "compute_units": units,
                    "audio": path.name,
                    "seconds": round(float(length[0]) / 100, 2),
                    "median_call_ms": round(1000 * float(np.median(seconds[1:] or seconds)), 2),
                    **compare(reference, candidate),
                }
                report["results"].append(result)
                print(
                    f"{model_path.name} {units} {path.name}: mean {result['mean_abs_difference']:.2e}, "
                    f"max {result['max_abs_difference']:.3f}, agreement {100 * result['agreement_at_0.5']:.2f}% "
                    f"({result['disagreements_at_0.5']} of {result['frames'] * sf.SPEAKERS}), "
                    f"post-processed {100 * result['callhome_agreement_10ms']:.2f}%, "
                    f"call {result['median_call_ms']} ms",
                    flush=True,
                )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
