# SPDX-License-Identifier: Apache-2.0

"""Check converted Parakeet Core ML artifacts against NeMo and FluidInference's artifacts.

The checks:

1. Interface: each compiled model's outputs have the contract's names, types, and shapes.
2. Encoder parity: Core ML encoder output versus NeMo's float32 PyTorch encoder
   on the same NeMo mel features (valid frames only).
3. Decoder and joint parity: the Core ML prediction network and joint decision
   versus PyTorch on the inputs that occur during greedy decoding.
4. Transcripts: NeMo's own transcribe() versus greedy TDT decoding in Python over
   the Core ML models (MLModel predictions), reported as word error rate.
   The same windowed decoder also runs over the PyTorch modules, which separates
   conversion error from the effect of 15-second windows on long clips.
5. The same checks for FluidInference's published artifacts at a pinned revision.

Usage:
    uv run python verify.py --variant v3
    uv run python verify.py --variant ultra --compute-units CPU_AND_NE CPU_ONLY
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coremltools as ct
import jiwer
import numpy as np
import torch

import parakeet as pk

# Synthetic speech fixtures rendered with macOS `say`.
FIXTURES: list[tuple[str, str, str]] = [
    ("en-us-fox", "Samantha", "The quick brown fox jumps over the lazy dog."),
    (
        "en-gb-stella",
        "Daniel",
        (
            "Please call Stella. Ask her to bring these things with her from the store: "
            "six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob."
        ),
    ),
    (
        "en-au-meeting",
        "Karen",
        "On Tuesday the thirteenth of October, the meeting started at nine forty-five and ended two hours later.",
    ),
    ("de-weather", "Anna", "Morgen wird es in Berlin sonnig und warm, am Nachmittag ziehen aber Wolken auf."),
    ("fr-train", "Thomas", "Le train pour Lyon partira du quai numéro trois à dix-huit heures."),
    (
        "en-us-long",
        "Samantha",
        (
            "The lighthouse keeper woke before dawn, as he had every morning for eleven years. "
            "He climbed the narrow stairs, checked the lamp, and wrote the weather in a worn notebook. "
            "On calm days the sea was grey and quiet, and the gulls slept on the rocks below the tower. "
            "On stormy nights the wind shook the windows, and he listened for ships that might need help. "
            "When the supply boat arrived each Friday, he traded letters with the captain and asked for news from the mainland. "
            "Most weeks the news was small: a new bakery, a broken bridge, a wedding in the village. "
            "He liked it that way. "
            "In the evening he read old novels by the fire and fell asleep to the sound of the waves."
        ),
    ),
]

LEFT_CONTEXT_FRAMES = 25  # 2.0 s
RIGHT_CONTEXT_FRAMES = 25  # 2.0 s

COMPUTE_UNITS = {
    "ALL": ct.ComputeUnit.ALL,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
}


# MARK: - Fixtures


def render_fixtures(directory: Path) -> list[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for name, voice, text in FIXTURES:
        path = directory / f"{name}.wav"
        if not path.exists():
            subprocess.run(
                ["say", "-v", voice, "-o", str(path), "--data-format=LEF32@16000", text],
                check=True,
            )
        audio = pk.load_audio(path)
        fixtures.append({"name": name, "voice": voice, "text": text, "path": path, "audio": audio})
    return fixtures


# MARK: - Inference backends


class Backend:
    name: str

    def encode(self, mel: np.ndarray, mel_length: np.ndarray) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def decode(self, token: int, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def joint(self, encoder_step: np.ndarray, decoder_step: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError


class TorchBackend(Backend):
    """NeMo's float32 PyTorch modules behind the same interface as the Core ML models."""

    def __init__(self, model: Any) -> None:
        self.name = "pytorch"
        self.model = model
        self.encoder = pk.EncoderWrapper(model.encoder.eval())
        self.decoder = pk.DecoderWrapper(model.decoder.eval())
        self.joint_decision = pk.JointDecisionSingleStep(
            pk.JointWrapper(model.joint.eval()), pk.VOCAB_SIZE, len(pk.TDT_DURATIONS)
        )
        model.decoder._rnnt_export = True  # Do not prepend a start symbol in forward().

    def encode(self, mel, mel_length):
        with torch.inference_mode():
            encoded, length = self.encoder(torch.from_numpy(mel), torch.from_numpy(mel_length))
        return encoded.numpy(), int(length[0])

    def decode(self, token, h, c):
        with torch.inference_mode():
            out, h2, c2 = self.decoder(
                torch.tensor([[token]], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                torch.from_numpy(h),
                torch.from_numpy(c),
            )
        return out.numpy(), h2.numpy(), c2.numpy()

    def joint(self, encoder_step, decoder_step):
        with torch.inference_mode():
            outputs = self.joint_decision(torch.from_numpy(encoder_step), torch.from_numpy(decoder_step))
        names = ("token_id", "token_prob", "duration", "top_k_ids", "top_k_logits")
        return {name: value.numpy() for name, value in zip(names, outputs)}


class CoreMLBackend(Backend):
    def __init__(self, name: str, directory: Path, files: dict[str, str], compute_units: str) -> None:
        self.name = name
        self.directory = directory
        self.files = files
        started = time.time()
        units = COMPUTE_UNITS[compute_units]
        self.encoder = ct.models.CompiledMLModel(str(directory / files["encoder"]), compute_units=units)
        self.decoder = ct.models.CompiledMLModel(str(directory / files["decoder"]), compute_units=units)
        self.joint_model = ct.models.CompiledMLModel(str(directory / files["joint"]), compute_units=units)
        self.load_seconds = time.time() - started

    def encode(self, mel, mel_length):
        out = self.encoder.predict({"mel": mel.astype(np.float32), "mel_length": mel_length.astype(np.int32)})
        self.last_encoder_outputs = out
        return np.asarray(out["encoder"], dtype=np.float32), int(np.asarray(out["encoder_length"]).reshape(-1)[0])

    def decode(self, token, h, c):
        out = self.decoder.predict(
            {
                "targets": np.array([[token]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "h_in": h.astype(np.float32),
                "c_in": c.astype(np.float32),
            }
        )
        self.last_decoder_outputs = out
        return (
            np.asarray(out["decoder"], dtype=np.float32),
            np.asarray(out["h_out"], dtype=np.float32),
            np.asarray(out["c_out"], dtype=np.float32),
        )

    def joint(self, encoder_step, decoder_step):
        out = self.joint_model.predict(
            {"encoder_step": encoder_step.astype(np.float32), "decoder_step": decoder_step.astype(np.float32)}
        )
        self.last_joint_outputs = out
        return {key: np.asarray(value) for key, value in out.items()}


def check_interface(backend: CoreMLBackend, mel: np.ndarray, mel_length: np.ndarray) -> dict[str, Any]:
    """Run each model once and compare output names, types, and shapes with the contract."""
    encoded, _ = backend.encode(mel, mel_length)
    zeros = np.zeros((pk.DECODER_LAYERS, 1, pk.DECODER_DIM), dtype=np.float32)
    decoded, _, _ = backend.decode(pk.BLANK_ID, zeros, zeros)
    backend.joint(np.ascontiguousarray(encoded[:, :, :1]), decoded)
    result: dict[str, Any] = {}
    for component, outputs in (
        ("encoder", backend.last_encoder_outputs),
        ("decoder", backend.last_decoder_outputs),
        ("joint", backend.last_joint_outputs),
    ):
        actual = {name: (str(np.asarray(v).dtype), tuple(np.asarray(v).shape)) for name, v in outputs.items()}
        expected = pk.CONTRACT[component]["outputs"]
        result[component] = {
            "matches": actual == expected,
            "outputs": {k: [d, list(s)] for k, (d, s) in actual.items()},
        }
    return result


# MARK: - Greedy TDT decoding


@dataclass
class Token:
    id: int
    frame: int
    probability: float


def greedy_tdt(
    backend: Backend,
    mel_for: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]],
    audio: np.ndarray,
    record: list | None = None,
) -> tuple[list[Token], int]:
    """Greedy TDT decoding over 15-second windows with left and right context.

    Tokens decoded before the right context of each window are committed.
    The next window starts LEFT_CONTEXT_FRAMES before the first uncommitted frame.
    Decoder state carries across windows. Returns the tokens and the window count.
    """
    h = np.zeros((pk.DECODER_LAYERS, 1, pk.DECODER_DIM), dtype=np.float32)
    c = np.zeros_like(h)
    decoder_out, h_next, c_next = backend.decode(pk.BLANK_ID, h, c)
    h, c = h_next, c_next
    tokens: list[Token] = []
    next_frame = 0
    window_start = 0
    windows = 0
    while True:
        sample_start = window_start * pk.SAMPLES_PER_ENCODER_FRAME
        is_last = sample_start + pk.WINDOW_SAMPLES >= len(audio)
        window, valid = pk.window_audio(audio[sample_start:])
        mel, mel_length = mel_for(window, valid)
        encoded, encoded_length = backend.encode(mel, mel_length)
        windows += 1
        commit_end = encoded_length if is_last else min(encoded_length, pk.ENCODER_FRAMES - RIGHT_CONTEXT_FRAMES)
        t = next_frame - window_start
        symbols = 0
        while t < commit_end:
            step = np.ascontiguousarray(encoded[:, :, t : t + 1])
            out = backend.joint(step, decoder_out)
            if record is not None:
                record.append((step, decoder_out, out))
            token = int(np.asarray(out["token_id"]).reshape(-1)[0])
            duration = pk.TDT_DURATIONS[int(np.asarray(out["duration"]).reshape(-1)[0])]
            if token != pk.BLANK_ID:
                tokens.append(Token(token, window_start + t, float(np.asarray(out["token_prob"]).reshape(-1)[0])))
                decoder_out, h, c = backend.decode(token, h, c)
                symbols += 1
            if token == pk.BLANK_ID and duration == 0:
                duration = 1
            if duration == 0 and symbols >= pk.MAX_SYMBOLS_PER_FRAME:
                duration = 1
            if duration > 0:
                symbols = 0
            t += duration
        next_frame = window_start + t
        if is_last:
            break
        window_start = next_frame - LEFT_CONTEXT_FRAMES
    return tokens, windows


def detokenize(tokens: list[Token], vocab: dict[int, str]) -> str:
    return "".join(vocab[t.id] for t in tokens).replace(pk.WORD_BOUNDARY, " ").strip()


# MARK: - Metrics


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(references: list[str], hypotheses: list[str]) -> float:
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not pairs:
        return 0.0
    return float(jiwer.wer([r for r, _ in pairs], [h if h.strip() else "<empty>" for _, h in pairs]))


def difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    a = reference.astype(np.float64).ravel()
    b = candidate.astype(np.float64).ravel()
    delta = np.abs(a - b)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "reference_mean_abs": float(np.abs(a).mean()),
        "relative_l2": float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)),
        "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
    }


def assert_eval(model: Any) -> None:
    training = [name for name, module in model.named_modules() if module.training]
    if training:
        raise RuntimeError(f"modules in training mode: {training[:5]}")


def nemo_transcribe(model: Any, paths: list[Path]) -> list[tuple[str, list[int]]]:
    model.decoder._rnnt_export = False
    try:
        with torch.inference_mode():
            hypotheses = model.transcribe([str(p) for p in paths], batch_size=1, verbose=False)
    finally:
        # transcribe() unfreezes submodules, which puts them back in training
        # mode (dropout on). Restore evaluation mode for every later check.
        model.eval()
        model.decoder._rnnt_export = True
    assert_eval(model)
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    results = []
    for hypothesis in hypotheses:
        if isinstance(hypothesis, str):
            results.append((hypothesis, []))
        else:
            ids = hypothesis.y_sequence
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            results.append((hypothesis.text, [int(i) for i in ids]))
    return results


# MARK: - Main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(pk.VARIANTS), required=True)
    parser.add_argument("--artifacts", type=Path, help="Converted artifacts (default: build/<variant>)")
    parser.add_argument("--cache-dir", type=Path, default=pk.DEFAULT_CACHE_DIR)
    parser.add_argument("--fixtures", type=Path, default=pk.BUILD_DIR / "fixtures")
    parser.add_argument("--audio", type=Path, nargs="*", default=[], help="Additional 16 kHz audio files")
    parser.add_argument("--compute-units", nargs="+", choices=sorted(COMPUTE_UNITS), default=["CPU_AND_NE"])
    parser.add_argument("--skip-reference", action="store_true", help="Skip FluidInference's artifacts")
    parser.add_argument(
        "--compare-artifacts",
        type=Path,
        metavar="DIRECTORY",
        help="Also check another directory with this tool's file names, such as a published revision",
    )
    parser.add_argument("--report", type=Path, help="JSON report path (default: build/reports/verify-<variant>.json)")
    args = parser.parse_args()

    variant = pk.VARIANTS[args.variant]
    artifacts = (args.artifacts or pk.BUILD_DIR / variant.name).resolve()
    report_path = (args.report or pk.BUILD_DIR / "reports" / f"verify-{variant.name}.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    fixtures = render_fixtures(args.fixtures)
    for path in args.audio:
        fixtures.append({"name": path.stem, "voice": None, "text": None, "path": path, "audio": pk.load_audio(path)})

    model, sources, _ = pk.load_model(variant, args.cache_dir.resolve())
    assert_eval(model)
    torch_backend = TorchBackend(model)

    def mel_for(window: np.ndarray, valid: int):
        return pk.nemo_mel(model, window, valid)

    artifact_sets: dict[str, tuple[Path, dict[str, str]]] = {
        "converted": (artifacts, {"encoder": pk.ENCODER_FILE, "decoder": pk.DECODER_FILE, "joint": pk.JOINT_FILE}),
    }
    if args.compare_artifacts:
        artifact_sets["compared"] = (
            args.compare_artifacts.resolve(),
            {"encoder": pk.ENCODER_FILE, "decoder": pk.DECODER_FILE, "joint": pk.JOINT_FILE},
        )
    if not args.skip_reference:
        reference_dir = pk.download(variant.reference, args.cache_dir.resolve())
        artifact_sets["fluidinference"] = (reference_dir, variant.reference_components)

    report: dict[str, Any] = {
        "variant": variant.name,
        "artifacts": str(artifacts),
        "reference": {"repo": variant.reference.repo, "revision": variant.reference.revision},
        "sources": sources,
        "tools": pk.tool_versions(),
        "decoding": {
            "window_seconds": pk.WINDOW_SAMPLES / pk.SAMPLE_RATE,
            "left_context_frames": LEFT_CONTEXT_FRAMES,
            "right_context_frames": RIGHT_CONTEXT_FRAMES,
            "max_symbols_per_frame": pk.MAX_SYMBOLS_PER_FRAME,
            "mel": "NeMo AudioToMelSpectrogramPreprocessor (float32, per-window normalization)",
        },
        "fixtures": [
            {
                "name": f["name"],
                "voice": f["voice"],
                "text": f["text"],
                "seconds": round(len(f["audio"]) / pk.SAMPLE_RATE, 2),
            }
            for f in fixtures
        ],
    }

    # Vocabulary
    converted_vocab_path = artifacts / pk.VOCAB_FILE
    vocab = {int(k): v for k, v in json.loads(converted_vocab_path.read_text()).items()}
    report["vocabulary"] = {
        "pieces": len(vocab),
        "matches_nemo_tokenizer": [vocab[i] for i in range(len(vocab))] == pk.vocabulary_entries(model),
    }
    if "fluidinference" in artifact_sets:
        reference_vocab_path = artifact_sets["fluidinference"][0] / pk.VOCAB_FILE
        report["vocabulary"]["same_mapping_as_reference"] = json.loads(reference_vocab_path.read_text()) == json.loads(
            converted_vocab_path.read_text()
        )
        report["vocabulary"]["byte_identical_to_reference"] = (
            reference_vocab_path.read_bytes() == converted_vocab_path.read_bytes()
        )

    # NeMo reference transcripts (full context, NeMo's own greedy batched TDT decoder).
    t = time.time()
    nemo_results = nemo_transcribe(model, [f["path"] for f in fixtures])
    report["nemo_transcribe_seconds"] = round(time.time() - t, 1)
    nemo_texts = [text for text, _ in nemo_results]

    # PyTorch modules through the same windowed decoder, recording joint inputs.
    torch_texts = []
    joint_records: list = []
    for fixture in fixtures:
        tokens, _ = greedy_tdt(torch_backend, mel_for, fixture["audio"], record=joint_records)
        torch_texts.append(detokenize(tokens, vocab))

    # Per-fixture first-window mel and PyTorch encoder output.
    first_windows = []
    for fixture in fixtures:
        window, valid = pk.window_audio(fixture["audio"])
        mel, mel_length = mel_for(window, valid)
        encoded, encoded_length = torch_backend.encode(mel, mel_length)
        first_windows.append((mel, mel_length, encoded, encoded_length))

    # Decoder inputs: NeMo's tokens for every fixture, fed in order.
    decoder_sequences = [[pk.BLANK_ID] + ids for _, ids in nemo_results]

    report["results"] = {}
    for units in args.compute_units:
        for set_name, (directory, files) in artifact_sets.items():
            key = f"{set_name}/{units}"
            print(f"[{variant.name}] {key}: loading {directory}", flush=True)
            backend = CoreMLBackend(set_name, directory, files, units)
            result: dict[str, Any] = {
                "directory": str(directory),
                "files": files,
                "compute_units": units,
                "load_seconds": round(backend.load_seconds, 1),
            }
            result["interface"] = check_interface(backend, first_windows[0][0], first_windows[0][1])

            # Encoder parity on valid frames.
            per_fixture = []
            all_ref, all_out = [], []
            for fixture, (mel, mel_length, encoded_ref, n) in zip(fixtures, first_windows):
                encoded, n_coreml = backend.encode(mel, mel_length)
                ref = encoded_ref[:, :, :n]
                out = encoded[:, :, :n]
                all_ref.append(ref.ravel())
                all_out.append(out.ravel())
                per_fixture.append(
                    {"fixture": fixture["name"], "frames": n, "length_matches": n == n_coreml, **difference(ref, out)}
                )
            result["encoder"] = {
                "overall": difference(np.concatenate(all_ref), np.concatenate(all_out)),
                "per_fixture": per_fixture,
            }

            # Decoder parity along NeMo's token sequences.
            max_out, max_state, steps = 0.0, 0.0, 0
            for sequence in decoder_sequences:
                h_t = np.zeros((pk.DECODER_LAYERS, 1, pk.DECODER_DIM), dtype=np.float32)
                c_t = h_t.copy()
                h_c, c_c = h_t.copy(), h_t.copy()
                for token in sequence:
                    out_t, h_t, c_t = torch_backend.decode(token, h_t, c_t)
                    out_c, h_c, c_c = backend.decode(token, h_c, c_c)
                    max_out = max(max_out, float(np.abs(out_t - out_c).max()))
                    max_state = max(max_state, float(np.abs(h_t - h_c).max()), float(np.abs(c_t - c_c).max()))
                    steps += 1
            result["decoder"] = {"steps": steps, "max_abs_output": max_out, "max_abs_state": max_state}

            # Joint parity on the inputs seen during PyTorch greedy decoding.
            token_agree = duration_agree = top1_in_topk = 0
            prob_errors, topk_overlap = [], []
            for encoder_step, decoder_step, out_t in joint_records:
                out_c = backend.joint(encoder_step, decoder_step)
                token_agree += int(out_t["token_id"].ravel()[0] == out_c["token_id"].ravel()[0])
                duration_agree += int(out_t["duration"].ravel()[0] == out_c["duration"].ravel()[0])
                prob_errors.append(abs(float(out_t["token_prob"].ravel()[0]) - float(out_c["token_prob"].ravel()[0])))
                ids_t = set(out_t["top_k_ids"].ravel().tolist())
                ids_c = set(np.asarray(out_c["top_k_ids"]).ravel().tolist())
                topk_overlap.append(len(ids_t & ids_c) / pk.TOP_K)
                top1_in_topk += int(int(out_t["token_id"].ravel()[0]) in ids_c)
            count = len(joint_records)
            result["joint"] = {
                "evaluations": count,
                "token_agreement": token_agree / count,
                "duration_agreement": duration_agree / count,
                "token_prob_max_abs": max(prob_errors),
                "token_prob_mean_abs": float(np.mean(prob_errors)),
                "top_k_overlap_mean": float(np.mean(topk_overlap)),
                "pytorch_token_in_top_k": top1_in_topk / count,
            }

            # Transcripts.
            t = time.time()
            texts, windows = [], []
            for fixture in fixtures:
                tokens, n_windows = greedy_tdt(backend, mel_for, fixture["audio"])
                texts.append(detokenize(tokens, vocab))
                windows.append(n_windows)
            decode_seconds = time.time() - t
            result["transcripts"] = {
                "decode_seconds": round(decode_seconds, 1),
                "wer_vs_nemo": wer(nemo_texts, texts),
                "wer_vs_nemo_normalized": wer([normalize(x) for x in nemo_texts], [normalize(x) for x in texts]),
                "wer_vs_pytorch_windowed": wer(torch_texts, texts),
                "wer_vs_say_text_normalized": wer(
                    [normalize(f["text"]) for f in fixtures if f["text"]],
                    [normalize(x) for f, x in zip(fixtures, texts) if f["text"]],
                ),
                "per_fixture": [
                    {
                        "fixture": f["name"],
                        "windows": w,
                        "text": x,
                        "wer_vs_nemo": wer([n], [x]),
                        "wer_vs_pytorch_windowed": wer([p], [x]),
                    }
                    for f, x, n, p, w in zip(fixtures, texts, nemo_texts, torch_texts, windows)
                ],
            }
            report["results"][key] = result
            print(
                f"[{variant.name}] {key}: encoder max {result['encoder']['overall']['max_abs']:.4f} "
                f"mean {result['encoder']['overall']['mean_abs']:.5f} cos {result['encoder']['overall']['cosine']:.6f}; "
                f"joint tokens {result['joint']['token_agreement']:.4f}; "
                f"WER vs NeMo {result['transcripts']['wer_vs_nemo']:.4f} "
                f"(normalized {result['transcripts']['wer_vs_nemo_normalized']:.4f}), "
                f"vs PyTorch windowed {result['transcripts']['wer_vs_pytorch_windowed']:.4f}",
                flush=True,
            )

    report["nemo"] = [
        {"fixture": f["name"], "text": n, "pytorch_windowed_text": p, "pytorch_windowed_wer_vs_nemo": wer([n], [p])}
        for f, n, p in zip(fixtures, nemo_texts, torch_texts)
    ]
    report["pytorch_windowed_wer_vs_nemo"] = wer(nemo_texts, torch_texts)
    report["pytorch_windowed_wer_vs_say_text_normalized"] = wer(
        [normalize(f["text"]) for f in fixtures if f["text"]],
        [normalize(x) for f, x in zip(fixtures, torch_texts) if f["text"]],
    )
    report["seconds"] = round(time.time() - started, 1)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[{variant.name}] report: {report_path}")


if __name__ == "__main__":
    main()
