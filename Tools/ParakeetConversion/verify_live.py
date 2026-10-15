# SPDX-License-Identifier: Apache-2.0

"""Measure live-transcription accuracy per encoder window and left context.

The script simulates AnySpeechRecognizer's `ParakeetRecognizer` in Python with the
Core ML models. Audio arrives in 100 ms blocks. Each regular window holds left context,
a 0.96 s chunk, and 2 s of right context; the decoder commits tokens emitted in the
chunk. Like the Swift recognizer, the next window resumes after the last committed
token or four frames before the commit frame, whichever is later, and each window
rebuilds the prediction network context from the blank token and the tokens committed
in the last 25 encoder frames. Volatile previews do not change committed tokens and
are not simulated.

Each configuration is `<window seconds>:<left context frames>`, for example `6:38`, with optional
`:r<right>`, `:c<chunk>`, `:h<decoder history>`, and `:m<boundary margin>` in encoder frames
(defaults 25, 12, 25, and 4, as in the Swift live configuration). `:h-1` keeps every
committed token in the decoder context, which is the same as carrying the decoder state.
The 15-second window uses `Encoder.mlmodelc`; other windows use `Encoder<seconds>s.mlmodelc`.
`file` runs the Swift file mode (15 s window, 2 s left and right context, largest chunk).

It also compares encoder outputs of short windows with the corresponding frames of
15-second windows (`--encoder-check`).

Usage:
    uv run python verify_live.py --variant v3 --artifacts build/v3-live \\
        --configs file 15:150 15:38 6:38 --audio /tmp/parakeet-fixtures/long.wav
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coremltools as ct
import jiwer
import numpy as np
import torch

import parakeet as pk
from verify import COMPUTE_UNITS, assert_eval, nemo_transcribe, normalize, render_fixtures

BLOCK_SAMPLES = 1_600  # The CLI's --live option reads 100 ms blocks.
FS = pk.SAMPLES_PER_ENCODER_FRAME

# MARK: - Configurations


@dataclass(frozen=True)
class LiveConfig:
    """The Swift `ParakeetRecognizer.Configuration`, with the encoder window."""

    name: str
    window: pk.EncoderWindow
    left: int
    chunk: int = 12
    right: int = 25
    margin: int = 4
    history: int = 25
    max_symbols: int = pk.MAX_SYMBOLS_PER_FRAME

    @staticmethod
    def parse(text: str, right: int) -> LiveConfig:
        """Parse `file` or `<seconds>:<left>[:r<right>][:c<chunk>][:h<history>][:m<margin>]` (frames)."""
        if text == "file":
            return LiveConfig("file", pk.DEFAULT_WINDOW, left=25, chunk=1 << 30, right=25)
        seconds, left, *options = text.split(":")
        values = {"r": right, "c": 12, "h": 25, "m": 4}
        for option in options:
            values[option[0]] = int(option[1:])
        name = text if right == 25 or any(o.startswith("r") for o in options) else f"{text}:r{right}"
        window = pk.EncoderWindow(int(seconds))
        config = LiveConfig(
            name,
            window,
            left=int(left),
            chunk=values["c"],
            right=values["r"],
            history=values["h"],
            margin=values["m"],
        )
        if config.left + config.right >= window.whole_frames or config.left < 0:
            raise ValueError(f"{text}: the window has no room for a chunk")
        if config.chunk <= config.margin:
            raise ValueError(f"{text}: the chunk must be longer than the boundary margin")
        return config


# MARK: - Core ML models


class Models:
    """Core ML models with the Swift engine's per-call interface."""

    def __init__(self, directory: Path, units: str) -> None:
        self.directory = directory
        self.units = COMPUTE_UNITS[units]
        self.encoders: dict[int, Any] = {}
        self.load_seconds: dict[str, float] = {}
        started = time.time()
        self.decoder = ct.models.CompiledMLModel(str(directory / pk.DECODER_FILE), compute_units=self.units)
        self.joint_model = ct.models.CompiledMLModel(str(directory / pk.JOINT_FILE), compute_units=self.units)
        self.load_seconds["decoder and joint"] = round(time.time() - started, 2)
        self.counts = {"encode": 0, "predict": 0, "joint": 0}
        self._context_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def encoder(self, window: pk.EncoderWindow):
        if window.seconds not in self.encoders:
            started = time.time()
            self.encoders[window.seconds] = ct.models.CompiledMLModel(
                str(self.directory / window.file), compute_units=self.units
            )
            self.load_seconds[window.file] = round(time.time() - started, 2)
        return self.encoders[window.seconds]

    def encode(self, window: pk.EncoderWindow, mel: np.ndarray, mel_length: np.ndarray) -> tuple[np.ndarray, int]:
        out = self.encoder(window).predict({"mel": mel, "mel_length": mel_length.astype(np.int32)})
        self.counts["encode"] += 1
        encoded = np.asarray(out["encoder"], dtype=np.float32)[0].T.copy()  # [frames, 1024]
        return encoded, int(np.asarray(out["encoder_length"]).reshape(-1)[0])

    def predict(self, token: int, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self.decoder.predict(
            {
                "targets": np.array([[token]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "h_in": h,
                "c_in": c,
            }
        )
        self.counts["predict"] += 1
        return (
            np.asarray(out["decoder"], dtype=np.float32),
            np.asarray(out["h_out"], dtype=np.float32),
            np.asarray(out["c_out"], dtype=np.float32),
        )

    def joint(self, frame: np.ndarray, decoder_out: np.ndarray) -> tuple[int, float, int]:
        out = self.joint_model.predict(
            {"encoder_step": frame.reshape(1, pk.ENCODER_DIM, 1), "decoder_step": decoder_out}
        )
        self.counts["joint"] += 1
        token = int(np.asarray(out["token_id"]).reshape(-1)[0])
        probability = float(np.asarray(out["token_prob"]).reshape(-1)[0])
        return token, probability, pk.TDT_DURATIONS[int(np.asarray(out["duration"]).reshape(-1)[0])]

    def context(self, history: tuple[int, ...]) -> np.ndarray:
        """The prediction network output after the blank token and `history`, from a zero state.

        The Swift recognizer recomputes this for every window; the cache only saves time here.
        `context_predictions` counts the calls that the Swift recognizer makes.
        """
        return self.context_state(history)[0]

    def context_state(self, history: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache = self._context_cache
        if history not in cache:
            known = len(history)
            while known > 0 and history[:known] not in cache:
                known -= 1
            if known == 0 and () not in cache:
                zero = np.zeros((pk.DECODER_LAYERS, 1, pk.DECODER_DIM), dtype=np.float32)
                cache[()] = self.predict(pk.BLANK_ID, zero, zero)
            state = cache[history[:known]]
            chain = {history[:i]: cache[history[:i]] for i in range(known + 1)}
            for i in range(known, len(history)):
                state = self.predict(history[i], state[1], state[2])
                chain[history[: i + 1]] = state
            # Keep only the current chain of prefixes, which bounds memory.
            self._context_cache = chain
        return self._context_cache[history]


# MARK: - Live recognizer simulation


@dataclass
class Committed:
    id: int
    frame: int
    duration: int
    probability: float


class LiveRecognizer:
    """The commit path of the Swift `ParakeetRecognizer`, ported line by line."""

    def __init__(self, config: LiveConfig, models: Models, mel_for) -> None:
        self.config = config
        self.models = models
        self.mel_for = mel_for
        self.window = config.window
        self.window_frames = config.window.whole_frames
        self.samples = np.zeros(0, dtype=np.float32)
        self.received = 0
        self.retained_start = 0
        self.committed_frame = 0
        self.history: list[tuple[int, int]] = []
        self.tokens: list[Committed] = []
        self.ended = False
        self.stats = {
            "windows": 0,
            "forced_windows": 0,
            "joint_calls": 0,
            "joint_runs": 0,
            "context_predictions": 0,
            "token_predictions": 0,
        }

    def append(self, block: np.ndarray) -> None:
        self.samples = np.concatenate([self.samples, block]) if self.samples.size else block.copy()
        self.received += len(block)

    @property
    def retained_frame(self) -> int:
        return (self.retained_start + FS - 1) // FS

    def window_start(self, available_frames: int) -> int:
        preferred = self.committed_frame - self.config.left
        return max(0, self.retained_frame, min(preferred, available_frames - self.window_frames))

    def process(self, end_of_input: bool = False) -> None:
        if end_of_input:
            self.ended = True
        available = self.received
        available_frames = (available + FS - 1) // FS
        target = available_frames if self.ended else None
        while self.committed_frame < available_frames:
            preferred = max(self.retained_frame, self.committed_frame - self.config.left)
            regular = min(
                preferred + self.window_frames - self.config.right,
                self.committed_frame + min(self.config.chunk, self.window_frames),
            )
            if (regular + self.config.right) * FS <= available:
                self.run_window(preferred, regular, self.config.margin)
            elif target is not None and target > self.committed_frame:
                self.stats["forced_windows"] += 1
                self.run_window(self.window_start(available_frames), target, 0)
            else:
                break

    def run_window(self, start: int, commit: int, margin: int) -> None:
        start_sample = start * FS
        count = min(self.window.samples, self.received - start_sample)
        assert start_sample >= self.retained_start and count > 0
        # Zero padding to a whole encoder frame counts as audio, as in the Swift recognizer.
        valid = min(self.window.samples, (count + FS - 1) // FS * FS)
        buffer = np.zeros(self.window.samples, dtype=np.float32)
        offset = start_sample - self.retained_start
        buffer[:count] = self.samples[offset : offset + count]
        mel, mel_length = self.mel_for(buffer, valid, self.window)
        encoded, encoded_length = self.models.encode(self.window, mel, mel_length)
        self.stats["windows"] += 1
        decodable = min(encoded_length, (count + FS - 1) // FS)
        local_commit = min(commit - start, decodable)
        if start + local_commit <= self.committed_frame:
            raise RuntimeError("the encoder reported too few valid frames")
        history = tuple(token for token, _ in self.history)
        self.stats["context_predictions"] += len(history) + 1
        decoder_out = self.models.context(history)
        _, h, c = self.models.context_state(history)
        committed = self.decode(encoded, self.committed_frame - start, local_commit, decoder_out, h, c, start)
        after_last = committed[-1].frame + 1 if committed else self.committed_frame
        self.committed_frame = max(after_last, start + local_commit - margin)
        self.tokens += committed
        self.history += [(t.id, t.frame) for t in committed]
        horizon = self.committed_frame - self.config.history
        while self.config.history >= 0 and self.history and self.history[0][1] < horizon:
            self.history.pop(0)
        # Discard audio more than one window before the commit frame.
        keep = max(0, (self.committed_frame - self.window_frames) * FS)
        drop = min(keep - self.retained_start, len(self.samples))
        if drop > 0:
            self.samples = self.samples[drop:]
            self.retained_start += drop

    def decode(self, encoded, start, end, decoder_out, h, c, frame_offset) -> list[Committed]:
        """Greedy TDT decoding from `start` while the frame is before `end` (ParakeetTDTDecoder.decode)."""
        tokens: list[Committed] = []
        frame = start
        symbols = 0
        new_run = True
        while frame < end:
            token, probability, duration = self.models.joint(encoded[frame], decoder_out)
            self.stats["joint_calls"] += 1
            self.stats["joint_runs"] += int(new_run)
            new_run = False
            if token == pk.BLANK_ID:
                frame += max(1, duration)
                symbols = 0
                continue
            tokens.append(Committed(token, frame + frame_offset, duration, probability))
            decoder_out, h, c = self.models.predict(token, h, c)
            self.stats["token_predictions"] += 1
            new_run = True
            symbols += 1
            if duration > 0:
                frame += duration
                symbols = 0
            elif symbols >= self.config.max_symbols:
                frame += 1
                symbols = 0
        return tokens


def run_live(config: LiveConfig, models: Models, mel_for, audio: np.ndarray) -> tuple[list[Committed], dict]:
    recognizer = LiveRecognizer(config, models, mel_for)
    started = time.time()
    for offset in range(0, len(audio), BLOCK_SAMPLES):
        recognizer.append(audio[offset : offset + BLOCK_SAMPLES])
        recognizer.process()
    recognizer.process(end_of_input=True)
    stats = dict(recognizer.stats, seconds=round(time.time() - started, 1))
    return recognizer.tokens, stats


def detokenize(tokens: list[Committed], vocab: dict[int, str]) -> str:
    return "".join(vocab[t.id] for t in tokens).strip()


# MARK: - Metrics

PUNCTUATION = re.compile(r"[.,?!;:]")
SENTENCE_END = re.compile(r"[.?!]")


def word_counts(reference: str, hypothesis: str) -> dict[str, int]:
    if not reference.strip():
        return {"words": 0, "substitutions": 0, "deletions": 0, "insertions": len(hypothesis.split())}
    out = jiwer.process_words(reference, hypothesis if hypothesis.strip() else "<empty>")
    return {
        "words": len(reference.split()),
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
    }


def style_counts(text: str) -> dict[str, int]:
    words = text.split()
    return {
        "punctuation": len(PUNCTUATION.findall(text)),
        "capitalized": sum(1 for w in words if w[:1].isupper()),
    }


def sum_counts(items: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            total[key] = total.get(key, 0) + value
    return total


def rate(counts: dict[str, int]) -> float | None:
    if not counts.get("words"):
        return None
    return (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / counts["words"]


def last_sentence_end_seconds(tokens: list[Committed], vocab: dict[int, str]) -> float | None:
    for token in reversed(tokens):
        if SENTENCE_END.search(vocab[token.id]):
            return round(token.frame * FS / pk.SAMPLE_RATE, 2)
    return None


def sentence_ends_per_minute(tokens: list[Committed], vocab: dict[int, str], seconds: float) -> list[int]:
    buckets = [0] * max(1, int(np.ceil(seconds / 60)))
    for token in tokens:
        if SENTENCE_END.search(vocab[token.id]):
            buckets[min(len(buckets) - 1, int(token.frame * FS / pk.SAMPLE_RATE // 60))] += 1
    return buckets


# MARK: - Encoder window check


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row (frame) of two [frames, channels] arrays."""
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def torch_encode(model: Any, mel: np.ndarray, mel_length: np.ndarray) -> tuple[np.ndarray, int]:
    with torch.inference_mode():
        encoded, length = model.encoder(
            audio_signal=torch.from_numpy(mel), length=torch.from_numpy(mel_length.astype(np.int64))
        )
    return encoded[0].numpy().T.copy(), int(length[0])


def encoder_check(model: Any, audio: np.ndarray, windows: list[pk.EncoderWindow]) -> tuple[dict, list]:
    """Compare short-window encoder outputs with 15-second windows over the same audio.

    For each short window W and several start offsets, with PyTorch's float32 encoder:

    - `masked_15s`: the 15 s encoder given the same W seconds of audio and a valid length
      of W seconds. The masks make this the same computation, so differences are numerical.
    - `same_mel`: W's frames versus the same frames of a 15 s window that starts at the same
      sample and continues with real audio. W's mel features are the first mel frames of the
      15 s window's features, so only the encoder's attention and convolution context differs.
    - `own_mel`: like `same_mel`, but W computes its own mel features, normalized over W
      seconds, as the runtime does.

    `coreml_encoder_check` adds `coreml`: the Core ML W encoder versus the PyTorch W encoder
    on the same mel features. Frame cosine similarity is grouped by distance from the end of
    W's valid audio. Returns the summary and the inputs and outputs for the Core ML comparison.

    In this environment, PyTorch's convolution can crash with a segmentation fault after
    Core ML has loaded models in the same process, so run this before loading Core ML models.
    """
    default = pk.DEFAULT_WINDOW
    starts = list(range(0, len(audio) - default.samples, 9 * pk.SAMPLE_RATE))[:6]
    result: dict[str, Any] = {"starts_seconds": [s / pk.SAMPLE_RATE for s in starts], "windows": {}}
    pending: list[tuple[pk.EncoderWindow, np.ndarray, np.ndarray, np.ndarray]] = []
    for window in windows:
        frames = window.whole_frames
        per: dict[str, list[np.ndarray]] = {"masked_15s": [], "same_mel": [], "own_mel": []}
        for start in starts:
            long_audio, long_valid = pk.window_audio(audio[start:], default.samples)
            long_mel, long_len = pk.nemo_mel(model, long_audio, long_valid)
            long_enc, _ = torch_encode(model, long_mel, long_len)

            short_audio, short_valid = pk.window_audio(audio[start:], window.samples)
            own_mel, own_len = pk.nemo_mel(model, short_audio, short_valid, window.mel_frames)
            own_enc, _ = torch_encode(model, own_mel, own_len)

            same_mel = np.ascontiguousarray(long_mel[:, :, : window.mel_frames])
            same_len = np.array([window.mel_frames], dtype=np.int32)
            same_enc, _ = torch_encode(model, same_mel, same_len)

            padded_mel = np.zeros_like(long_mel)
            padded_mel[:, :, : window.mel_frames] = own_mel
            masked_enc, _ = torch_encode(model, padded_mel, own_len)

            per["masked_15s"].append(cosine_rows(own_enc[:frames], masked_enc[:frames]))
            per["same_mel"].append(cosine_rows(same_enc[:frames], long_enc[:frames]))
            per["own_mel"].append(cosine_rows(own_enc[:frames], long_enc[:frames]))
            pending.append((window, own_mel, own_len, own_enc))
            maxima = result.setdefault("masked_15s_max_abs", {})
            delta = float(np.abs(own_enc[:frames] - masked_enc[:frames]).max())
            maxima[window.file] = max(delta, maxima.get(window.file, 0.0))
        summary: dict[str, Any] = {"whole_frames": frames}
        for key, rows in per.items():
            summary[key] = summarize_cosines(rows, frames)
        result["windows"][window.file] = summary
        print(f"[encoder check] {window.file}: {json.dumps(summary)}", flush=True)
    return result, pending


def summarize_cosines(rows: list[np.ndarray], frames: int) -> dict[str, Any]:
    bands = [(0, 2), (2, 6), (6, 13), (13, 26), (26, 1 << 30)]
    stacked = np.stack(rows)  # [starts, frames]
    distance = frames - 1 - np.arange(frames)  # frames before the end of the window
    return {
        "min": round(float(stacked.min()), 5),
        "mean": round(float(stacked.mean()), 5),
        "by_frames_before_end": {
            f"{lo}-{min(hi, frames) - 1}": round(float(stacked[:, (distance >= lo) & (distance < hi)].mean()), 5)
            for lo, hi in bands
            if lo < frames
        },
        "first_frame": round(float(stacked[:, 0].mean()), 5),
    }


def coreml_encoder_check(models: Models, pending: list, result: dict[str, Any]) -> None:
    """Compare each Core ML window encoder with PyTorch on the encoder check's mel features."""
    rows: dict[str, list[np.ndarray]] = {}
    for window, mel, mel_length, reference in pending:
        if not (models.directory / window.file).exists():
            continue
        encoded, _ = models.encode(window, mel, mel_length)
        frames = window.whole_frames
        rows.setdefault(window.file, []).append(cosine_rows(reference[:frames], encoded[:frames]))
        maxima = result.setdefault("coreml_max_abs", {})
        maxima[window.file] = max(
            float(np.abs(reference[:frames] - encoded[:frames]).max()), maxima.get(window.file, 0.0)
        )
    for name, values in rows.items():
        result["windows"][name]["coreml"] = summarize_cosines(values, len(values[0]))
        print(f"[encoder check] {name} Core ML: {json.dumps(result['windows'][name]['coreml'])}", flush=True)


# MARK: - Main


def file_key(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(pk.VARIANTS), required=True)
    parser.add_argument("--artifacts", type=Path, help="Converted artifacts (default: build/<variant>-live)")
    parser.add_argument("--cache-dir", type=Path, default=pk.DEFAULT_CACHE_DIR)
    parser.add_argument("--fixtures", type=Path, default=pk.BUILD_DIR / "fixtures")
    parser.add_argument(
        "--audio",
        nargs="*",
        default=[],
        metavar="PATH[=REFERENCE]",
        help="Additional audio, resampled to 16 kHz, with an optional reference text file",
    )
    parser.add_argument("--no-synthetic", action="store_true", help="Skip the six synthetic fixtures")
    parser.add_argument("--configs", nargs="+", default=["file", "15:150"], help="file or <seconds>:<left frames>")
    parser.add_argument("--right", type=int, default=25, help="Right context frames for live configurations")
    parser.add_argument("--compute-units", choices=sorted(COMPUTE_UNITS), default="CPU_AND_NE")
    parser.add_argument("--encoder-check", type=int, nargs="*", metavar="SECONDS", help="Run the encoder window check")
    parser.add_argument("--skip-nemo", action="store_true", help="Do not run NeMo's transcribe()")
    parser.add_argument(
        "--prepend-ms",
        type=int,
        default=0,
        help="Silence added before each clip in the simulation only, which moves every window boundary",
    )
    parser.add_argument("--report", type=Path, help="JSON report (default: build/reports/live-<variant>.json)")
    args = parser.parse_args()

    variant = pk.VARIANTS[args.variant]
    artifacts = (args.artifacts or pk.BUILD_DIR / f"{variant.name}-live").resolve()
    report_path = (args.report or pk.BUILD_DIR / "reports" / f"live-{variant.name}.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    configs = [LiveConfig.parse(text, args.right) for text in args.configs]
    vocab = {int(k): v for k, v in json.loads((artifacts / pk.VOCAB_FILE).read_text()).items()}

    fixtures: list[dict[str, Any]] = []
    if not args.no_synthetic:
        for f in render_fixtures(args.fixtures):
            fixtures.append(
                {"name": f["name"], "group": "synthetic", "path": f["path"], "audio": f["audio"], "text": f["text"]}
            )
    for spec in args.audio:
        path_text, _, reference_text = spec.partition("=")
        path = Path(path_text)
        audio = pk.load_audio_resampled(path)
        resampled = pk.BUILD_DIR / "fixtures-live" / f"{path.stem}-16k.wav"
        if not resampled.exists():
            import soundfile as sf

            resampled.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(resampled), audio, pk.SAMPLE_RATE, subtype="FLOAT")
        text = Path(reference_text).read_text().strip() if reference_text else None
        fixtures.append({"name": path.stem, "group": path.stem, "path": resampled, "audio": audio, "text": text})

    model, sources, _ = pk.load_model(variant, args.cache_dir.resolve())
    assert_eval(model)
    model.decoder._rnnt_export = True

    def mel_for(buffer: np.ndarray, valid: int, window: pk.EncoderWindow):
        return pk.nemo_mel(model, buffer, valid, window.mel_frames)

    report: dict[str, Any] = {
        "variant": variant.name,
        "artifacts": str(artifacts),
        "compute_units": args.compute_units,
        "tools": pk.tool_versions(),
        "simulation": {
            "block_samples": BLOCK_SAMPLES,
            "prepended_silence_ms": args.prepend_ms,
            "mel": "NeMo AudioToMelSpectrogramPreprocessor over each window (per-window normalization)",
            "configs": {
                c.name: {
                    "encoder": c.window.file,
                    "window_seconds": c.window.seconds,
                    "left_frames": c.left,
                    "chunk_frames": c.chunk if c.chunk < 1 << 20 else "largest",
                    "right_frames": c.right,
                    "boundary_margin_frames": c.margin,
                    "decoder_history_frames": c.history,
                }
                for c in configs
            },
        },
        "fixtures": [
            {"name": f["name"], "group": f["group"], "seconds": round(len(f["audio"]) / pk.SAMPLE_RATE, 2)}
            for f in fixtures
        ],
    }

    # PyTorch work first: see encoder_check.
    pending: list = []
    if args.encoder_check:
        check_audio = next((f["audio"] for f in fixtures if len(f["audio"]) > 40 * pk.SAMPLE_RATE), None)
        if check_audio is None:
            raise SystemExit("--encoder-check needs a fixture longer than 40 s")
        windows = [pk.EncoderWindow(s) for s in args.encoder_check]
        report["encoder_check"], pending = encoder_check(model, check_audio, windows)

    # NeMo's own transcripts, cached by file.
    nemo_cache_path = pk.BUILD_DIR / "reports" / f"nemo-transcripts-{variant.name}.json"
    nemo_cache = json.loads(nemo_cache_path.read_text()) if nemo_cache_path.exists() else {}
    if not args.skip_nemo:
        for f in fixtures:
            key = file_key(Path(f["path"]))
            if key not in nemo_cache:
                started = time.time()
                text, _ = nemo_transcribe(model, [Path(f["path"])])[0]
                model.decoder._rnnt_export = True
                nemo_cache[key] = {"path": str(f["path"]), "text": text, "seconds": round(time.time() - started, 1)}
                nemo_cache_path.write_text(json.dumps(nemo_cache, indent=2, ensure_ascii=False) + "\n")
                print(f"[nemo] {f['name']}: {nemo_cache[key]['seconds']} s", flush=True)
    for f in fixtures:
        f["nemo"] = nemo_cache.get(file_key(Path(f["path"])), {}).get("text")

    models = Models(artifacts, args.compute_units)
    if pending:
        coreml_encoder_check(models, pending, report["encoder_check"])

    report["sources"] = sources
    report["results"] = {}
    for config in configs:
        entries = []
        for f in fixtures:
            before = dict(models.counts)
            audio = f["audio"]
            if args.prepend_ms:
                silence = np.zeros(args.prepend_ms * pk.SAMPLE_RATE // 1000, dtype=np.float32)
                audio = np.concatenate([silence, audio])
            tokens, stats = run_live(config, models, mel_for, audio)
            stats["core_ml_calls"] = {k: models.counts[k] - before[k] for k in models.counts}
            text = detokenize(tokens, vocab)
            entry: dict[str, Any] = {"fixture": f["name"], "group": f["group"], "text": text, "stats": stats}
            entry["style"] = style_counts(text)
            if f["nemo"] is not None:
                entry["vs_nemo_raw"] = word_counts(f["nemo"], text)
                entry["vs_nemo_normalized"] = word_counts(normalize(f["nemo"]), normalize(text))
                entry["nemo_style"] = style_counts(f["nemo"])
            if f["text"]:
                entry["vs_reference_raw"] = word_counts(f["text"], text)
                entry["vs_reference_normalized"] = word_counts(normalize(f["text"]), normalize(text))
                entry["reference_style"] = style_counts(f["text"])
            seconds = len(f["audio"]) / pk.SAMPLE_RATE
            entry["last_sentence_end_seconds"] = last_sentence_end_seconds(tokens, vocab)
            if seconds > 120:
                entry["sentence_ends_per_minute"] = sentence_ends_per_minute(tokens, vocab, seconds)
            entries.append(entry)
            summary = {k: entry.get(k) for k in ("vs_nemo_normalized", "vs_reference_normalized")}
            print(
                f"[{variant.name}] {config.name} {f['name']}: {stats['seconds']} s, {stats['windows']} windows, "
                f"style {entry['style']}, {json.dumps(summary)}",
                flush=True,
            )
        report["results"][config.name] = {"fixtures": entries, "groups": summarize(entries)}
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    report["load_seconds"] = models.load_seconds
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print_table(report)
    print(f"[{variant.name}] report: {report_path}")


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in sorted({e["group"] for e in entries}):
        items = [e for e in entries if e["group"] == group]
        summary: dict[str, Any] = {}
        for key in ("vs_nemo_raw", "vs_nemo_normalized", "vs_reference_raw", "vs_reference_normalized"):
            present = [e[key] for e in items if key in e]
            if present:
                counts = sum_counts(present)
                summary[key] = {**counts, "wer": rate(counts)}
        style = sum_counts([e["style"] for e in items])
        summary["style"] = style
        for reference in ("nemo_style", "reference_style"):
            present = [e[reference] for e in items if reference in e]
            if present:
                ref = sum_counts(present)
                summary[f"retention_vs_{reference.removesuffix('_style')}"] = {
                    k: (round(style[k] / ref[k], 3) if ref[k] else None) for k in ref
                }
        summary["stats"] = sum_counts([{k: v for k, v in e["stats"].items() if isinstance(v, int)} for e in items])
        groups[group] = summary
    return groups


def print_table(report: dict[str, Any]) -> None:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.2f}%"

    print(
        "\n| Config | Group | WER vs NeMo raw / norm | WER vs ref norm | S / D / I vs ref norm | Punct / caps vs NeMo |"
    )
    print("| --- | --- | --- | --- | --- | --- |")
    for name, result in report["results"].items():
        for group, s in result["groups"].items():
            raw = s.get("vs_nemo_raw", {}).get("wer")
            norm = s.get("vs_nemo_normalized", {}).get("wer")
            ref = s.get("vs_reference_normalized", {})
            retention = s.get("retention_vs_nemo") or s.get("retention_vs_reference") or {}
            print(
                f"| {name} | {group} | {pct(raw)} / {pct(norm)} | {pct(ref.get('wer'))} | "
                f"{ref.get('substitutions', '-')} / {ref.get('deletions', '-')} / {ref.get('insertions', '-')} | "
                f"{retention.get('punctuation')} / {retention.get('capitalized')} |"
            )


if __name__ == "__main__":
    main()
