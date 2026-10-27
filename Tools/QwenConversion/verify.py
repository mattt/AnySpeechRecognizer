# SPDX-License-Identifier: Apache-2.0
"""Verify converted Qwen3-ASR Core ML models against the transformers reference.

Usage:
    uv run python verify.py --variant 0.6b --compute-units CPU_AND_GPU CPU_ONLY
    uv run python verify.py --artifacts build/0.6b-float16 --audio clip.wav=clip.txt

For each fixture, the verifier computes transformers' log-mel features and checks:

1. Encoder: the Core ML audio embeddings against `get_audio_features`, over valid tokens.
2. Decoder: with the reference audio embeddings, the Core ML decoder's greedy token after each
   prefix of the reference output (teacher forcing), and its log probability.
3. Transcripts: greedy decoding with the Core ML encoder and decoder against `generate()`.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile
import torch

import qwen

# Text, voice, and language of each synthetic fixture.
FIXTURES = {
    "en": (
        "Samantha",
        (
            "The quick brown fox jumps over the lazy dog. Speech recognition on device keeps "
            "audio private and works without a network connection."
        ),
    ),
    "en-gb": (
        "Daniel",
        (
            "Please call Stella. Ask her to bring these things with her from the store: six spoons "
            "of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob."
        ),
    ),
    "en-long": (
        "Samantha",
        (
            "Speech recognition has improved quickly over the last few years. Large models "
            "trained on many languages can now transcribe conversations, lectures, and meetings with "
            "few errors. Running these models on a phone or a laptop keeps recordings private, works "
            "without a network, and avoids the delay of sending audio to a server. The challenge is to "
            "fit a model with hundreds of millions of parameters into the memory and power budget of a "
            "small device."
        ),
    ),
    "zh": ("Tingting", "今天天气很好，我们一起去公园散步吧。语音识别技术在手机上运行得越来越快。"),
    "ja": (
        "Kyoko",
        "今日はとても良い天気ですね。音声認識の技術は、スマートフォンの上でもますます速く動くようになりました。",
    ),
    "ko": ("Yuna", "오늘은 날씨가 정말 좋네요. 음성 인식 기술은 휴대폰에서도 점점 더 빠르게 작동합니다."),
    "yue": ("Sinji", "今日天氣好好，我哋一齊去公園行吓啦。"),
    "de": (
        "Anna",
        (
            "Für die besten Aussichten auf Hongkong sollten Sie die Insel verlassen und zum "
            "gegenüberliegenden Ufer von Kowloon fahren."
        ),
    ),
}

MAX_NEW_TOKENS = 400


def render_fixtures() -> list[tuple[str, Path]]:
    qwen.FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (voice, text) in FIXTURES.items():
        path = qwen.FIXTURES_DIR / f"{name}.wav"
        if not path.exists():
            subprocess.run(["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text], check=True)
        paths.append((name, path))
    return paths


def load_audio(path: Path) -> np.ndarray:
    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != qwen.SAMPLE_RATE:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, qwen.SAMPLE_RATE, rate).astype(np.float32)
    return audio


def token_count(frames: int) -> int:
    whole, rest = divmod(frames, qwen.CHUNK_FRAMES)
    return whole * qwen.TOKENS_PER_CHUNK + (rest + 7) // 8


def coreml_audio_embeddings(encoder, features: np.ndarray, valid: int) -> np.ndarray:
    """Runs the window encoder over [128, F] features with `valid` frames; returns [N, D]."""
    outputs = []
    for start in range(0, valid, qwen.WINDOW_FRAMES):
        length = min(valid - start, qwen.WINDOW_FRAMES)
        window = np.zeros((1, qwen.MEL_BINS, qwen.WINDOW_FRAMES), dtype=np.float32)
        window[0, :, :length] = features[:, start : start + length]
        result = encoder.predict({"mel": window, "mel_length": np.array([length], dtype=np.int32)})
        outputs.append(result["audio_embeddings"][0, : token_count(length)])
    return np.concatenate(outputs)


class CoreMLDecoder(qwen.BlockFeeder):
    def __init__(self, path: Path, units: ct.ComputeUnit, config: dict):
        self.model = ct.models.CompiledMLModel(str(path), compute_units=units)
        super().__init__(
            self.run,
            config["block_length"],
            config["context_length"],
            config["hidden_size"],
            config["special_tokens"]["end_of_text"],
        )

    def start(self) -> None:
        super().start()
        self.state = self.model.make_state()

    def run(self, input_ids, audio_embeddings, position_ids, causal_mask):
        inputs = {
            "input_ids": input_ids,
            "audio_embeddings": audio_embeddings,
            "position_ids": position_ids,
            "causal_mask": causal_mask,
        }
        out = self.model.predict(inputs, state=self.state)
        return out["token_ids"], out["token_log_probs"]


class TorchEncoder:
    """The PyTorch encoder wrapper with the compiled encoder's `predict` interface."""

    def __init__(self, model):
        self.wrapper = qwen.EncoderWindow(model).eval()

    def predict(self, inputs: dict) -> dict:
        with torch.no_grad():
            out = self.wrapper(torch.from_numpy(inputs["mel"]), torch.from_numpy(inputs["mel_length"]))
        return {"audio_embeddings": out.numpy()}


class TorchDecoder(qwen.BlockFeeder):
    """The PyTorch wrapper that the converter traces, for separating wrapper and conversion error."""

    def __init__(self, model, config: dict):
        self.wrapper = qwen.TextDecoder(model, config["context_length"], config["block_length"]).eval()
        super().__init__(
            self.run,
            config["block_length"],
            config["context_length"],
            config["hidden_size"],
            config["special_tokens"]["end_of_text"],
        )

    def start(self) -> None:
        super().start()
        self.wrapper.reset()

    def run(self, input_ids, audio_embeddings, position_ids, causal_mask):
        with torch.no_grad():
            tokens, log_probs = self.wrapper(
                torch.from_numpy(input_ids).long(),
                torch.from_numpy(audio_embeddings),
                torch.from_numpy(position_ids).long(),
                torch.from_numpy(causal_mask),
            )
        return tokens.numpy(), log_probs.numpy()


def prompt(config: dict, audio_tokens: int) -> tuple[list[int], slice]:
    prefix, suffix = config["prompt"]["prefix"], config["prompt"]["suffix"]
    ids = prefix + [config["special_tokens"]["audio_pad"]] * audio_tokens + suffix
    return ids, slice(len(prefix), len(prefix) + audio_tokens)


def greedy(decoder: qwen.BlockFeeder, config: dict, audio: np.ndarray) -> list[int]:
    ids, span = prompt(config, len(audio))
    embeddings = np.zeros((len(ids), decoder.hidden), dtype=np.float32)
    embeddings[span] = audio
    decoder.start()
    token, _ = decoder.feed(ids, embeddings)
    tokens = [token]
    while tokens[-1] not in config["end_tokens"] and len(tokens) < MAX_NEW_TOKENS:
        token, _ = decoder.feed([tokens[-1]])
        tokens.append(token)
    return tokens


def teacher_forced(decoder: qwen.BlockFeeder, config: dict, audio: np.ndarray, reference: list[int]):
    """The decoder's greedy token after each reference prefix, and the log probabilities."""
    ids, span = prompt(config, len(audio))
    embeddings = np.zeros((len(ids), decoder.hidden), dtype=np.float32)
    embeddings[span] = audio
    decoder.start()
    predicted = [decoder.feed(ids, embeddings)]
    for token in reference[:-1]:
        predicted.append(decoder.feed([token]))
    return predicted


def error_rate(reference: list, hypothesis: list) -> float:
    import jiwer

    if not reference:
        return float(bool(hypothesis))
    return jiwer.wer(" ".join(map(str, reference)), " ".join(map(str, hypothesis))) if hypothesis else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", choices=sorted(qwen.VARIANTS), default="0.6b")
    parser.add_argument("--artifacts", type=Path, help="The converted directory (default: build/<variant>).")
    parser.add_argument(
        "--compute-units",
        nargs="+",
        default=["CPU_AND_GPU"],
        choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL", "WRAPPER"],
        help="Core ML compute units; WRAPPER runs the traced PyTorch wrappers instead.",
    )
    parser.add_argument("--audio", nargs="*", default=[], help="More audio files, as path or path=reference.txt.")
    parser.add_argument("--cache-dir", type=Path, default=qwen.DEFAULT_CACHE_DIR)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    variant = qwen.VARIANTS[args.variant]
    artifacts = (args.artifacts or qwen.BUILD_DIR / variant.name).resolve()
    import json

    config = json.loads((artifacts / qwen.CONFIG_FILE).read_text())
    model, processor = qwen.load_reference(qwen.snapshot(variant, args.cache_dir))
    tokenizer = processor.tokenizer

    fixtures = render_fixtures()
    for item in args.audio:
        path = Path(item.split("=", 1)[0])
        fixtures.append((path.stem, path))

    # Reference features, audio embeddings, and generate() tokens.
    references = {}
    for name, path in fixtures:
        audio = load_audio(path)
        inputs = processor.apply_transcription_request(audio=audio)
        with torch.no_grad():
            features = model.get_audio_features(inputs["input_features"], inputs["input_features_mask"])
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        tokens = output[0, inputs["input_ids"].shape[1] :].tolist()
        references[name] = {
            "features": inputs["input_features"][0].numpy(),
            "valid": int(inputs["input_features_mask"].sum()),
            "embeddings": features.pooler_output.numpy(),
            "tokens": tokens,
            "seconds": len(audio) / qwen.SAMPLE_RATE,
        }
        print(f"{name}: {tokenizer.decode(tokens)!r}")

    report = {"artifacts": str(artifacts), "source": f"{variant.repo}@{variant.revision}", "units": {}}
    for units in args.compute_units:
        unit = getattr(ct.ComputeUnit, units, None)
        started = time.monotonic()
        if units == "WRAPPER":
            encoder, decoder = TorchEncoder(model), TorchDecoder(model, config)
        else:
            encoder = ct.models.CompiledMLModel(str(artifacts / qwen.ENCODER_FILE), compute_units=unit)
            decoder = CoreMLDecoder(artifacts / qwen.DECODER_FILE, unit, config)
        print(f"\n{units}: loaded in {time.monotonic() - started:.1f} s")
        rows = []
        for name, _ in fixtures:
            reference = references[name]
            embeddings = coreml_audio_embeddings(encoder, reference["features"], reference["valid"])
            expected = reference["embeddings"]
            difference = np.abs(embeddings - expected)
            cosines = (embeddings * expected).sum(-1) / (
                np.linalg.norm(embeddings, axis=-1) * np.linalg.norm(expected, axis=-1)
            )
            forced = teacher_forced(decoder, config, expected, reference["tokens"])
            agreement = sum(p == t for (p, _), t in zip(forced, reference["tokens"]))
            started = time.monotonic()
            tokens = greedy(decoder, config, embeddings)
            elapsed = time.monotonic() - started
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            row = {
                "fixture": name,
                "seconds": reference["seconds"],
                "audio_tokens": len(expected),
                "encoder_max_abs": float(difference.max()),
                "encoder_mean_abs": float(difference.mean()),
                "encoder_reference_mean_abs": float(np.abs(expected).mean()),
                "encoder_mean_cosine": float(cosines.mean()),
                "encoder_min_cosine": float(cosines.min()),
                "forced_agreement": [agreement, len(reference["tokens"])],
                "tokens_match": tokens == reference["tokens"],
                "token_error_rate": error_rate(reference["tokens"], tokens),
                "decode_seconds": elapsed,
                "text": text,
            }
            rows.append(row)
            print(
                f"{name:8} enc max {row['encoder_max_abs']:.4f} mean {row['encoder_mean_abs']:.5f} "
                f"cos {row['encoder_mean_cosine']:.5f}/{row['encoder_min_cosine']:.4f} "
                f"forced {agreement}/{len(reference['tokens'])} match {row['tokens_match']} "
                f"TER {row['token_error_rate']:.2%} {elapsed:.2f} s"
            )
            if not row["tokens_match"]:
                print(f"         reference {tokenizer.decode(reference['tokens'])!r}")
                print(f"         coreml    {text!r}")
        report["units"][units] = rows
    destination = args.report or qwen.REPORTS_DIR / f"verify-{artifacts.name}.json"
    qwen.write_json(destination, report)
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
