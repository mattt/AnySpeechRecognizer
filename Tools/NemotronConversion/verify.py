# SPDX-License-Identifier: Apache-2.0

"""Verify converted Nemotron Core ML models chunk by chunk against NeMo.

Usage:
    uv run python verify.py --artifacts build/nemotron --compute-units CPU_AND_NE CPU_ONLY
    uv run python verify.py --artifacts build/nemotron --fast-reference --audio clip.wav=de-DE

The fixtures are rendered with `say`; `--audio` adds files with a locale or `auto`.
For each chunk size in the encoder and each fixture, the report compares:

1. Interface: the names, types, and shapes of each function's outputs.
2. Encoder steps: each Core ML step, given the PyTorch reference's caches, against the
   PyTorch step, and the caches it returns.
3. Streaming: Core ML steps that carry their own caches through the whole recording.
4. Decoder and joint: the Core ML prediction network along the reference's tokens, and the
   Core ML joint decision on every joint input of the reference's greedy decoding.
5. Transcripts: greedy decoding over the Core ML models against the PyTorch reference,
   and with `--nemo`, the reference against NeMo's `conformer_stream_step`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np

import nemotron as nt

FIXTURES = (
    (
        "en-samantha",
        "Samantha",
        "en-US",
        (
            "The quick brown fox jumps over the lazy dog. Streaming recognition should keep up with "
            "natural speech, even when sentences run long."
        ),
    ),
    (
        "en-daniel",
        "Daniel",
        "en-US",
        (
            "Please send the quarterly report to the finance team by Friday afternoon, and copy "
            "everyone who attended the planning meeting."
        ),
    ),
    (
        "de-anna",
        "Anna",
        "de-DE",
        "Guten Morgen. Heute besprechen wir den Zeitplan für das nächste Quartal und die offenen Fragen.",
    ),
    (
        "fr-thomas",
        "Thomas",
        "fr-FR",
        "Bonjour à tous. Nous allons parler du calendrier et des prochaines étapes du projet.",
    ),
    ("ja-kyoko", "Kyoko", "ja-JP", "今日は良い天気ですね。会議は午後三時から始まります。"),
    (
        "en-long",
        "Samantha",
        "auto",
        (
            "Cache-aware streaming keeps the attention and convolution caches of every layer between steps, "
            "so each chunk of audio is encoded once. The attention cache holds fifty-six frames, about four and a "
            "half seconds, so this clip runs long enough for the cache to fill and then roll forward several times. "
            "After that, the encoder output should still match the reference, and the transcript should not drift. "
            "Finally, the last chunk is shorter than the others, which checks the valid length of the input."
        ),
    ),
)


def render_fixtures(directory: Path) -> list[tuple[str, Path, str, str]]:
    """Render the fixtures with `say` as 16 kHz mono WAV files."""
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for name, voice, locale, text in FIXTURES:
        path = directory / f"{name}.wav"
        if not path.exists():
            aiff = directory / f"{name}.aiff"
            subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(path)], check=True
            )
            aiff.unlink()
        fixtures.append((name, path, locale, text))
    return fixtures


class CoreMLSteps(nt.Steps):
    def __init__(self, directory: Path, chunk: nt.Chunk, units: ct.ComputeUnit, cache_dtype: str) -> None:
        self.cache_dtype = np.dtype(cache_dtype)
        self.encoder = ct.models.CompiledMLModel(
            str(directory / nt.ENCODER_FILE), compute_units=units, function_name=chunk.function
        )
        self.decoder = ct.models.CompiledMLModel(str(directory / nt.DECODER_FILE), compute_units=units)
        self.joint_decision = ct.models.CompiledMLModel(str(directory / nt.JOINT_FILE), compute_units=units)

    def encode(self, inputs):
        inputs = {
            name: value.astype(self.cache_dtype) if name in ("cache_channel", "cache_time") else value
            for name, value in inputs.items()
        }
        outputs = self.encoder.predict(inputs)
        return {
            name: np.asarray(value).astype(np.float32) if np.asarray(value).dtype == np.float16 else np.asarray(value)
            for name, value in outputs.items()
        }

    def predict(self, token, h, c):
        out = self.decoder.predict(
            {
                "targets": np.array([[token]], dtype=np.int32),
                "target_length": np.array([1], dtype=np.int32),
                "h_in": h,
                "c_in": c,
            }
        )
        return out["decoder"].astype(np.float32), out["h_out"].astype(np.float32), out["c_out"].astype(np.float32)

    def joint(self, frame, decoder):
        out = self.joint_decision.predict({"encoder_step": frame, "decoder_step": decoder})
        return int(np.asarray(out["token_id"]).reshape(-1)[0]), float(np.asarray(out["token_prob"]).reshape(-1)[0])


def compare(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Error statistics over `[channels, frames]` arrays, with cosine similarity per frame."""
    difference = np.abs(reference - actual)
    dot = np.sum(reference * actual, axis=0)
    norms = np.linalg.norm(reference, axis=0) * np.linalg.norm(actual, axis=0)
    cosine = dot / np.maximum(norms, 1e-12)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "relative_l2": float(np.linalg.norm(reference - actual) / max(np.linalg.norm(reference), 1e-12)),
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
    }


def summarize(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        "max_abs": max(v["max_abs"] for v in values),
        "mean_abs": float(np.mean([v["mean_abs"] for v in values])),
        "relative_l2": float(np.mean([v["relative_l2"] for v in values])),
        "mean_cosine": float(np.mean([v["mean_cosine"] for v in values])),
        "min_cosine": min(v["min_cosine"] for v in values),
    }


def normalized_words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def error_rate(reference: str, hypothesis: str, characters: bool = False) -> float:
    import jiwer

    if characters:
        reference_chars = " ".join("".join(normalized_words(reference)))
        hypothesis_chars = " ".join("".join(normalized_words(hypothesis)))
        return float(jiwer.wer(reference_chars, hypothesis_chars)) if reference_chars else 0.0
    words = " ".join(normalized_words(reference))
    return float(jiwer.wer(words, " ".join(normalized_words(hypothesis)))) if words else 0.0


def nemo_tokens(model: Any, mel: np.ndarray, chunk: nt.Chunk, locale: str) -> list[int]:
    """Tokens of NeMo's own cache-aware streaming loop with `pad_and_drop_preencoded`."""
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    nt.set_chunk(model, chunk)
    model.set_inference_prompt(locale)
    buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=True)
    buffer.append_processed_signal(torch.from_numpy(mel)[None], stream_id=-1)
    channel, time_cache, channel_len = model.encoder.get_initial_cache_state(batch_size=1)
    hypotheses = predictions = None
    for chunk_audio, lengths in buffer:
        with torch.inference_mode():
            (predictions, _, channel, time_cache, channel_len, hypotheses) = model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=lengths,
                cache_last_channel=channel,
                cache_last_time=time_cache,
                cache_last_channel_len=channel_len,
                keep_all_outputs=buffer.is_buffer_empty(),
                previous_hypotheses=hypotheses,
                previous_pred_out=predictions,
                drop_extra_pre_encoded=nt.DROPPED_PRE_ENCODED_FRAMES,
                return_transcription=True,
            )
    return [int(t) for t in hypotheses[0].y_sequence]


@dataclass
class Reference:
    """One fixture's PyTorch reference stream for one chunk size."""

    name: str
    locale: str
    text: str
    samples: int
    mel: np.ndarray
    stream: nt.StreamResult
    seconds: float
    nemo_tokens: list[int] | None


def references(
    model: Any, chunk: nt.Chunk, fixtures: list[tuple[str, Path, str, str]], prompts: dict[str, int], run_nemo: bool
) -> list[Reference]:
    """The PyTorch reference, and with `run_nemo` NeMo's own loop, for each fixture.

    PyTorch runs before the verifier loads any Core ML model: on macOS 27.0.1, PyTorch's CPU
    `Linear` crashed with a segmentation fault after Core ML predictions in the same process.
    """
    nt.set_chunk(model, chunk)
    steps = nt.TorchSteps(model)
    result = []
    for name, path, locale, text in fixtures:
        audio = nt.load_audio(path)
        mel = nt.nemo_mel(model, audio)
        started = time.monotonic()
        stream = nt.stream(steps, mel, chunk, prompts[locale], keep_caches=True, trace=True)
        seconds = time.monotonic() - started
        nemo = nemo_tokens(model, mel, chunk, locale) if run_nemo else None
        result.append(Reference(name, locale, text, len(audio), mel, stream, seconds, nemo))
    return result


def verify_chunk(
    model: Any,
    directory: Path,
    chunk: nt.Chunk,
    units: str,
    fixtures: list[Reference],
    prompts: dict[str, int],
    cache_dtype: str,
) -> dict[str, Any]:
    coreml = CoreMLSteps(directory, chunk, getattr(ct.ComputeUnit, units), cache_dtype)
    record: dict[str, Any] = {"chunk_ms": chunk.milliseconds, "compute_units": units, "fixtures": []}

    # Interface: one step with zero caches.
    contract = chunk.contract(cache_dtype)
    inputs = {
        **nt.initial_caches(),
        "mel": np.zeros((1, nt.MEL_BINS, chunk.input_frames), np.float32),
        "mel_length": np.array([chunk.input_frames], np.int32),
        "prompt": nt.prompt_vector(0).numpy(),
    }
    raw = coreml.encoder.predict(
        {k: v.astype(coreml.cache_dtype) if k in ("cache_channel", "cache_time") else v for k, v in inputs.items()}
    )
    interface = {name: (str(np.asarray(value).dtype), tuple(np.asarray(value).shape)) for name, value in raw.items()}
    expected = {name: (dtype, shape) for name, (dtype, shape) in contract["outputs"].items()}
    record["interface_matches"] = interface == expected
    if not record["interface_matches"]:
        record["interface"] = {"actual": interface, "expected": expected}

    all_steps, all_closed, all_channel, decoder_errors = [], [], [], []
    joint_agree = joint_total = 0
    for fixture_reference in fixtures:
        name, locale, text, mel = (
            fixture_reference.name,
            fixture_reference.locale,
            fixture_reference.text,
            fixture_reference.mel,
        )
        prompt = prompts[locale]
        reference = fixture_reference.stream
        # Each Core ML step with the reference's caches.
        inputs = [nt.initial_caches()] + reference.caches[:-1]
        isolated = nt.stream(coreml, mel, chunk, prompt, encoder_inputs=inputs, keep_caches=True)
        steps = [compare(r, a) for r, a in zip(reference.encoder, isolated.encoder, strict=True)]
        # The valid frames of the attention cache that each step returns, layer by layer.
        channel = []
        for r, a in zip(reference.caches, isolated.caches, strict=True):
            valid = int(r["cache_length"][0])
            if valid > 0:
                channel.append(
                    compare(
                        r["cache_channel"][:, 0, -valid:, :].reshape(-1, nt.ENCODER_DIM).T,
                        a["cache_channel"][:, 0, -valid:, :].reshape(-1, nt.ENCODER_DIM).T,
                    )
                )
        # Core ML with its own caches.
        started = time.monotonic()
        closed = nt.stream(coreml, mel, chunk, prompt)
        coreml_seconds = time.monotonic() - started
        drift = [compare(r, a) for r, a in zip(reference.encoder, closed.encoder, strict=True)]
        # Prediction network along the reference tokens, and the joint on the reference's inputs.
        for token, h, c, output in reference.predictions:
            actual, _, _ = coreml.predict(token, h, c)
            decoder_errors.append(float(np.abs(actual - output).max()))
        agree = sum(coreml.joint(frame, decoder)[0] == token for frame, decoder, token in reference.joint_calls)
        joint_agree += agree
        joint_total += len(reference.joint_calls)

        reference_ids = [t for t, _, _ in reference.tokens]
        coreml_ids = [t for t, _, _ in closed.tokens]
        reference_text = nt.text(model, reference_ids)
        coreml_text = nt.text(model, coreml_ids)
        characters = locale.startswith(("ja", "zh"))
        fixture = {
            "name": name,
            "locale": locale,
            "seconds": round(fixture_reference.samples / nt.SAMPLE_RATE, 2),
            "steps": len(reference.encoder),
            "isolated_steps": summarize(steps),
            "isolated_cache_channel": summarize(channel) if channel else None,
            "streaming": summarize(drift),
            "streaming_last_step_cosine": drift[-1]["mean_cosine"],
            "tokens_equal": coreml_ids == reference_ids,
            "reference_tokens": len(reference_ids),
            "coreml_tokens": len(coreml_ids),
            "joint_agreement": [int(agree), len(reference.joint_calls)],
            "error_rate_vs_reference": error_rate(reference_text, coreml_text, characters),
            "error_rate_vs_text": error_rate(text, coreml_text, characters),
            "reference_error_rate_vs_text": error_rate(text, reference_text, characters),
            "reference_text": reference_text,
            "coreml_text": coreml_text,
            "reference_seconds": round(fixture_reference.seconds, 1),
            "coreml_seconds": round(coreml_seconds, 1),
        }
        nemo_ids = fixture_reference.nemo_tokens
        if nemo_ids is not None:
            fixture["reference_equals_nemo"] = nemo_ids == reference_ids
            if nemo_ids != reference_ids:
                fixture["nemo_text"] = nt.text(model, nemo_ids)
        record["fixtures"].append(fixture)
        all_steps += steps
        all_closed += drift
        all_channel += channel
        print(
            f"[{chunk.milliseconds} ms {units}] {name}: step cosine {fixture['isolated_steps']['mean_cosine']:.5f} "
            f"(min {fixture['isolated_steps']['min_cosine']:.4f}), streaming {fixture['streaming']['mean_cosine']:.5f}, "
            f"tokens equal {fixture['tokens_equal']}, error vs reference {fixture['error_rate_vs_reference']:.2%}",
            flush=True,
        )
    record["isolated_steps"] = summarize(all_steps)
    record["isolated_cache_channel"] = summarize(all_channel) if all_channel else None
    record["streaming"] = summarize(all_closed)
    record["decoder_max_abs"] = max(decoder_errors) if decoder_errors else None
    record["joint_agreement"] = [int(joint_agree), joint_total]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=nt.BUILD_DIR / "nemotron")
    parser.add_argument("--cache-dir", type=Path, default=nt.DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--compute-units", nargs="+", default=["CPU_AND_NE"], choices=["CPU_AND_NE", "CPU_ONLY", "CPU_AND_GPU", "ALL"]
    )
    parser.add_argument("--chunks", type=int, nargs="+", help="Chunk sizes to check (default: every function)")
    parser.add_argument(
        "--audio", nargs="*", default=[], metavar="FILE=LOCALE", help="More recordings, with a locale or auto"
    )
    parser.add_argument("--fast-reference", action="store_true", help="Batch NeMo's depthwise convolutions on the CPU")
    parser.add_argument(
        "--nemo", action="store_true", help="Also compare the reference with NeMo's conformer_stream_step"
    )
    parser.add_argument("--report", type=Path, default=nt.BUILD_DIR / "reports" / "verify.json")
    args = parser.parse_args()

    record = json.loads((args.artifacts / nt.CONVERSION_FILE).read_text())
    cache_dtype = record["conversion"]["streaming"].get("cache_dtype", "float32")
    functions = [f["chunk_frames"] for f in record["conversion"]["components"]["encoder"]["functions"]]
    available = {nt.Chunk(frames - 1).milliseconds for frames in functions}
    chunks = [nt.CHUNKS[ms] for ms in sorted(args.chunks or available) if ms in available]
    if args.fast_reference:
        nt.use_fast_depthwise_convolution()
    model, _ = nt.load_model(args.cache_dir.resolve())
    prompts = nt.prompt_dictionary(model)
    fixtures = render_fixtures(nt.BUILD_DIR / "fixtures")
    for item in args.audio:
        path, _, locale = item.partition("=")
        fixtures.append((Path(path).stem, Path(path), locale or "auto", ""))

    report = {
        "artifacts": str(args.artifacts),
        "tools": nt.tool_versions(),
        "fast_reference": args.fast_reference,
        "results": [],
    }
    computed = {chunk: references(model, chunk, fixtures, prompts, args.nemo) for chunk in chunks}
    for chunk in chunks:
        for units in args.compute_units:
            report["results"].append(
                verify_chunk(model, args.artifacts, chunk, units, computed[chunk], prompts, cache_dtype)
            )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(
        "\n| Chunk | Compute units | Step max / mean abs error | Step cosine, mean / min | Streaming cosine, mean / min | Decoder max abs | Joint agreement | Fixtures with equal tokens |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for result in report["results"]:
        steps, streaming = result["isolated_steps"], result["streaming"]
        equal = sum(f["tokens_equal"] for f in result["fixtures"])
        print(
            f"| {result['chunk_ms']} ms | `{result['compute_units']}` | {steps['max_abs']:.3f} / {steps['mean_abs']:.5f} "
            f"| {steps['mean_cosine']:.5f} / {steps['min_cosine']:.4f} | {streaming['mean_cosine']:.5f} / {streaming['min_cosine']:.4f} "
            f"| {result['decoder_max_abs']:.4f} | {result['joint_agreement'][0]} / {result['joint_agreement'][1]} "
            f"| {equal} / {len(result['fixtures'])} |"
        )
    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
