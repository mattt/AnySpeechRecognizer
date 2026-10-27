# SPDX-License-Identifier: Apache-2.0
"""Transcribe a natural-speech evaluation subset with the transformers reference.

The runs have the layout and records of `Tools/natural-speech-eval.py run`, so its `score`
command reports them beside the Core ML runs:

    uv run python reference_runs.py --root /tmp/natural-eval --set fleurs-zh --variant 0.6b
    uv run ../natural-speech-eval.py score --root /tmp/natural-eval

Each file is transcribed in one call, as `generate()` with greedy decoding, without the runtime's
segmentation; the original library splits only audio longer than 20 minutes.
Times are the elapsed time of feature extraction and generation; peak memory is the process's.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import soundfile
import torch

import qwen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="The evaluation scratch directory.")
    parser.add_argument("--set", required=True, help="A subset name, such as fleurs-zh.")
    parser.add_argument("--variant", choices=sorted(qwen.VARIANTS), default="0.6b")
    parser.add_argument("--label", help="The run directory name (default: transformers-<variant>).")
    parser.add_argument("--device", default="cpu", help="The PyTorch device, such as cpu or mps.")
    parser.add_argument("--threads", type=int, help="CPU threads for PyTorch.")
    parser.add_argument("--limit", type=int, help="Only the first files of the subset.")
    parser.add_argument("--cache-dir", type=Path, default=qwen.DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    variant = qwen.VARIANTS[args.variant]
    model, processor = qwen.load_reference(qwen.snapshot(variant, args.cache_dir))
    model.to(args.device)
    lines = (args.root / "subsets" / f"{args.set}.tsv").read_text().splitlines()[1:]
    output = args.root / "runs" / (args.label or f"transformers-{args.variant}") / args.set
    output.mkdir(parents=True, exist_ok=True)
    for index, line in enumerate(lines[: args.limit] if args.limit else lines):
        identifier, audio_path, seconds, _ = line.split("\t")
        destination = output / f"{identifier}.jsonl"
        if destination.exists():
            continue
        audio, rate = soundfile.read(str(args.root / audio_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != qwen.SAMPLE_RATE:
            raise ValueError(f"{audio_path} is not 16 kHz.")
        started = time.monotonic()
        inputs = processor.apply_transcription_request(audio=np.ascontiguousarray(audio)).to(args.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=4096)
        text = processor.decode(generated[:, inputs["input_ids"].shape[1] :], return_format="transcription_only")[0]
        elapsed = time.monotonic() - started
        duration = float(seconds)
        records = [
            {
                "type": "metadata",
                "diagnostics": {"provider": "transformers", "modelID": f"{variant.repo}@{variant.revision}"},
                "audio": {"durationSeconds": duration},
                "preparationSeconds": 0.0,
            },
            {
                "type": "result",
                "text": text,
                "isFinal": True,
                "startSeconds": 0.0,
                "endSeconds": duration,
                "finalizationSeconds": duration,
                "words": [],
            },
            {
                "type": "evaluation",
                "elapsedSeconds": elapsed,
                # ru_maxrss is in bytes on macOS.
                "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "finalSourceSeconds": duration,
            },
            {"type": "run", "wallSeconds": elapsed, "command": ["reference_runs.py", args.set]},
        ]
        destination.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        print(f"[{index + 1}/{len(lines)}] {identifier} {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
