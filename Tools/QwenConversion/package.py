# SPDX-License-Identifier: Apache-2.0
"""Package verified Qwen3-ASR Core ML artifacts for publication on Hugging Face.

Usage:
    uv run python package.py --variant 1.7b --trace-dtype float16 --output /tmp/qwen-1.7b-publish

The package contains:

- The converted models, vocabulary, and configuration from build/<variant>, copied unchanged
  after their sizes and digests are checked against that directory's conversion.json.
- LICENSE: the Apache License text as the Qwen3-ASR repository distributes it.
- NOTICE: the source, the credit, and the changes, as section 4(b) of the License requires.
- README.md: the model card, with the evaluation in evaluation/<variant>.json
  and the verification reports in build/reports.
- conversion.json: the conversion record, with the size and SHA-256 digest of every file
  except conversion.json and README.md.

Nothing is uploaded. Check the package with upload_dry_run.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import qwen

MODEL_FILES = (qwen.ENCODER_FILE, qwen.DECODER_FILE, qwen.VOCAB_FILE, qwen.CONFIG_FILE)
UNLISTED = (qwen.CONVERSION_FILE, qwen.MODEL_CARD_FILE)
SET_NAMES = {
    "fleurs-zh": "FLEURS Mandarin",
    "fleurs-ja": "FLEURS Japanese",
    "fleurs-ko": "FLEURS Korean",
    "fleurs-yue": "FLEURS Cantonese",
    "ls-clean-utt": "LibriSpeech test-clean",
    "ls-other-utt": "LibriSpeech test-other",
}
APP_REPOSITORY = "https://github.com/mattt/AnySpeechRecognizer"


def check_artifacts(artifacts: Path, record: dict[str, Any]) -> list[str]:
    """Problems with the artifacts against their conversion record."""
    problems = []
    for entry in record["files"]:
        path = artifacts / entry["path"]
        if entry["path"] == qwen.MODEL_CARD_FILE:
            continue
        if not path.is_file():
            problems.append(f"missing {entry['path']}")
        elif path.stat().st_size != entry["size"] or qwen.sha256(path) != entry["sha256"]:
            problems.append(f"digest mismatch {entry['path']}")
    return problems


def describe_size(size: int) -> str:
    """A decimal size: kilobytes below one megabyte, megabytes otherwise."""
    return f"{size / 1e3:,.1f} kB" if size < 1_000_000 else f"{size / 1e6:,.1f} MB"


def error_rate(counts: dict[str, int]) -> float:
    return (counts["S"] + counts["D"] + counts["I"]) / counts["N"]


def evaluation_table(evaluation: dict[str, Any], label: str, reference: str) -> list[str]:
    results = evaluation["results"]
    rows = [
        (
            "| Test set | Measure | Error rate | Substitutions / deletions / insertions | "
            "Transformers, same files | Real-time factor |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for subset, name in SET_NAMES.items():
        result = results.get(f"{label}/{subset}")
        if result is None:
            continue
        counts = result["basic"]
        measure = "WER" if subset.startswith("ls-") else "CER"
        other = results.get(f"{reference}/{subset}")
        reference_rate = f"{error_rate(other['basic']):.2%}" if other else "not run"
        rows.append(
            f"| {name} ({result['files']} files) | {measure} | {error_rate(counts):.2%} "
            f"| {counts['S']} / {counts['D']} / {counts['I']} of {counts['N']:,} "
            f"| {reference_rate} | {result['analysisRTF']:.2f} |"
        )
    return rows


def verification_summary(reports: list[Path]) -> list[str]:
    rows = [
        (
            "| Model | Clips | Encoder mean / lowest token cosine | Decoder tokens that match, teacher-forced "
            "| Transcripts identical to `generate()` |"
        ),
        "| --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        data = json.loads(report.read_text())
        for units, rows_of_unit in data["units"].items():
            model = "PyTorch wrappers (before Core ML)" if units == "WRAPPER" else f"Core ML, {units}"
            groups = [
                ("synthetic speech (`say`)", [r for r in rows_of_unit if not r["fixture"][0].isdigit()]),
                ("FLEURS and LibriSpeech", [r for r in rows_of_unit if r["fixture"][0].isdigit()]),
            ]
            for name, group in groups:
                if not group:
                    continue
                agreed = sum(r["forced_agreement"][0] for r in group)
                total = sum(r["forced_agreement"][1] for r in group)
                mean = sum(r["encoder_mean_cosine"] for r in group) / len(group)
                lowest = min(r["encoder_min_cosine"] for r in group)
                identical = sum(r["tokens_match"] for r in group)
                rows.append(
                    f"| {model} | {len(group)} {name} | {mean:.5f} / {lowest:.3f} | {agreed} of {total} "
                    f"| {identical} of {len(group)} |"
                )
    return rows


def notice(variant: qwen.Variant, record: dict[str, Any]) -> str:
    conversion = record["conversion"]
    context = conversion["context_length"]
    block = conversion["block_length"]
    created = record["created"][:10]
    return f"""{variant.title} for Core ML
Copyright 2026 Alibaba Cloud

This repository contains a modified version of {variant.title} by the Qwen team,
Alibaba Cloud: https://huggingface.co/{variant.repo}
at revision {variant.revision}.
The original model is licensed under the Apache License, Version 2.0.
LICENSE contains the license text as the Qwen3-ASR repository distributes it
({qwen.LICENSE_TEXT_SOURCE}).

AnySpeechRecognizer ({APP_REPOSITORY}) changed the model on {created}
with its Tools/QwenConversion tool. The changes are:

- The model was converted from PyTorch to two Core ML programs,
  AudioEncoder.mlmodelc and TextDecoder.mlmodelc.
- The audio encoder and projector take one fixed window of 800 mel frames (8 seconds).
  In a partial window, attention ignores the padding tokens.
- The text decoder takes fixed blocks of {block} tokens and keeps its key and value cache
  for {context:,} positions in Core ML states, one key and one value state per layer.
  It writes the cache at given positions and returns the greedy next token and its
  log probability instead of logits.
- RMSNorm divides each row by its largest magnitude before it squares the values,
  so that float16 does not overflow.
- Weights are stored in float16. Weight tensors of 2,048 or more elements are quantized
  to int8 (linear symmetric, one scale per output channel), except the token embedding
  table, which the output head shares, and the rotary position tables.
- vocab.json and qwen3_asr_config.json are new files. They are derived from the original
  tokenizer and chat template.

The changes are Copyright AnySpeechRecognizer contributors and are licensed under
the Apache License, Version 2.0.
"""


def model_card(
    variant: qwen.Variant,
    record: dict[str, Any],
    files: list[dict[str, Any]],
    evaluation: dict[str, Any],
    label: str,
    reference: str,
    reports: list[Path],
    verification_note: str | None,
) -> str:
    conversion = record["conversion"]
    sizes: dict[str, int] = {}
    for entry in files:
        top = entry["path"].split("/")[0]
        sizes[top] = sizes.get(top, 0) + entry["size"]
    total = sum(sizes.values())
    results = [r for key, r in evaluation["results"].items() if key.startswith(f"{label}/")]
    peak = max(r["peakMB"] for r in results) / 1000
    rtf = sorted(r["analysisRTF"] for r in results)
    languages = sorted(qwen.LANGUAGES)
    names = ", ".join(sorted(qwen.LANGUAGES.values()))
    tools = record["tools"]
    front_matter = "\n".join(
        [
            "---",
            "license: apache-2.0",
            "license_link: LICENSE",
            f"base_model: {variant.repo}",
            "base_model_relation: quantized",
            "pipeline_tag: automatic-speech-recognition",
            "library_name: coreml",
            "language:",
            *(f"- {code}" for code in languages),
            "tags:",
            "- coreml",
            "- speech-recognition",
            "---",
        ]
    )
    file_rows = "\n".join(
        f"| `{name}` | {describe_size(size)} |"
        for name, size in sorted(sizes.items())
        if name in MODEL_FILES or name in (qwen.LICENSE_FILE, qwen.NOTICE_FILE)
    )
    return f"""{front_matter}

# {variant.title} for Core ML

This repository contains a Core ML conversion of
[{variant.title}](https://huggingface.co/{variant.repo}) by the Qwen team, Alibaba Cloud,
at revision `{variant.revision}`.
[AnySpeechRecognizer]({APP_REPOSITORY}) runs it on Apple silicon
with the same API as Apple's `SpeechTranscriber`.

## Languages

The model transcribes speech in 30 languages: {names}.
It identifies the language of each segment, or the caller can set the language.
The original model card also lists 22 Chinese dialects.
This conversion was evaluated on Mandarin, Japanese, Korean, Cantonese, and English only.

## Use

```swift
let transcriber = SpeechTranscriber(
    model: QwenSpeechModel(variant: .large),
    transcriptionOptions: [],
    reportingOptions: [],
    attributeOptions: []
)
```

A version of AnySpeechRecognizer that pins this repository downloads and verifies these files
through `AssetInventory`. Other versions can load a local copy of this repository with
`QwenSpeechModel(variant: .large, assets: .directory(url))`.

## Limitations

- The model transcribes files. It does not stream, and it gives no volatile (partial) results.
  Each final result covers one segment of up to 40 seconds of audio.
- Results have no word timings, only the time range of their segment.
  The separate Qwen3-ForcedAligner-0.6B model, which predicts word timings, is not included.
- Results have no confidence values or alternatives, and the context (hotword) prompt
  of the original model is not supported.
- The language that the model identifies is not reported.
- The decoder runs on the GPU or the CPU. On macOS 27.0.1, Core ML cannot build it
  for the Neural Engine.
- The conversion was tested on a Mac with an M4 Max and macOS 27.0.1 only.
  It was not tested on iPhone or iPad.

## Size and resources

| Item | Value |
| --- | --- |
| Files | {total / 1e9:.2f} GB |
| Peak memory while transcribing | {peak:.1f} GB |
| Real-time factor (processing time divided by audio duration) | {rtf[0]:.2f} to {rtf[-1]:.2f} |
| Core ML specification | iOS 18 and macOS 15 (stateful ML programs) |

The real-time factor was measured on a Mac with an M4 Max while another speech job used the GPU,
so it is higher than on an idle Mac.

## Evaluation

The table gives the error rates of AnySpeechRecognizer's runtime with this conversion
in file mode, measured on {evaluation["date"]} with its `Tools/natural-speech-eval.py`.
The FLEURS sets are the first 100 test files of each language.
Character error rates (CER) count characters after NFKC normalization and lowercasing,
without punctuation, symbols, or spaces.
For Cantonese, both texts are first converted to simplified characters with OpenCC,
because the references use traditional characters and the model writes either script.
Word error rates (WER) count words after lowercasing and removing punctuation.
The transformers column is the original model in transformers with greedy decoding
on the same files.

{chr(10).join(evaluation_table(evaluation, label, reference))}

The FLEURS references write numbers as digits, names in Latin script,
and some English glosses in parentheses that speakers did not always read.
The model often writes numbers in words and names in the local script,
so these error rates are higher than those in the original model card.

## Conversion and verification

The models were converted with this command in `Tools/QwenConversion`:

```sh
uv run python convert.py --variant {variant.name} --dtype {conversion.get("trace_dtype", "float32")}
```

The tools were Python {tools["python"]}, PyTorch {tools["torch"]}, transformers {tools["transformers"]},
and coremltools {tools["coremltools"]} on macOS {tools["macos"]}.
The KV cache holds {conversion["context_length"]:,} tokens, and the decoder takes
{conversion["block_length"]} tokens in each prediction.

`Tools/QwenConversion/verify.py` compared the conversion with the transformers model.
It compared the audio embeddings for each token,
the decoder's greedy token after each prefix of the transformers output,
and complete greedy transcripts.

{chr(10).join(verification_summary(reports))}
{chr(10) + verification_note + chr(10) if verification_note else ""}
## Files

| File | Size |
| --- | --- |
{file_rows}

`conversion.json` lists the size and SHA-256 digest of every file except itself and this card.

## License

The original model and this conversion are licensed under the
[Apache License, Version 2.0]({qwen.LICENSE_URL}). See `LICENSE` and `NOTICE`.

{qwen.attribution(variant)}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", choices=sorted(qwen.VARIANTS), required=True)
    parser.add_argument("--artifacts", type=Path, help="Verified artifacts (default: build/<variant>).")
    parser.add_argument("--output", type=Path, required=True, help="An empty or new package directory.")
    parser.add_argument("--evaluation", type=Path, help="Evaluation results (default: evaluation/<variant>.json).")
    parser.add_argument("--label", help="The runtime's run label (default: final-qwen-<variant>).")
    parser.add_argument("--reference-label", help="The transformers run label (default: transformers-<variant>).")
    parser.add_argument(
        "--verification",
        type=Path,
        nargs="+",
        help="verify.py reports (default: build/reports/verify-<variant>-wrapper.json and -natural.json).",
    )
    parser.add_argument(
        "--verification-note", help="A sentence for the model card that describes the differing transcripts."
    )
    parser.add_argument(
        "--trace-dtype",
        choices=["float32", "float16"],
        help="The convert.py --dtype of the artifacts, when their conversion.json does not record it.",
    )
    args = parser.parse_args()

    variant = qwen.VARIANTS[args.variant]
    artifacts = (args.artifacts or qwen.BUILD_DIR / variant.name).resolve()
    output = args.output.resolve()
    evaluation_path = args.evaluation or qwen.EVALUATION_DIR / f"{variant.name}.json"
    label = args.label or f"final-qwen-{variant.name}"
    reference = args.reference_label or f"transformers-{variant.name}"
    reports = args.verification or [
        qwen.REPORTS_DIR / f"verify-{variant.name}-wrapper.json",
        qwen.REPORTS_DIR / f"verify-{variant.name}-natural.json",
    ]

    record = json.loads((artifacts / qwen.CONVERSION_FILE).read_text())
    if record["source"] != {"repository": variant.repo, "revision": variant.revision}:
        print(f"error: {artifacts} was converted from {record['source']}", file=sys.stderr)
        return 1
    conversion = record["conversion"]
    if "trace_dtype" not in conversion:
        if not args.trace_dtype:
            print("error: conversion.json does not record the trace type; pass --trace-dtype", file=sys.stderr)
            return 1
        conversion["trace_dtype"] = args.trace_dtype
        conversion["trace_dtype_source"] = "package.py --trace-dtype"
    problems = check_artifacts(artifacts, record)
    if qwen.sha256(qwen.LICENSE_TEXT) != qwen.LICENSE_TEXT_SHA256:
        problems.append(f"{qwen.LICENSE_TEXT.name} does not have the pinned digest")
    if problems:
        print("error:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    if output.exists() and any(output.iterdir()):
        print(f"error: {output} is not empty", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)

    for name in MODEL_FILES:
        source = artifacts / name
        if source.is_dir():
            shutil.copytree(source, output / name)
        else:
            shutil.copy2(source, output / name)
    shutil.copyfile(qwen.LICENSE_TEXT, output / qwen.LICENSE_FILE)
    (output / qwen.NOTICE_FILE).write_text(notice(variant, record))

    evaluation = json.loads(evaluation_path.read_text())
    files = [entry for entry in qwen.tree_files(output) if entry["path"] not in UNLISTED]
    model_files = {entry["path"]: entry for entry in record["files"]}
    for entry in files:
        original = model_files.get(entry["path"])
        if entry["path"].split("/")[0] in MODEL_FILES and original != entry:
            print(f"error: {entry['path']} changed while copying", file=sys.stderr)
            return 1
    (output / qwen.MODEL_CARD_FILE).write_text(
        model_card(variant, record, files, evaluation, label, reference, reports, args.verification_note)
    )
    package = {
        "variant": variant.name,
        "repository": variant.upload_repo,
        "source": record["source"],
        "created": record["created"],
        "packaged": datetime.now(UTC).isoformat(timespec="seconds"),
        "tools": record["tools"],
        "conversion": conversion,
        "license": qwen.LICENSE,
        "license_text": {"source": qwen.LICENSE_TEXT_SOURCE, "sha256": qwen.LICENSE_TEXT_SHA256},
        "attribution": record["attribution"],
        "evaluation": {
            "file": f"Tools/QwenConversion/evaluation/{evaluation_path.name}",
            "date": evaluation["date"],
            "label": label,
            "results": {key: value for key, value in evaluation["results"].items() if key.startswith(f"{label}/")},
        },
        "verification": [report.name for report in reports],
        "files": files,
    }
    qwen.write_json(output / qwen.CONVERSION_FILE, package)
    total = sum(entry["size"] for entry in files)
    print(f"Wrote {output}: {len(files)} listed files, {total / 1e9:.3f} GB, plus conversion.json and README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
