# SPDX-License-Identifier: Apache-2.0
"""Package verified Nemotron 3.5 ASR Streaming Core ML artifacts for publication on Hugging Face.

Usage:
    uv run python package.py
    uv run python package.py --artifacts build/nemotron --output build/package

The package contains:

- The compiled models, vocabulary, and prompts from the artifacts directory, copied unchanged
  after their sizes and digests are checked against its conversion.json.
- LICENSE: the OpenMDW-1.1 text, from the pinned commit of the OpenMDW repository.
- NOTICE: the notices of origin (NVIDIA, the source repository and revision) and the changes.
- README.md: the model card, with the evaluation in evaluation.json and the verification
  reports in build/reports.
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

import nemotron as nt

MODEL_FILES = (nt.ENCODER_FILE, nt.DECODER_FILE, nt.JOINT_FILE, nt.VOCAB_FILE, nt.PROMPTS_FILE)
UNLISTED = (nt.CONVERSION_FILE, nt.MODEL_CARD_FILE)
APP_REPOSITORY = "https://github.com/mattt/AnySpeechRecognizer"
EVALUATION = nt.TOOL_DIR / "evaluation.json"
REPORTS_DIR = nt.BUILD_DIR / "reports"


def check_artifacts(artifacts: Path, record: dict[str, Any]) -> list[str]:
    """Problems with the model files against their conversion record."""
    problems = []
    listed = {entry["path"]: entry for entry in record["files"]}
    for entry in listed.values():
        if entry["path"].split("/")[0] not in MODEL_FILES:
            continue
        path = artifacts / entry["path"]
        if not path.is_file():
            problems.append(f"missing {entry['path']}")
        elif path.stat().st_size != entry["size"] or nt.sha256_file(path) != entry["sha256"]:
            problems.append(f"digest mismatch {entry['path']}")
    for name in MODEL_FILES:
        if not any(path == name or path.startswith(name + "/") for path in listed):
            problems.append(f"conversion.json lists no {name}")
    return problems


def describe_size(size: int) -> str:
    """A decimal size: kilobytes below one megabyte, megabytes otherwise."""
    return f"{size / 1e3:,.1f} kB" if size < 1_000_000 else f"{size / 1e6:,.1f} MB"


def functions(record: dict[str, Any]) -> list[dict[str, Any]]:
    return record["conversion"]["components"]["encoder"]["functions"]


def chunk_list(record: dict[str, Any]) -> str:
    sizes = [f"{f['chunk_frames'] * 80:,} ms" for f in functions(record)]
    return ", ".join(sizes[:-1]) + f", and {sizes[-1]}"


def notice(record: dict[str, Any]) -> str:
    source = record["source"]
    created = record["created"][:10]
    return f"""{nt.TITLE} for Core ML

This repository contains a modified version of {nt.TITLE},
a model by NVIDIA Corporation:
https://huggingface.co/{source["repo"]}
at revision {source["revision"]},
converted from the file {source["file"]}
(SHA-256 {source["sha256"]}).

NVIDIA provides the model under the OpenMDW License Agreement, version 1.1
(OpenMDW-1.1). The original model card states: "Use of the model is governed by
the OpenMDW-1.1 license." LICENSE contains the text of the agreement, copied from
{nt.LICENSE_TEXT_SOURCE}.
This modified version is distributed under the same agreement.

AnySpeechRecognizer ({APP_REPOSITORY}) modified the model on {created}
with its Tools/NemotronConversion tool. The modifications are:

- The NeMo checkpoint was converted to three Core ML programs:
  Encoder.mlmodelc, Decoder.mlmodelc, and JointDecision.mlmodelc.
- Encoder.mlmodelc is a multifunction model with one function for each of five chunk
  sizes ({chunk_list(record)}), which share their weights.
- Each encoder function is one cache-aware streaming step. It takes the attention and
  convolution caches of every layer as explicit inputs and returns the updated caches
  as outputs.
- The language prompt projection is part of the encoder step, which takes the language
  prompt as a one-hot input.
- The joint decision returns the greedy token and its probability instead of logits.
- Weights are stored in float16. Encoder weight tensors of 2,048 or more elements are
  quantized to int8 (linear symmetric, one scale per output channel).
- vocab.json and prompts.json are new files, derived from the original tokenizer and
  model configuration.
"""


def evaluation_table(evaluation: dict[str, Any]) -> list[str]:
    systems = evaluation["systems"]
    names = evaluation["system_names"]
    rows = [
        "| Test set | Measure | " + " | ".join(names[s] for s in systems) + " |",
        "| --- | --- | " + " | ".join("---" for _ in systems) + " |",
    ]
    for row in evaluation["sets"]:
        cells = [f"{row['results'][s]:.2f}%" if s in row["results"] else "" for s in systems]
        rows.append(f"| {row['name']} | {row['measure']} | " + " | ".join(cells) + " |")
    return rows


def latency_table(evaluation: dict[str, Any]) -> list[str]:
    rows = [
        "| System | Median / 90th percentile delay | First word, median | Final when shown |",
        "| --- | --- | --- | --- |",
    ]
    for row in evaluation["latency"]["rows"]:
        rows.append(
            f"| {row['system']} | {row['median_ms']:,} / {row['p90_ms']:,} ms | {row['first_word_ms']:,} ms "
            f"| {'Yes' if row['final_when_shown'] else 'No'} |"
        )
    return rows


def verification_table(reports: list[Path]) -> list[str]:
    rows = [
        (
            "| Chunk | Compute units | Encoder step cosine, mean / lowest | Joint decisions that match "
            "| Fixtures with the same tokens | Reference matches NeMo's loop |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        for result in json.loads(report.read_text())["results"]:
            steps = result["isolated_steps"]
            agreed, total = result["joint_agreement"]
            fixtures = result["fixtures"]
            same = sum(f["tokens_equal"] for f in fixtures)
            nemo = [f["reference_equals_nemo"] for f in fixtures if "reference_equals_nemo" in f]
            nemo_text = f"{sum(nemo)} of {len(nemo)}" if nemo else "not checked"
            rows.append(
                f"| {result['chunk_ms']:,} ms | `{result['compute_units']}` "
                f"| {steps['mean_cosine']:.4f} / {steps['min_cosine']:.3f} | {agreed:,} of {total:,} "
                f"| {same} of {len(fixtures)} | {nemo_text} |"
            )
    return rows


def locale_lines() -> list[str]:
    def describe(tier: dict[str, list[str]]) -> str:
        return ", ".join(f"{name} ({', '.join(codes)})" for name, codes in tier.items())

    return [
        f"- Ready to transcribe (19 locales): {describe(nt.TRANSCRIPTION_READY)}.",
        f"- Broad coverage (13 locales): {describe(nt.BROAD_COVERAGE)}.",
        f"- Adaptation-ready (8 locales), which NVIDIA says need fine-tuning: {describe(nt.ADAPTATION_READY)}.",
    ]


def language_codes() -> list[str]:
    codes = {
        code.split("-")[0] for tier in (nt.TRANSCRIPTION_READY, nt.BROAD_COVERAGE) for c in tier.values() for code in c
    }
    return sorted(codes)


def model_card(
    record: dict[str, Any], files: list[dict[str, Any]], evaluation: dict[str, Any], reports: list[Path]
) -> str:
    source = record["source"]
    sizes: dict[str, int] = {}
    for entry in files:
        top = entry["path"].split("/")[0]
        sizes[top] = sizes.get(top, 0) + entry["size"]
    total = sum(sizes.values())
    tools = record["tools"]
    rtf = evaluation["real_time_factor"]
    check = evaluation["reference_check"]
    front_matter = "\n".join(
        [
            "---",
            "license: other",
            "license_name: openmdw-1.1",
            "license_link: LICENSE",
            f"base_model: {source['repo']}",
            "base_model_relation: quantized",
            "pipeline_tag: automatic-speech-recognition",
            "library_name: coreml",
            "language:",
            *(f"- {code}" for code in language_codes()),
            "tags:",
            "- coreml",
            "- speech-recognition",
            "- streaming-asr",
            "- rnnt",
            "---",
        ]
    )
    file_rows = "\n".join(
        f"| `{name}` | {describe_size(size)} |"
        for name, size in sorted(sizes.items())
        if name in MODEL_FILES or name in (nt.LICENSE_FILE, nt.NOTICE_FILE)
    )
    return f"""{front_matter}

# {nt.TITLE} for Core ML

This repository contains a Core ML conversion of
[{nt.TITLE}](https://huggingface.co/{source["repo"]}) by NVIDIA,
at revision `{source["revision"]}`.
[AnySpeechRecognizer]({APP_REPOSITORY}) runs it on Apple silicon
with the same API as Apple's `SpeechTranscriber`.

The model is a cache-aware streaming transducer. Its encoder keeps the attention and
convolution caches of every layer between steps, so it encodes each chunk of audio once,
and each word is final when the model decodes it. The chunk size sets the latency.
This conversion has encoder functions for {chunk_list(record)} chunks.

The runtime in AnySpeechRecognizer is a prototype. It was tested on a Mac only.

## Accuracy

This conversion is less accurate than Parakeet v3 and Parakeet Ultra in AnySpeechRecognizer's
live mode on every test set below except LibriSpeech test-clean long files.
Most of the difference comes from the model, not from the conversion: on the same files,
NVIDIA's model in PyTorch had {check[0]["reference_wer"]:.2f}% WER on {check[0]["set"]}
(this conversion {check[0]["runtime_wer"]:.2f}%) and {check[1]["reference_wer"]:.2f}% on
{check[1]["set"]} (this conversion {check[1]["runtime_wer"]:.2f}%), with {check[0]["chunk_ms"]} ms chunks.

Reasons to choose this model anyway:

- Words appear sooner, and the model does not revise them. With 160 ms chunks, the median
  delay from the end of a word to its display was 370 ms, against 729 ms for Parakeet v3's
  live mode, whose first results can still change for about 2 seconds.
- It transcribes languages that Parakeet does not. This evaluation measured Japanese and
  Mandarin; NVIDIA's model card also lists Korean, Arabic, Hindi, and others.
- The caller can set the language, or let the model detect it.

Error rates in percent, measured on {evaluation["date"]} on a {evaluation["machine"]}
with `Tools/natural-speech-eval.py` in live mode (100 ms blocks of input).
Error rates are computed after lowercasing and removing punctuation, without number or
spelling normalization, so they are higher than those in NVIDIA's model card.
This model received the language of each test set, except in the last column.
Japanese and Mandarin are character error rates (CER).
The evaluation used an earlier conversion with the same tools and settings;
the verification below gave the same results for both, in every row that both runs measured.

{chr(10).join(evaluation_table(evaluation))}

On the AMI meetings, this model deleted 3,605 reference words at 560 ms, against 2,408
for Parakeet v3. It leaves out more filler words and short replies, such as "um", "uh",
and "yeah", than Parakeet does, and it also misses more of the other words.

## Latency

AnySpeechRecognizer's `hear` command-line tool delivered each 100 ms block of audio at the time
that a microphone would deliver it.
The delay is the time from the end of a word's audio, from forced alignments,
until the word first appears in any result.
These figures come from the first 25 utterances of LibriSpeech test-clean.

{chr(10).join(latency_table(evaluation))}

Smaller chunks need more computation per second of audio. Processing took
{rtf["nemotron-160"][0]:.2f} to {rtf["nemotron-160"][1]:.2f} of real time with 160 ms chunks,
{rtf["nemotron-560"][0]:.3f} to {rtf["nemotron-560"][1]:.2f} with 560 ms chunks,
and {rtf["nemotron-1120"][0]:.3f} to {rtf["nemotron-1120"][1]:.2f} with 1,120 ms chunks,
on a Mac that other jobs were using at the same time.

## Languages

NVIDIA's model card lists 40 locales in three tiers:

{chr(10).join(locale_lines())}

`prompts.json` maps locale codes, such as `de-DE`, and `auto` to the model's language prompts.
With `auto`, the model detects the language and writes a tag such as `<en-US>` after each
sentence. AnySpeechRecognizer removes the tags from its results.
In the evaluation above, automatic detection changed the error rates by at most 0.54 points.
This conversion was evaluated on English, German, French, Japanese, and Mandarin only.

## Use

```swift
let transcriber = SpeechTranscriber(
    model: NemotronSpeechModel(language: .locale(Locale(identifier: "de-DE")), chunkSize: .ms160),
    transcriptionOptions: [],
    reportingOptions: [.volatileResults],
    attributeOptions: [.audioTimeRange]
)
```

A version of AnySpeechRecognizer that pins this repository downloads and verifies these files
through `AssetInventory`. Other versions can load a local copy of this repository with
`NemotronSpeechModel(assets: .directory(url))`.

## Limitations

- The runtime is a prototype. It was tested on a Mac with an M4 Max and macOS 27.0.1 only,
  not on iPhone or iPad.
- The language that the model detects is not reported.
- The chunk size of 160 ms is not one that NVIDIA trained. The model card reports results
  for it, and it worked in this evaluation.
- The first load of each encoder function takes 10 to 26 seconds on a Mac,
  while Core ML compiles it for the Neural Engine.

## Size and resources

| Item | Value |
| --- | --- |
| Files | {total / 1e6:,.0f} MB |
| Encoder functions | {chunk_list(record)} chunks, in one model with shared weights |
| Core ML specification | iOS 18 and macOS 15 (multifunction ML programs) |

## Conversion and verification

The models were converted with this command in `Tools/NemotronConversion`:

```sh
uv run python convert.py
```

The tools were Python {tools["python"]}, PyTorch {tools["torch"]}, NeMo {tools["nemo_toolkit"]},
and coremltools {tools["coremltools"]} on macOS {tools["macos"]}.
The conversion traces NeMo's own cache-aware encoder step for each chunk size.

`Tools/NemotronConversion/verify.py` compared each Core ML encoder step with NVIDIA's model
in PyTorch, given the same inputs and caches, and compared the transcripts of six short
recordings in English, German, French, and Japanese, one of them with automatic language
detection. It also compared the PyTorch reference with NeMo's own streaming loop.

{chr(10).join(verification_table(reports))}

Where the tokens differ, the transcripts differ by a period, a capital letter, a language tag,
or one word. Where the reference differs from NeMo's loop, the recording's last chunk has
fewer than 8 mel frames, which NeMo's loop does not encode.

## Files

| File | Size |
| --- | --- |
{file_rows}

`conversion.json` lists the size and SHA-256 digest of every file except itself and this card.

## License

NVIDIA provides the original model under the OpenMDW License Agreement, version 1.1,
and this conversion is distributed under the same agreement. See `LICENSE` and `NOTICE`.

{nt.ATTRIBUTION}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifacts", type=Path, default=nt.BUILD_DIR / "nemotron", help="Verified artifacts.")
    parser.add_argument(
        "--output", type=Path, default=nt.BUILD_DIR / "package", help="An empty or new package directory."
    )
    parser.add_argument("--evaluation", type=Path, default=EVALUATION, help="Evaluation results.")
    parser.add_argument(
        "--verification",
        type=Path,
        nargs="+",
        default=[REPORTS_DIR / "verify.json"],
        help="verify.py reports (default: build/reports/verify.json).",
    )
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    output = args.output.resolve()
    record = json.loads((artifacts / nt.CONVERSION_FILE).read_text())
    if (record["source"]["repo"], record["source"]["revision"]) != (nt.REPOSITORY, nt.REVISION):
        print(f"error: {artifacts} was converted from {record['source']}", file=sys.stderr)
        return 1
    problems = check_artifacts(artifacts, record)
    if nt.sha256_file(nt.LICENSE_TEXT) != nt.LICENSE_TEXT_SHA256:
        problems.append(f"{nt.LICENSE_TEXT.name} does not have the pinned digest")
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
    shutil.copyfile(nt.LICENSE_TEXT, output / nt.LICENSE_FILE)
    (output / nt.NOTICE_FILE).write_text(notice(record))

    files = nt.file_inventory(output, exclude=UNLISTED)
    model_files = {entry["path"]: entry for entry in record["files"]}
    for entry in files:
        if entry["path"].split("/")[0] in MODEL_FILES and model_files.get(entry["path"]) != entry:
            print(f"error: {entry['path']} changed while copying", file=sys.stderr)
            return 1
    evaluation = json.loads(args.evaluation.read_text())
    (output / nt.MODEL_CARD_FILE).write_text(model_card(record, files, evaluation, args.verification))
    package = {
        "format_version": 1,
        "title": nt.TITLE,
        "repository": nt.UPLOAD_REPO,
        "source": record["source"],
        "created": record["created"],
        "packaged": datetime.now(UTC).isoformat(timespec="seconds"),
        "tools": record["tools"],
        "conversion": record["conversion"],
        "license": nt.LICENSE,
        "license_text": {"source": nt.LICENSE_TEXT_SOURCE, "sha256": nt.LICENSE_TEXT_SHA256},
        "attribution": nt.ATTRIBUTION,
        "evaluation": {"file": "Tools/NemotronConversion/evaluation.json", "date": evaluation["date"]},
        "verification": [report.name for report in args.verification],
        "files": files,
    }
    (output / nt.CONVERSION_FILE).write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n")
    total = sum(entry["size"] for entry in files)
    print(f"Wrote {output}: {len(files)} listed files, {total / 1e6:.1f} MB, plus conversion.json and README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
