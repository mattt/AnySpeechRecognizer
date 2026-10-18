#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy==2.3.5", "rapidfuzz==3.14.6", "soundfile==0.14.0", "whisper-normalizer==0.1.15",
# ]
# ///
"""Evaluates the hear CLI on natural speech.

The data are LibriSpeech test-clean and test-other (CC BY 4.0), the AMI Meeting Corpus
headset-mix audio of the full-corpus test meetings (CC BY 4.0) with the BUT diarization
references (Apache-2.0), and the LibriSpeech forced alignments of Lugosch et al. on Zenodo
(CC BY 4.0). With `--held-out`, `fetch` also downloads LibriSpeech dev-clean and dev-other,
four AMI dev meetings, and, with `--fleurs`, the first FLEURS dev files, for checks on data
that no runtime change was tuned on. Everything is downloaded into a scratch directory;
nothing is added to the repository.

    uv run Tools/natural-speech-eval.py fetch --root /tmp/natural-eval
    uv run Tools/natural-speech-eval.py build --root /tmp/natural-eval
    uv run Tools/natural-speech-eval.py run --root /tmp/natural-eval \
        --cli .build/release/hear --set ls-clean-long --system v3-live
    uv run Tools/natural-speech-eval.py score --root /tmp/natural-eval

`run` writes one JSON-lines file per audio file under `runs/<label>/<set>/` and skips
files that already have output, so an interrupted run continues where it stopped.
Run one process at a time.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

LIBRISPEECH = "https://www.openslr.org/resources/12/{}.tar.gz"
AMI_AUDIO = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{0}/audio/{0}.Mix-Headset.wav"
)
# BUTSpeechFIT/AMI-diarization-setup, "only_words" references, pinned.
AMI_SETUP_COMMIT = "2509d8933721023fab4def2618aabd5c28eb82e9"
AMI_RTTM = (
    "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/"
    + AMI_SETUP_COMMIT
    + "/only_words/rttms/{}/{}.rttm"
)
AMI_TEST = [
    f"{prefix}{part}" for prefix in ["EN2002", "ES2004", "IS1009", "TS3003"] for part in "abcd"
]
AMI_DEFAULT = ["ES2004a", "IS1009a", "TS3003a", "EN2002a"]
# Held-out meetings: the first meeting of each dev series of the full-corpus partition.
AMI_DEV = ["ES2011a", "IS1008a", "TS3004a", "IB4001"]
# Lugosch et al., "Speech Model Pre-training for End-to-End Spoken Language Understanding",
# Montreal Forced Aligner alignments of LibriSpeech, DOI 10.5281/zenodo.2619474.
ALIGNMENTS = "https://zenodo.org/api/records/2619474/files/librispeech_alignments.zip/content"
# AMI manual annotations 1.6.2 (CC BY 4.0), for meeting reference transcripts.
AMI_ANNOTATIONS = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)

# google/fleurs (CC BY 4.0), pinned; a subset is the first files of a test or dev archive.
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/" + FLEURS_REVISION + "/data/{}/{}"
FLEURS_LANGUAGES = {"de": "de_de", "fr": "fr_fr"}
FLEURS_FILES = 100

# LibriSpeech splits and the subset name prefix of each.
LIBRISPEECH_SPLITS = {
    "test-clean": "ls-clean",
    "test-other": "ls-other",
    "dev-clean": "ls-dev-clean",
    "dev-other": "ls-dev-other",
}
HELD_OUT_SPLITS = ["dev-clean", "dev-other"]

# Seconds of audio in each utterance subset, and the lengths of the concatenated files.
UTTERANCE_BUDGET = 1800.0
LONG_TARGETS = [480.0, 420.0, 360.0, 300.0, 240.0, 180.0]

SYSTEMS = {
    "v3-file": ["--backend", "parakeet", "--version", "v3"],
    "v3-live": ["--backend", "parakeet", "--version", "v3", "--live"],
    "v3-fast": ["--backend", "parakeet", "--version", "v3", "--live", "--fast"],
    "ultra-file": ["--backend", "parakeet", "--version", "ultra"],
    "ultra-live": ["--backend", "parakeet", "--version", "ultra", "--live"],
    "ultra-fast": ["--backend", "parakeet", "--version", "ultra", "--live", "--fast"],
    "unified-file": ["--backend", "parakeet", "--version", "unified"],
    "unified-live": ["--backend", "parakeet", "--version", "unified", "--live"],
    "apple-file": ["--backend", "apple", "--locale", "en-US"],
    "apple-live": ["--backend", "apple", "--locale", "en-US", "--live"],
}
for version in ["v2", "v2.1"]:
    for mode in ["file", "live"]:
        for units in ["cpuAndNeuralEngine", "cpuAndGPU"]:
            name = f"sortformer-{version}-{mode}" + ("-gpu" if units == "cpuAndGPU" else "")
            SYSTEMS[name] = (
                ["--backend", "parakeet", "--version", "v3", "--diarize"]
                + ["--diarizer-version", version, "--diarizer-compute-units", units]
                + (["--live"] if mode == "live" else [])
            )
    for preset in ["none", "dihard3"]:
        SYSTEMS[f"sortformer-{version}-file-gpu-{preset}"] = SYSTEMS[
            f"sortformer-{version}-file-gpu"
        ] + ["--diarizer-post-processing", preset]


def download(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.rename(path)


def fetch(args) -> None:
    root = Path(args.root)
    data = root / "data"
    splits = ["test-clean", "test-other"] + (HELD_OUT_SPLITS if args.held_out else [])
    for split in splits:
        if not (data / "LibriSpeech" / split).exists():
            archive = root / "dl" / f"{split}.tar.gz"
            download(LIBRISPEECH.format(split), archive)
            with tarfile.open(archive) as tar:
                tar.extractall(data, filter="data")
    meetings = [("test", "rttm", name) for name in (AMI_TEST if args.all_meetings else AMI_DEFAULT)]
    if args.held_out:
        meetings += [("dev", "rttm-dev", name) for name in AMI_DEV]
    for partition, directory, meeting in meetings:
        download(AMI_RTTM.format(partition, meeting), data / "ami" / directory / f"{meeting}.rttm")
        download(AMI_AUDIO.format(meeting), data / "ami" / "audio" / f"{meeting}.Mix-Headset.wav")
    download(AMI_ANNOTATIONS, root / "dl" / "ami_public_manual_1.6.2.zip")
    if args.fleurs:
        archives = [("test", "fleurs")] + ([("dev", "fleurs-dev")] if args.held_out else [])
        for (split, directory), language in [
            (archive, language) for archive in archives for language in FLEURS_LANGUAGES.values()
        ]:
            target = data / directory / language
            download(FLEURS.format(language, f"{split}.tsv"), target / f"{split}.tsv")
            if len(list(target.glob("*.wav"))) >= FLEURS_FILES:
                continue
            # Read the archive as a stream and stop after the first files.
            url = FLEURS.format(language, f"audio/{split}.tar.gz")
            with (
                urllib.request.urlopen(url) as response,
                tarfile.open(fileobj=response, mode="r|gz") as tar,
            ):
                count = 0
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".wav"):
                        continue
                    member.name = Path(member.name).name
                    tar.extract(member, target, filter="data")
                    count += 1
                    if count == FLEURS_FILES:
                        break
    if args.alignments:
        archive = root / "dl" / "librispeech_alignments.zip"
        download(ALIGNMENTS, archive)
        target = data / "alignments"
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if any(f"{split}/" in name for split in splits) and not name.endswith("/"):
                    zipped.extract(name, target)


# Subsets.


def librispeech_utterances(data: Path, split: str) -> dict:
    """Utterances by (speaker, chapter), in ID order, with text and duration."""
    import soundfile

    chapters = {}
    for transcript in sorted((data / "LibriSpeech" / split).glob("*/*/*.trans.txt")):
        speaker, chapter = (int(part) for part in transcript.name.split(".")[0].split("-"))
        rows = []
        for line in transcript.read_text().splitlines():
            utterance, text = line.split(" ", 1)
            audio = transcript.parent / f"{utterance}.flac"
            rows.append((utterance, audio, soundfile.info(str(audio)).duration, text))
        chapters[(speaker, chapter)] = sorted(rows)
    return dict(sorted(chapters.items()))


def build(args) -> None:
    import numpy
    import soundfile

    root = Path(args.root)
    data = root / "data"
    subsets = root / "subsets"
    subsets.mkdir(parents=True, exist_ok=True)
    for split, prefix in LIBRISPEECH_SPLITS.items():
        if not (data / "LibriSpeech" / split).exists():
            continue
        short = prefix.removeprefix("ls-")
        chapters = librispeech_utterances(data, split)
        speakers = sorted({speaker for speaker, _ in chapters})
        # Utterances: consecutive utterances of each speaker's first chapters, in ID order,
        # until the speaker's equal share of the budget.
        budget = UTTERANCE_BUDGET / len(speakers)
        rows = []
        for speaker in speakers:
            total = 0.0
            for (owner, _), utterances in chapters.items():
                if owner != speaker:
                    continue
                for utterance, audio, duration, text in utterances:
                    if total >= budget:
                        break
                    rows.append((utterance, audio.relative_to(root), duration, text))
                    total += duration
        write_subset(subsets / f"{prefix}-utt.tsv", rows)
        # Long files: the longest chapters of distinct speakers, from their first utterance,
        # concatenated without added silence until the target length.
        ranked = sorted(
            chapters.items(), key=lambda item: (-sum(row[2] for row in item[1]), item[0])
        )
        used, rows = set(), []
        long_directory = data / "long"
        long_directory.mkdir(parents=True, exist_ok=True)
        for target in LONG_TARGETS:
            (speaker, chapter), utterances = next(
                item
                for item in ranked
                if item[0][0] not in used and sum(row[2] for row in item[1]) >= target
            )
            used.add(speaker)
            parts, total = [], 0.0
            for row in utterances:
                if total >= target:
                    break
                parts.append(row)
                total += row[2]
            name = f"{speaker}-{chapter}-long"
            audio = long_directory / f"{short}-{name}.wav"
            offsets, samples, position = [], [], 0
            for utterance, path, _, text in parts:
                signal, rate = soundfile.read(str(path), dtype="int16")
                assert rate == 16_000
                offsets.append((utterance, position / rate, len(signal) / rate, text))
                samples.append(signal)
                position += len(signal)
            if not audio.exists():
                soundfile.write(str(audio), numpy.concatenate(samples), 16_000, subtype="PCM_16")
            (long_directory / f"{short}-{name}.segments.tsv").write_text(
                "".join(f"{u}\t{s:.4f}\t{d:.4f}\t{t}\n" for u, s, d, t in offsets)
            )
            text = " ".join(part[3] for part in parts)
            rows.append((name, audio.relative_to(root), position / 16_000, text))
        write_subset(subsets / f"{prefix}-long.tsv", rows)
    for (split, suffix), (short, language) in [
        (archive, item)
        for archive in [("test", ""), ("dev", "-dev")]
        for item in FLEURS_LANGUAGES.items()
    ]:
        directory = data / ("fleurs" + suffix) / language
        if not (directory / f"{split}.tsv").exists():
            continue
        transcripts = {}
        for line in (directory / f"{split}.tsv").read_text().splitlines():
            fields = line.split("\t")
            transcripts[fields[1]] = fields[2]
        rows = []
        for audio in sorted(directory.glob("*.wav")):
            rows.append(
                (
                    audio.stem,
                    audio.relative_to(root),
                    soundfile.info(str(audio)).duration,
                    transcripts[audio.name],
                )
            )
        write_subset(subsets / f"fleurs-{short}{suffix}.tsv", rows)
    # Synthetic fixtures from Tools/make-diarization-audio.swift and
    # Tools/make-evaluation-audio.swift, with a reference text where the generator wrote one.
    if (root / "synthetic").exists():
        rows = []
        for audio in sorted((root / "synthetic").glob("*.wav")):
            text = audio.with_suffix(".txt")
            reference = " ".join(text.read_text().split()) if text.exists() else ""
            duration = soundfile.info(str(audio)).duration
            rows.append((audio.stem, audio.relative_to(root), duration, reference))
        write_subset(subsets / "synthetic.tsv", rows)
    for directory in ["rttm", "rttm-dev"]:
        rows = []
        for rttm in sorted((data / "ami" / directory).glob("*.rttm")):
            audio = data / "ami" / "audio" / f"{rttm.stem}.Mix-Headset.wav"
            if not audio.exists():
                continue
            duration = soundfile.info(str(audio)).duration
            reference = ami_reference(root, rttm.stem)
            rows.append((rttm.stem, audio.relative_to(root), duration, reference))
        if directory == "rttm":
            write_subset(subsets / "ami.tsv", rows)
            write_subset(subsets / "ami-4.tsv", [row for row in rows if row[0] in AMI_DEFAULT])
        elif rows:
            write_subset(subsets / "ami-dev-4.tsv", [row for row in rows if row[0] in AMI_DEV])
    for path in sorted(subsets.glob("*.tsv")):
        items = read_subset(path)
        hours = sum(item["duration"] for item in items) / 3600
        print(f"{path.name}: {len(items)} files, {hours:.3f} h")


def ami_reference(root: Path, meeting: str) -> str:
    """The words of every speaker in a meeting, ordered by start time.

    Punctuation and truncated words are left out. Overlapping speech has no single correct
    order, so word error rates of whole meetings are approximate.
    """
    from xml.etree import ElementTree

    archive = root / "dl" / "ami_public_manual_1.6.2.zip"
    if not archive.exists():
        return ""
    words = []
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if not (name.startswith(f"words/{meeting}.") and name.endswith(".words.xml")):
                continue
            for element in ElementTree.fromstring(zipped.read(name)):
                if element.tag != "w" or element.get("punc") or element.get("trunc"):
                    continue
                words.append((float(element.get("starttime")), element.text.strip()))
    return " ".join(word for _, word in sorted(words))


def write_subset(path: Path, rows) -> None:
    path.write_text(
        "id\taudio\tseconds\treference\n"
        + "".join(f"{i}\t{a}\t{d:.4f}\t{t}\n" for i, a, d, t in rows)
    )


def read_subset(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()[1:]
    items = []
    for line in lines:
        identifier, audio, seconds, reference = line.split("\t")
        items.append(
            {"id": identifier, "audio": audio, "duration": float(seconds), "reference": reference}
        )
    return items


# Runs.


def run(args) -> None:
    root = Path(args.root)
    items = read_subset(root / "subsets" / f"{args.set}.tsv")
    if args.limit:
        items = items[: args.limit]
    output = root / "runs" / (args.label or args.system) / args.set
    output.mkdir(parents=True, exist_ok=True)
    arguments = list(SYSTEMS[args.system])
    if args.model_directory:
        arguments += ["--model-directory", args.model_directory]
    started = time.monotonic()
    for index, item in enumerate(items):
        destination = output / f"{item['id']}.jsonl"
        if destination.exists():
            continue
        command = [args.cli, "evaluate", str(root / item["audio"]), *arguments]
        if "--diarize" in arguments:
            directory = "rttm-dev" if item["id"] in AMI_DEV else "rttm"
            rttm = root / "data" / "ami" / directory / f"{item['id']}.rttm"
            command += ["--rttm", str(rttm), "--collar", "0.25"]
        if index == 0 and args.warm_up:
            # The first run of a new binary compiles the Core ML models again.
            subprocess.run(command, capture_output=True, check=True)
        wall = time.monotonic()
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{item['id']}: exit {result.returncode}: {result.stderr}", file=sys.stderr)
            continue
        record = {"type": "run", "wallSeconds": time.monotonic() - wall, "command": command[1:]}
        destination.write_text(result.stdout + json.dumps(record) + "\n")
        print(f"[{index + 1}/{len(items)}] {item['id']}", file=sys.stderr)
    print(f"{args.set} {args.system}: {time.monotonic() - started:.1f} s", file=sys.stderr)


def load_run(path: Path) -> dict:
    run = {"finals": [], "words": []}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        kind = record["type"]
        if kind == "result" and record["isFinal"]:
            run["finals"].append(record["text"])
            run["words"] += record["words"]
        elif kind in ["metadata", "evaluation", "run"]:
            run[kind] = record
    run["text"] = "".join(run["finals"])
    return run


# Text normalization and alignment.


def basic_words(text: str) -> list[str]:
    """Lowercase words without punctuation, with apostrophes inside words kept.

    Hyphens, dashes, and slashes separate words, as in LibriSpeech's references.
    Numbers and spellings are not normalized.
    """
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[-‐‑–—/]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    return [word for word in (token.strip("'") for token in text.split()) if word]


_whisper = None


def whisper_words(text: str) -> list[str]:
    """Words after Whisper's English normalizer, which the Open ASR Leaderboard uses."""
    global _whisper
    if _whisper is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _whisper = EnglishTextNormalizer()
    return _whisper(text).split()


def align(reference: list, hypothesis: list) -> list[tuple]:
    """A minimum edit alignment: ("C" | "S" | "D" | "I", reference index, hypothesis index)."""
    from rapidfuzz.distance import Levenshtein

    operations = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(reference, hypothesis):
        if tag == "equal":
            operations += [("C", i, j) for i, j in zip(range(i1, i2), range(j1, j2))]
        elif tag == "replace":
            operations += [("S", i, j) for i, j in zip(range(i1, i2), range(j1, j2))]
        elif tag == "delete":
            operations += [("D", i, None) for i in range(i1, i2)]
        else:
            operations += [("I", None, j) for j in range(j1, j2)]
    return operations


def deletion_runs(reference: list, hypothesis: list, minimum: int = 8) -> int:
    """The number of runs of at least `minimum` consecutive deleted reference words."""
    runs = streak = 0
    for operation, _, _ in align(reference, hypothesis) + [("C", None, None)]:
        if operation == "D":
            streak += 1
        elif operation != "I":
            runs += streak >= minimum
            streak = 0
    return runs


def insertion_runs(reference: list, hypothesis: list, minimum: int = 3) -> tuple[int, int]:
    """The runs of at least `minimum` consecutive inserted words, and the words in them.

    Repeated filler, such as "and the other thing and the other thing", shows as such runs.
    """
    runs = words = streak = 0
    for operation, _, _ in align(reference, hypothesis) + [("C", None, None)]:
        if operation == "I":
            streak += 1
            continue
        if streak >= minimum:
            runs += 1
            words += streak
        streak = 0
    return runs, words


def errors(reference: list, hypothesis: list) -> dict:
    counts = {"N": len(reference), "S": 0, "D": 0, "I": 0}
    for operation, _, _ in align(reference, hypothesis):
        if operation != "C":
            counts[operation] += 1
    return counts


def timed_words(run: dict) -> list[tuple[str, float, float]]:
    """Normalized words with the time range of the recognized word that contains them."""
    words = []
    for word in run["words"]:
        for token in basic_words(word["text"]):
            words.append((token, word["startSeconds"], word["endSeconds"]))
    return words


# Scores.


def score(args) -> None:
    root = Path(args.root)
    results = root / "results"
    results.mkdir(exist_ok=True)
    report = []
    summary = {}
    asr_sets = [
        "ls-clean-utt",
        "ls-other-utt",
        "ls-clean-long",
        "ls-other-long",
        "ami-4",
        "ami",
        "fleurs-de",
        "fleurs-fr",
        "synthetic",
        "ls-dev-clean-utt",
        "ls-dev-other-utt",
        "ls-dev-clean-long",
        "ls-dev-other-long",
        "ami-dev-4",
        "fleurs-de-dev",
        "fleurs-fr-dev",
    ]
    labels = sorted(path.name for path in (root / "runs").iterdir() if path.is_dir())
    report.append("## ASR\n")
    report.append(
        "| Label | Set | Files | Words | WER, basic (S/D/I) | WER, Whisper normalizer "
        "| Runs of 8+ deleted words | Runs of 3+ inserted words (words) | Analysis RTF "
        "| Mean preparation | Peak RSS |"
    )
    report.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for label in labels:
        for subset in asr_sets:
            directory = root / "runs" / label / subset
            if not directory.exists():
                continue
            items = {item["id"]: item for item in read_subset(root / "subsets" / f"{subset}.tsv")}
            basic = {"N": 0, "S": 0, "D": 0, "I": 0}
            whisper = {"N": 0, "S": 0, "D": 0, "I": 0}
            analysis = duration = preparation = 0.0
            peak = 0
            files = runs = inserted_runs = inserted_words = 0
            for path in sorted(directory.glob("*.jsonl")):
                run = load_run(path)
                item = items[path.stem]
                if not item["reference"]:
                    continue
                runs += deletion_runs(basic_words(item["reference"]), basic_words(run["text"]))
                inserted = insertion_runs(basic_words(item["reference"]), basic_words(run["text"]))
                inserted_runs += inserted[0]
                inserted_words += inserted[1]
                english = not subset.startswith("fleurs")
                for total, counts in [
                    (basic, errors(basic_words(item["reference"]), basic_words(run["text"]))),
                    (
                        whisper,
                        errors(whisper_words(item["reference"]), whisper_words(run["text"]))
                        if english
                        else {"N": 0, "S": 0, "D": 0, "I": 0},
                    ),
                ]:
                    for key in total:
                        total[key] += counts[key]
                prepared = run["metadata"]["preparationSeconds"]
                analysis += run["evaluation"]["elapsedSeconds"] - prepared
                preparation += prepared
                duration += item["duration"]
                peak = max(peak, run["evaluation"]["peakResidentBytes"])
                files += 1
            if not files:
                continue

            def rate(counts):
                return (counts["S"] + counts["D"] + counts["I"]) / max(1, counts["N"])

            summary[f"{label}/{subset}"] = {
                "files": files,
                "basic": basic,
                "deletionRuns": runs,
                "insertionRuns": inserted_runs,
                "insertionRunWords": inserted_words,
                "whisper": whisper,
                "analysisRTF": analysis / duration,
                "meanPreparation": preparation / files,
                "peakMB": peak / 1e6,
            }
            report.append(
                f"| {label} | {subset} | {files} | {basic['N']} | {rate(basic):.2%} "
                f"({basic['S']}/{basic['D']}/{basic['I']}) | {rate(whisper):.2%} | {runs} "
                f"| {inserted_runs} ({inserted_words}) | {analysis / duration:.4f} "
                f"| {preparation / files:.2f} s | {peak / 1e6:.0f} MB |"
            )
    report += live_drops(root, labels, asr_sets)
    report += diarization(root, labels)
    report += word_timing(root, labels)
    (results / "report.md").write_text("\n".join(report) + "\n")
    (results / "asr.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(report))


def live_drops(root: Path, labels: list[str], sets: list[str]) -> list[str]:
    """Words of file mode that live and fast modes delete, and runs of deleted words."""
    lines = ["\n## Live and fast modes against file mode\n"]
    lines.append(
        "| Label | Set | File words | Deleted | Inserted | Substituted "
        "| Runs of 4+ deleted words | Deleted against reference, file / live |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    runs_of_deletions = []
    for label in labels:
        match = re.match(r"(.*?)(v3|ultra|unified)-(live|fast)$", label)
        if not match:
            continue
        file_label = f"{match[1]}{match[2]}-file"
        if file_label not in labels:
            file_label = f"{match[2]}-file"
        for subset in sets:
            directory = root / "runs" / label / subset
            if not directory.exists():
                continue
            items = {item["id"]: item for item in read_subset(root / "subsets" / f"{subset}.tsv")}
            totals = {"N": 0, "S": 0, "D": 0, "I": 0}
            reference_deletions = [0, 0]
            long_runs = 0
            for path in sorted(directory.glob("*.jsonl")):
                file_path = root / "runs" / file_label / subset / path.name
                if not file_path.exists():
                    continue
                file_words = timed_words(load_run(file_path))
                live_run = load_run(path)
                live_words = timed_words(live_run)
                operations = align([w[0] for w in file_words], [w[0] for w in live_words])
                for operation, _, _ in operations:
                    if operation != "C":
                        totals[operation] += 1
                totals["N"] += len(file_words)
                reference = basic_words(items[path.stem]["reference"])
                reference_deletions[0] += errors(reference, [w[0] for w in file_words])["D"]
                reference_deletions[1] += errors(reference, [w[0] for w in live_words])["D"]
                streak = []
                for operation, i, _ in operations + [("C", None, None)]:
                    if operation == "D":
                        streak.append(i)
                        continue
                    if operation == "I":
                        continue
                    if len(streak) >= 4:
                        long_runs += 1
                        first, last = file_words[streak[0]], file_words[streak[-1]]
                        runs_of_deletions.append(
                            f"| {label} | {subset}/{path.stem} | {first[1]:.2f}–{last[2]:.2f} s "
                            f"| {len(streak)} | {' '.join(file_words[k][0] for k in streak)} |"
                        )
                    streak = []
            lines.append(
                f"| {label} | {subset} | {totals['N']} | {totals['D']} | {totals['I']} "
                f"| {totals['S']} | {long_runs} | {reference_deletions[0]} / "
                f"{reference_deletions[1]} |"
            )
    if runs_of_deletions:
        lines.append("\n| Label | File | File-mode time | Words | Deleted words |")
        lines.append("| --- | --- | --- | --- | --- |")
        lines += runs_of_deletions
    return lines


def diarization(root: Path, labels: list[str]) -> list[str]:
    lines = ["\n## Diarization\n"]
    lines.append(
        "| Label | Meeting | Hours | Speakers, reference / found | DER, no collar "
        "(miss / FA / confusion) | DER, 0.25 s collar | Analysis RTF | Peak RSS |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for label, subset in [
        (label, subset) for label in labels for subset in ["ami-4", "ami", "ami-dev-4"]
    ]:
        directory = root / "runs" / label / subset
        if not directory.exists():
            continue
        totals = {key: [0.0, 0.0, 0.0, 0.0] for key in ["withoutCollar", "withCollar"]}
        duration = analysis = 0.0
        correct = meetings = 0
        peak = 0
        for path in sorted(directory.glob("*.jsonl")):
            run = load_run(path)
            score = run["evaluation"].get("diarization")
            if score is None:
                continue
            for key, total in totals.items():
                part = score[key]
                total[0] += part["scoredSpeakerSeconds"]
                total[1] += part["missedSeconds"]
                total[2] += part["falseAlarmSeconds"]
                total[3] += part["confusionSeconds"]
            seconds = run["metadata"]["audio"]["durationSeconds"]
            elapsed = run["evaluation"]["elapsedSeconds"] - run["metadata"]["preparationSeconds"]
            duration += seconds
            analysis += elapsed
            meetings += 1
            correct += score["referenceSpeakerCount"] == score["hypothesisSpeakerCount"]
            peak = max(peak, run["evaluation"]["peakResidentBytes"])
            lines.append(
                f"| {label} | {path.stem} | {seconds / 3600:.2f} "
                f"| {score['referenceSpeakerCount']} / {score['hypothesisSpeakerCount']} "
                f"| {component(score['withoutCollar'])} "
                f"| {score['withCollar']['diarizationErrorRate']:.2%} "
                f"| {elapsed / seconds:.4f} | {run['evaluation']['peakResidentBytes'] / 1e6:.0f} MB |"
            )
        if meetings:
            scored, missed, false_alarm, confusion = totals["withoutCollar"]
            collared = totals["withCollar"]
            lines.append(
                f"| {label} | {subset} ({meetings}) | {duration / 3600:.2f} | {correct} of {meetings} "
                f"correct | {(missed + false_alarm + confusion) / scored:.2%} "
                f"({missed / scored:.2%} / {false_alarm / scored:.2%} / {confusion / scored:.2%}) "
                f"| {sum(collared[1:]) / collared[0]:.2%} | {analysis / duration:.4f} "
                f"| {peak / 1e6:.0f} MB |"
            )
    return lines


def component(part: dict) -> str:
    scored = part["scoredSpeakerSeconds"]
    return (
        f"{part['diarizationErrorRate']:.2%} ({part['missedSeconds'] / scored:.2%} / "
        f"{part['falseAlarmSeconds'] / scored:.2%} / {part['confusionSeconds'] / scored:.2%})"
    )


def read_alignment(root: Path, utterance: str) -> list[tuple[str, float, float]] | None:
    """Words and times from the Montreal Forced Aligner TextGrid of an utterance."""
    speaker, chapter, _ = utterance.split("-")
    alignments = root / "data" / "alignments"
    matches = [
        path
        for split in ["test-clean", "test-other"]
        if (path := alignments / split / speaker / chapter / f"{utterance}.TextGrid").exists()
    ]
    if not matches:
        return None
    words, tier = [], None
    interval = {}
    for line in matches[0].read_text().splitlines():
        line = line.strip()
        if line.startswith("name ="):
            tier = line.split("=", 1)[1].strip().strip('"')
        elif tier == "words" and line.startswith(("xmin", "xmax", "text")):
            key, value = (part.strip() for part in line.split("=", 1))
            interval[key] = value
            if key == "text":
                text = value.strip('"').lower()
                if text and text not in ["sil", "sp", "<unk>"]:
                    words.append((text, float(interval["xmin"]), float(interval["xmax"])))
                interval = {}
    return words


def word_timing(root: Path, labels: list[str]) -> list[str]:
    lines = ["\n## Word timing against forced alignment\n"]
    if not (root / "data" / "alignments").exists():
        return lines + ["No alignments."]
    lines.append(
        "| Label | Set | Utterances | Matched words | Start error, median / p95 "
        "| End error, median / p95 | Start bias, mean |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for label in labels:
        for subset in ["ls-clean-utt", "ls-other-utt"]:
            directory = root / "runs" / label / subset
            if not directory.exists() or "sortformer" in label:
                continue
            starts, ends, signed = [], [], []
            utterances = 0
            for path in sorted(directory.glob("*.jsonl")):
                reference = read_alignment(root, path.stem)
                if not reference:
                    continue
                hypothesis = timed_words(load_run(path))
                if not hypothesis:
                    continue
                utterances += 1
                operations = align([w[0] for w in reference], [w[0] for w in hypothesis])
                for operation, i, j in operations:
                    if operation != "C":
                        continue
                    # A recognized word that normalizes to several words, such as a hyphenated
                    # word, has one time range; compare only one-to-one words.
                    if sum(1 for w in hypothesis if w[1:] == hypothesis[j][1:]) > 1:
                        continue
                    starts.append(abs(hypothesis[j][1] - reference[i][1]))
                    ends.append(abs(hypothesis[j][2] - reference[i][2]))
                    signed.append(hypothesis[j][1] - reference[i][1])
            if not starts:
                continue
            lines.append(
                f"| {label} | {subset} | {utterances} | {len(starts)} "
                f"| {statistics.median(starts) * 1000:.0f} / {percentile(starts, 95) * 1000:.0f} ms "
                f"| {statistics.median(ends) * 1000:.0f} / {percentile(ends, 95) * 1000:.0f} ms "
                f"| {statistics.mean(signed) * 1000:+.0f} ms |"
            )
    return lines


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(percent / 100 * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["fetch", "build", "run", "score"]:
        command = commands.add_parser(name)
        command.add_argument("--root", required=True, help="The scratch directory.")
        if name == "fetch":
            command.add_argument(
                "--all-meetings", action="store_true", help="All 16 AMI test meetings."
            )
            command.add_argument(
                "--alignments", action="store_true", help="LibriSpeech forced alignments."
            )
            command.add_argument(
                "--fleurs", action="store_true", help="The FLEURS German and French subsets."
            )
            command.add_argument(
                "--held-out",
                action="store_true",
                help="LibriSpeech dev-clean and dev-other, AMI dev meetings, and FLEURS dev files.",
            )
        if name == "run":
            command.add_argument("--cli", required=True, help="The hear binary.")
            command.add_argument("--set", required=True, help="A subset name, such as ami.")
            command.add_argument("--system", required=True, choices=sorted(SYSTEMS))
            command.add_argument("--label", help="The output directory name; the system's name.")
            command.add_argument("--limit", type=int, help="Only the first files of the subset.")
            command.add_argument(
                "--model-directory", help="A staged Parakeet directory, passed to the CLI."
            )
            command.add_argument(
                "--warm-up", action="store_true", help="Run the first file once more, unrecorded."
            )
    args = parser.parse_args()
    {"fetch": fetch, "build": build, "run": run, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
