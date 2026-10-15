# SPDX-License-Identifier: Apache-2.0

"""Check the converted output directory and print the commands that would publish it.

This script never uploads or creates anything on Hugging Face.
It re-hashes every file listed in conversion.json, checks that the directory contains
nothing else except conversion.json and the model card, and prints `hf` commands for a
maintainer to review and run by hand.

Usage:
    uv run python upload_dry_run.py
    uv run python upload_dry_run.py --artifacts build/sortformer --repo mattt/diar-streaming-sortformer-4spk-v2-coreml
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import sortformer as sf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path, default=sf.DEFAULT_OUTPUT_DIR, help="Converted output directory")
    parser.add_argument("--repo", default=sf.UPLOAD_REPO, help="Destination repository")
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    record = json.loads((artifacts / sf.CONVERSION_FILE).read_text())

    problems = []
    if not (artifacts / sf.MODEL_CARD_FILE).is_file():
        problems.append(f"missing {sf.MODEL_CARD_FILE}")
    if record.get("license") != sf.LICENSE:
        problems.append(f"conversion.json license is {record.get('license')!r}, not {sf.LICENSE}")
    source = record.get("source", {})
    if (source.get("repo"), source.get("revision"), source.get("sha256")) != (
        sf.CHECKPOINT.repo,
        sf.CHECKPOINT.revision,
        sf.CHECKPOINT.sha256,
    ):
        problems.append("conversion.json does not record the pinned source checkpoint")
    listed = {entry["path"]: entry for entry in record["files"]}
    actual = {
        entry["path"]: entry for entry in sf.file_inventory(artifacts, exclude=(sf.CONVERSION_FILE, sf.MODEL_CARD_FILE))
    }
    for path in sorted(set(listed) | set(actual)):
        if path not in actual:
            problems.append(f"missing {path}")
        elif path not in listed:
            problems.append(f"unlisted {path}")
        elif (listed[path]["size"], listed[path]["sha256"]) != (actual[path]["size"], actual[path]["sha256"]):
            problems.append(f"digest mismatch {path}")
    if not (artifacts / sf.MODEL_FILE).is_dir():
        problems.append(f"missing {sf.MODEL_FILE}")
    if sum(entry["size"] for entry in actual.values()) != record.get("total_size"):
        problems.append("total_size does not match the files")

    model_files = sorted(path for path in actual if path.startswith(f"{sf.MODEL_FILE}/"))
    print(f"Artifacts: {artifacts}")
    print(
        f"Files: {len(actual)} ({sum(e['size'] for e in actual.values()):,} bytes), plus "
        f"{sf.CONVERSION_FILE} and {sf.MODEL_CARD_FILE}"
    )
    print(
        f"Source: {source.get('repo')}@{source.get('revision')} {source.get('file')} (SHA-256 {source.get('sha256')})"
    )
    if problems:
        print("Problems:", *problems, sep="\n  ")
        return 1
    print(f"All file digests match {sf.CONVERSION_FILE}.")

    print("\nRepository layout:")
    print(f"  {sf.MODEL_CARD_FILE}")
    print(f"  {sf.CONVERSION_FILE}")
    for path in model_files:
        print(f"  {path}")
    print(f"\nThe runtime manifest's model path inside the repository is {sf.MODEL_FILE!r}.")

    commit_message = f"Add {sf.TITLE} Core ML model"
    commit_description = (
        f"Converted from {source['repo']}@{source['revision']} with coremltools {record['tools']['coremltools']}, "
        f"PyTorch {record['tools']['torch']}, and NeMo {record['tools']['nemo_toolkit']}."
    )
    print("\nDry run. Review, then run these commands yourself:\n")
    print(shlex.join(["hf", "repo", "create", args.repo, "--repo-type", "model"]))
    print(
        shlex.join(
            [
                "hf",
                "upload",
                args.repo,
                str(artifacts),
                ".",
                "--repo-type",
                "model",
                "--commit-message",
                commit_message,
                "--commit-description",
                commit_description,
            ]
        )
    )
    print("\nAfter the upload, record the new commit hash for the runtime manifest:")
    print(shlex.join(["hf", "models", "info", args.repo]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
