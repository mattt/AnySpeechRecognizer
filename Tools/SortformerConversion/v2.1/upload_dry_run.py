# SPDX-License-Identifier: Apache-2.0

"""Check a converted output directory and print the commands that would publish it.

This script never uploads or creates anything on Hugging Face.
It re-hashes every file listed in conversion.json, checks that the directory contains
nothing else except the model card, that the license and notice files are present,
and that the notice contains the sentence the NVIDIA Open Model License requires.
Then it prints `hf` commands for a maintainer to review and run by hand.

Usage:
    uv run python upload_dry_run.py
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
    parser.add_argument("--artifacts", type=Path, default=sf.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo", default=sf.UPLOAD_REPO)
    args = parser.parse_args()

    artifacts = args.artifacts.resolve()
    record = json.loads((artifacts / sf.CONVERSION_FILE).read_text())
    problems = []
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
    for required in (sf.MODEL_FILE, sf.LICENSE_FILE, sf.NOTICE_FILE, sf.MODEL_CARD_FILE):
        if not (artifacts / required).exists():
            problems.append(f"missing {required}")
    if (artifacts / sf.NOTICE_FILE).exists() and sf.REQUIRED_NOTICE not in (artifacts / sf.NOTICE_FILE).read_text():
        problems.append("NOTICE does not contain the required notice")
    license_text = (sf.TOOL_DIR / "NVIDIA-Open-Model-License.txt").read_bytes()
    if (artifacts / sf.LICENSE_FILE).exists() and (artifacts / sf.LICENSE_FILE).read_bytes() != license_text:
        problems.append("LICENSE differs from NVIDIA-Open-Model-License.txt")
    card = (artifacts / sf.MODEL_CARD_FILE).read_text() if (artifacts / sf.MODEL_CARD_FILE).exists() else ""
    for line in ("license: other", f"license_name: {sf.LICENSE_ID}", f"license_link: {sf.LICENSE_URL}"):
        if line not in card:
            problems.append(f"the model card has no `{line}`")

    source = record["source"]
    print(f"Artifacts: {artifacts}")
    print(
        f"Files: {len(actual)} ({sum(e['size'] for e in actual.values()) / 1e6:.1f} MB), plus conversion.json and README.md"
    )
    print(f"Source: {source['repo']}@{source['revision']} {source['file']}")
    if problems:
        print("Problems:", *problems, sep="\n  ")
        return 1
    print("All file digests match conversion.json; LICENSE, NOTICE, and the card's license metadata are present.")

    message = f"Add {sf.TITLE} Core ML artifacts"
    description = (
        f"Converted from {source['repo']}@{source['revision']} with coremltools {record['tools']['coremltools']}. "
        f"{sf.REQUIRED_NOTICE}."
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
                message,
                "--commit-description",
                description,
            ]
        )
    )
    print("\nAfter the upload, record the new commit hash for the runtime manifest:")
    print(shlex.join(["hf", "models", "info", args.repo]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
