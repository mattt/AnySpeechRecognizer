# SPDX-License-Identifier: Apache-2.0

"""Check a converted output directory and print the commands that would publish it.

This script never uploads or creates anything on Hugging Face.
It re-hashes every file listed in conversion.json, checks that the directory
contains nothing else except the model card, and prints `hf` commands for a
maintainer to review and run by hand.

Usage:
    uv run python upload_dry_run.py --variant v3
    uv run python upload_dry_run.py --variant v3 --update   # a new revision of an existing repository
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import parakeet as pk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(pk.VARIANTS), required=True)
    parser.add_argument("--artifacts", type=Path, help="Converted artifacts (default: build/<variant>)")
    parser.add_argument("--repo", help="Destination repository (default: the variant's proposed repository)")
    parser.add_argument(
        "--update", action="store_true", help="Propose a new revision of an existing repository with a full conversion"
    )
    args = parser.parse_args()

    variant = pk.VARIANTS[args.variant]
    artifacts = (args.artifacts or pk.BUILD_DIR / variant.name).resolve()
    repo = args.repo or variant.upload_repo
    record = json.loads((artifacts / pk.CONVERSION_FILE).read_text())

    problems = []
    if record.get("variant") != variant.name:
        problems.append(f"conversion.json describes variant {record.get('variant')!r}")
    if not (artifacts / pk.MODEL_CARD_FILE).is_file():
        problems.append(f"missing {pk.MODEL_CARD_FILE}")
    listed = {entry["path"]: entry for entry in record["files"]}
    actual = {
        entry["path"]: entry for entry in pk.file_inventory(artifacts, exclude=(pk.CONVERSION_FILE, pk.MODEL_CARD_FILE))
    }
    for path in sorted(set(listed) | set(actual)):
        if path not in actual:
            problems.append(f"missing {path}")
        elif path not in listed:
            problems.append(f"unlisted {path}")
        elif (listed[path]["size"], listed[path]["sha256"]) != (actual[path]["size"], actual[path]["sha256"]):
            problems.append(f"digest mismatch {path}")
    windows = [w["file"] for w in record["conversion"]["components"].get("encoder_windows", [])]
    for required in (pk.ENCODER_FILE, *windows, pk.DECODER_FILE, pk.JOINT_FILE, pk.VOCAB_FILE):
        if not (artifacts / required).exists():
            problems.append(f"missing {required}")
    if variant.license_text and (
        not (artifacts / pk.LICENSE_FILE).is_file()
        or (artifacts / pk.LICENSE_FILE).read_bytes() != variant.license_text.read_bytes()
    ):
        problems.append(f"{pk.LICENSE_FILE} is not a copy of {variant.license_text.name}")
    if variant.required_notice:
        notice = artifacts / pk.NOTICE_FILE
        if not notice.is_file() or variant.required_notice not in notice.read_text():
            problems.append(f"{pk.NOTICE_FILE} does not contain the required notice")
        card = (artifacts / pk.MODEL_CARD_FILE).read_text() if (artifacts / pk.MODEL_CARD_FILE).is_file() else ""
        if variant.card_license not in card:
            problems.append(f"{pk.MODEL_CARD_FILE} does not have the license metadata")

    print(f"Artifacts: {artifacts}")
    print(
        f"Files: {len(actual)} ({sum(e['size'] for e in actual.values()) / 1e6:.1f} MB), plus conversion.json and README.md"
    )
    for source in record["sources"]:
        print(f"Source: {source['repo']}@{source['revision']} {source['file']} ({source['role']})")
    if problems:
        print("Problems:", *problems, sep="\n  ")
        return 1
    print("All file digests match conversion.json.")

    commit_message = f"Add {variant.title} Core ML artifacts"
    sources = ", ".join(f"{s['repo']}@{s['revision']}" for s in record["sources"] if s["role"] != "tokenizer check")
    commit_description = f"Converted from {sources} with coremltools {record['tools']['coremltools']}."
    added = record.get("encoder_windows_added")
    if args.update and not added:
        nemo = record["tools"]["nemo_toolkit"]
        commit_message = f"Convert {variant.title} again with NeMo {nemo}"
        commit_description = (
            f"Converted from {sources} with NeMo {nemo} and coremltools {record['tools']['coremltools']}. "
            "The encoders take mel features with floor(samples / 160) valid frames and mask frames "
            "after mel_length between the subsampling convolutions, as NeMo 2.5 and later do."
        )
    if added:
        # convert.py --reuse-from copied every other file from an earlier output unchanged.
        seconds = [str(w["window_seconds"]) for w in record["conversion"]["components"]["encoder_windows"]]
        commit_message = f"Add {variant.title} encoders for {', '.join(seconds)} s windows"
        commit_description += f" Other model files are unchanged from {added['other_files_copied_from']}."
    print("\nDry run. Review, then run these commands yourself:\n")
    if added:
        print("# The repository already exists; the upload adds the new encoders and updates")
        print("# conversion.json and README.md. Check that the other files match the published revision.")
    elif args.update:
        print("# The repository already exists; the upload replaces every file with the same names.")
    else:
        print(shlex.join(["hf", "repo", "create", repo, "--repo-type", "model"]))
    print(
        shlex.join(
            [
                "hf",
                "upload",
                repo,
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
    print(shlex.join(["hf", "models", "info", repo]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
