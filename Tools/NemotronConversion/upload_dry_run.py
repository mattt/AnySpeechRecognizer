# SPDX-License-Identifier: Apache-2.0
"""Check a package from package.py and print the commands that would publish it.

This script never uploads or creates anything on Hugging Face.
It re-hashes every file that conversion.json lists, checks that the package contains nothing
else except conversion.json and the model card, compares the model files with the verified
conversion, checks the license, notice, and card, and prints `hf` commands for a maintainer
to review and run by hand.

Usage:
    uv run python upload_dry_run.py
    uv run python upload_dry_run.py --package build/package --artifacts build/nemotron
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import nemotron as nt

MODEL_FILES = (nt.ENCODER_FILE, nt.DECODER_FILE, nt.JOINT_FILE, nt.VOCAB_FILE, nt.PROMPTS_FILE)
REQUIRED = (*MODEL_FILES, nt.LICENSE_FILE, nt.NOTICE_FILE, nt.CONVERSION_FILE, nt.MODEL_CARD_FILE)
CHUNK_FUNCTIONS = tuple(chunk.function for chunk in nt.CHUNKS.values())


def check(package: Path, artifacts: Path | None) -> tuple[list[str], dict]:
    problems: list[str] = []
    for name in REQUIRED:
        if not (package / name).exists():
            problems.append(f"missing {name}")
    record = json.loads((package / nt.CONVERSION_FILE).read_text())
    source = record.get("source", {})
    if (source.get("repo"), source.get("revision")) != (nt.REPOSITORY, nt.REVISION):
        problems.append(f"conversion.json names the source {source}")
    functions = [f["function"] for f in record["conversion"]["components"]["encoder"]["functions"]]
    if tuple(functions) != CHUNK_FUNCTIONS:
        problems.append(f"the encoder has the functions {functions}, not {list(CHUNK_FUNCTIONS)}")

    # Every listed file, and nothing else.
    listed = {entry["path"]: entry for entry in record["files"]}
    actual = {
        entry["path"]: entry for entry in nt.file_inventory(package, exclude=(nt.CONVERSION_FILE, nt.MODEL_CARD_FILE))
    }
    for path in sorted(set(listed) | set(actual)):
        if path not in actual:
            problems.append(f"missing {path}")
        elif path not in listed:
            problems.append(f"unlisted {path}")
        elif (listed[path]["size"], listed[path]["sha256"]) != (actual[path]["size"], actual[path]["sha256"]):
            problems.append(f"digest mismatch {path}")

    # The model files are the verified conversion's, byte for byte.
    if artifacts is not None:
        verified = {e["path"]: e for e in json.loads((artifacts / nt.CONVERSION_FILE).read_text())["files"]}
        model_paths = [path for path in listed if path.split("/")[0] in MODEL_FILES]
        for path in model_paths:
            if verified.get(path) != listed[path]:
                problems.append(f"{path} differs from {artifacts / nt.CONVERSION_FILE}")
        for path in verified:
            if path.split("/")[0] in MODEL_FILES and path not in listed:
                problems.append(f"{path} of the verified conversion is not in the package")

    # License, notice, and card.
    license_path = package / nt.LICENSE_FILE
    if license_path.is_file():
        if nt.sha256_file(license_path) != nt.LICENSE_TEXT_SHA256:
            problems.append("LICENSE is not the pinned OpenMDW-1.1 text")
        if not license_path.read_text().startswith("OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)"):
            problems.append("LICENSE does not start with the OpenMDW-1.1 title")
    if record.get("license_text", {}).get("sha256") != nt.LICENSE_TEXT_SHA256:
        problems.append("conversion.json does not record the pinned license text")
    notice_path = package / nt.NOTICE_FILE
    if notice_path.is_file():
        text = notice_path.read_text()
        for required in (
            "NVIDIA Corporation",
            f"https://huggingface.co/{nt.REPOSITORY}",
            nt.REVISION,
            "OpenMDW-1.1",
            nt.LICENSE_TEXT_SOURCE,
            "modifications",
            "int8",
            "multifunction",
            "explicit",
        ):
            if required not in text:
                problems.append(f"NOTICE does not mention {required!r}")
    card_path = package / nt.MODEL_CARD_FILE
    if card_path.is_file():
        card = card_path.read_text()
        front_matter = card.split("---", 2)[1] if card.startswith("---") else ""
        for line in (
            "license: other",
            "license_name: openmdw-1.1",
            "license_link: LICENSE",
            f"base_model: {nt.REPOSITORY}",
            "library_name: coreml",
        ):
            if line not in front_matter.splitlines():
                problems.append(f"the model card's metadata lacks {line!r}")
        for text in (
            "## Accuracy",
            "## Latency",
            "## Languages",
            "## Limitations",
            "## License",
            "less accurate than Parakeet",
            "prototype",
        ):
            if text not in card:
                problems.append(f"the model card lacks {text!r}")
    return problems, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, default=nt.BUILD_DIR / "package", help="A directory from package.py.")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=nt.BUILD_DIR / "nemotron",
        help="The verified conversion to compare model files with (default: build/nemotron when it exists).",
    )
    parser.add_argument("--repo", default=nt.UPLOAD_REPO, help="The destination repository.")
    args = parser.parse_args()

    package = args.package.resolve()
    artifacts = args.artifacts.resolve() if (args.artifacts / nt.CONVERSION_FILE).is_file() else None
    problems, record = check(package, artifacts)

    files = record["files"]
    print(f"Package: {package}")
    print(f"Files: {len(files)} ({sum(e['size'] for e in files) / 1e6:.1f} MB), plus conversion.json and README.md")
    print(f"Source: {record['source']['repo']}@{record['source']['revision']}")
    if artifacts is not None:
        print(f"Model files compared with: {artifacts / nt.CONVERSION_FILE}")
    if problems:
        print("Problems:", *problems, sep="\n  ")
        return 1
    print("All file digests match conversion.json; LICENSE, NOTICE, and the model card pass their checks.")

    commit_message = f"Add {nt.TITLE} Core ML artifacts"
    commit_description = (
        f"Converted from {nt.REPOSITORY}@{nt.REVISION} with coremltools {record['tools']['coremltools']}. "
        "Licensed under OpenMDW-1.1; see LICENSE and NOTICE."
    )
    print("\nDry run. Review, then run these commands yourself:\n")
    print(shlex.join(["hf", "repo", "create", args.repo, "--repo-type", "model", "--no-private"]))
    print(
        shlex.join(
            [
                "hf",
                "upload",
                args.repo,
                str(package),
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
    print("\nAfter the upload, record the commit and pin the runtime manifest from the repository root:")
    print(shlex.join(["hf", "models", "info", args.repo]))
    print(
        "uv run Tools/make-nemotron-manifest.py <commit> "
        "> Sources/AnySpeechRecognizer/Nemotron/NemotronModelManifest+BuiltIn.swift"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
