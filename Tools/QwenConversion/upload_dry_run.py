# SPDX-License-Identifier: Apache-2.0
"""Check a package from package.py and print the commands that would publish it.

This script never uploads or creates anything on Hugging Face.
It re-hashes every file that conversion.json lists, checks that the package contains nothing
else except conversion.json and the model card, checks the license, notice, and card,
and prints `hf` commands for a maintainer to review and run by hand.

Usage:
    uv run python upload_dry_run.py --variant 1.7b --package /tmp/qwen-1.7b-publish
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import qwen

REQUIRED = (
    qwen.ENCODER_FILE,
    qwen.DECODER_FILE,
    qwen.VOCAB_FILE,
    qwen.CONFIG_FILE,
    qwen.LICENSE_FILE,
    qwen.NOTICE_FILE,
)


def check(package: Path, variant: qwen.Variant, artifacts: Path | None) -> tuple[list[str], dict]:
    problems: list[str] = []
    record = json.loads((package / qwen.CONVERSION_FILE).read_text())
    if record.get("variant") != variant.name:
        problems.append(f"conversion.json describes variant {record.get('variant')!r}")
    if record.get("source") != {"repository": variant.repo, "revision": variant.revision}:
        problems.append(f"conversion.json names the source {record.get('source')}")

    # Every listed file, and nothing else.
    listed = {entry["path"]: entry for entry in record["files"]}
    actual = {
        entry["path"]: entry
        for entry in qwen.tree_files(package)
        if entry["path"] not in (qwen.CONVERSION_FILE, qwen.MODEL_CARD_FILE)
    }
    for path in sorted(set(listed) | set(actual)):
        if path not in actual:
            problems.append(f"missing {path}")
        elif path not in listed:
            problems.append(f"unlisted {path}")
        elif (listed[path]["size"], listed[path]["sha256"]) != (actual[path]["size"], actual[path]["sha256"]):
            problems.append(f"digest mismatch {path}")
    for name in REQUIRED:
        if not (package / name).exists():
            problems.append(f"missing {name}")

    # The model files are the verified conversion's, byte for byte.
    if artifacts is not None:
        verified = {e["path"]: e for e in json.loads((artifacts / qwen.CONVERSION_FILE).read_text())["files"]}
        for path, entry in listed.items():
            if path.split("/")[0] in REQUIRED[:4] and verified.get(path) != entry:
                problems.append(f"{path} differs from {artifacts / qwen.CONVERSION_FILE}")

    # License, notice, and card.
    license_path = package / qwen.LICENSE_FILE
    if license_path.is_file():
        text = license_path.read_text()
        if qwen.sha256(license_path) != qwen.LICENSE_TEXT_SHA256:
            problems.append("LICENSE is not the pinned Apache License text")
        if "Apache License" not in text or "Version 2.0, January 2004" not in text:
            problems.append("LICENSE is not the Apache License, Version 2.0")
    notice_path = package / qwen.NOTICE_FILE
    if notice_path.is_file():
        text = notice_path.read_text()
        for required in (variant.repo, variant.revision, "Alibaba Cloud", "Apache License, Version 2.0", "changes"):
            if required not in text:
                problems.append(f"NOTICE does not mention {required!r}")
    card_path = package / qwen.MODEL_CARD_FILE
    if not card_path.is_file():
        problems.append(f"missing {qwen.MODEL_CARD_FILE}")
    else:
        card = card_path.read_text()
        front_matter = card.split("---", 2)[1] if card.startswith("---") else ""
        for line in ("license: apache-2.0", f"base_model: {variant.repo}", "library_name: coreml"):
            if line not in front_matter.splitlines():
                problems.append(f"the model card's metadata lacks {line!r}")
        for heading in ("## Limitations", "## Size and resources", "## Evaluation", "## License"):
            if heading not in card:
                problems.append(f"the model card lacks {heading!r}")
    return problems, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(qwen.VARIANTS), required=True)
    parser.add_argument("--package", type=Path, required=True, help="A directory that package.py wrote.")
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="The verified conversion to compare model files with (default: build/<variant> when it exists).",
    )
    parser.add_argument("--repo", help="Destination repository (default: the variant's proposed repository)")
    args = parser.parse_args()

    variant = qwen.VARIANTS[args.variant]
    package = args.package.resolve()
    artifacts = args.artifacts or qwen.BUILD_DIR / variant.name
    artifacts = artifacts.resolve() if (artifacts / qwen.CONVERSION_FILE).is_file() else None
    repo = args.repo or variant.upload_repo
    problems, record = check(package, variant, artifacts)

    files = record["files"]
    print(f"Package: {package}")
    print(f"Files: {len(files)} ({sum(e['size'] for e in files) / 1e9:.3f} GB), plus conversion.json and README.md")
    print(f"Source: {record['source']['repository']}@{record['source']['revision']}")
    if artifacts is not None:
        print(f"Model files compared with: {artifacts / qwen.CONVERSION_FILE}")
    if problems:
        print("Problems:", *problems, sep="\n  ")
        return 1
    print("All file digests match conversion.json; LICENSE, NOTICE, and the model card pass their checks.")

    commit_message = f"Add {variant.title} Core ML artifacts"
    commit_description = (
        f"Converted from {variant.repo}@{variant.revision} with coremltools {record['tools']['coremltools']}. "
        "Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE."
    )
    print("\nDry run. Review, then run these commands yourself:\n")
    print(shlex.join(["hf", "repo", "create", repo, "--repo-type", "model", "--no-private"]))
    print(
        shlex.join(
            [
                "hf",
                "upload",
                repo,
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
    print(shlex.join(["hf", "models", "info", repo]))
    print(
        f"uv run Tools/make-qwen-manifest.py {variant.name}@<commit> "
        "> Sources/AnySpeechRecognizer/Qwen/QwenModelManifest+BuiltIn.swift"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
