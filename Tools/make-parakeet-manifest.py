#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print built-in Parakeet model manifests as Swift source.

Usage:
    uv run Tools/make-parakeet-manifest.py v3@<revision> ultra@<revision> \
        > Sources/AnySpeechRecognizer/Parakeet/Assets/ParakeetModelManifest+BuiltIn.swift

Omit `@<revision>` to resolve the repository's current `main` commit.
Each preset names an optional live encoder, a shorter fixed-window encoder for live input.
The manifest lists it only when the revision contains it.
The runtime installs its files only for modules with volatile or fast results.
The script reads sizes and SHA-256 digests of Git LFS files from the Hugging Face API.
It downloads only small non-LFS files to hash them and refuses files over 8 MiB,
so it never downloads model weights.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass

HUB = "https://huggingface.co"
NON_LFS_LIMIT = 8 * 1024 * 1024


@dataclass(frozen=True)
class Preset:
    swift_name: str
    summary: str
    identifier: str
    repository: str
    contract_version: int
    encoder: str
    decoder: str
    joint: str
    vocabulary: str
    license: str
    attribution: str
    live_encoder: str | None = None


PRESETS = {
    "v3": Preset(
        swift_name="v3",
        summary="The package's Core ML conversion of NVIDIA's parakeet-tdt-0.6b-v3 "
        "with an int8 encoder.",
        identifier="parakeet-tdt-0.6b-v3",
        repository="mattt/parakeet-tdt-0.6b-v3-coreml",
        contract_version=1,
        encoder="Encoder.mlmodelc",
        decoder="Decoder.mlmodelc",
        joint="JointDecisionv3.mlmodelc",
        vocabulary="parakeet_vocab.json",
        live_encoder="Encoder8s.mlmodelc",
        license="CC-BY-4.0",
        attribution=(
            "parakeet-tdt-0.6b-v3 by NVIDIA (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), "
            "licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
            "Converted to Core ML with an int8 encoder by AnySpeechRecognizer "
            "(https://huggingface.co/mattt/parakeet-tdt-0.6b-v3-coreml), "
            "following FluidInference's conversion (https://github.com/FluidInference/mobius)."
        ),
    ),
    "ultra": Preset(
        swift_name="ultra",
        summary="The package's Core ML conversion of Moondream's Parakeet Ultra "
        "with an int8 encoder.",
        identifier="parakeet-ultra",
        repository="mattt/parakeet-ultra-coreml",
        contract_version=1,
        encoder="Encoder.mlmodelc",
        decoder="Decoder.mlmodelc",
        joint="JointDecisionv3.mlmodelc",
        vocabulary="parakeet_vocab.json",
        live_encoder="Encoder8s.mlmodelc",
        license="CC-BY-4.0",
        attribution=(
            "Parakeet Ultra by Moondream (https://huggingface.co/moondream/parakeet-ultra), "
            "a post-trained version of parakeet-tdt-0.6b-v3 by NVIDIA "
            "(https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), "
            "licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
            "Converted to Core ML with an int8 encoder by AnySpeechRecognizer "
            "(https://huggingface.co/mattt/parakeet-ultra-coreml), "
            "following FluidInference's conversion (https://github.com/FluidInference/mobius)."
        ),
    ),
    "unified": Preset(
        swift_name="unified",
        summary="The package's Core ML conversion of NVIDIA's parakeet-unified-en-0.6b "
        "with an int8 encoder. The attribution contains the notice that the NVIDIA Open "
        "Model License requires.",
        identifier="parakeet-unified-en-0.6b",
        repository="mattt/parakeet-unified-en-0.6b-coreml",
        contract_version=1,
        encoder="Encoder.mlmodelc",
        decoder="Decoder.mlmodelc",
        joint="JointDecisionv3.mlmodelc",
        vocabulary="parakeet_vocab.json",
        live_encoder="Encoder8s.mlmodelc",
        license="LicenseRef-NVIDIA-Open-Model-License",
        attribution=(
            "parakeet-unified-en-0.6b by NVIDIA "
            "(https://huggingface.co/nvidia/parakeet-unified-en-0.6b). "
            "Licensed by NVIDIA Corporation under the NVIDIA Open Model License "
            "(https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). "
            "Converted to Core ML with an int8 encoder by AnySpeechRecognizer "
            "(https://huggingface.co/mattt/parakeet-unified-en-0.6b-coreml)."
        ),
    ),
}


def fetch(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "make-parakeet-manifest"})
    with urllib.request.urlopen(request) as response:
        return response.read(), dict(response.headers)


def resolve_revision(repository: str) -> str:
    body, _ = fetch(f"{HUB}/api/models/{repository}")
    return json.loads(body)["sha"]


def list_files(repository: str, revision: str, path: str) -> list[dict]:
    quoted = urllib.parse.quote(path)
    url = f"{HUB}/api/models/{repository}/tree/{revision}/{quoted}?recursive=true&expand=true"
    entries: list[dict] = []
    while url:
        body, headers = fetch(url)
        entries.extend(json.loads(body))
        link = headers.get("Link") or headers.get("link") or ""
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return [entry for entry in entries if entry["type"] == "file"]


def hash_small_file(repository: str, revision: str, path: str, size: int) -> str:
    if size > NON_LFS_LIMIT:
        sys.exit(f"error: {path} is not stored in LFS and exceeds {NON_LFS_LIMIT} bytes")
    quoted = urllib.parse.quote(path)
    body, _ = fetch(f"{HUB}/{repository}/resolve/{revision}/{quoted}")
    if len(body) != size:
        sys.exit(f"error: {path} has {len(body)} bytes; the API reports {size}")
    return hashlib.sha256(body).hexdigest()


def manifest_files(
    preset: Preset, revision: str
) -> tuple[list[tuple[str, int, str]], str | None]:
    """Return the files and the live encoder, or `None` when the revision lacks it."""
    files: dict[str, tuple[int, str]] = {}
    for root in (preset.encoder, preset.decoder, preset.joint):
        entries = list_files(preset.repository, revision, root)
        if not entries:
            sys.exit(f"error: {preset.repository}@{revision} has no files under {root}")
        for entry in entries:
            files[entry["path"]] = file_record(preset, revision, entry)
    entries = list_files(preset.repository, revision, "")
    vocabulary = [entry for entry in entries if entry["path"] == preset.vocabulary]
    if not vocabulary:
        sys.exit(f"error: {preset.repository}@{revision} has no {preset.vocabulary}")
    files[preset.vocabulary] = file_record(preset, revision, vocabulary[0])
    live_encoder = preset.live_encoder
    if live_encoder:
        prefix = live_encoder + "/"
        live_entries = [entry for entry in entries if entry["path"].startswith(prefix)]
        if live_entries:
            for entry in live_entries:
                files[entry["path"]] = file_record(preset, revision, entry)
        else:
            print(
                f"{preset.repository}@{revision} has no {live_encoder}; "
                "the manifest omits the optional live encoder",
                file=sys.stderr,
            )
            live_encoder = None
    listed = [(path, size, digest) for path, (size, digest) in sorted(files.items())]
    return listed, live_encoder


def file_record(preset: Preset, revision: str, entry: dict) -> tuple[int, str]:
    size = int(entry["size"])
    lfs = entry.get("lfs")
    if lfs:
        if int(lfs["size"]) != size:
            sys.exit(f"error: {entry['path']} LFS size does not match its tree size")
        return size, lfs["oid"]
    return size, hash_small_file(preset.repository, revision, entry["path"], size)


def swift_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def swift_integer(value: int) -> str:
    return f"{value:_}"


def wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render(preset: Preset, revision: str) -> str:
    files, live_encoder = manifest_files(preset, revision)
    total = sum(size for _, size, _ in files)
    print(
        f"{preset.repository}@{revision}: {len(files)} files, {total} bytes",
        file=sys.stderr,
    )
    out = []
    for line in wrap(preset.summary, 88):
        out.append(f"    /// {line}")
    out.append(f"    static let {preset.swift_name} = ParakeetModelManifest(")
    out.append(f"        identifier: {swift_string(preset.identifier)},")
    out.append(f"        repository: {swift_string(preset.repository)},")
    out.append(f"        revision: {swift_string(revision)},")
    out.append(f"        contractVersion: {preset.contract_version},")
    out.append("        components: .init(")
    out.append(
        f"            encoder: {swift_string(preset.encoder)}, "
        f"decoder: {swift_string(preset.decoder)},"
    )
    out.append(
        f"            joint: {swift_string(preset.joint)}, "
        f"vocabulary: {swift_string(preset.vocabulary)}" + ("," if live_encoder else "),")
    )
    if live_encoder:
        out.append(f"            liveEncoder: {swift_string(live_encoder)}),")
    out.append("        files: [")
    for path, size, digest in files:
        out.append("            .init(")
        out.append(f"                path: {swift_string(path)}, size: {swift_integer(size)},")
        out.append(f"                sha256: {swift_string(digest)}),")
    out.append("        ],")
    out.append(f"        license: {swift_string(preset.license)},")
    out.append('        attribution: """')
    attribution = wrap(preset.attribution, 84)
    for index, line in enumerate(attribution):
        suffix = " \\" if index < len(attribution) - 1 else ""
        out.append(f"            {line}{suffix}")
    out.append('            """')
    out.append("    )")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "presets",
        nargs="+",
        metavar="preset[@revision]",
        help=f"one of {', '.join(PRESETS)}, optionally pinned to a 40-character commit",
    )
    arguments = parser.parse_args()
    pinned: list[tuple[Preset, str]] = []
    for argument in arguments.presets:
        name, _, revision = argument.partition("@")
        if name not in PRESETS:
            parser.error(f"unknown preset {name!r}")
        preset = PRESETS[name]
        revision = revision or resolve_revision(preset.repository)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            parser.error(f"{argument}: the revision must be a 40-character commit hash")
        pinned.append((preset, revision))

    pins = [f"{preset.swift_name}@{revision}" for preset, revision in pinned]
    print("// Generated by Tools/make-parakeet-manifest.py. Do not edit.")
    print("// uv run Tools/make-parakeet-manifest.py \\")
    print(" \\\n".join(f"//     {argument}" for argument in pins))
    print()
    print("extension ParakeetModelManifest {")
    print("\n\n".join(render(preset, revision) for preset, revision in pinned))
    print("}")


if __name__ == "__main__":
    main()
