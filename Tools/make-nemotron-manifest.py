#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print the built-in Nemotron 3.5 ASR Streaming model manifest as Swift source.

Usage:
    uv run Tools/make-nemotron-manifest.py <revision> \
        > Sources/AnySpeechRecognizer/Nemotron/NemotronModelManifest+BuiltIn.swift

Without arguments, the script prints no manifest, so the model loads only from a staged
directory. With a 40-character commit of the published repository, or `main` to resolve
its current commit, the manifest lists the compiled models, the vocabulary, the prompts,
and the license and notice files. The script reads sizes and SHA-256 digests of Git LFS
files from the Hugging Face API. It downloads only small non-LFS files to hash them and
refuses files over 8 MiB, so it never downloads model weights.

With `--package <directory>`, the script reads the files of a local package from
Tools/NemotronConversion/package.py instead, to check the generated source before an
upload. Do not commit that output: its revision is not a published commit.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"
NON_LFS_LIMIT = 8 * 1024 * 1024
REPOSITORY = "mattt/nemotron-3.5-asr-streaming-0.6b-coreml"
IDENTIFIER = "nemotron-3.5-asr-streaming-0.6b"
COMPONENTS = (
    "Encoder.mlmodelc",
    "Decoder.mlmodelc",
    "JointDecision.mlmodelc",
    "vocab.json",
    "prompts.json",
)
LICENSE_FILES = ("LICENSE", "NOTICE")
ATTRIBUTION = (
    "Nemotron 3.5 ASR Streaming 0.6B by NVIDIA "
    "(https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), licensed under the "
    "OpenMDW License 1.1 (https://openmdw.ai/license/1-1/). Converted to Core ML by "
    f"AnySpeechRecognizer (https://huggingface.co/{REPOSITORY}): the model is split into a "
    "cache-aware encoder step with the language prompt projection, a prediction network, and a "
    "joint decision; weights are stored in float16, and encoder weights are quantized to int8."
)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "make-nemotron-manifest"})
    with urllib.request.urlopen(request) as response:
        return response.read(), dict(response.headers)


def resolve_revision(repository):
    body, _ = fetch(f"{HUB}/api/models/{repository}")
    return json.loads(body)["sha"]


def list_files(repository, revision):
    url = f"{HUB}/api/models/{repository}/tree/{revision}?recursive=true&expand=true"
    entries = []
    while url:
        body, headers = fetch(url)
        entries.extend(json.loads(body))
        link = headers.get("Link") or headers.get("link") or ""
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return [entry for entry in entries if entry["type"] == "file"]


def digest(repository, revision, entry):
    size = int(entry["size"])
    lfs = entry.get("lfs")
    if lfs:
        if int(lfs["size"]) != size:
            sys.exit(f"error: {entry['path']} LFS size does not match its tree size")
        return size, lfs["oid"]
    if size > NON_LFS_LIMIT:
        sys.exit(f"error: {entry['path']} is not stored in LFS and exceeds {NON_LFS_LIMIT} bytes")
    quoted = urllib.parse.quote(entry["path"])
    body, _ = fetch(f"{HUB}/{repository}/resolve/{revision}/{quoted}")
    if len(body) != size:
        sys.exit(f"error: {entry['path']} has {len(body)} bytes; the API reports {size}")
    return size, hashlib.sha256(body).hexdigest()


def wanted(path):
    return path.split("/")[0] in COMPONENTS or path in LICENSE_FILES


def check_complete(paths, source):
    found = {path.split("/")[0] for path in paths}
    missing = [name for name in (*COMPONENTS, *LICENSE_FILES) if name not in found]
    if missing:
        sys.exit(f"error: {source} lacks {', '.join(missing)}")


def published_files(repository, revision):
    """Every file of the components and the license files, with sizes and digests."""
    entries = [entry for entry in list_files(repository, revision) if wanted(entry["path"])]
    check_complete([entry["path"] for entry in entries], f"{repository}@{revision}")
    return sorted((entry["path"], *digest(repository, revision, entry)) for entry in entries)


def package_files(package):
    """The same files of a local package, hashed from disk."""
    files = []
    for path in sorted(p for p in package.rglob("*") if p.is_file()):
        relative = path.relative_to(package).as_posix()
        if wanted(relative):
            files.append(
                (relative, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            )
    check_complete([path for path, _, _ in files], str(package))
    return files


def swift_string(value):
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def wrap(text, width):
    lines, current = [], ""
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


def render(files, revision):
    out = [
        "    /// AnySpeechRecognizer's Core ML conversion of Nemotron 3.5 ASR Streaming 0.6B.",
        "    static let published = NemotronModelManifest(",
        f"        identifier: {swift_string(IDENTIFIER)},",
        f"        repository: {swift_string(REPOSITORY)},",
        f"        revision: {swift_string(revision)},",
        "        contractVersion: 1,",
        "        files: [",
    ]
    for path, size, sha256 in files:
        # Keep lines within swift-format's 100 columns.
        line = f"                path: {swift_string(path)}, size: {size:_},"
        if len(line) > 100:
            line = f"                path: {swift_string(path)},\n                size: {size:_},"
        out += ["            .init(", line, f"                sha256: {swift_string(sha256)}),"]
    out += ["        ],", '        license: "OpenMDW-1.1",', '        attribution: """']
    lines = wrap(ATTRIBUTION, 84)
    for index, line in enumerate(lines):
        out.append(f"            {line}" + (" \\" if index < len(lines) - 1 else ""))
    out += ['            """', "    )"]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "revision", nargs="?", help="a 40-character commit of the published repository, or main"
    )
    parser.add_argument(
        "--package",
        type=Path,
        help="read a local package instead of the repository, for checks only",
    )
    arguments = parser.parse_args()
    revision = arguments.revision
    if arguments.package and not revision:
        parser.error("--package needs a revision to write into the manifest")
    if revision == "main" and not arguments.package:
        revision = resolve_revision(REPOSITORY)
    if revision and not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("the revision must be a 40-character commit hash")

    command = " ".join(
        ["uv run Tools/make-nemotron-manifest.py", *([revision] if revision else [])]
    )
    out = [
        "// Generated by Tools/make-nemotron-manifest.py. Do not edit.",
        f"// {command}",
        "",
        "extension NemotronModelManifest {",
        "    /// The manifest of the published conversion, or `nil` when none is pinned,",
        "    /// so that the model loads only from a staged directory.",
        f"    static let builtIn: NemotronModelManifest? = {'.published' if revision else 'nil'}",
    ]
    if revision:
        files = (
            package_files(arguments.package)
            if arguments.package
            else published_files(REPOSITORY, revision)
        )
        source = arguments.package or f"{REPOSITORY}@{revision}"
        print(f"{source}: {len(files)} files, {sum(f[1] for f in files)} bytes", file=sys.stderr)
        out += ["", render(files, revision)]
    out.append("}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
