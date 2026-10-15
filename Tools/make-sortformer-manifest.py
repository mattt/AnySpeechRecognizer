#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""Print a built-in Sortformer diarization manifest as Swift source.

Usage:
    uv run Tools/make-sortformer-manifest.py <revision> \
        > Sources/AnySpeechRecognizer/Diarization/SortformerModelManifest+BuiltIn.swift
    uv run Tools/make-sortformer-manifest.py --version v2.1 <revision> \
        > Sources/AnySpeechRecognizer/Diarization/SortformerModelManifest+V2_1.swift

Each manifest pins the files of one compiled model directory in a Hugging Face repository:
v2 in mattt/diar-streaming-sortformer-4spk-v2-coreml, and v2.1 in
mattt/diar-streaming-sortformer-4spk-v2.1-coreml, which Tools/SortformerConversion/v2.1 produces.
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

HUB = "https://huggingface.co"
NON_LFS_LIMIT = 8 * 1024 * 1024
VERSIONS = {
    "v2": {
        "name": "builtIn",
        "summary": [
            "AnySpeechRecognizer's Core ML conversion of NVIDIA's Streaming Sortformer 4spk v2,",
            "with NVIDIA's low-latency streaming configuration.",
        ],
        "repository": "mattt/diar-streaming-sortformer-4spk-v2-coreml",
        "model": "SortformerLowLatency.mlmodelc",
        "identifier": "diar-streaming-sortformer-4spk-v2",
        "license": "CC-BY-4.0",
        "attribution": (
            "Streaming Sortformer 4spk v2 by NVIDIA "
            "(https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2), "
            "licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
            "Converted to Core ML for AnySpeechRecognizer "
            "(https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2-coreml), "
            "following FluidInference's conversion (https://github.com/FluidInference/mobius)."
        ),
    },
    "v2.1": {
        "name": "v2_1",
        "summary": [
            "AnySpeechRecognizer's Core ML conversion of NVIDIA's Streaming Sortformer 4spk v2.1,",
            "with NVIDIA's low-latency streaming configuration.",
            "The attribution contains the notice that the NVIDIA Open Model License requires.",
        ],
        "repository": "mattt/diar-streaming-sortformer-4spk-v2.1-coreml",
        "model": "SortformerLowLatency.mlmodelc",
        "identifier": "diar-streaming-sortformer-4spk-v2.1",
        "license": "LicenseRef-NVIDIA-Open-Model-License",
        "attribution": (
            "Streaming Sortformer 4spk v2.1 by NVIDIA "
            "(https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1). "
            "Licensed by NVIDIA Corporation under the NVIDIA Open Model License "
            "(https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). "
            "Converted to Core ML for AnySpeechRecognizer "
            "(https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2.1-coreml)."
        ),
    },
}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "make-sortformer-manifest"})
    with urllib.request.urlopen(request) as response:
        return response.read(), dict(response.headers)


def list_files(repository, revision, path):
    quoted = urllib.parse.quote(path)
    url = f"{HUB}/api/models/{repository}/tree/{revision}/{quoted}?recursive=true&expand=true"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--version", choices=sorted(VERSIONS), default="v2", help="the model version (default: v2)"
    )
    parser.add_argument("revision", help="a 40-character commit of the version's repository")
    arguments = parser.parse_args()
    revision = arguments.revision
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        parser.error("the revision must be a 40-character commit hash")
    version = VERSIONS[arguments.version]
    repository, model = version["repository"], version["model"]
    entries = list_files(repository, revision, model)
    if not entries:
        sys.exit(f"error: {repository}@{revision} has no files under {model}")
    files = sorted((entry["path"], *digest(repository, revision, entry)) for entry in entries)
    print(
        f"{repository}@{revision}: {len(files)} files, {sum(f[1] for f in files)} bytes",
        file=sys.stderr,
    )
    command = "uv run Tools/make-sortformer-manifest.py"
    if arguments.version != "v2":
        command += f" --version {arguments.version}"
    out = [
        "// Generated by Tools/make-sortformer-manifest.py. Do not edit.",
        f"// {command} {revision}",
        "",
        "extension SortformerModelManifest {",
        *(f"    /// {line}" for line in version["summary"]),
        # The name follows the version, as `SortformerDiarizationModel.Version.v2_1` does.
        *(["    // swift-format-ignore: AlwaysUseLowerCamelCase"] if "_" in version["name"] else []),
        f"    static let {version['name']} = SortformerModelManifest(",
        f"        identifier: {swift_string(version['identifier'])},",
        f"        repository: {swift_string(repository)},",
        f"        revision: {swift_string(revision)},",
        "        contractVersion: 1,",
        f"        model: {swift_string(model)},",
        "        files: [",
    ]
    for path, size, sha256 in files:
        # Keep lines within swift-format's 100 columns.
        line = f"                path: {swift_string(path)}, size: {size:_},"
        if len(line) > 100:
            line = f"                path: {swift_string(path)},\n                size: {size:_},"
        out += ["            .init(", line, f"                sha256: {swift_string(sha256)}),"]
    out += [
        "        ],",
        f"        license: {swift_string(version['license'])},",
        '        attribution: """',
    ]
    lines = wrap(version["attribution"], 84)
    for index, line in enumerate(lines):
        out.append(f"            {line}" + (" \\" if index < len(lines) - 1 else ""))
    out += ['            """', "    )", "}"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
