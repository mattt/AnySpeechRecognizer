#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Print the built-in Qwen3-ASR model manifests as Swift source.

Usage:
    uv run Tools/make-qwen-manifest.py 1.7b@<revision> \
        > Sources/AnySpeechRecognizer/Qwen/QwenModelManifest+BuiltIn.swift

Without arguments, the script prints a table with no manifests, so that every size loads only
from a staged directory. Each argument pins one size's published repository to a 40-character
commit; omit `@<revision>` to resolve the repository's current `main` commit.
The manifest lists the compiled models, the vocabulary, the configuration, and the license
and notice files. The script reads sizes and SHA-256 digests of Git LFS files from the
Hugging Face API. It downloads only small non-LFS files to hash them and refuses files over
8 MiB, so it never downloads model weights.
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
COMPONENTS = {
    "encoder": "AudioEncoder.mlmodelc",
    "decoder": "TextDecoder.mlmodelc",
    "vocabulary": "vocab.json",
    "configuration": "qwen3_asr_config.json",
}
LICENSE_FILES = ("LICENSE", "NOTICE")
PRESETS = {
    "0.6b": {
        "case": "small",
        "summary": "AnySpeechRecognizer's Core ML conversion of Qwen3-ASR 0.6B.",
        "identifier": "qwen3-asr-0.6b",
        "repository": "mattt/qwen3-asr-0.6b-coreml",
        "title": "Qwen3-ASR 0.6B",
        "source": "Qwen/Qwen3-ASR-0.6B-hf",
    },
    "1.7b": {
        "case": "large",
        "summary": "AnySpeechRecognizer's Core ML conversion of Qwen3-ASR 1.7B.",
        "identifier": "qwen3-asr-1.7b",
        "repository": "mattt/qwen3-asr-1.7b-coreml",
        "title": "Qwen3-ASR 1.7B",
        "source": "Qwen/Qwen3-ASR-1.7B-hf",
    },
}


def attribution(preset):
    return (
        f"{preset['title']} by the Qwen team, Alibaba Cloud "
        f"(https://huggingface.co/{preset['source']}), licensed under the Apache License, "
        "Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). "
        "Converted to Core ML by AnySpeechRecognizer "
        f"(https://huggingface.co/{preset['repository']}): the model is split into an audio encoder "
        "and a text decoder with a KV cache, weights are stored in float16, "
        "and weights of large layers are quantized to int8."
    )


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "make-qwen-manifest"})
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


def pinned_files(repository, revision):
    """Every file of the components and the license files, with sizes and digests."""
    wanted = []
    for entry in list_files(repository, revision):
        path = entry["path"]
        top = path.split("/")[0]
        if top in COMPONENTS.values() or path in LICENSE_FILES:
            wanted.append(entry)
    found = {entry["path"].split("/")[0] for entry in wanted}
    missing = [name for name in (*COMPONENTS.values(), *LICENSE_FILES) if name not in found]
    if missing:
        sys.exit(f"error: {repository}@{revision} lacks {', '.join(missing)}")
    return sorted((entry["path"], *digest(repository, revision, entry)) for entry in wanted)


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


def render(preset, revision):
    repository = preset["repository"]
    files = pinned_files(repository, revision)
    print(
        f"{repository}@{revision}: {len(files)} files, {sum(f[1] for f in files)} bytes",
        file=sys.stderr,
    )
    out = [
        f"    /// {preset['summary']}",
        f"    static let {preset['case']} = QwenModelManifest(",
        f"        identifier: {swift_string(preset['identifier'])},",
        f"        repository: {swift_string(repository)},",
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
    out += ["        ],", '        license: "Apache-2.0",', '        attribution: """']
    lines = wrap(attribution(preset), 84)
    for index, line in enumerate(lines):
        out.append(f"            {line}" + (" \\" if index < len(lines) - 1 else ""))
    out += ['            """', "    )"]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "sizes",
        nargs="*",
        metavar="size[@revision]",
        help=f"one of {', '.join(PRESETS)}, optionally pinned to a 40-character commit",
    )
    arguments = parser.parse_args()
    pinned = []
    for argument in arguments.sizes:
        name, _, revision = argument.partition("@")
        if name not in PRESETS:
            parser.error(f"unknown size {name!r}")
        preset = PRESETS[name]
        revision = revision or resolve_revision(preset["repository"])
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            parser.error(f"{argument}: the revision must be a 40-character commit hash")
        pinned.append((name, preset, revision))

    command = " ".join(["uv run Tools/make-qwen-manifest.py", *(f"{n}@{r}" for n, _, r in pinned)])
    entries = ", ".join(f".{preset['case']}: .{preset['case']}" for _, preset, _ in pinned) or ":"
    out = [
        "// Generated by Tools/make-qwen-manifest.py. Do not edit.",
        f"// {command}",
        "",
        "extension QwenModelManifest {",
        "    /// The manifests of the published conversions, by size.",
        "    /// Sizes without a manifest load only from a staged directory.",
        f"    static let builtIn: [QwenSpeechModel.Variant: QwenModelManifest] = [{entries}]",
    ]
    for _, preset, revision in pinned:
        out += ["", render(preset, revision)]
    out.append("}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
