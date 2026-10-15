# SPDX-License-Identifier: Apache-2.0

"""Compare the weights of two compiled Sortformer models tensor by tensor.

Usage:
    uv run python compare_weights.py build/sortformer/SortformerLowLatency.mlmodelc \
        <reference>/SortformerNvidiaLow_v2.mlmodelc [--json build/reports/weights.json]

Each `model.mil` names its constant tensors and their offsets in the model's weight file.
The script pairs tensors by name, reads each one, and reports whether the bytes match and,
for those that do not, how far the values differ. It also checks inline constants
(values written in `model.mil` itself) by comparing the program text without its
`buildInfo` line.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

BLOB = re.compile(
    r"tensor<(?P<dtype>\w+), \[(?P<shape>[^\]]*)\]> (?P<name>\S+) = const\(\)\[name = tensor<string, \[\]>\(\"[^\"]*\"\), "
    r"val = tensor<\w+, \[[^\]]*\]>\(BLOBFILE\(path = tensor<string, \[\]>\(\"@model_path/weights/(?P<file>[^\"]+)\"\), "
    r"offset = tensor<uint64, \[\]>\((?P<offset>\d+)\)\)\)\]"
)
DTYPES = {"fp16": np.float16, "fp32": np.float32, "int32": np.int32, "int8": np.int8, "uint8": np.uint8}


def blobs(model_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Return {(submodel, tensor name): {dtype, shape, file, offset}} for every BLOBFILE constant."""
    out = {}
    for mil in sorted(model_dir.rglob("model.mil")):
        submodel = mil.parent.relative_to(model_dir).as_posix()
        for match in BLOB.finditer(mil.read_text()):
            shape = tuple(int(x) for x in match["shape"].split(",") if x.strip())
            out[(submodel, match["name"])] = {
                "dtype": match["dtype"],
                "shape": shape,
                "path": mil.parent / "weights" / match["file"],
                "offset": int(match["offset"]),
            }
    return out


def read_blob(entry: dict[str, Any]) -> np.ndarray:
    """Read one tensor. Each blob starts with a 64-byte header; bytes 8 to 16 hold the data size and offset."""
    with entry["path"].open("rb") as handle:
        handle.seek(entry["offset"])
        header = handle.read(64)
        _sentinel, _dtype, size, data_offset = struct.unpack_from("<IIQQ", header)
        handle.seek(data_offset)
        data = handle.read(size)
    return np.frombuffer(data, dtype=DTYPES[entry["dtype"]]).reshape(entry["shape"])


def program_text(model_dir: Path) -> dict[str, list[str]]:
    """Each model.mil without its buildInfo line, by submodel."""
    return {
        mil.parent.relative_to(model_dir).as_posix(): [
            line for line in mil.read_text().splitlines() if not line.startswith("[buildInfo")
        ]
        for mil in sorted(model_dir.rglob("model.mil"))
    }


def compare(ours: Path, reference: Path) -> dict[str, Any]:
    a, b = blobs(ours), blobs(reference)
    report: dict[str, Any] = {
        "tensors": len(a),
        "reference_tensors": len(b),
        "only_in_ours": sorted(f"{s}/{n}" for s, n in set(a) - set(b)),
        "only_in_reference": sorted(f"{s}/{n}" for s, n in set(b) - set(a)),
        "identical": 0,
        "different": [],
    }
    for key in sorted(set(a) & set(b)):
        x, y = read_blob(a[key]), read_blob(b[key])
        if x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes():
            report["identical"] += 1
            continue
        entry = {"tensor": f"{key[0]}/{key[1]}", "dtype": a[key]["dtype"], "shape": list(x.shape)}
        if x.shape == y.shape:
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            diff = np.abs(xf - yf)
            entry.update(
                {
                    "values_different": int(np.count_nonzero(x != y)),
                    "values": int(x.size),
                    "max_abs_difference": float(diff.max()),
                    "max_abs_value": float(np.abs(yf).max()),
                }
            )
            if x.dtype == np.float16:
                # Distance in units in the last place (ULP) of float16.
                ulp = np.abs(x.view(np.int16).astype(np.int32) - y.view(np.int16).astype(np.int32))
                entry["max_ulp_difference"] = int(ulp.max())
        report["different"].append(entry)
    texts_a, texts_b = program_text(ours), program_text(reference)
    report["programs"] = {
        submodel: {
            "identical_without_build_info": texts_a.get(submodel) == texts_b.get(submodel),
            "lines_different": sum(
                1 for la, lb in zip(texts_a.get(submodel, []), texts_b.get(submodel, [])) if la != lb
            )
            + abs(len(texts_a.get(submodel, [])) - len(texts_b.get(submodel, []))),
        }
        for submodel in sorted(set(texts_a) | set(texts_b))
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ours", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--json", type=Path, help="Write the report as JSON")
    args = parser.parse_args()
    report = compare(args.ours, args.reference)
    print(f"Tensors in weight files: {report['tensors']} (reference {report['reference_tensors']})")
    print(f"Byte-identical: {report['identical']}; different: {len(report['different'])}")
    for entry in report["different"]:
        print(
            f"  {entry['tensor']} {entry['dtype']} {entry['shape']}: "
            f"{entry.get('values_different')} of {entry.get('values')} values differ, "
            f"max |difference| {entry.get('max_abs_difference')}, max ULP {entry.get('max_ulp_difference')}"
        )
    for name in report["only_in_ours"]:
        print(f"  only in ours: {name}")
    for name in report["only_in_reference"]:
        print(f"  only in reference: {name}")
    for submodel, program in report["programs"].items():
        print(
            f"{submodel}/model.mil without buildInfo: "
            + ("identical" if program["identical_without_build_info"] else f"{program['lines_different']} lines differ")
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
