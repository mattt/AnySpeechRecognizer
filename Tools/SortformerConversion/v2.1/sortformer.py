# SPDX-License-Identifier: Apache-2.0

"""Shared definitions for converting and verifying Streaming Sortformer 4spk v2.1 for Core ML.

The converted model has the tensor interface of the Streaming Sortformer v2 model that
AnySpeechRecognizer's `CoreMLSortformerEngine` loads: one call runs the pre-encoder on a
chunk of log-mel features, packs the speaker cache, FIFO, and chunk embeddings, and runs
the Fast Conformer encoder, the Transformer, and the speaker heads.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

TOOL_DIR = Path(__file__).resolve().parent
BUILD_DIR = TOOL_DIR / "build"
DEFAULT_CACHE_DIR = BUILD_DIR / "cache"
DEFAULT_OUTPUT_DIR = BUILD_DIR / "artifacts"
DEFAULT_WORK_DIR = BUILD_DIR / "work"
REPORTS_DIR = BUILD_DIR / "reports"


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    file: str
    sha256: str
    size: int


SOURCE = Source(
    repo="nvidia/diar_streaming_sortformer_4spk-v2.1",
    revision="cd03eee90fbec18297ac31b8c21546e596b7f71c",
    file="diar_streaming_sortformer_4spk-v2.1.nemo",
    sha256="8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8",
    size=471_367_680,
)
UPLOAD_REPO = "mattt/diar-streaming-sortformer-4spk-v2.1-coreml"
MODEL_FILE = "SortformerLowLatency.mlmodelc"
CONVERSION_FILE = "conversion.json"
MODEL_CARD_FILE = "README.md"
LICENSE_FILE = "LICENSE"
NOTICE_FILE = "NOTICE"
TITLE = "Streaming Sortformer 4spk v2.1"

# NVIDIA's "low latency" configuration from the model card, in 80 ms frames.
STREAMING = {
    "chunk_len": 6,
    "chunk_left_context": 1,
    "chunk_right_context": 7,
    "fifo_len": 188,
    "spkcache_len": 188,
    "spkcache_update_period": 144,
}
SUBSAMPLING = 8
MEL_BINS = 128
EMBEDDING_SIZE = 512
SPEAKERS = 4
FRAME_SECONDS = 0.08
CHUNK_MEL_FRAMES = (
    STREAMING["chunk_left_context"] + STREAMING["chunk_len"] + STREAMING["chunk_right_context"]
) * SUBSAMPLING
CHUNK_EMBEDDINGS = CHUNK_MEL_FRAMES // SUBSAMPLING
PACKED_ROWS = STREAMING["spkcache_len"] + STREAMING["fifo_len"] + CHUNK_EMBEDDINGS

# The tensor interface of the runtime's v2 model, which this conversion keeps.
CONTRACT: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {
    "inputs": {
        "chunk": ("float32", (1, CHUNK_MEL_FRAMES, MEL_BINS)),
        "chunk_lengths": ("int32", (1,)),
        "spkcache": ("float32", (1, STREAMING["spkcache_len"], EMBEDDING_SIZE)),
        "spkcache_lengths": ("int32", (1,)),
        "fifo": ("float32", (1, STREAMING["fifo_len"], EMBEDDING_SIZE)),
        "fifo_lengths": ("int32", (1,)),
    },
    "outputs": {
        "speaker_preds": ("float32", (1, PACKED_ROWS, SPEAKERS)),
        "chunk_pre_encoder_embs_out": ("float32", (1, CHUNK_EMBEDDINGS, EMBEDDING_SIZE)),
        "chunk_pre_encoder_lengths_out": ("int32", (1,)),
    },
}

# Creator-defined metadata that the runtime reads, with the same keys as the v2 model.
STREAMING_METADATA = {
    **{key: str(value) for key, value in STREAMING.items()},
    "subsampling_factor": str(SUBSAMPLING),
    "frame_duration": str(FRAME_SECONDS),
    "mel_feature_frames": str(STREAMING["chunk_len"] * SUBSAMPLING),
}

# License facts. The NVIDIA Open Model License has no SPDX identifier.
LICENSE_NAME = "NVIDIA Open Model License Agreement"
LICENSE_ID = "nvidia-open-model-license"
LICENSE_URL = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/"
LICENSE_PDF_URL = (
    "https://www.nvidia.com/content/dam/en-zz/Solutions/license-agreements/enterprise-software/"
    "nvidia-open-model-license-agreements-24-10-2025.pdf"
)
LICENSE_VERSION = "October 24, 2025"
TRUSTWORTHY_AI_URL = "https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/"
# Section 3.1 requires this sentence in a "Notice" text file with every copy.
REQUIRED_NOTICE = "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
ATTRIBUTION = (
    f"{TITLE} by NVIDIA (https://huggingface.co/{SOURCE.repo}). "
    f"{REQUIRED_NOTICE} ({LICENSE_URL}). "
    f"Converted to Core ML for AnySpeechRecognizer (https://huggingface.co/{UPLOAD_REPO})."
)
MODIFICATIONS = (
    "The NeMo checkpoint was converted to a Core ML pipeline for NVIDIA's low-latency streaming "
    "configuration: the pre-encoder runs in float32, the encoder, Transformer, and speaker heads run "
    "with float16 weights and compute precision, and the speaker cache, FIFO, and chunk embeddings "
    "are packed inside the model. The weights were not trained, fine-tuned, or pruned."
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(root: Path, exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Return the relative path, size, and SHA-256 digest of every file under root."""
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude or path.name == ".DS_Store":
            continue
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tool_versions() -> dict[str, Any]:
    from importlib import metadata

    packages = ["torch", "torchaudio", "nemo_toolkit", "coremltools", "numpy", "huggingface_hub"]
    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    versions["coremlcompiler"] = _command_output(["xcrun", "coremlcompiler", "version"])
    versions["macos"] = platform.mac_ver()[0]
    versions["macos_build"] = _command_output(["sw_vers", "-buildVersion"])
    versions["machine"] = _command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.machine()
    return versions


def download_checkpoint(cache_dir: Path) -> Path:
    """Download the pinned `.nemo` checkpoint and check its size and SHA-256 digest."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=SOURCE.repo,
            filename=SOURCE.file,
            revision=SOURCE.revision,
            local_dir=cache_dir / SOURCE.repo.replace("/", "--") / SOURCE.revision,
        )
    )
    if path.stat().st_size != SOURCE.size:
        raise RuntimeError(f"{path} has {path.stat().st_size} bytes; expected {SOURCE.size}")
    digest = sha256_file(path)
    if digest != SOURCE.sha256:
        raise RuntimeError(f"{path} has SHA-256 {digest}; expected {SOURCE.sha256}")
    return path


def load_model(checkpoint: Path):
    """Load the checkpoint on the CPU with NVIDIA's low-latency streaming configuration."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(str(checkpoint), map_location="cpu")
    model.eval()
    model.streaming_mode = True
    model.async_streaming = False
    modules = model.sortformer_modules
    for key, value in STREAMING.items():
        setattr(modules, key, value)
    modules._check_streaming_parameters()
    featurizer = model.preprocessor.featurizer
    featurizer.dither = 0.0
    featurizer.pad_to = 0
    if (modules.n_spk, modules.fc_d_model, modules.subsampling_factor) != (SPEAKERS, EMBEDDING_SIZE, SUBSAMPLING):
        raise RuntimeError("the checkpoint does not have 4 speakers, 512-dimensional embeddings, and 8x subsampling")
    return model


def pack(
    spkcache: torch.Tensor,
    spkcache_lengths: torch.Tensor,
    fifo: torch.Tensor,
    fifo_lengths: torch.Tensor,
    chunk: torch.Tensor,
    chunk_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place the valid rows of the cache, FIFO, and chunk one after another, then zeros.

    This is NeMo's `SortformerModules.concat_and_pad` for a batch of one with fixed shapes:
    output row `p` reads row `p` of the cache, row `p - cache` of the FIFO, or row
    `p - cache - fifo` of the chunk, so tracing produces a gather instead of dynamic slices.
    """
    cache_rows, fifo_rows = spkcache.shape[1], fifo.shape[1]
    stacked = torch.cat([spkcache, fifo, chunk], dim=1)
    rows = stacked.shape[1]
    position = torch.arange(rows, dtype=torch.int64)
    cache_end = spkcache_lengths.to(torch.int64).reshape(())
    fifo_end = cache_end + fifo_lengths.to(torch.int64).reshape(())
    total = fifo_end + chunk_lengths.to(torch.int64).reshape(())
    source = torch.where(
        position < cache_end,
        position,
        torch.where(
            position < fifo_end, position - cache_end + cache_rows, position - fifo_end + cache_rows + fifo_rows
        ),
    ).clamp(0, rows - 1)
    index = source.reshape(1, rows, 1).expand(stacked.shape[0], rows, stacked.shape[2])
    packed = torch.gather(stacked, 1, index) * (position < total).to(stacked.dtype).reshape(1, rows, 1)
    return packed, total.reshape(1)


class PreEncoder(nn.Module):
    """The float32 stage: chunk pre-encoder and packing."""

    def __init__(self, model):
        super().__init__()
        self.pre_encode = model.encoder.pre_encode

    def forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        embeddings, lengths = self.pre_encode(x=chunk, lengths=chunk_lengths)
        lengths = lengths.to(torch.int64)
        packed, total = pack(spkcache, spkcache_lengths, fifo, fifo_lengths, embeddings, lengths)
        return packed, total, embeddings, lengths


class Head(nn.Module):
    """The float16 stage: Fast Conformer encoder, Transformer, and speaker sigmoids."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pre_encoder_embs, pre_encoder_lengths):
        embeddings, lengths = self.model.frontend_encoder(
            processed_signal=pre_encoder_embs, processed_signal_length=pre_encoder_lengths, bypass_pre_encode=True
        )
        return self.model.forward_infer(embeddings, lengths)
