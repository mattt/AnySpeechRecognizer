# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from FluidInference/mobius
# (https://github.com/FluidInference/mobius, commit 26f5fa8059d5ab3d34449adce6ff6ed86734e522),
# Copyright Fluid Inference, licensed under the Apache License, Version 2.0:
#
#   - fixed_concat_and_pad, PreEncoderWrapper, and HeadWrapper come from
#     models/speaker-diarization/sortformer-streaming/conversion/coreml_wrappers.py
#     (fixed_concat_and_pad, PreEncoderWrapper.forward_concat, and SortformerHeadWrapper).
#
# The wrappers call NVIDIA NeMo's SortformerEncLabelModel (https://github.com/NVIDIA/NeMo,
# Apache License, Version 2.0) and follow its forward_for_export.
# The modifications are Copyright AnySpeechRecognizer contributors and are
# licensed under the same terms. See NOTICE in this directory.

"""Shared definitions for converting and verifying the Streaming Sortformer Core ML model."""

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
DEFAULT_OUTPUT_DIR = BUILD_DIR / "sortformer"
DEFAULT_WORK_DIR = BUILD_DIR / "work"
REPORTS_DIR = BUILD_DIR / "reports"

MOBIUS_REPOSITORY = "https://github.com/FluidInference/mobius"
MOBIUS_COMMIT = "26f5fa8059d5ab3d34449adce6ff6ed86734e522"
MOBIUS_PATH = "models/speaker-diarization/sortformer-streaming/conversion"


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    file: str
    size: int | None = None
    sha256: str | None = None


# NVIDIA's checkpoint. The .nemo file last changed in commit 59620264e008f2b06a0e969688e0af3e8705478b;
# later commits change only the model card and add a GGUF file.
CHECKPOINT = Source(
    repo="nvidia/diar_streaming_sortformer_4spk-v2",
    revision="84edd514b8ef68004c10086918cd62f2148cbd59",
    file="diar_streaming_sortformer_4spk-v2.nemo",
    size=471_367_680,
    sha256="b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329",
)

# The Core ML model that the runtime's built-in manifest pins today; used for comparison only.
REFERENCE = Source(
    repo="FluidInference/diar-streaming-sortformer-coreml",
    revision="ae9a27ab45dc0aa3abede7d2d6bad2b7a69aa6d1",
    file="v3/fp16/SortformerNvidiaLow_v2.mlmodelc",
    size=246_275_129,
)

UPLOAD_REPO = "mattt/diar-streaming-sortformer-4spk-v2-coreml"
MODEL_FILE = "SortformerLowLatency.mlmodelc"
PACKAGE_FILE = "SortformerLowLatency.mlpackage"
CONVERSION_FILE = "conversion.json"
MODEL_CARD_FILE = "README.md"

TITLE = "Streaming Sortformer 4spk v2"
LICENSE = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = (
    "Streaming Sortformer 4spk v2 by NVIDIA "
    "(https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2), "
    "licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
    "Converted to Core ML for AnySpeechRecognizer: the pre-encoder, FastConformer encoder, "
    "and Transformer are exported for one streaming step with a fixed speaker cache and FIFO, "
    "and the encoder and Transformer weights are stored in float16."
)

# NVIDIA's low-latency streaming configuration (1.04 s algorithmic latency), as in
# FluidInference's SortformerNvidiaLow variants and the runtime's SortformerConfiguration.
STREAMING = {
    "chunk_len": 6,
    "chunk_left_context": 1,
    "chunk_right_context": 7,
    "fifo_len": 188,
    "spkcache_len": 188,
    "spkcache_update_period": 144,
}
SAMPLE_RATE = 16_000
SUBSAMPLING_FACTOR = 8
FRAME_SECONDS = 0.08
MEL_BINS = 128
EMBEDDING_SIZE = 512
SPEAKER_COUNT = 4
CHUNK_EMBEDDING_ROWS = STREAMING["chunk_left_context"] + STREAMING["chunk_len"] + STREAMING["chunk_right_context"]
CHUNK_MEL_FRAMES = CHUNK_EMBEDDING_ROWS * SUBSAMPLING_FACTOR
PREDICTION_ROWS = STREAMING["spkcache_len"] + STREAMING["fifo_len"] + CHUNK_EMBEDDING_ROWS

# Creator-defined metadata. CoreMLSortformerEngine reads the streaming keys and subsampling_factor.
# frame_duration and mel_feature_frames (the chunk's mel frames without context) keep the keys of
# FluidInference's model.
STREAMING_METADATA = {
    **{key: str(value) for key, value in STREAMING.items()},
    "subsampling_factor": str(SUBSAMPLING_FACTOR),
    "frame_duration": str(FRAME_SECONDS),
    "mel_feature_frames": str(STREAMING["chunk_len"] * SUBSAMPLING_FACTOR),
}

# The tensor interface of the compiled pipeline: (dtype, shape).
# It matches FluidInference's v3/fp16/SortformerNvidiaLow_v2.mlmodelc and the shapes that
# CoreMLSortformerEngine checks when it loads a model.
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
        "speaker_preds": ("float32", (1, PREDICTION_ROWS, SPEAKER_COUNT)),
        "chunk_pre_encoder_embs_out": ("float16", (1, CHUNK_EMBEDDING_ROWS, EMBEDDING_SIZE)),
        "chunk_pre_encoder_lengths_out": ("int32", (1,)),
    },
}

INPUT_DESCRIPTIONS = {
    "chunk": "Log-mel features of the chunk with its left and right context, time-major, 128 bins",
    "chunk_lengths": "Number of valid mel frames in chunk",
    "spkcache": "Arrival-order speaker cache: pre-encoder embeddings of earlier frames",
    "spkcache_lengths": "Number of valid frames in spkcache",
    "fifo": "FIFO queue: pre-encoder embeddings of the most recent frames",
    "fifo_lengths": "Number of valid frames in fifo",
}
OUTPUT_DESCRIPTIONS = {
    "speaker_preds": "Speaker activity probabilities for the valid speaker cache, FIFO, and chunk frames, in that order",
    "chunk_pre_encoder_embs_out": "Pre-encoder embeddings of the chunk, for the speaker cache and FIFO update",
    "chunk_pre_encoder_lengths_out": "Number of valid frames in chunk_pre_encoder_embs_out",
}


# MARK: - Files and digests


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


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def download_checkpoint(cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Download the pinned .nemo checkpoint and check its size and SHA-256 digest.

    The digest must equal both the pinned digest and the digest Hugging Face records for the file.
    """
    from huggingface_hub import HfApi, hf_hub_download

    local_dir = cache_dir / CHECKPOINT.repo.replace("/", "--") / CHECKPOINT.revision
    path = Path(
        hf_hub_download(
            repo_id=CHECKPOINT.repo, filename=CHECKPOINT.file, revision=CHECKPOINT.revision, local_dir=local_dir
        )
    )
    digest = sha256_file(path)
    hub_digest = None
    for info in HfApi().get_paths_info(CHECKPOINT.repo, [CHECKPOINT.file], revision=CHECKPOINT.revision):
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            hub_digest = lfs.sha256
    if digest != CHECKPOINT.sha256 or path.stat().st_size != CHECKPOINT.size:
        raise RuntimeError(f"{path}: SHA-256 {digest}, {path.stat().st_size} bytes; expected {CHECKPOINT.sha256}")
    if hub_digest is not None and hub_digest != digest:
        raise RuntimeError(f"{path}: Hugging Face records SHA-256 {hub_digest}")
    return path, {
        "repo": CHECKPOINT.repo,
        "revision": CHECKPOINT.revision,
        "file": CHECKPOINT.file,
        "size": path.stat().st_size,
        "sha256": digest,
        "sha256_matches_hub": hub_digest == digest if hub_digest else None,
    }


# MARK: - Tool versions


def _command_output(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
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


# MARK: - NeMo model


def load_model(checkpoint: Path) -> Any:
    """Restore the NeMo model on the CPU with the low-latency streaming configuration."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(str(checkpoint), map_location="cpu")
    model.eval()
    model.streaming_mode = True
    model.async_streaming = False
    modules = model.sortformer_modules
    for key, value in STREAMING.items():
        setattr(modules, key, value)
    featurizer = model.preprocessor.featurizer
    featurizer.dither = 0.0
    featurizer.pad_to = 0
    if int(modules.subsampling_factor) != SUBSAMPLING_FACTOR or int(modules.n_spk) != SPEAKER_COUNT:
        raise RuntimeError("unexpected subsampling factor or speaker count")
    if int(modules.fc_d_model) != EMBEDDING_SIZE or int(model.encoder._feat_in) != MEL_BINS:
        raise RuntimeError("unexpected embedding size or mel bins")
    return model


# MARK: - Export wrappers (from FluidInference mobius coreml_wrappers.py)


def fixed_concat_and_pad(
    embs: list[torch.Tensor], lengths: list[torch.Tensor], max_total_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the valid frames of [spkcache, fifo, chunk] at the start of a fixed-length sequence.

    The same result as NeMo's concat_and_pad, computed with one gather and arithmetic indices,
    so the Core ML graph has static shapes and no zero-length slices.
    """
    batch, _, dim = embs[0].shape
    size0, size1, size2 = embs[0].shape[1], embs[1].shape[1], embs[2].shape[1]
    total_input_size = size0 + size1 + size2
    full_concat = torch.cat(embs, dim=1)
    len0 = lengths[0].reshape(())
    len1 = lengths[1].reshape(())
    len2 = lengths[2].reshape(())
    total_length = len0 + len1 + len2
    out_pos = torch.arange(max_total_len, dtype=torch.long)
    # Output position p reads p in the cache, p + (size0 - len0) in the FIFO, and
    # p + (size0 - len0) + (size1 - len1) in the chunk.
    cumsum0 = len0
    cumsum1 = len0 + len1
    in_seg1_or_2 = (out_pos >= cumsum0).long()
    in_seg2 = (out_pos >= cumsum1).long()
    offset = in_seg1_or_2 * (size0 - len0) + in_seg2 * (size1 - len1)
    gather_idx = (out_pos + offset).clamp(0, total_input_size - 1)
    gather_idx = gather_idx.unsqueeze(0).unsqueeze(-1).expand(batch, max_total_len, dim)
    output = torch.gather(full_concat, dim=1, index=gather_idx)
    output = output * (out_pos < total_length).float().unsqueeze(0).unsqueeze(-1)
    return output, total_length


class PreEncoderWrapper(nn.Module):
    """Pre-encodes a chunk and packs it after the valid speaker cache and FIFO frames."""

    def __init__(self, model: Any):
        super().__init__()
        self.model = model
        modules = model.sortformer_modules
        chunk_length = modules.chunk_left_context + modules.chunk_len + modules.chunk_right_context
        self.pre_encoder_length = modules.spkcache_len + modules.fifo_len + chunk_length

    def forward(self, *args):
        # mobius's signature; the trace's value names, and so the model.mil text, depend on it.
        return self.forward_concat(*args)

    def forward_concat(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.model.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)
        packed, packed_lengths = fixed_concat_and_pad(
            [spkcache, fifo, chunk_pre_encode_embs],
            [spkcache_lengths, fifo_lengths, chunk_pre_encode_lengths],
            self.pre_encoder_length,
        )
        return packed, packed_lengths, chunk_pre_encode_embs, chunk_pre_encode_lengths


class HeadWrapper(nn.Module):
    """Runs the FastConformer encoder, Transformer, and speaker sigmoids on the packed sequence.

    The chunk embeddings and lengths pass through identity operations, so the outputs are distinct
    tensors from the inputs. The macOS 26 BNNS graph compiler rejects a function whose input
    is also its output.
    """

    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(self, pre_encoder_embs, pre_encoder_lengths, chunk_pre_encoder_embs, chunk_pre_encoder_lengths):
        embs, lengths = self.model.frontend_encoder(
            processed_signal=pre_encoder_embs,
            processed_signal_length=pre_encoder_lengths,
            bypass_pre_encode=True,
        )
        preds = self.model.forward_infer(embs, lengths)
        return preds, chunk_pre_encoder_embs + 0.0, chunk_pre_encoder_lengths + 0


def step_reference(
    model: Any,
    chunk: torch.Tensor,
    chunk_lengths: torch.Tensor,
    spkcache: torch.Tensor,
    spkcache_lengths: torch.Tensor,
    fifo: torch.Tensor,
    fifo_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One streaming step with NeMo's float32 PyTorch modules and NeMo's own concat_and_pad."""
    with torch.inference_mode():
        return model.forward_for_export(chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths)
