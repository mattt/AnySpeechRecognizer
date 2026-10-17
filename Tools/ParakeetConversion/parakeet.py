# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from FluidInference/mobius
# (https://github.com/FluidInference/mobius, commit 26f5fa8059d5ab3d34449adce6ff6ed86734e522),
# Copyright Fluid Inference, licensed under the Apache License, Version 2.0:
#
#   - The component wrappers (EncoderWrapper, DecoderWrapper, JointWrapper,
#     JointDecisionSingleStep) come from
#     models/stt/parakeet-tdt-v3-0.6b/coreml/individual_components.py.
#   - hf_to_nemo_name and load_hf_weights come from
#     models/stt/parakeet-redux/coreml/redux_weights.py.
#
# The modifications are Copyright AnySpeechRecognizer contributors and are
# licensed under the same terms. See NOTICE in this directory.

"""Shared definitions for converting and verifying Parakeet TDT and RNN-T Core ML models."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

TOOL_DIR = Path(__file__).resolve().parent
# PARAKEET_BUILD_DIR moves checkpoints, outputs, fixtures, and reports to another directory.
BUILD_DIR = Path(os.environ.get("PARAKEET_BUILD_DIR", TOOL_DIR / "build")).resolve()
DEFAULT_CACHE_DIR = BUILD_DIR / "cache"

MOBIUS_REPOSITORY = "https://github.com/FluidInference/mobius"
MOBIUS_COMMIT = "26f5fa8059d5ab3d34449adce6ff6ed86734e522"

# Fixed 15-second window contract shared with FluidInference's artifacts.
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 240_000
MEL_BINS = 128
MEL_FRAMES = 1_501
ENCODER_DIM = 1_024
ENCODER_FRAMES = 188
SAMPLES_PER_ENCODER_FRAME = 1_280
DECODER_DIM = 640
DECODER_LAYERS = 2
VOCAB_SIZE = 8_192
BLANK_ID = 8_192
TDT_DURATIONS = (0, 1, 2, 3, 4)
TOP_K = 64
MAX_SYMBOLS_PER_FRAME = 10
WORD_BOUNDARY = "\u2581"  # SentencePiece word-boundary marker

ENCODER_FILE = "Encoder.mlmodelc"
DECODER_FILE = "Decoder.mlmodelc"
JOINT_FILE = "JointDecisionv3.mlmodelc"
VOCAB_FILE = "parakeet_vocab.json"
CONVERSION_FILE = "conversion.json"
MODEL_CARD_FILE = "README.md"
LICENSE_FILE = "LICENSE"
NOTICE_FILE = "NOTICE"

# Input and output names, data types, and shapes of each component.
CONTRACT: dict[str, dict[str, dict[str, tuple[str, tuple[int, ...]]]]] = {
    "encoder": {
        "inputs": {
            "mel": ("float32", (1, MEL_BINS, MEL_FRAMES)),
            "mel_length": ("int32", (1,)),
        },
        "outputs": {
            "encoder": ("float32", (1, ENCODER_DIM, ENCODER_FRAMES)),
            "encoder_length": ("int32", (1,)),
        },
    },
    "decoder": {
        "inputs": {
            "targets": ("int32", (1, 1)),
            "target_length": ("int32", (1,)),
            "h_in": ("float32", (DECODER_LAYERS, 1, DECODER_DIM)),
            "c_in": ("float32", (DECODER_LAYERS, 1, DECODER_DIM)),
        },
        "outputs": {
            "decoder": ("float32", (1, DECODER_DIM, 1)),
            "h_out": ("float32", (DECODER_LAYERS, 1, DECODER_DIM)),
            "c_out": ("float32", (DECODER_LAYERS, 1, DECODER_DIM)),
        },
    },
    "joint": {
        "inputs": {
            "encoder_step": ("float32", (1, ENCODER_DIM, 1)),
            "decoder_step": ("float32", (1, DECODER_DIM, 1)),
        },
        "outputs": {
            "token_id": ("int32", (1, 1, 1)),
            "token_prob": ("float32", (1, 1, 1)),
            "duration": ("int32", (1, 1, 1)),
            "top_k_ids": ("int32", (1, 1, 1, TOP_K)),
            "top_k_logits": ("float32", (1, 1, 1, TOP_K)),
        },
    },
}


MEL_HOP_SAMPLES = 160
# Encoder frames of attention context to each side in windows longer than 15 seconds:
# NeMo's local attention (`rel_pos_local_attn`, [256, 256]), which the model card recommends
# for long audio. The conversion computes it as full attention with keys more than 256 frames
# away masked, because coremltools cannot convert NeMo's sliding-chunk implementation.
LOCAL_ATTENTION_CONTEXT = (256, 256)
# Each of the encoder's three stride-2 subsampling convolutions (kernel 3, padding 1)
# maps L frames to floor((L - 1) / 2) + 1.
SUBSAMPLING_STAGES = 3


@dataclass(frozen=True)
class EncoderWindow:
    """A fixed encoder input window.

    The 15-second window is the published `Encoder.mlmodelc`. Other windows are named
    `Encoder<seconds>s.mlmodelc` so that they can share a repository with it.
    Windows longer than 15 seconds use local attention; see ``uses_local_attention``.
    """

    seconds: int

    @property
    def samples(self) -> int:
        return self.seconds * SAMPLE_RATE

    @property
    def mel_frames(self) -> int:
        return self.samples // MEL_HOP_SAMPLES + 1

    @property
    def encoder_frames(self) -> int:
        frames = self.mel_frames
        for _ in range(SUBSAMPLING_STAGES):
            frames = (frames - 1) // 2 + 1
        return frames

    @property
    def whole_frames(self) -> int:
        """Encoder frames whose audio fits completely in the window."""
        return self.samples // SAMPLES_PER_ENCODER_FRAME

    @property
    def is_default(self) -> bool:
        return self.samples == WINDOW_SAMPLES

    @property
    def uses_local_attention(self) -> bool:
        """Whether the window is longer than 15 seconds and limits attention to
        ``LOCAL_ATTENTION_CONTEXT``."""
        return self.samples > WINDOW_SAMPLES

    @property
    def file(self) -> str:
        return ENCODER_FILE if self.is_default else f"Encoder{self.seconds}s.mlmodelc"

    @property
    def package(self) -> str:
        return "Encoder.mlpackage" if self.is_default else f"Encoder{self.seconds}s.mlpackage"

    def contract(self) -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
        return {
            "inputs": {
                "mel": ("float32", (1, MEL_BINS, self.mel_frames)),
                "mel_length": ("int32", (1,)),
            },
            "outputs": {
                "encoder": ("float32", (1, ENCODER_DIM, self.encoder_frames)),
                "encoder_length": ("int32", (1,)),
            },
        }


DEFAULT_WINDOW = EncoderWindow(WINDOW_SAMPLES // SAMPLE_RATE)
assert (DEFAULT_WINDOW.mel_frames, DEFAULT_WINDOW.encoder_frames) == (MEL_FRAMES, ENCODER_FRAMES)


@dataclass(frozen=True)
class Source:
    """A file in a Hugging Face repository at an immutable revision."""

    repo: str
    revision: str
    files: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/tree/{self.revision}"


V3_LANGUAGES = "en, de, fr, es, it, pt, ru, uk, hr, sl, lv, lt, et, fi, sv, da, nl, pl, cs, sk, hu, ro, bg, el, mt"
CC_BY_4_0_URL = "https://creativecommons.org/licenses/by/4.0/"


@dataclass(frozen=True)
class Variant:
    name: str
    title: str
    # The NeMo checkpoint supplies the architecture, tokenizer, and preprocessor.
    base: Source
    # Optional replacement weights in Hugging Face transformers layout.
    weights: Source | None
    # FluidInference's published artifacts with the same contract, for comparison,
    # or None when no such artifacts exist.
    reference: Source | None
    reference_components: dict[str, str]
    upload_repo: str
    license: str
    attribution: str
    notes: tuple[str, ...] = field(default_factory=tuple)
    # SentencePiece pieces before the blank token, which has the next ID.
    vocab_size: int = VOCAB_SIZE
    # TDT duration classes in frames, or () for an RNN-T joint, which predicts no durations.
    durations: tuple[int, ...] = TDT_DURATIONS
    languages: str = V3_LANGUAGES
    # The Hugging Face model card's license metadata and the license's name and link.
    card_license: str = "license: cc-by-4.0"
    license_name: str = "CC BY 4.0"
    license_url: str = CC_BY_4_0_URL
    # For licenses that require them with every copy: the agreement, which the output
    # directory gets as LICENSE, the notice for its NOTICE file, and the modifications.
    license_text: Path | None = None
    required_notice: str | None = None
    modifications: str | None = None

    @property
    def blank_id(self) -> int:
        return self.vocab_size

    @property
    def is_tdt(self) -> bool:
        return bool(self.durations)

    @property
    def duration_frames(self) -> tuple[int, ...]:
        """Frames to advance for each value of the joint decision's `duration` output.

        An RNN-T joint decision always outputs class 0, which advances no frames, so greedy
        decoding emits tokens at a frame until a blank or the symbol limit, as RNN-T does.
        """
        return self.durations or (0,)


NVIDIA_V3 = Source(
    repo="nvidia/parakeet-tdt-0.6b-v3",
    revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    files=("parakeet-tdt-0.6b-v3.nemo",),
)

NVIDIA_OPEN_MODEL_LICENSE_URL = "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/"
# The agreement released October 24, 2025, which the Sortformer v2.1 conversion also publishes.
NVIDIA_OPEN_MODEL_LICENSE_TEXT = TOOL_DIR.parent / "SortformerConversion" / "v2.1" / "NVIDIA-Open-Model-License.txt"
NVIDIA_OPEN_MODEL_LICENSE_VERSION = "October 24, 2025"
NVIDIA_TRUSTWORTHY_AI_URL = "https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/"
# Section 3.1 of the agreement requires this sentence in a "Notice" text file with every copy.
NVIDIA_REQUIRED_NOTICE = "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
UNIFIED_REVISION = "fe53cd885760c96b6a5f51a0bfd362cb4584a98b"

VARIANTS: dict[str, Variant] = {
    "v3": Variant(
        name="v3",
        title="Parakeet TDT 0.6B v3",
        base=NVIDIA_V3,
        weights=None,
        reference=Source(
            repo="FluidInference/parakeet-tdt-0.6b-v3-coreml",
            revision="7dd20fe6b1797d35f5e3307e8b1732d9a178edfe",
            files=(
                "Encoder_v2.mlmodelc",
                "Decoder.mlmodelc",
                "JointDecisionv3.mlmodelc",
                "Preprocessor.mlmodelc",
                VOCAB_FILE,
            ),
        ),
        reference_components={
            "encoder": "Encoder_v2.mlmodelc",
            "decoder": "Decoder.mlmodelc",
            "joint": "JointDecisionv3.mlmodelc",
            "preprocessor": "Preprocessor.mlmodelc",
        },
        upload_repo="mattt/parakeet-tdt-0.6b-v3-coreml",
        license="CC-BY-4.0",
        attribution=(
            "Parakeet TDT 0.6B v3 by NVIDIA "
            "(https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), "
            "licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
            "Converted to Core ML for AnySpeechRecognizer: the model is split into "
            "encoder, prediction network, and joint decision components, "
            "weights are stored in float16, and encoder weights are quantized to int8."
        ),
    ),
    "ultra": Variant(
        name="ultra",
        title="Parakeet Ultra",
        base=NVIDIA_V3,
        weights=Source(
            repo="moondream/parakeet-ultra",
            revision="73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b",
            files=("model.safetensors", "tokenizer.json"),
        ),
        reference=Source(
            repo="FluidInference/parakeet-ultra-coreml",
            revision="95eaa59a39d4394f047a4dc5cce480388a60d1b6",
            files=(
                "Encoder.mlmodelc",
                "Decoder.mlmodelc",
                "JointDecisionv3.mlmodelc",
                "Preprocessor.mlmodelc",
                VOCAB_FILE,
            ),
        ),
        reference_components={
            "encoder": "Encoder.mlmodelc",
            "decoder": "Decoder.mlmodelc",
            "joint": "JointDecisionv3.mlmodelc",
            "preprocessor": "Preprocessor.mlmodelc",
        },
        upload_repo="mattt/parakeet-ultra-coreml",
        license="CC-BY-4.0",
        attribution=(
            "Parakeet Ultra by Moondream (https://huggingface.co/moondream/parakeet-ultra), "
            "a post-trained version of Parakeet TDT 0.6B v3 by NVIDIA "
            "(https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3); "
            "both licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). "
            "Converted to Core ML for AnySpeechRecognizer: the model is split into "
            "encoder, prediction network, and joint decision components, "
            "weights are stored in float16, and encoder weights are quantized to int8."
        ),
        notes=(
            (
                "The NVIDIA v3 checkpoint supplies the architecture, tokenizer, and "
                "preprocessor; Moondream's weights replace every model parameter."
            ),
            "Moondream's vad_head tensors are not exported.",
        ),
    ),
    "unified": Variant(
        name="unified",
        title="Parakeet Unified EN 0.6B",
        base=Source(
            repo="nvidia/parakeet-unified-en-0.6b",
            revision=UNIFIED_REVISION,
            files=("parakeet-unified-en-0.6b.nemo",),
        ),
        weights=None,
        # FluidInference's parakeet-unified-en-0.6b-coreml has other file names and interfaces.
        reference=None,
        reference_components={},
        upload_repo="mattt/parakeet-unified-en-0.6b-coreml",
        license="LicenseRef-NVIDIA-Open-Model-License",
        attribution=(
            "Parakeet Unified EN 0.6B by NVIDIA (https://huggingface.co/nvidia/parakeet-unified-en-0.6b). "
            f"{NVIDIA_REQUIRED_NOTICE} ({NVIDIA_OPEN_MODEL_LICENSE_URL}). "
            "Converted to Core ML with an int8 encoder for AnySpeechRecognizer "
            "(https://huggingface.co/mattt/parakeet-unified-en-0.6b-coreml)."
        ),
        notes=(
            "Recognizes English speech only, with punctuation and capitalization.",
            (
                "An RNN-T model: the joint decision's duration output is always 0, so greedy decoding "
                "stays on a frame after each token and advances after a blank or after 10 tokens."
            ),
            (
                "The encoders use full attention, the model's offline mode; "
                "NeMo's chunked attention masks for streaming are not exported."
            ),
        ),
        vocab_size=1_024,
        durations=(),
        languages="en",
        card_license=(
            f"license: other\nlicense_name: nvidia-open-model-license\nlicense_link: {NVIDIA_OPEN_MODEL_LICENSE_URL}"
        ),
        license_name="NVIDIA Open Model License Agreement",
        license_url=NVIDIA_OPEN_MODEL_LICENSE_URL,
        license_text=NVIDIA_OPEN_MODEL_LICENSE_TEXT,
        required_notice=NVIDIA_REQUIRED_NOTICE,
        modifications=(
            f"The NeMo checkpoint (revision {UNIFIED_REVISION}) was converted to Core ML "
            "and split into encoder, prediction network, and joint decision components. "
            "The encoders, for 15-second and 8-second windows, use full attention and have int8 weights "
            "(linear symmetric quantization, one scale per output channel); the other components have "
            "float16 weights, and all components compute in float16. "
            "The joint decision outputs the most likely token, its probability, the 64 highest token "
            "logits, and a duration that is always 0. "
            "The weights were not trained, fine-tuned, or pruned."
        ),
    ),
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


def download(source: Source, cache_dir: Path, patterns: list[str] | None = None) -> Path:
    """Download files from a pinned revision into a local directory and return it.

    Each file name in `source.files` is treated as a file or a directory prefix.
    """
    from huggingface_hub import snapshot_download

    local_dir = cache_dir / source.repo.replace("/", "--") / source.revision
    allow = patterns or [p for f in source.files for p in (f, f"{f}/**")]
    snapshot_download(
        repo_id=source.repo,
        revision=source.revision,
        allow_patterns=allow,
        local_dir=local_dir,
    )
    return local_dir


def verify_lfs_digest(source: Source, local_dir: Path, filename: str) -> dict[str, Any]:
    """Compare a downloaded file's SHA-256 digest with the digest Hugging Face records."""
    from huggingface_hub import HfApi

    path = local_dir / filename
    digest = sha256_file(path)
    expected = None
    for info in HfApi().get_paths_info(source.repo, [filename], revision=source.revision):
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            expected = lfs.sha256
    if expected is not None and expected != digest:
        raise RuntimeError(f"{source.repo}@{source.revision}/{filename}: SHA-256 {digest} != {expected}")
    return {
        "repo": source.repo,
        "revision": source.revision,
        "file": filename,
        "size": path.stat().st_size,
        "sha256": digest,
        "sha256_matches_hub": expected == digest if expected else None,
    }


# MARK: - Tool versions


def _command_output(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tool_versions() -> dict[str, Any]:
    from importlib import metadata

    packages = ["torch", "torchaudio", "nemo_toolkit", "coremltools", "numpy", "huggingface_hub", "safetensors"]
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


# MARK: - NeMo model loading


def hf_to_nemo_name(name: str) -> str:
    """Map Hugging Face transformers Parakeet parameter names to NeMo names.

    Derived from mobius `redux_weights.py`.
    """
    n = name
    n = re.sub(r"^encoder\.subsampling\.layers\.(\d+)\.", r"encoder.pre_encode.conv.\1.", n)
    n = n.replace("encoder.subsampling.linear.", "encoder.pre_encode.out.")
    n = n.replace(".self_attn.q_proj.", ".self_attn.linear_q.")
    n = n.replace(".self_attn.k_proj.", ".self_attn.linear_k.")
    n = n.replace(".self_attn.v_proj.", ".self_attn.linear_v.")
    n = n.replace(".self_attn.o_proj.", ".self_attn.linear_out.")
    n = n.replace(".self_attn.relative_k_proj.", ".self_attn.linear_pos.")
    n = n.replace(".self_attn.bias_u", ".self_attn.pos_bias_u")
    n = n.replace(".self_attn.bias_v", ".self_attn.pos_bias_v")
    n = n.replace(".conv.norm.", ".conv.batch_norm.")
    n = n.replace("decoder.embedding.weight", "decoder.prediction.embed.weight")
    n = n.replace("decoder.lstm.", "decoder.prediction.dec_rnn.lstm.")
    n = n.replace("decoder.decoder_projector.", "joint.pred.")
    n = n.replace("encoder_projector.", "joint.enc.")
    n = n.replace("joint.head.", "joint.joint_net.2.")
    return n


def load_hf_weights(model: torch.nn.Module, safetensors_path: Path) -> dict[str, Any]:
    """Replace a NeMo Parakeet model's parameters with full-precision Hugging Face weights.

    Derived from mobius `redux_weights.py`. Ternary (Redux) checkpoints are rejected.
    """
    from safetensors import safe_open

    with safe_open(str(safetensors_path), "np") as handle:
        raw = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118 (safe_open is not a dict)
    if any(key.endswith(".qweight") for key in raw):
        raise ValueError("ternary checkpoints are not supported")

    state: dict[str, torch.Tensor] = {}
    skipped = []
    source_dtypes: dict[str, int] = {}
    for key, value in raw.items():
        source_dtypes[str(value.dtype)] = source_dtypes.get(str(value.dtype), 0) + 1
        if key.startswith("vad_head."):
            skipped.append(key)
            continue
        tensor = value if value.dtype == np.int64 else value.astype(np.float32)
        state[hf_to_nemo_name(key)] = torch.from_numpy(np.ascontiguousarray(tensor))

    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    # The mel filterbank and window are not part of the transformers checkpoint.
    for key, value in original.items():
        if key.startswith("preprocessor."):
            state[key] = value
    missing = sorted(set(original) - set(state))
    unexpected = sorted(set(state) - set(original))
    if missing or unexpected:
        raise KeyError(f"state dict mismatch: missing={missing[:10]} unexpected={unexpected[:10]}")
    for key, value in original.items():
        # The transformers checkpoint stores batch-norm counters with shape (1,).
        if key.endswith("num_batches_tracked") and state[key].numel() == value.numel() == 1:
            state[key] = state[key].reshape(value.shape)
        if tuple(value.shape) != tuple(state[key].shape):
            raise ValueError(f"shape mismatch for {key}: {tuple(value.shape)} != {tuple(state[key].shape)}")
    model.load_state_dict(state, strict=True)

    changed = sum(
        1
        for key, value in original.items()
        if not key.startswith("preprocessor.") and not torch.equal(value.float(), state[key].float())
    )
    return {
        "tensors": len(raw),
        "source_dtypes": source_dtypes,
        "skipped": skipped,
        "parameters_changed_from_base": changed,
        "parameters_total": sum(1 for key in original if not key.startswith("preprocessor.")),
    }


def load_model(variant: Variant, cache_dir: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Load the variant's NeMo model in evaluation mode on the CPU.

    Returns the model, the source records, and weight-loading details.
    """
    import nemo.collections.asr as nemo_asr

    base_dir = download(variant.base, cache_dir)
    base_file = variant.base.files[0]
    sources = [
        {
            "role": "architecture, tokenizer, preprocessor, weights",
            **verify_lfs_digest(variant.base, base_dir, base_file),
        }
    ]
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(base_dir / base_file), map_location="cpu")
    model.eval()
    # NeMo 3.0.0's transcribe() reads validation_ds, which the unified checkpoint sets to null.
    if model.cfg.get("validation_ds") is None:
        from omegaconf import open_dict

        with open_dict(model.cfg):
            model.cfg.validation_ds = {}

    details: dict[str, Any] = {}
    if variant.weights is not None:
        sources[0]["role"] = "architecture, tokenizer, preprocessor"
        weights_dir = download(variant.weights, cache_dir)
        weights_file = variant.weights.files[0]
        sources.append({"role": "weights", **verify_lfs_digest(variant.weights, weights_dir, weights_file)})
        details = load_hf_weights(model, weights_dir / weights_file)
        model.eval()
        tokenizer_json = weights_dir / "tokenizer.json"
        if tokenizer_json.exists():
            sources.append(
                {"role": "tokenizer check", **verify_lfs_digest(variant.weights, weights_dir, "tokenizer.json")}
            )
            details["tokenizer_pieces_match_base"] = tokenizer_pieces_match(model, tokenizer_json)
    return model, sources, details


def tokenizer_pieces_match(model: Any, tokenizer_json: Path) -> bool:
    """Check that a transformers tokenizer.json has the NeMo model's SentencePiece pieces."""
    data = json.loads(tokenizer_json.read_text())
    vocab = data.get("model", {}).get("vocab")
    if isinstance(vocab, list):
        pieces = [entry[0] for entry in vocab]
    elif isinstance(vocab, dict):
        pieces = [piece for piece, _ in sorted(vocab.items(), key=lambda item: item[1])]
    else:
        return False
    model_pieces = vocabulary_pieces(model)
    return pieces[: len(model_pieces)] == model_pieces


def vocabulary_pieces(model: Any) -> list[str]:
    sp = model.tokenizer.tokenizer
    return [sp.id_to_piece(i) for i in range(int(model.tokenizer.vocab_size))]


def vocabulary_entries(model: Any) -> list[str]:
    """Return the vocabulary as FluidInference's `parakeet_vocab.json` stores it.

    FluidInference's file replaces the SentencePiece word-boundary marker
    U+2581 with an ASCII space; the runtime depends on that form.
    U+2581 only occurs as the first character of a piece.
    """
    return [piece.replace(WORD_BOUNDARY, " ") for piece in vocabulary_pieces(model)]


def vocabulary_json(model: Any) -> str:
    """Serialize the vocabulary in FluidInference's `parakeet_vocab.json` format."""
    vocab = {str(i): entry for i, entry in enumerate(vocabulary_entries(model))}
    return json.dumps(vocab, indent=2, ensure_ascii=False)


# MARK: - Component wrappers (derived from mobius individual_components.py)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, features: torch.Tensor, length: torch.Tensor):
        encoded, encoded_lengths = self.module(audio_signal=features, length=length.to(dtype=torch.long))
        return encoded, encoded_lengths


class DecoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, targets, target_lengths, h_in, c_in):
        decoder_output, _, new_state = self.module(
            targets=targets.to(dtype=torch.long),
            target_length=target_lengths.to(dtype=torch.long),
            states=[h_in, c_in],
        )
        return decoder_output, new_state[0], new_state[1]


class JointWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor) -> torch.Tensor:
        # [B, D, T] and [B, D, U] to [B, T, D] and [B, U, D].
        encoder_outputs = encoder_outputs.transpose(1, 2)
        decoder_outputs = decoder_outputs.transpose(1, 2)
        enc_proj = self.module.enc(encoder_outputs)
        dec_proj = self.module.pred(decoder_outputs)
        x = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)
        x = self.module.joint_net[0](x)  # ReLU
        x = self.module.joint_net[1](x)  # Dropout, inactive in evaluation mode
        return self.module.joint_net[2](x)  # [B, T, U, vocabulary + blank + durations]


class JointDecisionSingleStep(torch.nn.Module):
    """Joint network and greedy decision for one encoder frame and one decoder step.

    Outputs the argmax token over the vocabulary and blank, its softmax probability,
    the argmax duration class index, and the top-K token logits and indices.
    An RNN-T joint (`num_durations` 0) has no duration logits, and its duration is always 0.
    """

    def __init__(self, joint: JointWrapper, vocab_size: int, num_durations: int, top_k: int = TOP_K) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.num_durations = int(num_durations)
        self.top_k = int(top_k)

    def forward(self, encoder_step: torch.Tensor, decoder_step: torch.Tensor):
        logits = self.joint(encoder_step, decoder_step)  # [1, 1, 1, V + 1 + durations]
        token_logits = logits[..., : self.vocab_with_blank]
        token_id = torch.argmax(token_logits, dim=-1).to(dtype=torch.int32)
        token_probs_all = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(token_probs_all, dim=-1, index=token_id.long().unsqueeze(-1)).squeeze(-1)
        if self.num_durations:
            duration_logits = logits[..., -self.num_durations :]
            duration = torch.argmax(duration_logits, dim=-1).to(dtype=torch.int32)
        else:
            duration = torch.zeros_like(token_id)
        top_k_logits, top_k_ids = torch.topk(token_logits, k=min(self.top_k, token_logits.shape[-1]), dim=-1)
        return token_id, token_prob, duration, top_k_ids.to(dtype=torch.int32), top_k_logits


# MARK: - Features


def window_audio(audio: np.ndarray, window_samples: int = WINDOW_SAMPLES) -> tuple[np.ndarray, int]:
    """Zero-pad or trim audio to one window (15 s by default); return it and the valid sample count."""
    valid = min(len(audio), window_samples)
    window = np.zeros(window_samples, dtype=np.float32)
    window[:valid] = audio[:valid]
    return window, valid


def nemo_mel(
    model: Any, window: np.ndarray, valid_samples: int, mel_frames: int = MEL_FRAMES
) -> tuple[np.ndarray, np.ndarray]:
    """Compute NeMo's normalized log-mel features for one window (15 s by default)."""
    if model.preprocessor.training:
        raise RuntimeError("the preprocessor is in training mode, which adds dither")
    with torch.inference_mode():
        mel, mel_length = model.preprocessor(
            input_signal=torch.from_numpy(window)[None],
            length=torch.tensor([valid_samples], dtype=torch.long),
        )
    mel_np = mel.numpy().astype(np.float32)
    if mel_np.shape != (1, MEL_BINS, mel_frames):
        raise ValueError(f"unexpected mel shape {mel_np.shape}")
    return mel_np, mel_length.numpy().astype(np.int32)


def load_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {rate} Hz; expected {SAMPLE_RATE} Hz")
    return np.ascontiguousarray(data[:, 0])


def load_audio_resampled(path: Path) -> np.ndarray:
    """Load the first channel of an audio file and resample it to 16 kHz with a polyphase filter."""
    import math

    import soundfile as sf
    from scipy.signal import resample_poly

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data[:, 0]
    if rate != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, int(rate) // divisor).astype(np.float32)
    return np.ascontiguousarray(audio)
