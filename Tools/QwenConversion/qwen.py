# SPDX-License-Identifier: Apache-2.0
"""Shared definitions for converting and verifying Qwen3-ASR Core ML models.

The wrappers reimplement the forward passes of transformers' `Qwen3ASRForConditionalGeneration`
(Apache License, Version 2.0) with the checkpoint's own modules, in a form that coremltools can
trace: one fixed audio encoder window, and a text decoder with a stateful KV cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOOL_DIR = Path(__file__).resolve().parent
# QWEN_BUILD_DIR moves checkpoints, outputs, fixtures, and reports to another directory.
BUILD_DIR = Path(os.environ.get("QWEN_BUILD_DIR", TOOL_DIR / "build")).resolve()
DEFAULT_CACHE_DIR = BUILD_DIR / "cache"
REPORTS_DIR = BUILD_DIR / "reports"
FIXTURES_DIR = BUILD_DIR / "fixtures"

SAMPLE_RATE = 16_000
MEL_BINS = 128
MEL_HOP_SAMPLES = 160
# The encoder convolves each 100-frame (1 s) chunk separately; three stride-2 convolutions
# (kernel 3, padding 1) map L frames to ceil(L / 8) tokens, so a whole chunk gives 13.
CHUNK_FRAMES = 100
TOKENS_PER_CHUNK = 13
# Self-attention covers 8 chunks (`n_window_infer` 800 frames, 104 tokens) at a time,
# so 8-second windows of audio are encoded independently.
WINDOW_CHUNKS = 8
WINDOW_FRAMES = WINDOW_CHUNKS * CHUNK_FRAMES
WINDOW_TOKENS = WINDOW_CHUNKS * TOKENS_PER_CHUNK

ENCODER_FILE = "AudioEncoder.mlmodelc"
DECODER_FILE = "TextDecoder.mlmodelc"
VOCAB_FILE = "vocab.json"
CONFIG_FILE = "qwen3_asr_config.json"
CONVERSION_FILE = "conversion.json"
MODEL_CARD_FILE = "README.md"
LICENSE_FILE = "LICENSE"
NOTICE_FILE = "NOTICE"
# The Apache License text that a package ships, as the Qwen3-ASR repository distributes it:
# the License with its appendix filled in as "Copyright 2026 Alibaba Cloud".
LICENSE_TEXT = TOOL_DIR / "LICENSE-Qwen3-ASR"
LICENSE_TEXT_SOURCE = "https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/LICENSE"
LICENSE_TEXT_SHA256 = "a44a6081c73ad75f0255bb2bb5cab74ef1829565a895a24e53a4f11290ab7655"
EVALUATION_DIR = TOOL_DIR / "evaluation"


@dataclass(frozen=True)
class Variant:
    name: str
    repo: str
    revision: str
    title: str
    upload_repo: str


VARIANTS = {
    "0.6b": Variant(
        name="0.6b",
        repo="Qwen/Qwen3-ASR-0.6B-hf",
        revision="7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
        title="Qwen3-ASR 0.6B",
        upload_repo="mattt/qwen3-asr-0.6b-coreml",
    ),
    "1.7b": Variant(
        name="1.7b",
        repo="Qwen/Qwen3-ASR-1.7B-hf",
        revision="bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        title="Qwen3-ASR 1.7B",
        upload_repo="mattt/qwen3-asr-1.7b-coreml",
    ),
}

LICENSE = "Apache-2.0"
LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"


def attribution(variant: Variant) -> str:
    return (
        f"{variant.title} by the Qwen team, Alibaba Cloud (https://huggingface.co/{variant.repo}), "
        f"licensed under the Apache License, Version 2.0 ({LICENSE_URL}). "
        "Converted to Core ML for AnySpeechRecognizer: the model is split into an audio encoder "
        "and a text decoder with a KV cache, weights are stored in float16, "
        "and weights of large layers are quantized to int8."
    )


# The forced-language prefill names that the model was trained with ("language <NAME><asr_text>").
LANGUAGES = {
    "ar": "Arabic", "yue": "Cantonese", "zh": "Chinese", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "fil": "Filipino", "fi": "Finnish", "fr": "French",
    "de": "German", "el": "Greek", "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "mk": "Macedonian", "ms": "Malay",
    "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "es": "Spanish", "sv": "Swedish", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
}  # fmt: skip


def snapshot(variant: Variant, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(variant.repo, revision=variant.revision, cache_dir=str(cache_dir)))


def load_reference(path: Path, dtype: torch.dtype = torch.float32):
    """The transformers model, in float32 unless `dtype` selects another type, and its processor."""
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(path))
    model = Qwen3ASRForConditionalGeneration.from_pretrained(str(path), dtype=dtype)
    model.eval()
    return model, processor


# Encoder.


def window_token_frames() -> torch.Tensor:
    """The first mel frame that each token of a window reads, in window order.

    Token t of chunk c is valid when its first frame, 100 c + 8 t, is before the valid length:
    the chunk then has more than 8 t valid frames, so ceil(frames / 8) > t.
    """
    chunks = torch.arange(WINDOW_CHUNKS).unsqueeze(1) * CHUNK_FRAMES
    tokens = torch.arange(TOKENS_PER_CHUNK).unsqueeze(0) * 8
    return (chunks + tokens).reshape(-1).to(torch.int32)


class EncoderWindow(nn.Module):
    """The audio encoder and projector for one 8-second window of mel features.

    Input `mel` is [1, 128, 800] and `mel_length` is the number of valid frames (1 to 800).
    Frames after the valid length must be zero, as the feature extractor pads them.
    Output `audio_embeddings` is [1, 104, D], where D is the decoder's hidden size; tokens after
    `ceil(mel_length / 100) - 1` whole chunks and the partial chunk's tokens are not valid.
    """

    def __init__(self, model):
        super().__init__()
        tower = model.model.audio_tower
        self.conv2d1, self.conv2d2, self.conv2d3 = tower.conv2d1, tower.conv2d2, tower.conv2d3
        self.conv_out = tower.conv_out
        self.layers = tower.layers
        self.ln_post = tower.ln_post
        self.projector = model.model.multi_modal_projector
        self.heads = tower.config.encoder_attention_heads
        self.register_buffer("positions", tower.positional_embedding.positional_embedding[:TOKENS_PER_CHUNK].clone())
        self.register_buffer("token_frames", window_token_frames())

    def forward(self, mel: torch.Tensor, mel_length: torch.Tensor) -> torch.Tensor:
        chunks = mel.reshape(1, MEL_BINS, WINDOW_CHUNKS, CHUNK_FRAMES).permute(2, 0, 1, 3)
        x = F.gelu(self.conv2d1(chunks))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        chunk_count, channels, bins, steps = x.shape
        x = x.permute(0, 3, 1, 2).reshape(chunk_count, steps, channels * bins)
        x = self.conv_out(x) + self.positions.to(x.dtype)
        x = x.reshape(1, WINDOW_TOKENS, -1)
        valid = self.token_frames < mel_length
        mask = torch.where(valid, 0.0, -1e4).to(x.dtype).reshape(1, 1, 1, WINDOW_TOKENS)
        for layer in self.layers:
            x = x + self.attention(layer.self_attn, layer.self_attn_layer_norm(x), mask)
            h = layer.final_layer_norm(x)
            x = x + layer.fc2(F.gelu(layer.fc1(h)))
        x = self.ln_post(x)
        return self.projector(x)

    def attention(self, attn, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        head = width // self.heads

        def split(t):
            return t.reshape(batch, length, self.heads, head).transpose(1, 2)

        q, k, v = split(attn.q_proj(x)), split(attn.k_proj(x)), split(attn.v_proj(x))
        scores = torch.matmul(q, k.transpose(2, 3)) * (head**-0.5) + mask
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, length, width)
        return attn.out_proj(out)


# Decoder.


class RMSNorm(nn.Module):
    """Qwen3's RMSNorm, scaled by each row's maximum magnitude first.

    Squaring hidden states of a few hundred overflows float16; after the scaling every square
    is at most 1 and the largest is 1, so the mean stays representable.
    """

    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.eps = norm.variance_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        y = x / scale
        variance = y.pow(2).mean(-1, keepdim=True)
        return self.weight * (y * torch.rsqrt(variance + self.eps / (scale * scale)))


class TextDecoder(nn.Module):
    """The Qwen3 text decoder for a fixed block of tokens, with a KV cache in Core ML states.

    Inputs:
    - `input_ids` [1, T]. Positions whose ID is the audio placeholder take their embedding
      from `audio_embeddings` [1, T, D] instead of the token embedding table.
    - `position_ids` [T]: the cache position of each token. The block writes its keys and values
      there, so positions after the real tokens of a partial block are overwritten later.
    - `causal_mask` [1, 1, T, C]: an additive mask over all C cache positions, 0 where a token may
      attend and a large negative number elsewhere. Every row must allow at least one position.

    Outputs: `token_ids` [1, T], the greedy next token after each input position, and
    `token_log_probs` [1, T], its log probability.

    Every shape is fixed. On macOS 27.0.1, Core ML gave wrong results on the GPU for a stateful
    decoder with flexible token and cache lengths.
    """

    def __init__(self, model, context: int, block: int = 1):
        super().__init__()
        lm = model.model.language_model
        config = lm.config
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = RMSNorm(lm.norm)
        self.input_norms = nn.ModuleList(RMSNorm(layer.input_layernorm) for layer in lm.layers)
        self.post_norms = nn.ModuleList(RMSNorm(layer.post_attention_layernorm) for layer in lm.layers)
        self.q_norms = nn.ModuleList(RMSNorm(layer.self_attn.q_norm) for layer in lm.layers)
        self.k_norms = nn.ModuleList(RMSNorm(layer.self_attn.k_norm) for layer in lm.layers)
        self.lm_head = model.lm_head
        self.audio_token_id = model.config.audio_token_id
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.context = context
        self.block = block
        # One key and one value state per layer.
        for index in range(len(lm.layers)):
            self.register_buffer(f"key_cache_{index}", torch.zeros(self.cache_shape, dtype=torch.float16))
            self.register_buffer(f"value_cache_{index}", torch.zeros(self.cache_shape, dtype=torch.float16))
        # Rotary tables for every cache position, computed in float32 by transformers' own
        # rotary embedding. The decoder gathers rows; float16 angles would be too coarse.
        positions = torch.arange(context).unsqueeze(0)
        # A new rotary embedding in float32, whatever the model's type: float16 inverse
        # frequencies would round the smallest of them.
        rotary = type(lm.rotary_emb)(config=config).float()
        cos, sin = rotary(torch.zeros(1, dtype=torch.float32), positions)
        self.register_buffer("rope_cos", cos[0].clone())
        self.register_buffer("rope_sin", sin[0].clone())

    @property
    def cache_shape(self) -> tuple[int, ...]:
        return (1, self.kv_heads, self.context, self.head_dim)

    def cache_names(self) -> list[str]:
        return [f"{kind}_cache_{index}" for index in range(len(self.layers)) for kind in ("key", "value")]

    def reset(self) -> None:
        for name in self.cache_names():
            getattr(self, name).zero_()

    def example_inputs(self) -> tuple[torch.Tensor, ...]:
        dtype = self.embed_tokens.weight.dtype
        return (
            torch.zeros(1, self.block, dtype=torch.long),
            torch.zeros(1, self.block, self.hidden_size, dtype=dtype),
            torch.arange(self.block, dtype=torch.long),
            torch.zeros(1, 1, self.block, self.context, dtype=dtype),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = input_ids.shape[1]
        text = self.embed_tokens(input_ids)
        is_audio = (input_ids == self.audio_token_id).unsqueeze(-1)
        x = torch.where(is_audio, audio_embeddings, text)
        cos = self.rope_cos[position_ids].to(x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin[position_ids].to(x.dtype).unsqueeze(0).unsqueeze(0)
        groups = self.heads // self.kv_heads
        rows = position_ids.reshape(1, 1, length, 1).expand(1, self.kv_heads, length, self.head_dim)
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = self.input_norms[index](x)
            q = self.q_norms[index](attn.q_proj(h).reshape(1, length, self.heads, self.head_dim))
            k = self.k_norms[index](attn.k_proj(h).reshape(1, length, self.kv_heads, self.head_dim))
            v = attn.v_proj(h).reshape(1, length, self.kv_heads, self.head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            key_cache = getattr(self, f"key_cache_{index}")
            value_cache = getattr(self, f"value_cache_{index}")
            # A functional scatter in float32: coremltools 9.0 has no conversion for `index_copy_`,
            # and it fails to convert an in-place `scatter_`, or a scatter of a float16 trace,
            # whose state and updates it reads with different types.
            key_cache[:] = key_cache.float().scatter(2, rows, k.float()).to(torch.float16)
            value_cache[:] = value_cache.float().scatter(2, rows, v.float()).to(torch.float16)
            # Grouped-query attention: each key head serves `groups` consecutive query heads.
            keys = key_cache.to(q.dtype).repeat_interleave(groups, dim=1)
            values = value_cache.to(q.dtype).repeat_interleave(groups, dim=1)
            scores = torch.matmul(q, keys.transpose(-1, -2)) * (self.head_dim**-0.5) + causal_mask
            weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(weights, values).transpose(1, 2).reshape(1, length, self.heads * self.head_dim)
            x = x + attn.o_proj(out)
            h = self.post_norms[index](x)
            mlp = layer.mlp
            x = x + mlp.down_proj(F.silu(mlp.gate_proj(h)) * mlp.up_proj(h))
        logits = self.lm_head(self.norm(x))
        log_probs = torch.log_softmax(logits, dim=-1)
        tokens = torch.argmax(logits, dim=-1).to(torch.int32)
        return tokens, log_probs.amax(dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class BlockFeeder:
    """Feeds tokens to a fixed-block decoder, as the Swift runtime does.

    `predict(input_ids, audio_embeddings, position_ids, causal_mask)` runs one block on NumPy
    arrays and returns the `token_ids` and `token_log_probs` arrays. The padding rows of a
    partial block repeat the end-of-text token at the following positions, which later blocks
    overwrite before any token attends to them.
    """

    def __init__(self, predict, block: int, context: int, hidden: int, padding_id: int):
        self.predict = predict
        self.block = block
        self.context = context
        self.hidden = hidden
        self.padding_id = padding_id
        self.position = 0

    def start(self) -> None:
        self.position = 0

    def feed(self, ids: list[int], audio: np.ndarray | None = None) -> tuple[int, float]:
        """Appends tokens, with audio embeddings for placeholder rows; returns the next token."""
        if self.position + len(ids) > self.context:
            raise ValueError("The prompt and output exceed the decoder's context.")
        result = (0, 0.0)
        for start in range(0, len(ids), self.block):
            chunk = ids[start : start + self.block]
            count = len(chunk)
            embeddings = np.zeros((1, self.block, self.hidden), dtype=np.float32)
            if audio is not None:
                embeddings[0, :count] = audio[start : start + count]
            positions = np.minimum(self.position + np.arange(self.block), self.context - 1).astype(np.int32)
            mask = np.where(np.arange(self.context)[None, :] > positions[:, None], -1e4, 0.0)
            tokens, log_probs = self.predict(
                np.array([chunk + [self.padding_id] * (self.block - count)], dtype=np.int32),
                embeddings,
                positions,
                mask.astype(np.float32).reshape(1, 1, self.block, self.context),
            )
            self.position += count
            result = (int(tokens.reshape(-1)[count - 1]), float(log_probs.reshape(-1)[count - 1]))
        return result


# Prompt.


def prompt_ids(tokenizer) -> dict[str, Any]:
    """Token IDs of the chat prompt around the audio placeholders, and the language prefills.

    The prompt is the chat template with an empty system message:
    `<|im_start|>system\\n<|im_end|>\\n<|im_start|>user\\n<|audio_start|>`, one `<|audio_pad|>`
    per audio embedding, then `<|audio_end|><|im_end|>\\n<|im_start|>assistant\\n`.
    A forced language appends `language <NAME><asr_text>`.
    """

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    return {
        "prefix": encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"),
        "suffix": encode("<|audio_end|><|im_end|>\n<|im_start|>assistant\n"),
        "languages": {
            code: {"name": name, "prefill": encode(f"language {name}<asr_text>")}
            for code, name in sorted(LANGUAGES.items())
        },
    }


# Records.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tree_files(root: Path) -> list[dict[str, Any]]:
    """Every file under a directory with its size and SHA-256 digest, in path order."""
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == CONVERSION_FILE or relative.startswith("."):
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": sha256(path)})
    return files


def environment() -> dict[str, str]:
    import coremltools
    import transformers

    xcode = subprocess.run(["xcodebuild", "-version"], capture_output=True, text=True, check=True).stdout
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "coremltools": coremltools.__version__,
        "numpy": np.__version__,
        "macos": platform.mac_ver()[0],
        "xcode": " ".join(xcode.split()),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
