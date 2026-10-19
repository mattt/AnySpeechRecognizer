# SPDX-License-Identifier: Apache-2.0

"""Shared definitions for converting and verifying Nemotron 3.5 ASR Streaming Core ML models.

The encoder is NVIDIA's cache-aware FastConformer. Each streaming step encodes one chunk of
mel frames with the attention and convolution caches of the previous step and returns the
updated caches, so no audio is encoded twice. The step follows NeMo's
`CacheAwareStreamingAudioBuffer` with `pad_and_drop_preencoded=True`: every step, including
the first, receives the 9 mel frames before the chunk (zeros before the audio) and drops the
first 2 subsampled frames, so one fixed-shape model serves every step.
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

TOOL_DIR = Path(__file__).resolve().parent
# NEMOTRON_BUILD_DIR moves checkpoints, outputs, fixtures, and reports to another directory.
BUILD_DIR = Path(os.environ.get("NEMOTRON_BUILD_DIR", TOOL_DIR / "build")).resolve()
DEFAULT_CACHE_DIR = BUILD_DIR / "cache"

REPOSITORY = "nvidia/nemotron-3.5-asr-streaming-0.6b"
REVISION = "ea30d66debe3740a08b573244286791d423d6b3e"
CHECKPOINT = "nemotron-3.5-asr-streaming-0.6b.nemo"
TITLE = "Nemotron 3.5 ASR Streaming 0.6B"
LICENSE = "OpenMDW-1.1"
LICENSE_URL = "https://openmdw.ai/license/1-1/"
# The repository that would publish the package; nothing in this directory uploads.
UPLOAD_REPO = "mattt/nemotron-3.5-asr-streaming-0.6b-coreml"
# The canonical license text that a package ships as LICENSE.
LICENSE_TEXT = TOOL_DIR / "LICENSE-OpenMDW-1.1"
LICENSE_TEXT_SOURCE = (
    "https://github.com/OpenMDW/OpenMDW/blob/9067ba54b2f84bc9e15a05734f570ec88e9f445d/1.1/LICENSE.OpenMDW-1.1"
)
LICENSE_TEXT_SHA256 = "2ab44b68365473c112f5092211a38f231cb23e50de68b75a13369adbd76a74df"
ATTRIBUTION = (
    f"{TITLE} by NVIDIA (https://huggingface.co/{REPOSITORY}), "
    f"licensed under the OpenMDW License 1.1 ({LICENSE_URL}). "
    "Converted to Core ML for AnySpeechRecognizer: the model is split into a cache-aware "
    "encoder step with the language prompt projection, a prediction network, and a joint "
    "decision; weights are stored in float16, and encoder weights are quantized to int8."
)

SAMPLE_RATE = 16_000
MEL_HOP_SAMPLES = 160
MEL_BINS = 128
ENCODER_DIM = 1_024
ENCODER_LAYERS = 24
SUBSAMPLING = 8
SAMPLES_PER_ENCODER_FRAME = SUBSAMPLING * MEL_HOP_SAMPLES
# Encoder frames of left attention context, the attention cache of each layer.
ATTENTION_CACHE_FRAMES = 56
# Frames of the causal depthwise convolution cache (kernel 9).
CONVOLUTION_CACHE_FRAMES = 8
# Mel frames before each chunk that the causal subsampling needs, and the subsampled frames
# that they produce, which each step drops.
PRE_ENCODE_CACHE_FRAMES = 9
DROPPED_PRE_ENCODED_FRAMES = 2
DECODER_DIM = 640
DECODER_LAYERS = 2
VOCAB_SIZE = 13_087
BLANK_ID = 13_087
NUM_PROMPTS = 128
AUTO_PROMPT = "auto"
MAX_SYMBOLS_PER_FRAME = 10
WORD_BOUNDARY = "▁"

ENCODER_FILE = "Encoder.mlmodelc"
DECODER_FILE = "Decoder.mlmodelc"
JOINT_FILE = "JointDecision.mlmodelc"
VOCAB_FILE = "vocab.json"
PROMPTS_FILE = "prompts.json"
CONVERSION_FILE = "conversion.json"
MODEL_CARD_FILE = "README.md"
LICENSE_FILE = "LICENSE"
NOTICE_FILE = "NOTICE"

# The locales of the upstream model card, by tier. Transcription-ready and broad-coverage
# locales transcribe without fine-tuning; adaptation-ready locales need it.
TRANSCRIPTION_READY = {
    "English": ["en-US", "en-GB"],
    "Spanish": ["es-US", "es-ES"],
    "French": ["fr-FR", "fr-CA"],
    "Italian": ["it-IT"],
    "Portuguese": ["pt-BR", "pt-PT"],
    "Dutch": ["nl-NL"],
    "German": ["de-DE"],
    "Turkish": ["tr-TR"],
    "Russian": ["ru-RU"],
    "Arabic": ["ar-AR"],
    "Hindi": ["hi-IN"],
    "Japanese": ["ja-JP"],
    "Korean": ["ko-KR"],
    "Vietnamese": ["vi-VN"],
    "Ukrainian": ["uk-UA"],
}
BROAD_COVERAGE = {
    "Polish": ["pl-PL"],
    "Swedish": ["sv-SE"],
    "Czech": ["cs-CZ"],
    "Norwegian Bokmål": ["nb-NO"],
    "Danish": ["da-DK"],
    "Bulgarian": ["bg-BG"],
    "Finnish": ["fi-FI"],
    "Croatian": ["hr-HR"],
    "Slovak": ["sk-SK"],
    "Mandarin": ["zh-CN"],
    "Hungarian": ["hu-HU"],
    "Romanian": ["ro-RO"],
    "Estonian": ["et-EE"],
}
ADAPTATION_READY = {
    "Greek": ["el-GR"],
    "Lithuanian": ["lt-LT"],
    "Latvian": ["lv-LV"],
    "Maltese": ["mt-MT"],
    "Slovenian": ["sl-SI"],
    "Hebrew": ["he-IL"],
    "Thai": ["th-TH"],
    "Norwegian Nynorsk": ["nn-NO"],
}


@dataclass(frozen=True)
class Chunk:
    """A streaming chunk size: `att_context_size` [56, right] in NeMo.

    A chunk has `right + 1` encoder frames of 80 ms. Frames attend to every frame of their
    chunk and to the 56 frames before it, so the chunk size is also the latency of the encoder.
    """

    right: int

    @property
    def frames(self) -> int:
        return self.right + 1

    @property
    def milliseconds(self) -> int:
        return self.frames * 80

    @property
    def mel_frames(self) -> int:
        """New mel frames per step."""
        return self.frames * SUBSAMPLING

    @property
    def input_frames(self) -> int:
        """Mel frames of the encoder input: the pre-encode cache and the chunk."""
        return PRE_ENCODE_CACHE_FRAMES + self.mel_frames

    @property
    def function(self) -> str:
        """The function name in the multifunction encoder."""
        return f"chunk_{self.milliseconds}ms"

    @property
    def package(self) -> str:
        return f"Encoder{self.milliseconds}ms.mlpackage"

    def contract(self, cache_dtype: str = "float32") -> dict[str, dict[str, tuple[str, tuple[int, ...]]]]:
        """The tensor interface, with float32 or float16 caches."""
        channel = (cache_dtype, (ENCODER_LAYERS, 1, ATTENTION_CACHE_FRAMES, ENCODER_DIM))
        time_cache = (cache_dtype, (ENCODER_LAYERS, 1, ENCODER_DIM, CONVOLUTION_CACHE_FRAMES))
        return {
            "inputs": {
                "mel": ("float32", (1, MEL_BINS, self.input_frames)),
                "mel_length": ("int32", (1,)),
                "cache_channel": channel,
                "cache_time": time_cache,
                "cache_length": ("int32", (1,)),
                "prompt": ("float32", (1, NUM_PROMPTS)),
            },
            "outputs": {
                "encoder": ("float32", (1, ENCODER_DIM, self.frames)),
                "encoder_length": ("int32", (1,)),
                "cache_channel_out": channel,
                "cache_time_out": time_cache,
                "cache_length_out": ("int32", (1,)),
            },
        }


# The chunk sizes on the model card. NeMo trained [56, 0], [56, 3], [56, 6], and [56, 13];
# the card also reports [56, 1].
CHUNKS = {chunk.milliseconds: chunk for chunk in (Chunk(0), Chunk(1), Chunk(3), Chunk(6), Chunk(13))}

CONTRACT: dict[str, dict[str, dict[str, tuple[str, tuple[int, ...]]]]] = {
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
        },
    },
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
    """Download the pinned checkpoint and check its digest against the one Hugging Face records."""
    from huggingface_hub import HfApi, hf_hub_download

    local_dir = cache_dir / REPOSITORY.replace("/", "--") / REVISION
    path = Path(hf_hub_download(REPOSITORY, CHECKPOINT, revision=REVISION, local_dir=local_dir))
    digest = sha256_file(path)
    expected = None
    for info in HfApi().get_paths_info(REPOSITORY, [CHECKPOINT], revision=REVISION):
        lfs = getattr(info, "lfs", None)
        if lfs is not None:
            expected = lfs.sha256
    if expected is not None and expected != digest:
        raise RuntimeError(f"{REPOSITORY}@{REVISION}/{CHECKPOINT}: SHA-256 {digest} != {expected}")
    record = {
        "repo": REPOSITORY,
        "revision": REVISION,
        "file": CHECKPOINT,
        "size": path.stat().st_size,
        "sha256": digest,
        "sha256_matches_hub": expected == digest if expected else None,
    }
    return path, record


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


def load_model(cache_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load the NeMo model in evaluation mode on the CPU, with the checkpoint's source record."""
    from nemo.collections.asr.models import ASRModel

    path, record = download_checkpoint(cache_dir)
    model = ASRModel.restore_from(str(path), map_location="cpu")
    model.eval()
    if not getattr(model, "concat", False):
        raise RuntimeError("the checkpoint has no language prompt projection")
    checks = {
        "vocabulary": int(model.tokenizer.vocab_size) == VOCAB_SIZE,
        "blank": int(model.decoder.blank_idx) == BLANK_ID,
        "prompts": int(model.num_prompts) == NUM_PROMPTS,
        "layers": len(model.encoder.layers) == ENCODER_LAYERS,
        "decoder": (int(model.decoder.pred_hidden), int(model.decoder.pred_rnn_layers))
        == (DECODER_DIM, DECODER_LAYERS),
        "attention": model.encoder.att_context_style == "chunked_limited",
    }
    if not all(checks.values()):
        raise RuntimeError(f"unexpected model: {checks}")
    return model, record


def use_fast_depthwise_convolution() -> None:
    """Computes NeMo's depthwise causal convolutions with one batched product on the CPU.

    PyTorch 2.7 computes a `conv1d` with 1,024 groups on the CPU as one small convolution per
    group, so one encoder step takes more than a second. The replacement computes the same
    sums in another order. `verify.py --fast-reference` uses it; the conversion does not.
    """
    from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D

    original = CausalConv1D.forward
    if getattr(original, "fast_depthwise", False):
        return

    def forward(self, x, cache=None):
        if self.groups != self.in_channels or self.stride != (1,) or self.dilation != (1,):
            return original(self, x, cache=cache)
        x, cache = self.update_cache(x, cache=cache)
        windows = x.unfold(2, self.kernel_size[0], 1)
        out = (windows * self.weight[:, 0, :][None, :, None, :]).sum(-1)
        if self.bias is not None:
            out = out + self.bias[None, :, None]
        return out if cache is None else (out, cache)

    forward.fast_depthwise = True
    CausalConv1D.forward = forward


def prompt_dictionary(model: Any) -> dict[str, int]:
    return {str(key): int(value) for key, value in model.cfg.model_defaults.prompt_dictionary.items()}


def prompt_vector(index: int) -> torch.Tensor:
    vector = torch.zeros(1, NUM_PROMPTS)
    vector[0, index] = 1
    return vector


def set_chunk(model: Any, chunk: Chunk) -> None:
    """Select the attention context of a chunk size and NeMo's matching streaming settings."""
    model.encoder.set_default_att_context_size([ATTENTION_CACHE_FRAMES, chunk.right])
    config = model.encoder.streaming_cfg
    expected = (config.last_channel_cache_size, config.pre_encode_cache_size, config.drop_extra_pre_encoded)
    if expected != (ATTENTION_CACHE_FRAMES, [0, PRE_ENCODE_CACHE_FRAMES], DROPPED_PRE_ENCODED_FRAMES):
        raise RuntimeError(f"unexpected streaming configuration {config}")
    if config.valid_out_len != chunk.frames or config.chunk_size[1] != chunk.mel_frames:
        raise RuntimeError(f"unexpected streaming configuration {config} for {chunk}")


def vocabulary_pieces(model: Any) -> list[str]:
    sp = model.tokenizer.tokenizer
    return [sp.id_to_piece(i) for i in range(int(model.tokenizer.vocab_size))]


def vocabulary_json(model: Any) -> str:
    """Token ID to piece, with the word-boundary marker written as a space, as for Parakeet."""
    pieces = vocabulary_pieces(model)
    return json.dumps(
        {str(i): piece.replace(WORD_BOUNDARY, " ") for i, piece in enumerate(pieces)}, indent=2, ensure_ascii=False
    )


# MARK: - Component wrappers


class EncoderStep(torch.nn.Module):
    """One cache-aware encoder step and the language prompt projection.

    The prompt is a one-hot vector over the 128 prompt slots; `prompts.json` maps locales and
    `auto` to slots. The output is the projected encoder frame that the joint network takes.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.prompt_kernel = model.prompt_kernel

    def forward(self, mel, mel_length, cache_channel, cache_time, cache_length, prompt):
        encoded, encoded_length, channel, time_cache, length = self.encoder.cache_aware_stream_step(
            processed_signal=mel,
            processed_signal_length=mel_length.to(dtype=torch.long),
            cache_last_channel=cache_channel,
            cache_last_time=cache_time,
            cache_last_channel_len=cache_length.to(dtype=torch.long),
            keep_all_outputs=True,
            drop_extra_pre_encoded=DROPPED_PRE_ENCODED_FRAMES,
        )
        frames = encoded.transpose(1, 2)
        prompts = prompt[:, None, :].expand(frames.shape[0], frames.shape[1], prompt.shape[-1])
        projected = self.prompt_kernel(torch.cat([frames, prompts], dim=-1)).transpose(1, 2)
        return (
            projected,
            encoded_length.to(dtype=torch.int32),
            channel,
            time_cache,
            length.to(dtype=torch.int32),
        )


class DecoderWrapper(torch.nn.Module):
    """One step of the prediction network with explicit LSTM state, as for Parakeet.

    `RNNTDecoder.forward` prepends the blank start token unless the module is prepared for
    export, so the wrapper calls `predict` without it. `target_lengths` is part of the
    interface that the runtime shares with Parakeet; one step always has length 1.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, targets, target_lengths, h_in, c_in):
        output, state = self.module.predict(targets.to(dtype=torch.long), state=[h_in, c_in], add_sos=False)
        return output.transpose(1, 2), state[0], state[1]


class JointDecision(torch.nn.Module):
    """The joint network and greedy decision for one projected encoder frame.

    Returns the argmax over the 13,087 tokens and the blank and that token's probability.
    """

    def __init__(self, joint: torch.nn.Module) -> None:
        super().__init__()
        self.joint = joint

    def forward(self, encoder_step, decoder_step):
        enc = self.joint.enc(encoder_step.transpose(1, 2))
        pred = self.joint.pred(decoder_step.transpose(1, 2))
        hidden = enc.unsqueeze(2) + pred.unsqueeze(1)
        logits = self.joint.joint_net[2](self.joint.joint_net[1](self.joint.joint_net[0](hidden)))
        token = torch.argmax(logits, dim=-1)
        probability = torch.gather(torch.softmax(logits, dim=-1), -1, token.unsqueeze(-1)).squeeze(-1)
        return token.to(dtype=torch.int32), probability


# MARK: - Features and audio


def nemo_mel(model: Any, audio: np.ndarray) -> np.ndarray:
    """NeMo's log-mel features for a whole recording, `[128, floor(samples / 160)]`.

    The preprocessor does not normalize the features of this model, so a frame does not depend
    on the frames around it, and streaming features equal those of the whole recording.
    """
    if model.preprocessor.training:
        raise RuntimeError("the preprocessor is in training mode, which adds dither")
    with torch.inference_mode():
        mel, length = model.preprocessor(
            input_signal=torch.from_numpy(audio)[None], length=torch.tensor([len(audio)], dtype=torch.long)
        )
    return mel[0, :, : int(length[0])].numpy().astype(np.float32)


def chunk_input(mel: np.ndarray, start: int, chunk: Chunk) -> tuple[np.ndarray, int]:
    """The encoder input of the step that starts at mel frame `start`, and its valid length.

    The pre-encode cache is the 9 mel frames before `start`, with zeros before the audio.
    The last chunk is padded with zeros after its valid frames.
    """
    window = np.zeros((MEL_BINS, chunk.input_frames), dtype=np.float32)
    first = start - PRE_ENCODE_CACHE_FRAMES
    source = mel[:, max(0, first) : start + chunk.mel_frames]
    offset = max(0, -first)
    window[:, offset : offset + source.shape[1]] = source
    return window[None], offset + source.shape[1]


def initial_caches() -> dict[str, np.ndarray]:
    return {
        "cache_channel": np.zeros((ENCODER_LAYERS, 1, ATTENTION_CACHE_FRAMES, ENCODER_DIM), dtype=np.float32),
        "cache_time": np.zeros((ENCODER_LAYERS, 1, ENCODER_DIM, CONVOLUTION_CACHE_FRAMES), dtype=np.float32),
        "cache_length": np.zeros((1,), dtype=np.int32),
    }


def load_audio(path: Path) -> np.ndarray:
    """Load the first channel of an audio file and resample it to 16 kHz with a polyphase filter."""
    import math

    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data[:, 0]
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = math.gcd(int(rate), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, int(rate) // divisor).astype(np.float32)
    return np.ascontiguousarray(audio)


# MARK: - Streaming


class Steps:
    """Runs encoder steps, decoder steps, and joint decisions with one backend."""

    def encode(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def predict(self, token: int, h: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def joint(self, frame: np.ndarray, decoder: np.ndarray) -> tuple[int, float]:
        raise NotImplementedError


class TorchSteps(Steps):
    """The float32 PyTorch modules, the reference."""

    def __init__(self, model: Any) -> None:
        self.encoder = EncoderStep(model).eval()
        self.decoder = DecoderWrapper(model.decoder).eval()
        self.joint_decision = JointDecision(model.joint).eval()

    def encode(self, inputs):
        with torch.inference_mode():
            outputs = self.encoder(*(torch.from_numpy(inputs[name]) for name in ENCODER_INPUTS))
        return {name: value.numpy() for name, value in zip(ENCODER_OUTPUTS, outputs, strict=True)}

    def predict(self, token, h, c):
        with torch.inference_mode():
            output, h, c = self.decoder(
                torch.tensor([[token]], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                torch.from_numpy(h),
                torch.from_numpy(c),
            )
        return output.numpy(), h.numpy(), c.numpy()

    def joint(self, frame, decoder):
        with torch.inference_mode():
            token, probability = self.joint_decision(torch.from_numpy(frame), torch.from_numpy(decoder))
        return int(token.reshape(-1)[0]), float(probability.reshape(-1)[0])


ENCODER_INPUTS = ("mel", "mel_length", "cache_channel", "cache_time", "cache_length", "prompt")
ENCODER_OUTPUTS = ("encoder", "encoder_length", "cache_channel_out", "cache_time_out", "cache_length_out")


@dataclass
class StreamResult:
    tokens: list[tuple[int, int, float]]
    """Token ID, encoder frame, and probability."""
    encoder: list[np.ndarray]
    """The projected encoder output of each step, `[1024, valid frames]`."""
    caches: list[dict[str, np.ndarray]]
    """The caches after each step, with `keep_caches`."""
    joint_calls: list[tuple[np.ndarray, np.ndarray, int]]
    """The encoder frame, prediction network output, and token of each joint decision, with `trace`."""
    predictions: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]
    """The token, LSTM state, and output of each prediction network step, with `trace`."""


def stream(
    steps: Steps,
    mel: np.ndarray,
    chunk: Chunk,
    prompt: int,
    *,
    encoder_inputs: list[dict[str, np.ndarray]] | None = None,
    keep_caches: bool = False,
    trace: bool = False,
) -> StreamResult:
    """Encodes and greedily decodes a recording chunk by chunk with RNN-T decoding.

    With `encoder_inputs`, each step takes those caches instead of the previous step's outputs,
    which isolates the error of one step from the error that the caches accumulate.
    """
    caches = initial_caches()
    prompt_input = prompt_vector(prompt).numpy()
    h = np.zeros((DECODER_LAYERS, 1, DECODER_DIM), dtype=np.float32)
    c = h.copy()
    decoder_output, h, c = steps.predict(BLANK_ID, h, c)
    result = StreamResult(tokens=[], encoder=[], caches=[], joint_calls=[], predictions=[])
    frames = mel.shape[1]
    frame_offset = 0
    for step, start in enumerate(range(0, frames, chunk.mel_frames)):
        window, valid = chunk_input(mel, start, chunk)
        inputs = dict(caches if encoder_inputs is None else encoder_inputs[step])
        inputs.update(mel=window, mel_length=np.array([valid], dtype=np.int32), prompt=prompt_input)
        outputs = steps.encode(inputs)
        caches = {
            "cache_channel": outputs["cache_channel_out"],
            "cache_time": outputs["cache_time_out"],
            "cache_length": outputs["cache_length_out"].astype(np.int32),
        }
        if keep_caches:
            result.caches.append(caches)
        count = int(outputs["encoder_length"].reshape(-1)[0])
        encoded = outputs["encoder"][0, :, :count]
        result.encoder.append(encoded)
        for local in range(count):
            frame = np.ascontiguousarray(encoded[None, :, local : local + 1])
            for _ in range(MAX_SYMBOLS_PER_FRAME):
                token, probability = steps.joint(frame, decoder_output)
                if trace:
                    result.joint_calls.append((frame, decoder_output, token))
                if token == BLANK_ID:
                    break
                result.tokens.append((token, frame_offset + local, probability))
                state = (h, c)
                decoder_output, h, c = steps.predict(token, h, c)
                if trace:
                    result.predictions.append((token, *state, decoder_output))
        frame_offset += count
    return result


def text(model: Any, tokens: list[int]) -> str:
    """Decoded text without the language tags that the model emits after a sentence."""
    import re

    decoded = model.tokenizer.ids_to_text([t for t in tokens if t < VOCAB_SIZE])
    return re.sub(r"\s*<[a-z]{2}-[A-Z]{2}>", "", decoded).strip()
