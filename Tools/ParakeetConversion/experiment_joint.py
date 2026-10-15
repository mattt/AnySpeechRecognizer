# SPDX-License-Identifier: Apache-2.0

"""Export experimental joint-network variants that need fewer Core ML calls per decoding step.

These models are for measurement only. They are not part of the published outputs, and
the published `JointDecisionv3.mlmodelc` and `Decoder.mlmodelc` do not change.

Variants, all float16 ML programs like the published joint decision:

- `JointDecisionNoTopK`: the published joint decision for one frame without the top-K outputs.
- `JointDecisionProjected`: one frame of the encoder projection (`joint.enc`, 1024 to 640)
  computed in advance; the model adds the prediction projection, applies ReLU and the output layer.
- `EncoderProjection<frames>`: `joint.enc` for a whole encoder window, one call per window.
- `JointDecisionFrames<K>`: greedy decisions for K consecutive encoder frames with one
  prediction network output. Decoding uses the decisions until it emits a token, which
  changes the prediction network output.
- `JointDecisionFrames<K>Projected`: the same with projected encoder frames.

Each model also checks its decisions against the PyTorch joint network on random inputs.

Usage:
    uv run python experiment_joint.py --variant v3 --frames 16 --window-frames 76 188
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch

import parakeet as pk
from convert import compile_model, convert_traced


class JointFrames(torch.nn.Module):
    """Greedy TDT decisions for K encoder frames and one prediction network output."""

    def __init__(self, joint: torch.nn.Module, projected: bool, top_k: int = 0) -> None:
        super().__init__()
        self.joint = joint
        self.projected = projected
        self.vocab_with_blank = pk.VOCAB_SIZE + 1
        self.top_k = top_k

    def forward(self, encoder_steps: torch.Tensor, decoder_step: torch.Tensor):
        # encoder_steps [1, D, K]; decoder_step [1, 640, 1]
        enc = encoder_steps.transpose(1, 2)  # [1, K, D]
        if not self.projected:
            enc = self.joint.enc(enc)  # [1, K, 640]
        dec = self.joint.pred(decoder_step.transpose(1, 2))  # [1, 1, 640]
        x = torch.relu(enc + dec)
        logits = self.joint.joint_net[2](x)  # [1, K, V + 1 + durations]
        token_logits = logits[..., : self.vocab_with_blank]
        duration_logits = logits[..., self.vocab_with_blank :]
        token_id = torch.argmax(token_logits, dim=-1).to(dtype=torch.int32)
        probs = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(probs, dim=-1, index=token_id.long().unsqueeze(-1)).squeeze(-1)
        duration = torch.argmax(duration_logits, dim=-1).to(dtype=torch.int32)
        return token_id, token_prob, duration


class EncoderProjection(torch.nn.Module):
    def __init__(self, joint: torch.nn.Module) -> None:
        super().__init__()
        self.joint = joint

    def forward(self, encoder: torch.Tensor) -> torch.Tensor:
        return self.joint.enc(encoder.transpose(1, 2)).transpose(1, 2)  # [1, 640, T]


def export(module: torch.nn.Module, example: tuple, inputs: list[ct.TensorType], outputs: list[str], path: Path):
    traced = torch.jit.trace(module, example, strict=False).eval()
    model = convert_traced(traced, inputs, [ct.TensorType(name=n) for n in outputs])
    package = path.with_suffix(".mlpackage")
    if package.exists():
        shutil.rmtree(package)
    model.save(str(package))
    compile_model(package, path)
    shutil.rmtree(package)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(pk.VARIANTS), default="v3")
    parser.add_argument("--frames", type=int, nargs="+", default=[16], help="K for JointDecisionFrames<K>")
    parser.add_argument("--window-frames", type=int, nargs="+", default=[76, 188], help="EncoderProjection sizes")
    parser.add_argument("--output-dir", type=Path, help="Default: build/experiments/<variant>")
    parser.add_argument("--cache-dir", type=Path, default=pk.DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    variant = pk.VARIANTS[args.variant]
    output_dir = (args.output_dir or pk.BUILD_DIR / "experiments" / variant.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = pk.load_model(variant, args.cache_dir.resolve())
    joint = model.joint.eval()
    # PyTorch encoder work runs before Core ML loads any model in this process (see verify_live.py).
    inputs = agreement_inputs(model)
    f32 = np.float32

    def tensor(name: str, shape: tuple[int, ...]) -> ct.TensorType:
        return ct.TensorType(name=name, shape=shape, dtype=f32)

    decision_outputs = ["token_id", "token_prob", "duration"]
    exported: dict[str, Any] = {}
    generator = torch.Generator().manual_seed(1)
    dec = torch.randn(1, pk.DECODER_DIM, 1, generator=generator)

    one = torch.randn(1, pk.ENCODER_DIM, 1, generator=generator)
    export(
        JointFrames(joint, projected=False),
        (one, dec),
        [tensor("encoder_step", (1, pk.ENCODER_DIM, 1)), tensor("decoder_step", (1, pk.DECODER_DIM, 1))],
        decision_outputs,
        output_dir / "JointDecisionNoTopK.mlmodelc",
    )
    exported["JointDecisionNoTopK.mlmodelc"] = {"frames": 1, "projected": False}
    proj_one = torch.randn(1, pk.DECODER_DIM, 1, generator=generator)
    export(
        JointFrames(joint, projected=True),
        (proj_one, dec),
        [tensor("encoder_step", (1, pk.DECODER_DIM, 1)), tensor("decoder_step", (1, pk.DECODER_DIM, 1))],
        decision_outputs,
        output_dir / "JointDecisionProjected.mlmodelc",
    )
    exported["JointDecisionProjected.mlmodelc"] = {"frames": 1, "projected": True}
    for k in args.frames:
        steps = torch.randn(1, pk.ENCODER_DIM, k, generator=generator)
        export(
            JointFrames(joint, projected=False),
            (steps, dec),
            [tensor("encoder_steps", (1, pk.ENCODER_DIM, k)), tensor("decoder_step", (1, pk.DECODER_DIM, 1))],
            decision_outputs,
            output_dir / f"JointDecisionFrames{k}.mlmodelc",
        )
        exported[f"JointDecisionFrames{k}.mlmodelc"] = {"frames": k, "projected": False}
        projected = torch.randn(1, pk.DECODER_DIM, k, generator=generator)
        export(
            JointFrames(joint, projected=True),
            (projected, dec),
            [tensor("encoder_steps", (1, pk.DECODER_DIM, k)), tensor("decoder_step", (1, pk.DECODER_DIM, 1))],
            decision_outputs,
            output_dir / f"JointDecisionFrames{k}Projected.mlmodelc",
        )
        exported[f"JointDecisionFrames{k}Projected.mlmodelc"] = {"frames": k, "projected": True}
    for frames in args.window_frames:
        encoder = torch.randn(1, pk.ENCODER_DIM, frames, generator=generator)
        export(
            EncoderProjection(joint),
            (encoder,),
            [tensor("encoder", (1, pk.ENCODER_DIM, frames))],
            ["encoder_projected"],
            output_dir / f"EncoderProjection{frames}.mlmodelc",
        )
        exported[f"EncoderProjection{frames}.mlmodelc"] = {"frames": frames}

    # Agreement with PyTorch on realistic inputs: encoder frames of speech and decoder outputs
    # along NeMo's tokens for one fixture.
    agreement = check_agreement(inputs, joint, output_dir, args.frames)
    (output_dir / "experiment.json").write_text(
        json.dumps({"variant": variant.name, "models": exported, "agreement": agreement}, indent=2) + "\n"
    )
    print(json.dumps(agreement, indent=2))


def agreement_inputs(model: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Encoder frames of speech and prediction network outputs after a few tokens."""
    fixture = pk.BUILD_DIR / "fixtures" / "en-us-long.wav"
    audio = pk.load_audio(fixture)
    window, valid = pk.window_audio(audio)
    mel, mel_length = pk.nemo_mel(model, window, valid)
    with torch.inference_mode():
        encoded, length = model.encoder(audio_signal=torch.from_numpy(mel), length=torch.from_numpy(mel_length).long())
        model.decoder._rnnt_export = True
        state = None
        outputs = []
        token = torch.tensor([[pk.BLANK_ID]])
        for token_id in [pk.BLANK_ID, 100, 200, 300, 400]:
            token = torch.tensor([[token_id]])
            out, _, state = model.decoder(targets=token, target_length=torch.tensor([1]), states=state)
            outputs.append(out[:, :, :1])
    return encoded[:, :, : int(length[0])], outputs


def check_agreement(
    inputs: tuple[torch.Tensor, list[torch.Tensor]], joint: torch.nn.Module, output_dir: Path, frames: list[int]
) -> dict[str, Any]:
    enc, outputs = inputs
    n = enc.shape[-1]
    reference = JointFrames(joint, projected=False)
    projection = EncoderProjection(joint)
    units = ct.ComputeUnit.CPU_AND_NE
    one = ct.models.CompiledMLModel(str(output_dir / "JointDecisionProjected.mlmodelc"), compute_units=units)
    result: dict[str, Any] = {}
    with torch.inference_mode():
        enc_proj = projection(enc)
        for k in frames:
            multi = ct.models.CompiledMLModel(str(output_dir / f"JointDecisionFrames{k}.mlmodelc"), compute_units=units)
            agree = total = agree_one = 0
            for dec in outputs:
                for start in range(0, n - k, k):
                    ref_tokens, _, ref_durations = reference(enc[:, :, start : start + k], dec)
                    out = multi.predict(
                        {"encoder_steps": enc[:, :, start : start + k].numpy(), "decoder_step": dec.numpy()}
                    )
                    agree += int(
                        (np.asarray(out["token_id"]).ravel() == ref_tokens.numpy().ravel()).sum()
                        + (np.asarray(out["duration"]).ravel() == ref_durations.numpy().ravel()).sum()
                    )
                    total += 2 * k
                    for i in range(k):
                        out1 = one.predict(
                            {
                                "encoder_step": enc_proj[:, :, start + i : start + i + 1].numpy(),
                                "decoder_step": dec.numpy(),
                            }
                        )
                        agree_one += int(
                            int(np.asarray(out1["token_id"]).ravel()[0]) == int(ref_tokens.ravel()[i])
                            and int(np.asarray(out1["duration"]).ravel()[0]) == int(ref_durations.ravel()[i])
                        )
            result[f"JointDecisionFrames{k}"] = {"token_and_duration_agreement": agree / total}
            result["JointDecisionProjected"] = {"token_and_duration_agreement": agree_one / (total // 2)}
    return result


if __name__ == "__main__":
    main()
