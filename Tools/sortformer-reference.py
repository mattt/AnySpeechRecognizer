#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["nemo_toolkit[asr]==3.0.0", "soundfile"]
# ///
"""Write NeMo's Sortformer features and streaming predictions for a 16 kHz mono file.

Usage:
    uv run Tools/sortformer-reference.py <diar_streaming_sortformer_4spk-v2.nemo> \
        <audio.wav> <output-prefix>

The v2.1 checkpoint (diar_streaming_sortformer_4spk-v2.1.nemo) has the same architecture
and configuration and works the same way.

Writes `<prefix>.mel.f32`, log-mel features as little-endian Float32, time-major with
128 values per 10 ms frame, and `<prefix>.preds.f32`, speaker probabilities with
4 values per 80 ms frame. The streaming settings match the package's built-in models:
NVIDIA's low-latency configuration with 6-frame chunks, 1 frame of left and 7 of right
context, a 188-frame FIFO and speaker cache, and an update period of 144 frames.
The parity tests compare the Swift frontend and pipeline with these files.
"""

import sys

import numpy as np
import soundfile
import torch
from nemo.collections.asr.models import SortformerEncLabelModel


def main() -> None:
    checkpoint, audio_path, prefix = sys.argv[1:4]
    model = SortformerEncLabelModel.restore_from(checkpoint, map_location="cpu")
    model.eval()
    model.streaming_mode = True
    model.async_streaming = False
    modules = model.sortformer_modules
    modules.chunk_len = 6
    modules.chunk_right_context = 7
    modules.chunk_left_context = 1
    modules.fifo_len = 188
    modules.spkcache_len = 188
    modules.spkcache_update_period = 144
    featurizer = model.preprocessor.featurizer
    featurizer.dither = 0.0
    featurizer.pad_to = 0

    audio, rate = soundfile.read(audio_path, dtype="float32")
    if rate != 16_000 or audio.ndim != 1:
        sys.exit("error: the audio must be 16 kHz mono")
    signal = torch.from_numpy(audio).unsqueeze(0)
    length = torch.tensor([signal.shape[1]])
    with torch.no_grad():
        features, feature_length = model.process_signal(
            audio_signal=signal, audio_signal_length=length
        )
        features = features[:, :, : int(feature_length.max())]
        predictions = model.forward_streaming(features, feature_length)
    mel = features[0].transpose(0, 1).contiguous().numpy().astype("<f4")
    preds = predictions[0].numpy().astype("<f4")
    mel.tofile(prefix + ".mel.f32")
    preds.tofile(prefix + ".preds.f32")
    print(f"{mel.shape[0]} mel frames, {preds.shape[0]} prediction frames", file=sys.stderr)


if __name__ == "__main__":
    main()
