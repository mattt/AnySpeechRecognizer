# Streaming Sortformer v2.1 conversion

This maintainer tool converts NVIDIA's
[Streaming Sortformer 4spk v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
speaker diarization checkpoint to Core ML for AnySpeechRecognizer's Sortformer runtime,
`SortformerDiarizationModel(version: .v2_1)`.
The output has the same tensor interface as the v2 model that the runtime also loads.
The tool is not part of the Swift package products.

## v2.1 and v2

NVIDIA's model cards and checkpoints, compared at
`cd03eee90fbec18297ac31b8c21546e596b7f71c` (v2.1) and
`84edd514b8ef68004c10086918cd62f2148cbd59` (v2):

| Fact | v2 | v2.1 |
| --- | --- | --- |
| License | CC-BY-4.0 | NVIDIA Open Model License Agreement |
| Architecture | 17-layer Fast Conformer (NEST) encoder, 18-layer Transformer, 4 sigmoid outputs | Same; 990 tensors with the same names and shapes |
| Trained tensors | | All 988 trained tensors differ from v2; the mel filter bank and window are identical |
| Preprocessor | 128 mel bins, 25 ms window, 10 ms hop, 512-point FFT, no normalization | Same |
| Streaming and speaker cache parameters | NeMo defaults | The same values, written out in `model_config.yaml` (`pred_score_threshold` 0.25, `scores_boost_latest` 0.05, `sil_threshold` 0.2, `strong_boost_rate` 0.75, `weak_boost_rate` 1.5, `min_pos_scores_rate` 0.5, `spkcache_sil_frames_per_spk` 3) |
| Recommended low-latency configuration | 6-frame chunks, 7 frames of right context, FIFO 188, update period 144, cache 188 | Same |
| Training data | Fisher, AMI, VoxConverse, ICSI, AISHELL-4, DIHARD III dev, NIST SRE, LibriSpeech, AliMeeting, DiPCo | The same and NOTSOFAR1 |
| Maximum speakers | 4 | 4 |
| Post-processing parameters in NeMo | CALLHOME part 1 and DIHARD III dev presets | None published |

The checkpoint's only other configuration changes are the training warmup steps
and the NeMo version that saved it (2.6.0rc0).
`causal_attn_rate` and `causal_attn_rc` apply only in training.
NVIDIA reports lower DER for v2.1 at 1.04 s latency on AliMeeting, AMI, NOTSOFAR1,
CALLHOME, and CH109, and higher DER on DIHARD III (15.09% against 14.49% with up to
four speakers).

Because nothing but the weights differ, the conversion keeps the v2 model's interface,
and the runtime needs no change except a manifest, an attribution, and a version case.
The runtime's speaker cache constants equal the v2.1 configuration values.

## License

Use of v2.1 is governed by the NVIDIA Open Model License Agreement,
version released October 24, 2025, which the model card links.
`NVIDIA-Open-Model-License.txt` is a copy of the agreement, transcribed from NVIDIA's PDF;
its words match the PDF's text.
The checkpoint is not gated.
The relevant terms:

- Preamble: "You are free to create and distribute Derivative Models."
- 1.1: "'Derivative Model' means all (a) modifications to the Model, (b) works based on
  the Model, and (c) any other derivative works of the Model."
  A Core ML conversion of the weights is a Derivative Model.
- 2.2 grants a license to "reproduce, use, create derivative works of, make, have made,
  sell, offer for sale, distribute (through multiple tiers of distribution) and import
  the Model", conditioned on compliance with the agreement.
- 3: "You may reproduce and distribute copies of the Model or Derivative Models thereof
  in any medium, with or without modifications, provided that You meet the following
  conditions".
- 3.1: "If you distribute the Model, You must give any other recipients of the Model a
  copy of this Agreement and include the following attribution notice within a 'Notice'
  text file with such copies: 'Licensed by NVIDIA Corporation under the NVIDIA Open
  Model License'".
- 2.3: "Use of the Models under the Agreement must be consistent with NVIDIA's
  Trustworthy AI terms".
  Those terms prohibit use "in violation of applicable law or regulation, including but
  not limited to (a) illegal surveillance, (b) illegal collection or processing of
  biometric information without the consent of the subject where required under
  applicable law, or (c) illegal harassment, abuse, threatening or bullying".
- 2.1: rights terminate if you "bypass, disable, reduce the efficacy of, or circumvent
  any technical limitation, safety guardrail ... contained in the Model without a
  substantially similar Guardrail".
  The checkpoint documents no guardrail; the conversion keeps the architecture,
  the four outputs, and the streaming algorithm, and changes no weights except by
  float16 rounding.
- 2.1 also lets NVIDIA update the agreement; licensees must comply with the update
  or stop distributing.
- 3.2 applies only to NVIDIA Cosmos models; Sortformer is not one.
- 5 permits NVIDIA's names only to describe the model's origin and to reproduce the notice.
- 3.3 permits separate terms for modifications; this tool adds none to the model files.

How the output meets them:

| Requirement | Where |
| --- | --- |
| Copy of the agreement | `LICENSE` in the output and the Hugging Face repository |
| Notice file with the required sentence | `NOTICE`, which also names the source revision and the modifications |
| License metadata | Model card `license: other`, `license_name: nvidia-open-model-license`, `license_link` to NVIDIA's page, as on NVIDIA's own card; Hugging Face has no identifier for this license |
| Trustworthy AI terms | Model card and `NOTICE` |
| Attribution in applications | `SortformerDiarizationModel.attribution` for `.v2_1` contains the required sentence and the links |
| Core ML metadata | The model's `license` field is the required sentence and the agreement's URL |

## Outputs

`convert.py` writes `build/artifacts/`, which has the Hugging Face repository's layout:

| File | Contents |
| --- | --- |
| `SortformerLowLatency.mlmodelc` | Compiled Core ML pipeline: a float32 pre-encoder with cache, FIFO, and chunk packing, then a float16 encoder, Transformer, and speaker heads |
| `LICENSE` | The NVIDIA Open Model License Agreement |
| `NOTICE` | The required notice, the source, and the modifications |
| `conversion.json` | Source revision and digest, tool versions, settings, interface, and the size and SHA-256 digest of every file |
| `README.md` | Hugging Face model card |

### Tensor interface

The pipeline takes one streaming step, as the runtime's `CoreMLSortformerEngine` expects:

| Name | Type | Contents |
| --- | --- | --- |
| `chunk` | Float32 [1, 112, 128] | Log-mel features of 1 + 6 + 7 frames of 80 ms, time-major, zero-padded |
| `chunk_lengths` | Int32 [1] | Valid mel frames |
| `spkcache`, `fifo` | Float32 [1, 188, 512] | Speaker cache and FIFO embeddings, valid rows first |
| `spkcache_lengths`, `fifo_lengths` | Int32 [1] | Valid rows |
| `speaker_preds` (output) | Float32 [1, 390, 4] | Probabilities for the valid cache, FIFO, and chunk rows, then zeros |
| `chunk_pre_encoder_embs_out` (output) | Float32 [1, 14, 512] | Pre-encoder embeddings of the chunk |
| `chunk_pre_encoder_lengths_out` (output) | Int32 [1] | Valid chunk embedding rows |

The creator-defined metadata has the v2 model's keys: `chunk_len` 6,
`chunk_left_context` 1, `chunk_right_context` 7, `fifo_len` 188, `spkcache_len` 188,
`spkcache_update_period` 144, `subsampling_factor` 8, `frame_duration` 0.08,
and `mel_feature_frames` 48, plus `source_checkpoint`.
FluidInference's v2 model returns the chunk embeddings as Float16; this model returns
Float32, and the runtime reads both.

The packing is NeMo's `concat_and_pad` with fixed shapes: a gather whose indices come
from the three lengths, so the trace has no data-dependent slices.
The pre-encoder stays in float32 because its embeddings enter the speaker cache
and FIFO and are reused in later chunks.
A Core ML pipeline joins the two programs; the pre-encoder's chunk embeddings and
lengths are pipeline outputs because the head does not consume them,
so no tensor is both an input and an output of a program.

## Setup

Requirements: macOS on Apple silicon with Xcode or the Command Line Tools
(`xcrun coremlcompiler`) and [uv](https://docs.astral.sh/uv/).
The environment takes 1.4 GB, the checkpoint 471 MB,
and the intermediate package and the output 235 MB each.

```sh
cd Tools/SortformerConversion/v2.1
uv sync
```

The lock file pins Python 3.12, PyTorch 2.7.0, NeMo 2.7.3, coremltools 9.0,
and NumPy 2.3.5.
coremltools 9.0 converts one-element arrays with `int()`, which NumPy 2.4 and later reject.
FluidInference reports that tracing with PyTorch 2.9 produced a model that the
macOS 26 BNNS compiler rejects; 2.7.0 avoids it.

## Convert

```sh
uv run python convert.py
```

The converter downloads the pinned checkpoint to `build/cache/`
(or takes `--checkpoint`), checks its size and SHA-256 digest against the digest that
Hugging Face records, traces and converts the two stages, builds the pipeline,
checks its interface against the contract, compiles it, and compares the compiled model
with NeMo's `forward_for_export` on random inputs with a partly filled cache and FIFO.
On an M4 Max it takes about 20 seconds.
The maximum differences were 5.9e-5 for the probabilities and 8.4e-5 for the
embeddings (`CPU_ONLY`).

A new conversion does not reproduce earlier files byte for byte,
because `coremldata.bin` records the conversion date;
publish a new revision and regenerate the manifest instead of replacing files.

## Verify

```sh
uv run python verify.py --audio two-speakers.wav four-speakers.wav overlap.wav long.wav \
    --compute-units CPU_AND_NE CPU_ONLY CPU_AND_GPU --write-reference build/reference
uv run python convert.py --update-card --verify-report build/reports/verify.json
```

`verify.py` computes NeMo's features for each file and runs NeMo's `forward_streaming`
in float32 on the CPU with the low-latency configuration.
It then runs NeMo's synchronous streaming loop again, with the same chunks and NeMo's
own `streaming_update` for the speaker cache and FIFO, but with each model call replaced
by the Core ML model.
It compares frame probabilities, decisions at a 0.5 threshold per frame and speaker,
and speech regions after NeMo's `ts_vad_post_processing` with the v2 CALLHOME
parameters, at 10 ms.
`--model` takes other compiled models with the same interface for comparison.
`--write-reference` writes NeMo's features and probabilities in the format of
`Tools/sortformer-reference.py` for the Swift parity tests.
`--update-card` rewrites only the model card, with a summary of the report.

### Results

Observed on October 14, 2026, on an M4 Max with macOS 27.0.1 (26A434).
The fixtures are the synthetic conversations from `Tests/Fixtures/Diarization`
(made with `Tools/make-diarization-audio.swift`): 62 s, 75 s, 36 s, and the 307 s concatenation `long`.
Agreement counts frame and speaker pairs (4 per 80 ms frame).
"Call" is the median time of one `MLModel.predict` from Python.

| Fixture | Compute units | Probability difference, mean / max | Agreement at 0.5 | Agreement after post-processing | Call |
| --- | --- | --- | --- | --- | --- |
| two-speakers | `CPU_AND_NE` | 3.2e-4 / 0.043 | 99.97% (1 of 3,108 differ) | 100.00% | 57.6 ms |
| four-speakers | `CPU_AND_NE` | 1.4e-3 / 0.086 | 99.97% (1 of 3,764) | 100.00% | 58.3 ms |
| overlap | `CPU_AND_NE` | 3.3e-4 / 0.018 | 100% | 99.95% | 59.1 ms |
| long | `CPU_AND_NE` | 7.1e-4 / 0.095 | 99.96% (6 of 15,352) | 99.97% | 56.8 ms |
| two-speakers | `CPU_ONLY` | 2.3e-4 / 0.045 | 100% | 100.00% | 59.6 ms |
| four-speakers | `CPU_ONLY` | 6.9e-4 / 0.098 | 100% | 100.00% | 59.8 ms |
| overlap | `CPU_ONLY` | 1.7e-4 / 0.014 | 100% | 100.00% | 60.8 ms |
| long | `CPU_ONLY` | 7.1e-4 / 0.085 | 99.98% (3 of 15,352) | 99.98% | 60.6 ms |
| two-speakers | `CPU_AND_GPU` | 4.3e-5 / 0.010 | 99.97% (1 of 3,108) | 100.00% | 18.6 ms |
| four-speakers | `CPU_AND_GPU` | 5.0e-5 / 0.007 | 100% | 100.00% | 18.6 ms |
| overlap | `CPU_AND_GPU` | 2.5e-5 / 0.001 | 100% | 100.00% | 18.6 ms |
| long | `CPU_AND_GPU` | 1.1e-4 / 0.039 | 99.99% (2 of 15,352) | 99.99% | 18.6 ms |

- NeMo 2.7.3 (this environment) and NeMo 3.0.0 with PyTorch 2.14.1
  (`Tools/sortformer-reference.py`) give identical features and probabilities within
  1.7e-6 on `four-speakers` and `overlap`.
- FluidInference's `v3/fp16/SortformerNvidiaLow_v2.1.mlmodelc`
  (`FluidInference/diar-streaming-sortformer-coreml` at
  `ae9a27ab45dc0aa3abede7d2d6bad2b7a69aa6d1`) has the same interface and gives similar
  numbers: on `long`, 9.6e-4 / 0.166 and 99.95% with `CPU_AND_NE`,
  and 1.7e-4 / 0.045 and 99.99% with `CPU_AND_GPU`.
  This tool's code is independent of FluidInference's scripts, which were only read.
- As with v2, `cpuAndNeuralEngine` is no faster than the CPU,
  so most of the model probably does not run on the Neural Engine; this was not profiled.

### Package runtime

The package ran the compiled model staged under `/tmp` through its own Swift pipeline
(features, chunking, speaker cache, and post-processing).
The opt-in `SortformerModelTests` with `ANYSPEECH_SORTFORMER_VERSION=v2.1` and NeMo
references from `--write-reference` passed on `two-speakers`, `four-speakers`, and
`overlap`; the Swift pipeline's probabilities differed from NeMo's by
3.1e-4 / 0.052, 1.3e-3 / 0.089, and 3.3e-4 / 0.016 (mean / max),
with 100% agreement at 0.5.

The CLI's `evaluate --diarize --diarizer-version v2.1` with a Parakeet v3 transcriber
compared both versions on the same fixtures with `cpuAndNeuralEngine`.
DER uses 10 ms frames, includes overlap, and maps speakers one to one.

| Fixture | Speakers, reference / found (v2, v2.1) | DER, 0.25 s collar, v2 / v2.1 | DER, no collar, v2 / v2.1 |
| --- | --- | --- | --- |
| two-speakers, 62.1 s | 2 / 2, 2 | 0.0% / 0.0% | 9.1% / 9.9% |
| four-speakers, 75.2 s | 4 / 3, 3 | 32.6% / 18.8% | 40.7% / 31.6% |
| overlap, 36.4 s | 3 / 3, 3 | 0.0% / 0.1% | 15.0% / 16.8% |
| long, 307.0 s | 4 / 3, 3 | 18.1% / 13.5% | 29.0% / 26.3% |

File input and live input in 100 ms blocks gave the same DER for every fixture and version.
Elapsed time and peak resident memory were the same for both versions within 0.3 s
and 7 MB (for example, `long` in file mode: 36.9 s and 37.1 s, 127 MB each).
v2.1, like v2, merges two of the four synthetic voices on `four-speakers`;
the lower confusion (9.8 s against 15.6 s with the collar) accounts for its lower DER.
Synthetic voices are an easy case; these numbers do not predict accuracy on recordings.

## Upload

`upload_dry_run.py` re-hashes every file against `conversion.json`,
checks that `LICENSE` is `NVIDIA-Open-Model-License.txt`, that `NOTICE` contains the
required sentence, and that the model card has the license metadata,
then prints the commands to publish the directory:

```sh
uv run python upload_dry_run.py
hf repo create mattt/diar-streaming-sortformer-4spk-v2.1-coreml --repo-type model
hf upload mattt/diar-streaming-sortformer-4spk-v2.1-coreml build/artifacts . --repo-type model \
    --commit-message "Add Streaming Sortformer 4spk v2.1 Core ML artifacts"
```

Then generate the runtime manifest from the new commit:

```sh
uv run Tools/make-sortformer-manifest.py --version v2.1 <commit> \
    > Sources/AnySpeechRecognizer/Diarization/SortformerModelManifest+V2_1.swift
```

Do not change files in place after publication;
publish a fixed artifact under a new revision and update the manifest.

The artifacts were published on October 14, 2026, as commit
[`1fa9c6d17eb1cfa252f53562ec6d12287aac7110`](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2.1-coreml/commit/1fa9c6d17eb1cfa252f53562ec6d12287aac7110)
of [mattt/diar-streaming-sortformer-4spk-v2.1-coreml](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2.1-coreml):
the eleven model files (246,279,192 bytes), `LICENSE`, `NOTICE`, `conversion.json`,
and `README.md`.
With an empty `HF_HUB_CACHE` and a staged Parakeet directory,
`prepare --diarize --diarizer-version v2.1` installed them from that revision in 35.4 s,
including the download, the digest checks, and loading both models,
and `evaluate` with the installed model gave the same DER on `four-speakers`
as the staged copy.
