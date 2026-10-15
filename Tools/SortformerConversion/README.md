# Sortformer conversion

This maintainer tool converts NVIDIA's Streaming Sortformer 4spk v2
speaker diarization checkpoint to Core ML for AnySpeechRecognizer's
`SortformerDiarizationModel`.
It produces a package-owned replacement for the model that the runtime loads today,
FluidInference's `v3/fp16/SortformerNvidiaLow_v2.mlmodelc`,
with the same tensor interface, streaming configuration, and precision,
and with metadata that names NVIDIA's v2 checkpoint and its CC BY 4.0 license.
The tool is not part of the Swift package products.

| | |
| --- | --- |
| Source | [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) at `84edd514b8ef68004c10086918cd62f2148cbd59`, `diar_streaming_sortformer_4spk-v2.nemo` (SHA-256 `b371afce…83329`) |
| Comparison | [FluidInference/diar-streaming-sortformer-coreml](https://huggingface.co/FluidInference/diar-streaming-sortformer-coreml) at `ae9a27ab45dc0aa3abede7d2d6bad2b7a69aa6d1`, `v3/fp16/SortformerNvidiaLow_v2.mlmodelc` |
| Proposed repository | `mattt/diar-streaming-sortformer-4spk-v2-coreml` |

The repository name keeps the version because NVIDIA's v2.1 checkpoint
is under a different license (the NVIDIA Open Model License).
Revisions, the streaming configuration, the interface, and attribution text
are defined in `sortformer.py`.

## Outputs

`convert.py` writes `build/sortformer/`:

| File | Contents |
| --- | --- |
| `SortformerLowLatency.mlmodelc` | Compiled Core ML pipeline of two ML programs: the pre-encoder and the packing of cache, FIFO, and chunk frames (float32), then the FastConformer encoder, Transformer, and speaker sigmoids (float16). |
| `conversion.json` | Source revision and digest, tool versions, settings, and the size and SHA-256 digest of every file. |
| `README.md` | Hugging Face model card with the license, attribution, base model, and usage. |

The name describes NVIDIA's low-latency streaming configuration,
which the input and output shapes fix.

### Tensor interface

The pipeline is a Core ML pipeline model with an iOS 16 and macOS 13 deployment target.

| Name | Type | Shape |
| --- | --- | --- |
| input `chunk` | Float32 | [1, 112, 128] |
| input `chunk_lengths` | Int32 | [1] |
| input `spkcache` | Float32 | [1, 188, 512] |
| input `spkcache_lengths` | Int32 | [1] |
| input `fifo` | Float32 | [1, 188, 512] |
| input `fifo_lengths` | Int32 | [1] |
| output `speaker_preds` | Float32 | [1, 390, 4] |
| output `chunk_pre_encoder_embs_out` | Float16 | [1, 14, 512] |
| output `chunk_pre_encoder_lengths_out` | Int32 | [1] |

The creator-defined metadata has the keys that `CoreMLSortformerEngine` reads
(`chunk_len` 6, `chunk_left_context` 1, `chunk_right_context` 7, `fifo_len` 188,
`spkcache_len` 188, `spkcache_update_period` 144, `subsampling_factor` 8),
FluidInference's `frame_duration` (0.08) and `mel_feature_frames` (48, the chunk's
mel frames without context),
and `source_checkpoint` and `source_checkpoint_sha256`.
The author, license, and description fields name NVIDIA, CC BY 4.0, and the conversion.

`chunk_pre_encoder_embs_out` is Float16, as in FluidInference's model.
The head casts every embedding to float16 before its first operation,
so rounding the chunk embeddings that return as cache and FIFO input changes
nothing in the head; only the runtime's mean silence embedding is averaged from
the rounded values.

## Setup

Requirements:

- macOS on Apple silicon with Xcode or the Command Line Tools (for `swiftc`).
- [uv](https://docs.astral.sh/uv/).
- About 2.6 GB of disk space: 1.4 GB for the environment in `.venv/`,
  471 MB for the checkpoint in `build/cache/`,
  235 MB each for the package in `build/work/` and the compiled output,
  235 MB for FluidInference's model if `verify.py` downloads it,
  and 25 MB of NeMo references for the four fixtures.

```sh
cd Tools/SortformerConversion
uv sync
```

The lock file pins Python 3.12, PyTorch 2.7.0, coremltools 9.0, NeMo 2.7.3,
and NumPy 2.3.5.
PyTorch 2.7.0 and coremltools 9.0 are the versions in the `buildInfo` of
FluidInference's v3 model.
NeMo 2.7.3 was the current release when FluidInference rebuilt v3 (June 2026).
NumPy is pinned to 2.3 because coremltools 9.0 converts one-element arrays with
`int()`, which NumPy 2.4 rejects, and NeMo's pyarrow dependency needs NumPy 2.

## Convert

```sh
uv run python convert.py
```

The converter downloads the checkpoint to `build/cache/` and checks its size and
SHA-256 digest against the pinned values and the digest that Hugging Face records.
It traces the pre-encoder and the head with `torch.jit.trace`,
converts the pre-encoder with float32 and the head with float16 compute precision,
joins them with `coremltools.utils.make_pipeline`,
checks the pipeline's interface against `sortformer.CONTRACT`,
and compiles it with `coremltools.utils.compile_model`.
It takes about 20 seconds on an M4 Max.

The steps follow FluidInference's
[mobius](https://github.com/FluidInference/mobius/tree/26f5fa8059d5ab3d34449adce6ff6ed86734e522/models/speaker-diarization/sortformer-streaming/conversion)
`convert_to_coreml.py` and `build_all_variants.py`, which built the v3 models:

1. The pre-encoder wrapper runs NeMo's `encoder.pre_encode` on the chunk and packs the
   valid speaker cache, FIFO, and chunk frames at the start of a 390-frame sequence with
   one gather (`fixed_concat_and_pad`), so the graph has static shapes.
2. The head wrapper runs `frontend_encoder` with `bypass_pre_encode` and `forward_infer`,
   and passes the chunk embeddings and lengths through identity operations.
   The macOS 26 BNNS graph compiler rejects a function whose input is also its output.

Converting twice gives the same `model.mil` and weight files.
Each `coremldata.bin` differs between runs: it records the conversion date and stores
the user-defined metadata in an order that changes between runs.

## Verify

The fixtures are the synthetic multi-speaker recordings from
`Tools/make-diarization-audio.swift` and the plans in `Tests/Fixtures/Diarization`:

```sh
xcrun swift ../make-diarization-audio.swift ../../Tests/Fixtures/Diarization/overlap.tsv /tmp/diar-fixtures/overlap
uv run python verify.py --audio /tmp/diar-fixtures/two-speakers.wav /tmp/diar-fixtures/four-speakers.wav \
    /tmp/diar-fixtures/overlap.wav /tmp/diar-fixtures/long.wav --compute-units CPU_AND_NE CPU_ONLY
```

`long.wav` concatenates the plans two-speakers, four-speakers, overlap, two-speakers,
and four-speakers (307.0 s).
`--reference-model` uses a local copy of FluidInference's model instead of downloading it.

`verify.py` checks:

1. **Interface.** `describe_model.swift` loads each compiled model with Core ML and prints
   the input and output names, types, and shapes and the metadata, as the Swift runtime
   sees them. The script compares both models with each other and with the contract.
2. **Weights.** `compare_weights.py` pairs every weight tensor of the two models by name
   from `model.mil` and compares the bytes.
3. **Parity.** NeMo's float32 `forward_streaming` on each file, against the same streaming
   loop with each Core ML model in place of the PyTorch modules.
   Both use NeMo's mel features and NeMo's `streaming_update` for the speaker cache and FIFO,
   so the differences come from the models.
   Each Core ML step is also compared with NeMo's `forward_for_export` on the same inputs.
   Decisions are compared at a threshold of 0.5 and with the CALLHOME hysteresis
   thresholds (onset 0.641, offset 0.561), per frame and speaker.

It writes NeMo's features and predictions to `build/reference/<name>.mel.f32` and
`<name>.preds.f32`, in the format of `Tools/sortformer-reference.py`,
for the Swift opt-in tests, and the report to `build/reports/verify.json`.

### Results

Observed on October 14, 2026, on an M4 Max with macOS 27.0.1 (26A434),
with the pinned environment.

**Interface.** Core ML reports the same input and output names, types, and shapes
for both models, and the same values for the seven metadata keys that the runtime reads.

**Weights.** 883 of the 900 weight tensors are byte-identical to FluidInference's,
including the whole pre-encoder weight file (`model0/weights/0-weight.bin`,
the same SHA-256 digest as in the runtime's built-in manifest).
Both `model.mil` files are identical except their `buildInfo` line (Core ML compiler
3600.25.2 here, 3520.5.1 for FluidInference).
The 17 different tensors are the relative positional projections of the 17 encoder
layers ([1, 8, 64, 779], float16), which the trace folds into constants:
the float32 product of NeMo's positional encoding and each layer's `linear_pos` weight,
rounded to float16.
In each, 136 to 287 of 398,848 values differ.
The largest difference is 0.031, one float16 step at magnitudes from 32 to 64;
the largest values are 24 to 55.
A float32 product computed in this environment reproduces this conversion's values
exactly, and both conversions differ from a float64 product in about 0.06% of values
(this conversion 4,085, FluidInference 3,757 of 6,780,416),
so the difference is float32 rounding on different machines.

**Parity.** Mean and maximum absolute difference of the speaker probabilities, and the
share of frame and speaker decisions that agree at 0.5 and with hysteresis.

| Fixture | Frames | Model | Compute units | vs NeMo, mean / max | Agreement at 0.5 / hysteresis | One step vs PyTorch, max |
| --- | --- | --- | --- | --- | --- | --- |
| two-speakers | 777 | this conversion | `CPU_AND_NE` | 2.0e-4 / 0.035 | 99.97% / 100% | 0.026 |
| two-speakers | 777 | FluidInference | `CPU_AND_NE` | 2.0e-4 / 0.034 | 99.97% / 100% | 0.025 |
| two-speakers | 777 | this conversion | `CPU_ONLY` | 2.6e-4 / 0.085 | 100% / 99.94% | 0.019 |
| two-speakers | 777 | FluidInference | `CPU_ONLY` | 2.2e-4 / 0.040 | 100% / 99.97% | 0.021 |
| four-speakers | 941 | this conversion | `CPU_AND_NE` | 2.0e-3 / 0.140 | 99.92% / 99.55% | 0.143 |
| four-speakers | 941 | FluidInference | `CPU_AND_NE` | 2.0e-3 / 0.129 | 99.92% / 99.55% | 0.138 |
| four-speakers | 941 | this conversion | `CPU_ONLY` | 9.6e-4 / 0.114 | 99.97% / 99.95% | 0.062 |
| four-speakers | 941 | FluidInference | `CPU_ONLY` | 8.0e-4 / 0.083 | 99.97% / 100% | 0.032 |
| overlap | 455 | this conversion | `CPU_AND_NE` | 6.3e-4 / 0.040 | 100% / 100% | 0.027 |
| overlap | 455 | FluidInference | `CPU_AND_NE` | 3.7e-4 / 0.018 | 100% / 100% | 0.027 |
| overlap | 455 | this conversion | `CPU_ONLY` | 4.5e-4 / 0.030 | 100% / 100% | 0.017 |
| overlap | 455 | FluidInference | `CPU_ONLY` | 4.5e-4 / 0.046 | 100% / 100% | 0.013 |
| long | 3,838 | this conversion | `CPU_AND_NE` | 4.6e-3 / 0.639 | 99.54% / 99.61% | 0.264 |
| long | 3,838 | FluidInference | `CPU_AND_NE` | 3.8e-3 / 0.526 | 99.60% / 99.65% | 0.099 |
| long | 3,838 | this conversion | `CPU_ONLY` | 3.8e-3 / 0.567 | 99.64% / 99.65% | 0.043 |
| long | 3,838 | FluidInference | `CPU_ONLY` | 4.6e-3 / 0.569 | 99.44% / 99.64% | 0.019 |

This conversion against FluidInference's, same compute units:

| Fixture | `CPU_AND_NE` mean / max, agreement at 0.5 | `CPU_ONLY` mean / max, agreement at 0.5 |
| --- | --- | --- |
| two-speakers | 3.1e-5 / 0.018, 100% | 2.4e-4 / 0.073, 100% |
| four-speakers | 2.8e-4 / 0.046, 100% | 4.4e-4 / 0.045, 100% |
| overlap | 3.5e-4 / 0.023, 100% | 1.9e-4 / 0.016, 100% |
| long | 3.5e-3 / 0.550, 99.80% | 1.5e-3 / 0.313, 99.75% |

- The two models are equally close to NeMo; neither is closer on every fixture.
  On `long`, every model and compute-unit setting differs from NeMo, and from each other,
  by 0.3 to 0.6 at some frames; the speaker cache feeds each step's predictions back into
  later steps, so small differences accumulate over 307 s.
- NeMo 2.7.3 with PyTorch 2.7.0 reproduces the earlier references from NeMo 3.0.0 with
  PyTorch 2.14.1 (`Tools/sortformer-reference.py`) within 2.5e-6 on `four-speakers`
  and `overlap`.
- One Core ML call took 53 to 56 ms with either compute-unit setting and either model,
  measured with `coremltools` while other processes used the CPU.

### Runtime

With the converted model staged as a compiled model directory,
the package's opt-in tests pass without code changes:

```sh
ANYSPEECH_SORTFORMER_MODEL=<build/sortformer/SortformerLowLatency.mlmodelc> \
ANYSPEECH_DIARIZATION_AUDIO=/tmp/diar-fixtures/four-speakers.wav \
ANYSPEECH_SORTFORMER_REFERENCE=<build/reference/four-speakers> \
swift test --filter SortformerModelTests
```

The Swift pipeline's predictions differ from NeMo by 2.0e-3 mean and 0.140 maximum
with 99.92% agreement on `four-speakers`, and by 6.3e-4 and 0.040 with 100% agreement
on `overlap`, the same as the Python loop.

`hear evaluate` with `--diarize --diarizer-model` and Parakeet v3,
file mode, default compute units (`cpuAndNeuralEngine`).
DER uses 10 ms frames, includes overlap, and maps speakers one to one;
the reference boundaries come from an energy detector, so the collar-free DER counts
padding and short pauses as false alarms.

| Fixture | Speakers, reference / found | DER, 0.25 s collar: this conversion / FluidInference | DER, no collar: this conversion / FluidInference |
| --- | --- | --- | --- |
| two-speakers | 2 / 2 | 0.0% / 0.0% | 9.15% / 9.15% |
| four-speakers | 4 / 3 | 32.65% / 32.65% | 40.73% / 40.73% |
| overlap | 3 / 3 | 0.0% / 0.0% | 15.04% / 15.04% |
| long | 4 / 3 | 18.97% / 18.10% | 29.71% / 28.97% |

The first three fixtures give the same error components with both models.
On `long`, confusion is the same (39.21 s with the collar), and this conversion misses
2.26 s of speech where FluidInference's misses 0.33 s.
The four-speaker plans have three distinct voices for this model, as with NeMo itself.
Synthetic voices are an easy case; these numbers do not predict accuracy on recordings.

## License and attribution

The code in this directory is licensed under the Apache License, Version 2.0.
Parts of it are derived from FluidInference's mobius and follow NVIDIA NeMo;
see `NOTICE` and the header of each derived file.
Keep those notices in copies and derivatives.

The converted model is an adaptation of a model licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Its `README.md`, `conversion.json`, and Core ML metadata (`license`) contain this
attribution:

> Streaming Sortformer 4spk v2 by NVIDIA
> (https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2),
> licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
> Converted to Core ML for AnySpeechRecognizer: the pre-encoder, FastConformer encoder,
> and Transformer are exported for one streaming step with a fixed speaker cache and FIFO,
> and the encoder and Transformer weights are stored in float16.

Applications that ship or download the model must show the attribution to their users.

## Upload

Nothing in this directory uploads to Hugging Face.
`upload_dry_run.py` re-hashes every file against `conversion.json`,
checks the license and source records, prints the repository layout,
and prints the commands to publish the directory:

```sh
uv run python upload_dry_run.py
```

The repository has the output directory's layout at its root:

```
README.md
conversion.json
SortformerLowLatency.mlmodelc/{analytics/coremldata.bin, coremldata.bin,
    model0/{analytics/coremldata.bin, coremldata.bin, model.mil, weights/0-weight.bin},
    model1/{analytics/coremldata.bin, coremldata.bin, model.mil, weights/1-weight.bin}}
```

After review, a maintainer authenticated as `mattt` runs:

```sh
hf repo create mattt/diar-streaming-sortformer-4spk-v2-coreml --repo-type model
hf upload mattt/diar-streaming-sortformer-4spk-v2-coreml build/sortformer . --repo-type model \
    --commit-message "Add Streaming Sortformer 4spk v2 Core ML model"
```

Then record the commit hash from `hf models info <repository>` in the runtime manifest.
`Tools/make-sortformer-manifest.py` reads sizes and digests from the Hub;
for this repository, its `REPOSITORY`, `MODEL` (`SortformerLowLatency.mlmodelc`),
and `ATTRIBUTION` constants need the new values.
Do not change files in place after publication;
publish a fixed artifact under a new revision and update the manifest.
