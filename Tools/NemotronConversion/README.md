# Nemotron conversion

This maintainer tool converts NVIDIA's
[Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
to cache-aware Core ML models for AnySpeechRecognizer's `NemotronSpeechModel`.
The runtime is a prototype:
no repository publishes the converted models,
so the runtime loads a directory that this tool writes.
The tool is not part of the Swift package products.

| | |
| --- | --- |
| Source | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) at `ea30d66debe3740a08b573244286791d423d6b3e`, `nemotron-3.5-asr-streaming-0.6b.nemo` |
| License | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) |
| Model | NeMo `EncDecRNNTBPEModelWithPrompt`: a 24-layer cache-aware FastConformer encoder, a language prompt projection, a two-layer LSTM prediction network, and an RNN-T joint network over 13,087 tokens and the blank |

The revision, file names, streaming constants, and tensor interface
are defined in `nemotron.py`.

## The model

The encoder uses `chunked_limited` self-attention with `att_context_size` [56, R]:
a chunk of R + 1 encoder frames (80 ms each) attends to itself
and to the 56 frames before it, and every convolution is causal.
Cache-aware streaming keeps, for each of the 24 layers,
the last 56 inputs of the self-attention (`cache_last_channel`)
and the last 8 inputs of the depthwise convolution (`cache_last_time`),
so each chunk is encoded once.
The chunk size is the encoder's latency.
NeMo trained R = 0, 3, 6, and 13 (80, 320, 560, and 1,120 ms);
the model card also reports R = 1 (160 ms).

The language prompt is a one-hot vector over 128 slots,
concatenated with each 1,024-dimension encoder frame and projected back to 1,024 dimensions
by a two-layer MLP (`prompt_kernel`).
`prompts.json` maps 120 locale keys and `auto` to slots.
With `auto`, and sometimes with a given language,
the model writes a language tag token such as `<en-US>` after terminal punctuation.
The 40 tag pieces have the form `<xx-XX>`;
the Swift vocabulary treats them as control pieces, so results omit them.

The preprocessor does not normalize features (`normalize: NA`),
so a mel frame depends only on its own 25 ms of audio,
and streaming features equal those of the whole recording.

## Outputs

`convert.py` writes `build/nemotron/`:

| File | Contents |
| --- | --- |
| `Encoder.mlmodelc` | Multifunction encoder step with the prompt projection. One function per chunk size, `chunk_80ms` to `chunk_1120ms`, sharing int8 weights (linear symmetric, one scale per output channel). |
| `Decoder.mlmodelc` | Prediction network for one token with explicit LSTM state. Float16 weights. |
| `JointDecision.mlmodelc` | Joint network and greedy decision for one encoder frame. Float16 weights. |
| `vocab.json` | Token ID (`"0"` to `"13086"`) to SentencePiece piece, with the word-boundary marker written as a space, as for Parakeet. |
| `prompts.json` | Locale keys and `auto` to prompt slots, from the checkpoint. |
| `conversion.json` | Source, tool versions, settings, and the size and SHA-256 digest of every file. |
| `README.md` | A model card with the license and attribution. |

All models are ML programs with an iOS 18 and macOS 15 deployment target,
which multifunction models need.

### Tensor interface

Each encoder function, for a chunk of C = R + 1 encoder frames:

| Name | Type | Shape |
| --- | --- | --- |
| input `mel` | Float32 | [1, 128, 9 + 8C] |
| input `mel_length` | Int32 | [1] |
| input `cache_channel` | Float32 | [24, 1, 56, 1024] |
| input `cache_time` | Float32 | [24, 1, 1024, 8] |
| input `cache_length` | Int32 | [1] |
| input `prompt` | Float32 | [1, 128] |
| output `encoder` | Float32 | [1, 1024, C] |
| output `encoder_length` | Int32 | [1] |
| output `cache_channel_out` | Float32 | [24, 1, 56, 1024] |
| output `cache_time_out` | Float32 | [24, 1, 1024, 8] |
| output `cache_length_out` | Int32 | [1] |

The prediction network has Parakeet's interface
(`targets`, `target_length`, `h_in`, `c_in` to `decoder`, `h_out`, `c_out`, with [2, 1, 640] states).
The joint decision takes `encoder_step` [1, 1024, 1] and `decoder_step` [1, 640, 1]
and returns `token_id` and `token_prob`, the argmax over the tokens and the blank
and its softmax probability. It has no duration output: RNN-T decoding emits up to
10 tokens at a frame and moves to the next frame on the blank.

### Streaming steps

Each step follows NeMo's `CacheAwareStreamingAudioBuffer` with `pad_and_drop_preencoded=True`.
The encoder input is the 9 mel frames before the chunk, zeros before the recording,
followed by the chunk's 8C mel frames.
The causal subsampling turns the 9 frames into 2 encoder frames, which the step drops.
Every step, including the first, has the same shape, so one function serves the whole stream.
NeMo's default instead encodes a first chunk of 1 + 8R frames without the 9 frames,
which needs a second model; on the fixtures, the two modes differ only in a word or a period
at the start or end of a clip.

The last chunk of a recording can be shorter. Its mel frames are followed by zeros,
and `mel_length` gives the valid frames, which the subsampling and attention masks use.
`encoder_length` then reports ⌈valid mel frames / 8⌉ encoder frames.

The caches pass from each step's outputs to the next step's inputs.
`cache_length` counts the valid frames of the attention cache, up to 56;
the attention masks the others.

The conversion traces NeMo's own `ConformerEncoder.cache_aware_stream_step`
and `prompt_kernel` for each chunk size,
converts each trace with float16 compute precision and float32 lengths and masks,
quantizes its weights to int8,
and combines the packages with `coremltools.utils.save_multifunction`.
The functions share every weight except the positional projections that the trace
folds into constants for each input length.
With two functions, the combined package was 570 MiB and each single-function package 566 MiB.

Explicit cache inputs and outputs work on macOS 15 and iOS 18 and later and keep
each step a pure function, which the verifier needs.
The Swift engine keeps two sets of cache arrays and lets Core ML write each step's caches
into the other set through output backings, so the caches do not pass through Swift.
A stateful model (`MLState`) could keep the caches inside Core ML and avoid the copies;
it was not tried.

## Setup

Requirements:

- macOS on Apple silicon with Xcode or the Command Line Tools (for `xcrun coremlcompiler`).
- [uv](https://docs.astral.sh/uv/).
- About 2.4 GB of disk for the checkpoint in `build/cache/`,
  about 1.2 GB for each chunk size's intermediate package,
  and about 650 MB for the output.

```sh
cd Tools/NemotronConversion
uv sync
```

The lock file pins the Parakeet tool's environment:
Python 3.12, PyTorch 2.7.0, NeMo 3.0.0, coremltools 9.0, and NumPy 2.3.5.
NeMo 3.0.0 restores the checkpoint as `EncDecRNNTBPEModelWithPrompt`.
Its `transcribe()` fails in the Lhotse prompt dataloader (`Unknown prompt key: 'None'`),
so the tool computes features and decodes without it.

Set `NEMOTRON_BUILD_DIR` to keep checkpoints, outputs, fixtures, and reports
outside `build/`.

## Convert

```sh
uv run python convert.py
uv run python convert.py --chunks 160 560 --output-dir build/nemotron-160-560
```

The converter downloads the checkpoint, checks its SHA-256 digest against the one that
Hugging Face records, and exports every chunk size by default.
Each chunk size takes about one minute on an M4 Max; combining and compiling take about
15 seconds more.

## Verify

```sh
uv run python verify.py --artifacts build/nemotron --compute-units CPU_AND_NE CPU_ONLY --fast-reference
uv run python verify.py --artifacts build/nemotron --chunks 160 560 --nemo
```

The verifier renders six fixtures with `say` into `build/fixtures/`:
English in two voices, German, French, Japanese, and a 36-second English clip,
long enough for the attention cache to fill and roll forward,
with automatic language detection.
`--audio file=locale` adds recordings.

For each encoder function and fixture, it compares:

1. **Interface.** The names, types, and shapes of the function's outputs.
2. **Encoder steps.** Each Core ML step, given the PyTorch reference's caches,
   against the float32 PyTorch step: the encoder frames, and the valid frames
   of the attention cache it returns.
3. **Streaming.** Core ML steps that carry their own caches through the recording,
   against the reference.
4. **Decoder and joint.** The Core ML prediction network along the reference's tokens,
   and the Core ML joint decision on every joint input of the reference's greedy decoding.
5. **Transcripts.** Greedy decoding over the Core ML models against the reference.
   With `--nemo`, also the reference against NeMo's own `conformer_stream_step` loop.

The reference runs NeMo's modules in Python (`nemotron.stream`).
PyTorch 2.7 runs NeMo's depthwise convolution (1,024 groups) on the CPU
as one convolution per group, about 1.7 s per encoder step.
`--fast-reference` computes it with one batched product instead;
over 12 steps of the long clip, its encoder frames and caches differ from NeMo's
by at most 8.1e-6.

The verifier runs all PyTorch work for every chunk size before it loads a Core ML model.
With Core ML loaded first, PyTorch's CPU `Linear` crashed with a segmentation fault
in four of six runs on macOS 27.0.1.
The report is `build/reports/verify.json`.

### Results

Observed on October 16, 2026, on an M4 Max with macOS 27.0.1 (26A434),
with the pinned environment, while other jobs used the machine.

**Reference against NeMo.** A standalone run of the reference loop, with NeMo's exact
depthwise convolution, gave the same tokens as NeMo's `conformer_stream_step`
with `CacheAwareStreamingAudioBuffer` (`pad_and_drop_preencoded=True`)
for English, German, Japanese, and French (with `auto`) fixtures at 160 and 560 ms,
8 of 8 cases.
`verify.py --nemo --fast-reference` on the six fixtures agreed in 11 of 12 cases;
in the twelfth (`en-daniel`, 160 ms), NeMo's loop omits the clip's final period.

**Core ML against the reference**, six fixtures (about 1,200 joint decisions per row),
from `--fast-reference` runs of the two-function and the five-function encoders:

| Chunk | Compute units | Step max / mean abs error | Step cosine, mean / min | Streaming cosine, mean / min | Decoder max abs | Joint agreement | Fixtures with the same tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 80 ms | `CPU_AND_NE` | 0.282 / 0.00321 | 0.99727 / 0.7758 | 0.99098 / 0.5568 | 0.0171 | 1191 / 1192 | 5 / 6 |
| 160 ms | `CPU_AND_NE` | 5.549 / 0.00370 | 0.99675 / 0.7660 | 0.99274 / 0.7021 | 0.0171 | 1194 / 1195 | 6 / 6 |
| 160 ms | `CPU_ONLY` | 0.406 / 0.00317 | 0.99740 / 0.7634 | 0.99503 / 0.7494 | 0.0171 | 1194 / 1195 | 6 / 6 |
| 320 ms | `CPU_AND_NE` | 0.359 / 0.00337 | 0.99691 / 0.7079 | 0.99387 / 0.6116 | 0.0176 | 1199 / 1200 | 5 / 6 |
| 560 ms | `CPU_AND_NE` | 0.299 / 0.00361 | 0.99670 / 0.8037 | 0.99215 / 0.4160 | 0.0136 | 1196 / 1198 | 4 / 6 |
| 560 ms | `CPU_ONLY` | 0.328 / 0.00341 | 0.99752 / 0.9218 | 0.99620 / 0.6494 | 0.0136 | 1196 / 1198 | 5 / 6 |
| 1120 ms | `CPU_AND_NE` | 0.504 / 0.00393 | 0.99529 / 0.6347 | 0.99457 / 0.6494 | 0.0163 | 1197 / 1198 | 6 / 6 |

- Cosine similarity is per encoder frame. The minimums are single frames;
  the 160 ms maximum error of 5.5 is one frame of the long clip on the Neural Engine,
  whose tokens still match.
- Where the tokens differ, the normalized transcripts are identical
  (a period or a language tag differs), except at 560 ms, where Core ML drops the first
  word, "The", of a clip whose speech starts at its first sample, with both compute units.
  NeMo's own streaming loop flips the same "The" between its two first-chunk modes.

**On natural speech.** The PyTorch reference and the Swift runtime (live mode, 560 ms)
on the first files of two evaluation sets (see [Evaluation](#evaluation)):

| Set | Files | Reference WER | Swift WER | Same normalized text | Swift against the reference |
| --- | --- | --- | --- | --- | --- |
| LibriSpeech test-other utterances | 60 | 8.62% | 9.16% | 46 / 60 | 1.79% |
| FLEURS German | 50 | 11.17% | 11.50% | 39 / 50 | 1.82% |

The conversion (int8 weights, float16 compute, the Neural Engine) costs about 0.3 to 0.5
points of WER on these files; the rest of the gap to Parakeet is the model's.

### Latency

`latency.swift` loads each function and times `prediction(from:)` with zero caches,
passing the caches through output backings as the runtime does:

```sh
xcrun swiftc -O latency.swift -o build/latency
build/latency build/nemotron cpuAndNeuralEngine 200
```

With `cpuAndNeuralEngine`, while evaluations ran on the same machine:

| Model | Median / p95 per call | Encoder time per second of audio |
| --- | --- | --- |
| `chunk_160ms` | 9.1 / 27.1 ms | 57 ms |
| `chunk_560ms` | 10.5 / 12.6 ms | 19 ms |
| Decoder | 0.39 / 1.20 ms | |
| Joint decision | 0.58 / 2.09 ms | |

A call costs about the same at every chunk size, so small chunks cost more per second of
audio. Greedy RNN-T decoding calls the joint decision at least once per 80 ms frame,
12.5 times per second of audio, and about once more per token.
The first load of each function took 10 to 26 s, for the Neural Engine compilation.
With `--cache-dtype float16`, which declares float16 cache inputs and outputs,
a call took 8.4 ms instead of 9.4 ms at 160 ms and about the same at 560 ms,
within the noise of the shared machine; the default stays float32.

## Evaluation

`Tools/natural-speech-eval.py` ran the release `hear` CLI in live mode
(`--live`, 100 ms blocks) on the README sets, FLEURS German and French,
and the first 100 FLEURS Japanese and Mandarin test files, scored by character error rate.
Nemotron received the set's language (`en-US` for LibriSpeech and AMI),
except in the `auto` row. Parakeet v3 and Ultra ran at the same commit on the same files.
Observed on October 16, 2026, on an M4 Max with macOS 27.0.1, with other evaluations
and conversions running at the same time.

| Set | v3 live | Ultra live | Nemotron 160 ms | Nemotron 560 ms | Nemotron 1,120 ms | Nemotron 560 ms, `auto` |
| --- | --- | --- | --- | --- | --- | --- |
| LibriSpeech test-clean, utterances | 3.28% | 2.51% | 3.87% | 3.37% | 3.54% | |
| LibriSpeech test-other, utterances | 4.19% | 3.92% | 8.11% | 7.15% | 6.62% | |
| LibriSpeech test-clean, 3–8 min files | 4.73% | 2.65% | 3.03% | 2.53% | 2.34% | |
| LibriSpeech test-other, 3–8 min files | 4.29% | 3.28% | 5.10% | 4.60% | 4.35% | |
| AMI meetings, headset mix | 23.95% | 23.56% | 31.33% | 31.16% | | |
| FLEURS German | 7.30% | 4.39% | 11.65% | 10.85% | 10.47% | 10.72% |
| FLEURS French | 9.13% | 5.82% | 14.52% | 13.14% | 12.29% | 13.17% |
| FLEURS Japanese (CER) | | | 16.76% | 15.37% | | 15.91% |
| FLEURS Mandarin (CER) | | | 19.51% | 19.63% | | 19.69% |

- Word error rates are computed after lowercasing and removing punctuation,
  without number or spelling normalization, so they are higher than the model card's.
- On AMI, Nemotron deleted 3,605 reference words at 560 ms, against 2,408 for v3,
  and had 88 runs of 8 or more deleted words, against 27. It drops more backchannels and
  fillers ("yeah" 493 times, against 359; "um" 108 times, against 32), but with Whisper's
  English normalizer, which removes fillers, it is still 28.3% against 21.4%.
- Automatic detection changed the error rate by at most 0.54 points (Japanese)
  against the given language.
- Analysis took 0.09 to 0.15 of real time at 160 ms, 0.035 to 0.06 at 560 ms,
  0.025 to 0.04 at 1,120 ms, and 0.04 to 0.06 for Parakeet live.
  Other jobs shared the machine, so these figures are unreliable.
- The evaluation binary predates two runtime changes: finalizations now encode the
  last 12.5 ms of audio provisionally, and the executor skips updates without new tokens.
  The first affects at most the last two mel frames of each file.

### Latency in real time

With `--real-time`, the CLI delivers each 100 ms block when its audio would have been
captured. Latency is the time from the end of a word's audio until the word first appears
in any result, volatile or final, over the first 25 LibriSpeech test-clean utterances.
Word ends come from the forced alignments of Lugosch et al.
Against Nemotron's own word end times, its median latencies were 40 to 300 ms lower,
because it emits a word after its audio ends.

| System | Median / p90 word latency | First word, median |
| --- | --- | --- |
| Parakeet v3 live | 729 / 1,297 ms | 427 ms |
| Nemotron 80 ms | 380 / 554 ms | 487 ms |
| Nemotron 160 ms | 370 / 552 ms | 435 ms |
| Nemotron 560 ms | 479 / 738 ms | 635 ms |
| Nemotron 1,120 ms | 667 / 1,176 ms | 843 ms |

Nemotron's words are final when they appear. Parakeet's first appear in volatile results,
which later windows can revise, and become final after about 2 s of right context.

## License and attribution

The code in this directory is licensed under the Apache License, Version 2.0.

The converted models are adaptations of a model licensed under the
[OpenMDW License 1.1](https://openmdw.ai/license/1-1/).
Each output's `README.md` and `conversion.json` contain this attribution,
and the Core ML model metadata (`license`) repeats it:

> Nemotron 3.5 ASR Streaming 0.6B by NVIDIA
> (https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
> licensed under the OpenMDW License 1.1 (https://openmdw.ai/license/1-1/).
> Converted to Core ML for AnySpeechRecognizer: the model is split into a cache-aware
> encoder step with the language prompt projection, a prediction network, and a joint
> decision; weights are stored in float16, and encoder weights are quantized to int8.

Nothing in this directory uploads to Hugging Face.
