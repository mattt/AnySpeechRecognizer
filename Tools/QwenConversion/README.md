# Qwen3-ASR conversion

This maintainer tool converts the Qwen team's Qwen3-ASR checkpoints to Core ML
for AnySpeechRecognizer's `QwenSpeechModel`, which transcribes files.
It also packages a verified conversion for Hugging Face
and checks the package before a maintainer uploads it.
The tool is not part of the Swift package products, and it never uploads.

## Source

| Variant | Checkpoint | Status |
| --- | --- | --- |
| `0.6b` | [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) at `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c` | Converted and verified |
| `1.7b` | [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) at `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` | Converted and verified |

The `-hf` repositories hold the transformers-native checkpoints
(`Qwen3ASRForConditionalGeneration`, bfloat16).
The converter loads them with transformers, in float32 by default,
and traces its own wrappers around the checkpoint's modules.
The 1.7B model has a 24-layer encoder (width 1,024)
and a 28-layer decoder (hidden size 2,048); the tensor interface is the same
except for that hidden size.
Revisions are defined in `qwen.py`.

## Outputs

`convert.py` writes one directory per variant, `build/<variant>/`:

| File | Contents |
| --- | --- |
| `AudioEncoder.mlmodelc` | Audio encoder and projector for one 8-second window. 187 MB for 0.6B, 319 MB for 1.7B. |
| `TextDecoder.mlmodelc` | Qwen3 text decoder for a block of 16 tokens, with a 1,024-token KV cache in Core ML states. 754 MB for 0.6B, 2,035 MB for 1.7B. |
| `vocab.json` | The byte-level BPE string of each token ID, `0` to `151935`. |
| `qwen3_asr_config.json` | Prompt token IDs, language prefills, special and skipped tokens, and audio settings. |
| `conversion.json` | Source, tool versions, settings, and the size and SHA-256 digest of every file. |
| `README.md` | Model card with the license and attribution. |

Weights are int8, linear symmetric, with one scale per output channel,
for tensors of 2,048 or more elements.
The decoder's token embedding table, which the output head shares,
and the rotary tables stay float16; see [Findings](#findings).
`--weights float16` keeps every weight in float16
(for 0.6B, a 373 MB encoder and a 1.19 GB decoder).

### Tensor interface

Both models are ML programs with an iOS 18 and macOS 15 deployment target,
which stateful models need.
Every shape is fixed.

| Model | Inputs | Outputs | States |
| --- | --- | --- | --- |
| Encoder | `mel` Float32 [1, 128, 800], `mel_length` Int32 [1] | `audio_embeddings` Float32 [1, 104, 1024] | — |
| Decoder | `input_ids` Int32 [1, 16], `audio_embeddings` Float32 [1, 16, 1024], `position_ids` Int32 [16], `causal_mask` Float32 [1, 1, 16, 1024] | `token_ids` Int32 [1, 16], `token_log_probs` Float32 [1, 16] | `key_cache_<layer>` and `value_cache_<layer>` Float16 [1, 8, 1024, 128], for layers 0 to 27 |

The encoder takes 128-bin log-mel features with frames after `mel_length` set to zero.
The checkpoint's encoder convolves each 100-frame chunk separately
(three stride-2 convolutions give `ceil(frames / 8)` tokens, 13 for a whole chunk)
and attends within windows of 8 chunks (`n_window_infer` 800 frames, 104 tokens),
so 8-second windows are independent,
and one fixed window reproduces the encoder for audio of any length.
For a partial window, the converted encoder masks the keys of padding tokens;
its output rows after the valid tokens are not used.

The decoder writes the keys and values of each input row at its `position_ids`
and attends over all 1,024 cache positions with `causal_mask`,
an additive mask that is 0 where a row may attend and -10,000 elsewhere.
Rows whose `input_ids` is the audio placeholder (`<|audio_pad|>`, 151676)
take their embedding from `audio_embeddings`.
`token_ids` holds the greedy next token after each row,
and `token_log_probs` its log probability.
A partial block repeats `<|endoftext|>` after the real tokens,
at the following positions;
later blocks overwrite those positions before any row may attend to them.

### Prompt

The prompt is the chat template with an empty system message,
one placeholder per audio embedding, and the assistant turn:

```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>…<|audio_end|><|im_end|>\n<|im_start|>assistant\n
```

The model answers `language <name><asr_text><transcript>`.
To force a language, the prompt ends with that prefill,
such as `language Chinese<asr_text>`,
and the model writes only the transcript.
`qwen3_asr_config.json` lists the prefill tokens for the 30 language codes
of transformers' `LANGUAGE_CODE_TO_NAME`.
The context and hotword system prompt that the model card describes is not supported,
because the runtime has no BPE encoder.

## Setup

Requirements:

- macOS on Apple silicon with Xcode, for `xcrun coremlcompiler`.
- [uv](https://docs.astral.sh/uv/).
- About 10 GB of free memory, and 4 GB of disk space for 0.6B or 9 GB for 1.7B.

```sh
cd Tools/QwenConversion
uv sync
```

The lock file pins Python 3.12, PyTorch 2.8.0, transformers 5.20.0,
coremltools 9.0, and NumPy 2.3.5.
transformers 5.20 imports `torch.distributed.checkpoint.hf_storage`,
which PyTorch 2.7 lacks,
so the tool uses PyTorch 2.8,
which coremltools 9.0 warns that it has not tested.
Set `QWEN_BUILD_DIR` to keep checkpoints, outputs, fixtures, and reports
outside `build/`.

## Convert

```sh
uv run python convert.py --variant 0.6b
```

```sh
uv run python convert.py --variant 1.7b --dtype float16
```

The first run downloads the checkpoint to `build/cache/` (1.6 GB for 0.6B, 4.1 GB for 1.7B).
`--dtype float16` traces the decoder from float16 weights,
which halves the conversion's memory;
the encoder still traces in float32,
because coremltools 9.0 fails on its float16 convolutions.
The checkpoint is bfloat16, so both types give the same float16 weights:
for 0.6B, both `weight.bin` files of each model had the same SHA-256 digest,
and verification gave the same numbers.
On an M4 Max with macOS 27.0.1, the 0.6B conversion in float32 took 86 s
and peaked at 9.6 GB (October 16, 2026),
and the 1.7B conversion with `--dtype float16` took 143 s and peaked at 10.4 GB
(October 17, 2026).
`--context` sets the KV cache length and `--block` the decoder block.

## Verify

```sh
uv run python verify.py --variant 0.6b --compute-units CPU_AND_GPU
uv run python verify.py --compute-units WRAPPER
uv run python verify.py --compute-units CPU_AND_GPU --audio clip.wav other.flac
```

The verifier renders eight fixtures with `say` into `build/fixtures/`:
English in two voices, a 25-second English clip that needs four encoder windows,
Mandarin, Japanese, Korean, Cantonese, and German.
`--audio` adds files.
For each fixture it computes transformers' features and checks:

1. **Encoder.** The Core ML audio embeddings against `get_audio_features`,
   over the valid tokens: maximum and mean absolute error, and token cosine similarity.
2. **Decoder.** With the reference audio embeddings,
   the Core ML decoder's greedy token after each prefix of `generate()`'s output
   (teacher forcing).
3. **Transcripts.** Greedy decoding with the Core ML encoder and decoder
   against `generate()`, token for token.

`WRAPPER` runs the traced PyTorch wrappers instead of Core ML,
which separates wrapper error from conversion error.
The report is `build/reports/verify-<directory>.json`.
`mel_reference.py` writes the transformers features that the Swift
`QwenMelFrontendTests` compare with.

### Results

Observed on October 16 (0.6B) and October 17, 2026 (1.7B),
on an M4 Max with macOS 27.0.1, with `CPU_AND_GPU`.
The natural clips are the first four FLEURS test files in Mandarin, Japanese, and Korean,
and four LibriSpeech test-other utterances.
The mean absolute value of the reference embeddings is 0.015 to 0.017.

| Size | Artifacts | Fixtures | Encoder max / mean abs error | Encoder mean / min token cosine | Teacher-forced tokens | Identical transcripts |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6B | PyTorch wrappers | 8 synthetic | < 0.000001 | 1.00000 / 1.0000 | 311 / 311 | 8 / 8 |
| 0.6B | int8 | 8 synthetic | 0.023 / 0.00059 | 0.99901 / 0.958 | 311 / 311 | 8 / 8 |
| 0.6B | int8 | 16 natural | 0.101 / 0.00062 | 0.99829 / 0.512 | 559 / 562 | 13 / 16 |
| 0.6B | float16 | 8 synthetic | 0.0033 / 0.00002 | 1.00000 / 0.9988 | 311 / 311 | 8 / 8 |
| 0.6B | float16 | 16 natural | 0.017 / 0.00004 | 0.99999 / 0.981 | 562 / 562 | 15 / 16 |
| 1.7B | PyTorch wrappers | 8 synthetic | < 0.000001 | 1.00000 / 1.0000 | 313 / 313 | 8 / 8 |
| 1.7B | int8 | 8 synthetic | 0.059 / 0.00046 | 0.99899 / 0.710 | 312 / 313 | 8 / 8 |
| 1.7B | int8 | 16 natural | 0.076 / 0.00060 | 0.99892 / 0.718 | 558 / 562 | 14 / 16 |

The int8 differences are near-ties:
a Japanese particle, "1만" for "일만" in Korean,
and commas for em dashes in English.
The float16 difference is in one Japanese clip,
where the encoder's float16 output changes a word that the decoder then follows.
Both 1.7B differences are word spacing in Korean
("마무리되었으며" for "마무리 되었으며", "북발트해" for "북 발트해").
The Swift runtime, release build, wrote the same transcripts as `generate()`
for all eight synthetic fixtures.

### Reference runs

`reference_runs.py` transcribes a subset of `Tools/natural-speech-eval.py`
with the transformers model and writes runs that its `score` command reads:

```sh
uv run python reference_runs.py --root /tmp/natural-eval --set fleurs-zh --variant 1.7b --threads 4
```

On October 16, 2026, the Core ML 0.6B runtime and the transformers 0.6B model
had error rates within 0.32 points of each other
on FLEURS Mandarin, Japanese, and Korean and on LibriSpeech utterances;
see the [README](../../README.md#accuracy-and-performance) for the table.

## Findings

These are from macOS 27.0.1 (26A434), Xcode 27.1, and coremltools 9.0.

- **Fixed shapes.** A decoder with flexible token and cache lengths (`ct.RangeDim`)
  and states, the pattern of Apple's stateful Llama example,
  returned wrong tokens on the GPU from the third layer on,
  and an earlier variant that reshaped grouped query heads failed an MPS assertion
  (`K dimension is inconsistently defined`).
  Without states, or with states and fixed shapes, the same layers matched PyTorch.
  The converter therefore uses a fixed block,
  writes the cache with a scatter at `position_ids`,
  gathers rotary rows from float32 tables,
  and attends over the whole cache.
  coremltools 9.0 cannot convert `index_copy_`,
  and an in-place `scatter_` on a float16 state fails to convert,
  so the cache update is a functional scatter on the state cast to the compute type.
- **Traced slice updates.** Assigning a tensor to a state slice with broadcasting
  traced a reshape to the example's length, and Core ML then failed to load the model
  (error -14). The update must have the slice's rank.
- **Embedding table.** With the shared token embedding and output head quantized to int8,
  the first token after the prompt was wrong for every fixture.
  The other decoder weights quantize without visible harm.
- **Float16 overflow.** The decoder's residual stream reaches about 5,800 on the first token
  from the third layer on (dimension 35).
  Squaring such values in RMSNorm overflows float16,
  so the wrapper divides each row by its largest magnitude first.
- **Neural Engine.** The decoder runs on the GPU; see [Neural Engine](#neural-engine).
  The encoder loads and is about three times faster on the Neural Engine
  (7.4 ms per window against 22.6 ms on the GPU),
  but some tokens deviate (minimum cosine 0.26; 0.23 with `CPU_ONLY`),
  so the runtime runs both models on the GPU.
- **CPU.** In the verifier, which also holds the transformers model,
  `CPU_ONLY` decoding crashed the process with a segmentation fault
  in the first prediction. A standalone process decoded on the CPU without crashing.
  The cause was not investigated.
- **GPU.** Once, the Swift integration tests with the 1.7B model, which load two decoders
  in one process, stopped at an MPSGraph assertion
  (`shape.count = 0 != strides.count = 2`).
  Four repetitions with each size passed,
  and none of 1,866 evaluation processes failed.
  The cause was not investigated.

### Neural Engine

On October 17, 2026, probes of 0.6B decoder variants (block 16, context 1,024, float16)
asked Core ML for a compute plan with `CPU_AND_NE` (`MLComputePlan`).
Constructing an `MLModel` is not a test:
coremltools logs a failed load and returns a model object anyway.

| Decoder variant | Plan with `CPU_AND_NE` |
| --- | --- |
| 1 layer, with the embedding and output head, scatter cache | Builds |
| 28 layers, embedding and head removed, no cache | Builds; every layer on the Neural Engine |
| 28 layers, embedding and head removed, one-hot cache, 56 states | Error -14 |
| The same with a 512-token cache (half the state memory) | Error -14 |
| The same with 2 states that hold all layers | Error -14 |
| 24 layers, embedding and head removed, one-hot cache, 48 states | Builds |
| 21 or 14 layers, embedding and head removed, one-hot cache | Builds |

So error -14 is not caused by the embedding table or the output head,
nor by the number or the size of the states.
It appears when a stateful program holds more than about 24 of these layers
(roughly 3,000 operations); the cause inside Core ML is not documented.
Two 14-layer models, as the aufklarer conversion uses, build.
But every cache update was slow on the Neural Engine.
Time per prediction of one 14-layer half, block 16:

| Cache update | Neural Engine | GPU |
| --- | --- | --- |
| None (attends to the block only) | 4.6 ms | 8.1 ms |
| One-hot rewrite of the whole cache | 20.1 ms | 9.0 ms |
| Functional scatter (runs on the CPU) | 19.2 ms | — |
| Slice update at `position_ids[0]` (runs on the CPU) | 18.0 ms | 20.4 ms |
| Cache as an input, new rows as outputs, no states | 29.8 ms | 21.9 ms |

With states, Core ML runs `read_state`, `write_state`, `scatter_along_axis`,
and `slice_update` on the CPU between Neural Engine segments of each layer,
except for the one-hot update, which keeps them on the Neural Engine
but rewrites both 2 MB states of each layer in every prediction.
Two halves would take at least 36 ms per prediction against about 25 ms
for the 28-layer decoder on the GPU,
so the converter keeps one decoder for the GPU.
These probes ran while other jobs used the CPU and GPU.

### Latency and memory

Other jobs used the GPU and memory of this Mac during these measurements,
so treat them as rough.
One decoder prediction took 25 ms on the GPU with blocks of 16 tokens,
20 ms with blocks of 4, and 32 to 39 ms with blocks of 1;
22 ms of the 16-token prediction remained without cache writes.
The time hardly depends on the block,
so it is mostly per-operation overhead, not arithmetic.
Blocks of 16 balance prefill (about 35 predictions for a 40-second segment)
against decoding (one prediction per token).

With the Swift runtime, a 309-second LibriSpeech file took 51 s with int8 weights
(peak resident memory 1.74 GB) and 46 s with float16 weights (3.14 GB).
On October 17, 2026, over FLEURS and LibriSpeech utterances, the real-time factor was
0.22 to 0.32 for 0.6B and 0.30 to 0.36 for 1.7B, with peaks of 1.7 GB and 4.4 GB.
The 1.7B decoder is 2.7 times as large on disk but took only 1.1 to 1.4 times as long,
which fits per-operation overhead.
Another speech pipeline used the GPU during most of these runs.

## Package and upload

The 1.7B conversion is packaged for the public repository `mattt/qwen3-asr-1.7b-coreml`,
and the 0.6B conversion for `mattt/qwen3-asr-0.6b-coreml`.

```sh
uv run python package.py --variant 1.7b --trace-dtype float16 --output /tmp/qwen-1.7b-publish \
    --verification-note 'The two differing transcripts differ only in word spacing in Korean ("마무리되었으며" for "마무리 되었으며", "북발트해" for "북 발트해").'
uv run python upload_dry_run.py --variant 1.7b --package /tmp/qwen-1.7b-publish

uv run python package.py --variant 0.6b --trace-dtype float32 --output /tmp/qwen-0.6b-publish \
    --verification-note 'The three differing transcripts differ in near-tied words: in Japanese, "カタルーニャを好み" for "カタルーニャがお好み" and "カタルーニャと" for "カタルーニャ語と"; in Korean, "일만" for "1만"; and in English, commas for dashes.'
uv run python upload_dry_run.py --variant 0.6b --package /tmp/qwen-0.6b-publish
```

`package.py` copies the compiled models, `vocab.json`, and `qwen3_asr_config.json`
from `build/<variant>` after it checks their sizes and digests against that directory's
`conversion.json`; the files are byte for byte the verified ones.
It adds:

| File | Contents |
| --- | --- |
| `LICENSE` | The Apache License, Version 2.0, as the Qwen3-ASR repository distributes it ([`LICENSE`](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/LICENSE) at `c17a131`, with the appendix filled in as "Copyright 2026 Alibaba Cloud"). `LICENSE-Qwen3-ASR` in this directory is a copy; both scripts check its SHA-256 digest. The upstream repositories have no NOTICE file. |
| `NOTICE` | The source and revision, the credit to the Qwen team, Alibaba Cloud, and the changes, as section 4(b) of the License requires. |
| `README.md` | The model card: license metadata, languages, use, limitations, size and resources, evaluation, and verification. |
| `conversion.json` | The conversion record, with the size and SHA-256 digest of every file except itself and the card. |

The card's evaluation comes from `evaluation/<variant>.json`,
whose fields follow the `score` command's `results/asr.json`.
`evaluation/0.6b.json` and `evaluation/1.7b.json` hold the counts that the command printed
for the October 16 and 17, 2026 runs;
the run files were in `/tmp` and no longer exist.
To measure again, run `Tools/natural-speech-eval.py` with the system `qwen-0.6b-file`
or `qwen-1.7b-file` and pass the new `results/asr.json` to `package.py --evaluation`.
The verification table comes from `build/reports/verify-<variant>-wrapper.json`
and `verify-<variant>-natural.json`.
`--trace-dtype` records how the artifacts were traced,
because `convert.py` began to record it after both conversions.

`upload_dry_run.py` re-hashes every listed file, rejects unlisted files,
compares the model files with `build/<variant>/conversion.json`,
checks the license digest, the notice's source, credit, and changes,
and the card's metadata and sections, then prints the `hf` commands for a maintainer to run.

After the upload, pin the runtime's manifest to the uploaded commit
from the repository root.
Pass every published size, because the generator replaces the whole file;
the dry run prints the command with the sizes that are already pinned.

```sh
hf models info mattt/qwen3-asr-0.6b-coreml
uv run Tools/make-qwen-manifest.py 0.6b@<commit> 1.7b@<commit> \
    > Sources/AnySpeechRecognizer/Qwen/QwenModelManifest+BuiltIn.swift
```

The generator reads sizes and digests from the Hugging Face API without downloading weights.
It pins the compiled models, the vocabulary, the configuration, `LICENSE`, and `NOTICE`.
With the manifests, `QwenSpeechModel(variant: .small)` and `QwenSpeechModel(variant: .large)`
install through `AssetInventory`, which verifies every file, as Parakeet does.
Without arguments, the generator writes a table without manifests,
which is the state before the upload.

## License and attribution

The code in this directory is licensed under the Apache License, Version 2.0.
The wrappers in `qwen.py` reimplement forward passes of transformers'
`Qwen3ASRForConditionalGeneration`, also licensed under the Apache License, Version 2.0.

Qwen3-ASR is licensed under the Apache License, Version 2.0.
Each output's `README.md`, `conversion.json`, and model metadata contain this attribution
(here for 0.6B):

> Qwen3-ASR 0.6B by the Qwen team, Alibaba Cloud
> (https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf),
> licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
> Converted to Core ML for AnySpeechRecognizer: the model is split into an audio encoder
> and a text decoder with a KV cache, weights are stored in float16,
> and weights of large layers are quantized to int8.
