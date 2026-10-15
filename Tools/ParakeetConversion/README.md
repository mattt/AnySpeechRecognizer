# Parakeet conversion

This maintainer tool converts NVIDIA's Parakeet TDT checkpoints to Core ML
for AnySpeechRecognizer's Parakeet runtime.
It produces package-owned replacements for the files that the runtime loads
from FluidInference's Hugging Face repositories,
with the same tensor interface.
The tool is not part of the Swift package products.

## Variants

| Variant | Weights | Architecture, tokenizer, and preprocessor | Proposed repository |
| --- | --- | --- | --- |
| `v3` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) at `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | Same | `mattt/parakeet-tdt-0.6b-v3-coreml` |
| `ultra` | [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) at `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b` | NVIDIA v3, same revision | `mattt/parakeet-ultra-coreml` |

Moondream publishes Parakeet Ultra as a Hugging Face transformers checkpoint
(`model.safetensors`, float16 and float32 tensors), not as a `.nemo` file.
The converter loads the NVIDIA v3 `.nemo` checkpoint and replaces every model
parameter with Ultra's weights.
It drops Ultra's `vad_head` tensors and checks that Ultra's `tokenizer.json`
has the same pieces as the v3 tokenizer.

Revisions, comparison artifacts, and attribution text are defined in
`parakeet.py`.
To move to a new upstream revision, change it there and run the conversion
and verification again.

## Outputs

`convert.py` writes one directory per variant, `build/<variant>/`:

| File | Contents |
| --- | --- |
| `Encoder.mlmodelc` | Encoder for one 15-second window. Weights are int8, linear symmetric, one scale per output channel. |
| `Encoder<seconds>s.mlmodelc` | Optional, with `--encoder-windows`: the same encoder for a shorter fixed window, such as `Encoder8s.mlmodelc`. Same trained weights and encoding. |
| `Decoder.mlmodelc` | Prediction network for one step with explicit LSTM state. Float16 weights. |
| `JointDecisionv3.mlmodelc` | Joint network and greedy TDT decision for one encoder frame. Float16 weights. |
| `parakeet_vocab.json` | Token ID (`"0"` to `"8191"`) to SentencePiece piece. |
| `conversion.json` | Sources, tool versions, settings, and the size and SHA-256 digest of every file. |
| `README.md` | Hugging Face model card with the license and attribution. |

The file names match the layout of FluidInference's `parakeet-ultra-coreml`
repository, which the package's Parakeet runtime loads.
`Encoder.mlmodelc` has the same encoding as FluidInference's v3
`Encoder_v2.mlmodelc`, not the 6-bit palettized v3 `Encoder.mlmodelc`
([FluidAudio#760](https://github.com/FluidInference/FluidAudio/issues/760)).
The name `JointDecisionv3` keeps the top-K interface distinct from
FluidInference's older `JointDecision.mlmodelc`, which has no top-K outputs.

There is no preprocessor model.
The Swift runtime computes mel features itself.

### Tensor interface

All models are ML programs with an iOS 17 and macOS 14 deployment target.

| Model | Inputs | Outputs |
| --- | --- | --- |
| Encoder | `mel` Float32 [1, 128, 1501], `mel_length` Int32 [1] | `encoder` Float32 [1, 1024, 188], `encoder_length` Int32 [1] |
| Encoder, W-second window | `mel` Float32 [1, 128, 100 W + 1], `mel_length` Int32 [1] | `encoder` Float32 [1, 1024, F], `encoder_length` Int32 [1] |
| Decoder | `targets` Int32 [1, 1], `target_length` Int32 [1], `h_in` and `c_in` Float32 [2, 1, 640] | `decoder` Float32 [1, 640, 1], `h_out` and `c_out` Float32 [2, 1, 640] |
| Joint decision | `encoder_step` Float32 [1, 1024, 1], `decoder_step` Float32 [1, 640, 1] | `token_id` Int32 [1, 1, 1], `token_prob` Float32 [1, 1, 1], `duration` Int32 [1, 1, 1], `top_k_ids` Int32 [1, 1, 1, 64], `top_k_logits` Float32 [1, 1, 1, 64] |

The joint decision takes the argmax over the 8,192 tokens and the blank (ID 8192),
returns that token's softmax probability,
and returns the argmax index over the five TDT durations (0, 1, 2, 3, 4 frames).
`top_k_ids` and `top_k_logits` are the 64 highest token logits, in descending order.

A W-second encoder takes `floor(16000 W / 160) + 1` mel frames.
Each of the three stride-2 subsampling convolutions maps L frames to
`floor((L - 1) / 2) + 1`, which gives F encoder frames of 80 ms.
The last frame covers only part of an 80 ms step.

`mel_length` follows NeMo 2.5 and later:
N valid samples give `floor(N / 160)` valid frames, so a full 15-second window
has 1,500 valid frames of its 1,501.
Features after the valid frames are zero.
See [NeMo 3](#nemo-3).
`conversion.json` records each window's shapes under
`conversion.components.encoder_windows`.

| Window | Mel frames | Encoder frames (F) | Whole 80 ms frames |
| --- | --- | --- | --- |
| 4 s | 401 | 51 | 50 |
| 6 s | 601 | 76 | 75 |
| 8 s | 801 | 101 | 100 |
| 10 s | 1,001 | 126 | 125 |
| 15 s (`Encoder.mlmodelc`) | 1,501 | 188 | 187 |

The vocabulary file follows FluidInference's format:
the SentencePiece word-boundary marker U+2581 (`▁`) is written as an ASCII space.
The marker only occurs at the start of a piece.
Join the entries and trim the result to get text.

## Setup

Requirements:

- macOS on Apple silicon with Xcode or the Command Line Tools,
  for `xcrun coremlcompiler`.
- [uv](https://docs.astral.sh/uv/).
- About 8 GB of free memory and 6 GB of disk space for both variants,
  plus about 1.2 GB for each window encoder of each variant
  (the intermediate package and the compiled model).

Install the locked environment:

```sh
cd Tools/ParakeetConversion
uv sync
```

The lock file pins Python 3.12, PyTorch 2.7.0, NeMo 3.0.0, coremltools 9.0,
and NumPy 2.3.5.
NumPy stays below 2.4: coremltools 9.0 converts one-element arrays with `int()`,
which NumPy 2.4 rejects.
The artifacts published before October 14, 2026, were converted with
NeMo 2.3.1 and NumPy 1.26.4, the versions FluidInference used.

Set `PARAKEET_BUILD_DIR` to keep checkpoints, outputs, fixtures, and reports
outside `build/`, for example on a scratch volume.

## Convert

```sh
uv run python convert.py --variant v3
uv run python convert.py --variant ultra
```

The first run downloads the checkpoints to `build/cache/`
(2.5 GB for v3; 1.3 GB more for Ultra).
The converter checks each checkpoint's SHA-256 digest against the digest
that Hugging Face records for it.
It writes intermediate `.mlpackage` files to `build/work/<variant>/`,
compiles them with `xcrun coremlcompiler`,
and writes the outputs to `build/<variant>/`.
Each conversion takes about one minute on an M4 Max.

### Shorter encoder windows

`--encoder-windows` also exports the encoder for shorter fixed windows,
in whole seconds from 2 to 14.
The decoder and joint decision do not depend on the window, so they are shared.

To add window encoders to a directory that is already published,
pass that directory to `--reuse-from` and write to a new directory:

```sh
uv run python convert.py --variant v3 --reuse-from build/v3 \
    --encoder-windows 4 6 8 10 --output-dir build/v3-live --work-dir build/work/v3-live
uv run python convert.py --variant ultra --reuse-from build/ultra \
    --encoder-windows 4 6 8 10 --output-dir build/ultra-live --work-dir build/work/ultra-live
```

`--reuse-from` checks every file of the earlier directory against its
`conversion.json`, copies the files unchanged, and exports only the new encoders.
The new `conversion.json` keeps the earlier record, lists the window encoders
under `conversion.components.encoder_windows`,
and records when and with which tools they were added under `encoder_windows_added`.
The model card lists the new files.
Each window takes about 40 seconds.

Without `--reuse-from`, the converter exports every component again.
That does not reproduce the earlier files byte for byte:
`weight.bin`, `model.mil`, and `metadata.json` match,
but each `coremldata.bin` contains the conversion date and stores the
user-defined metadata in an order that changes between runs.

Each window encoder is 588 to 592 MB, almost all of it the same int8 weights
as `Encoder.mlmodelc`.

### Long encoder windows (experimental)

`--encoder-windows` also takes whole seconds from 16 to 120,
for an experimental file-mode encoder such as `Encoder60s.mlmodelc`.
These encoders limit self-attention to 256 encoder frames (20.48 s) on each side,
as NeMo's local attention (`rel_pos_local_attn` with context [256, 256]) does;
the model card recommends local attention for long audio.
coremltools 9.0 cannot convert NeMo's sliding-chunk implementation (`as_strided`),
so the converter keeps NeMo's dense `rel_pos` attention and masks keys
more than 256 frames away (`att_context_size` [256, 256]).
In PyTorch, the two differ by at most 2.1e-5 on the checked windows.
Length arithmetic and comparisons stay in float32:
float16 represents integers exactly only up to 2,048,
and with float16 lengths a 60 s encoder reported 751 valid frames instead of 750.
On October 15, 2026, the v3 encoders measured 605 MB (30 s), 627 MB (60 s),
and 675 MB (120 s); the converter peaked at 6.5 to 10 GB of memory.
The Swift runtime uses the longest such encoder of a staged directory for file results,
and its blank-gap recovery then keeps looking for gaps after one that it cannot recover.
These encoders are not published.

The conversion follows FluidInference's
[mobius](https://github.com/FluidInference/mobius) scripts:

1. Trace NeMo's encoder, prediction network, and joint decision with
   `torch.jit.trace` on a fixed 15-second window.
2. Convert each trace with `coremltools.convert` to an ML program
   with float16 compute precision.
3. Quantize the encoder weights with
   `coremltools.optimize.coreml.linear_quantize_weights`
   (`linear_symmetric`, `int8`, `per_channel`, `weight_threshold=2048`).
4. Compile each package and write the vocabulary from NeMo's SentencePiece
   tokenizer.

## Verify

```sh
uv run python verify.py --variant v3 --compute-units CPU_AND_NE CPU_ONLY
uv run python verify.py --variant ultra --compute-units CPU_AND_NE CPU_ONLY
```

The verifier renders six fixtures with `say` into `build/fixtures/`:
English in three voices, German, French, and one 36-second English clip
that needs three windows.
Pass `--audio` with additional 16 kHz files.
It uses NeMo's preprocessor for all mel features,
so every backend receives the same input.

It checks:

1. **Interface.** The output names, types, and shapes of each compiled model.
2. **Encoder.** Core ML output versus NeMo's float32 PyTorch encoder on the
   first window of each fixture, over valid frames: maximum and mean absolute
   error, relative L2 error, and cosine similarity.
3. **Decoder and joint.** The Core ML prediction network along NeMo's token
   sequences, and the Core ML joint decision on every joint input that occurs
   during greedy decoding with the PyTorch modules.
4. **Transcripts.** NeMo's own `transcribe()` versus greedy TDT decoding in
   Python over the Core ML models with `MLModel.predict`.
   Long clips use 15-second windows with 2 seconds of left and right context.
   The same windowed decoder also runs over the PyTorch modules,
   so the report separates conversion error from windowing error.
   Word error rate is reported on raw text and on lowercased text without
   punctuation.
5. **Comparison.** The same checks for FluidInference's published artifacts
   at the pinned revisions, and with `--compare-artifacts <directory>`,
   for another directory such as a published revision of this tool's output.

The report is `build/reports/verify-<variant>.json`.
`verify_live.py` measures live windowing; see [Live windows](#live-windows).

### Results

These results are from NeMo 2.3.1; [NeMo 3](#nemo-3) has the current results.
Observed on October 13, 2026, on an M4 Max with macOS 27.0.1 (26A434),
with the environment pinned at the time (NeMo 2.3.1).
Encoder error is over the valid frames of the first window of all six fixtures.
"WER vs NeMo" compares with NeMo's full-context `transcribe()` output;
"WER vs PyTorch windowed" compares with the same windowed decoder
running NeMo's PyTorch modules.

| Variant | Compute units | Encoder max / mean abs error | Encoder cosine | Joint token agreement | WER vs NeMo, raw / normalized | WER vs PyTorch windowed |
| --- | --- | --- | --- | --- | --- | --- |
| v3 | `CPU_AND_NE` | 0.147 / 0.00103 | 0.9895 | 430 / 430 | 2.87% / 0% | 0.48% |
| v3 | `CPU_ONLY` | 0.140 / 0.00084 | 0.9925 | 430 / 430 | 2.39% / 0% | 0.96% |
| Ultra | `CPU_AND_NE` | 0.145 / 0.00109 | 0.9949 | 425 / 425 | 2.40% / 0% | 0% |
| Ultra | `CPU_ONLY` | 0.100 / 0.00083 | 0.9976 | 425 / 425 | 2.40% / 0% | 0% |

- All five single-window fixtures decode to exactly NeMo's text.
  Every raw difference is punctuation or capitalization in the 36-second clip,
  where the 15-second windows change sentence boundaries.
  The windowed PyTorch decoder shows the same kind of change
  (3.35% raw WER versus NeMo for v3, 2.40% for Ultra).
- The mean absolute value of the reference encoder output is 0.0157 for v3
  and 0.0217 for Ultra.
  The lowest per-fixture cosine is on the shortest clip
  (v3, 2.7 s, 34 valid frames: 0.878 on `CPU_AND_NE`, 0.907 on `CPU_ONLY`);
  its tokens still match.
- The prediction network's output differs from PyTorch by at most 0.015 (v3)
  and 0.025 (Ultra); its LSTM state differs by at most 0.40 and 0.45.
- The vocabulary is byte-identical to FluidInference's `parakeet_vocab.json`
  for both variants.
- FluidInference's artifacts at the pinned revisions give the same numbers in
  every row.
  The `weights/weight.bin` file of each converted model is byte-identical to the
  corresponding FluidInference file (`Encoder_v2.mlmodelc` for v3,
  `Encoder.mlmodelc` for Ultra, and `Decoder.mlmodelc` and
  `JointDecisionv3.mlmodelc` for both).
  Each `model.mil` differs only in its `buildInfo` line (compiler version).
  The remaining differences are in `coremldata.bin` and `metadata.json`,
  which hold the model description, author, and license.

## Live windows

The Swift recognizer's live mode commits 0.96 s chunks (12 encoder frames)
with 12 s of left context and 2 s of right context,
so each update encodes a full 15-second window.
A shorter fixed window encodes only the left context, chunk, and right context.
This section measures what shorter windows change.
All measurements are from October 13, 2026,
on an M4 Max with macOS 27.0.1 (26A434), with NeMo 2.3.1.

### Encoder windows

The window encoders are traces of the same NeMo encoder at a different input length.
The relative positional encoding and the subsampling do not depend on absolute time,
but the trace stores each layer's positional projection for the window length
as a constant, so each window needs its own model.
That constant is why `weight.bin` grows by about 24.6 KB per relative position
(587.5 MB for 4 s, 589.9 MB for 8 s).

`verify_live.py --encoder-check` compares the float32 PyTorch encoder
at six positions of `long.wav` (0 to 45 s):

- **Masked 15 s window.** The 15 s encoder given the same W seconds of audio and
  `mel_length` for W seconds computes the same function as the W-second encoder,
  apart from the last, partial encoder frame:
  in the 15 s window the second and third subsampling convolutions read
  `ReLU(bias)` activations past the valid length, where the short window reads
  zero padding.
  After subsampling, that frame's cosine similarity is 0.95 to 0.97;
  the other frames differ by at most 0.0015.
  After the encoder, the mean frame cosine is 0.9970 to 0.9999.
  A few isolated frames inside the window differ more (minimum 0.68 for 8 s),
  which is also true of the Core ML conversion on identical input.
  This was NeMo 2.3.1. NeMo 3.0.0 masks frames after the valid length between
  the subsampling convolutions, and the masked 15 s window then matches the
  8 s encoder on every frame (see [NeMo 3](#nemo-3)).
- **Same audio with more right context.** Frames of a W-second window differ
  from the same frames of a 15 s window that starts at the same sample,
  because full self-attention lets every frame depend on all later audio.
  The difference is largest at the end of the window and reaches its first frame.
  Computing the mel features over W seconds instead of slicing the 15 s window's
  features changes the mean cosine by less than 0.03.
- **Core ML.** Each Core ML window encoder against the PyTorch encoder
  on the same mel features, with `CPU_AND_NE`.

Each W-second window computes its own mel features in the comparison with the
15 s window.

| Window | Masked 15 s, mean / min cosine | Mean cosine with the 15 s window, by frames before the end: 0–1 / 2–5 / 6–12 / 13–25 / 26+ | First frame | Core ML vs PyTorch, mean / min cosine |
| --- | --- | --- | --- | --- |
| 4 s | 0.9970 / 0.727 | 0.07 / 0.25 / 0.47 / 0.82 / 0.87 | 0.71 | 0.9920 / 0.373 |
| 6 s | 0.9999 / 0.994 | 0.12 / 0.17 / 0.32 / 0.72 / 0.91 | 0.69 | 0.9986 / 0.864 |
| 8 s | 0.9990 / 0.679 | 0.15 / 0.30 / 0.50 / 0.81 / 0.95 | 0.83 | 0.9991 / 0.972 |
| 10 s | 0.9994 / 0.978 | 0.09 / 0.20 / 0.31 / 0.74 / 0.94 | 0.76 | 0.9976 / 0.595 |

The live recognizer decodes frames 25 to 40 frames before the end of a window
whatever its length, so a 15 s window gives those frames no more right context
than a short one.
Only more left context distinguishes it.
When the left context fits, a short window and the 15 s window receive the same
audio, and the transcripts are the same
(`6:25` and `15:25` below produce identical results).

### Live simulation

`verify_live.py` ports the commit path of the Swift `ParakeetRecognizer`
to Python and runs it over the Core ML models:

- Audio arrives in 100 ms blocks, as with the CLI's `--live` option.
- A regular window runs when its right context is available.
  Its features come from NeMo's preprocessor over the window,
  with the valid length rounded up to a whole encoder frame.
- Decoding starts after the last committed token or four frames before
  the previous commit frame, whichever is later, and commits tokens emitted
  before the commit frame.
- Each window rebuilds the prediction network context from the blank token
  and the committed tokens of the last `decoderHistoryFrames` frames.
- At the end of input, a final window commits the rest.

Volatile previews and lookahead tokens do not change committed tokens,
so the simulation does not compute them.
A configuration is written `<window seconds>:<left frames>`, with optional
`:h<history frames>`, `:r<right frames>`, `:c<chunk frames>`, and `:m<margin frames>`.
The defaults are the Swift live configuration's 25, 25, 12, and 4.
`file` is the Swift file mode: 15 s windows with 2 s of left and right context
and the largest chunk.
The current live mode is `15:150`.

```sh
uv run python verify_live.py --variant v3 --artifacts build/v3-live \
    --audio long.wav=long.txt ten.wav=ten.txt --encoder-check 4 6 8 10 \
    --configs file 15:150 15:150:h75 8:63:h63
```

`--audio` takes 16 kHz or other sample rates (resampled with a polyphase filter)
and an optional reference text.
`--prepend-ms` adds silence before each clip, which moves every window boundary.
The report is `build/reports/live-<variant>.json`;
NeMo's transcripts are cached in `build/reports/nemo-transcripts-<variant>.json`.

The fixtures are the six synthetic clips,
four long clips (497 s in total),
and a 10-minute stress clip:

- `long.wav` (112.7 s) and `long2.wav` (137.0 s), with reference texts.
- The same two texts rendered with `say -v Daniel` and `say -v Karen`
  (113.4 s and 133.4 s).
- `ten.wav`: 2 s of silence, 64.67 repeats of a 9.2 s seed, and a 3.3 s ending
  (`Tools/make-evaluation-audio.swift`).
  Its reference is the seed text 64 times, the seed's first two sentences,
  and the ending; the last repeat stops 74 ms into the next "The".

Word error rate (WER) uses NeMo's `transcribe()` on the whole clip
as the reference for the synthetic and long clips.
Against the long clips' reference texts, NeMo itself has 7.7% and 8.0%
normalized WER, because it writes numbers as digits.
For `ten.wav`, NeMo's full-context transcript writes "test" for "tests"
65 times, so `ten.wav` uses its reference text.
"Normalized" means lowercase without punctuation.
Punctuation and capitalization retention is the number of punctuation marks
and capitalized words in the result divided by the number in the reference.

### Results

v3, all fixtures, windows at their natural positions.
S, D, and I are substituted, deleted, and inserted words after normalization.

| Config | Long clips normalized WER vs NeMo | Long clips S / D / I | Long clips punctuation / capitals | `ten.wav` S / D / I | `ten.wav` punctuation / capitals |
| --- | --- | --- | --- | --- | --- |
| `file` | 0.75% | 9 / 0 / 2 | 1.00 / 0.97 | 2 / 0 / 0 | 0.89 / 0.89 |
| `15:150` (current live) | 1.71% | 22 / 0 / 3 | 0.93 / 0.97 | 0 / 0 / 0 | 0.07 / 0.07 |
| `15:150:h75` | 1.50% | 19 / 0 / 3 | 0.99 / 0.99 | 0 / 0 / 0 | 1.00 / 1.00 |
| `15:150:h150` | 1.44% | 18 / 0 / 3 | 1.00 / 0.98 | 0 / 0 / 0 | 1.00 / 1.00 |
| `15:112:h112` | 3.08% | 26 / 12 / 7 | 0.98 / 0.91 | 0 / 0 / 0 | 1.00 / 1.00 |
| `15:88` | 2.67% | 21 / 14 / 4 | 0.85 / 0.77 | 0 / 0 / 0 | 0.48 / 0.47 |
| `15:88:h88` | 4.65% | 22 / 41 / 5 | 0.96 / 0.90 | 0 / 0 / 0 | 1.00 / 1.00 |
| `15:63` | 1.50% | 20 / 0 / 2 | 0.94 / 0.97 | 0 / 0 / 0 | 0.86 / 0.75 |
| `15:63:h63` | 2.46% | 24 / 7 / 5 | 0.99 / 0.96 | 0 / 0 / 0 | 1.00 / 1.00 |
| `15:38` | 3.97% | 23 / 23 / 12 | 0.99 / 0.85 | 0 / 0 / 0 | 0.73 / 0.66 |
| `15:38:h38` | 3.15% | 24 / 8 / 14 | 1.02 / 0.88 | 0 / 0 / 0 | 1.00 / 0.98 |
| `15:25` | 1.98% | 15 / 12 / 2 | 0.97 / 0.80 | 0 / 26 / 0 | 0.93 / 0.44 |
| `10:88` | 6.84% | 18 / 78 / 4 | 0.75 / 0.71 | 0 / 0 / 0 | 0.48 / 0.48 |
| `10:88:h88` | 6.91% | 17 / 82 / 2 | 0.91 / 0.91 | 0 / 0 / 0 | 1.00 / 1.00 |
| `8:63` | 1.09% | 14 / 0 / 2 | 0.95 / 0.98 | 0 / 0 / 0 | 0.61 / 0.43 |
| `8:63:h63` | 1.30% | 17 / 0 / 2 | 0.99 / 0.98 | 0 / 0 / 0 | 1.00 / 1.00 |
| `8:63:h75` | 1.09% | 14 / 0 / 2 | 0.99 / 0.97 | 0 / 0 / 0 | 1.00 / 1.00 |
| `6:38` | 3.21% | 26 / 12 / 9 | 0.89 / 0.80 | 0 / 0 / 0 | 0.54 / 0.52 |
| `6:38:h50` | 3.01% | 22 / 12 / 10 | 0.99 / 0.91 | 0 / 0 / 0 | 1.00 / 0.99 |
| `4:13` | 6.09% | 29 / 47 / 13 | 0.92 / 0.72 | 0 / 27 / 0 | 0.74 / 0.18 |
| `4:13:h13` | 3.90% | 27 / 15 / 15 | 0.95 / 0.67 | 0 / 27 / 0 | 0.84 / 0.22 |

Every configuration transcribes the six synthetic clips with the same words as
NeMo, except `15:25` and `6:25` (one word, 0.48%) and `4:13:h13` (2.38%).
The deletions I inspected are contiguous runs of up to 39 words,
mostly at the start of a paragraph after a pause, where the decoder emits only blanks.
Results are sensitive to small changes in window content:
the windows of `15:88` and `10:88` differ by at most 1,600 samples (100 ms)
of audio at the end, and `10:88` deletes 64 more words.

To check that the leading results do not depend on where window boundaries
fall, the same clips ran with 330 ms and 650 ms of silence prepended:

| Config | Long clips normalized WER at 0 / 330 / 650 ms | Long clips deletions | Long clips insertions | `ten.wav` punctuation |
| --- | --- | --- | --- | --- |
| `file` | 0.75% / 0.89% / 0.68% | 0 / 0 / 0 | 2 / 1 / 1 | 0.89 / 0.93 / 0.96 |
| `15:150` | 1.71% / 1.23% / 1.23% | 0 / 0 / 0 | 3 / 1 / 2 | 0.07 / 0.05 / 0.06 |
| `15:150:h75` | 1.50% / 1.37% / 1.30% | 0 / 0 / 0 | 3 / 2 / 3 | 1.00 / 1.00 / 1.00 |
| `8:63:h63` | 1.30% / 1.44% / 1.23% | 0 / 0 / 0 | 2 / 2 / 2 | 1.00 / 1.00 / 1.00 |
| `8:63:h75` | 1.09% / 1.44% / 1.23% | 0 / 0 / 0 | 2 / 2 / 2 | 1.00 / 1.00 / 1.00 |
| `6:38:h50` | 3.01% / 1.98% / 2.19% | 12 / 1 / 1 | 10 / 10 / 10 | 1.00 / 1.00 / 1.00 |
| `10:88:h88` | 6.91% / 6.43% / 5.34% | 82 / 70 / 51 | 2 / 2 / 2 | 1.00 / 1.00 / 1.00 |

Ultra, all fixtures, windows at their natural positions:

| Config | Long clips normalized WER vs NeMo | Long clips S / D / I | Long clips punctuation / capitals | `ten.wav` S / D / I | `ten.wav` punctuation / capitals |
| --- | --- | --- | --- | --- | --- |
| `file` | 1.30% | 17 / 0 / 2 | 1.00 / 1.00 | 0 / 0 / 0 | 0.75 / 0.75 |
| `15:150` (current live) | 2.19% | 28 / 0 / 4 | 0.94 / 1.03 | 0 / 0 / 0 | 0.04 / 0.04 |
| `15:150:h75` | 2.19% | 27 / 0 / 5 | 0.99 / 1.03 | 0 / 0 / 0 | 1.00 / 1.00 |
| `8:63:h63` | 1.64% | 24 / 0 / 0 | 0.99 / 0.98 | 0 / 0 / 0 | 1.00 / 1.00 |
| `8:63:h75` | 1.57% | 23 / 0 / 0 | 1.00 / 0.99 | 0 / 0 / 0 | 1.00 / 1.00 |
| `6:38:h50` | 1.30% | 9 / 1 / 9 | 1.01 / 0.92 | 0 / 0 / 0 | 1.00 / 1.00 |

All Ultra configurations transcribe the synthetic clips with NeMo's words.

### Punctuation and capitalization in live mode

The simulation reproduces the loss of punctuation and capitalization on
`ten.wav` in the current live mode.
After the first three sentences (about 15 s), `15:150` writes 0 to 2
sentence ends per minute; file mode writes 12 to 20, and the audio has 19.6.
The words stay correct.

The cause is the decoder history.
Each window rebuilds the prediction network context from the blank token and
the tokens of the last 25 frames (2 s), while the encoder sees 12 s of left context.
In file mode the left context is also 2 s.
With a history of 75 or 150 frames, `15:150` keeps all 195 sentence ends and
capitals on `ten.wav` (Ultra was tested with 75 frames),
and the long clips' normalized WER does not increase
(v3: 1.71% to 1.50%; Ultra: 2.19% to 2.19%).
With no history (`15:150:h0`), the decoder writes no punctuation and deletes
1,298 of 1,694 words.
Less left context with the default history also restores part of the punctuation
(0.48 at 88 frames and 0.86 at 63 frames).
A history much longer than the left context causes deletions
(`6:38:h150`: 47 deleted words on `long2.wav`),
and carrying the whole decoder state (`15:150:h-1`) deleted 13 words on
`long.wav`.

### Latency

`latency.swift` loads a compiled model with `MLModel` and times
`prediction(from:)` with random inputs after five warm-up calls:

```sh
xcrun swiftc -O latency.swift -o build/latency
build/latency build/v3-live/Encoder8s.mlmodelc cpuAndNeuralEngine 100 fresh
```

Encoders, 100 calls each.
"First load" loads a new copy of the model in a new process.
The Ultra window encoders, which no process had loaded before,
took 7.1 to 8.9 s with `cpuAndNeuralEngine` and 2.1 to 3.5 s with `cpuOnly`.
Loading the same path again in a new process of this command-line tool
took as long (7.6 to 9.9 s with `cpuAndNeuralEngine`),
so the tool saw no compiled-model cache between processes.
A second load in the same process took 51 to 58 ms with `cpuAndNeuralEngine`
and under 1 ms with `cpuOnly`.
The simulation runs 1.08 to 1.10 windows per second of audio,
more than 1 / 0.96 because the boundary margin repeats up to four frames;
the last column multiplies the median by 1.09.

| v3 encoder | `cpuAndNeuralEngine` median / p95 | `cpuOnly` median / p95 | First load, `cpuAndNeuralEngine` / `cpuOnly` | Encoder time per second of audio, `cpuAndNeuralEngine` / `cpuOnly` |
| --- | --- | --- | --- | --- |
| 15 s | 26.8 / 27.1 ms | 70.2 / 72.2 ms | 10.7 / 2.0 s | 29 / 77 ms |
| 10 s | 14.8 / 15.0 ms | 50.9 / 51.8 ms | 7.3 / 1.9 s | 16 / 55 ms |
| 8 s | 12.1 / 12.3 ms | 46.7 / 49.1 ms | 7.0 / 1.8 s | 13 / 51 ms |
| 6 s | 11.1 / 11.4 ms | 38.6 / 42.8 ms | 7.7 / 2.5 s | 12 / 42 ms |
| 4 s | 8.2 / 8.5 ms | 28.1 / 28.8 ms | 8.3 / 2.3 s | 9 / 31 ms |

The Ultra encoders have the same latency within 0.3 ms with `cpuAndNeuralEngine`.

### Fewer calls per decoding step

`experiment_joint.py` exports joint variants to `build/experiments/<variant>/`
for measurement only; it does not change the published joint or decoder.
Latency over 1,000 calls:

| Model | `cpuAndNeuralEngine` median / p95 | `cpuOnly` median / p95 |
| --- | --- | --- |
| `Decoder.mlmodelc` | 0.122 / 0.197 ms | 0.120 / 0.178 ms |
| `JointDecisionv3.mlmodelc` (published) | 0.208 / 0.316 ms | 0.229 / 0.311 ms |
| Joint decision without top-K outputs | 0.184 / 0.290 ms | 0.190 / 0.284 ms |
| Joint decision on a projected encoder frame | 0.184 / 0.280 ms | 0.195 / 0.293 ms |
| Joint decisions for 8 frames | 0.380 / 0.488 ms | 0.423 / 0.498 ms |
| Joint decisions for 16 frames | 0.413 / 0.511 ms | 0.620 / 0.922 ms |
| Joint decisions for 16 projected frames | 0.603 / 0.865 ms | 0.604 / 0.902 ms |
| Encoder projection, 101 frames (8 s window) | 0.126 / 0.180 ms | 0.087 / 0.108 ms |

The multi-frame models agree with PyTorch on every token and duration in the
check; the projected single-frame model agrees on 99.8% of frames.

Calls per second of audio in the simulation (v3, all fixtures, 1,160 s):

| Config | Windows | Joint calls | Joint calls with a new decoder output | Token predictions | Context predictions | Model time per second of audio |
| --- | --- | --- | --- | --- | --- | --- |
| `15:150` | 1.10 | 6.88 | 5.74 | 5.68 | 13.9 | 33 ms |
| `15:150:h75` | 1.08 | 6.81 | 5.87 | 5.85 | 38.3 | 36 ms |
| `8:63:h63` | 1.09 | 6.79 | 5.87 | 5.85 | 32.7 | 19 ms |
| `8:63:h75` | 1.09 | 6.79 | 5.87 | 5.85 | 38.4 | 20 ms |

The model time multiplies these rates by the `cpuAndNeuralEngine` medians.
It excludes the mel features and the lookahead decoding for volatile results.
TDT durations already skip most frames, and 83% to 86% of joint calls use a new
prediction network output, so a multi-frame joint would save at most 14% to 17%
of joint calls at two to three times the cost per call.
Dropping the top-K outputs saves 0.024 ms per call, about 0.16 ms per second of audio;
moving the encoder projection out of the joint saves nothing more.
Neither change is worth a new artifact; the encoder dominates.

### Recommendation

For live input, use `Encoder8s.mlmodelc` with 63 frames (5.04 s) of left context,
12-frame chunks, 25 frames of right context, a 4-frame boundary margin,
and a decoder history of 63 to 75 frames.
On these fixtures, its normalized WER on the long clips is 1.23% to 1.44% at three
window placements, against 1.23% to 1.71% for the current live mode.
It keeps punctuation and capitalization, has no deletions at any placement,
and cuts the encoder time per update from 26.8 ms to 12.1 ms.

To keep the 15 s window, change only the decoder history to 75 frames;
that fixes punctuation and capitalization at about 3 ms more decoder time
per second of audio.
Avoid 88 to 112 frames of left context, which caused deletions in every test.

The 6 s window saves 1 ms per call over 8 s and inserted about 10 words per run;
the 4 s window lost too many words.

A Core ML multifunction model can hold the 15 s and 8 s encoders with shared
weights: a test package with both functions had one 599 MB `weight.bin`
instead of the separate 594 MB and 590 MB files, and each function's output was
identical to the separate model's.
It needs the iOS 18 and macOS 15 model specification and
`MLModelConfiguration.functionName`, so `convert.py` does not produce it.

## NeMo 3

NeMo 2.5.0 ([NVIDIA-NeMo/Speech#13827](https://github.com/NVIDIA-NeMo/Speech/pull/13827))
changed the FastConformer preprocessing, and NeMo 3.0.0
([NVIDIA-NeMo/Speech#15562](https://github.com/NVIDIA-NeMo/Speech/pull/15562))
zero-fills normalized features after the valid frames.
Compared with NeMo 2.3.1, the preprocessor:

- pads the STFT with zeros instead of reflecting the window at its ends,
- counts `floor(N / 160)` valid frames instead of `floor(N / 160) + 1`, and
- sets pre-emphasized samples at and after the valid sample count to zero.

The encoder's subsampling is a `MaskedConvSequential`, which multiplies frames
after the valid length by zero before each convolution.
The traced encoder therefore has six more `mul` operations,
and `mel_length` now affects the subsampling as well as the attention mask.
The tensor interface and file names are unchanged.

The tool converts with NeMo 3.0.0 since October 14, 2026.
The Swift mel frontend follows the same rule;
`mel_reference.py` writes the NeMo values that its unit test checks.

### Conversion

Observed on October 14, 2026, on an M4 Max with macOS 27.0.1 (26A434).
`convert.py --variant v3 --encoder-windows 8` and the same for Ultra each took 80 s.

- `Decoder.mlmodelc` and `JointDecisionv3.mlmodelc` have the same `weight.bin`
  and `model.mil` bytes as the published revisions
  (`f2f43ea` for v3, `4038b9e` for Ultra), and `parakeet_vocab.json` is unchanged.
- Each encoder `weight.bin` has the same size as before;
  165 of its 594,211,328 bytes differ for v3 (134 for Ultra),
  and 77 (63) for the 8 s encoder, spread across the 24 layers.
  The cause was not investigated;
  the encoder parity below is as good as or better than the published encoders'.

### Verification against NeMo 3.0.0

`verify.py --compute-units CPU_AND_NE CPU_ONLY --skip-reference --compare-artifacts <published>`
checks the new and the published artifacts against NeMo 3.0.0 with the same
NeMo 3.0.0 features.
Encoder error is over the valid frames of the first window of the six fixtures.

| Variant | Artifacts | Compute units | Encoder max / mean abs error | Encoder cosine | Lowest fixture cosine | Joint token agreement | WER vs NeMo, raw / normalized |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3 | NeMo 3.0.0 | `CPU_AND_NE` | 0.048 / 0.00071 | 0.99844 | 0.9970 | 433 / 433 | 2.87% / 0% |
| v3 | Published | `CPU_AND_NE` | 0.068 / 0.00083 | 0.99662 | 0.9931 | 433 / 433 | 2.87% / 0% |
| v3 | NeMo 3.0.0 | `CPU_ONLY` | 0.025 / 0.00059 | 0.99904 | 0.9987 | 433 / 433 | 2.87% / 0% |
| v3 | Published | `CPU_ONLY` | 0.033 / 0.00066 | 0.99878 | 0.9971 | 433 / 433 | 2.87% / 0% |
| Ultra | NeMo 3.0.0 | `CPU_AND_NE` | 0.141 / 0.00104 | 0.99570 | 0.9863 | 425 / 425 | 1.92% / 0.48% |
| Ultra | Published | `CPU_AND_NE` | 0.141 / 0.00108 | 0.99575 | 0.9864 | 425 / 425 | 1.92% / 0.48% |
| Ultra | NeMo 3.0.0 | `CPU_ONLY` | 0.126 / 0.00078 | 0.99810 | 0.9942 | 425 / 425 | 1.44% / 0% |
| Ultra | Published | `CPU_ONLY` | 0.162 / 0.00087 | 0.99738 | 0.9926 | 425 / 425 | 1.92% / 0.48% |

The decoders are identical, so their error is the same for both:
at most 0.015 (v3) and 0.025 (Ultra) in the output.
The windowed PyTorch decoder has 2.87% (v3) and 2.40% (Ultra) raw WER against
NeMo's full-context `transcribe()`.

`verify_live.py --encoder-check 8 --configs file 8:63:h75` with `long.wav`,
`long2.wav`, and `ten.wav`:

- **Masked 15 s window.** With NeMo 3.0.0, the 15 s PyTorch encoder given 8 s of
  audio and its `mel_length` matches the 8 s encoder on every frame
  (minimum cosine 1.0, maximum absolute difference 4.3e-6).
  With NeMo 2.3.1 the last frame differed (cosine 0.95 to 0.97).
- **Core ML 8 s encoder versus PyTorch**, mean / minimum frame cosine:
  v3 0.99486 / 0.382 (new) and 0.99474 / 0.382 (published);
  Ultra 0.99646 / 0.466 (new) and 0.99639 / 0.463 (published).
- **Simulation.** New and published artifacts give the same counts:
  normalized WER against NeMo on the synthetic and long clips of
  0.21% (v3, file), 0.53% (v3, `8:63:h75`), 0.85% (Ultra, file),
  and 1.60% (Ultra, `8:63:h75`);
  all 195 sentence ends of `ten.wav` in `8:63:h75`.

### Swift runtime

The Swift recognizer with the NeMo 3 frontend, release build, CLI `evaluate`
on 11 fixtures (4,876 reference words):
the six fixtures of the October 13 evaluation, three English `say` voices
(Moira, Rishi, Tessa), German (Anna), and French (Thomas).
"Errors" are word edits against the reference texts:
raw (case-sensitive whitespace tokens) and normalized
(lowercase, without punctuation).

| Variant | Mode | Published, raw / normalized errors | NeMo 3.0.0 conversion, raw / normalized errors |
| --- | --- | --- | --- |
| v3 | File | 231 / 149 | 238 / 148 |
| v3 | Live (8 s encoder) | 224 / 124 | 224 / 124 |
| Ultra | File | 215 / 137 | 216 / 137 |
| Ultra | Live (8 s encoder) | 211 / 135 | 211 / 135 |

The extra raw errors of v3 file mode are three sentences that the new encoder
splits in two on `long2.wav` and one that it joins on `long.wav`.
Word start times of the two conversions differ by 80 ms for at most 2.1% of words.
On the 60-minute fixture, both v3 conversions produced 10,178 words in each mode;
live mode was word for word the same, and file mode differed in 10 words
("test" and "tests").
File mode took 17.6 s with the new encoders and 17.3 s with the published ones;
live mode took 93.1 s and 90.3 s, with peaks of 87 to 93 MB.

## License and attribution

The code in this directory is licensed under the Apache License, Version 2.0.
Parts of it are derived from FluidInference's mobius;
see `NOTICE` and the header of each derived file.
Keep those notices in copies and derivatives.

The converted models are adaptations of models licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
CC BY 4.0 permits redistribution and modification if the distribution
credits the creator, links to the license, and indicates the changes.
Each output's `README.md` and `conversion.json` contain this attribution,
and the Core ML model metadata (`license`) repeats it.

v3:

> Parakeet TDT 0.6B v3 by NVIDIA
> (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
> licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
> Converted to Core ML for AnySpeechRecognizer: the model is split into
> encoder, prediction network, and joint decision components,
> weights are stored in float16, and encoder weights are quantized to int8.

Ultra:

> Parakeet Ultra by Moondream (https://huggingface.co/moondream/parakeet-ultra),
> a post-trained version of Parakeet TDT 0.6B v3 by NVIDIA
> (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3);
> both licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
> Converted to Core ML for AnySpeechRecognizer: the model is split into
> encoder, prediction network, and joint decision components,
> weights are stored in float16, and encoder weights are quantized to int8.

Applications that ship or download these models must show the same attribution
to their users, for example in an acknowledgments screen.

## Upload

Nothing in this directory uploads to Hugging Face.
`upload_dry_run.py` re-hashes every output file against `conversion.json`
and prints the commands to publish the directory:

```sh
uv run python upload_dry_run.py --variant v3
```

Each repository has the output directory's layout at its root:

```
README.md
conversion.json
parakeet_vocab.json
Encoder.mlmodelc/{analytics/coremldata.bin, coremldata.bin, metadata.json, model.mil, weights/weight.bin}
Decoder.mlmodelc/...
JointDecisionv3.mlmodelc/...
```

After review, a maintainer authenticated as `mattt` runs:

```sh
hf repo create mattt/parakeet-tdt-0.6b-v3-coreml --repo-type model
hf upload mattt/parakeet-tdt-0.6b-v3-coreml build/v3 . --repo-type model \
    --commit-message "Add Parakeet TDT 0.6B v3 Core ML artifacts"

hf repo create mattt/parakeet-ultra-coreml --repo-type model
hf upload mattt/parakeet-ultra-coreml build/ultra . --repo-type model \
    --commit-message "Add Parakeet Ultra Core ML artifacts"
```

For a directory made with `--reuse-from`, the dry run also checks the window
encoders, omits `hf repo create`, and proposes a commit message that names the
added windows, for example:

```sh
uv run python upload_dry_run.py --variant v3 --artifacts build/v3-live
```

For a full conversion that replaces the files of an existing repository,
pass `--update`; the dry run omits `hf repo create` and proposes a commit
message with the NeMo version.

The upload includes every file in the directory,
so for publication export only the windows that the runtime uses.
Before uploading, check that the other files match the published revision;
`--reuse-from` keeps them byte-identical to its source directory.

Then record the commit hash from `hf models info <repository>`
and the paths, sizes, and digests from `conversion.json`
in the runtime's built-in manifest.
Do not change files in place after publication;
publish a fixed artifact under a new revision and update the manifest.
