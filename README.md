# AnySpeechRecognizer

AnySpeechRecognizer is a Swift package that implements Apple's modern Speech API
with a choice of recognition engine.
Code written for `SpeechAnalyzer`, `SpeechTranscriber`, `DictationTranscriber`,
`SpeechDetector`, and `AssetInventory` builds against this package
when you change the import statement:

```diff
- import Speech
+ import AnySpeechRecognizer
```

The default backend is Apple's Speech framework.
The package also has its own runtime for NVIDIA's
[Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) models on Core ML,
with file and live input, volatile results, and word timing.
To use it, pass a model instead of a locale when you create the transcriber:

```diff
  let transcriber = SpeechTranscriber(
-     locale: locale,
+     model: ParakeetSpeechModel(variant: .v3),
      transcriptionOptions: [],
      reportingOptions: [.volatileResults],
      attributeOptions: [.audioTimeRange]
  )
```

Asset installation, analysis, and results use the same API for both backends.
As a package extension, `SpeakerDiarizer` identifies who speaks when
with NVIDIA's Streaming Sortformer model, on the same timeline as the transcriber.
The package does not implement the older `SFSpeechRecognizer` API.

## Requirements

- Swift 6.2 or later
- macOS 26 or later, or iOS 26 or later
- Apple silicon for the Parakeet and Sortformer models

## Installation

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/mattt/AnySpeechRecognizer", from: "0.1.0")
```

The package uses
[swift-huggingface](https://github.com/huggingface/swift-huggingface)
to download and cache model files.

## Usage

### Transcribe a file

Resolve a locale, install assets explicitly,
and read results while the analyzer reads the file:

```swift
import AVFoundation
import AnySpeechRecognizer

func transcribe(_ url: URL) async throws -> AttributedString {
    guard let locale = await SpeechTranscriber.supportedLocale(
        equivalentTo: Locale(identifier: "en-US")
    ) else {
        throw SpeechRecognitionError.unsupportedLanguage("English is unavailable.")
    }
    let transcriber = SpeechTranscriber(
        locale: locale,
        transcriptionOptions: [],
        reportingOptions: [],
        attributeOptions: [.audioTimeRange]
    )
    if let request = try await AssetInventory.assetInstallationRequest(
        supporting: [transcriber]
    ) {
        try await request.downloadAndInstall()
    }

    let analyzer = SpeechAnalyzer(modules: [transcriber])
    async let text: AttributedString = {
        var text = AttributedString()
        for try await result in transcriber.results {
            text.append(result.text)
        }
        return text
    }()

    do {
        let file = try AVAudioFile(forReading: url)
        _ = try await analyzer.analyzeSequence(from: file)
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        return try await text
    } catch {
        await analyzer.cancelAndFinishNow()
        throw error
    }
}
```

Each final result is a phrase, emitted once in source order,
with attributed text, alternatives, and a source time range.
Times stay on the source recording's timeline,
including a file's starting position.

### Transcribe live audio

Supply buffers as an `AsyncStream<AnalyzerInput>` and request volatile results.
A volatile result replaces earlier volatile text for the same audio;
a final result never changes.

```swift
let transcriber = SpeechTranscriber(locale: locale, preset: .progressiveTranscription)
let (audio, input) = AsyncStream<AnalyzerInput>.makeStream()
let analyzer = SpeechAnalyzer(inputSequence: audio, modules: [transcriber])

Task {
    var finalized = AttributedString()
    for try await result in transcriber.results {
        if result.isFinal {
            finalized.append(result.text)
            show(finalized, pending: nil)
        } else {
            show(finalized, pending: result.text)
        }
    }
}

// Feed capture buffers as they arrive, then finish.
input.yield(AnalyzerInput(buffer: buffer))
input.finish()
try await analyzer.finalizeAndFinishThroughEndOfInput()
```

The package does not capture audio.
Supply buffers in any PCM format; the analyzer converts them for each module.
`finalize(through:)` makes results final through a time while input continues,
and `volatileRange` reports the audio whose results can still change.

> [!WARNING]
> On macOS 27.0.1, Apple's `AnalyzerInput(buffer:)` initializer traps
> for Float32 buffers.
> Convert capture buffers to Int16 before you wrap them.

### Dictation and speech detection

`DictationTranscriber` and `SpeechDetector` use Apple's backend.
All Apple modules in one analyzer share one native analyzer:

```swift
let dictation = DictationTranscriber(locale: locale, preset: .longDictation)
let detector = SpeechDetector()
let analyzer = SpeechAnalyzer(modules: [dictation, detector])
```

### Results and cancellation

A module belongs to one analyzer and permits one result iterator.
One input operation runs at a time.
Read results concurrently with analysis:
each module buffers at most 256 results and 4 MiB of text.
A newer volatile result replaces an unread one,
but final results are never dropped;
too many unread final results fail the session with
`SpeechRecognitionError.bufferOverflow`.

If you stop reading results early, cancel the analyzer:

```swift
for try await result in transcriber.results {
    if shouldStop(result) {
        await analyzer.cancelAndFinishNow()
        break
    }
    display(result)
}
```

Cancellation discards buffered results, ends every module's results,
and waits for cleanup.
The active input operation throws `CancellationError`.
If the session already failed, it keeps the original error.

### Differences from Apple's Speech framework

The package has its own analyzer, module, and result types that match Apple's,
and it reuses Speech value types such as `AnalyzerInput` and `AnalysisContext`.
Package modules do not conform to `Speech.SpeechModule`,
so the package analyzer does not accept Apple's module instances.
If a file imports both modules, qualify ambiguous names,
such as `AnySpeechRecognizer.SpeechAnalyzer`.
To read word timing and confidence, import Speech for its attribute keys:

```swift
import Speech

let range = run[AttributeScopes.SpeechAttributes.TimeRangeAttribute.self]
let confidence = run[AttributeScopes.SpeechAttributes.ConfidenceAttribute.self]
```

`SpeechAnalyzer.state` is an extension that reports the session's lifecycle.
The package does not provide custom language models
or the OS 27 capture and asset input providers.

## Backends

| Capability | Apple | Parakeet | Qwen3-ASR |
| --- | --- | --- | --- |
| `SpeechTranscriber` | Yes | Yes | Yes |
| `DictationTranscriber`, `SpeechDetector` | Yes | No | No |
| Live input and volatile results | Yes | Yes | No; live input gives final results per segment |
| Fast results | Yes | Yes, with less right context | No |
| Alternatives, etiquette replacement | Yes | No | No |
| Word timing | Yes | Yes, in 80 ms frames | No; each result has its segment's range |
| Confidence | Yes | Yes, the geometric mean of token probabilities | No |
| Language | Selected by locale | Detected automatically, not reported; English only for `.unified` | Detected automatically, not reported, or forced |
| Assets | Apple's inventory | Pinned files in the Hugging Face cache | Pinned files in the Hugging Face cache |

Requests for an unsupported module or option fail before the analyzer reads audio.

### Apple

Modules created with a locale use Apple's Speech framework through `SystemSpeechModel`,
with every preset and option that Speech supports.
For these modules, `AssetInventory` forwards to Apple's inventory,
including locale reservations.

### Parakeet

`ParakeetSpeechModel` has three variants:

- `.v3` is NVIDIA's
  [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
  which recognizes 25 European languages.
- `.ultra` is Moondream's
  [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra),
  a post-trained version of v3 that is more accurate in most of our measurements.
- `.unified` is NVIDIA's
  [Parakeet Unified EN 0.6B](https://huggingface.co/nvidia/parakeet-unified-en-0.6b),
  which recognizes English only.
  It is the most accurate variant on English read speech,
  especially in live mode, but not on meetings.
  It is licensed under the NVIDIA Open Model License (see [License](#license)).

Creating a model does not download or load weights.
`AssetInventory` downloads pinned files from Hugging Face
and verifies the size and SHA-256 digest of each one.
`.v3` and `.ultra` need about 632 MB each, and `.unified` about 614 MB.
Modules with volatile or fast results also use an 8-second live encoder,
about 591 MB more, which makes live transcription faster
but raises its word error rate on long files.
Until it is installed, those modules use the 15-second encoder,
and `AssetInventory.status(forModules:)` reports them as not installed.

The files go into the standard Hugging Face cache:
`HF_HUB_CACHE` or `$HF_HOME/hub` if set,
`~/.cache/huggingface/hub` for Mac apps outside the sandbox,
and `Application Support/AnySpeechRecognizer/HuggingFace/hub`
for sandboxed apps and iOS apps.

To use the model offline, stage a directory with the compiled models and vocabulary:

```swift
let model = ParakeetSpeechModel(
    variant: .v3,
    assets: .directory(URL(fileURLWithPath: "/path/to/parakeet-models")),
    computeUnits: .cpuAndNeuralEngine
)
```

The directory contains `Encoder_v2.mlmodelc` or `Encoder.mlmodelc`,
`Decoder.mlmodelc`, `JointDecisionv3.mlmodelc`, `parakeet_vocab.json`,
and optionally the live encoder, `Encoder8s.mlmodelc`.
The runtime does not verify a staged directory.

### Qwen3-ASR

`QwenSpeechModel` runs Alibaba's
[Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) or
[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) on Core ML
for file transcription.
It recognizes 30 languages and 22 Chinese dialects,
including Mandarin, Cantonese, Japanese, and Korean, which Parakeet does not cover.
The conversions are published at
[mattt/qwen3-asr-0.6b-coreml](https://huggingface.co/mattt/qwen3-asr-0.6b-coreml)
and [mattt/qwen3-asr-1.7b-coreml](https://huggingface.co/mattt/qwen3-asr-1.7b-coreml),
and `AssetInventory` installs and verifies their pinned files as it does Parakeet's:

```swift
let transcriber = SpeechTranscriber(
    model: QwenSpeechModel(
        variant: .large,
        language: Locale.Language(identifier: "yue")  // or nil to detect it
    ),
    transcriptionOptions: [],
    reportingOptions: [],
    attributeOptions: []
)
```

To use a conversion that you make with [`Tools/QwenConversion`](Tools/QwenConversion),
pass its output directory as `assets: .directory(url)`.

The model transcribes audio in segments of up to 40 seconds,
cut at the quietest point near 30 seconds, as the original Qwen3-ASR library cuts long audio.
Each segment becomes one final result with the segment's source range.
Results arrive when a segment is complete:
at a cut, at a gap in the input, or at finalization.
Live input works, but results trail the audio by up to 40 seconds.
Requests for volatile or fast results, word timing, confidence, or dictation
fail with `SpeechRecognitionError.unsupportedConfiguration` before the analyzer reads audio.
The model card's separate forced-alignment model, Qwen3-ForcedAligner-0.6B,
predicts word timing but is not integrated,
so results have no time attributes.

Qwen3-ASR-1.7B is more accurate than 0.6B in every test below.
Each converted model is an int8 audio encoder and an int8 text decoder
with a float16 embedding table:
0.94 GB for 0.6B (187 MB and 754 MB) and 2.36 GB for 1.7B (319 MB and 2,035 MB).
Both models run on the GPU; Core ML cannot build the decoder for the Neural Engine.
While it transcribes, a Mac needs about 1.7 GB of memory for 0.6B and 4.4 GB for 1.7B,
for a 10-second file as for a 36-minute meeting.

## Speaker diarization

`SpeakerDiarizer` identifies who speaks when.
Apple's Speech framework has no diarization module,
so code that uses it doesn't build with `import Speech`.
Add it to the same analyzer as a transcriber from either backend.
Both modules receive the same input,
and the diarizer does not change the transcriber's results:

```swift
let transcriber = SpeechTranscriber(
    model: ParakeetSpeechModel(variant: .v3),
    attributeOptions: [.audioTimeRange]
)
let diarizer = SpeakerDiarizer(model: SortformerDiarizationModel())
if let request = try await AssetInventory.assetInstallationRequest(
    supporting: [transcriber, diarizer]
) {
    try await request.downloadAndInstall()
}
let analyzer = SpeechAnalyzer(modules: [transcriber, diarizer])

async let transcript: AttributedString = {
    var text = AttributedString()
    for try await result in transcriber.results where result.isFinal {
        text.append(result.text)
    }
    return text
}()
async let speakers: SpeakerTimeline = {
    var timeline = SpeakerTimeline()
    for try await result in diarizer.results { timeline.apply(result) }
    return timeline
}()

let file = try AVAudioFile(forReading: url)
_ = try await analyzer.analyzeSequence(from: file)
try await analyzer.finalizeAndFinishThroughEndOfInput()

let attributed = try await speakers.attributing(try await transcript)
for turn in attributed.speakerTurns {
    print(turn.speaker?.description ?? "Unknown", String(turn.text.characters))
}
```

Each result lists speaker segments in a source range;
overlapping speech appears as overlapping segments.
`SpeakerTimeline` collects results
and attributes each timed word to the speaker who overlaps it the most.
The speaker attribute is in the package's own attribute scope,
`AttributeScopes.AnySpeechRecognizerAttributes`.
Request `.audioTimeRange` from the transcriber so that words have times.
A transcriber usually recognizes only the louder voice in overlapping speech,
so a short interjection can lack its own words.

`SortformerDiarizationModel` runs NVIDIA's Streaming Sortformer on Core ML.
It reports up to four speakers, numbered from 0 in order of arrival
and stable within one analyzer session.
Accuracy degrades with five or more speakers.
Final results trail the input by about 2 seconds.
File input and live input give the same results,
except near a `finalize(through:)` call or a gap in the input.
Each model version needs about 246 MB.

- `.v2`, the default, is
  [Streaming Sortformer 4spk v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2),
  licensed under CC BY 4.0.
- `.v2_1` is
  [Streaming Sortformer 4spk v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1),
  which NVIDIA trained with additional meeting data.
  It has the same architecture and latency,
  and is licensed under the NVIDIA Open Model License (see [License](#license)).

The default compute units are `cpuAndNeuralEngine`.
`cpuAndGPU` runs about three times faster
but uses about 530 MB of memory instead of about 135 MB.
To use a staged model, pass its version and the compiled model:
`SortformerDiarizationModel(version: .v2_1, assets: .compiledModel(url))`.

## Custom providers

To add a recognition engine,
implement `SpeechRecognitionModel` for its configuration
and `SpeechRecognitionExecutor` for one module's session.
The executor pulls timed buffers from its input
and publishes results through a sink that validates their timing.
Capabilities declare the options the provider supports;
other requests fail before the analyzer reads audio.

This provider reports placeholder text instead of recognizing speech:

```swift
import AVFoundation
import AnySpeechRecognizer
import CoreMedia

struct ExampleModel: SpeechRecognitionModel {
    let modelID = "example/placeholder"
    let capabilities = SpeechRecognitionCapabilities(
        transcription: .init(reportingOptions: [.volatileResults], attributeOptions: []),
        liveInput: true
    )
    func makeExecutor() -> any SpeechRecognitionExecutor { ExampleExecutor() }
}

actor ExampleExecutor: SpeechRecognitionExecutor {
    private var results: SpeechRecognitionResultSink?
    private var finalized = CMTime.zero
    private var consumed = CMTime.zero

    func prepare(
        request: SpeechRecognitionRequest, format: AVAudioFormat, context: AnalysisContext
    ) async throws {
        try request.validate(for: ExampleModel().capabilities)
    }

    func analyze(_ input: SpeechAnalysisInput, results: SpeechRecognitionResultSink) async throws {
        self.results = results
        for try await buffer in input {
            consumed = buffer.endTime
            let pending = CMTimeRange(start: finalized, end: consumed)
            results.setVolatileRange(pending)
            try await results.emit(
                .init(text: "Speech so far", range: pending, resultsFinalizationTime: finalized)
            )
        }
    }

    func finalize(through time: CMTime) async throws {
        guard let results, time > finalized else { return }
        let range = CMTimeRange(start: finalized, end: min(time, consumed))
        try await results.emit(
            .init(text: "Final text. ", range: range, resultsFinalizationTime: range.end)
        )
        finalized = range.end
        results.setVolatileRange(nil)
    }

    func cancel() async {}
}

let transcriber = SpeechTranscriber(model: ExampleModel(), reportingOptions: [.volatileResults])
```

`finalize(through:)` and other controls refer to source times,
so the executor applies them when its input reaches them.
A provider whose assets depend on the request can implement
`availability(for:)` and `assetInstallationRequest(supporting:)`.

## Command-line tool

The `hear` executable transcribes, diarizes, and evaluates local files:

```console
$ swift run hear prepare --backend parakeet --live --diarize
$ swift run hear transcribe meeting.wav --backend parakeet --live --diarize
$ swift run hear transcribe recording.wav --backend apple --locale en-US --json
```

Every command requires an explicit `--backend`.
`--backend qwen` uses the built-in assets of the size that `--version` selects
(default: 1.7b) unless `--model-directory` names a staged directory,
and its `--locale` forces a language, such as `--locale ja`.
`prepare` is the only command that downloads;
`status` checks assets without downloading.
`--live` reads the file in 100 ms blocks with volatile results,
and `--diarize` adds a Sortformer diarizer
and prints one line for each speaker turn.
`--json` writes one record per line with word ranges, speaker segments,
and finalization times.
`evaluate` also reports elapsed time, peak memory,
and, with `--reference` or `--rttm`, the word error rate or diarization error rate.
Run `swift run hear --help` for all options.

## Accuracy and performance

We measured the `hear` CLI on an M4 Max Mac Studio with macOS 27.0.1.
Each LibriSpeech row has about 30 minutes of audio,
each FLEURS row the first 100 test files,
and the AMI row four test meetings (1.5 hours).
Word error rates are computed after lowercasing and removing punctuation.
Parakeet Unified recognizes English only, so it was not run on FLEURS.
Apple's transcriber was not run on FLEURS.

| Test set | Parakeet v3, file | v3, live | Ultra, file | Ultra, live | Unified, file | Unified, live | Apple, file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LibriSpeech test-clean, utterances | 2.57% | 3.28% | 2.27% | 2.51% | 2.07% | 2.24% | 2.81% |
| LibriSpeech test-other, utterances | 3.51% | 4.19% | 3.53% | 3.92% | 3.09% | 3.23% | 4.54% |
| LibriSpeech test-clean, 3–8 min files | 1.55% | 4.73% | 1.66% | 2.65% | 1.43% | 1.45% | 2.12% |
| LibriSpeech test-other, 3–8 min files | 2.78% | 4.29% | 2.55% | 3.28% | 2.19% | 2.42% | 3.97% |
| FLEURS German | 4.90% | 7.30% | 4.22% | 4.39% | | | |
| FLEURS French | 6.55% | 9.13% | 5.62% | 5.82% | | | |
| AMI meetings, headset mix | 26.72% | 23.95% | 25.17% | 23.56% | 25.25% | 25.41% | 29.24% |

Parakeet v3 and Ultra process an hour of audio in 18 to 31 seconds in file mode
and in 93 to 112 seconds in live mode;
Unified takes 33 to 42 seconds and 124 to 136 seconds.
Peak memory is about 100 MB except while Core ML compiles a model,
which can take about 680 MB.
Parakeet v3 and Ultra word start times have a median error of 60 ms
against forced alignments of LibriSpeech.

For Qwen3-ASR, the table below gives character error rates on the first 100 FLEURS test files,
counted after NFKC normalization, without punctuation, symbols, or spaces;
Cantonese texts are first converted to simplified characters with OpenCC,
because the references use traditional characters and the model writes either script.
The references write numbers as digits, Latin names in Latin script,
and some English glosses in parentheses that speakers did not always read.
The model often writes such numbers in words and such names in the local script,
so these rates are higher than the model card's.
The transformers columns transcribe each file in one call with greedy decoding on the CPU;
the package's runtime stays within 0.32 points of them.

| Test set | 0.6B | 0.6B, transformers | 1.7B | 1.7B, transformers |
| --- | --- | --- | --- | --- |
| FLEURS Mandarin | 7.11% | 7.14% | 6.28% | 6.25% |
| FLEURS Japanese | 9.49% | 9.31% | 6.10% | 6.08% |
| FLEURS Korean | 5.29% | 4.97% | 3.93% | 3.80% |
| FLEURS Cantonese | 8.15% | | 6.25% | |
| LibriSpeech test-clean, utterances (WER) | 2.14% | 2.10% | 1.97% | 1.86% |
| LibriSpeech test-other, utterances (WER) | 4.65% | 4.63% | 3.46% | 3.40% |

On the other sets of the first table,
Qwen3-ASR had word error rates of 1.68% with 0.6B and 1.58% with 1.7B
on the test-clean 3–8 min files,
2.63% and 1.94% on the test-other 3–8 min files,
6.67% and 3.88% on FLEURS German, 8.71% and 5.12% on FLEURS French,
and 26.31% and 24.49% on the AMI meetings.

On the sets of this table, Qwen3-ASR file mode took 10.5 to 15.4 minutes per hour of audio
with 0.6B and 14.5 to 21.2 minutes with 1.7B;
the 3–8 min files took 7.9 to 8.1 minutes per hour with 0.6B
and 12.6 to 12.9 minutes with 1.7B.
These times leave out model loading and Core ML compilation,
and each is the median of two runs that differed by less than 2%.
Peak memory was 1.7 GB for 0.6B and 4.4 GB for 1.7B.

With Parakeet v3 in the same analyzer,
Sortformer's diarization error rate on the 16 AMI test meetings,
with overlapping speech and no collar,
was 25.73% for v2 and 23.42% for v2.1.

## Limitations

- Fast results are not usable on natural speech:
  they insert repeated filler, such as "and the other thing",
  and reach 15% to 22% WER on long recordings.
  Live mode does this less often.
- Parakeet sometimes decodes a window of speech as silence,
  mostly in live mode and in meetings with overlapping speech.
- Parakeet Unified's word start times are currently about 300 ms late.
- Parakeet on a physical iPhone, the OS 26 runtime,
  and the minimum Swift 6.2 toolchain are not yet tested.
- Qwen3-ASR transcribes files only, with no word timing and no context prompt,
  and it is not tested on iOS devices.
  Its decoder runs only on the GPU, and file mode takes 8 to 21 minutes per hour of audio,
  where Parakeet v3 and Ultra take 18 to 31 seconds.
- With Apple's analyzer, `finalize(through:)` for the newest audio
  can wait until more input arrives while the input stream stays open.
- An interrupted download keeps the files that finished,
  but a partial file starts over
  ([swift-huggingface#97](https://github.com/huggingface/swift-huggingface/issues/97)).

## Development

The repository uses [mise](https://mise.jdx.dev/) tasks.
Swift comes from the selected Xcode.

| Task | Description |
| --- | --- |
| `mise run setup` | Resolve package dependencies. |
| `mise run build` | Build the library and CLI. |
| `mise run build:ios` | Compile the library for iOS. |
| `mise run test` | Run the tests without model downloads. |
| `mise run test:compatibility` | Build the [fixtures](Tests/Fixtures/Compatibility) with `import Speech` and with `import AnySpeechRecognizer`. |
| `mise run test:integration` | Run opt-in tests with local recordings and installed assets. |
| `mise run evaluate` | Run `hear evaluate` on a local recording. |
| `mise run prepare:apple` | Download and prepare Apple Speech assets. |
| `mise run prepare:parakeet` | Download and prepare Parakeet assets. |

Integration tests read their inputs from environment variables:
`ANYSPEECH_INTEGRATION_AUDIO` selects the Apple tests,
`ANYSPEECH_PARAKEET_MODELS` with `ANYSPEECH_PARAKEET_LONG_AUDIO`
selects the Parakeet tests,
and `ANYSPEECH_QWEN_MODELS` with `ANYSPEECH_QWEN_AUDIO` selects the Qwen3-ASR tests.

The [`Tools`](Tools) directory contains the development tools:

- [`ParakeetConversion`](Tools/ParakeetConversion) and
  [`SortformerConversion`](Tools/SortformerConversion)
  convert NVIDIA's checkpoints to the Core ML models that the package downloads.
- [`QwenConversion`](Tools/QwenConversion) converts, verifies, and packages Qwen3-ASR
  for `QwenSpeechModel`.
- `make-parakeet-manifest.py`, `make-sortformer-manifest.py`, and `make-qwen-manifest.py`
  generate the pinned model manifests.
- `natural-speech-eval.py` downloads the test sets
  and reproduces the measurements above.
- `make-evaluation-audio.swift` and `make-diarization-audio.swift`
  generate synthetic recordings for stress and diarization tests.

## License

AnySpeechRecognizer is available under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for details.

The package downloads model weights at run time
and does not include them.
If an app downloads or bundles a model,
list the model in the app's acknowledgments
with its license and any required notice.
The `attribution` property of each model has this text.
If the app distributes the model files, for example in an app bundle,
also include the license text and the notice file from the model's repository.

These models are licensed under CC BY 4.0:

- Parakeet v3 by NVIDIA
  ([source](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
  [Core ML conversion](https://huggingface.co/mattt/parakeet-tdt-0.6b-v3-coreml)).
- Parakeet Ultra by Moondream
  ([source](https://huggingface.co/moondream/parakeet-ultra),
  [Core ML conversion](https://huggingface.co/mattt/parakeet-ultra-coreml)).
- Sortformer v2 by NVIDIA
  ([source](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2),
  [Core ML conversion](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2-coreml)).

These models are licensed under the
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0):

- Qwen3-ASR 0.6B by the Qwen team, Alibaba Cloud
  ([source](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf),
  [Core ML conversion](https://huggingface.co/mattt/qwen3-asr-0.6b-coreml)).
- Qwen3-ASR 1.7B by the Qwen team, Alibaba Cloud
  ([source](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf),
  [Core ML conversion](https://huggingface.co/mattt/qwen3-asr-1.7b-coreml)).

These models are licensed under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/):

- Parakeet Unified by NVIDIA
  ([source](https://huggingface.co/nvidia/parakeet-unified-en-0.6b),
  [Core ML conversion](https://huggingface.co/mattt/parakeet-unified-en-0.6b-coreml)).
- Sortformer v2.1 by NVIDIA
  ([source](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1),
  [Core ML conversion](https://huggingface.co/mattt/diar-streaming-sortformer-4spk-v2.1-coreml)).

<details>
<summary>NVIDIA Open Model License conditions</summary>

The Core ML conversions are Derivative Models under the
NVIDIA Open Model License Agreement, version released October 24, 2025.
Section 3.1 of the agreement requires this notice
in a `Notice` text file with every copy of the model files:

```text
Licensed by NVIDIA Corporation under the NVIDIA Open Model License
```

Each conversion's repository has the agreement in its `LICENSE` file
and the notice in its `NOTICE` file.
Each model's `attribution` text also contains the notice.
Use must be consistent with NVIDIA's
[Trustworthy AI terms](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/),
which prohibit use in violation of applicable law,
such as illegal surveillance
or illegal collection or processing of biometric information.

</details>

## Acknowledgments

The Parakeet runtime follows the window layout, duration mapping,
and frame padding of
[FluidAudio](https://github.com/FluidInference/FluidAudio),
and the conversion tools derive from FluidInference's
[möbius](https://github.com/FluidInference/mobius);
both are licensed under the Apache License, Version 2.0.
The Sortformer speaker cache, features, chunking, and post-processing
port or follow [NVIDIA NeMo](https://github.com/NVIDIA/NeMo),
licensed under the Apache License, Version 2.0.
