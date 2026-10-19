# ``AnySpeechRecognizer``

Analyze speech with Apple's modern Speech API and a choice of recognition engine.

## Overview

AnySpeechRecognizer implements ``SpeechAnalyzer``, ``SpeechTranscriber``,
``DictationTranscriber``, ``SpeechDetector``, and ``AssetInventory``
with the same call shape as Apple's Speech framework.
Create a ``SpeechTranscriber`` with a locale for Apple Speech,
or with a ``SpeechRecognitionModel`` such as ``ParakeetSpeechModel``.
Use ``AssetInventory`` to install assets explicitly;
analysis never downloads.
Consume each module's results while ``SpeechAnalyzer`` reads a file or live input,
then finalize the analyzer to close result delivery.

``SpeakerDiarizer`` is a package extension that identifies speakers
on the same timeline as a transcriber.

Unsupported configuration fails before input is consumed.
If input or result consumption stops early, call
``SpeechAnalyzer/cancelAndFinishNow()`` and wait for cleanup.

## Topics

### Analysis

- ``SpeechAnalyzer``
- ``SpeechTranscriber``
- ``DictationTranscriber``
- ``SpeechDetector``
- ``SpeechModule``

### Models

- ``SystemSpeechModel``
- ``ParakeetSpeechModel``
- ``NemotronSpeechModel``

### Custom providers

- ``SpeechRecognitionModel``
- ``SpeechRecognitionExecutor``
- ``SpeechRecognitionCapabilities``
- ``SpeechRecognitionConfiguration``
- ``SpeechRecognitionRequest``
- ``SpeechRecognitionResultSink``
- ``SpeechAnalysisInput``

### Speaker diarization (package extension)

- ``SpeakerDiarizer``
- ``SortformerDiarizationModel``
- ``SpeakerDiarizationModel``
- ``SpeakerDiarizationExecutor``
- ``SpeakerDiarizationCapabilities``
- ``SpeakerDiarizationRequest``
- ``SpeakerTimeline``
- ``SpeakerTurn``

### Preparation and diagnosis

- ``AssetInventory``
- ``AssetInstallationRequest``
- ``SpeechRecognitionAvailability``
- ``SpeechRecognitionDiagnostics``
- ``SpeechRecognitionError``
