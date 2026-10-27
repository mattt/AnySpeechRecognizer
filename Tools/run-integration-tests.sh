#!/usr/bin/env bash
# Runs the opt-in recognition tests whose inputs are set in the environment.
# Apple:    ANYSPEECH_INTEGRATION_AUDIO=<speech file>
# Parakeet: ANYSPEECH_PARAKEET_MODELS=<staged model directory>
#           ANYSPEECH_PARAKEET_LONG_AUDIO=<speech longer than 15 s>
#           ANYSPEECH_PARAKEET_SHORT_AUDIO=<speech shorter than 15 s> (optional)
# Qwen:     ANYSPEECH_QWEN_MODELS=<directory from Tools/QwenConversion/convert.py>
#           ANYSPEECH_QWEN_AUDIO=<speech file>
set -euo pipefail

cd "$(dirname "$0")/.."
filters=()
if [ -n "${ANYSPEECH_INTEGRATION_AUDIO:-}" ]; then
    filters+=(--filter AppleIntegration)
fi
if [ -n "${ANYSPEECH_PARAKEET_MODELS:-}" ]; then
    filters+=(--filter ParakeetAnalyzer --filter ParakeetModelTests)
fi
if [ -n "${ANYSPEECH_QWEN_MODELS:-}" ]; then
    filters+=(--filter QwenAnalyzer)
fi
if [ "${#filters[@]}" -eq 0 ]; then
    echo "error: Set ANYSPEECH_INTEGRATION_AUDIO, ANYSPEECH_PARAKEET_MODELS, or ANYSPEECH_QWEN_MODELS." >&2
    exit 1
fi
exec xcrun swift test --no-parallel "${filters[@]}"
