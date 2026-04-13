#!/usr/bin/env bash
# Download all inference models for torch-inference server.
#
# Models already present (committed via git-lfs or pre-seeded):
#   models/kokoro-82m/          — Kokoro ONNX TTS (primary engine, 88 MB + voices)
#   models/yolo/yolov8n.onnx    — YOLOv8 Nano object detection
#   models/classify/*.onnx      — EfficientNet-Lite4 / MobileNetV2 image classification
#
# This script downloads the remaining models that are NOT committed to git:
#   models/tts/piper_lessac/    — Piper en_US-lessac-medium ONNX TTS (~60 MB)
#   models/whisper/             — OpenAI Whisper base (~140 MB)
#
# Note: VITS, StyleTTS2, XTTS, Bark engines are stubs that delegate to Kokoro;
# their config.json files are in the repo but no large model files are needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HF_BASE="https://huggingface.co"
OPENAI_WHISPER="https://openaipublic.azureedge.net/main/whisper/models"

echo "=== Downloading Piper en_US-lessac-medium ONNX TTS ==="
mkdir -p models/tts/piper_lessac
if [[ ! -f models/tts/piper_lessac/model.onnx ]]; then
  curl -L --progress-bar \
    "${HF_BASE}/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
    -o models/tts/piper_lessac/model.onnx
  echo "  ✓ model.onnx"
else
  echo "  ✓ model.onnx (already present)"
fi

if [[ ! -f models/tts/piper_lessac/config.json ]]; then
  curl -L --progress-bar \
    "${HF_BASE}/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
    -o models/tts/piper_lessac/config.json
  echo "  ✓ config.json"
else
  echo "  ✓ config.json (already present)"
fi

echo ""
echo "=== Downloading OpenAI Whisper base STT model ==="
mkdir -p models/whisper
WHISPER_HASH="ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
if [[ ! -f models/whisper/whisper-base.pt ]]; then
  curl -L --progress-bar \
    "${OPENAI_WHISPER}/${WHISPER_HASH}/base.pt" \
    -o models/whisper/whisper-base.pt
  echo "  ✓ whisper-base.pt"
else
  echo "  ✓ whisper-base.pt (already present)"
fi

echo ""
echo "=== Creating stub engine directories ==="
for dir in models/vits models/styletts2 models/xtts models/bark; do
  mkdir -p "${dir}"
  echo "  ✓ ${dir}/"
done

echo ""
echo "=== All models ready ==="
echo ""
echo "Engine          Model                                        Size    Status"
echo "─────────────────────────────────────────────────────────────────────────"

check() {
  local label="$1" path="$2"
  if [[ -f "$path" ]]; then
    size=$(du -sh "$path" 2>/dev/null | cut -f1)
    printf "%-16s %-44s %-8s ✓\n" "$label" "$(basename "$path")" "$size"
  else
    printf "%-16s %-44s %-8s MISSING\n" "$label" "$(basename "$path")" "-"
  fi
}

check "Kokoro ONNX"   models/kokoro-82m/kokoro-v1.0.int8.onnx
check "YOLOv8n"       models/yolo/yolov8n.onnx
check "EfficientNet"  models/classify/efficientnet-lite4-11.onnx
check "MobileNetV2"   models/classify/mobilenetv2-7.onnx
check "Piper ONNX"    models/tts/piper_lessac/model.onnx
check "Whisper base"  models/whisper/whisper-base.pt

echo ""
echo "Kokoro voices: $(ls models/kokoro-82m/voices/*.bin 2>/dev/null | wc -l | tr -d ' ') files"
