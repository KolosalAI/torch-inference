#!/usr/bin/env bash
set -euo pipefail

# Download MiniCPM-V 2.6 Q2_K (~2 GB) + mmproj (~400 MB) from HuggingFace.
# Models are placed in services/llm/models/.

MODELS_DIR="$(dirname "$0")/../services/llm/models"
mkdir -p "$MODELS_DIR"

MODEL_URL="https://huggingface.co/bartowski/MiniCPM-V-2_6-GGUF/resolve/main/MiniCPM-V-2_6-Q2_K.gguf"
MMPROJ_URL="https://huggingface.co/bartowski/MiniCPM-V-2_6-GGUF/resolve/main/mmproj-MiniCPM-V-2_6-f16.gguf"

MODEL_FILE="$MODELS_DIR/minicpm-v-2_6-q2_k.gguf"
MMPROJ_FILE="$MODELS_DIR/minicpm-v-2_6-mmproj-f16.gguf"

echo "=== Downloading MiniCPM-V 2.6 Q2_K (~2 GB) ==="
if [ ! -f "$MODEL_FILE" ]; then
    curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
    echo "Model saved to $MODEL_FILE"
else
    echo "Model already present: $MODEL_FILE"
fi

echo "=== Downloading mmproj (~400 MB) ==="
if [ ! -f "$MMPROJ_FILE" ]; then
    curl -L --progress-bar -o "$MMPROJ_FILE" "$MMPROJ_URL"
    echo "mmproj saved to $MMPROJ_FILE"
else
    echo "mmproj already present: $MMPROJ_FILE"
fi

echo ""
echo "All models ready. Run: make llm-run"
