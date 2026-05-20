#!/usr/bin/env bash
set -euo pipefail

# Download LLaVA-1.6-Mistral-7B IQ1_S (~1.8 GB) + mmproj (~624 MB) from HuggingFace.
# IQ1_S = true 1-bit quantisation via llama.cpp. Supports text + image input.
# Models are placed in services/llm/models/.

MODELS_DIR="$(dirname "$0")/../services/llm/models"
mkdir -p "$MODELS_DIR"

MODEL_URL="https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.IQ1_S.gguf"
MMPROJ_URL="https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf"

MODEL_FILE="$MODELS_DIR/llava-v1.6-mistral-7b.IQ1_S.gguf"
MMPROJ_FILE="$MODELS_DIR/llava-v1.6-mistral-7b-mmproj-f16.gguf"

echo "=== Downloading LLaVA-1.6-Mistral-7B IQ1_S (~1.8 GB) ==="
if [ ! -f "$MODEL_FILE" ]; then
    curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
    echo "Model saved to $MODEL_FILE"
else
    echo "Model already present: $MODEL_FILE"
fi

echo "=== Downloading mmproj (~624 MB) ==="
if [ ! -f "$MMPROJ_FILE" ]; then
    curl -L --progress-bar -o "$MMPROJ_FILE" "$MMPROJ_URL"
    echo "mmproj saved to $MMPROJ_FILE"
else
    echo "mmproj already present: $MMPROJ_FILE"
fi

echo ""
echo "All models ready. Run: make llm-run"
