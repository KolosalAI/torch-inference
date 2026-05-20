#!/usr/bin/env bash
set -euo pipefail

# Downloads pre-exported HRM-Text-1B artifacts from a GitHub Release on
# KolosalAI/torch-inference. Set RELEASE_TAG to pin a specific release.

RELEASE_TAG="${RELEASE_TAG:-hrm-text-1b-v0}"
ASSET_NAME="hrm-text-1b.tar.gz"
OUT_DIR="services/llm/models/hrm-text-1b"
URL="https://github.com/KolosalAI/torch-inference/releases/download/${RELEASE_TAG}/${ASSET_NAME}"

if [ -f "${OUT_DIR}/model.onnx" ]; then
    echo "Artifacts already present at ${OUT_DIR}/. Delete to re-download."
    exit 0
fi

mkdir -p "${OUT_DIR}"
echo "Downloading ${URL}..."
curl -L --fail --progress-bar -o /tmp/hrm-text-1b.tar.gz "${URL}"
tar -xzf /tmp/hrm-text-1b.tar.gz -C "${OUT_DIR}" --strip-components=1
rm /tmp/hrm-text-1b.tar.gz
echo "Done. Artifacts at ${OUT_DIR}/"
