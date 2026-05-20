#!/usr/bin/env bash
# Compatibility shim. The LLM service now uses HRM-Text. See the new script.
echo "scripts/download_llm_model.sh is deprecated."
echo "Run: bash scripts/download_hrm_text_artifacts.sh   (or: make hrm-download)"
exec bash "$(dirname "$0")/download_hrm_text_artifacts.sh" "$@"
