# Registry Downloadability Design

**Date:** 2026-04-04  
**Status:** Approved

---

## Goal

Make every model in `model_registry.json` actually downloadable via the existing `/models/sota/{model_id}` API endpoint, and ensure downloaded files land in the correct on-disk directory.

---

## Problem Summary

Two structural bugs prevent downloads from working correctly:

### Bug 1: `model_type` missing from all registry entries

All 65 entries in `model_registry.json` lack a `model_type` field. The `download_model_async` function (`src/api/models.rs:258`) defaults to `"tts"` when the field is empty, so every model — YOLO, Whisper, ResNet — lands in `models/tts/` regardless of its actual type.

Additionally, the `cache_dir` match block has no arms for `"speech-to-text"`, `"text-classification"`, `"feature-extraction"`, or `"nlp"`, so those types would fall through to `models/other/`.

### Bug 2: Non-HuggingFace URLs silently skipped

The handler has three branches:
- `url.contains("resolve")` → direct file download
- `url.starts_with("https://huggingface.co/")` → repo download
- else → `log::warn!` only, nothing downloaded

YOLO models use GitHub release URLs (`https://github.com/ultralytics/...`). Torchvision models use `https://download.pytorch.org/models/...`. Both hit the `else` branch — a warning is logged, no file is saved.

---

## Fix A: Add `model_type` to `model_registry.json`

Add a `"model_type"` string field to every entry using these values:

| Entry category | `model_type` value |
|---|---|
| TTS models (Kokoro, Bark, XTTS, Piper, F5-TTS, StyleTTS2, etc.) | `"tts"` |
| STT models (Whisper variants) | `"speech-to-text"` |
| Image classification (ResNet, MobileNet, EVA, ConvNeXt, ViT, etc.) | `"image-classification"` |
| Object detection (YOLO variants) | `"object-detection"` |
| Text classification (BERT, RoBERTa) | `"text-classification"` |
| Feature extraction (MPNet, multilingual MPNet) | `"feature-extraction"` |
| Zero-shot classification (BART MNLI) | `"text-classification"` |

---

## Fix B: Extend `download_model_async` in `src/api/models.rs`

### Change 1: Add missing `cache_dir` arms (lines 264–271)

Add three new match arms after the existing ones:

```rust
"speech-to-text" => std::path::Path::new("models/stt").join(model_id),
"text-classification" | "feature-extraction" => std::path::Path::new("models/nlp").join(model_id),
"nlp" => std::path::Path::new("models/nlp").join(model_id),
```

### Change 2: Add 4th download branch (after line 315)

After the `else if model.url.starts_with("https://huggingface.co/")` block and before the final `else` warning, add:

```rust
} else if model.url.starts_with("https://") && {
    let u = model.url.as_str();
    u.ends_with(".pt") || u.ends_with(".pth") || u.ends_with(".onnx")
        || u.ends_with(".bin") || u.ends_with(".safetensors") || u.ends_with(".gguf")
} {
    // Direct HTTPS file download (GitHub releases, pytorch.org, etc.)
    let extension = model.url.rsplit('.').next().unwrap_or("bin");
    let filepath = cache_dir.join(format!("model.{}", extension));
    download_file_streaming(get_http_client(), &model.url, &filepath).await?;
    log::info!("Downloaded {} to {:?}", model.name, filepath);
```

---

## Scope

- **In scope:** `model_registry.json` `model_type` population; `cache_dir` match extension; direct HTTPS download branch
- **Out of scope:** HuggingFace authentication tokens; URL reachability checking; `models.json` changes; new download formats
