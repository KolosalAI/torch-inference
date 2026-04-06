# Model Registry Expansion Design

**Date:** 2026-04-04  
**Status:** Approved  
**Scope:** Add all supportable models to both `models.json` and `model_registry.json`

---

## Goal

Populate both registry files with every model that the existing engine code can support, so the inference API exposes the full picture of available models without requiring code changes.

---

## Architecture

No code changes. This is a data-only expansion of two JSON registry files:

- **`models.json`** — detailed inference registry (`/models/available`). Full schema: hardware requirements, batch config, feature flags. Used by the engine to load and serve models.
- **`model_registry.json`** — lightweight SOTA/discovery registry (`/models/sota`). Used for listing and downloading models via the API.

New entries follow the exact schemas of existing entries in each file. All new entries: `"enabled": true`, `"auto_load": false`.

---

## Model Inventory

### TTS (19 new entries → `models.json` only; already present in `model_registry.json`)

| Key | Model | Source | Size |
|-----|-------|--------|------|
| `kokoro_v019` | Kokoro v0.19 | huggingface/hexgrad | 312 MB |
| `kokoro_v10` | Kokoro v1.0 | huggingface/hexgrad | 312 MB |
| `kokoro_onnx` | Kokoro v1.0 (ONNX) | huggingface/hexgrad | 326 MB |
| `kokoro_onnx_int8` | Kokoro v1.0 INT8 (ONNX) | github/thewh1teagle | 83 MB |
| `xtts_v2` | XTTS v2 | huggingface/coqui | ~2 GB |
| `piper_lessac` | Piper (Lessac) | huggingface/rhasspy | 60 MB |
| `styletts2` | StyleTTS2 | huggingface/yl4579 | ~500 MB |
| `f5_tts` | F5-TTS v1 | huggingface/SWivid | ~736 MB |
| `parler_tts_mini` | Parler-TTS Mini v1 | huggingface/parler-tts | ~3.6 GB |
| `chatterbox` | Chatterbox TTS | huggingface/ResembleAI | ~2.1 GB |
| `outetts_0_3_500m` | OuteTTS 0.3 (500M) | huggingface/OuteAI | ~1 GB |
| `sesame_csm_1b` | Sesame CSM 1B | huggingface/sesame | ~3.8 GB |
| `cosyvoice2_0_5b` | CosyVoice2 0.5B | huggingface/FunAudioLLM | ~2 GB |
| `index_tts_2` | IndexTTS-2 | huggingface/IndexTeam | ~2 GB |
| `melotts_english_v3` | MeloTTS English v3 | huggingface/myshell-ai | ~208 MB |
| `bark_small` | Bark Small | huggingface/suno | ~1.5 GB |
| `matcha_tts` | Matcha-TTS | huggingface/shivammehta25 | ~200 MB |
| `vits_vctk` | VITS VCTK | huggingface/jaywalnut310 | ~175 MB |
| `amphion_maskgct` | Amphion MaskGCT | huggingface/amphion | ~2 GB |
| `metavoice` | MetaVoice | huggingface/metavoiceio | ~1 GB |
| `openvoice` | OpenVoice | huggingface/myshell-ai | ~600 MB |
| `fish_speech_v15` | Fish Speech v1.5 | huggingface/fishaudio | ~1 GB |

### STT (5 new entries → both files)

| Key | Model | Size | Notes |
|-----|-------|------|-------|
| `whisper_small` | Whisper Small | 244 MB | 39M params |
| `whisper_medium` | Whisper Medium | 769 MB | 307M params |
| `whisper_large` | Whisper Large | 1.5 GB | 1.5B params |
| `whisper_large_v3` | Whisper Large v3 | 1.5 GB | best accuracy |
| `whisper_turbo` | Whisper Turbo | 809 MB | 809M params, fast |

### Image Classification (19 new entries → both files)

**TorchVision family (6 new):**

| Key | Model | Source | Size |
|-----|-------|--------|------|
| `resnet34` | ResNet-34 | torchvision | 83.3 MB |
| `resnet50` | ResNet-50 | torchvision | 97.8 MB |
| `resnet101` | ResNet-101 | torchvision | 170.5 MB |
| `resnet152` | ResNet-152 | torchvision | 230.5 MB |
| `mobilenet_v3_small` | MobileNetV3 Small | torchvision | 9.8 MB |
| `mobilenet_v3_large` | MobileNetV3 Large | torchvision | 21.1 MB |

**TIMM SOTA (already in `model_registry.json`, new to `models.json`):**

| Key | Model | Accuracy |
|-----|-------|----------|
| `eva02_large` | EVA-02 Large | 90.054% |
| `eva_giant` | EVA Giant | 89.792% |
| `convnextv2_huge` | ConvNeXt V2 Huge | 88.848% |
| `convnext_xxlarge_clip` | ConvNeXt XXLarge CLIP | 88.612% |
| `maxvit_xlarge` | MaxViT XLarge | 88.53% |
| `beit_large` | BEiT Large | 88.6% |
| `swin_large` | Swin Transformer Large | 87.3% |
| `deit3_huge` | DeiT-III Huge | 87.7% |
| `vit_giant_clip` | ViT Giant (CLIP) | ~88.5% |
| `coatnet_3` | CoAtNet-3 | ~86.0% |
| `efficientnetv2_xl` | EfficientNetV2 XL | 87.3% |
| `mobilenetv4_hybrid_large` | MobileNetV4 Hybrid Large | 84.36% |

### Object Detection (12 new entries → both files)

| Key | Model | Source | mAP50-95 |
|-----|-------|--------|----------|
| `yolov8x` | YOLOv8 XLarge | ultralytics | 0.418 |
| `yolov5m` | YOLOv5 Medium | pytorch_hub | 0.283 |
| `yolov5l` | YOLOv5 Large | pytorch_hub | 0.318 |
| `yolov5x` | YOLOv5 XLarge | pytorch_hub | 0.337 |
| `yolov9c` | YOLOv9 Compact | ultralytics | 0.530 |
| `yolov9e` | YOLOv9 Extended | ultralytics | 0.555 |
| `yolov10n` | YOLOv10 Nano | ultralytics | 0.386 |
| `yolov10s` | YOLOv10 Small | ultralytics | 0.462 |
| `yolov10m` | YOLOv10 Medium | ultralytics | 0.512 |
| `yolov10b` | YOLOv10 Balanced | ultralytics | 0.526 |
| `yolov10l` | YOLOv10 Large | ultralytics | 0.534 |
| `yolov10x` | YOLOv10 XLarge | ultralytics | 0.544 |

### NLP (5 new entries → both files)

| Key | Model | Task | Size |
|-----|-------|------|------|
| `bert_base_uncased` | BERT Base Uncased | text-classification | 417.7 MB |
| `roberta_base` | RoBERTa Base | text-classification | 476.5 MB |
| `all_mpnet_base_v2` | all-mpnet-base-v2 | feature-extraction | 438 MB |
| `paraphrase_multilingual_mpnet` | paraphrase-multilingual-mpnet-base-v2 | feature-extraction | 1.11 GB |
| `bart_large_mnli` | BART Large MNLI | zero-shot-classification | 1.63 GB |

---

## Group Updates (`models.json`)

| Group | Added model IDs |
|-------|----------------|
| `text_to_speech` | all 22 new TTS keys |
| `speech_to_text` | whisper_small, whisper_medium, whisper_large, whisper_large_v3, whisper_turbo |
| `image_classification` | resnet34, resnet50, resnet101, resnet152, mobilenet_v3_small, mobilenet_v3_large, + 12 TIMM SOTA |
| `object_detection` | yolov8x, yolov5m/l/x, yolov9c/e, yolov10n/s/m/b/l/x |
| `text_classification` | bert_base_uncased, roberta_base |
| `feature_extraction` | all_mpnet_base_v2, paraphrase_multilingual_mpnet |
| `zero_shot_classification` (new group) | bart_large_mnli |

## Hardware Profile Updates (`models.json`)

- **`cpu_only`**: add whisper_small, whisper_medium, resnet34/50, mobilenet_v3_small/large, mobilenet_v4_hybrid_large, yolov8x, yolov5m/l/x, yolov10n/s, bert_base_uncased, roberta_base, all_mpnet_base_v2, kokoro_onnx, kokoro_onnx_int8, piper_lessac, melotts_english_v3, vits_vctk, matcha_tts
- **`gpu_basic`**: above + whisper_large, resnet101/152, yolov9c, yolov10m/b, swin_large, deit3_huge, coatnet_3, efficientnetv2_xl, xtts_v2, styletts2, f5_tts, bark_small, metavoice, openvoice, fish_speech_v15, cosyvoice2_0_5b, index_tts_2, chatterbox, paraphrase_multilingual_mpnet, bart_large_mnli
- **`gpu_advanced`**: all models (empty blocked_models list)

---

## Summary

- **`models.json`**: ~61 new entries + group/profile updates  
- **`model_registry.json`**: ~23 new entries  
- No code changes required  
- All entries: `"enabled": true`, `"auto_load": false`
