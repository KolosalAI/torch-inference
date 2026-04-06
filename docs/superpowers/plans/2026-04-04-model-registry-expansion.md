# Model Registry Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add all supportable models to both `models.json` (inference registry) and `model_registry.json` (SOTA discovery registry) without any code changes.

**Architecture:** Pure data expansion. `models.json` receives full inference entries (hardware reqs, batch config, feature flags). `model_registry.json` receives lightweight discovery entries. Both files use their existing schemas — no new fields introduced. All entries: `"enabled": true`, `"auto_load": false`.

**Tech Stack:** JSON, `jq` for validation, `cargo check` for compile-time regression check.

---

## File Map

| File | Change |
|------|--------|
| `models.json` | Add ~61 entries to `available_models`; update `model_groups`; update `hardware_profiles` |
| `model_registry.json` | Add ~23 entries to `models` object |

---

### Task 1: Add TTS models to `models.json`

**Files:**
- Modify: `models.json` — `available_models` object

Add the following 22 entries inside the `"available_models"` object in `models.json`. Insert them after the existing `"bark_tts"` entry. All entries follow the same schema as `"bark_tts"`.

- [ ] **Step 1: Add kokoro_v019 through fish_speech_v15 inside `available_models`**

Open `models.json`. After the closing `}` of the `"bark_tts"` entry (and its trailing comma), insert:

```json
    "kokoro_v019": {
      "name": "kokoro_v019",
      "display_name": "Kokoro v0.19",
      "description": "Lightweight 82M-parameter multi-voice TTS model based on StyleTTS2 + ISTFTNet",
      "source": "huggingface",
      "model_id": "hexgrad/kLegacy",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 2048,
        "gpu_required": false,
        "min_gpu_memory_mb": 1024
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 60,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 82000000,
        "size_mb": 312,
        "architecture": "StyleTTS2 + ISTFTNet",
        "framework": "pytorch",
        "version": "0.19",
        "languages": ["en"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "fast",
        "supports_voice_cloning": false,
        "supports_emotions": false,
        "max_text_length": 2000
      }
    },
    "kokoro_v10": {
      "name": "kokoro_v10",
      "display_name": "Kokoro v1.0",
      "description": "Improved Kokoro with 54 voices; StyleTTS2 + ISTFTNet architecture",
      "source": "huggingface",
      "model_id": "hexgrad/Kokoro-82M",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 2,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 2048,
        "gpu_required": false,
        "min_gpu_memory_mb": 1024
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 60,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 82000000,
        "size_mb": 312,
        "architecture": "StyleTTS2 + ISTFTNet",
        "framework": "pytorch",
        "version": "1.0",
        "languages": ["en"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "fast",
        "supports_voice_cloning": false,
        "supports_emotions": false,
        "max_text_length": 2000
      }
    },
    "kokoro_onnx": {
      "name": "kokoro_onnx",
      "display_name": "Kokoro v1.0 (ONNX)",
      "description": "Kokoro v1.0 exported to ONNX — cross-platform inference without PyTorch",
      "source": "huggingface",
      "model_id": "hexgrad/Kokoro-82M",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 2,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 1024,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 82000000,
        "size_mb": 326,
        "architecture": "StyleTTS2 + ISTFTNet (ONNX)",
        "framework": "onnxruntime",
        "version": "1.0",
        "languages": ["en"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "fast",
        "supports_voice_cloning": false,
        "supports_emotions": false,
        "max_text_length": 2000
      }
    },
    "kokoro_onnx_int8": {
      "name": "kokoro_onnx_int8",
      "display_name": "Kokoro v1.0 INT8 (ONNX)",
      "description": "INT8-quantized Kokoro — ~4x smaller, faster on CPU",
      "source": "github",
      "model_id": "thewh1teagle/kokoro-onnx",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 1,
      "hardware_requirements": {
        "min_memory_mb": 256,
        "recommended_memory_mb": 512,
        "gpu_required": false,
        "min_gpu_memory_mb": 0
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 82000000,
        "size_mb": 83,
        "architecture": "StyleTTS2 + ISTFTNet (ONNX INT8)",
        "framework": "onnxruntime",
        "version": "1.0",
        "languages": ["en"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "very_fast",
        "supports_voice_cloning": false,
        "supports_emotions": false,
        "max_text_length": 2000
      }
    },
    "xtts_v2": {
      "name": "xtts_v2",
      "display_name": "XTTS v2",
      "description": "Coqui XTTS v2 — high-quality zero-shot voice cloning via Transformer + HiFiGAN",
      "source": "huggingface",
      "model_id": "coqui/XTTS-v2",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 467000000,
        "size_mb": 1900,
        "architecture": "Transformer + HiFiGAN",
        "framework": "pytorch",
        "version": "2.0",
        "languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "ko", "hu"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "medium",
        "supports_voice_cloning": true,
        "supports_emotions": false,
        "max_text_length": 500
      }
    },
    "piper_lessac": {
      "name": "piper_lessac",
      "display_name": "Piper (Lessac Medium)",
      "description": "Piper VITS-based ONNX TTS — fast, lightweight, runs entirely on CPU",
      "source": "huggingface",
      "model_id": "rhasspy/piper-voices",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 1,
      "hardware_requirements": {
        "min_memory_mb": 128,
        "recommended_memory_mb": 256,
        "gpu_required": false,
        "min_gpu_memory_mb": 0
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 30,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 28000000,
        "size_mb": 60,
        "architecture": "VITS (ONNX)",
        "framework": "onnxruntime",
        "version": "1.0",
        "languages": ["en"],
        "sample_rate": 22050
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "medium",
        "speed": "very_fast",
        "supports_voice_cloning": false,
        "supports_emotions": false,
        "max_text_length": 5000
      }
    },
    "styletts2": {
      "name": "styletts2",
      "display_name": "StyleTTS2",
      "description": "StyleTTS2 base model for LJSpeech — single-speaker, high naturalness",
      "source": "huggingface",
      "model_id": "yl4579/StyleTTS2-LJSpeech",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 1024,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 90,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 148000000,
        "size_mb": 500,
        "architecture": "StyleTTS2",
        "framework": "pytorch",
        "version": "2.0",
        "languages": ["en"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "medium",
        "supports_voice_cloning": false,
        "supports_emotions": true,
        "max_text_length": 2000
      }
    },
    "f5_tts": {
      "name": "f5_tts",
      "display_name": "F5-TTS v1",
      "description": "Flow Matching DiT TTS — zero-shot voice cloning, non-autoregressive, MIT licensed",
      "source": "huggingface",
      "model_id": "SWivid/F5-TTS",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 6144,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 335000000,
        "size_mb": 736,
        "architecture": "Flow Matching (DiT)",
        "framework": "pytorch",
        "version": "1.0",
        "languages": ["en", "zh"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "very_high",
        "speed": "medium",
        "supports_voice_cloning": true,
        "supports_emotions": false,
        "max_text_length": 1000
      }
    },
    "parler_tts_mini": {
      "name": "parler_tts_mini",
      "display_name": "Parler-TTS Mini v1",
      "description": "Conditional 0.9B transformer TTS — controllable gender, pace, pitch via natural language prompt",
      "source": "huggingface",
      "model_id": "parler-tts/parler-tts-mini-v1",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 8192,
        "recommended_memory_mb": 16384,
        "gpu_required": true,
        "min_gpu_memory_mb": 6144
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 1,
        "timeout_seconds": 180,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 880000000,
        "size_mb": 3600,
        "architecture": "Conditional Transformer",
        "framework": "transformers",
        "version": "1.0",
        "languages": ["en"],
        "sample_rate": 44100
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "slow",
        "supports_voice_cloning": false,
        "supports_emotions": true,
        "max_text_length": 500
      }
    },
    "chatterbox": {
      "name": "chatterbox",
      "display_name": "Chatterbox TTS",
      "description": "ResembleAI Chatterbox — emotion exaggeration control, Perth watermarking, 23 languages",
      "source": "huggingface",
      "model_id": "ResembleAI/chatterbox",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 500000000,
        "size_mb": 2100,
        "architecture": "Llama 0.5B backbone",
        "framework": "pytorch",
        "version": "1.0",
        "languages": ["en", "de", "es", "fr", "it", "ja", "ko", "pl", "pt", "ru", "zh"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "very_high",
        "speed": "medium",
        "supports_voice_cloning": true,
        "supports_emotions": true,
        "max_text_length": 1000
      }
    },
    "outetts_0_3_500m": {
      "name": "outetts_0_3_500m",
      "display_name": "OuteTTS 0.3 (500M)",
      "description": "Qwen2.5-0.5B-based TTS with voice cloning; 6 languages; GGUF quantizations available",
      "source": "huggingface",
      "model_id": "OuteAI/OuteTTS-0.3-500M",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 6144,
        "gpu_required": true,
        "min_gpu_memory_mb": 3072
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 500000000,
        "size_mb": 1000,
        "architecture": "Qwen2.5-0.5B based",
        "framework": "pytorch",
        "version": "0.3",
        "languages": ["en", "ja", "ko", "zh", "fr", "de"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "medium",
        "supports_voice_cloning": true,
        "supports_emotions": false,
        "max_text_length": 1000
      }
    },
    "sesame_csm_1b": {
      "name": "sesame_csm_1b",
      "display_name": "Sesame CSM 1B",
      "description": "SOTA contextual/conversational speech — Llama backbone + RVQ audio decoder, MOS 4.7",
      "source": "huggingface",
      "model_id": "sesame/csm-1b",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 8192,
        "recommended_memory_mb": 16384,
        "gpu_required": true,
        "min_gpu_memory_mb": 8192
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 1,
        "timeout_seconds": 180,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 1000000000,
        "size_mb": 3800,
        "architecture": "Llama backbone + RVQ audio decoder",
        "framework": "pytorch",
        "version": "1.0",
        "languages": ["en"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "sota",
        "speed": "slow",
        "supports_voice_cloning": true,
        "supports_emotions": true,
        "max_text_length": 1000
      }
    },
    "cosyvoice2_0_5b": {
      "name": "cosyvoice2_0_5b",
      "display_name": "CosyVoice2 0.5B",
      "description": "LLM-based TTS with SOTA content consistency; 9 languages + 18 Chinese dialects",
      "source": "huggingface",
      "model_id": "FunAudioLLM/CosyVoice2-0.5B",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 500000000,
        "size_mb": 2000,
        "architecture": "LLM-based TTS",
        "framework": "pytorch",
        "version": "2.0",
        "languages": ["zh", "en", "ja", "ko", "fr", "de", "es", "ru", "pt"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "very_high",
        "speed": "medium",
        "supports_voice_cloning": true,
        "supports_emotions": false,
        "max_text_length": 1000
      }
    },
    "index_tts_2": {
      "name": "index_tts_2",
      "display_name": "IndexTTS-2",
      "description": "2025 SOTA zero-shot controllable TTS — detailed expressiveness control",
      "source": "huggingface",
      "model_id": "IndexTeam/IndexTTS-2",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 1,
        "timeout_seconds": 180,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 800000000,
        "size_mb": 2000,
        "architecture": "Zero-shot controllable TTS",
        "framework": "pytorch",
        "version": "2.0",
        "languages": ["en", "zh"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "sota",
        "speed": "slow",
        "supports_voice_cloning": true,
        "supports_emotions": true,
        "max_text_length": 1000
      }
    },
    "melotts_english_v3": {
      "name": "melotts_english_v3",
      "display_name": "MeloTTS English v3",
      "description": "Fast real-time CPU inference; multiple English accents (US, UK, Indian, Australian)",
      "source": "huggingface",
      "model_id": "myshell-ai/MeloTTS-English-v3",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 2,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 1024,
        "gpu_required": false,
        "min_gpu_memory_mb": 0
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 30,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 65000000,
        "size_mb": 208,
        "architecture": "VITS/VITS2",
        "framework": "pytorch",
        "version": "3.0",
        "languages": ["en"],
        "sample_rate": 44100
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "medium_high",
        "speed": "very_fast",
        "supports_voice_cloning": false,
        "supports_emotions": false,
        "max_text_length": 5000
      }
    },
    "bark_small": {
      "name": "bark_small",
      "display_name": "Bark Small",
      "description": "Smaller/faster Bark — same capabilities as full Bark, lower quality, 1.5 GB",
      "source": "huggingface",
      "model_id": "suno/bark-small",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 300000000,
        "size_mb": 1500,
        "architecture": "Bark (small)",
        "framework": "transformers",
        "version": "1.0",
        "languages": ["en", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "tr", "zh"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "medium_high",
        "speed": "medium",
        "supports_voice_cloning": true,
        "supports_emotions": true,
        "max_text_length": 250,
        "supports_music": true,
        "supports_sound_effects": true
      }
    },
    "matcha_tts": {
      "name": "matcha_tts",
      "display_name": "Matcha-TTS",
      "description": "Fast ODE-based synthesis via conditional flow matching; ONNX-exportable; LJSpeech quality",
      "source": "huggingface",
      "model_id": "shivammehta25/Matcha-TTS",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 1024,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 60,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 18000000,
        "size_mb": 200,
        "architecture": "Conditional Flow Matching",
        "framework": "pytorch",
        "version": "1.0",
        "languages": ["en"],
        "sample_rate": 22050
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "fast",
        "supports_voice_cloning": false,
        "supports_emotions": false,
        "max_text_length": 2000
      }
    },
    "vits_vctk": {
      "name": "vits_vctk",
      "display_name": "VITS VCTK",
      "description": "Multi-speaker VCTK VITS model — 109 speakers, fast real-time synthesis",
      "source": "huggingface",
      "model_id": "jaywalnut310/vits",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 1024,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 30,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 36000000,
        "size_mb": 175,
        "architecture": "VITS",
        "framework": "pytorch",
        "version": "1.0",
        "languages": ["en"],
        "sample_rate": 22050
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "medium_high",
        "speed": "very_fast",
        "supports_voice_cloning": false,
        "supports_emotions": false,
        "max_text_length": 2000
      }
    },
    "amphion_maskgct": {
      "name": "amphion_maskgct",
      "display_name": "Amphion MaskGCT",
      "description": "Masked Generative Codec Transformer — zero-shot voice cloning, non-autoregressive",
      "source": "huggingface",
      "model_id": "amphion/MaskGCT",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 400000000,
        "size_mb": 2000,
        "architecture": "Masked Generative Codec Transformer",
        "framework": "pytorch",
        "version": "1.0",
        "languages": ["en", "zh", "fr", "de", "ja", "ko", "es"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "very_high",
        "speed": "medium",
        "supports_voice_cloning": true,
        "supports_emotions": false,
        "max_text_length": 1000
      }
    },
    "metavoice": {
      "name": "metavoice",
      "display_name": "MetaVoice",
      "description": "MetaVoice-1B — 1B parameter voice cloning model from MetaVoice",
      "source": "huggingface",
      "model_id": "metavoiceio/metavoice-1B-v0.1",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 1200000000,
        "size_mb": 1000,
        "architecture": "Transformer",
        "framework": "pytorch",
        "version": "0.1",
        "languages": ["en"],
        "sample_rate": 24000
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "medium_high",
        "speed": "medium",
        "supports_voice_cloning": true,
        "supports_emotions": false,
        "max_text_length": 1000
      }
    },
    "openvoice": {
      "name": "openvoice",
      "display_name": "OpenVoice",
      "description": "MyShell OpenVoice — flexible voice style control with VITS backbone",
      "source": "huggingface",
      "model_id": "myshell-ai/OpenVoice",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 90,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 140000000,
        "size_mb": 600,
        "architecture": "VITS",
        "framework": "pytorch",
        "version": "1.0",
        "languages": ["en", "zh"],
        "sample_rate": 22050
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "medium",
        "speed": "fast",
        "supports_voice_cloning": true,
        "supports_emotions": false,
        "max_text_length": 2000
      }
    },
    "fish_speech_v15": {
      "name": "fish_speech_v15",
      "display_name": "Fish Speech v1.5",
      "description": "VQGAN + Llama TTS — multi-lingual, high quality, ~1GB",
      "source": "huggingface",
      "model_id": "fishaudio/fish-speech-1.5",
      "model_type": "tts",
      "task": "text-to-speech",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 500000000,
        "size_mb": 1000,
        "architecture": "VQGAN + Llama",
        "framework": "pytorch",
        "version": "1.5",
        "languages": ["en", "zh", "ja", "ko", "fr", "de", "es"],
        "sample_rate": 44100
      },
      "tts_features": {
        "supports_tts": true,
        "quality": "high",
        "speed": "medium",
        "supports_voice_cloning": true,
        "supports_emotions": false,
        "max_text_length": 1000
      }
    },
```

- [ ] **Step 2: Validate JSON syntax**

```bash
jq . models.json > /dev/null && echo "JSON valid" || echo "JSON INVALID"
```

Expected: `JSON valid`

- [ ] **Step 3: Commit**

```bash
git add models.json
git commit -m "feat(registry): add 22 TTS models to models.json"
```

---

### Task 2: Add Whisper STT variants to both files

**Files:**
- Modify: `models.json` — `available_models` object
- Modify: `model_registry.json` — `models` object

- [ ] **Step 1: Add Whisper variants to `models.json` after existing `whisper_base` entry**

```json
    "whisper_small": {
      "name": "whisper_small",
      "display_name": "Whisper Small",
      "description": "OpenAI Whisper small — 244 MB, 39M parameters, multilingual",
      "source": "huggingface",
      "model_id": "openai/whisper-small",
      "model_type": "speech",
      "task": "speech-to-text",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 2,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 1024,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 120,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 39000000,
        "size_mb": 244,
        "architecture": "Whisper",
        "framework": "transformers",
        "version": "1.0",
        "languages": ["multilingual"],
        "sample_rate": 16000
      },
      "stt_features": {
        "supports_stt": true,
        "quality": "medium",
        "speed": "fast",
        "supports_timestamps": true,
        "supports_language_detection": true,
        "max_audio_length_seconds": 3600
      }
    },
    "whisper_medium": {
      "name": "whisper_medium",
      "display_name": "Whisper Medium",
      "description": "OpenAI Whisper medium — 769 MB, 307M parameters, multilingual",
      "source": "huggingface",
      "model_id": "openai/whisper-medium",
      "model_type": "speech",
      "task": "speech-to-text",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 1024,
        "recommended_memory_mb": 2048,
        "gpu_required": false,
        "min_gpu_memory_mb": 1024
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 120,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 307000000,
        "size_mb": 769,
        "architecture": "Whisper",
        "framework": "transformers",
        "version": "1.0",
        "languages": ["multilingual"],
        "sample_rate": 16000
      },
      "stt_features": {
        "supports_stt": true,
        "quality": "high",
        "speed": "medium",
        "supports_timestamps": true,
        "supports_language_detection": true,
        "max_audio_length_seconds": 3600
      }
    },
    "whisper_large": {
      "name": "whisper_large",
      "display_name": "Whisper Large",
      "description": "OpenAI Whisper large — 1.5 GB, multilingual, highest base accuracy",
      "source": "huggingface",
      "model_id": "openai/whisper-large",
      "model_type": "speech",
      "task": "speech-to-text",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 6144,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 180,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 1500000000,
        "size_mb": 1500,
        "architecture": "Whisper",
        "framework": "transformers",
        "version": "1.0",
        "languages": ["multilingual"],
        "sample_rate": 16000
      },
      "stt_features": {
        "supports_stt": true,
        "quality": "very_high",
        "speed": "slow",
        "supports_timestamps": true,
        "supports_language_detection": true,
        "max_audio_length_seconds": 3600
      }
    },
    "whisper_large_v3": {
      "name": "whisper_large_v3",
      "display_name": "Whisper Large v3",
      "description": "OpenAI Whisper large-v3 — best accuracy, improved multilingual",
      "source": "huggingface",
      "model_id": "openai/whisper-large-v3",
      "model_type": "speech",
      "task": "speech-to-text",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 6144,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 180,
        "warmup_iterations": 2
      },
      "metadata": {
        "parameters": 1550000000,
        "size_mb": 1500,
        "architecture": "Whisper",
        "framework": "transformers",
        "version": "3.0",
        "languages": ["multilingual"],
        "sample_rate": 16000
      },
      "stt_features": {
        "supports_stt": true,
        "quality": "sota",
        "speed": "slow",
        "supports_timestamps": true,
        "supports_language_detection": true,
        "max_audio_length_seconds": 3600
      }
    },
    "whisper_turbo": {
      "name": "whisper_turbo",
      "display_name": "Whisper Turbo",
      "description": "OpenAI Whisper turbo — 809 MB, fast large-v3 distillation",
      "source": "huggingface",
      "model_id": "openai/whisper-large-v3-turbo",
      "model_type": "speech",
      "task": "speech-to-text",
      "category": "audio",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 1024,
        "recommended_memory_mb": 2048,
        "gpu_required": false,
        "min_gpu_memory_mb": 1024
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 120,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 809000000,
        "size_mb": 809,
        "architecture": "Whisper",
        "framework": "transformers",
        "version": "turbo",
        "languages": ["multilingual"],
        "sample_rate": 16000
      },
      "stt_features": {
        "supports_stt": true,
        "quality": "very_high",
        "speed": "fast",
        "supports_timestamps": true,
        "supports_language_detection": true,
        "max_audio_length_seconds": 3600
      }
    },
```

- [ ] **Step 2: Add Whisper variants to `model_registry.json` inside the `"models"` object**

```json
    "whisper-small": {
      "status": "Available",
      "architecture": "Whisper",
      "note": "39M parameters — fast multilingual transcription",
      "voices": "N/A",
      "quality": "Medium",
      "rank": 4,
      "size": "244 MB",
      "url": "https://huggingface.co/openai/whisper-small",
      "score": 0,
      "name": "Whisper Small"
    },
    "whisper-medium": {
      "status": "Available",
      "architecture": "Whisper",
      "note": "307M parameters — good accuracy/speed balance",
      "voices": "N/A",
      "quality": "High",
      "rank": 3,
      "size": "769 MB",
      "url": "https://huggingface.co/openai/whisper-medium",
      "score": 0,
      "name": "Whisper Medium"
    },
    "whisper-large": {
      "status": "Available",
      "architecture": "Whisper",
      "note": "1.5B parameters — high accuracy multilingual",
      "voices": "N/A",
      "quality": "Very High",
      "rank": 2,
      "size": "~1.5 GB",
      "url": "https://huggingface.co/openai/whisper-large",
      "score": 0,
      "name": "Whisper Large"
    },
    "whisper-large-v3": {
      "status": "Available",
      "architecture": "Whisper",
      "note": "Best multilingual accuracy; updated tokenizer",
      "voices": "N/A",
      "quality": "SOTA",
      "rank": 1,
      "size": "~1.5 GB",
      "url": "https://huggingface.co/openai/whisper-large-v3",
      "score": 0,
      "name": "Whisper Large v3"
    },
    "whisper-turbo": {
      "status": "Available",
      "architecture": "Whisper",
      "note": "Distilled large-v3 — fast inference, near-large accuracy",
      "voices": "N/A",
      "quality": "Very High",
      "rank": 2,
      "size": "809 MB",
      "url": "https://huggingface.co/openai/whisper-large-v3-turbo",
      "score": 0,
      "name": "Whisper Turbo"
    },
```

- [ ] **Step 3: Validate both files**

```bash
jq . models.json > /dev/null && echo "models.json valid" || echo "models.json INVALID"
jq . model_registry.json > /dev/null && echo "model_registry.json valid" || echo "model_registry.json INVALID"
```

Expected: both print `valid`

- [ ] **Step 4: Commit**

```bash
git add models.json model_registry.json
git commit -m "feat(registry): add Whisper small/medium/large/large-v3/turbo STT variants"
```

---

### Task 3: Add torchvision image classification variants to both files

**Files:**
- Modify: `models.json` — `available_models` object
- Modify: `model_registry.json` — `models` object

- [ ] **Step 1: Add ResNet and MobileNetV3 variants to `models.json` after the existing `mobilenet_v2` entry**

```json
    "resnet34": {
      "name": "resnet34",
      "display_name": "ResNet-34",
      "description": "34-layer residual network for image classification",
      "source": "torchvision",
      "model_id": "resnet34",
      "model_type": "cnn",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 256,
        "recommended_memory_mb": 1024,
        "gpu_required": false,
        "min_gpu_memory_mb": 256
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 32,
        "timeout_seconds": 30,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 21797672,
        "size_mb": 83.3,
        "architecture": "ResNet",
        "framework": "torchvision",
        "version": "1.0.0",
        "input_size": [224, 224],
        "num_classes": 1000,
        "pretrained": true,
        "accuracy_top1": 0.7361,
        "accuracy_top5": 0.9168
      }
    },
    "resnet50": {
      "name": "resnet50",
      "display_name": "ResNet-50",
      "description": "50-layer residual network — strong baseline for image classification",
      "source": "torchvision",
      "model_id": "resnet50",
      "model_type": "cnn",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 2048,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 16,
        "timeout_seconds": 30,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 25557032,
        "size_mb": 97.8,
        "architecture": "ResNet",
        "framework": "torchvision",
        "version": "1.0.0",
        "input_size": [224, 224],
        "num_classes": 1000,
        "pretrained": true,
        "accuracy_top1": 0.7613,
        "accuracy_top5": 0.9290
      }
    },
    "resnet101": {
      "name": "resnet101",
      "display_name": "ResNet-101",
      "description": "101-layer residual network — high accuracy image classification",
      "source": "torchvision",
      "model_id": "resnet101",
      "model_type": "cnn",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 1024,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 1024
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 44549160,
        "size_mb": 170.5,
        "architecture": "ResNet",
        "framework": "torchvision",
        "version": "1.0.0",
        "input_size": [224, 224],
        "num_classes": 1000,
        "pretrained": true,
        "accuracy_top1": 0.7745,
        "accuracy_top5": 0.9366
      }
    },
    "resnet152": {
      "name": "resnet152",
      "display_name": "ResNet-152",
      "description": "152-layer residual network — highest accuracy in standard ResNet family",
      "source": "torchvision",
      "model_id": "resnet152",
      "model_type": "cnn",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 60192808,
        "size_mb": 230.5,
        "architecture": "ResNet",
        "framework": "torchvision",
        "version": "1.0.0",
        "input_size": [224, 224],
        "num_classes": 1000,
        "pretrained": true,
        "accuracy_top1": 0.7830,
        "accuracy_top5": 0.9416
      }
    },
    "mobilenet_v3_small": {
      "name": "mobilenet_v3_small",
      "display_name": "MobileNetV3 Small",
      "description": "Ultra-lightweight MobileNetV3 small — optimized for edge/mobile CPU inference",
      "source": "torchvision",
      "model_id": "mobilenet_v3_small",
      "model_type": "cnn",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 1,
      "hardware_requirements": {
        "min_memory_mb": 64,
        "recommended_memory_mb": 256,
        "gpu_required": false,
        "min_gpu_memory_mb": 128
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 64,
        "timeout_seconds": 30,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 2542856,
        "size_mb": 9.8,
        "architecture": "MobileNetV3",
        "framework": "torchvision",
        "version": "1.0.0",
        "input_size": [224, 224],
        "num_classes": 1000,
        "pretrained": true,
        "accuracy_top1": 0.6757,
        "accuracy_top5": 0.8756
      }
    },
    "mobilenet_v3_large": {
      "name": "mobilenet_v3_large",
      "display_name": "MobileNetV3 Large",
      "description": "MobileNetV3 large variant — better accuracy than V2 with similar efficiency",
      "source": "torchvision",
      "model_id": "mobilenet_v3_large",
      "model_type": "cnn",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 2,
      "hardware_requirements": {
        "min_memory_mb": 128,
        "recommended_memory_mb": 512,
        "gpu_required": false,
        "min_gpu_memory_mb": 256
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 64,
        "timeout_seconds": 30,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 5483032,
        "size_mb": 21.1,
        "architecture": "MobileNetV3",
        "framework": "torchvision",
        "version": "1.0.0",
        "input_size": [224, 224],
        "num_classes": 1000,
        "pretrained": true,
        "accuracy_top1": 0.7402,
        "accuracy_top5": 0.9171
      }
    },
```

- [ ] **Step 2: Add torchvision models to `model_registry.json`**

```json
    "resnet34": {
      "status": "Available",
      "architecture": "ResNet",
      "task": "Image Classification",
      "note": "34-layer residual network",
      "accuracy": "73.61% Top-1",
      "rank": 14,
      "size": "83.3 MB",
      "url": "https://download.pytorch.org/models/resnet34-b627a593.pth",
      "score": 0,
      "name": "ResNet-34"
    },
    "resnet50": {
      "status": "Available",
      "architecture": "ResNet",
      "task": "Image Classification",
      "note": "50-layer residual network — standard benchmark baseline",
      "accuracy": "76.13% Top-1",
      "rank": 13,
      "size": "97.8 MB",
      "url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
      "score": 0,
      "name": "ResNet-50"
    },
    "resnet101": {
      "status": "Available",
      "architecture": "ResNet",
      "task": "Image Classification",
      "note": "101-layer residual network",
      "accuracy": "77.45% Top-1",
      "rank": 12,
      "size": "170.5 MB",
      "url": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
      "score": 0,
      "name": "ResNet-101"
    },
    "resnet152": {
      "status": "Available",
      "architecture": "ResNet",
      "task": "Image Classification",
      "note": "152-layer residual network — deepest standard ResNet",
      "accuracy": "78.30% Top-1",
      "rank": 11,
      "size": "230.5 MB",
      "url": "https://download.pytorch.org/models/resnet152-f82ba261.pth",
      "score": 0,
      "name": "ResNet-152"
    },
    "mobilenet-v3-small": {
      "status": "Available",
      "architecture": "MobileNetV3",
      "task": "Image Classification",
      "note": "Ultra-lightweight — 9.8 MB, CPU/edge optimized",
      "accuracy": "67.57% Top-1",
      "rank": 16,
      "size": "9.8 MB",
      "url": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
      "score": 0,
      "name": "MobileNetV3 Small"
    },
    "mobilenet-v3-large": {
      "status": "Available",
      "architecture": "MobileNetV3",
      "task": "Image Classification",
      "note": "Better accuracy than MobileNetV2, still CPU-friendly",
      "accuracy": "74.02% Top-1",
      "rank": 15,
      "size": "21.1 MB",
      "url": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
      "score": 0,
      "name": "MobileNetV3 Large"
    },
```

- [ ] **Step 3: Validate**

```bash
jq . models.json > /dev/null && echo "models.json valid" || echo "models.json INVALID"
jq . model_registry.json > /dev/null && echo "model_registry.json valid" || echo "model_registry.json INVALID"
```

- [ ] **Step 4: Commit**

```bash
git add models.json model_registry.json
git commit -m "feat(registry): add ResNet-34/50/101/152 and MobileNetV3 Small/Large"
```

---

### Task 4: Add TIMM SOTA image models to `models.json`

**Files:**
- Modify: `models.json` — `available_models` object

These 12 models are already in `model_registry.json`. Add them to `models.json` after the `mobilenet_v3_large` entry.

- [ ] **Step 1: Add TIMM SOTA entries to `models.json`**

```json
    "eva02_large": {
      "name": "eva02_large",
      "display_name": "EVA-02 Large",
      "description": "SOTA ImageNet-1K accuracy at 90.054% Top-1; MIM pretraining",
      "source": "huggingface",
      "model_id": "timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 90,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 304000000,
        "size_mb": 1200,
        "architecture": "ViT (EVA-02)",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [448, 448],
        "num_classes": 1000,
        "accuracy_top1": 0.90054
      }
    },
    "eva_giant": {
      "name": "eva_giant",
      "display_name": "EVA Giant",
      "description": "Very large EVA ViT — 4 GB, 89.792% Top-1 on ImageNet-1K",
      "source": "huggingface",
      "model_id": "timm/eva_giant_patch14_560.m30m_ft_in22k_in1k",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 8192,
        "recommended_memory_mb": 16384,
        "gpu_required": true,
        "min_gpu_memory_mb": 8192
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 180,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 1012000000,
        "size_mb": 4000,
        "architecture": "ViT (EVA Giant)",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [560, 560],
        "num_classes": 1000,
        "accuracy_top1": 0.89792
      }
    },
    "convnextv2_huge": {
      "name": "convnextv2_huge",
      "display_name": "ConvNeXt V2 Huge",
      "description": "SOTA ConvNet — 88.848% Top-1 with FCMAE pretraining + 512px fine-tuning",
      "source": "huggingface",
      "model_id": "timm/convnextv2_huge.fcmae_ft_in22k_in1k_512",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 8192,
        "recommended_memory_mb": 16384,
        "gpu_required": true,
        "min_gpu_memory_mb": 6144
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 660000000,
        "size_mb": 2600,
        "architecture": "ConvNeXt V2",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [512, 512],
        "num_classes": 1000,
        "accuracy_top1": 0.88848
      }
    },
    "convnext_xxlarge_clip": {
      "name": "convnext_xxlarge_clip",
      "display_name": "ConvNeXt XXLarge CLIP",
      "description": "CLIP LAION-2B pretrained ConvNeXt XXLarge with model soup fine-tuning — 88.612% Top-1",
      "source": "huggingface",
      "model_id": "timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 8192,
        "recommended_memory_mb": 16384,
        "gpu_required": true,
        "min_gpu_memory_mb": 6144
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 846000000,
        "size_mb": 3500,
        "architecture": "ConvNeXt XXLarge",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [256, 256],
        "num_classes": 1000,
        "accuracy_top1": 0.88612
      }
    },
    "maxvit_xlarge": {
      "name": "maxvit_xlarge",
      "display_name": "MaxViT XLarge",
      "description": "MaxViT hybrid architecture — 88.53% Top-1 at 512px, ImageNet-21K pretrained",
      "source": "huggingface",
      "model_id": "timm/maxvit_xlarge_tf_512.in21k_ft_in1k",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 90,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 475000000,
        "size_mb": 1800,
        "architecture": "MaxViT",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [512, 512],
        "num_classes": 1000,
        "accuracy_top1": 0.8853
      }
    },
    "beit_large": {
      "name": "beit_large",
      "display_name": "BEiT Large",
      "description": "BERT-style pretraining for vision — 88.6% Top-1 at 512px",
      "source": "huggingface",
      "model_id": "timm/beit_large_patch16_512.in22k_ft_in22k_in1k",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 90,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 307000000,
        "size_mb": 1200,
        "architecture": "BEiT",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [512, 512],
        "num_classes": 1000,
        "accuracy_top1": 0.886
      }
    },
    "swin_large": {
      "name": "swin_large",
      "display_name": "Swin Transformer Large",
      "description": "Hierarchical Swin Transformer — 87.3% Top-1 at 384px, ImageNet-22K pretrained",
      "source": "huggingface",
      "model_id": "timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 197000000,
        "size_mb": 790,
        "architecture": "Swin Transformer",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [384, 384],
        "num_classes": 1000,
        "accuracy_top1": 0.873
      }
    },
    "deit3_huge": {
      "name": "deit3_huge",
      "display_name": "DeiT-III Huge",
      "description": "Data-efficient image transformer v3 — 87.7% Top-1 at 224px",
      "source": "huggingface",
      "model_id": "timm/deit3_huge_patch14_224.fb_in22k_ft_in1k",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 90,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 632000000,
        "size_mb": 2500,
        "architecture": "DeiT-III",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [224, 224],
        "num_classes": 1000,
        "accuracy_top1": 0.877
      }
    },
    "vit_giant_clip": {
      "name": "vit_giant_clip",
      "display_name": "ViT Giant (CLIP)",
      "description": "Google ViT Giant pretrained on CLIP LAION-2B — ~88.5% Top-1",
      "source": "huggingface",
      "model_id": "timm/vit_giant_patch14_224.clip_laion2b",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 8192,
        "recommended_memory_mb": 16384,
        "gpu_required": true,
        "min_gpu_memory_mb": 8192
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 2,
        "timeout_seconds": 120,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 1012000000,
        "size_mb": 5000,
        "architecture": "ViT Giant",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [224, 224],
        "num_classes": 1000,
        "accuracy_top1": 0.885
      }
    },
    "coatnet_3": {
      "name": "coatnet_3",
      "display_name": "CoAtNet-3",
      "description": "Efficient Conv + Attention hybrid — ~86.0% Top-1 at 224px",
      "source": "huggingface",
      "model_id": "timm/coatnet_3_rw_224.sw_in1k",
      "model_type": "transformer",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 168000000,
        "size_mb": 700,
        "architecture": "CoAtNet",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [224, 224],
        "num_classes": 1000,
        "accuracy_top1": 0.860
      }
    },
    "efficientnetv2_xl": {
      "name": "efficientnetv2_xl",
      "display_name": "EfficientNetV2 XL",
      "description": "EfficientNetV2 XL — 87.3% Top-1 with ImageNet-21K pretraining",
      "source": "huggingface",
      "model_id": "timm/tf_efficientnetv2_xl.in21k_ft_in1k",
      "model_type": "cnn",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 6144,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 208000000,
        "size_mb": 850,
        "architecture": "EfficientNetV2",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [512, 512],
        "num_classes": 1000,
        "accuracy_top1": 0.873
      }
    },
    "mobilenetv4_hybrid_large": {
      "name": "mobilenetv4_hybrid_large",
      "display_name": "MobileNetV4 Hybrid Large",
      "description": "Edge-optimized MobileNetV4 — 84.36% Top-1 at 448px, CPU/mobile friendly",
      "source": "huggingface",
      "model_id": "timm/mobilenetv4_hybrid_large.e600_r448_in1k",
      "model_type": "cnn",
      "task": "image-classification",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 2,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 1024,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 32,
        "timeout_seconds": 30,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 37000000,
        "size_mb": 140,
        "architecture": "MobileNetV4",
        "framework": "timm",
        "version": "1.0.0",
        "input_size": [448, 448],
        "num_classes": 1000,
        "accuracy_top1": 0.8436
      }
    },
```

- [ ] **Step 2: Validate**

```bash
jq . models.json > /dev/null && echo "models.json valid" || echo "models.json INVALID"
```

- [ ] **Step 3: Commit**

```bash
git add models.json
git commit -m "feat(registry): add 12 TIMM SOTA image classification models to models.json"
```

---

### Task 5: Add YOLO variants to both files

**Files:**
- Modify: `models.json` — `available_models` object
- Modify: `model_registry.json` — `models` object

- [ ] **Step 1: Add YOLOv8x, YOLOv5m/l/x, YOLOv9c/e, YOLOv10n/s/m/b/l/x to `models.json` after the `yolov5s` entry**

```json
    "yolov8x": {
      "name": "yolov8x",
      "display_name": "YOLOv8 XLarge",
      "description": "YOLOv8 XLarge — highest accuracy YOLOv8 variant",
      "source": "ultralytics",
      "model_id": "yolov8x.pt",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 120,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 68229648,
        "size_mb": 136.7,
        "architecture": "YOLOv8",
        "framework": "ultralytics",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.555,
        "map50_95": 0.418
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
    "yolov5m": {
      "name": "yolov5m",
      "display_name": "YOLOv5 Medium",
      "description": "YOLOv5 medium — balanced speed and accuracy",
      "source": "pytorch_hub",
      "model_id": "ultralytics/yolov5:yolov5m",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 2048,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 21172173,
        "size_mb": 42.2,
        "architecture": "YOLOv5",
        "framework": "pytorch",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.456,
        "map50_95": 0.283
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
    "yolov5l": {
      "name": "yolov5l",
      "display_name": "YOLOv5 Large",
      "description": "YOLOv5 large — high accuracy object detection",
      "source": "pytorch_hub",
      "model_id": "ultralytics/yolov5:yolov5l",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 1024,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 1024
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 90,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 46533693,
        "size_mb": 93.2,
        "architecture": "YOLOv5",
        "framework": "pytorch",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.491,
        "map50_95": 0.318
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
    "yolov5x": {
      "name": "yolov5x",
      "display_name": "YOLOv5 XLarge",
      "description": "YOLOv5 XLarge — maximum accuracy in YOLOv5 family",
      "source": "pytorch_hub",
      "model_id": "ultralytics/yolov5:yolov5x",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 120,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 86705005,
        "size_mb": 170.3,
        "architecture": "YOLOv5",
        "framework": "pytorch",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.506,
        "map50_95": 0.337
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
    "yolov9c": {
      "name": "yolov9c",
      "display_name": "YOLOv9 Compact",
      "description": "YOLOv9 Compact — GELAN + PGI architecture, 53.0% mAP50-95",
      "source": "ultralytics",
      "model_id": "yolov9c.pt",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 90,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 25483328,
        "size_mb": 51.5,
        "architecture": "YOLOv9 (GELAN)",
        "framework": "ultralytics",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.762,
        "map50_95": 0.530
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
    "yolov9e": {
      "name": "yolov9e",
      "display_name": "YOLOv9 Extended",
      "description": "YOLOv9 Extended — highest accuracy YOLOv9 variant, 55.5% mAP50-95",
      "source": "ultralytics",
      "model_id": "yolov9e.pt",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 120,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 57380160,
        "size_mb": 115.5,
        "architecture": "YOLOv9 (GELAN)",
        "framework": "ultralytics",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.787,
        "map50_95": 0.555
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
    "yolov10n": {
      "name": "yolov10n",
      "display_name": "YOLOv10 Nano",
      "description": "YOLOv10 Nano — NMS-free, 2.3MB, 38.5% mAP50-95",
      "source": "ultralytics",
      "model_id": "yolov10n.pt",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 1,
      "hardware_requirements": {
        "min_memory_mb": 128,
        "recommended_memory_mb": 256,
        "gpu_required": false,
        "min_gpu_memory_mb": 128
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 32,
        "timeout_seconds": 30,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 2300000,
        "size_mb": 5.6,
        "architecture": "YOLOv10",
        "framework": "ultralytics",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.536,
        "map50_95": 0.386
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": true
      }
    },
    "yolov10s": {
      "name": "yolov10s",
      "display_name": "YOLOv10 Small",
      "description": "YOLOv10 Small — NMS-free, 46.2% mAP50-95",
      "source": "ultralytics",
      "model_id": "yolov10s.pt",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 2,
      "hardware_requirements": {
        "min_memory_mb": 256,
        "recommended_memory_mb": 512,
        "gpu_required": false,
        "min_gpu_memory_mb": 256
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 16,
        "timeout_seconds": 30,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 7200000,
        "size_mb": 16.5,
        "architecture": "YOLOv10",
        "framework": "ultralytics",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.632,
        "map50_95": 0.462
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": true
      }
    },
    "yolov10m": {
      "name": "yolov10m",
      "display_name": "YOLOv10 Medium",
      "description": "YOLOv10 Medium — NMS-free, 51.1% mAP50-95",
      "source": "ultralytics",
      "model_id": "yolov10m.pt",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 1024,
        "recommended_memory_mb": 2048,
        "gpu_required": true,
        "min_gpu_memory_mb": 1024
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 15400000,
        "size_mb": 32.1,
        "architecture": "YOLOv10",
        "framework": "ultralytics",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.692,
        "map50_95": 0.511
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
    "yolov10b": {
      "name": "yolov10b",
      "display_name": "YOLOv10 Balanced",
      "description": "YOLOv10 Balanced — NMS-free, wider backbone, 52.7% mAP50-95",
      "source": "ultralytics",
      "model_id": "yolov10b.pt",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 4096,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 19100000,
        "size_mb": 37.4,
        "architecture": "YOLOv10",
        "framework": "ultralytics",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.705,
        "map50_95": 0.526
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
    "yolov10l": {
      "name": "yolov10l",
      "display_name": "YOLOv10 Large",
      "description": "YOLOv10 Large — NMS-free, 53.4% mAP50-95",
      "source": "ultralytics",
      "model_id": "yolov10l.pt",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 6144,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 90,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 24400000,
        "size_mb": 49.0,
        "architecture": "YOLOv10",
        "framework": "ultralytics",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.712,
        "map50_95": 0.534
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
    "yolov10x": {
      "name": "yolov10x",
      "display_name": "YOLOv10 XLarge",
      "description": "YOLOv10 XLarge — NMS-free, highest accuracy in YOLOv10 family, 54.4% mAP50-95",
      "source": "ultralytics",
      "model_id": "yolov10x.pt",
      "model_type": "yolo",
      "task": "object-detection",
      "category": "computer-vision",
      "enabled": true,
      "auto_load": false,
      "priority": 5,
      "hardware_requirements": {
        "min_memory_mb": 4096,
        "recommended_memory_mb": 8192,
        "gpu_required": true,
        "min_gpu_memory_mb": 4096
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 4,
        "timeout_seconds": 120,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 29500000,
        "size_mb": 58.8,
        "architecture": "YOLOv10",
        "framework": "ultralytics",
        "version": "1.0.0",
        "input_size": [640, 640],
        "num_classes": 80,
        "pretrained": true,
        "map50": 0.722,
        "map50_95": 0.544
      },
      "detection_features": {
        "supports_detection": true,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "supports_batch_inference": true,
        "real_time_capable": false
      }
    },
```

- [ ] **Step 2: Add YOLO variants to `model_registry.json`**

```json
    "yolov8x": {
      "status": "Available",
      "architecture": "YOLOv8",
      "task": "Object Detection",
      "note": "XLarge — highest accuracy YOLOv8, 41.8% mAP",
      "accuracy": "55.5% mAP50 / 41.8% mAP50-95",
      "rank": 5,
      "size": "136.7 MB",
      "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
      "score": 0,
      "name": "YOLOv8 XLarge"
    },
    "yolov5m": {
      "status": "Available",
      "architecture": "YOLOv5",
      "task": "Object Detection",
      "note": "Medium — balanced speed/accuracy",
      "accuracy": "45.6% mAP50 / 28.3% mAP50-95",
      "rank": 8,
      "size": "42.2 MB",
      "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
      "score": 0,
      "name": "YOLOv5 Medium"
    },
    "yolov5l": {
      "status": "Available",
      "architecture": "YOLOv5",
      "task": "Object Detection",
      "note": "Large — high accuracy",
      "accuracy": "49.1% mAP50 / 31.8% mAP50-95",
      "rank": 7,
      "size": "93.2 MB",
      "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
      "score": 0,
      "name": "YOLOv5 Large"
    },
    "yolov5x": {
      "status": "Available",
      "architecture": "YOLOv5",
      "task": "Object Detection",
      "note": "XLarge — maximum accuracy in YOLOv5 family",
      "accuracy": "50.6% mAP50 / 33.7% mAP50-95",
      "rank": 6,
      "size": "170.3 MB",
      "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt",
      "score": 0,
      "name": "YOLOv5 XLarge"
    },
    "yolov9c": {
      "status": "Available",
      "architecture": "YOLOv9 (GELAN + PGI)",
      "task": "Object Detection",
      "note": "Compact — 53.0% mAP50-95",
      "accuracy": "76.2% mAP50 / 53.0% mAP50-95",
      "rank": 3,
      "size": "51.5 MB",
      "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt",
      "score": 0,
      "name": "YOLOv9 Compact"
    },
    "yolov9e": {
      "status": "Available",
      "architecture": "YOLOv9 (GELAN + PGI)",
      "task": "Object Detection",
      "note": "Extended — 55.5% mAP50-95, SOTA for YOLO family",
      "accuracy": "78.7% mAP50 / 55.5% mAP50-95",
      "rank": 2,
      "size": "115.5 MB",
      "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt",
      "score": 0,
      "name": "YOLOv9 Extended"
    },
    "yolov10n": {
      "status": "Available",
      "architecture": "YOLOv10 (NMS-free)",
      "task": "Object Detection",
      "note": "Nano — NMS-free, 38.5% mAP, real-time capable",
      "accuracy": "53.6% mAP50 / 38.5% mAP50-95",
      "rank": 9,
      "size": "5.6 MB",
      "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
      "score": 0,
      "name": "YOLOv10 Nano"
    },
    "yolov10s": {
      "status": "Available",
      "architecture": "YOLOv10 (NMS-free)",
      "task": "Object Detection",
      "note": "Small — NMS-free, 46.2% mAP",
      "accuracy": "63.2% mAP50 / 46.2% mAP50-95",
      "rank": 8,
      "size": "16.5 MB",
      "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
      "score": 0,
      "name": "YOLOv10 Small"
    },
    "yolov10m": {
      "status": "Available",
      "architecture": "YOLOv10 (NMS-free)",
      "task": "Object Detection",
      "note": "Medium — NMS-free, 51.1% mAP",
      "accuracy": "69.2% mAP50 / 51.1% mAP50-95",
      "rank": 6,
      "size": "32.1 MB",
      "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
      "score": 0,
      "name": "YOLOv10 Medium"
    },
    "yolov10b": {
      "status": "Available",
      "architecture": "YOLOv10 (NMS-free)",
      "task": "Object Detection",
      "note": "Balanced — wider, 52.7% mAP",
      "accuracy": "70.5% mAP50 / 52.7% mAP50-95",
      "rank": 5,
      "size": "37.4 MB",
      "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt",
      "score": 0,
      "name": "YOLOv10 Balanced"
    },
    "yolov10l": {
      "status": "Available",
      "architecture": "YOLOv10 (NMS-free)",
      "task": "Object Detection",
      "note": "Large — NMS-free, 53.4% mAP",
      "accuracy": "71.2% mAP50 / 53.4% mAP50-95",
      "rank": 4,
      "size": "49.0 MB",
      "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt",
      "score": 0,
      "name": "YOLOv10 Large"
    },
    "yolov10x": {
      "status": "Available",
      "architecture": "YOLOv10 (NMS-free)",
      "task": "Object Detection",
      "note": "XLarge — highest accuracy in YOLOv10 family, 54.4% mAP",
      "accuracy": "72.2% mAP50 / 54.4% mAP50-95",
      "rank": 3,
      "size": "58.8 MB",
      "url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt",
      "score": 0,
      "name": "YOLOv10 XLarge"
    },
```

- [ ] **Step 3: Validate**

```bash
jq . models.json > /dev/null && echo "models.json valid" || echo "models.json INVALID"
jq . model_registry.json > /dev/null && echo "model_registry.json valid" || echo "model_registry.json INVALID"
```

- [ ] **Step 4: Commit**

```bash
git add models.json model_registry.json
git commit -m "feat(registry): add YOLOv8x, YOLOv5m/l/x, YOLOv9c/e, YOLOv10 full family"
```

---

### Task 6: Add NLP models to both files

**Files:**
- Modify: `models.json` — `available_models` object
- Modify: `model_registry.json` — `models` object

- [ ] **Step 1: Add NLP models to `models.json` after the `sentence_transformer` entry**

```json
    "bert_base_uncased": {
      "name": "bert_base_uncased",
      "display_name": "BERT Base Uncased",
      "description": "Google BERT base — general-purpose transformer for text classification and NLU",
      "source": "huggingface",
      "model_id": "bert-base-uncased",
      "model_type": "transformer",
      "task": "text-classification",
      "category": "nlp",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 2048,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 16,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 109482240,
        "size_mb": 417.7,
        "architecture": "BertForSequenceClassification",
        "framework": "transformers",
        "version": "1.0.0",
        "languages": ["en"],
        "max_sequence_length": 512
      }
    },
    "roberta_base": {
      "name": "roberta_base",
      "display_name": "RoBERTa Base",
      "description": "Robustly optimized BERT pretraining — stronger text classification than BERT base",
      "source": "huggingface",
      "model_id": "roberta-base",
      "model_type": "transformer",
      "task": "text-classification",
      "category": "nlp",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 2048,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 16,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 124645632,
        "size_mb": 476.5,
        "architecture": "RobertaForSequenceClassification",
        "framework": "transformers",
        "version": "1.0.0",
        "languages": ["en"],
        "max_sequence_length": 512
      }
    },
    "all_mpnet_base_v2": {
      "name": "all_mpnet_base_v2",
      "display_name": "all-mpnet-base-v2",
      "description": "Best-quality sentence embeddings from sentence-transformers; 768-dim",
      "source": "huggingface",
      "model_id": "sentence-transformers/all-mpnet-base-v2",
      "model_type": "transformer",
      "task": "feature-extraction",
      "category": "nlp",
      "enabled": true,
      "auto_load": false,
      "priority": 3,
      "hardware_requirements": {
        "min_memory_mb": 512,
        "recommended_memory_mb": 2048,
        "gpu_required": false,
        "min_gpu_memory_mb": 512
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 32,
        "timeout_seconds": 60,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 109482240,
        "size_mb": 438,
        "architecture": "MPNet",
        "framework": "sentence-transformers",
        "version": "1.0.0",
        "max_sequence_length": 384,
        "embedding_dimension": 768
      }
    },
    "paraphrase_multilingual_mpnet": {
      "name": "paraphrase_multilingual_mpnet",
      "display_name": "paraphrase-multilingual-mpnet-base-v2",
      "description": "Multilingual sentence embeddings — 50+ languages, 768-dim",
      "source": "huggingface",
      "model_id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
      "model_type": "transformer",
      "task": "feature-extraction",
      "category": "nlp",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 1024,
        "recommended_memory_mb": 4096,
        "gpu_required": false,
        "min_gpu_memory_mb": 1024
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 16,
        "timeout_seconds": 90,
        "warmup_iterations": 5
      },
      "metadata": {
        "parameters": 278000000,
        "size_mb": 1110,
        "architecture": "XLM-RoBERTa + MPNet",
        "framework": "sentence-transformers",
        "version": "2.0",
        "max_sequence_length": 128,
        "embedding_dimension": 768,
        "languages": ["multilingual"]
      }
    },
    "bart_large_mnli": {
      "name": "bart_large_mnli",
      "display_name": "BART Large MNLI",
      "description": "facebook/bart-large-mnli — zero-shot text classification via NLI",
      "source": "huggingface",
      "model_id": "facebook/bart-large-mnli",
      "model_type": "transformer",
      "task": "zero-shot-classification",
      "category": "nlp",
      "enabled": true,
      "auto_load": false,
      "priority": 4,
      "hardware_requirements": {
        "min_memory_mb": 2048,
        "recommended_memory_mb": 6144,
        "gpu_required": true,
        "min_gpu_memory_mb": 2048
      },
      "inference_config": {
        "batch_size": 1,
        "max_batch_size": 8,
        "timeout_seconds": 90,
        "warmup_iterations": 3
      },
      "metadata": {
        "parameters": 406291456,
        "size_mb": 1630,
        "architecture": "BartForSequenceClassification",
        "framework": "transformers",
        "version": "1.0.0",
        "languages": ["en"],
        "max_sequence_length": 1024
      }
    },
```

- [ ] **Step 2: Add NLP models to `model_registry.json`**

```json
    "bert-base-uncased": {
      "status": "Available",
      "architecture": "BERT",
      "task": "Text Classification",
      "note": "General-purpose BERT base — fine-tune for downstream tasks",
      "voices": "N/A",
      "quality": "High",
      "rank": 3,
      "size": "417.7 MB",
      "url": "https://huggingface.co/bert-base-uncased",
      "score": 0,
      "name": "BERT Base Uncased"
    },
    "roberta-base": {
      "status": "Available",
      "architecture": "RoBERTa",
      "task": "Text Classification",
      "note": "Stronger pretraining than BERT — better downstream performance",
      "voices": "N/A",
      "quality": "High",
      "rank": 2,
      "size": "476.5 MB",
      "url": "https://huggingface.co/roberta-base",
      "score": 0,
      "name": "RoBERTa Base"
    },
    "all-mpnet-base-v2": {
      "status": "Available",
      "architecture": "MPNet",
      "task": "Feature Extraction",
      "note": "Best quality sentence embeddings — 768-dim",
      "voices": "N/A",
      "quality": "High",
      "rank": 1,
      "size": "438 MB",
      "url": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
      "score": 0,
      "name": "all-mpnet-base-v2"
    },
    "paraphrase-multilingual-mpnet": {
      "status": "Available",
      "architecture": "XLM-RoBERTa + MPNet",
      "task": "Feature Extraction",
      "note": "50+ language sentence embeddings — 768-dim",
      "voices": "N/A",
      "quality": "High",
      "rank": 2,
      "size": "1.11 GB",
      "url": "https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
      "score": 0,
      "name": "paraphrase-multilingual-mpnet-base-v2"
    },
    "bart-large-mnli": {
      "status": "Available",
      "architecture": "BART",
      "task": "Zero-Shot Classification",
      "note": "Zero-shot text classification via NLI — no fine-tuning required",
      "voices": "N/A",
      "quality": "High",
      "rank": 1,
      "size": "1.63 GB",
      "url": "https://huggingface.co/facebook/bart-large-mnli",
      "score": 0,
      "name": "BART Large MNLI"
    },
```

- [ ] **Step 3: Validate**

```bash
jq . models.json > /dev/null && echo "models.json valid" || echo "models.json INVALID"
jq . model_registry.json > /dev/null && echo "model_registry.json valid" || echo "model_registry.json INVALID"
```

- [ ] **Step 4: Commit**

```bash
git add models.json model_registry.json
git commit -m "feat(registry): add BERT, RoBERTa, MPNet, multilingual, BART zero-shot NLP models"
```

---

### Task 7: Update `model_groups` in `models.json`

**Files:**
- Modify: `models.json` — `model_groups` object

Replace each group's `"models"` array with the full expanded list. The existing `model_groups` block (lines starting at `"model_groups"`) gets these updated values.

- [ ] **Step 1: Replace the `model_groups` block**

Find the `"model_groups"` key in `models.json` and replace its entire value with:

```json
  "model_groups": {
    "text_to_speech": {
      "name": "Text-to-Speech Models",
      "description": "Models for converting text to speech",
      "models": [
        "speecht5_tts", "bark_tts", "bark_small",
        "kokoro_v019", "kokoro_v10", "kokoro_onnx", "kokoro_onnx_int8",
        "xtts_v2", "piper_lessac", "styletts2", "f5_tts",
        "parler_tts_mini", "chatterbox", "outetts_0_3_500m",
        "sesame_csm_1b", "cosyvoice2_0_5b", "index_tts_2",
        "melotts_english_v3", "matcha_tts", "vits_vctk",
        "amphion_maskgct", "metavoice", "openvoice", "fish_speech_v15"
      ],
      "default_model": "kokoro_onnx",
      "enabled": true
    },
    "speech_to_text": {
      "name": "Speech-to-Text Models",
      "description": "Models for converting speech to text",
      "models": [
        "whisper_base", "whisper_small", "whisper_medium",
        "whisper_large", "whisper_large_v3", "whisper_turbo"
      ],
      "default_model": "whisper_base",
      "enabled": true
    },
    "image_classification": {
      "name": "Image Classification Models",
      "description": "Models for classifying images",
      "models": [
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "vit_base",
        "eva02_large", "eva_giant", "convnextv2_huge", "convnext_xxlarge_clip",
        "maxvit_xlarge", "beit_large", "swin_large", "deit3_huge",
        "vit_giant_clip", "coatnet_3", "efficientnetv2_xl", "mobilenetv4_hybrid_large"
      ],
      "default_model": "resnet18",
      "enabled": true
    },
    "object_detection": {
      "name": "Object Detection Models",
      "description": "Models for detecting and localizing objects in images",
      "models": [
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
        "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
        "yolov9c", "yolov9e",
        "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"
      ],
      "default_model": "yolov8n",
      "enabled": true
    },
    "text_classification": {
      "name": "Text Classification Models",
      "description": "Models for classifying text",
      "models": ["distilbert_sentiment", "bert_base_uncased", "roberta_base"],
      "default_model": "distilbert_sentiment",
      "enabled": true
    },
    "feature_extraction": {
      "name": "Feature Extraction Models",
      "description": "Models for extracting features from text or images",
      "models": [
        "sentence_transformer", "all_mpnet_base_v2", "paraphrase_multilingual_mpnet"
      ],
      "default_model": "sentence_transformer",
      "enabled": true
    },
    "zero_shot_classification": {
      "name": "Zero-Shot Classification Models",
      "description": "Models for classifying text without fine-tuning",
      "models": ["bart_large_mnli"],
      "default_model": "bart_large_mnli",
      "enabled": true
    }
  },
```

- [ ] **Step 2: Validate**

```bash
jq . models.json > /dev/null && echo "models.json valid" || echo "models.json INVALID"
```

- [ ] **Step 3: Commit**

```bash
git add models.json
git commit -m "feat(registry): update model_groups to include all new model entries"
```

---

### Task 8: Update `hardware_profiles` in `models.json`

**Files:**
- Modify: `models.json` — `hardware_profiles` object

- [ ] **Step 1: Replace the `hardware_profiles` block**

Find the `"hardware_profiles"` key in `models.json` and replace its entire value with:

```json
  "hardware_profiles": {
    "cpu_only": {
      "name": "CPU Only",
      "description": "Profile for CPU-only inference",
      "allowed_models": [
        "example",
        "distilbert_sentiment", "bert_base_uncased", "roberta_base",
        "sentence_transformer", "all_mpnet_base_v2",
        "resnet18", "resnet34", "resnet50", "mobilenet_v2",
        "mobilenet_v3_small", "mobilenet_v3_large", "mobilenetv4_hybrid_large",
        "whisper_base", "whisper_small", "whisper_medium", "whisper_turbo",
        "yolov8n", "yolov8s", "yolov5n", "yolov5s",
        "yolov10n", "yolov10s",
        "kokoro_onnx", "kokoro_onnx_int8", "piper_lessac",
        "melotts_english_v3", "vits_vctk", "matcha_tts",
        "kokoro_v019", "kokoro_v10"
      ],
      "blocked_models": [
        "bark_tts", "bark_small", "yolov8m", "yolov8l", "yolov8x",
        "yolov5m", "yolov5l", "yolov5x", "yolov9c", "yolov9e",
        "yolov10m", "yolov10b", "yolov10l", "yolov10x",
        "eva02_large", "eva_giant", "convnextv2_huge", "convnext_xxlarge_clip",
        "maxvit_xlarge", "beit_large", "swin_large", "deit3_huge",
        "vit_giant_clip", "coatnet_3", "efficientnetv2_xl",
        "xtts_v2", "styletts2", "f5_tts", "parler_tts_mini",
        "chatterbox", "outetts_0_3_500m", "sesame_csm_1b",
        "cosyvoice2_0_5b", "index_tts_2", "amphion_maskgct",
        "metavoice", "openvoice", "fish_speech_v15",
        "whisper_large", "whisper_large_v3",
        "resnet101", "resnet152",
        "paraphrase_multilingual_mpnet", "bart_large_mnli"
      ],
      "max_memory_usage_mb": 8192,
      "optimization_level": "medium"
    },
    "gpu_basic": {
      "name": "Basic GPU",
      "description": "Profile for systems with basic GPU (4-8GB VRAM)",
      "allowed_models": [
        "example",
        "distilbert_sentiment", "bert_base_uncased", "roberta_base",
        "sentence_transformer", "all_mpnet_base_v2",
        "resnet18", "resnet34", "resnet50", "resnet101",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "mobilenetv4_hybrid_large", "coatnet_3", "swin_large",
        "whisper_base", "whisper_small", "whisper_medium",
        "whisper_large", "whisper_turbo",
        "speecht5_tts", "kokoro_v019", "kokoro_v10",
        "kokoro_onnx", "kokoro_onnx_int8", "piper_lessac",
        "styletts2", "f5_tts", "bark_small", "metavoice",
        "melotts_english_v3", "vits_vctk", "matcha_tts",
        "xtts_v2", "openvoice", "fish_speech_v15",
        "cosyvoice2_0_5b", "index_tts_2", "chatterbox",
        "yolov8n", "yolov8s", "yolov8m", "yolov8l",
        "yolov5n", "yolov5s", "yolov5m", "yolov5l",
        "yolov9c", "yolov10n", "yolov10s", "yolov10m", "yolov10b",
        "paraphrase_multilingual_mpnet", "bart_large_mnli"
      ],
      "blocked_models": [
        "bark_tts", "yolov8x", "yolov5x", "yolov9e",
        "yolov10l", "yolov10x",
        "eva02_large", "eva_giant", "convnextv2_huge", "convnext_xxlarge_clip",
        "maxvit_xlarge", "beit_large", "deit3_huge", "vit_giant_clip",
        "efficientnetv2_xl", "resnet152",
        "parler_tts_mini", "outetts_0_3_500m", "sesame_csm_1b",
        "amphion_maskgct", "whisper_large_v3"
      ],
      "max_memory_usage_mb": 16384,
      "max_gpu_memory_mb": 6144,
      "optimization_level": "high"
    },
    "gpu_advanced": {
      "name": "Advanced GPU",
      "description": "Profile for systems with high-end GPU (8GB+ VRAM)",
      "allowed_models": [],
      "blocked_models": [],
      "max_memory_usage_mb": 65536,
      "max_gpu_memory_mb": 24576,
      "optimization_level": "maximum"
    }
  },
```

- [ ] **Step 2: Validate**

```bash
jq . models.json > /dev/null && echo "models.json valid" || echo "models.json INVALID"
```

- [ ] **Step 3: Commit**

```bash
git add models.json
git commit -m "feat(registry): update hardware_profiles to include all new models"
```

---

### Task 9: Final validation

**Files:** Both registry files (read-only validation)

- [ ] **Step 1: Count entries to verify completeness**

```bash
echo "=== models.json available_models count ==="
jq '.available_models | length' models.json

echo "=== model_registry.json models count ==="
jq '.models | length' model_registry.json

echo "=== TTS models in models.json ==="
jq '[.available_models[] | select(.task == "text-to-speech")] | length' models.json

echo "=== STT models in models.json ==="
jq '[.available_models[] | select(.task == "speech-to-text")] | length' models.json

echo "=== Image classification models in models.json ==="
jq '[.available_models[] | select(.task == "image-classification")] | length' models.json

echo "=== Object detection models in models.json ==="
jq '[.available_models[] | select(.task == "object-detection")] | length' models.json

echo "=== NLP models in models.json ==="
jq '[.available_models[] | select(.category == "nlp")] | length' models.json
```

Expected counts:
- `available_models`: 76+
- `model_registry.json models`: 60+
- TTS: 24+
- STT: 6
- image-classification: 21+
- object-detection: 18+
- NLP: 7+

- [ ] **Step 2: Verify all group model IDs exist in available_models**

```bash
jq -r '
  . as $root |
  .model_groups | to_entries[] | .value.models[] as $m |
  if ($root.available_models[$m] != null) then empty else "MISSING: \($m)" end
' models.json
```

Expected: no output (all referenced IDs exist)

- [ ] **Step 3: Cargo check — ensure Rust code still compiles with new registry**

```bash
cargo check 2>&1 | tail -5
```

Expected: `Finished` or only warnings (no errors)

- [ ] **Step 4: Final commit**

```bash
git add models.json model_registry.json
git commit -m "feat(registry): complete model registry expansion — 60+ models across TTS/STT/CV/NLP

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
