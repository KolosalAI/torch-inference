# Kolosal Inference — Engineering Guidelines

## Project
High-performance multimodal inference server written in Rust (actix-web). Serves TTS synthesis, audio/STT transcription, image classification, YOLO object detection, and LLM chat completion over HTTP.

## Scope — What to engineer
| Feature | Status | Notes |
|---------|--------|-------|
| TTS (`/tts/*`) | ✅ Active | Kokoro-ONNX, Bark, XTTS, StyleTTS2, Piper, VITS |
| STT / Audio (`/stt/*`, `/audio/*`) | ✅ Active | Whisper-based via `whisper_stt.rs` |
| Classification (`/classify/*`) | ✅ Active | ORT/ONNX image classifier |
| Detection (`/detect/*`, `/yolo/*`) | ✅ Active | YOLO via ORT/ONNX |
| Dashboard / Health / Metrics | ✅ Active | `/health`, `/system/info`, `/metrics`, `/performance` |
| **LLM / Assistant** | ✅ Active | 1-bit LLM chat via `/v1/chat/completions` (OpenAI-compatible, streaming SSE) |

## Build & Run

```bash
# Build (release, ~2 min)
cargo build --release

# Run server (port 8000)
./target/release/torch-inference-server

# Run all tests
cargo test

# Run benchmarks
cargo bench --bench throughput_bench
cargo bench --bench tts_bench
cargo bench --bench audio_bench
cargo bench --bench classification_bench
cargo bench --bench detection_bench
```

## Architecture
```
src/
  main.rs           — server startup, state init (TTS 6 engines, ORT, GPU)
  config.rs         — Config struct; reads config.yaml or env vars
  lib.rs            — crate root, re-exports all modules
  api/
    handlers.rs     — playground.html embedded via include_str!
    tts.rs          — POST /tts/stream (sentence-level streaming WAV)
    audio.rs        — POST /stt/transcribe
    classify.rs     — POST /classify/batch
    classification.rs
    yolo.rs         — POST /detect
    health.rs       — GET /health
    system.rs       — GET /system/info
    ...
  core/
    tts_manager.rs  — TTSManager: loads/routes to 6 TTS engines
    tts_engine.rs   — TTSEngine trait
    kokoro_onnx.rs  — Primary TTS engine (Kokoro-82M, ONNX INT8)
    audio.rs        — AudioProcessor: decode, resample, WAV I/O
    yolo.rs         — YoloDetector, BoundingBox, NMS
    image_pipeline.rs — ImagePipeline: preprocess for classifier/YOLO
    model_cache.rs  — FNV-1a 64-bit keyed model output cache
    ...
  benches/
    tts_bench.rs
    audio_bench.rs
    classification_bench.rs
    detection_bench.rs
    throughput_bench.rs
    cache_bench.rs
    memory_bench.rs
```

## Key Conventions
- **Playground**: `src/api/playground.html` is embedded via `include_str!`. All UI changes require `cargo build --release` to take effect.
- **Cache keys**: FNV-1a 64-bit hash with NUL byte separators. Never use `DefaultHasher` (not stable across runs).
- **Async**: actix-web uses `current_thread` executor. Never use `tokio::task::block_in_place` inside handlers — use `reqwest::blocking::Client` or `spawn_blocking` instead.
- **ORT**: Always enabled (`ort = "=2.0.0-rc.10"`). Loads `/opt/homebrew/lib/libonnxruntime.dylib` on macOS at runtime.
- **Candle feature**: Optional (`--features candle`). Not used in production — do not enable.
- **Config**: `config.yaml` or environment variables. Only fields mapped in `Config` struct affect runtime.
- **Error handling**: Use `anyhow::Result` internally; map to `actix_web::Error` at handler boundary.
- **Logging**: `tracing` crate with JSON output. Use `tracing::info!`, `tracing::error!`, etc.

## TTS Models
Located in `models/kokoro-82m/`:
- `kokoro-v1.0.int8.onnx` — primary model
- `voices/*.bin` — voice embeddings (af_heart, af_bella, etc.)

Server loads 6 TTS engines at startup via `TTSManager::initialize_defaults()`.

## Configuration
`config.yaml` key sections (runtime-relevant):
- `server.host`, `server.port` (default `0.0.0.0:8000`)
- `tts.*` — engine settings
- `security.*` — API key, rate limits
