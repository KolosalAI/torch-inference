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
| **LLM / Assistant** | ✅ Active | HRM-Text-1B via ONNX/`ort`; OpenAI-compatible `/v1/chat/completions` with streaming SSE. Image inputs are bridged through `/classify/batch` + `/yolo/detect` (caption-then-text). |

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
  config.rs         — Config struct; reads config.toml
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
- **Config**: `config.toml` (read by `Config::load()`). Env vars are not auto-loaded — only fields mapped in `Config` struct affect runtime.
- **Error handling**: Use `anyhow::Result` internally; map to `actix_web::Error` at handler boundary.
- **Logging**: `tracing` crate with JSON output. Use `tracing::info!`, `tracing::error!`, etc.

## TTS Models
Located in `models/kokoro-82m/`:
- `kokoro-v1.0.int8.onnx` — primary model
- `voices/*.bin` — voice embeddings (af_heart, af_bella, etc.)

Server loads 6 TTS engines at startup via `TTSManager::initialize_defaults()`.

## Open follow-ups (deferred from the audit landed across commits cf2b566 → dcd7cf2)

These are intentionally not in any of the audit batches because each needs
either (a) representative benchmarks to validate or (b) a design RFC. Pick
them up only with the relevant supporting work in hand.

- **Whisper KV-cache decoder** (`src/core/whisper_onnx.rs`): the greedy
  decode loop currently re-runs the full decoder over the full prefix
  per token (O(N²) compute) and clones the encoder output (~3 MB) on
  every iteration. A KV-cache rewrite is the single biggest STT
  latency win, but it needs a bench harness.
- **Sharded TTS synthesis cache** (`src/core/tts_manager.rs`): single
  global `Mutex<LruCache>` clones full `AudioData` inside the lock.
  Sharded LRU (16-way like `dedup.rs`) plus `Arc<AudioData>` would
  remove the contention point but the AudioData → Arc<AudioData>
  change is a public API churn.
- **Drop ModelCache serde round-trip** (`src/core/model_cache.rs`):
  every cache hit deserializes JSON. Replace with `Arc<dyn Any>`.
- **YOLO preprocess on the SIMD pipeline** (`src/core/ort_yolo.rs`):
  goes through `image::imageops::Lanczos3` + scalar normalize. The
  classifier path already uses `fast_image_resize` + `wide::f32x8`;
  routing YOLO through the same pipeline would 3-5× preprocess.
- **Error envelope unification across handlers** — currently four
  shapes (ApiError, registry::ApiResponse, predict ad-hoc, OpenAI).
  Pick one and migrate; needs a small RFC to settle which.
- **Audio resampler cache** (`src/core/audio.rs`): `FftFixedInOut`
  is rebuilt per call. Cache-by-key, but the resampler is `&mut self`
  per-process so a pool is needed. Worth 5-15 ms per STT request.

## Configuration
`config.toml` key sections (runtime-relevant):
- `server.host`, `server.port` — default `127.0.0.1:8000`. Use `0.0.0.0` only with auth + TLS in front.
- `models.*` — cache dir, registry path
- `auth.*` — JWT secret (must be set, no placeholder), bcrypt cost
- `microservices.*` — LLM/STT proxy upstream URLs
