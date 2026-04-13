# LLM Microservice Design

**Date:** 2026-04-13
**Status:** Approved

## Overview

A standalone Rust microservice that runs 1-bit (IQ1_S/Q2_K) multimodal LLM inference alongside the existing Kolosal Torch Inference server. Lives in `services/llm/` — a separate Cargo crate, not part of the main workspace. Exposes an OpenAI-compatible API on port 8001. The main server proxies `/llm/*` to it and returns 503 when it is not running.

## Repository Structure

```
torch-inference/
├── src/                             ← existing server (untouched except proxy route)
├── services/
│   └── llm/
│       ├── Cargo.toml               ← standalone crate
│       ├── config.toml              ← runtime config
│       ├── src/
│       │   ├── main.rs              ← actix-web server, port 8001
│       │   ├── config.rs            ← LlmConfig struct
│       │   ├── handler.rs           ← POST /v1/chat/completions, GET /v1/models
│       │   └── engine.rs            ← LlamaEngine wrapping llama-cpp-2
│       └── models/                  ← GGUF model files (gitignored)
├── scripts/
│   └── download_llm_model.sh        ← fetches MiniCPM-V 2.6 Q2_K + mmproj
└── Makefile                         ← llm-build, llm-run targets added
```

## Model

**Primary:** MiniCPM-V 2.6 Q2_K
- Multimodal: image + text input, text output
- ~2 GB GGUF + ~400 MB mmproj clip projector
- Supports single and multi-turn conversation with images
- Source: HuggingFace `openbmb/MiniCPM-V-2_6-gguf`

**Fallback:** LLaVA-1.6-Mistral-7B IQ1_S (~1.8 GB) — true 1-bit quantisation, text + image.

## API

### `POST /v1/chat/completions`

OpenAI-compatible. Supports `stream: true` (SSE) and `stream: false` (single JSON).

**Text-only:**
```json
{
  "model": "minicpm-v",
  "messages": [{"role": "user", "content": "Explain YOLO in one sentence."}],
  "stream": true,
  "max_tokens": 512
}
```

**Multimodal (image + text):**
```json
{
  "model": "minicpm-v",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>"}},
      {"type": "text", "text": "What objects are in this image?"}
    ]
  }],
  "stream": false
}
```

**Streaming response (SSE):**
```
data: {"id":"...","choices":[{"delta":{"content":"token"},"index":0}]}

data: [DONE]
```

### `GET /v1/models`

Returns the currently loaded model name and metadata.

### Proxy (main server)

`POST /llm/v1/chat/completions` on port 8000 → strips `/llm` prefix → forwards to `127.0.0.1:8001`. Returns `503 {"error":"LLM service unavailable"}` when the microservice is not running.

## Components

### `LlamaEngine` (engine.rs)

- Held as `Arc<LlamaEngine>` in actix `Data`
- Loads GGUF model via `llama_cpp_2::LlamaModel::load_from_file()`
- Loads clip projector via `llama_cpp_2::ClipContext` when `mmproj_path` is set
- Inference runs in `tokio::task::spawn_blocking` — mirrors pattern in `ort_yolo.rs`
- Token stream bridged to async via `tokio::sync::mpsc` channel → SSE response

### `LlmConfig` (config.rs)

Read from `services/llm/config.toml`:
```toml
port          = 8001
model_path    = "models/minicpm-v-2_6-q2_k.gguf"
mmproj_path   = "models/minicpm-v-2_6-mmproj.gguf"
ctx_size      = 4096
n_threads     = 4
n_gpu_layers  = 0
```

`mmproj_path` is optional — omit for text-only operation.

## Data Flow

```
Client
  → POST /v1/chat/completions
  → handler.rs: parse messages, extract base64 image bytes if present
  → engine.rs: LlamaEngine::infer(prompt, Option<image_bytes>, params)
      → spawn_blocking:
          → decode image → clip embed (if mmproj loaded)
          → llama token generation loop
          → send tokens via mpsc::Sender
  → SSE: stream tokens as OpenAI delta chunks  (or collect → single JSON)
  → data: [DONE]
```

## Error Handling

| Scenario | Response |
|----------|----------|
| Model file missing at startup | Log error, exit 1 with clear message |
| `mmproj_path` missing but image sent | `400 {"error":"multimodal not configured"}` |
| Image base64 decode failure | `400 {"error":"invalid image"}` |
| Inference error | `500 {"error":"inference failed: <msg>"}` |
| LLM service down (proxy) | `503 {"error":"LLM service unavailable"}` |

## Build & Run

```bash
# Build the LLM microservice
make llm-build
# equivalent: cd services/llm && cargo build --release

# Download models
bash scripts/download_llm_model.sh

# Run (standalone)
make llm-run
# equivalent: cd services/llm && ./target/release/llm-service

# Run both servers
make run        # main server on 8000
make llm-run    # LLM service on 8001
```

## Dependencies (`services/llm/Cargo.toml`)

```toml
llama-cpp-2   = "0.1"     # llama.cpp Rust bindings
actix-web     = "4.8"
actix-rt      = "2.10"
tokio         = { version = "1", features = ["rt-multi-thread", "macros", "sync"] }
serde         = { version = "1", features = ["derive"] }
serde_json    = "1"
anyhow        = "1"
tracing       = "0.1"
tracing-subscriber = "0.3"
base64        = "0.21"
```

`llama-cpp-2` builds llama.cpp via `cmake` at compile time. On macOS, Metal GPU acceleration is enabled automatically.

## Makefile Additions

```makefile
llm-build:
	cd services/llm && cargo build --release

llm-run:
	cd services/llm && ./target/release/llm-service
```
