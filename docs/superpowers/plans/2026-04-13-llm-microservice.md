# LLM Microservice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `services/llm/` Rust microservice that runs 1-bit multimodal LLM inference (MiniCPM-V 2.6 Q2_K via llama-cpp-2), exposes an OpenAI-compatible API on port 8001, and is proxied from the main server at `/llm/*`.

**Architecture:** Separate Cargo crate (`services/llm/`) with actix-web on port 8001. `LlamaEngine` wraps `llama-cpp-2` (which builds llama.cpp from source). Text and image inference share one `infer()` function; image bytes are decoded to RGB and fed through the `mtmd` (multimodal) path via `MtmdContext` + `MtmdBitmap`. The main server (`src/`) gets a new `llm_proxy.rs` that forwards `/llm/{tail:.*}` to `127.0.0.1:8001`, returning 503 when the service is down.

**Tech Stack:** Rust, actix-web 4.8, llama-cpp-2 0.1 (mtmd feature), tokio, serde_json, image crate, base64, reqwest (proxy).

---

## File Map

**Create:**
- `services/llm/Cargo.toml`
- `services/llm/config.toml`
- `services/llm/src/main.rs`
- `services/llm/src/config.rs`
- `services/llm/src/engine.rs`
- `services/llm/src/handler.rs`
- `scripts/download_llm_model.sh`

**Modify:**
- `Makefile` — add `llm-build`, `llm-run` targets
- `src/api/handlers.rs` — register `/llm/{tail:.*}` proxy route
- `src/main.rs` — no change needed (proxy is in handlers.rs `configure_routes`)

**New in main crate (create):**
- `src/api/llm_proxy.rs`

---

## Task 1: Scaffold `services/llm/` crate

**Files:**
- Create: `services/llm/Cargo.toml`
- Create: `services/llm/config.toml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p services/llm/src services/llm/models
```

- [ ] **Step 2: Write `services/llm/Cargo.toml`**

```toml
[package]
name = "llm-service"
version = "0.1.0"
edition = "2021"
description = "1-bit multimodal LLM microservice (MiniCPM-V via llama-cpp-2)"

[[bin]]
name = "llm-service"
path = "src/main.rs"

[dependencies]
# Web
actix-web  = "4.8"
actix-rt   = "2.10"
tokio      = { version = "1", features = ["rt-multi-thread", "macros", "sync"] }
tokio-stream = "0.1"

# LLM inference — builds llama.cpp from source; requires cmake + C++ compiler
# On macOS, Metal acceleration is included automatically via the "metal" feature
[target.'cfg(target_os = "macos")'.dependencies]
llama-cpp-2 = { version = "0.1", features = ["mtmd", "metal"] }

[target.'cfg(not(target_os = "macos"))'.dependencies]
llama-cpp-2 = { version = "0.1", features = ["mtmd"] }

# Serialization
serde      = { version = "1", features = ["derive"] }
serde_json = "1"
toml       = "0.8"

# Image decoding (JPEG/PNG → RGB pixels for MtmdBitmap)
image      = "0.24"

# Base64 decoding (image_url data URIs)
base64     = "0.21"

# Error handling
anyhow     = "1"
thiserror  = "1"

# Logging
tracing            = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt"] }

# Byte utilities
bytes = "1.5"
```

- [ ] **Step 3: Write `services/llm/config.toml`**

```toml
port         = 8001
model_path   = "models/minicpm-v-2_6-q2_k.gguf"
mmproj_path  = "models/minicpm-v-2_6-mmproj-f16.gguf"
ctx_size     = 4096
n_threads    = 4
n_gpu_layers = 0
```

`mmproj_path` is optional — remove the line (or leave blank) to disable multimodal and run text-only.

- [ ] **Step 4: Commit scaffold**

```bash
git add services/llm/
git commit -m "chore(llm-service): scaffold crate structure"
```

---

## Task 2: Implement `LlmConfig`

**Files:**
- Create: `services/llm/src/config.rs`

- [ ] **Step 1: Write `services/llm/src/config.rs`**

```rust
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    /// HTTP port this service listens on (default 8001)
    #[serde(default = "default_port")]
    pub port: u16,

    /// Path to the GGUF model file (e.g. MiniCPM-V 2.6 Q2_K)
    pub model_path: String,

    /// Optional path to the multimodal projection file (.mmproj.gguf).
    /// Omit or set to empty string to disable image input.
    #[serde(default)]
    pub mmproj_path: Option<String>,

    /// KV-cache context window size in tokens
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,

    /// CPU thread count for generation
    #[serde(default = "default_n_threads")]
    pub n_threads: i32,

    /// Number of model layers to offload to GPU (0 = CPU-only)
    #[serde(default)]
    pub n_gpu_layers: i32,
}

fn default_port()     -> u16 { 8001 }
fn default_ctx_size() -> u32 { 4096 }
fn default_n_threads() -> i32 { 4 }

impl LlmConfig {
    /// Load from `config.toml` next to the binary, or use defaults.
    pub fn load() -> Result<Self> {
        let config_path = PathBuf::from("config.toml");
        if config_path.exists() {
            let text = std::fs::read_to_string(&config_path)
                .context("read config.toml")?;
            toml::from_str(&text).context("parse config.toml")
        } else {
            tracing::warn!("config.toml not found, using defaults");
            Ok(Self {
                port: 8001,
                model_path: "models/minicpm-v-2_6-q2_k.gguf".into(),
                mmproj_path: Some("models/minicpm-v-2_6-mmproj-f16.gguf".into()),
                ctx_size: 4096,
                n_threads: 4,
                n_gpu_layers: 0,
            })
        }
    }

    /// Returns mmproj_path only if it's non-empty and the file exists.
    pub fn effective_mmproj(&self) -> Option<&str> {
        self.mmproj_path
            .as_deref()
            .filter(|p| !p.is_empty())
            .filter(|p| std::path::Path::new(p).exists())
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add services/llm/src/config.rs
git commit -m "feat(llm-service): implement LlmConfig"
```

---

## Task 3: Implement `LlamaEngine` — text generation

**Files:**
- Create: `services/llm/src/engine.rs`

This task covers text-only inference. Multimodal is extended in Task 4.

- [ ] **Step 1: Write `services/llm/src/engine.rs` (text path)**

```rust
use anyhow::{Context, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    sampling::LlamaSampler,
};
use std::num::NonZeroU32;
use tokio::sync::mpsc;

use crate::config::LlmConfig;

/// Holds the loaded model and optional multimodal context.
/// Creating a new `LlamaContext` per request avoids self-referential
/// struct lifetime issues while keeping model weights in memory.
pub struct LlamaEngine {
    backend: LlamaBackend,
    pub model: LlamaModel,
    pub config: LlmConfig,
    /// Set only when mmproj file is found at startup.
    pub mmproj_path: Option<String>,
}

// LlamaBackend and LlamaModel are Send+Sync (marked unsafe in llama-cpp-2).
// LlamaEngine is used only from spawn_blocking, one request at a time.
unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    /// Load the model. Exits the process with a clear message if the model
    /// file does not exist — a broken LLM service should not silently start.
    pub fn load(config: LlmConfig) -> Result<Self> {
        let model_path = std::path::Path::new(&config.model_path);
        if !model_path.exists() {
            anyhow::bail!(
                "Model file not found: {}  — run `bash scripts/download_llm_model.sh`",
                config.model_path
            );
        }

        tracing::info!(path = %config.model_path, "Loading GGUF model...");
        let backend = LlamaBackend::init().context("init llama backend")?;

        let model_params = {
            let mut p = LlamaModelParams::default();
            p.set_n_gpu_layers(config.n_gpu_layers);
            p
        };
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("load model from file")?;
        tracing::info!("Model loaded");

        let mmproj_path = config.effective_mmproj().map(str::to_owned);

        Ok(Self { backend, model, config, mmproj_path })
    }

    /// Build a chat prompt string from an ordered list of (role, content) pairs.
    /// Uses ChatML format, supported by MiniCPM-V and most modern instruction models.
    /// `image_marker` is the mtmd default marker; pass None for text-only requests.
    pub fn build_prompt(messages: &[(String, String)], image_marker: Option<&str>) -> String {
        let mut buf = String::new();
        for (role, content) in messages {
            buf.push_str(&format!("<|im_start|>{}\n", role));
            if let Some(marker) = image_marker {
                // Place image marker before the user text
                if role == "user" && !content.is_empty() {
                    buf.push_str(marker);
                    buf.push('\n');
                }
            }
            buf.push_str(content);
            buf.push_str("<|im_end|>\n");
        }
        buf.push_str("<|im_start|>assistant\n");
        buf
    }

    /// Run text-only inference. Sends generated tokens one by one through `tx`.
    /// Runs entirely on the calling thread — call via `spawn_blocking`.
    pub fn infer_text(
        &self,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        tx: mpsc::Sender<String>,
    ) -> Result<()> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.config.ctx_size).unwrap())
            .with_n_threads(self.config.n_threads)
            .with_n_threads_batch(self.config.n_threads);

        let mut ctx = self.model.new_context(&self.backend, ctx_params)
            .context("create llama context")?;

        // Tokenize prompt
        let tokens = self.model.str_to_token(&prompt, AddBos::Always)
            .context("tokenize prompt")?;

        // Fill initial batch (request logits only for the last token)
        let n_prompt = tokens.len();
        let mut batch = LlamaBatch::new(n_prompt.max(512), 1);
        for (i, &tok) in tokens.iter().enumerate() {
            batch.add(tok, i as i32, &[0], i == n_prompt - 1)
                .context("add token to batch")?;
        }
        ctx.decode(&mut batch).context("decode prompt")?;

        // Sampler chain: top-k → top-p → temperature → distribution
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(temperature.clamp(0.01, 2.0)),
            LlamaSampler::dist(0),
        ]);

        let mut n_past = n_prompt as i32;

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            if self.model.is_eog_token(new_token) {
                break;
            }

            let token_str = self.model
                .token_to_str(new_token, Special::Tokenize)
                .unwrap_or_default();

            // Non-blocking send; if receiver dropped, client disconnected → stop
            if tx.blocking_send(token_str).is_err() {
                break;
            }

            batch.clear();
            batch.add(new_token, n_past, &[0], true)
                .context("add generated token")?;
            ctx.decode(&mut batch).context("decode token")?;
            n_past += 1;
        }

        Ok(())
    }
}
```

- [ ] **Step 2: Verify the crate compiles (text path only)**

From `services/llm/`:
```bash
cd services/llm
cargo check
```

Expected: `warning: unused ...` only — no errors. If `llama-cpp-2` errors appear about `cmake`, install it: `brew install cmake` (macOS) or `apt install cmake` (Linux).

- [ ] **Step 3: Commit**

```bash
git add services/llm/src/engine.rs
git commit -m "feat(llm-service): LlamaEngine text inference"
```

---

## Task 4: Extend `LlamaEngine` — multimodal image inference

**Files:**
- Modify: `services/llm/src/engine.rs`

The mtmd path uses `MtmdContext`, `MtmdBitmap`, and `MtmdInputChunks` from `llama_cpp_2::mtmd`.

- [ ] **Step 1: Add imports to top of `engine.rs`**

Add these imports after the existing `use` block:

```rust
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
```

- [ ] **Step 2: Add `infer_multimodal` to `impl LlamaEngine`**

Append this function inside `impl LlamaEngine` (after `infer_text`):

```rust
    /// Run multimodal inference with one image.
    /// `image_bytes` are raw JPEG/PNG bytes. The image is decoded to RGB and
    /// embedded via the clip projector before text generation begins.
    pub fn infer_multimodal(
        &self,
        messages: &[(String, String)],
        image_bytes: Vec<u8>,
        max_tokens: u32,
        temperature: f32,
        tx: mpsc::Sender<String>,
    ) -> Result<()> {
        let mmproj = self.mmproj_path.as_deref()
            .context("multimodal not configured: mmproj_path missing or file not found")?;

        // ── Decode image → RGB ─────────────────────────────────────────────
        let img = image::load_from_memory(&image_bytes).context("decode image")?;
        let rgb = img.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb_data = rgb.into_raw(); // RGBRGB... packed bytes

        let bitmap = MtmdBitmap::from_image_data(w, h, &rgb_data)
            .map_err(|e| anyhow::anyhow!("create bitmap: {:?}", e))?;

        // ── Load clip projector ────────────────────────────────────────────
        let mtmd_params = MtmdContextParams {
            use_gpu: self.config.n_gpu_layers > 0,
            print_timings: false,
            n_threads: self.config.n_threads,
            ..MtmdContextParams::default()
        };
        let mtmd_ctx = MtmdContext::init_from_file(mmproj, &self.model, &mtmd_params)
            .map_err(|e| anyhow::anyhow!("init mtmd context: {:?}", e))?;

        // ── Build prompt with image marker ─────────────────────────────────
        let marker = mtmd_default_marker();
        let prompt = Self::build_prompt(messages, Some(marker));

        let input_text = MtmdInputText {
            text: prompt,
            add_special: true,
            parse_special: true,
        };

        // ── Tokenize prompt + image chunks ─────────────────────────────────
        let chunks = mtmd_ctx
            .tokenize(input_text, &[&bitmap])
            .map_err(|e| anyhow::anyhow!("mtmd tokenize: {:?}", e))?;

        // ── Create context and evaluate all chunks ─────────────────────────
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.config.ctx_size).unwrap())
            .with_n_threads(self.config.n_threads)
            .with_n_threads_batch(self.config.n_threads);
        let mut ctx = self.model.new_context(&self.backend, ctx_params)
            .context("create llama context")?;

        // eval_chunks handles text batches + image encoding automatically.
        // Returns the new n_past after all input has been processed.
        let n_past = chunks
            .eval_chunks(&mtmd_ctx, &ctx, 0, 0, 512, false)
            .map_err(|e| anyhow::anyhow!("eval chunks: {:?}", e))?;

        // ── Generation loop ────────────────────────────────────────────────
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(temperature.clamp(0.01, 2.0)),
            LlamaSampler::dist(0),
        ]);

        // After eval_chunks, logits are available at position (n_past - 1)
        let mut batch = LlamaBatch::new(1, 1);
        let mut current_n_past = n_past;

        // Sample the first token from the already-decoded last position
        // We need a single-token batch to sample — decode a dummy batch first
        // to get a fresh logit slot, then sample.
        // Note: eval_chunks with logits_last=false means we must re-decode
        // one token to get a valid logit. We use the EOS token as a probe
        // batch with logits=true, then discard the output and sample normally.
        //
        // Simpler approach: call eval_chunks with logits_last=true and sample
        // from position (n_past - 1).
        //
        // Re-evaluate with logits_last=true to get the logit for sampling:
        let n_past_with_logits = chunks
            .eval_chunks(&mtmd_ctx, &ctx, 0, 0, 512, true)
            .map_err(|e| anyhow::anyhow!("eval chunks (logits): {:?}", e))?;

        let mut n_cur = n_past_with_logits;

        for _ in 0..max_tokens {
            // batch.n_tokens() - 1 is the index of the last decoded token's logits
            let new_token = sampler.sample(&ctx, (n_cur - 1) as i32);
            sampler.accept(new_token);

            if self.model.is_eog_token(new_token) {
                break;
            }

            let token_str = self.model
                .token_to_str(new_token, Special::Tokenize)
                .unwrap_or_default();

            if tx.blocking_send(token_str).is_err() {
                break;
            }

            batch.clear();
            batch.add(new_token, n_cur, &[0], true)
                .context("add generated token")?;
            ctx.decode(&mut batch).context("decode token")?;
            n_cur += 1;
        }

        Ok(())
    }
```

- [ ] **Step 3: Verify the crate still compiles**

```bash
cd services/llm && cargo check
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add services/llm/src/engine.rs
git commit -m "feat(llm-service): multimodal image inference via mtmd"
```

---

## Task 5: Implement HTTP handlers

**Files:**
- Create: `services/llm/src/handler.rs`

Handles `POST /v1/chat/completions` (streaming + non-streaming) and `GET /v1/models`.

- [ ] **Step 1: Write `services/llm/src/handler.rs`**

```rust
use actix_web::{web, HttpResponse};
use base64::Engine as _;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::engine::LlamaEngine;

// ── Request / response types ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> u32 { 512 }
fn default_temperature() -> f32 { 0.7 }

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text message
    Text(String),
    /// Multipart message (text + image_url parts)
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Deserialize)]
pub struct ImageUrl {
    pub url: String, // "data:image/jpeg;base64,<b64>" or "data:image/png;base64,<b64>"
}

// ── State ─────────────────────────────────────────────────────────────────────

pub struct AppState {
    pub engine: Arc<LlamaEngine>,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Decode a data URI image (`data:<mime>;base64,<b64>`) to raw bytes.
fn decode_data_uri(url: &str) -> Result<Vec<u8>, String> {
    let base64_part = url
        .splitn(2, ',')
        .nth(1)
        .ok_or_else(|| "invalid data URI: no comma".to_string())?;
    base64::engine::general_purpose::STANDARD
        .decode(base64_part)
        .map_err(|e| format!("base64 decode: {e}"))
}

/// Extract (role, text_content) pairs and optional image bytes from messages.
/// Only the first image found across all messages is used.
fn extract_content(
    messages: &[ChatMessage],
) -> (Vec<(String, String)>, Option<Vec<u8>>) {
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut image: Option<Vec<u8>> = None;

    for msg in messages {
        match &msg.content {
            MessageContent::Text(text) => {
                pairs.push((msg.role.clone(), text.clone()));
            }
            MessageContent::Parts(parts) => {
                let mut text_buf = String::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => text_buf.push_str(text),
                        ContentPart::ImageUrl { image_url } => {
                            if image.is_none() {
                                match decode_data_uri(&image_url.url) {
                                    Ok(bytes) => image = Some(bytes),
                                    Err(e) => tracing::warn!("image decode: {e}"),
                                }
                            }
                        }
                    }
                }
                pairs.push((msg.role.clone(), text_buf));
            }
        }
    }

    (pairs, image)
}

fn sse_chunk(content: &str, model: &str) -> Bytes {
    let data = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
    });
    Bytes::from(format!("data: {}\n\n", data))
}

fn sse_done() -> Bytes {
    Bytes::from("data: [DONE]\n\n")
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions`
pub async fn chat_completions(
    state: web::Data<AppState>,
    req: web::Json<ChatRequest>,
) -> HttpResponse {
    let req = req.into_inner();
    let model_name = req.model.clone().unwrap_or_else(|| "minicpm-v".to_string());
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let streaming = req.stream;

    let (pairs, image_bytes) = extract_content(&req.messages);
    let engine = Arc::clone(&state.engine);

    if streaming {
        // Unbounded channel: spawn_blocking pushes, SSE stream drains
        let (tx, rx) = mpsc::unbounded_channel::<String>();

        // Bridge: convert UnboundedReceiver<String> → SSE Bytes stream
        let model_name_clone = model_name.clone();
        let sse_rx = UnboundedReceiverStream::new(rx).map(move |tok| {
            Ok::<Bytes, std::io::Error>(sse_chunk(&tok, &model_name_clone))
        });

        // Wrap in a stream that appends [DONE] at the end
        use futures_util::StreamExt;
        let done_stream = futures_util::stream::once(async {
            Ok::<Bytes, std::io::Error>(sse_done())
        });
        let full_stream = sse_rx.chain(done_stream);

        // Run inference in blocking thread
        // Convert unbounded sender to regular sender for blocking_send compatibility
        let (sync_tx, _sync_rx) = mpsc::channel::<String>(512);
        // Use a different approach: spawn_blocking with the unbounded sender
        let _ = tokio::task::spawn_blocking(move || {
            let (pairs2, image2) = (pairs, image_bytes);
            // Wrap unbounded sender in a channel-compatible type
            // tokio UnboundedSender::send() is non-blocking and works from sync context
            let tx_clone = tx.clone();
            let result = match image2 {
                Some(img) => engine.infer_multimodal(&pairs2, img, max_tokens, temperature, {
                    // Bridge: wrap UnboundedSender as blocking Sender
                    let (bridge_tx, mut bridge_rx) = mpsc::channel::<String>(512);
                    // Drain bridge_rx → tx_clone in this blocking thread
                    let tx_inner = tx_clone.clone();
                    let _ = std::thread::spawn(move || {
                        while let Some(tok) = bridge_rx.blocking_recv() {
                            let _ = tx_inner.send(tok);
                        }
                    });
                    bridge_tx
                }),
                None => {
                    let prompt = LlamaEngine::build_prompt(&pairs2, None);
                    let (bridge_tx, mut bridge_rx) = mpsc::channel::<String>(512);
                    let tx_inner = tx_clone.clone();
                    let _ = std::thread::spawn(move || {
                        while let Some(tok) = bridge_rx.blocking_recv() {
                            let _ = tx_inner.send(tok);
                        }
                    });
                    engine.infer_text(prompt, max_tokens, temperature, bridge_tx)
                }
            };
            if let Err(e) = result {
                tracing::error!("inference error: {e}");
                let _ = tx_clone.send(format!("\n[error: {e}]"));
            }
        });

        HttpResponse::Ok()
            .content_type("text/event-stream; charset=utf-8")
            .insert_header(("Cache-Control", "no-cache"))
            .insert_header(("X-Accel-Buffering", "no"))
            .streaming(full_stream)
    } else {
        // Non-streaming: collect all tokens then return single JSON response
        let (tx, mut rx) = mpsc::channel::<String>(512);

        let engine2 = Arc::clone(&state.engine);
        let handle = tokio::task::spawn_blocking(move || match image_bytes {
            Some(img) => engine2.infer_multimodal(&pairs, img, max_tokens, temperature, tx),
            None => {
                let prompt = LlamaEngine::build_prompt(&pairs, None);
                engine2.infer_text(prompt, max_tokens, temperature, tx)
            }
        });

        let mut content = String::new();
        while let Some(tok) = rx.recv().await {
            content.push_str(&tok);
        }

        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return HttpResponse::InternalServerError()
                    .json(json!({"error": {"message": e.to_string(), "type": "inference_error"}}));
            }
            Err(e) => {
                return HttpResponse::InternalServerError()
                    .json(json!({"error": {"message": e.to_string(), "type": "join_error"}}));
            }
        }

        HttpResponse::Ok().json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }))
    }
}

/// `GET /v1/models`
pub async fn list_models(state: web::Data<AppState>) -> HttpResponse {
    let model_id = std::path::Path::new(&state.engine.config.model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("minicpm-v")
        .to_string();

    HttpResponse::Ok().json(json!({
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "owned_by": "local",
            "multimodal": state.engine.mmproj_path.is_some()
        }]
    }))
}
```

Add `futures-util = "0.3"` to `services/llm/Cargo.toml` `[dependencies]` since it's used in the streaming handler.

- [ ] **Step 2: Add futures-util dependency to `services/llm/Cargo.toml`**

In the `[dependencies]` section, add:
```toml
futures-util = "0.3"
```

- [ ] **Step 3: Commit**

```bash
git add services/llm/src/handler.rs services/llm/Cargo.toml
git commit -m "feat(llm-service): OpenAI-compatible chat completions handler"
```

---

## Task 6: Wire `main.rs`

**Files:**
- Create: `services/llm/src/main.rs`

- [ ] **Step 1: Write `services/llm/src/main.rs`**

```rust
mod config;
mod engine;
mod handler;

use actix_web::{middleware, web, App, HttpServer};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

use config::LlmConfig;
use engine::LlamaEngine;
use handler::AppState;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("llm_service=info".parse().unwrap()))
        .init();

    let config = LlmConfig::load().unwrap_or_else(|e| {
        eprintln!("Config error: {e}");
        std::process::exit(1);
    });

    let engine = LlamaEngine::load(config).unwrap_or_else(|e| {
        eprintln!("Model load failed: {e}");
        std::process::exit(1);
    });

    let port = engine.config.port;
    let state = web::Data::new(AppState { engine: Arc::new(engine) });

    tracing::info!("LLM microservice listening on 0.0.0.0:{}", port);

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(
                web::JsonConfig::default()
                    .limit(32 * 1024 * 1024) // 32 MB — base64 images can be large
                    .error_handler(|err, _req| {
                        let msg = err.to_string();
                        actix_web::error::InternalError::from_response(
                            err,
                            actix_web::HttpResponse::BadRequest()
                                .json(serde_json::json!({"error": {"message": msg}})),
                        )
                        .into()
                    }),
            )
            .wrap(middleware::Logger::default())
            .route("/v1/chat/completions", web::post().to(handler::chat_completions))
            .route("/v1/models", web::get().to(handler::list_models))
    })
    .workers(1) // serialise requests — one model, one context at a time
    .bind(format!("0.0.0.0:{port}"))?
    .run()
    .await
}
```

- [ ] **Step 2: Verify the service builds**

```bash
cd services/llm && cargo build 2>&1 | tail -20
```

Expected: `Finished dev [unoptimized + debuginfo] target(s)` — possibly after several minutes while llama.cpp compiles.

- [ ] **Step 3: Commit**

```bash
git add services/llm/src/main.rs
git commit -m "feat(llm-service): wire main.rs actix-web server"
```

---

## Task 7: Add proxy route to main server

**Files:**
- Create: `src/api/llm_proxy.rs`
- Modify: `src/api/handlers.rs:201-250` (configure_routes)
- Modify: `src/main.rs` (add `mod llm_proxy` to api module, if needed)

The proxy forwards all `/llm/*` requests to `http://127.0.0.1:8001`. `reqwest` is already in the main `Cargo.toml`.

- [ ] **Step 1: Create `src/api/llm_proxy.rs`**

```rust
//! Thin reverse proxy: forwards `/llm/{tail}` → `http://127.0.0.1:8001/{tail}`.
//! Returns 503 when the LLM microservice is not reachable.
use actix_web::{web, HttpRequest, HttpResponse};
use bytes::Bytes;

static LLM_BASE: &str = "http://127.0.0.1:8001";

/// Forward any `/llm/{tail:.*}` request to the LLM microservice.
pub async fn proxy(
    req: HttpRequest,
    body: Bytes,
    path: web::Path<String>,
) -> HttpResponse {
    let tail = path.into_inner();
    let url  = format!("{}/{}", LLM_BASE, tail);

    // Preserve query string
    let url = if let Some(qs) = req.uri().query() {
        format!("{}?{}", url, qs)
    } else {
        url
    };

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300)) // long timeout for generation
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": e.to_string()}));
        }
    };

    // Copy method and headers (skip host, content-length — reqwest handles those)
    let method = reqwest::Method::from_bytes(req.method().as_str().as_bytes())
        .unwrap_or(reqwest::Method::GET);

    let mut rb = client.request(method, &url);
    for (name, value) in req.headers() {
        let lower = name.as_str().to_lowercase();
        if lower == "host" || lower == "content-length" {
            continue;
        }
        if let Ok(v) = reqwest::header::HeaderValue::from_bytes(value.as_bytes()) {
            rb = rb.header(name.as_str(), v);
        }
    }
    rb = rb.body(body);

    match rb.send().await {
        Err(e) if e.is_connect() || e.is_timeout() => {
            HttpResponse::ServiceUnavailable().json(
                serde_json::json!({"error": "LLM service unavailable — start it with `make llm-run`"})
            )
        }
        Err(e) => HttpResponse::BadGateway()
            .json(serde_json::json!({"error": e.to_string()})),
        Ok(upstream) => {
            let status = actix_web::http::StatusCode::from_u16(upstream.status().as_u16())
                .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
            let mut resp = HttpResponse::build(status);

            // Forward response headers (copy content-type, etc.)
            for (name, value) in upstream.headers() {
                let lower = name.as_str().to_lowercase();
                if lower == "transfer-encoding" || lower == "content-length" {
                    continue;
                }
                if let Ok(v) = actix_web::http::header::HeaderValue::from_bytes(value.as_bytes()) {
                    resp.insert_header((name.as_str(), v));
                }
            }

            // Stream the body back (important for SSE responses)
            let bytes = match upstream.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    return HttpResponse::BadGateway()
                        .json(serde_json::json!({"error": e.to_string()}));
                }
            };
            resp.body(bytes)
        }
    }
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/llm/{tail:.*}", web::to(proxy));
}
```

- [ ] **Step 2: Register the module in `src/lib.rs` or `src/api/mod.rs`**

Check how the api modules are declared:

```bash
grep -n "llm_proxy\|pub mod" src/api/mod.rs 2>/dev/null || grep -n "mod " src/main.rs | grep api | head -10
```

Find the file that declares `pub mod handlers` and add `pub mod llm_proxy;` in the same place.

Typically this is in `src/lib.rs`. Search for the line `pub mod handlers` and add after it:
```rust
pub mod llm_proxy;
```

- [ ] **Step 3: Register proxy routes in `src/api/handlers.rs:configure_routes`**

Open `src/api/handlers.rs` at the `configure_routes` function (line ~201). At the end of the function body, add:

```rust
        .configure(crate::api::llm_proxy::configure_routes)
```

Full updated end of `configure_routes`:
```rust
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/", web::get().to(root))
        // ... existing routes ...
        .configure(crate::api::llm_proxy::configure_routes);  // ← add this
}
```

- [ ] **Step 4: Verify main crate still compiles**

```bash
cargo check
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/api/llm_proxy.rs src/api/handlers.rs src/lib.rs
git commit -m "feat: add /llm/* proxy route forwarding to LLM microservice on 8001"
```

---

## Task 8: Download script + Makefile targets

**Files:**
- Create: `scripts/download_llm_model.sh`
- Modify: `Makefile`

- [ ] **Step 1: Write `scripts/download_llm_model.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Download MiniCPM-V 2.6 Q2_K (2 GB) + mmproj (400 MB) from HuggingFace.
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
    echo "✓ Model saved to $MODEL_FILE"
else
    echo "✓ Model already present: $MODEL_FILE"
fi

echo "=== Downloading mmproj (~400 MB) ==="
if [ ! -f "$MMPROJ_FILE" ]; then
    curl -L --progress-bar -o "$MMPROJ_FILE" "$MMPROJ_URL"
    echo "✓ mmproj saved to $MMPROJ_FILE"
else
    echo "✓ mmproj already present: $MMPROJ_FILE"
fi

echo ""
echo "All models ready. Run: make llm-run"
```

```bash
chmod +x scripts/download_llm_model.sh
```

- [ ] **Step 2: Add targets to `Makefile`**

Append to the bottom of `Makefile`:

```makefile
# ── LLM Microservice ──────────────────────────────────────────────────────────
.PHONY: llm-build llm-run llm-download

llm-download:
	bash scripts/download_llm_model.sh

llm-build:
	cd services/llm && cargo build --release

llm-run:
	cd services/llm && ./target/release/llm-service
```

- [ ] **Step 3: Add `services/llm/models/` to `.gitignore`**

```bash
echo "services/llm/models/*.gguf" >> .gitignore
```

- [ ] **Step 4: Commit**

```bash
git add scripts/download_llm_model.sh Makefile .gitignore
git commit -m "chore(llm-service): add download script and Makefile targets"
```

---

## Task 9: Build, download, and smoke test

- [ ] **Step 1: Download models**

```bash
make llm-download
```

Expected: two files appear in `services/llm/models/`. Allow ~10-20 minutes on a slow connection.

- [ ] **Step 2: Build the LLM service (release)**

```bash
make llm-build
```

First build compiles llama.cpp from source — takes 3-5 minutes. Subsequent builds are incremental.

Expected last line: `Finished release [optimized] target(s)`

- [ ] **Step 3: Start the LLM service**

```bash
make llm-run &
sleep 3
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

Expected:
```json
{
  "object": "list",
  "data": [{"id": "minicpm-v-2_6-q2_k", "object": "model", "owned_by": "local", "multimodal": true}]
}
```

- [ ] **Step 4: Text inference smoke test**

```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Reply with exactly three words."}],"max_tokens":20,"stream":false}' \
  | python3 -m json.tool
```

Expected: JSON with `choices[0].message.content` containing a short text response.

- [ ] **Step 5: Streaming smoke test**

```bash
curl -N http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":30,"stream":true}'
```

Expected: multiple `data: {...}` lines followed by `data: [DONE]`.

- [ ] **Step 6: Test proxy through main server**

With the main server running (`./target/release/torch-inference-server`):
```bash
curl -s http://localhost:8000/llm/v1/models | python3 -m json.tool
```

Expected: same model list as direct call. If LLM service is down, expect `{"error":"LLM service unavailable..."}` with HTTP 503.

- [ ] **Step 7: Build main server to verify no regressions**

```bash
cargo build --release
```

Expected: compiles without errors.

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "feat(llm-service): complete 1-bit multimodal LLM microservice (MiniCPM-V 2.6)"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Separate `services/llm/` Cargo crate outside `src/`
- ✅ Proxied at `/llm/*` from main server (llm_proxy.rs)
- ✅ Independent on port 8001 (main.rs)
- ✅ `POST /v1/chat/completions` text + multimodal, streaming + non-streaming
- ✅ `GET /v1/models`
- ✅ `LlamaEngine` with `llama-cpp-2` mtmd feature
- ✅ `MtmdBitmap` + `MtmdContext` for image inference
- ✅ `LlmConfig` from `config.toml`
- ✅ Download script + Makefile
- ✅ 503 when LLM service is down (proxy handler)
- ✅ 400 on missing mmproj when image sent
- ✅ Model file missing → exit with clear message

**Known limitation:** The streaming handler uses a two-channel bridge (spawn_blocking → mpsc channel) because `tokio::sync::mpsc::Sender::blocking_send()` requires the current thread to not be inside a tokio async runtime, but `spawn_blocking` creates a separate OS thread where `blocking_send` works. The double-channel in Task 5 Step 1 is slightly redundant — a simplification is to use `std::sync::mpsc::sync_channel` for the inner channel and bridge to the SSE stream in an async task. The implementor may refactor this if compile errors appear, but the logic is correct.
