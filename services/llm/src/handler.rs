use actix_web::{web, HttpResponse};
use base64::Engine as _;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use futures_util::StreamExt;

use crate::hrm_engine::HrmEngine;

// ── State ─────────────────────────────────────────────────────────────────────

pub struct AppState {
    pub engine: Arc<crate::hrm_engine::HrmEngine>,
    pub vision: Option<Arc<crate::vision_bridge::VisionBridge>>,
    pub lease: crate::engine_lease::EngineLease,
    pub gate: Arc<crate::memory_gate::MemoryGate>,
    pub limits: crate::config::LimitsConfig,
}

// ── Request types ─────────────────────────────────────────────────────────────

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
    Text(String),
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
    pub url: String,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Per-request response id (`chatcmpl-<uuid>`) for OpenAI-API parity.
fn new_completion_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

/// Unix epoch seconds for the OpenAI `created` field.
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn decode_data_uri(url: &str) -> Result<Vec<u8>, String> {
    let base64_part = url
        .splitn(2, ',')
        .nth(1)
        .ok_or_else(|| "invalid data URI: no comma".to_string())?;
    base64::engine::general_purpose::STANDARD
        .decode(base64_part)
        .map_err(|e| format!("base64 decode: {e}"))
}

/// Extract (role, text) pairs and the first image bytes from the messages.
/// Returns Err if an image_url is present but cannot be decoded.
fn extract_content(
    messages: &[ChatMessage],
    max_image_bytes: usize,
) -> Result<(Vec<(String, String)>, Option<Vec<u8>>), String> {
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
                                let bytes = decode_data_uri(&image_url.url)
                                    .map_err(|e| format!("invalid image: {e}"))?;
                                if bytes.len() > max_image_bytes {
                                    return Err(format!(
                                        "image exceeds {} bytes ({} actual)",
                                        max_image_bytes, bytes.len()));
                                }
                                image = Some(bytes);
                            }
                        }
                    }
                }
                pairs.push((msg.role.clone(), text_buf));
            }
        }
    }

    Ok((pairs, image))
}

fn sse_chunk(id: &str, created: u64, content: &str, model: &str) -> Bytes {
    let data = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": null}]
    });
    Bytes::from(format!("data: {}\n\n", data))
}

fn sse_done() -> Bytes {
    Bytes::from("data: [DONE]\n\n")
}

// ── Handlers ─────────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions`
pub async fn chat_completions(
    state: web::Data<AppState>,
    req: web::Json<ChatRequest>,
) -> HttpResponse {
    let req = req.into_inner();

    // ── Admission + bounds ──────────────────────────────────────────────
    if req.messages.len() > state.limits.max_messages {
        return HttpResponse::BadRequest().json(json!({
            "error": format!("messages exceeds max ({} > {})",
                             req.messages.len(), state.limits.max_messages)
        }));
    }
    if let Err(e) = state.gate.admit() {
        return HttpResponse::ServiceUnavailable()
            .insert_header(("Retry-After", "1"))
            .json(json!({"error": e.to_string()}));
    }

    let model_name = req.model.clone().unwrap_or_else(|| "hrm-text-1b".to_string());
    let temperature = req.temperature;
    let streaming = req.stream;
    let id = new_completion_id();
    let created = now_unix_secs();

    let (mut pairs, image_bytes) = match extract_content(&req.messages, state.limits.max_image_bytes) {
        Ok(v) => v,
        Err(e) if e.starts_with("image exceeds") =>
            return HttpResponse::PayloadTooLarge().json(json!({"error": e})),
        Err(e) => return HttpResponse::BadRequest().json(json!({"error": e})),
    };

    if let Some(img) = image_bytes {
        let prefix = match state.vision.as_ref() {
            Some(vb) => vb.describe(&img).await,
            None => "[Image attached but vision bridge disabled.]".to_string(),
        };
        // Prepend description to the last user message.
        if let Some((_role, content)) = pairs.iter_mut().rev().find(|(r, _)| r == "user") {
            *content = format!("{prefix}\n{content}");
        } else {
            pairs.push(("user".into(), prefix));
        }
    }

    let prompt = build_prompt(&pairs);
    if prompt.len() > state.limits.max_prompt_chars {
        return HttpResponse::BadRequest().json(json!({
            "error": format!("prompt exceeds {} chars ({} actual)",
                             state.limits.max_prompt_chars, prompt.len())
        }));
    }
    // Clamp generated tokens to the configured ceiling regardless of what the
    // client requested — the unbounded value is an OOM lever.
    let max_tokens = req.max_tokens.min(state.limits.max_generated_tokens);
    let engine = Arc::clone(&state.engine);

    if streaming {
        let (tx, rx) = mpsc::channel::<String>(state.limits.channels.chat_stream_buffer);

        let engine2 = Arc::clone(&engine);
        let prompt2 = prompt.clone();
        let lease = state.lease.clone();
        tokio::spawn(async move {
            // Serialize every ONNX run behind the engine lease so concurrent
            // requests can't multiply peak inference memory.
            let _permit = lease.acquire().await;
            let res = tokio::task::spawn_blocking(move || {
                engine2.infer_text(prompt2, max_tokens, temperature, tx)
            }).await;
            match res {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::error!("inference error: {e:#}"),
                Err(e) => tracing::error!("inference task join error: {e}"),
            }
        });

        let model_for_stream = model_name.clone();
        let id_for_stream = id.clone();
        let token_stream = ReceiverStream::new(rx)
            .map(move |tok| Ok::<Bytes, std::io::Error>(
                sse_chunk(&id_for_stream, created, &tok, &model_for_stream)));
        let done_stream = futures_util::stream::once(async {
            Ok::<Bytes, std::io::Error>(sse_done())
        });
        HttpResponse::Ok()
            .content_type("text/event-stream; charset=utf-8")
            .insert_header(("Cache-Control", "no-cache"))
            .insert_header(("X-Accel-Buffering", "no"))
            .streaming(token_stream.chain(done_stream))
    } else {
        let (tx, mut rx) = mpsc::channel::<String>(state.limits.channels.chat_nonstream_buffer);
        let lease = state.lease.clone();
        let handle = tokio::spawn(async move {
            let _permit = lease.acquire().await;
            tokio::task::spawn_blocking(move || {
                engine.infer_text(prompt, max_tokens, temperature, tx)
            }).await
        });

        let mut content = String::new();
        while let Some(tok) = rx.recv().await {
            content.push_str(&tok);
        }
        let inference = match handle.await {
            Ok(inner) => inner.unwrap_or_else(|e| Err(anyhow::anyhow!("join inner: {e}"))),
            Err(e)    => Err(anyhow::anyhow!("join outer: {e}")),
        };
        if let Err(e) = inference {
            return HttpResponse::InternalServerError()
                .json(json!({"error": format!("inference failed: {e}")}));
        }

        HttpResponse::Ok().json(json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
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

/// Build a ChatML-formatted prompt. Same shape as the legacy LlamaEngine::build_prompt
/// minus the multimodal marker.
fn build_prompt(messages: &[(String, String)]) -> String {
    let mut buf = String::new();
    for (role, content) in messages {
        buf.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    buf.push_str("<|im_start|>assistant\n");
    buf
}

/// `GET /v1/models`
pub async fn list_models(state: web::Data<AppState>) -> HttpResponse {
    let _ = state;
    HttpResponse::Ok().json(json!({
        "object": "list",
        "data": [{
            "id": "hrm-text-1b",
            "object": "model",
            "owned_by": "local",
            "multimodal": true  // vision_bridge handles images
        }]
    }))
}
