use actix_web::{web, HttpResponse};
use base64::Engine as _;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use futures_util::StreamExt;

use crate::engine::LlamaEngine;

// ── State ─────────────────────────────────────────────────────────────────────

pub struct AppState {
    pub engine: Arc<LlamaEngine>,
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
fn extract_content(messages: &[ChatMessage]) -> Result<(Vec<(String, String)>, Option<Vec<u8>>), String> {
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
                                image = Some(decode_data_uri(&image_url.url)
                                    .map_err(|e| format!("invalid image: {e}"))?);
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

// ── Handlers ─────────────────────────────────────────────────────────────────

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

    let (pairs, image_bytes) = match extract_content(&req.messages) {
        Ok(v) => v,
        Err(e) => return HttpResponse::BadRequest().json(json!({"error": e})),
    };

    // Pre-flight: image provided but no multimodal projector loaded
    if image_bytes.is_some() && state.engine.mmproj_path.is_none() {
        return HttpResponse::BadRequest()
            .json(json!({"error": "multimodal not configured: mmproj_path missing or file not found"}));
    }

    let engine = Arc::clone(&state.engine);

    if streaming {
        // Channel for token strings from the blocking inference thread
        let (tx, rx) = mpsc::channel::<String>(128);

        let engine2 = Arc::clone(&engine);
        let pairs2 = pairs.clone();
        tokio::task::spawn_blocking(move || {
            let result = match image_bytes {
                Some(img) => engine2.infer_multimodal(&pairs2, img, max_tokens, temperature, tx),
                None => {
                    let prompt = LlamaEngine::build_prompt(&pairs2, None);
                    engine2.infer_text(prompt, max_tokens, temperature, tx)
                }
            };
            if let Err(e) = result {
                tracing::error!("inference error: {e:#}");
            }
        });

        let model_for_stream = model_name.clone();
        let token_stream = ReceiverStream::new(rx)
            .map(move |tok| Ok::<Bytes, std::io::Error>(sse_chunk(&tok, &model_for_stream)));
        let done_stream = futures_util::stream::once(async {
            Ok::<Bytes, std::io::Error>(sse_done())
        });
        let full_stream = token_stream.chain(done_stream);

        HttpResponse::Ok()
            .content_type("text/event-stream; charset=utf-8")
            .insert_header(("Cache-Control", "no-cache"))
            .insert_header(("X-Accel-Buffering", "no"))
            .streaming(full_stream)
    } else {
        // Non-streaming: collect all tokens, return single JSON
        let (tx, mut rx) = mpsc::channel::<String>(512);

        let handle = tokio::task::spawn_blocking(move || match image_bytes {
            Some(img) => engine.infer_multimodal(&pairs, img, max_tokens, temperature, tx),
            None => {
                let prompt = LlamaEngine::build_prompt(&pairs, None);
                engine.infer_text(prompt, max_tokens, temperature, tx)
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
                    .json(json!({"error": format!("inference failed: {}", e)}));
            }
            Err(e) => {
                return HttpResponse::InternalServerError()
                    .json(json!({"error": format!("inference failed: {}", e)}));
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
