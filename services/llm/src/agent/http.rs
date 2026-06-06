//! actix handler at POST /v1/agent/run. Streams SSE frames.

use actix_web::{web, HttpResponse, http::header};
use bytes::Bytes;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};

use crate::agent::executor::{run_agent, ExecOptions, Input};
use crate::agent::planner::Planner;
use crate::agent::sse::AgentEvent;
use crate::agent::tool::ToolRegistry;
use crate::config::AgentConfig;

#[derive(Debug, Deserialize)]
pub struct AgentRunRequest {
    pub messages: Vec<ChatMsg>,
    #[serde(default)]
    pub input: Option<AgentInput>,
    #[serde(default)]
    pub config: Option<AgentConfigOverride>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Default, Deserialize)]
pub struct AgentInput {
    #[serde(default)] pub image: Option<String>,   // data URI or raw b64
    #[serde(default)] pub audio: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct AgentConfigOverride {
    #[serde(default)] pub max_steps:    Option<usize>,
    #[serde(default)] pub max_run_ms:   Option<u64>,
    #[serde(default)] pub per_tool_ms:  Option<u64>,
    #[serde(default)] pub temperature:  Option<f32>,
}

pub struct AgentLayer {
    pub planner:  Arc<dyn Planner>,
    pub registry: Arc<ToolRegistry>,
    pub config:   AgentConfig,
    pub sem:      Arc<Semaphore>,
    pub gate:     Arc<crate::memory_gate::MemoryGate>,
    pub limits:   crate::config::LimitsConfig,
}

impl AgentLayer {
    pub fn new(
        planner: Arc<dyn Planner>,
        registry: Arc<ToolRegistry>,
        config: AgentConfig,
        gate: Arc<crate::memory_gate::MemoryGate>,
        limits: crate::config::LimitsConfig,
    ) -> Self {
        let sem = Arc::new(Semaphore::new(config.max_concurrent_runs.max(1)));
        Self { planner, registry, config, sem, gate, limits }
    }
}

pub async fn run(
    layer: web::Data<AgentLayer>,
    req: web::Json<AgentRunRequest>,
) -> HttpResponse {
    if !layer.config.enabled {
        return HttpResponse::NotFound().json(serde_json::json!({"error":"agent disabled"}));
    }

    if let Err(e) = layer.gate.admit() {
        return HttpResponse::ServiceUnavailable()
            .insert_header(("Retry-After", "1"))
            .json(serde_json::json!({"error": e.to_string()}));
    }

    let permit = match layer.sem.clone().try_acquire_owned() {
        Ok(p)  => p,
        Err(_) => return HttpResponse::TooManyRequests()
            .json(serde_json::json!({"error":"max_concurrent_runs reached"})),
    };

    let req = req.into_inner();

    if req.messages.len() > layer.limits.max_messages {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": format!("messages exceeds max ({} > {})",
                             req.messages.len(), layer.limits.max_messages)
        }));
    }

    let user_msg = req.messages.iter().rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    if user_msg.is_empty() {
        return HttpResponse::BadRequest()
            .json(serde_json::json!({"error":"messages must contain a user message"}));
    }

    let inputs = match stage_inputs(&req.input, layer.limits.max_image_bytes) {
        Ok(m) => m,
        Err(e) => return HttpResponse::PayloadTooLarge().json(serde_json::json!({"error": e})),
    };

    let opts = ExecOptions {
        max_steps:           req.config.as_ref().and_then(|c| c.max_steps).unwrap_or(layer.config.max_steps),
        max_run_ms:          req.config.as_ref().and_then(|c| c.max_run_ms).unwrap_or(layer.config.max_run_ms),
        per_tool_ms:         req.config.as_ref().and_then(|c| c.per_tool_ms).unwrap_or(layer.config.per_tool_ms),
        planner_temperature: req.config.as_ref().and_then(|c| c.temperature).unwrap_or(layer.config.planner_temperature),
        planner_max_tokens:  256,
    };

    let rx = run_agent(
        layer.planner.clone(),
        layer.registry.clone(),
        user_msg,
        inputs,
        opts,
    ).await;

    let stream = receiver_to_sse(rx, permit);

    HttpResponse::Ok()
        .content_type("text/event-stream; charset=utf-8")
        .insert_header((header::CACHE_CONTROL, "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(stream)
}

fn stage_inputs(
    input: &Option<AgentInput>,
    max_image_bytes: usize,
) -> Result<HashMap<String, Input>, String> {
    let mut m = HashMap::new();
    let Some(i) = input else { return Ok(m); };
    if let Some(img) = &i.image {
        let (mime, b64) = split_data_uri_or_bare(img, "image/jpeg");
        // Cap the decoded byte count; b64 is ~4/3 the binary size.
        let approx_bytes = b64.len() * 3 / 4;
        if approx_bytes > max_image_bytes {
            return Err(format!("image exceeds {} bytes (~{} actual)",
                               max_image_bytes, approx_bytes));
        }
        m.insert("input".to_string(), Input::Image { b64, mime });
    } else if let Some(aud) = &i.audio {
        let (mime, b64) = split_data_uri_or_bare(aud, "audio/wav");
        let approx_bytes = b64.len() * 3 / 4;
        if approx_bytes > max_image_bytes {
            return Err(format!("audio exceeds {} bytes (~{} actual)",
                               max_image_bytes, approx_bytes));
        }
        m.insert("input".to_string(), Input::Audio { b64, mime });
    }
    Ok(m)
}

fn split_data_uri_or_bare(s: &str, default_mime: &str) -> (String, String) {
    if let Some(comma) = s.find(',') {
        if let Some(meta) = s.get(..comma) {
            if let Some(rest) = meta.strip_prefix("data:") {
                let (mime, _) = rest.split_once(';').unwrap_or((rest, ""));
                return (mime.to_string(), s[comma + 1..].to_string());
            }
        }
    }
    (default_mime.to_string(), s.to_string())
}

fn receiver_to_sse(
    mut rx: mpsc::Receiver<AgentEvent>,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> impl futures_util::Stream<Item = Result<Bytes, actix_web::Error>> {
    async_stream::stream! {
        // Move permit into stream so it's held until termination.
        let _hold = permit;
        while let Some(ev) = rx.recv().await {
            yield Ok::<_, actix_web::Error>(Bytes::from(ev.to_sse_frame()));
        }
        yield Ok::<_, actix_web::Error>(Bytes::from_static(b"data: [DONE]\n\n"));
    }
}
