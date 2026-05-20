//! stt(audio) → transcript
//!
//! POSTs audio bytes (multipart `file` field) to /stt/transcribe.

use async_trait::async_trait;
use base64::Engine as _;
use reqwest::multipart;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

use crate::agent::tool::{Tool, ToolError};
use crate::agent::tools::classify::encode_image; // same data-URI extraction

pub struct SttTool {
    pub client: reqwest::Client,
    pub url:    String,
}

impl SttTool {
    pub fn new(client: reqwest::Client, base: &str, endpoint: &str) -> Arc<Self> {
        Arc::new(Self { client, url: format!("{}{}", base, endpoint) })
    }
}

#[async_trait]
impl Tool for SttTool {
    fn name(&self) -> &'static str { "stt" }

    async fn invoke(&self, args: Value, deadline: Instant) -> Result<Value, ToolError> {
        let audio = args.get("audio")
            .ok_or_else(|| ToolError::BadArg("stt requires `audio`".into()))?;
        let b64 = encode_image(audio)?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(&b64)
            .map_err(|e| ToolError::BadArg(format!("invalid base64 audio: {}", e)))?;

        let part = multipart::Part::bytes(bytes)
            .file_name("input.wav")
            .mime_str("audio/wav")
            .map_err(|e| ToolError::Upstream(format!("stt mime: {}", e)))?;
        let form = multipart::Form::new().part("file", part);

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { return Err(ToolError::Timeout(0)); }

        let resp = self.client.post(&self.url)
            .multipart(form)
            .timeout(remaining)
            .send().await
            .map_err(|e| if e.is_timeout() {
                ToolError::Timeout(remaining.as_millis() as u64)
            } else {
                ToolError::Upstream(format!("stt: {}", e))
            })?;

        if !resp.status().is_success() {
            let s = resp.status();
            let b = resp.text().await.unwrap_or_default();
            return Err(ToolError::Upstream(format!("stt returned {}: {}", s, b)));
        }
        let raw: Value = resp.json().await
            .map_err(|e| ToolError::Upstream(format!("decode stt: {}", e)))?;
        let transcript = raw.get("transcript").or_else(|| raw.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        Ok(json!({ "transcript": transcript }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn stt_returns_transcript() {
        let mut server = mockito::Server::new_async().await;
        let _m = server.mock("POST", "/stt/transcribe")
            .with_status(200)
            .with_body(r#"{"transcript":"hello world"}"#)
            .create_async().await;
        let t = SttTool::new(reqwest::Client::new(), &server.url(), "/stt/transcribe");
        let out = t.invoke(json!({"audio":"UklGRiIAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="}),
                            Instant::now() + Duration::from_secs(2)).await.unwrap();
        assert_eq!(out["transcript"], "hello world");
    }
}
