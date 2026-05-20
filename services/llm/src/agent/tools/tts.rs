//! tts(text, voice="af_heart") → audio_url, duration_ms
//!
//! POSTs to /tts/stream and returns the response location header (or a
//! synthesized data URI if the upstream streams audio bytes).
//!
//! v1 contract: the main server's /tts/stream returns audio bytes directly
//! in the response. Since we don't want to ferry potentially-MB-sized audio
//! back through the SSE stream, we save the bytes to a temp file under the
//! server's /tmp dir and return a `file://...` URL. Future versions could
//! upload to a shared cache.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

use crate::agent::tool::{Tool, ToolError};

pub struct TtsTool {
    pub client: reqwest::Client,
    pub url:    String,
}

impl TtsTool {
    pub fn new(client: reqwest::Client, base: &str, endpoint: &str) -> Arc<Self> {
        Arc::new(Self { client, url: format!("{}{}", base, endpoint) })
    }
}

#[async_trait]
impl Tool for TtsTool {
    fn name(&self) -> &'static str { "tts" }

    async fn invoke(&self, args: Value, deadline: Instant) -> Result<Value, ToolError> {
        let text = args.get("text").and_then(Value::as_str)
            .ok_or_else(|| ToolError::BadArg("tts requires `text` (string)".into()))?;
        let voice = args.get("voice").and_then(Value::as_str).unwrap_or("af_heart");

        let body = json!({ "text": text, "voice": voice });
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { return Err(ToolError::Timeout(0)); }

        let started = Instant::now();
        let resp = self.client.post(&self.url)
            .json(&body)
            .timeout(remaining)
            .send().await
            .map_err(|e| if e.is_timeout() {
                ToolError::Timeout(remaining.as_millis() as u64)
            } else {
                ToolError::Upstream(format!("tts: {}", e))
            })?;

        if !resp.status().is_success() {
            let s = resp.status();
            let b = resp.text().await.unwrap_or_default();
            return Err(ToolError::Upstream(format!("tts returned {}: {}", s, b)));
        }

        let bytes = resp.bytes().await
            .map_err(|e| ToolError::Upstream(format!("tts body: {}", e)))?;

        let tmp = std::env::temp_dir().join(format!("agent_tts_{}.wav", ulid::Ulid::new()));
        std::fs::write(&tmp, &bytes)
            .map_err(|e| ToolError::Upstream(format!("tts write: {}", e)))?;

        Ok(json!({
            "audio_url":   format!("file://{}", tmp.display()),
            "duration_ms": started.elapsed().as_millis() as u64,
            "bytes":       bytes.len(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn tts_returns_audio_url_and_writes_file() {
        let mut server = mockito::Server::new_async().await;
        let _m = server.mock("POST", "/tts/stream")
            .with_status(200)
            .with_header("content-type", "audio/wav")
            .with_body(b"RIFF\0\0\0\0WAVE")
            .create_async().await;
        let t = TtsTool::new(reqwest::Client::new(), &server.url(), "/tts/stream");
        let out = t.invoke(json!({"text":"hi","voice":"af_heart"}),
                            Instant::now() + Duration::from_secs(2)).await.unwrap();
        let url = out["audio_url"].as_str().unwrap();
        assert!(url.starts_with("file://"));
        let path = url.trim_start_matches("file://");
        let written = std::fs::read(path).unwrap();
        assert_eq!(&written[..4], b"RIFF");
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn tts_missing_text_returns_badarg() {
        let t = TtsTool::new(reqwest::Client::new(), "http://x", "/tts/stream");
        let err = t.invoke(json!({}),
                            Instant::now() + Duration::from_secs(1)).await.unwrap_err();
        assert!(matches!(err, ToolError::BadArg(_)));
    }
}
