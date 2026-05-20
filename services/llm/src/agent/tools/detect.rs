//! detect(image, model_version="v8", model_size="n") → count, labels, raw
//!
//! Posts multipart to /yolo/detect with `model_version` and `model_size` as
//! query params. This is the shape the real server expects (see
//! src/api/yolo.rs:84). The vision_bridge module has a long-standing bug
//! using JSON for this same endpoint; this tool does it correctly.

use async_trait::async_trait;
use base64::Engine as _;
use reqwest::multipart;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

use crate::agent::tool::{Tool, ToolError};
use crate::agent::tools::classify::encode_image;

pub struct DetectTool {
    pub client: reqwest::Client,
    pub url:    String,
}

impl DetectTool {
    pub fn new(client: reqwest::Client, base: &str, endpoint: &str) -> Arc<Self> {
        Arc::new(Self { client, url: format!("{}{}", base, endpoint) })
    }
}

#[async_trait]
impl Tool for DetectTool {
    fn name(&self) -> &'static str { "detect" }

    async fn invoke(&self, args: Value, deadline: Instant) -> Result<Value, ToolError> {
        let image = args.get("image")
            .ok_or_else(|| ToolError::BadArg("detect requires `image`".into()))?;
        let version = args.get("model_version").and_then(Value::as_str).unwrap_or("v8").to_string();
        let size    = args.get("model_size").and_then(Value::as_str).unwrap_or("n").to_string();

        let b64 = encode_image(image)?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(&b64)
            .map_err(|e| ToolError::BadArg(format!("invalid base64 image: {}", e)))?;

        let part = multipart::Part::bytes(bytes)
            .file_name("input.jpg")
            .mime_str("image/jpeg")
            .map_err(|e| ToolError::Upstream(format!("multipart mime: {}", e)))?;
        let form = multipart::Form::new().part("image", part);

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() { return Err(ToolError::Timeout(0)); }

        let url = format!("{}?model_version={}&model_size={}", self.url, version, size);
        let resp = self.client.post(&url)
            .multipart(form)
            .timeout(remaining)
            .send().await
            .map_err(|e| if e.is_timeout() {
                ToolError::Timeout(remaining.as_millis() as u64)
            } else {
                ToolError::Upstream(format!("{}: {}", url, e))
            })?;

        if !resp.status().is_success() {
            let s = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::Upstream(format!("detect returned {}: {}", s, body)));
        }
        let raw: Value = resp.json().await
            .map_err(|e| ToolError::Upstream(format!("decode detect: {}", e)))?;

        // Response shape (success): { success: true, results: { detections: [{label, ...}] } }
        let detections = raw.pointer("/results/detections")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        let labels: Vec<String> = detections.iter()
            .filter_map(|d| d.get("label").and_then(Value::as_str))
            .map(String::from).collect();

        Ok(json!({
            "count":  detections.len(),
            "labels": labels,
            "raw":    raw,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn tiny_jpeg_b64() -> String {
        // 1×1 JPEG, base64. Small valid header to satisfy mime guess.
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AKp//9k=".to_string()
    }

    #[tokio::test]
    async fn detect_flattens_count_and_labels() {
        let mut server = mockito::Server::new_async().await;
        let _m = server.mock("POST", "/yolo/detect")
            .match_query(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded("model_version".into(), "v8".into()),
                mockito::Matcher::UrlEncoded("model_size".into(), "n".into()),
            ]))
            .with_status(200)
            .with_body(r#"{"success":true,"results":{"detections":[{"label":"person"},{"label":"dog"}]}}"#)
            .create_async().await;
        let t = DetectTool::new(reqwest::Client::new(), &server.url(), "/yolo/detect");
        let out = t.invoke(json!({"image": tiny_jpeg_b64()}),
                            Instant::now() + Duration::from_secs(2)).await.unwrap();
        assert_eq!(out["count"], 2);
        assert_eq!(out["labels"], json!(["person","dog"]));
    }

    #[tokio::test]
    async fn detect_400_returns_upstream() {
        let mut server = mockito::Server::new_async().await;
        let _m = server.mock("POST", "/yolo/detect")
            .with_status(400).with_body("bad request")
            .create_async().await;
        let t = DetectTool::new(reqwest::Client::new(), &server.url(), "/yolo/detect");
        let err = t.invoke(json!({"image": tiny_jpeg_b64()}),
                            Instant::now() + Duration::from_secs(2)).await.unwrap_err();
        assert!(matches!(err, ToolError::Upstream(_)));
    }

    #[tokio::test]
    async fn detect_uses_overridden_model_args() {
        let mut server = mockito::Server::new_async().await;
        let m = server.mock("POST", "/yolo/detect")
            .match_query(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded("model_version".into(), "v11".into()),
                mockito::Matcher::UrlEncoded("model_size".into(), "s".into()),
            ]))
            .with_status(200)
            .with_body(r#"{"success":true,"results":{"detections":[]}}"#)
            .create_async().await;
        let t = DetectTool::new(reqwest::Client::new(), &server.url(), "/yolo/detect");
        let _ = t.invoke(json!({"image": tiny_jpeg_b64(),
                                "model_version": "v11", "model_size": "s"}),
                          Instant::now() + Duration::from_secs(2)).await.unwrap();
        m.assert_async().await;
    }
}
