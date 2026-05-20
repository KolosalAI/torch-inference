//! classify(image, top_k) → label, confidence, all
//!
//! POSTs to the main server's /classify/batch as `{"images":[b64], "top_k":N}`.
//! Returns top-1 promoted to `label`/`confidence` plus the full list as `all`.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

use crate::agent::tool::{Tool, ToolError};
use crate::agent::tools::http_json::post_json;

pub struct ClassifyTool {
    pub client: reqwest::Client,
    pub url:    String,
}

impl ClassifyTool {
    pub fn new(client: reqwest::Client, base: &str, endpoint: &str) -> Arc<Self> {
        Arc::new(Self { client, url: format!("{}{}", base, endpoint) })
    }
}

#[async_trait]
impl Tool for ClassifyTool {
    fn name(&self) -> &'static str { "classify" }

    async fn invoke(&self, args: Value, deadline: Instant) -> Result<Value, ToolError> {
        let image = args.get("image")
            .ok_or_else(|| ToolError::BadArg("classify requires `image`".into()))?;
        let top_k = args.get("top_k").and_then(Value::as_i64).unwrap_or(1).max(1) as usize;
        let b64 = encode_image(image)?;

        let body = json!({ "images": [b64], "top_k": top_k });
        let resp = post_json(&self.client, &self.url, &body, deadline).await?;

        // Response shape: { "results": [[{label, confidence}, ...]] }
        let preds = resp.get("results")
            .and_then(|r| r.as_array())
            .and_then(|a| a.first())
            .and_then(|p| p.as_array())
            .ok_or_else(|| ToolError::Upstream("classify: missing results".into()))?;

        let top = preds.first()
            .ok_or_else(|| ToolError::Upstream("classify: empty predictions".into()))?;
        let label = top.get("label").and_then(Value::as_str)
            .ok_or_else(|| ToolError::Upstream("classify: missing label".into()))?
            .to_string();
        let confidence = top.get("confidence").and_then(Value::as_f64).unwrap_or(0.0);

        Ok(json!({
            "label": label,
            "confidence": confidence,
            "all": preds,
        }))
    }
}

/// Accept either a `data:image/...;base64,...` URI, a raw base64 string,
/// or the literal `"input"` (which will have already been substituted by
/// the executor with the request's staged image bytes as base64).
pub(crate) fn encode_image(v: &Value) -> Result<String, ToolError> {
    let s = v.as_str()
        .ok_or_else(|| ToolError::BadArg("image must be a string".into()))?;
    if let Some(idx) = s.find("base64,") {
        return Ok(s[idx + 7 ..].to_string());
    }
    // Otherwise assume already base64.
    Ok(s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn classify_extracts_top1_and_all() {
        let mut server = mockito::Server::new_async().await;
        let _m = server.mock("POST", "/classify/batch")
            .with_status(200)
            .with_body(r#"{"results":[[{"label":"cat","confidence":0.81},{"label":"dog","confidence":0.10}]]}"#)
            .create_async().await;
        let t = ClassifyTool::new(reqwest::Client::new(), &server.url(), "/classify/batch");
        let out = t.invoke(json!({"image":"FAKE","top_k":2}),
                            Instant::now() + Duration::from_secs(2)).await.unwrap();
        assert_eq!(out["label"], "cat");
        assert_eq!(out["confidence"], 0.81);
        assert_eq!(out["all"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn classify_500_returns_upstream() {
        let mut server = mockito::Server::new_async().await;
        let _m = server.mock("POST", "/classify/batch")
            .with_status(500).create_async().await;
        let t = ClassifyTool::new(reqwest::Client::new(), &server.url(), "/classify/batch");
        let err = t.invoke(json!({"image":"X"}),
                            Instant::now() + Duration::from_secs(2)).await.unwrap_err();
        assert!(matches!(err, ToolError::Upstream(_)));
    }

    #[tokio::test]
    async fn classify_missing_image_returns_badarg() {
        let t = ClassifyTool::new(reqwest::Client::new(), "http://x", "/classify/batch");
        let err = t.invoke(json!({}),
                            Instant::now() + Duration::from_secs(1)).await.unwrap_err();
        assert!(matches!(err, ToolError::BadArg(_)));
    }
}
