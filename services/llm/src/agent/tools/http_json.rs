//! Shared helper for tools that POST JSON to the main server and parse JSON
//! back. Times out at the tool's deadline.

use serde_json::Value;
use std::time::Instant;

use crate::agent::tool::ToolError;

pub async fn post_json(
    client: &reqwest::Client,
    url: &str,
    body: &Value,
    deadline: Instant,
) -> Result<Value, ToolError> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(ToolError::Timeout(0));
    }
    let resp = client.post(url)
        .json(body)
        .timeout(remaining)
        .send().await
        .map_err(|e| if e.is_timeout() {
            ToolError::Timeout(remaining.as_millis() as u64)
        } else {
            ToolError::Upstream(format!("{}: {}", url, e))
        })?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(ToolError::Upstream(format!("{} returned {}: {}", url, status, body)));
    }
    resp.json::<Value>().await
        .map_err(|e| ToolError::Upstream(format!("decode {}: {}", url, e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    #[tokio::test]
    async fn returns_decoded_json_on_200() {
        let mut server = mockito::Server::new_async().await;
        let m = server.mock("POST", "/x")
            .with_status(200).with_body(r#"{"ok":true}"#)
            .create_async().await;
        let client = reqwest::Client::new();
        let url = format!("{}/x", server.url());
        let out = post_json(&client, &url, &json!({"k":1}),
                            Instant::now() + Duration::from_secs(2)).await.unwrap();
        assert_eq!(out, json!({"ok": true}));
        m.assert_async().await;
    }

    #[tokio::test]
    async fn maps_non_2xx_to_upstream_error() {
        let mut server = mockito::Server::new_async().await;
        let _m = server.mock("POST", "/x").with_status(500).with_body("boom")
            .create_async().await;
        let client = reqwest::Client::new();
        let url = format!("{}/x", server.url());
        let err = post_json(&client, &url, &json!({}),
                            Instant::now() + Duration::from_secs(2)).await.unwrap_err();
        match err {
            ToolError::Upstream(s) => assert!(s.contains("500") && s.contains("boom")),
            other => panic!("expected Upstream, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn returns_timeout_when_deadline_passed() {
        let client = reqwest::Client::new();
        let err = post_json(&client, "http://127.0.0.1:1/x",
                            &json!({}), Instant::now()).await.unwrap_err();
        assert!(matches!(err, ToolError::Timeout(_)));
    }
}
