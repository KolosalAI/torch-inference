use anyhow::Result;
use base64::Engine as _;
use serde::Deserialize;
use std::time::Duration;

use crate::config::VisionBridgeConfig;

pub struct VisionBridge {
    cfg: VisionBridgeConfig,
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct ClassifyResp { results: Vec<Vec<ClassifyPred>> }

#[derive(Debug, Deserialize)]
struct ClassifyPred { label: String, confidence: f32 }

#[derive(Debug, Deserialize)]
struct DetectResp { detections: Vec<DetectBox> }

#[derive(Debug, Deserialize)]
struct DetectBox {
    label: String,
    #[serde(default)] confidence: f32,
    #[serde(default)] x1: f32, #[serde(default)] y1: f32,
    #[serde(default)] x2: f32, #[serde(default)] y2: f32,
}

impl VisionBridge {
    pub fn new(cfg: VisionBridgeConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(cfg.classify_timeout_ms + cfg.detect_timeout_ms))
            .build()
            .expect("build reqwest client");
        Self { cfg, http }
    }

    /// Produce a textual description of `image_bytes` by calling the main
    /// server's /classify/batch and /yolo/detect endpoints. Always returns
    /// a description; on errors, returns a stub and logs a warning.
    pub async fn describe(&self, image_bytes: &[u8]) -> String {
        let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

        let classify = self.classify(&b64).await;
        let detect = self.detect(&b64).await;

        match (classify, detect) {
            (Ok(cls), Ok(det)) => Self::compose(&cls, &det),
            (Ok(cls), Err(e)) => {
                tracing::warn!(error=%e, "vision_bridge: detect failed");
                Self::compose(&cls, &[])
            }
            (Err(e), Ok(det)) => {
                tracing::warn!(error=%e, "vision_bridge: classify failed");
                Self::compose("(classifier unavailable)", &det)
            }
            (Err(e1), Err(e2)) => {
                tracing::warn!(classify=%e1, detect=%e2, "vision_bridge: both failed");
                "[Image attached but vision tools unavailable.]".to_string()
            }
        }
    }

    async fn classify(&self, b64: &str) -> Result<String> {
        let url = format!("{}{}", self.cfg.main_server_base, self.cfg.classify_endpoint);
        let body = serde_json::json!({ "images": [b64], "top_k": 1 });
        let resp: ClassifyResp = self.http.post(&url)
            .json(&body)
            .timeout(Duration::from_millis(self.cfg.classify_timeout_ms))
            .send().await?
            .error_for_status()?
            .json().await?;
        let pred = resp.results.first()
            .and_then(|preds| preds.first())
            .ok_or_else(|| anyhow::anyhow!("classify: empty"))?;
        Ok(format!("'{}' ({:.2})", pred.label, pred.confidence))
    }

    async fn detect(&self, b64: &str) -> Result<Vec<DetectBox>> {
        let url = format!("{}{}", self.cfg.main_server_base, self.cfg.detect_endpoint);
        let body = serde_json::json!({ "image": b64 });
        let resp: DetectResp = self.http.post(&url)
            .json(&body)
            .timeout(Duration::from_millis(self.cfg.detect_timeout_ms))
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.detections)
    }

    fn compose(class_summary: &str, dets: &[DetectBox]) -> String {
        let mut s = format!("[Image attached. Classifier (top-1): {class_summary}.");
        if dets.is_empty() {
            s.push_str(" No YOLO detections.]");
        } else {
            s.push_str(" YOLO detections: ");
            for (i, d) in dets.iter().enumerate() {
                if i > 0 { s.push_str("; "); }
                s.push_str(&format!(
                    "{} at ({:.0},{:.0},{:.0},{:.0}) score {:.2}",
                    d.label, d.x1, d.y1, d.x2, d.y2, d.confidence
                ));
            }
            s.push_str(".]");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(base: &str) -> VisionBridgeConfig {
        VisionBridgeConfig {
            enabled: true,
            main_server_base: base.into(),
            classify_endpoint: "/classify/batch".into(),
            detect_endpoint: "/yolo/detect".into(),
            classify_timeout_ms: 500,
            detect_timeout_ms: 500,
        }
    }

    #[test]
    fn compose_with_both_results() {
        let det = DetectBox { label: "person".into(), confidence: 0.9, x1: 1.0, y1: 2.0, x2: 3.0, y2: 4.0 };
        let s = VisionBridge::compose("'cat' (0.81)", &[det]);
        assert!(s.contains("classifier") || s.contains("Classifier"));
        assert!(s.contains("'cat'"));
        assert!(s.contains("person at (1,2,3,4)"));
    }

    #[test]
    fn compose_with_no_detections() {
        let s = VisionBridge::compose("'cat' (0.81)", &[]);
        assert!(s.contains("No YOLO detections"));
    }

    #[tokio::test]
    async fn describe_returns_stub_when_server_down() {
        // No server running on this port.
        let vb = VisionBridge::new(cfg("http://127.0.0.1:1"));
        let out = vb.describe(b"\x89PNG\r\n\x1a\n").await;
        assert!(out.contains("vision tools unavailable"));
    }
}
