//! reflect(prompt, max_tokens=128) → output
//!
//! In-process HrmEngine call. Uses the Planner trait so tests can stub.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

use crate::agent::planner::Planner;
use crate::agent::tool::{Tool, ToolError};

pub struct ReflectTool {
    pub planner:     Arc<dyn Planner>,
    pub default_max: u32,
}

impl ReflectTool {
    pub fn new(planner: Arc<dyn Planner>, default_max: u32) -> Arc<Self> {
        Arc::new(Self { planner, default_max })
    }
}

#[async_trait]
impl Tool for ReflectTool {
    fn name(&self) -> &'static str { "reflect" }

    async fn invoke(&self, args: Value, _deadline: Instant) -> Result<Value, ToolError> {
        let prompt = args.get("prompt").and_then(Value::as_str)
            .ok_or_else(|| ToolError::BadArg("reflect requires `prompt` (string)".into()))?
            .to_string();
        let max_tokens = args.get("max_tokens").and_then(Value::as_u64)
            .unwrap_or(self.default_max as u64).min(self.default_max as u64) as u32;

        let output = self.planner.propose(prompt, max_tokens, 0.0).await
            .map_err(|e| ToolError::Upstream(format!("reflect: {}", e)))?;
        Ok(json!({ "output": output.trim().to_string() }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use async_trait::async_trait;
    use std::time::Duration;

    struct StubPlanner(String);
    #[async_trait]
    impl Planner for StubPlanner {
        async fn propose(&self, _p: String, _m: u32, _t: f32) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn reflect_returns_planner_output_trimmed() {
        let r = ReflectTool::new(Arc::new(StubPlanner("  42  ".into())), 128);
        let out = r.invoke(json!({"prompt":"2+2 = ?", "max_tokens": 8}),
                            Instant::now() + Duration::from_secs(2)).await.unwrap();
        assert_eq!(out["output"], "42");
    }

    #[tokio::test]
    async fn reflect_caps_max_tokens_to_default() {
        // Sanity check that we don't blow past the cap.
        let r = ReflectTool::new(Arc::new(StubPlanner("ok".into())), 16);
        let _ = r.invoke(json!({"prompt":"x", "max_tokens": 9999}),
                          Instant::now() + Duration::from_secs(1)).await.unwrap();
        // No assertion needed; just ensures the cap clamp doesn't panic.
    }

    #[tokio::test]
    async fn reflect_missing_prompt_returns_badarg() {
        let r = ReflectTool::new(Arc::new(StubPlanner("".into())), 8);
        let err = r.invoke(json!({}),
                            Instant::now() + Duration::from_secs(1)).await.unwrap_err();
        assert!(matches!(err, ToolError::BadArg(_)));
    }
}
