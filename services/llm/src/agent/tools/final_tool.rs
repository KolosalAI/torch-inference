//! `final(answer="…")` — terminates the run. The tool returns the answer in
//! its value; the executor's main loop detects `step.tool == "final"` and
//! emits a Final SSE event instead of continuing.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::time::Instant;

use crate::agent::tool::{Tool, ToolError};

pub struct FinalTool;

#[async_trait]
impl Tool for FinalTool {
    fn name(&self) -> &'static str { "final" }

    async fn invoke(&self, args: Value, _deadline: Instant) -> Result<Value, ToolError> {
        let answer = args.get("answer")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::BadArg("final requires `answer` (string)".into()))?;
        Ok(json!({ "answer": answer }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn final_returns_answer() {
        let out = FinalTool.invoke(json!({"answer": "42"}),
                                    Instant::now() + Duration::from_secs(1)).await.unwrap();
        assert_eq!(out["answer"], "42");
    }

    #[tokio::test]
    async fn final_rejects_missing_answer() {
        let err = FinalTool.invoke(json!({}),
                                    Instant::now() + Duration::from_secs(1)).await.unwrap_err();
        assert!(matches!(err, ToolError::BadArg(_)));
    }
}
