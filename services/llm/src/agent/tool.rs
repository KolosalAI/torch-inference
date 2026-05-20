//! Tool trait + registry. Each tool owns its argument validation and
//! returns a `serde_json::Value` whose top-level fields become available
//! to subsequent steps as `{{stepN.field}}` refs.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("argument error: {0}")]
    BadArg(String),
    #[error("denied: {0}")]
    Denied(String),
    #[error("timeout after {0} ms")]
    Timeout(u64),
    #[error("upstream error: {0}")]
    Upstream(String),
    #[error("unknown tool: {0}")]
    Unknown(String),
}

#[async_trait]
pub trait Tool: Send + Sync {
    /// Lowercase name matching the DSL grammar.
    fn name(&self) -> &'static str;

    /// `args` is a JSON object with already-resolved values (no refs).
    /// `deadline` is the wall-clock cutoff for this dispatch.
    async fn invoke(&self, args: Value, deadline: Instant) -> Result<Value, ToolError>;
}

pub struct ToolRegistry {
    tools: HashMap<&'static str, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn insert(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name(), tool);
    }

    pub async fn dispatch(
        &self,
        name: &str,
        args: Value,
        deadline: Instant,
    ) -> Result<Value, ToolError> {
        let tool = self.tools.get(name)
            .ok_or_else(|| ToolError::Unknown(name.to_string()))?;
        tool.invoke(args, deadline).await
    }
}

impl Default for ToolRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    struct EchoTool;
    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &'static str { "echo" }
        async fn invoke(&self, args: Value, _d: Instant) -> Result<Value, ToolError> {
            Ok(json!({"echoed": args}))
        }
    }

    #[tokio::test]
    async fn registry_dispatches_to_registered_tool() {
        let mut reg = ToolRegistry::new();
        reg.insert(Arc::new(EchoTool));
        let out = reg.dispatch("echo", json!({"x": 1}),
                                Instant::now() + Duration::from_secs(1)).await.unwrap();
        assert_eq!(out, json!({"echoed": {"x": 1}}));
    }

    #[tokio::test]
    async fn registry_unknown_returns_unknown_error() {
        let reg = ToolRegistry::new();
        let err = reg.dispatch("missing", json!({}),
                                Instant::now() + Duration::from_secs(1)).await.unwrap_err();
        assert!(matches!(err, ToolError::Unknown(_)));
    }
}
