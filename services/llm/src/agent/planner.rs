//! Planner abstraction so the executor can be tested without the real ONNX model.
//!
//! The production impl wraps `HrmEngine::infer_text` via spawn_blocking, draining
//! the streaming channel into a single String. Tests use a canned-output stub.

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

use crate::hrm_engine::HrmEngine;

#[async_trait]
pub trait Planner: Send + Sync {
    /// Run inference on `prompt` and return the full generated text.
    async fn propose(&self, prompt: String, max_tokens: u32, temperature: f32) -> Result<String>;
}

pub struct HrmPlanner {
    engine: Arc<HrmEngine>,
}

impl HrmPlanner {
    pub fn new(engine: Arc<HrmEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl Planner for HrmPlanner {
    async fn propose(&self, prompt: String, max_tokens: u32, temperature: f32) -> Result<String> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(256);
        let engine = self.engine.clone();
        let handle = tokio::task::spawn_blocking(move || {
            engine.infer_text(prompt, max_tokens, temperature, tx)
        });
        let mut buf = String::new();
        while let Some(s) = rx.recv().await { buf.push_str(&s); }
        handle.await
            .map_err(|e| anyhow::anyhow!("planner join: {}", e))?
            .map_err(|e| anyhow::anyhow!("planner inference: {}", e))?;
        Ok(buf)
    }
}
