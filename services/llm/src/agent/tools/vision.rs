//! vision(image) → description
//!
//! Reuses the existing services/llm/src/vision_bridge.rs (which already
//! handles graceful fallback when classify/detect upstreams are unavailable).

use async_trait::async_trait;
use base64::Engine as _;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

use crate::agent::tool::{Tool, ToolError};
use crate::agent::tools::classify::encode_image;
use crate::vision_bridge::VisionBridge;

pub struct VisionTool {
    pub bridge: Arc<VisionBridge>,
}

impl VisionTool {
    pub fn new(bridge: Arc<VisionBridge>) -> Arc<Self> {
        Arc::new(Self { bridge })
    }
}

#[async_trait]
impl Tool for VisionTool {
    fn name(&self) -> &'static str { "vision" }

    async fn invoke(&self, args: Value, _deadline: Instant) -> Result<Value, ToolError> {
        let image = args.get("image")
            .ok_or_else(|| ToolError::BadArg("vision requires `image`".into()))?;
        let b64 = encode_image(image)?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(&b64)
            .map_err(|e| ToolError::BadArg(format!("invalid base64: {}", e)))?;
        let description = self.bridge.describe(&bytes).await;
        Ok(json!({ "description": description }))
    }
}
