use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    /// HTTP port this service listens on
    #[serde(default = "default_port")]
    pub port: u16,

    /// Path to the GGUF model file
    pub model_path: String,

    /// Optional path to the multimodal projection file (.mmproj.gguf).
    /// Omit to disable image input.
    #[serde(default)]
    pub mmproj_path: Option<String>,

    /// KV-cache context window size in tokens
    #[serde(default = "default_ctx_size")]
    pub ctx_size: u32,

    /// CPU thread count for generation
    #[serde(default = "default_n_threads")]
    pub n_threads: i32,

    /// Number of model layers to offload to GPU (0 = CPU-only)
    #[serde(default)]
    pub n_gpu_layers: i32,
}

fn default_port() -> u16 { 8001 }
fn default_ctx_size() -> u32 { 4096 }
fn default_n_threads() -> i32 { 4 }

impl LlmConfig {
    /// Load from `config.toml` in the current working directory, or use defaults.
    pub fn load() -> Result<Self> {
        let config_path = std::path::PathBuf::from("config.toml");
        if config_path.exists() {
            let text = std::fs::read_to_string(&config_path)
                .context("read config.toml")?;
            toml::from_str(&text).context("parse config.toml")
        } else {
            tracing::warn!("config.toml not found, using defaults");
            Ok(Self {
                port: 8001,
                model_path: "models/minicpm-v-2_6-q2_k.gguf".into(),
                mmproj_path: Some("models/minicpm-v-2_6-mmproj-f16.gguf".into()),
                ctx_size: 4096,
                n_threads: 4,
                n_gpu_layers: 0,
            })
        }
    }

    /// Returns mmproj_path only if it's non-empty and the file exists on disk.
    pub fn effective_mmproj(&self) -> Option<&str> {
        self.mmproj_path
            .as_deref()
            .filter(|p| !p.is_empty())
            .filter(|p| std::path::Path::new(p).exists())
    }
}
