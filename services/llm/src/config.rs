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

    /// HRM-Text engine configuration. When present, the service runs the
    /// new HrmEngine; otherwise it falls back to the legacy LlamaEngine.
    /// Both engines share `port`; the HRM section provides the rest.
    #[serde(default)]
    pub hrm: Option<HrmConfig>,

    /// Vision bridge configuration for image description via classify+detect.
    #[serde(default)]
    pub vision_bridge: Option<VisionBridgeConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HrmConfig {
    /// Directory containing model.onnx, tokenizer.json, config.json.
    pub model_dir: String,

    /// Execution provider preference. "auto" picks CoreML on macOS, CUDA on
    /// Linux when n_gpu_layers > 0, else CPU. Other values: "cpu", "coreml",
    /// "cuda".
    #[serde(default = "default_ep_preference")]
    pub ep_preference: String,

    /// Use the int8 quantized variant (model.int8.onnx) if true. Defaults to
    /// false (fp16 model.onnx).
    #[serde(default)]
    pub use_quantized: Option<bool>,

    /// Number of CPU threads for ort sessions. Falls back to LlmConfig.n_threads
    /// if None.
    #[serde(default)]
    pub n_threads: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionBridgeConfig {
    #[serde(default = "default_vb_enabled")]
    pub enabled: bool,
    #[serde(default = "default_vb_base")]
    pub main_server_base: String,
    #[serde(default = "default_vb_classify")]
    pub classify_endpoint: String,
    #[serde(default = "default_vb_detect")]
    pub detect_endpoint: String,
    #[serde(default = "default_vb_classify_timeout")]
    pub classify_timeout_ms: u64,
    #[serde(default = "default_vb_detect_timeout")]
    pub detect_timeout_ms: u64,
}

fn default_port() -> u16 { 8001 }
fn default_ctx_size() -> u32 { 4096 }
fn default_n_threads() -> i32 { 4 }
fn default_ep_preference() -> String { "auto".to_string() }

fn default_vb_enabled() -> bool { true }
fn default_vb_base() -> String { "http://127.0.0.1:8000".to_string() }
fn default_vb_classify() -> String { "/classify/batch".to_string() }
fn default_vb_detect() -> String { "/yolo/detect".to_string() }
fn default_vb_classify_timeout() -> u64 { 1500 }
fn default_vb_detect_timeout() -> u64 { 2500 }

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
                model_path: "models/llava-v1.6-mistral-7b.IQ1_S.gguf".into(),
                mmproj_path: Some("models/llava-v1.6-mistral-7b-mmproj-f16.gguf".into()),
                ctx_size: 4096,
                n_threads: 4,
                n_gpu_layers: 0,
                hrm: None,
                vision_bridge: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hrm_section_with_defaults() {
        let toml_text = r#"
port = 8001
model_path = "models/llava-v1.6-mistral-7b.IQ1_S.gguf"

[hrm]
model_dir = "models/hrm-text-1b"
"#;
        let cfg: LlmConfig = toml::from_str(toml_text).unwrap();
        let hrm = cfg.hrm.expect("hrm section present");
        assert_eq!(hrm.model_dir, "models/hrm-text-1b");
        assert_eq!(hrm.ep_preference, "auto");
        assert!(hrm.use_quantized.is_none() || hrm.use_quantized == Some(false));
    }
}
