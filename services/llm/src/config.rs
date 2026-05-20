use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    /// HTTP port this service listens on
    #[serde(default = "default_port")]
    pub port: u16,

    /// HRM-Text engine configuration (required at startup).
    #[serde(default)]
    pub hrm: Option<HrmConfig>,

    /// Vision bridge configuration for image description via classify+detect.
    #[serde(default)]
    pub vision_bridge: Option<VisionBridgeConfig>,

    #[serde(default)]
    pub agent: Option<AgentConfig>,
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

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "default_agent_enabled")]
    pub enabled: bool,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_max_run_ms")]
    pub max_run_ms: u64,
    #[serde(default = "default_per_tool_ms")]
    pub per_tool_ms: u64,
    #[serde(default = "default_max_concurrent_runs")]
    pub max_concurrent_runs: usize,
    #[serde(default = "default_reflect_max_tokens")]
    pub reflect_max_tokens: u32,
    #[serde(default = "default_planner_temperature")]
    pub planner_temperature: f32,
    #[serde(default)]
    pub http_fetch: Option<HttpFetchConfig>,
    #[serde(default)]
    pub tools: Option<AgentToolsConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HttpFetchConfig {
    #[serde(default = "default_agent_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub allowlist: Vec<String>,
    #[serde(default = "default_http_fetch_max_bytes")]
    pub max_bytes: usize,
    #[serde(default)]
    pub follow_redirects: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AgentToolsConfig {
    #[serde(default = "default_main_server_base")]
    pub main_server_base: String,
    #[serde(default = "default_classify_endpoint")]
    pub classify_endpoint: String,
    #[serde(default = "default_detect_endpoint")]
    pub detect_endpoint: String,
    #[serde(default = "default_tts_endpoint")]
    pub tts_endpoint: String,
    #[serde(default = "default_stt_endpoint")]
    pub stt_endpoint: String,
}

impl Default for HttpFetchConfig {
    fn default() -> Self {
        Self {
            enabled: default_agent_enabled(),
            allowlist: Vec::new(),
            max_bytes: default_http_fetch_max_bytes(),
            follow_redirects: false,
        }
    }
}

fn default_agent_enabled() -> bool { true }
fn default_max_steps() -> usize { 8 }
fn default_max_run_ms() -> u64 { 60_000 }
fn default_per_tool_ms() -> u64 { 5_000 }
fn default_max_concurrent_runs() -> usize { 4 }
fn default_reflect_max_tokens() -> u32 { 128 }
fn default_planner_temperature() -> f32 { 0.0 }
fn default_http_fetch_max_bytes() -> usize { 65_536 }
fn default_main_server_base() -> String { "http://127.0.0.1:8000".to_string() }
fn default_classify_endpoint() -> String { "/classify/batch".to_string() }
fn default_detect_endpoint() -> String { "/yolo/detect".to_string() }
fn default_tts_endpoint() -> String { "/tts/stream".to_string() }
fn default_stt_endpoint() -> String { "/stt/transcribe".to_string() }

fn default_port() -> u16 { 8001 }
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
                hrm: None,
                vision_bridge: None,
                agent: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hrm_section_with_defaults() {
        let toml_text = r#"
port = 8001

[hrm]
model_dir = "models/hrm-text-1b"
"#;
        let cfg: LlmConfig = toml::from_str(toml_text).unwrap();
        let hrm = cfg.hrm.expect("hrm section present");
        assert_eq!(hrm.model_dir, "models/hrm-text-1b");
        assert_eq!(hrm.ep_preference, "auto");
        assert!(hrm.use_quantized.is_none() || hrm.use_quantized == Some(false));
    }

    #[test]
    fn parses_agent_section_with_defaults() {
        let toml_text = r#"
port = 8001
[agent]
enabled = true
"#;
        let cfg: LlmConfig = toml::from_str(toml_text).unwrap();
        let agent = cfg.agent.expect("agent section");
        assert!(agent.enabled);
        assert_eq!(agent.max_steps, 8);
        assert_eq!(agent.max_run_ms, 60_000);
        let hf = agent.http_fetch.unwrap_or_default();
        assert!(hf.allowlist.is_empty());
    }
}
