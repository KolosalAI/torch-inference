use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn default_json_body_limit_mb() -> usize { 50 }
fn default_server_host() -> String { "0.0.0.0".to_string() }
fn default_server_port() -> u16 { 8000 }
fn default_server_workers() -> usize { num_cpus::get() }
fn default_log_level() -> String { "info".to_string() }
fn default_keep_alive_secs() -> u64 { 75 }
fn default_request_timeout_secs() -> u64 { 5 }
fn default_disconnect_timeout_secs() -> u64 { 1 }
fn default_shutdown_timeout_secs() -> u64 { 30 }
fn default_proxy_timeout_secs() -> u64 { 300 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub device: DeviceConfig,
    #[serde(default)]
    pub batch: BatchConfig,
    #[serde(default)]
    pub performance: PerformanceConfig,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub microservices: MicroservicesConfig,
    #[serde(default)]
    pub guard: GuardConfig,
    #[serde(default)]
    pub sanitizer: SanitizerConfig,
    #[serde(default)]
    pub postprocess: PostprocessConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizerConfig {
    pub max_text_length: usize,
    pub sanitize_text: bool,
    pub sanitize_image_dimensions: bool,
    pub max_image_width: u32,
    pub max_image_height: u32,
    pub round_probabilities: bool,
    pub probability_decimals: u32,
    pub remove_null_values: bool,
}

impl Default for SanitizerConfig {
    fn default() -> Self {
        Self {
            max_text_length: 10000,
            sanitize_text: true,
            sanitize_image_dimensions: true,
            max_image_width: 4096,
            max_image_height: 4096,
            round_probabilities: true,
            probability_decimals: 4,
            remove_null_values: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_server_host")]
    pub host: String,
    #[serde(default = "default_server_port")]
    pub port: u16,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default = "default_server_workers")]
    pub workers: usize,
    /// TCP keep-alive for idle connections (seconds). Default: 75.
    #[serde(default = "default_keep_alive_secs")]
    pub keep_alive_secs: u64,
    /// Max time to wait for client to send request headers (seconds). Default: 5.
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Time to wait after the last byte before closing a keep-alive connection (seconds). Default: 1.
    #[serde(default = "default_disconnect_timeout_secs")]
    pub disconnect_timeout_secs: u64,
    /// Graceful shutdown drain window (seconds). Default: 30.
    #[serde(default = "default_shutdown_timeout_secs")]
    pub shutdown_timeout_secs: u64,
    /// Maximum JSON request body size (MiB). Default: 50.
    #[serde(default = "default_json_body_limit_mb")]
    pub json_body_limit_mb: usize,
    /// Timeout for outbound proxy requests to microservices (seconds). Default: 300.
    #[serde(default = "default_proxy_timeout_secs")]
    pub proxy_timeout_secs: u64,
    /// Per-request cap on multipart audio uploads (MiB). Default: 100.
    /// Protects `/stt/transcribe` from OOM via oversized files.
    #[serde(default = "default_multipart_audio_limit_mb")]
    pub multipart_audio_limit_mb: usize,
    /// Per-request cap on multipart image uploads (MiB). Default: 10.
    /// Protects `/yolo/detect` from OOM via oversized files.
    #[serde(default = "default_multipart_image_limit_mb")]
    pub multipart_image_limit_mb: usize,
    /// Per-image cap on base64 strings inside JSON requests (MiB). Default: 5.
    /// Protects `/classify/batch` and `/classify/stream` from OOM via large
    /// individual items even when the batch as a whole fits within
    /// `json_body_limit_mb`.
    #[serde(default = "default_classify_image_limit_mb")]
    pub classify_image_limit_mb: usize,
    /// Maximum decoded audio duration (seconds). Default: 1800 (30 min).
    /// Applied during WAV validation and Symphonia decode to bound memory.
    #[serde(default = "default_audio_max_duration_secs")]
    pub audio_max_duration_secs: u32,
    /// Per-chunk timeout for `/tts/stream` (seconds). Default: 30.
    /// A stuck engine no longer holds the connection open indefinitely.
    #[serde(default = "default_tts_chunk_timeout_secs")]
    pub tts_chunk_timeout_secs: u64,
    /// Per-image timeout for `/classify/stream` (seconds). Default: 30.
    /// A hung inference task no longer blocks subsequent images.
    #[serde(default = "default_classify_item_timeout_secs")]
    pub classify_item_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_server_host(),
            port: default_server_port(),
            log_level: default_log_level(),
            workers: default_server_workers(),
            keep_alive_secs: default_keep_alive_secs(),
            request_timeout_secs: default_request_timeout_secs(),
            disconnect_timeout_secs: default_disconnect_timeout_secs(),
            shutdown_timeout_secs: default_shutdown_timeout_secs(),
            json_body_limit_mb: default_json_body_limit_mb(),
            proxy_timeout_secs: default_proxy_timeout_secs(),
            multipart_audio_limit_mb: default_multipart_audio_limit_mb(),
            multipart_image_limit_mb: default_multipart_image_limit_mb(),
            classify_image_limit_mb: default_classify_image_limit_mb(),
            audio_max_duration_secs: default_audio_max_duration_secs(),
            tts_chunk_timeout_secs: default_tts_chunk_timeout_secs(),
            classify_item_timeout_secs: default_classify_item_timeout_secs(),
        }
    }
}

fn default_microservice_host() -> String {
    "127.0.0.1".to_string()
}
fn default_stt_port() -> u16 {
    8002
}
fn default_llm_port() -> u16 {
    8001
}
// Multipart / per-item limits introduced by the upstream hardening batches.
fn default_multipart_audio_limit_mb() -> usize { 100 }
fn default_multipart_image_limit_mb() -> usize { 10 }
fn default_classify_image_limit_mb() -> usize { 5 }
fn default_audio_max_duration_secs() -> u32 { 1800 }
fn default_tts_chunk_timeout_secs() -> u64 { 30 }
fn default_classify_item_timeout_secs() -> u64 { 30 }

/// Microservice host/port configuration. The main server spawns these as child
/// processes and proxies requests to them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroservicesConfig {
    /// Host for the STT microservice. Default: "127.0.0.1".
    #[serde(default = "default_microservice_host")]
    pub stt_host: String,
    /// Port for the STT microservice. Default: 8002.
    #[serde(default = "default_stt_port")]
    pub stt_port: u16,
    /// Host for the LLM microservice. Default: "127.0.0.1".
    #[serde(default = "default_microservice_host")]
    pub llm_host: String,
    /// Port for the LLM microservice. Default: 8001.
    #[serde(default = "default_llm_port")]
    pub llm_port: u16,
}

impl Default for MicroservicesConfig {
    fn default() -> Self {
        Self {
            stt_host: default_microservice_host(),
            stt_port: default_stt_port(),
            llm_host: default_microservice_host(),
            llm_port: default_llm_port(),
        }
    }
}

impl MicroservicesConfig {
    /// Base URL for the STT microservice proxy.
    pub fn stt_base_url(&self) -> String {
        format!("http://{}:{}", self.stt_host, self.stt_port)
    }

    /// Base URL for the LLM microservice proxy.
    pub fn llm_base_url(&self) -> String {
        format!("http://{}:{}", self.llm_host, self.llm_port)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceConfig {
    #[serde(default)]
    pub device_type: String,
    #[serde(default)]
    pub device_id: usize,
    #[serde(default)]
    pub device_ids: Option<Vec<usize>>,
    #[serde(default)]
    pub use_fp16: bool,
    #[serde(default)]
    pub use_tensorrt: bool,
    #[serde(default)]
    pub use_torch_compile: bool,

    // Metal-specific optimizations (macOS)
    #[serde(default)]
    pub metal_use_mlx: bool,
    #[serde(default)]
    pub metal_cache_shaders: bool,
    #[serde(default)]
    pub metal_optimize_for_apple_silicon: bool,

    // JIT Compilation settings
    #[serde(default)]
    pub enable_jit: bool,
    #[serde(default)]
    pub enable_jit_profiling: bool,
    #[serde(default)]
    pub enable_jit_executor: bool,
    #[serde(default)]
    pub enable_jit_fusion: bool,

    // PyTorch/LibTorch optimizations
    #[serde(default)]
    pub num_threads: usize,
    #[serde(default)]
    pub num_interop_threads: usize,
    #[serde(default)]
    pub cudnn_benchmark: bool,
    #[serde(default)]
    pub enable_autocast: bool,
    #[serde(default)]
    pub torch_warmup_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchConfig {
    #[serde(default)]
    pub batch_size: usize,
    #[serde(default)]
    pub max_batch_size: usize,
    #[serde(default)]
    pub enable_dynamic_batching: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceConfig {
    #[serde(default)]
    pub warmup_iterations: usize,
    #[serde(default)]
    pub enable_caching: bool,
    #[serde(default)]
    pub enable_profiling: bool,
    #[serde(default)]
    pub cache_size_mb: usize,
    #[serde(default)]
    pub enable_cuda_graphs: bool,
    #[serde(default)]
    pub enable_model_quantization: bool,
    #[serde(default)]
    pub quantization_bits: u8,
    #[serde(default)]
    pub enable_tensor_pooling: bool,
    #[serde(default)]
    pub max_pooled_tensors: usize,
    #[serde(default)]
    pub enable_async_model_loading: bool,
    #[serde(default)]
    pub preload_models_on_startup: bool,
    #[serde(default)]
    pub enable_result_compression: bool,
    #[serde(default)]
    pub compression_level: u32,
    #[serde(default)]
    pub enable_request_batching: bool,
    #[serde(default)]
    pub adaptive_batch_timeout: bool,
    #[serde(default)]
    pub min_batch_size: usize,
    #[serde(default)]
    pub enable_inflight_batching: bool,
    #[serde(default)]
    pub max_inflight_batches: usize,
    #[serde(default)]
    pub enable_worker_pool: bool,
    #[serde(default)]
    pub min_workers: usize,
    #[serde(default)]
    pub max_workers: usize,
    #[serde(default)]
    pub enable_auto_scaling: bool,
    #[serde(default)]
    pub enable_zero_scaling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub jwt_secret: String,
    #[serde(default)]
    pub jwt_algorithm: String,
    #[serde(default)]
    pub access_token_expire_minutes: u32,
    #[serde(default)]
    pub refresh_token_expire_days: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelsConfig {
    #[serde(default)]
    pub auto_load: Vec<String>,
    /// Root directory for all model files. Default: "models".
    #[serde(default)]
    pub cache_dir: PathBuf,
    #[serde(default)]
    pub max_loaded_models: usize,
    /// Path to the EfficientNet-Lite4 ONNX classification model.
    #[serde(default)]
    pub classify_model: PathBuf,
    /// Path to the ImageNet-1000 labels file for the classifier.
    #[serde(default)]
    pub classify_labels: PathBuf,
    /// Directory containing audio models (Whisper, etc.). Default: "models/audio".
    #[serde(default)]
    pub audio_model_dir: PathBuf,
    /// Default YOLO confidence threshold (0–1). Can be overridden per-request. Default: 0.25.
    #[serde(default)]
    pub yolo_conf_threshold: f32,
    /// Default YOLO IoU threshold for NMS (0–1). Can be overridden per-request. Default: 0.45.
    #[serde(default)]
    pub yolo_iou_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuardConfig {
    #[serde(default)]
    pub enable_guards: bool,
    #[serde(default)]
    pub max_memory_mb: usize,
    #[serde(default)]
    pub max_requests_per_second: usize,
    #[serde(default)]
    pub max_queue_depth: usize,
    #[serde(default)]
    pub min_cache_hit_rate: f64,
    #[serde(default)]
    pub max_error_rate: f64,
    #[serde(default)]
    pub enable_circuit_breaker: bool,
    #[serde(default)]
    pub enable_auto_mitigation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPostprocessConfig {
    #[serde(default = "AudioPostprocessConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default = "AudioPostprocessConfig::default_target_peak")]
    pub target_peak: f32,
    #[serde(default = "AudioPostprocessConfig::default_silence_threshold")]
    pub silence_threshold: f32,
    #[serde(default = "AudioPostprocessConfig::default_pad_ms")]
    pub pad_ms: u32,
}

impl AudioPostprocessConfig {
    fn default_enabled() -> bool {
        true
    }
    fn default_target_peak() -> f32 {
        0.95
    }
    fn default_silence_threshold() -> f32 {
        0.01
    }
    fn default_pad_ms() -> u32 {
        50
    }
}

impl Default for AudioPostprocessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_peak: 0.95,
            silence_threshold: 0.01,
            pad_ms: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyPostprocessConfig {
    #[serde(default = "ClassifyPostprocessConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default = "ClassifyPostprocessConfig::default_temperature")]
    pub temperature: f32,
    #[serde(default = "ClassifyPostprocessConfig::default_min_confidence")]
    pub min_confidence: f32,
}

impl ClassifyPostprocessConfig {
    fn default_enabled() -> bool {
        true
    }
    fn default_temperature() -> f32 {
        1.0
    }
    fn default_min_confidence() -> f32 {
        0.01
    }
}

impl Default for ClassifyPostprocessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            temperature: 1.0,
            min_confidence: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloPostprocessConfig {
    #[serde(default = "YoloPostprocessConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default = "YoloPostprocessConfig::default_high_confidence_threshold")]
    pub high_confidence_threshold: f32,
    #[serde(default = "YoloPostprocessConfig::default_medium_confidence_threshold")]
    pub medium_confidence_threshold: f32,
}

impl YoloPostprocessConfig {
    fn default_enabled() -> bool {
        true
    }
    fn default_high_confidence_threshold() -> f32 {
        0.7
    }
    fn default_medium_confidence_threshold() -> f32 {
        0.4
    }
}

impl Default for YoloPostprocessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            high_confidence_threshold: 0.7,
            medium_confidence_threshold: 0.4,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PostprocessConfig {
    #[serde(default)]
    pub audio: AudioPostprocessConfig,
    #[serde(default)]
    pub classify: ClassifyPostprocessConfig,
    #[serde(default)]
    pub yolo: YoloPostprocessConfig,
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        Self::load_from_path(std::path::Path::new("config.toml"))
    }

    pub fn load_from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let config: Config = if path.exists() {
            let content = std::fs::read_to_string(path)?;
            toml::from_str(&content)?
        } else {
            Config::default()
        };

        config.validate()?;
        Ok(config)
    }

    /// Reject configurations that would ship insecure or nonsensical defaults.
    /// Called at the end of `Config::load`.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.auth.enabled {
            if self.auth.jwt_secret == "your-secret-key-here" {
                anyhow::bail!(
                    "auth.enabled is true but jwt_secret is the placeholder \
                     'your-secret-key-here'; set a real secret in config.toml \
                     under [auth] or via the deployment environment"
                );
            }
            if self.auth.jwt_secret.len() < 32 {
                anyhow::bail!(
                    "auth.enabled is true but jwt_secret is shorter than 32 chars \
                     ({}); use a high-entropy secret",
                    self.auth.jwt_secret.len()
                );
            }
        }

        if self.server.port == 0 {
            anyhow::bail!("server.port must be in 1..=65535 (got 0)");
        }

        let min_w = self.performance.min_workers;
        let max_w = self.performance.max_workers;
        if min_w == 0 || max_w == 0 {
            anyhow::bail!(
                "performance.min_workers and max_workers must both be >= 1 \
                 (min={}, max={})",
                min_w,
                max_w
            );
        }
        if min_w > max_w {
            anyhow::bail!(
                "performance.min_workers ({}) must be <= max_workers ({})",
                min_w,
                max_w
            );
        }

        let conf = self.models.yolo_conf_threshold;
        if !(0.0..=1.0).contains(&conf) || conf.is_nan() {
            anyhow::bail!(
                "models.yolo_conf_threshold must be in [0.0, 1.0] (got {})",
                conf
            );
        }
        let iou = self.models.yolo_iou_threshold;
        if !(0.0..=1.0).contains(&iou) || iou.is_nan() {
            anyhow::bail!(
                "models.yolo_iou_threshold must be in [0.0, 1.0] (got {})",
                iou
            );
        }

        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8000,
                log_level: "info".to_string(),
                workers: num_cpus::get(),
                keep_alive_secs: 75,
                request_timeout_secs: 5,
                disconnect_timeout_secs: 1,
                shutdown_timeout_secs: 30,
                json_body_limit_mb: 50,
                proxy_timeout_secs: 300,
                multipart_audio_limit_mb: 100,
                multipart_image_limit_mb: 10,
                classify_image_limit_mb: 5,
                audio_max_duration_secs: 1800,
                tts_chunk_timeout_secs: 30,
                classify_item_timeout_secs: 30,
            },
            device: DeviceConfig {
                device_type: "auto".to_string(),
                device_id: 0,
                device_ids: None,
                use_fp16: false,
                use_tensorrt: false,
                use_torch_compile: false,
                metal_use_mlx: false,
                metal_cache_shaders: true,
                metal_optimize_for_apple_silicon: true,
                enable_jit: true,
                enable_jit_profiling: false,
                enable_jit_executor: true,
                enable_jit_fusion: true,
                num_threads: num_cpus::get(),
                num_interop_threads: 1,
                cudnn_benchmark: true,
                enable_autocast: false,
                torch_warmup_iterations: 5,
            },
            batch: BatchConfig {
                batch_size: 1,
                max_batch_size: 8,
                enable_dynamic_batching: true,
            },
            performance: PerformanceConfig {
                warmup_iterations: 3,
                enable_caching: true,
                enable_profiling: false,
                cache_size_mb: 1024,
                enable_cuda_graphs: false,
                enable_model_quantization: false,
                quantization_bits: 8,
                enable_tensor_pooling: true,
                max_pooled_tensors: 100,
                enable_async_model_loading: true,
                preload_models_on_startup: false,
                enable_result_compression: false,
                compression_level: 6,
                enable_request_batching: true,
                adaptive_batch_timeout: true,
                min_batch_size: 1,
                enable_inflight_batching: false,
                max_inflight_batches: 4,
                enable_worker_pool: true,
                min_workers: 2,
                max_workers: 16,
                enable_auto_scaling: true,
                enable_zero_scaling: false,
            },
            auth: AuthConfig {
                // Default is disabled: a server with no `[auth]` section in
                // config.toml runs unauthenticated. To enable auth, set
                // `auth.enabled = true` *and* provide a real `auth.jwt_secret`
                // (>= 32 chars, not the placeholder). `Config::validate`
                // refuses to start if the placeholder is paired with
                // `enabled = true`.
                enabled: false,
                jwt_secret: "your-secret-key-here".to_string(),
                jwt_algorithm: "HS256".to_string(),
                access_token_expire_minutes: 60,
                refresh_token_expire_days: 7,
            },
            models: ModelsConfig {
                auto_load: vec!["example".to_string()],
                cache_dir: PathBuf::from("models"),
                max_loaded_models: 5,
                classify_model: PathBuf::from("models/classify/efficientnet-lite4-11.onnx"),
                classify_labels: PathBuf::from("models/classify/imagenet1000.txt"),
                audio_model_dir: PathBuf::from("models/audio"),
                yolo_conf_threshold: 0.25,
                yolo_iou_threshold: 0.45,
            },
            microservices: MicroservicesConfig {
                stt_host: "127.0.0.1".to_string(),
                stt_port: 8002,
                llm_host: "127.0.0.1".to_string(),
                llm_port: 8001,
            },
            guard: GuardConfig {
                enable_guards: true,
                max_memory_mb: 8192,
                max_requests_per_second: 1000,
                max_queue_depth: 500,
                min_cache_hit_rate: 60.0,
                max_error_rate: 5.0,
                enable_circuit_breaker: true,
                enable_auto_mitigation: true,
            },
            sanitizer: SanitizerConfig {
                max_text_length: 10000,
                sanitize_text: true,
                sanitize_image_dimensions: true,
                max_image_width: 4096,
                max_image_height: 4096,
                round_probabilities: true,
                probability_decimals: 4,
                remove_null_values: true,
            },
            postprocess: PostprocessConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.server.port, 8000);
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.device.device_type, "auto");
        assert_eq!(config.batch.batch_size, 1);
        assert_eq!(config.batch.max_batch_size, 8);
    }

    #[test]
    fn test_server_config_defaults() {
        let config = Config::default();
        assert_eq!(config.server.log_level, "info");
        assert!(config.server.workers > 0);
    }

    #[test]
    fn test_device_config_defaults() {
        let config = Config::default();
        assert_eq!(config.device.device_id, 0);
        assert!(!config.device.use_fp16);
        assert!(!config.device.use_tensorrt);
        assert!(!config.device.use_torch_compile);
        assert!(!config.device.metal_use_mlx);
        assert!(config.device.metal_cache_shaders);
        assert!(config.device.metal_optimize_for_apple_silicon);
    }

    #[test]
    fn test_batch_config_defaults() {
        let config = Config::default();
        assert!(config.batch.enable_dynamic_batching);
    }

    #[test]
    fn test_performance_config_defaults() {
        let config = Config::default();
        assert_eq!(config.performance.warmup_iterations, 3);
        assert!(config.performance.enable_caching);
        assert!(!config.performance.enable_profiling);
        assert_eq!(config.performance.cache_size_mb, 1024);
    }

    #[test]
    fn test_auth_config_defaults() {
        let config = Config::default();
        // Auth is disabled in the default Config so that a fresh checkout boots
        // without forcing the operator to provision a JWT secret. Operators
        // turning auth on must also set a non-placeholder secret — see
        // `test_validate_rejects_placeholder_jwt_when_auth_enabled`.
        assert!(!config.auth.enabled);
        assert_eq!(config.auth.jwt_algorithm, "HS256");
        assert_eq!(config.auth.access_token_expire_minutes, 60);
        assert_eq!(config.auth.refresh_token_expire_days, 7);
    }

    #[test]
    fn test_validate_accepts_default_config() {
        let config = Config::default();
        assert!(
            config.validate().is_ok(),
            "default config should validate (auth disabled)"
        );
    }

    #[test]
    fn test_validate_rejects_placeholder_jwt_when_auth_enabled() {
        let mut config = Config::default();
        config.auth.enabled = true;
        // jwt_secret is still the placeholder
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("placeholder"), "got: {msg}");
    }

    #[test]
    fn test_validate_rejects_short_jwt_when_auth_enabled() {
        let mut config = Config::default();
        config.auth.enabled = true;
        config.auth.jwt_secret = "short".to_string();
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("32 chars"), "got: {msg}");
    }

    #[test]
    fn test_validate_accepts_strong_jwt_with_auth_enabled() {
        let mut config = Config::default();
        config.auth.enabled = true;
        config.auth.jwt_secret = "x".repeat(64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_port_zero() {
        let mut config = Config::default();
        config.server.port = 0;
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("server.port"));
    }

    #[test]
    fn test_validate_rejects_min_greater_than_max_workers() {
        let mut config = Config::default();
        config.performance.min_workers = 8;
        config.performance.max_workers = 4;
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("min_workers"));
    }

    #[test]
    fn test_validate_rejects_zero_workers() {
        let mut config = Config::default();
        config.performance.min_workers = 0;
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains(">= 1"));
    }

    #[test]
    fn test_validate_rejects_yolo_threshold_out_of_range() {
        let mut config = Config::default();
        config.models.yolo_conf_threshold = 1.5;
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("yolo_conf_threshold"));

        let mut config = Config::default();
        config.models.yolo_iou_threshold = -0.1;
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("yolo_iou_threshold"));
    }

    #[test]
    fn test_models_config_defaults() {
        let config = Config::default();
        assert_eq!(config.models.max_loaded_models, 5);
        assert_eq!(config.models.cache_dir, PathBuf::from("models"));
    }

    #[test]
    fn test_guard_config_defaults() {
        let config = Config::default();
        assert!(config.guard.enable_guards);
        assert_eq!(config.guard.max_memory_mb, 8192);
        assert_eq!(config.guard.max_requests_per_second, 1000);
        assert_eq!(config.guard.max_queue_depth, 500);
        assert_eq!(config.guard.min_cache_hit_rate, 60.0);
        assert_eq!(config.guard.max_error_rate, 5.0);
        assert!(config.guard.enable_circuit_breaker);
        assert!(config.guard.enable_auto_mitigation);
    }

    #[test]
    fn test_sanitizer_config_defaults() {
        let config = Config::default();
        assert_eq!(config.sanitizer.max_text_length, 10000);
        assert!(config.sanitizer.sanitize_text);
        assert!(config.sanitizer.sanitize_image_dimensions);
        assert_eq!(config.sanitizer.max_image_width, 4096);
        assert_eq!(config.sanitizer.max_image_height, 4096);
        assert!(config.sanitizer.round_probabilities);
        assert_eq!(config.sanitizer.probability_decimals, 4);
        assert!(config.sanitizer.remove_null_values);
    }

    #[test]
    fn test_sanitizer_config_standalone_default() {
        let san = SanitizerConfig::default();
        assert_eq!(san.max_text_length, 10000);
        assert!(san.sanitize_text);
        assert!(san.sanitize_image_dimensions);
        assert_eq!(san.max_image_width, 4096);
        assert_eq!(san.max_image_height, 4096);
        assert!(san.round_probabilities);
        assert_eq!(san.probability_decimals, 4);
        assert!(san.remove_null_values);
    }

    #[test]
    fn test_performance_config_extended_defaults() {
        let config = Config::default();
        assert!(!config.performance.enable_cuda_graphs);
        assert!(!config.performance.enable_model_quantization);
        assert_eq!(config.performance.quantization_bits, 8);
        assert!(config.performance.enable_tensor_pooling);
        assert_eq!(config.performance.max_pooled_tensors, 100);
        assert!(config.performance.enable_async_model_loading);
        assert!(!config.performance.preload_models_on_startup);
        assert!(!config.performance.enable_result_compression);
        assert_eq!(config.performance.compression_level, 6);
        assert!(config.performance.enable_request_batching);
        assert!(config.performance.adaptive_batch_timeout);
        assert_eq!(config.performance.min_batch_size, 1);
        assert!(!config.performance.enable_inflight_batching);
        assert_eq!(config.performance.max_inflight_batches, 4);
        assert!(config.performance.enable_worker_pool);
        assert_eq!(config.performance.min_workers, 2);
        assert_eq!(config.performance.max_workers, 16);
        assert!(config.performance.enable_auto_scaling);
        assert!(!config.performance.enable_zero_scaling);
    }

    #[test]
    fn test_device_config_jit_defaults() {
        let config = Config::default();
        assert!(config.device.enable_jit);
        assert!(!config.device.enable_jit_profiling);
        assert!(config.device.enable_jit_executor);
        assert!(config.device.enable_jit_fusion);
    }

    #[test]
    fn test_device_config_thread_defaults() {
        let config = Config::default();
        assert!(config.device.num_threads > 0);
        assert_eq!(config.device.num_interop_threads, 1);
        assert!(config.device.cudnn_benchmark);
        assert!(!config.device.enable_autocast);
        assert_eq!(config.device.torch_warmup_iterations, 5);
    }

    #[test]
    fn test_models_config_auto_load_defaults() {
        let config = Config::default();
        assert_eq!(config.models.auto_load, vec!["example".to_string()]);
    }

    #[test]
    fn test_auth_jwt_secret_defaults() {
        let config = Config::default();
        assert_eq!(config.auth.jwt_secret, "your-secret-key-here");
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.server.port, config.server.port);
        assert_eq!(deserialized.server.host, config.server.host);
        assert_eq!(deserialized.batch.batch_size, config.batch.batch_size);
        assert_eq!(deserialized.auth.jwt_algorithm, config.auth.jwt_algorithm);
        assert_eq!(deserialized.guard.max_memory_mb, config.guard.max_memory_mb);
        assert_eq!(
            deserialized.sanitizer.max_text_length,
            config.sanitizer.max_text_length
        );
    }

    #[test]
    fn test_server_config_standalone_default() {
        let srv = ServerConfig::default();
        assert_eq!(srv.host, "0.0.0.0");
        assert_eq!(srv.port, 8000);
        assert_eq!(srv.log_level, "info");
        assert!(srv.workers > 0);
        assert_eq!(srv.keep_alive_secs, 75);
        assert_eq!(srv.request_timeout_secs, 5);
        assert_eq!(srv.disconnect_timeout_secs, 1);
        assert_eq!(srv.shutdown_timeout_secs, 30);
        assert_eq!(srv.json_body_limit_mb, 50);
        assert_eq!(srv.proxy_timeout_secs, 300);
    }

    #[test]
    fn test_batch_config_standalone_default() {
        let batch = BatchConfig::default();
        assert_eq!(batch.batch_size, 0);
        assert_eq!(batch.max_batch_size, 0);
        assert!(!batch.enable_dynamic_batching);
    }

    #[test]
    fn test_auth_config_standalone_default() {
        let auth = AuthConfig::default();
        assert!(!auth.enabled);
        assert_eq!(auth.jwt_secret, "");
        assert_eq!(auth.jwt_algorithm, "");
        assert_eq!(auth.access_token_expire_minutes, 0);
        assert_eq!(auth.refresh_token_expire_days, 0);
    }

    #[test]
    fn test_guard_config_standalone_default() {
        let guard = GuardConfig::default();
        assert!(!guard.enable_guards);
        assert_eq!(guard.max_memory_mb, 0);
        assert_eq!(guard.max_requests_per_second, 0);
        assert!(!guard.enable_circuit_breaker);
        assert!(!guard.enable_auto_mitigation);
    }

    // ── PostprocessConfig ─────────────────────────────────────────────────────

    #[test]
    fn test_postprocess_config_defaults() {
        let cfg = PostprocessConfig::default();
        assert!(cfg.audio.enabled);
        assert!((cfg.audio.target_peak - 0.95).abs() < f32::EPSILON);
        assert!((cfg.audio.silence_threshold - 0.01).abs() < f32::EPSILON);
        assert_eq!(cfg.audio.pad_ms, 50);
        assert!(cfg.classify.enabled);
        assert!((cfg.classify.temperature - 1.0).abs() < f32::EPSILON);
        assert!((cfg.classify.min_confidence - 0.01).abs() < f32::EPSILON);
        assert!(cfg.yolo.enabled);
        assert!((cfg.yolo.high_confidence_threshold - 0.7).abs() < f32::EPSILON);
        assert!((cfg.yolo.medium_confidence_threshold - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_postprocess_config_partial_toml_audio_only_enabled() {
        // Verify per-field serde defaults: when only `enabled` is specified,
        // the other fields take their Default values rather than erroring.
        // Parsing into AudioPostprocessConfig directly exercises the per-field defaults.
        let toml = "enabled = false\n";
        let cfg: AudioPostprocessConfig = toml::from_str(toml).unwrap();
        assert!(!cfg.enabled);
        assert!(
            (cfg.target_peak - 0.95).abs() < f32::EPSILON,
            "target_peak should default to 0.95"
        );
        assert_eq!(cfg.pad_ms, 50, "pad_ms should default to 50");
    }

    #[test]
    fn test_postprocess_config_partial_toml_classify_only_temperature() {
        let toml = "temperature = 2.0\n";
        let cfg: ClassifyPostprocessConfig = toml::from_str(toml).unwrap();
        assert!(cfg.enabled, "enabled should default to true");
        assert!((cfg.temperature - 2.0).abs() < f32::EPSILON);
        assert!((cfg.min_confidence - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_postprocess_config_partial_toml_yolo_only_threshold() {
        let toml = "high_confidence_threshold = 0.9\n";
        let cfg: YoloPostprocessConfig = toml::from_str(toml).unwrap();
        assert!(cfg.enabled, "enabled should default to true");
        assert!((cfg.high_confidence_threshold - 0.9).abs() < f32::EPSILON);
        assert!((cfg.medium_confidence_threshold - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_postprocess_config_partial_toml_audio_only_peak() {
        // Specifying target_peak but NOT enabled — covers default_enabled helper.
        let toml = "target_peak = 0.8\n";
        let cfg: AudioPostprocessConfig = toml::from_str(toml).unwrap();
        assert!(cfg.enabled, "enabled should default to true");
        assert!((cfg.target_peak - 0.8).abs() < f32::EPSILON);
        assert_eq!(cfg.pad_ms, 50);
    }

    #[test]
    fn test_postprocess_config_partial_toml_classify_only_enabled() {
        // Specifying enabled but NOT temperature — covers default_temperature helper.
        let toml = "enabled = false\n";
        let cfg: ClassifyPostprocessConfig = toml::from_str(toml).unwrap();
        assert!(!cfg.enabled);
        assert!(
            (cfg.temperature - 1.0).abs() < f32::EPSILON,
            "temperature should default to 1.0"
        );
        assert!((cfg.min_confidence - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_postprocess_config_partial_toml_yolo_only_medium_threshold() {
        // Specifying medium_confidence_threshold but NOT high — covers default_high_confidence_threshold helper.
        let toml = "medium_confidence_threshold = 0.5\n";
        let cfg: YoloPostprocessConfig = toml::from_str(toml).unwrap();
        assert!(cfg.enabled, "enabled should default to true");
        assert!(
            (cfg.high_confidence_threshold - 0.7).abs() < f32::EPSILON,
            "high threshold should default to 0.7"
        );
        assert!((cfg.medium_confidence_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_postprocess_config_empty_section_uses_all_defaults() {
        // Empty string deserializes into all-default PostprocessConfig.
        let cfg: PostprocessConfig = toml::from_str("").unwrap();
        assert!(cfg.audio.enabled);
        assert!(cfg.classify.enabled);
        assert!(cfg.yolo.enabled);
    }

    #[test]
    fn test_config_load_returns_default_when_no_file() {
        // Since config.toml doesn't necessarily exist in the test environment,
        // Config::load() should return Ok with a default when the file is absent.
        // We test by ensuring it doesn't panic and returns a valid Config.
        // Note: if config.toml exists, this tests the parse path instead.
        let result = Config::load();
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_clone() {
        let config = Config::default();
        let cloned = config.clone();
        assert_eq!(cloned.server.port, config.server.port);
        assert_eq!(cloned.device.device_type, config.device.device_type);
        assert_eq!(cloned.batch.max_batch_size, config.batch.max_batch_size);
    }

    /// Covers line 158: Config::load() returns Config::default() when
    /// config.toml does not exist in the working directory.
    /// Loads config from a path where no config.toml exists — exercises the else branch.
    #[test]
    #[serial_test::serial]
    fn test_config_load_without_config_file() {
        let non_existent = std::path::Path::new("/tmp/torch_inference_test_definitely_absent.toml");
        let result = Config::load_from_path(non_existent);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.server.port, 8000);
    }
}
