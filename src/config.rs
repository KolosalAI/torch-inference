use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    pub guard: GuardConfig,
    #[serde(default)]
    pub sanitizer: SanitizerConfig,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerConfig {
    #[serde(default)] pub host: String,
    #[serde(default)] pub port: u16,
    #[serde(default)] pub log_level: String,
    #[serde(default)] pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceConfig {
    #[serde(default)] pub device_type: String,
    #[serde(default)] pub device_id: usize,
    #[serde(default)] pub device_ids: Option<Vec<usize>>,
    #[serde(default)] pub use_fp16: bool,
    #[serde(default)] pub use_tensorrt: bool,
    #[serde(default)] pub use_torch_compile: bool,

    // Metal-specific optimizations (macOS)
    #[serde(default)] pub metal_use_mlx: bool,
    #[serde(default)] pub metal_cache_shaders: bool,
    #[serde(default)] pub metal_optimize_for_apple_silicon: bool,
    
    // JIT Compilation settings
    #[serde(default)] pub enable_jit: bool,
    #[serde(default)] pub enable_jit_profiling: bool,
    #[serde(default)] pub enable_jit_executor: bool,
    #[serde(default)] pub enable_jit_fusion: bool,

    // PyTorch/LibTorch optimizations
    #[serde(default)] pub num_threads: usize,
    #[serde(default)] pub num_interop_threads: usize,
    #[serde(default)] pub cudnn_benchmark: bool,
    #[serde(default)] pub enable_autocast: bool,
    #[serde(default)] pub torch_warmup_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BatchConfig {
    #[serde(default)] pub batch_size: usize,
    #[serde(default)] pub max_batch_size: usize,
    #[serde(default)] pub enable_dynamic_batching: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceConfig {
    #[serde(default)] pub warmup_iterations: usize,
    #[serde(default)] pub enable_caching: bool,
    #[serde(default)] pub enable_profiling: bool,
    #[serde(default)] pub cache_size_mb: usize,
    #[serde(default)] pub enable_cuda_graphs: bool,
    #[serde(default)] pub enable_model_quantization: bool,
    #[serde(default)] pub quantization_bits: u8,
    #[serde(default)] pub enable_tensor_pooling: bool,
    #[serde(default)] pub max_pooled_tensors: usize,
    #[serde(default)] pub enable_async_model_loading: bool,
    #[serde(default)] pub preload_models_on_startup: bool,
    #[serde(default)] pub enable_result_compression: bool,
    #[serde(default)] pub compression_level: u32,
    #[serde(default)] pub enable_request_batching: bool,
    #[serde(default)] pub adaptive_batch_timeout: bool,
    #[serde(default)] pub min_batch_size: usize,
    #[serde(default)] pub enable_inflight_batching: bool,
    #[serde(default)] pub max_inflight_batches: usize,
    #[serde(default)] pub enable_worker_pool: bool,
    #[serde(default)] pub min_workers: usize,
    #[serde(default)] pub max_workers: usize,
    #[serde(default)] pub enable_auto_scaling: bool,
    #[serde(default)] pub enable_zero_scaling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthConfig {
    #[serde(default)] pub enabled: bool,
    #[serde(default)] pub jwt_secret: String,
    #[serde(default)] pub jwt_algorithm: String,
    #[serde(default)] pub access_token_expire_minutes: u32,
    #[serde(default)] pub refresh_token_expire_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelsConfig {
    #[serde(default)] pub auto_load: Vec<String>,
    #[serde(default)] pub cache_dir: PathBuf,
    #[serde(default)] pub max_loaded_models: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GuardConfig {
    #[serde(default)] pub enable_guards: bool,
    #[serde(default)] pub max_memory_mb: usize,
    #[serde(default)] pub max_requests_per_second: usize,
    #[serde(default)] pub max_queue_depth: usize,
    #[serde(default)] pub min_cache_hit_rate: f64,
    #[serde(default)] pub max_error_rate: f64,
    #[serde(default)] pub enable_circuit_breaker: bool,
    #[serde(default)] pub enable_auto_mitigation: bool,
}

impl Config {
    pub fn load() -> anyhow::Result<Self> {
        let config_file = "config.toml";
        
        if std::path::Path::new(config_file).exists() {
            let content = std::fs::read_to_string(config_file)?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
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
                enabled: true,
                jwt_secret: "your-secret-key-here".to_string(),
                jwt_algorithm: "HS256".to_string(),
                access_token_expire_minutes: 60,
                refresh_token_expire_days: 7,
            },
            models: ModelsConfig {
                auto_load: vec!["example".to_string()],
                cache_dir: PathBuf::from("models"),
                max_loaded_models: 5,
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
        assert!(config.auth.enabled);
        assert_eq!(config.auth.jwt_algorithm, "HS256");
        assert_eq!(config.auth.access_token_expire_minutes, 60);
        assert_eq!(config.auth.refresh_token_expire_days, 7);
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
        assert_eq!(srv.host, "");
        assert_eq!(srv.port, 0);
        assert_eq!(srv.log_level, "");
        assert_eq!(srv.workers, 0);
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
}
