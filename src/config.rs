use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub device: DeviceConfig,
    pub batch: BatchConfig,
    pub performance: PerformanceConfig,
    pub auth: AuthConfig,
    pub models: ModelsConfig,
    pub guard: GuardConfig,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub log_level: String,
    pub workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub device_type: String,
    pub device_id: usize,
    pub device_ids: Option<Vec<usize>>,
    pub use_fp16: bool,
    pub use_tensorrt: bool,
    pub use_torch_compile: bool,
    
    // Metal-specific optimizations (macOS)
    pub metal_use_mlx: bool,
    pub metal_cache_shaders: bool,
    pub metal_optimize_for_apple_silicon: bool,
    
    // JIT Compilation settings
    pub enable_jit: bool,
    pub enable_jit_profiling: bool,
    pub enable_jit_executor: bool,
    pub enable_jit_fusion: bool,
    
    // PyTorch/LibTorch optimizations
    pub num_threads: usize,
    pub num_interop_threads: usize,
    pub cudnn_benchmark: bool,
    pub enable_autocast: bool,
    pub torch_warmup_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub max_batch_size: usize,
    pub enable_dynamic_batching: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub warmup_iterations: usize,
    pub enable_caching: bool,
    pub enable_profiling: bool,
    pub cache_size_mb: usize,
    pub enable_cuda_graphs: bool,
    pub enable_model_quantization: bool,
    pub quantization_bits: u8,
    pub enable_tensor_pooling: bool,
    pub max_pooled_tensors: usize,
    pub enable_async_model_loading: bool,
    pub preload_models_on_startup: bool,
    pub enable_result_compression: bool,
    pub compression_level: u32,
    pub enable_request_batching: bool,
    pub adaptive_batch_timeout: bool,
    pub min_batch_size: usize,
    pub enable_inflight_batching: bool,
    pub max_inflight_batches: usize,
    pub enable_worker_pool: bool,
    pub min_workers: usize,
    pub max_workers: usize,
    pub enable_auto_scaling: bool,
    pub enable_zero_scaling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub enabled: bool,
    pub jwt_secret: String,
    pub jwt_algorithm: String,
    pub access_token_expire_minutes: u32,
    pub refresh_token_expire_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub auto_load: Vec<String>,
    pub cache_dir: PathBuf,
    pub max_loaded_models: usize,
    pub instances_per_model: usize,  // Number of model replicas for parallel inference
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardConfig {
    pub enable_guards: bool,
    pub max_memory_mb: usize,
    pub max_requests_per_second: usize,
    pub max_queue_depth: usize,
    pub min_cache_hit_rate: f64,
    pub max_error_rate: f64,
    pub enable_circuit_breaker: bool,
    pub enable_auto_mitigation: bool,
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
                instances_per_model: 1,  // Default to 1 instance
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
}
