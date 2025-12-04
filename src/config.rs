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
    pub use_fp16: bool,
    pub use_tensorrt: bool,
    pub use_torch_compile: bool,
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
                use_fp16: false,
                use_tensorrt: false,
                use_torch_compile: false,
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
        }
    }
}
