use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use crate::core::gpu::GpuManager;
use crate::error::ApiError;
use std::sync::Arc;

#[derive(Debug, Serialize)]
pub struct SystemInfoResponse {
    pub system: SystemDetails,
    pub gpu: GpuInfoDetails,
    pub runtime: RuntimeDetails,
    pub features: FeatureFlags,
}

#[derive(Debug, Serialize)]
pub struct SystemDetails {
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_memory_bytes: u64,
    pub total_memory_human: String,
    pub hostname: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GpuInfoDetails {
    pub available: bool,
    pub count: usize,
    pub devices: Vec<GpuDeviceInfo>,
}

#[derive(Debug, Serialize)]
pub struct GpuDeviceInfo {
    pub id: usize,
    pub name: String,
    pub total_memory: u64,
    pub total_memory_human: String,
    pub free_memory: u64,
    pub free_memory_human: String,
    pub utilization: Option<u32>,
    pub temperature: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct RuntimeDetails {
    pub version: String,
    pub build_date: String,
    pub rust_version: String,
    pub uptime_secs: u64,
}

#[derive(Debug, Serialize)]
pub struct FeatureFlags {
    pub cuda_enabled: bool,
    pub onnx_enabled: bool,
    pub torch_enabled: bool,
    pub audio_processing: bool,
    pub image_security: bool,
}

#[derive(Debug, Serialize)]
pub struct ConfigResponse {
    pub server: ServerConfig,
    pub inference: InferenceConfig,
    pub cache: CacheConfig,
}

#[derive(Debug, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub max_connections: usize,
}

#[derive(Debug, Serialize)]
pub struct InferenceConfig {
    pub default_batch_size: usize,
    pub max_batch_size: usize,
    pub timeout_secs: u64,
    pub device: String,
}

#[derive(Debug, Serialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub ttl_secs: u64,
    pub max_size_mb: usize,
}

pub struct SystemInfoState {
    pub gpu_manager: Arc<GpuManager>,
    pub start_time: std::time::Instant,
}

pub async fn get_system_info(
    state: web::Data<SystemInfoState>,
) -> Result<HttpResponse, ApiError> {
    let gpu_info = state.gpu_manager.get_info()
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    let sys = sysinfo::System::new_all();
    
    let system = SystemDetails {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_count: num_cpus::get(),
        total_memory_bytes: sys.total_memory(),
        total_memory_human: format_bytes(sys.total_memory()),
        hostname: sysinfo::System::host_name(),
    };

    let gpu_devices: Vec<GpuDeviceInfo> = gpu_info.devices.iter().map(|d| {
        GpuDeviceInfo {
            id: d.id,
            name: d.name.clone(),
            total_memory: d.total_memory,
            total_memory_human: format_bytes(d.total_memory),
            free_memory: d.free_memory,
            free_memory_human: format_bytes(d.free_memory),
            utilization: d.utilization,
            temperature: d.temperature,
        }
    }).collect();

    let gpu = GpuInfoDetails {
        available: gpu_info.available,
        count: gpu_info.count,
        devices: gpu_devices,
    };

    let runtime = RuntimeDetails {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_date: option_env!("BUILD_DATE").unwrap_or("unknown").to_string(),
        rust_version: option_env!("RUST_VERSION").unwrap_or(env!("CARGO_PKG_RUST_VERSION")).to_string(),
        uptime_secs: state.start_time.elapsed().as_secs(),
    };

    let features = FeatureFlags {
        cuda_enabled: cfg!(feature = "cuda"),
        onnx_enabled: cfg!(feature = "onnx"),
        torch_enabled: cfg!(feature = "torch"),
        audio_processing: true,
        image_security: true,
    };

    Ok(HttpResponse::Ok().json(SystemInfoResponse {
        system,
        gpu,
        runtime,
        features,
    }))
}

pub async fn get_config() -> Result<HttpResponse, ApiError> {
    // Load from environment or config file
    let config = ConfigResponse {
        server: ServerConfig {
            host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            workers: num_cpus::get(),
            max_connections: 1000,
        },
        inference: InferenceConfig {
            default_batch_size: 1,
            max_batch_size: 32,
            timeout_secs: 30,
            device: if cfg!(feature = "cuda") { "cuda" } else { "cpu" }.to_string(),
        },
        cache: CacheConfig {
            enabled: true,
            ttl_secs: 3600,
            max_size_mb: 1024,
        },
    };

    Ok(HttpResponse::Ok().json(config))
}

pub async fn get_gpu_stats(
    state: web::Data<SystemInfoState>,
) -> Result<HttpResponse, ApiError> {
    let stats = state.gpu_manager.collect_stats()
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(stats))
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}
