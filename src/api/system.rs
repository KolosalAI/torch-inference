use crate::core::gpu::GpuManager;
use crate::error::ApiError;
use actix_web::{web, HttpResponse, Result};
use serde::Serialize;
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

pub async fn get_system_info(state: web::Data<SystemInfoState>) -> Result<HttpResponse, ApiError> {
    let gpu_info = state
        .gpu_manager
        .get_info()
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

    let gpu_devices: Vec<GpuDeviceInfo> = gpu_info
        .devices
        .iter()
        .map(|d| GpuDeviceInfo {
            id: d.id,
            name: d.name.clone(),
            total_memory: d.total_memory,
            total_memory_human: format_bytes(d.total_memory),
            free_memory: d.free_memory,
            free_memory_human: format_bytes(d.free_memory),
            utilization: d.utilization,
            temperature: d.temperature,
        })
        .collect();

    let gpu = GpuInfoDetails {
        available: gpu_info.available,
        count: gpu_info.count,
        devices: gpu_devices,
    };

    let runtime = RuntimeDetails {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_date: option_env!("BUILD_DATE").unwrap_or("unknown").to_string(),
        rust_version: option_env!("RUST_VERSION")
            .unwrap_or(env!("CARGO_PKG_RUST_VERSION"))
            .to_string(),
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
            device: if cfg!(feature = "cuda") {
                "cuda"
            } else {
                "cpu"
            }
            .to_string(),
        },
        cache: CacheConfig {
            enabled: true,
            ttl_secs: 3600,
            max_size_mb: 1024,
        },
    };

    Ok(HttpResponse::Ok().json(config))
}

pub async fn get_gpu_stats(state: web::Data<SystemInfoState>) -> Result<HttpResponse, ApiError> {
    let stats = state
        .gpu_manager
        .collect_stats()
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

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::http::StatusCode;
    use actix_web::{test, web, App};

    fn make_system_state() -> web::Data<SystemInfoState> {
        web::Data::new(SystemInfoState {
            gpu_manager: Arc::new(GpuManager::new()),
            start_time: std::time::Instant::now(),
        })
    }

    // ── get_system_info ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_system_info_returns_200() {
        let state = make_system_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/system/info", web::get().to(get_system_info)),
        )
        .await;
        let req = test::TestRequest::get().uri("/system/info").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_system_info_response_shape() {
        let state = make_system_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/system/info", web::get().to(get_system_info)),
        )
        .await;
        let req = test::TestRequest::get().uri("/system/info").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["system"].is_object());
        assert!(body["gpu"].is_object());
        assert!(body["runtime"].is_object());
        assert!(body["features"].is_object());
    }

    #[tokio::test]
    async fn test_get_system_info_system_fields() {
        let state = make_system_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/system/info", web::get().to(get_system_info)),
        )
        .await;
        let req = test::TestRequest::get().uri("/system/info").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let sys = &body["system"];
        assert!(sys["os"].is_string());
        assert!(sys["arch"].is_string());
        assert!(sys["cpu_count"].as_u64().unwrap() > 0);
        assert!(sys["total_memory_human"].is_string());
    }

    #[tokio::test]
    async fn test_get_system_info_features_fields() {
        let state = make_system_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/system/info", web::get().to(get_system_info)),
        )
        .await;
        let req = test::TestRequest::get().uri("/system/info").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let features = &body["features"];
        assert!(features["audio_processing"].as_bool().unwrap());
        assert!(features["image_security"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_get_system_info_runtime_fields() {
        let state = make_system_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/system/info", web::get().to(get_system_info)),
        )
        .await;
        let req = test::TestRequest::get().uri("/system/info").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let rt = &body["runtime"];
        assert!(rt["version"].is_string());
        assert!(rt["uptime_secs"].is_number());
    }

    // ── get_config ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_config_returns_200() {
        let app =
            test::init_service(App::new().route("/system/config", web::get().to(get_config))).await;
        let req = test::TestRequest::get().uri("/system/config").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_config_response_shape() {
        let app =
            test::init_service(App::new().route("/system/config", web::get().to(get_config))).await;
        let req = test::TestRequest::get().uri("/system/config").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["server"].is_object());
        assert!(body["inference"].is_object());
        assert!(body["cache"].is_object());
    }

    #[tokio::test]
    async fn test_get_config_server_defaults() {
        std::env::remove_var("HOST");
        std::env::remove_var("PORT");
        let app =
            test::init_service(App::new().route("/system/config", web::get().to(get_config))).await;
        let req = test::TestRequest::get().uri("/system/config").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let server = &body["server"];
        assert_eq!(server["host"].as_str().unwrap(), "0.0.0.0");
        assert_eq!(server["port"].as_u64().unwrap(), 8080);
        assert!(server["workers"].as_u64().unwrap() > 0);
        assert_eq!(server["max_connections"].as_u64().unwrap(), 1000);
    }

    #[tokio::test]
    async fn test_get_config_inference_defaults() {
        let app =
            test::init_service(App::new().route("/system/config", web::get().to(get_config))).await;
        let req = test::TestRequest::get().uri("/system/config").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let inf = &body["inference"];
        assert_eq!(inf["default_batch_size"].as_u64().unwrap(), 1);
        assert_eq!(inf["max_batch_size"].as_u64().unwrap(), 32);
        assert_eq!(inf["timeout_secs"].as_u64().unwrap(), 30);
        assert!(inf["device"].is_string());
    }

    #[tokio::test]
    async fn test_get_config_cache_defaults() {
        let app =
            test::init_service(App::new().route("/system/config", web::get().to(get_config))).await;
        let req = test::TestRequest::get().uri("/system/config").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let cache = &body["cache"];
        assert_eq!(cache["enabled"].as_bool().unwrap(), true);
        assert_eq!(cache["ttl_secs"].as_u64().unwrap(), 3600);
        assert_eq!(cache["max_size_mb"].as_u64().unwrap(), 1024);
    }

    #[tokio::test]
    async fn test_get_config_port_from_env() {
        std::env::set_var("PORT", "9999");
        let app =
            test::init_service(App::new().route("/system/config", web::get().to(get_config))).await;
        let req = test::TestRequest::get().uri("/system/config").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["server"]["port"].as_u64().unwrap(), 9999);
        std::env::remove_var("PORT");
    }

    // ── get_gpu_stats ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_gpu_stats_returns_200() {
        let state = make_system_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/system/gpu", web::get().to(get_gpu_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/system/gpu").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_gpu_stats_response_is_object_or_array() {
        let state = make_system_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/system/gpu", web::get().to(get_gpu_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/system/gpu").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        // The response should be some valid JSON
        assert!(!body.is_null());
    }

    // ── format_bytes ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[tokio::test]
    async fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(512), "512.00 B");
        assert_eq!(format_bytes(1), "1.00 B");
        assert_eq!(format_bytes(1023), "1023.00 B");
    }

    #[tokio::test]
    async fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(2048), "2.00 KB");
        assert_eq!(format_bytes(1024 * 512), "512.00 KB");
    }

    #[tokio::test]
    async fn test_format_bytes_megabytes() {
        let one_mb = 1024u64 * 1024;
        assert_eq!(format_bytes(one_mb), "1.00 MB");
        assert_eq!(format_bytes(one_mb * 10), "10.00 MB");
    }

    #[tokio::test]
    async fn test_format_bytes_gigabytes() {
        let one_gb = 1024u64 * 1024 * 1024;
        assert_eq!(format_bytes(one_gb), "1.00 GB");
        assert_eq!(format_bytes(one_gb * 8), "8.00 GB");
    }

    #[tokio::test]
    async fn test_format_bytes_terabytes() {
        let one_tb = 1024u64 * 1024 * 1024 * 1024;
        assert_eq!(format_bytes(one_tb), "1.00 TB");
    }

    #[tokio::test]
    async fn test_format_bytes_large_tb() {
        // Values larger than 1 TB should stay in TB (last unit)
        let two_tb = 2u64 * 1024 * 1024 * 1024 * 1024;
        assert_eq!(format_bytes(two_tb), "2.00 TB");
    }

    // ── struct construction ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_server_config_struct() {
        let sc = ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            workers: 4,
            max_connections: 1000,
        };
        assert_eq!(sc.host, "127.0.0.1");
        assert_eq!(sc.port, 8080);
        assert_eq!(sc.workers, 4);
        assert_eq!(sc.max_connections, 1000);
    }

    #[tokio::test]
    async fn test_inference_config_struct() {
        let ic = InferenceConfig {
            default_batch_size: 1,
            max_batch_size: 32,
            timeout_secs: 30,
            device: "cpu".to_string(),
        };
        assert_eq!(ic.default_batch_size, 1);
        assert_eq!(ic.max_batch_size, 32);
        assert_eq!(ic.timeout_secs, 30);
        assert_eq!(ic.device, "cpu");
    }

    #[tokio::test]
    async fn test_cache_config_struct() {
        let cc = CacheConfig {
            enabled: true,
            ttl_secs: 3600,
            max_size_mb: 1024,
        };
        assert!(cc.enabled);
        assert_eq!(cc.ttl_secs, 3600);
        assert_eq!(cc.max_size_mb, 1024);
    }

    #[tokio::test]
    async fn test_feature_flags_struct() {
        let ff = FeatureFlags {
            cuda_enabled: false,
            onnx_enabled: true,
            torch_enabled: false,
            audio_processing: true,
            image_security: true,
        };
        assert!(!ff.cuda_enabled);
        assert!(ff.onnx_enabled);
        assert!(ff.audio_processing);
        assert!(ff.image_security);
    }

    #[tokio::test]
    async fn test_runtime_details_struct() {
        let rd = RuntimeDetails {
            version: "1.2.3".to_string(),
            build_date: "2026-03-27".to_string(),
            rust_version: "1.75.0".to_string(),
            uptime_secs: 42,
        };
        assert_eq!(rd.version, "1.2.3");
        assert_eq!(rd.uptime_secs, 42);
    }

    #[tokio::test]
    async fn test_gpu_device_info_struct() {
        let gd = GpuDeviceInfo {
            id: 0,
            name: "NVIDIA RTX 4090".to_string(),
            total_memory: 1024 * 1024 * 1024 * 24,
            total_memory_human: "24.00 GB".to_string(),
            free_memory: 1024 * 1024 * 1024 * 20,
            free_memory_human: "20.00 GB".to_string(),
            utilization: Some(42),
            temperature: Some(65),
        };
        assert_eq!(gd.id, 0);
        assert_eq!(gd.name, "NVIDIA RTX 4090");
        assert_eq!(gd.utilization, Some(42));
        assert_eq!(gd.temperature, Some(65));
    }

    #[tokio::test]
    async fn test_gpu_info_details_no_devices() {
        let gid = GpuInfoDetails {
            available: false,
            count: 0,
            devices: vec![],
        };
        assert!(!gid.available);
        assert_eq!(gid.count, 0);
        assert!(gid.devices.is_empty());
    }

    #[tokio::test]
    async fn test_system_details_struct() {
        let sd = SystemDetails {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            cpu_count: 8,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            total_memory_human: "16.00 GB".to_string(),
            hostname: Some("my-server".to_string()),
        };
        assert_eq!(sd.os, "linux");
        assert_eq!(sd.cpu_count, 8);
        assert_eq!(sd.hostname, Some("my-server".to_string()));
    }

    #[tokio::test]
    async fn test_config_response_serde() {
        let cr = ConfigResponse {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                workers: 2,
                max_connections: 500,
            },
            inference: InferenceConfig {
                default_batch_size: 1,
                max_batch_size: 16,
                timeout_secs: 60,
                device: "cpu".to_string(),
            },
            cache: CacheConfig {
                enabled: false,
                ttl_secs: 0,
                max_size_mb: 0,
            },
        };
        let json = serde_json::to_string(&cr).expect("serialize failed");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse failed");
        assert_eq!(v["server"]["port"], 8080);
        assert_eq!(v["inference"]["device"], "cpu");
        assert_eq!(v["cache"]["enabled"], false);
    }
}
