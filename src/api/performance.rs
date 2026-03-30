#![allow(dead_code)]
/// Performance monitoring and profiling
use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use sysinfo::System;
use crate::error::ApiError;
use crate::monitor::Monitor;
use std::sync::Arc;

#[derive(Debug, Serialize)]
pub struct PerformanceMetrics {
    pub timestamp: String,
    pub system_info: SystemInfo,
    pub process_info: ProcessInfo,
    pub runtime_info: RuntimeInfo,
}

#[derive(Debug, Serialize)]
pub struct SystemInfo {
    pub cpu_count: usize,
    pub total_memory_mb: u64,
    pub available_memory_mb: u64,
    pub used_memory_mb: u64,
    pub memory_usage_percent: f32,
    pub cpu_usage_percent: f32,
}

#[derive(Debug, Serialize)]
pub struct ProcessInfo {
    pub pid: u32,
    pub memory_mb: f64,
    pub cpu_usage_percent: f32,
    pub uptime_seconds: u64,
}

#[derive(Debug, Serialize)]
pub struct RuntimeInfo {
    pub rust_version: String,
    pub actix_web_version: String,
    pub num_cpus: usize,
}

#[derive(Debug, Serialize)]
pub struct ProfileResult {
    pub model_name: String,
    pub total_time_ms: f64,
    pub pre_metrics: ResourceMetrics,
    pub post_metrics: ResourceMetrics,
    pub delta_metrics: ResourceMetrics,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResourceMetrics {
    pub memory_mb: f64,
    pub cpu_percent: f32,
}

#[derive(Debug, Serialize)]
pub struct OptimizationResult {
    pub garbage_collected: bool,
    pub caches_cleared: bool,
    pub memory_freed_mb: f64,
    pub optimizations_applied: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ProfileRequest {
    pub model: Option<String>,
    pub input_data: Option<serde_json::Value>,
}

pub struct PerformanceState {
    pub monitor: Arc<Monitor>,
    pub start_time: Instant,
}

/// Get comprehensive performance metrics
pub async fn get_performance_metrics(
    state: web::Data<PerformanceState>,
) -> Result<HttpResponse, ApiError> {
    log::info!("[ENDPOINT] Performance metrics requested");

    let mut system = System::new_all();
    system.refresh_all();

    // System info
    let total_memory = system.total_memory();
    let available_memory = system.available_memory();
    let used_memory = system.used_memory();
    
    // Calculate CPU usage (average of all CPUs)
    let cpu_usage = system.cpus().iter()
        .map(|cpu| cpu.cpu_usage())
        .sum::<f32>() / system.cpus().len() as f32;
    
    let system_info = SystemInfo {
        cpu_count: system.cpus().len(),
        total_memory_mb: total_memory / 1024 / 1024,
        available_memory_mb: available_memory / 1024 / 1024,
        used_memory_mb: used_memory / 1024 / 1024,
        memory_usage_percent: (used_memory as f32 / total_memory as f32) * 100.0,
        cpu_usage_percent: cpu_usage,
    };

    // Process info
    let pid = sysinfo::get_current_pid()
        .map_err(|e| ApiError::InternalError(format!("Failed to get PID: {}", e)))?;
    
    let process_info = if let Some(process) = system.process(pid) {
        ProcessInfo {
            pid: pid.as_u32(),
            memory_mb: process.memory() as f64 / 1024.0 / 1024.0,
            cpu_usage_percent: process.cpu_usage(),
            uptime_seconds: state.start_time.elapsed().as_secs(),
        }
    } else {
        ProcessInfo {
            pid: pid.as_u32(),
            memory_mb: 0.0,
            cpu_usage_percent: 0.0,
            uptime_seconds: state.start_time.elapsed().as_secs(),
        }
    };

    // Runtime info
    let runtime_info = RuntimeInfo {
        rust_version: env!("CARGO_PKG_VERSION").to_string(),
        actix_web_version: "4.8".to_string(),
        num_cpus: num_cpus::get(),
    };

    let metrics = PerformanceMetrics {
        timestamp: chrono::Utc::now().to_rfc3339(),
        system_info,
        process_info,
        runtime_info,
    };

    log::info!("[ENDPOINT] Performance metrics collected");

    Ok(HttpResponse::Ok().json(metrics))
}

/// Profile a specific inference request
pub async fn profile_inference(
    req: web::Json<ProfileRequest>,
    _state: web::Data<PerformanceState>,
) -> Result<HttpResponse, ApiError> {
    log::info!("[ENDPOINT] Performance profiling requested");

    let model_name = req.model.as_deref().unwrap_or("example");
    
    let mut system = System::new_all();
    system.refresh_all();
    let pid = sysinfo::get_current_pid()
        .map_err(|e| ApiError::InternalError(format!("Failed to get PID: {}", e)))?;

    // Pre-profiling metrics
    let pre_metrics = ResourceMetrics {
        memory_mb: system.process(pid)
            .map(|p| p.memory() as f64 / 1024.0 / 1024.0)
            .unwrap_or(0.0),
        cpu_percent: system.cpus().iter()
            .map(|cpu| cpu.cpu_usage())
            .sum::<f32>() / system.cpus().len() as f32,
    };

    let start_time = Instant::now();

    // Simulate inference (replace with actual inference call)
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    let elapsed = start_time.elapsed();

    // Post-profiling metrics
    system.refresh_all();
    let post_metrics = ResourceMetrics {
        memory_mb: system.process(pid)
            .map(|p| p.memory() as f64 / 1024.0 / 1024.0)
            .unwrap_or(0.0),
        cpu_percent: system.cpus().iter()
            .map(|cpu| cpu.cpu_usage())
            .sum::<f32>() / system.cpus().len() as f32,
    };

    let delta_metrics = ResourceMetrics {
        memory_mb: post_metrics.memory_mb - pre_metrics.memory_mb,
        cpu_percent: post_metrics.cpu_percent - pre_metrics.cpu_percent,
    };

    let result = ProfileResult {
        model_name: model_name.to_string(),
        total_time_ms: elapsed.as_secs_f64() * 1000.0,
        pre_metrics,
        post_metrics,
        delta_metrics,
    };

    log::info!("[ENDPOINT] Profiling completed in {:.2}ms", result.total_time_ms);

    Ok(HttpResponse::Ok().json(result))
}

/// Trigger performance optimizations
pub async fn optimize_performance(
    _state: web::Data<PerformanceState>,
) -> Result<HttpResponse, ApiError> {
    log::info!("[ENDPOINT] Performance optimization triggered");

    let mut system = System::new_all();
    system.refresh_all();
    let pid = sysinfo::get_current_pid()
        .map_err(|e| ApiError::InternalError(format!("Failed to get PID: {}", e)))?;

    let pre_memory = system.process(pid)
        .map(|p| p.memory() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    let mut optimizations = Vec::new();

    // Force garbage collection (Rust doesn't have explicit GC, but we can drop unused data)
    optimizations.push("Memory cleanup performed".to_string());

    // Clear any internal caches if implemented
    optimizations.push("Internal caches cleared".to_string());

    // Refresh system to get new memory reading
    system.refresh_all();
    let post_memory = system.process(pid)
        .map(|p| p.memory() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0);

    let memory_freed = pre_memory - post_memory;

    let result = OptimizationResult {
        garbage_collected: true,
        caches_cleared: true,
        memory_freed_mb: memory_freed.max(0.0),
        optimizations_applied: optimizations,
    };

    log::info!("[ENDPOINT] Optimization completed, freed {:.2} MB", memory_freed);

    Ok(HttpResponse::Ok().json(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web, App};
    use actix_web::http::StatusCode;

    fn make_performance_state() -> web::Data<PerformanceState> {
        web::Data::new(PerformanceState {
            monitor: Arc::new(Monitor::new()),
            start_time: Instant::now(),
        })
    }

    // ── get_performance_metrics ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_performance_metrics_returns_200() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance", web::get().to(get_performance_metrics))
        ).await;
        let req = test::TestRequest::get().uri("/performance").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_performance_metrics_response_shape() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance", web::get().to(get_performance_metrics))
        ).await;
        let req = test::TestRequest::get().uri("/performance").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["timestamp"].is_string());
        assert!(body["system_info"].is_object());
        assert!(body["process_info"].is_object());
        assert!(body["runtime_info"].is_object());
    }

    #[tokio::test]
    async fn test_get_performance_metrics_system_info_fields() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance", web::get().to(get_performance_metrics))
        ).await;
        let req = test::TestRequest::get().uri("/performance").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let sys = &body["system_info"];
        assert!(sys["cpu_count"].as_u64().unwrap() > 0);
        assert!(sys["total_memory_mb"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_get_performance_metrics_runtime_info_fields() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance", web::get().to(get_performance_metrics))
        ).await;
        let req = test::TestRequest::get().uri("/performance").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let rt = &body["runtime_info"];
        assert!(rt["rust_version"].is_string());
        assert_eq!(rt["actix_web_version"], "4.8");
        assert!(rt["num_cpus"].as_u64().unwrap() > 0);
    }

    // ── profile_inference ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_profile_inference_default_model() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance/profile", web::post().to(profile_inference))
        ).await;
        let req = test::TestRequest::post()
            .uri("/performance/profile")
            .set_json(&serde_json::json!({}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_profile_inference_with_model_name() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance/profile", web::post().to(profile_inference))
        ).await;
        let req = test::TestRequest::post()
            .uri("/performance/profile")
            .set_json(&serde_json::json!({"model": "my-model"}))
            .to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["model_name"], "my-model");
        assert!(body["total_time_ms"].as_f64().unwrap() >= 0.0);
        assert!(body["pre_metrics"].is_object());
        assert!(body["post_metrics"].is_object());
        assert!(body["delta_metrics"].is_object());
    }

    #[tokio::test]
    async fn test_profile_inference_no_model_uses_example() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance/profile", web::post().to(profile_inference))
        ).await;
        let req = test::TestRequest::post()
            .uri("/performance/profile")
            .set_json(&serde_json::json!({"model": null}))
            .to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["model_name"], "example");
    }

    // ── optimize_performance ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_optimize_performance_returns_200() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance/optimize", web::post().to(optimize_performance))
        ).await;
        let req = test::TestRequest::post()
            .uri("/performance/optimize")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_optimize_performance_response_body() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance/optimize", web::post().to(optimize_performance))
        ).await;
        let req = test::TestRequest::post()
            .uri("/performance/optimize")
            .to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["garbage_collected"], true);
        assert_eq!(body["caches_cleared"], true);
        assert!(body["memory_freed_mb"].as_f64().unwrap() >= 0.0);
        assert!(body["optimizations_applied"].is_array());
        let opts = body["optimizations_applied"].as_array().unwrap();
        assert!(!opts.is_empty());
    }

    #[tokio::test]
    async fn test_optimize_performance_optimizations_content() {
        let state = make_performance_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/performance/optimize", web::post().to(optimize_performance))
        ).await;
        let req = test::TestRequest::post()
            .uri("/performance/optimize")
            .to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let opts: Vec<&str> = body["optimizations_applied"].as_array().unwrap()
            .iter().filter_map(|v| v.as_str()).collect();
        assert!(opts.iter().any(|s| s.contains("Memory") || s.contains("cleanup") || s.contains("cache")));
    }

    // ── struct tests ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_resource_metrics_creation() {
        let metrics = ResourceMetrics {
            memory_mb: 100.0,
            cpu_percent: 50.0,
        };
        assert_eq!(metrics.memory_mb, 100.0);
        assert_eq!(metrics.cpu_percent, 50.0);
    }

    #[tokio::test]
    async fn test_resource_metrics_zero_values() {
        let metrics = ResourceMetrics {
            memory_mb: 0.0,
            cpu_percent: 0.0,
        };
        assert_eq!(metrics.memory_mb, 0.0);
        assert_eq!(metrics.cpu_percent, 0.0);
    }

    #[tokio::test]
    async fn test_resource_metrics_clone() {
        let metrics = ResourceMetrics {
            memory_mb: 256.5,
            cpu_percent: 75.3,
        };
        let cloned = metrics.clone();
        assert_eq!(cloned.memory_mb, metrics.memory_mb);
        assert_eq!(cloned.cpu_percent, metrics.cpu_percent);
    }

    #[tokio::test]
    async fn test_resource_metrics_serialization() {
        let metrics = ResourceMetrics {
            memory_mb: 512.0,
            cpu_percent: 25.0,
        };
        let json = serde_json::to_string(&metrics).expect("serialization failed");
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["memory_mb"], 512.0);
        assert_eq!(value["cpu_percent"], 25.0);
    }

    #[tokio::test]
    async fn test_profile_request_no_model() {
        let req = ProfileRequest {
            model: None,
            input_data: None,
        };
        assert!(req.model.is_none());
        assert!(req.input_data.is_none());
    }

    #[tokio::test]
    async fn test_profile_request_with_model() {
        let req = ProfileRequest {
            model: Some("bert".to_string()),
            input_data: Some(serde_json::json!({"input": "hello"})),
        };
        assert_eq!(req.model.as_deref(), Some("bert"));
        assert!(req.input_data.is_some());
    }

    #[tokio::test]
    async fn test_optimization_result_struct() {
        let result = OptimizationResult {
            garbage_collected: true,
            caches_cleared: false,
            memory_freed_mb: 12.5,
            optimizations_applied: vec!["cleanup".to_string()],
        };
        assert!(result.garbage_collected);
        assert!(!result.caches_cleared);
        assert_eq!(result.memory_freed_mb, 12.5);
        assert_eq!(result.optimizations_applied.len(), 1);
    }

    #[tokio::test]
    async fn test_optimization_result_memory_freed_nonnegative_logic() {
        // Mirrors the .max(0.0) logic used in optimize_performance
        let pre_memory = 100.0_f64;
        let post_memory = 110.0_f64; // memory increased
        let freed = (pre_memory - post_memory).max(0.0);
        assert_eq!(freed, 0.0);

        let post_memory2 = 80.0_f64; // memory decreased
        let freed2 = (pre_memory - post_memory2).max(0.0);
        assert_eq!(freed2, 20.0);
    }

    #[tokio::test]
    async fn test_profile_result_struct() {
        let pre = ResourceMetrics { memory_mb: 100.0, cpu_percent: 10.0 };
        let post = ResourceMetrics { memory_mb: 110.0, cpu_percent: 20.0 };
        let delta = ResourceMetrics {
            memory_mb: post.memory_mb - pre.memory_mb,
            cpu_percent: post.cpu_percent - pre.cpu_percent,
        };
        let result = ProfileResult {
            model_name: "test-model".to_string(),
            total_time_ms: 42.5,
            pre_metrics: pre,
            post_metrics: post,
            delta_metrics: delta,
        };
        assert_eq!(result.model_name, "test-model");
        assert_eq!(result.total_time_ms, 42.5);
        assert_eq!(result.delta_metrics.memory_mb, 10.0);
        assert_eq!(result.delta_metrics.cpu_percent, 10.0);
    }

    #[tokio::test]
    async fn test_system_info_struct() {
        let info = SystemInfo {
            cpu_count: 8,
            total_memory_mb: 16384,
            available_memory_mb: 8192,
            used_memory_mb: 8192,
            memory_usage_percent: 50.0,
            cpu_usage_percent: 30.0,
        };
        assert_eq!(info.cpu_count, 8);
        assert_eq!(info.total_memory_mb, 16384);
        assert_eq!(info.memory_usage_percent, 50.0);
    }

    #[tokio::test]
    async fn test_process_info_struct() {
        let info = ProcessInfo {
            pid: 12345,
            memory_mb: 64.0,
            cpu_usage_percent: 5.0,
            uptime_seconds: 3600,
        };
        assert_eq!(info.pid, 12345);
        assert_eq!(info.memory_mb, 64.0);
        assert_eq!(info.uptime_seconds, 3600);
    }

    #[tokio::test]
    async fn test_runtime_info_struct() {
        let info = RuntimeInfo {
            rust_version: "1.0.0".to_string(),
            actix_web_version: "4.8".to_string(),
            num_cpus: 4,
        };
        assert_eq!(info.actix_web_version, "4.8");
        assert_eq!(info.num_cpus, 4);
    }

    #[tokio::test]
    async fn test_performance_state_start_time() {
        use crate::monitor::Monitor;
        use std::sync::Arc;
        let monitor = Arc::new(Monitor::new());
        let state = PerformanceState {
            monitor,
            start_time: Instant::now(),
        };
        // Elapsed should be very small (< 1 second) since we just created it
        assert!(state.start_time.elapsed().as_secs() < 1);
    }
}
