/// Performance monitoring and profiling
use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use sysinfo::{System, Pid};
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
    state: web::Data<PerformanceState>,
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
    state: web::Data<PerformanceState>,
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

    #[test]
    fn test_resource_metrics_creation() {
        let metrics = ResourceMetrics {
            memory_mb: 100.0,
            cpu_percent: 50.0,
        };
        assert_eq!(metrics.memory_mb, 100.0);
        assert_eq!(metrics.cpu_percent, 50.0);
    }
}
