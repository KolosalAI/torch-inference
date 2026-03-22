use actix_web::{web, HttpResponse, Result};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use crate::monitor::Monitor;

#[derive(Serialize, Deserialize)]
pub struct HealthCheck {
    pub status: String,
    pub version: String,
    pub timestamp: String,
    pub uptime_seconds: u64,
    pub checks: HashMap<String, ComponentHealth>,
}

#[derive(Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: String,
    pub message: Option<String>,
    pub latency_ms: u64,
}

/// Liveness probe - basic health check
/// Returns 200 if service is alive (even if degraded)
pub async fn liveness(
    monitor: web::Data<Arc<Monitor>>,
) -> Result<HttpResponse> {
    let health = monitor.get_health_status();
    
    let response = HealthCheck {
        status: if health.healthy { "healthy".to_string() } else { "degraded".to_string() },
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        uptime_seconds: health.uptime_seconds,
        checks: HashMap::new(),
    };
    
    Ok(HttpResponse::Ok().json(response))
}

/// Readiness probe - strict health check
/// Returns 200 only if service is ready to accept traffic
pub async fn readiness(
    monitor: web::Data<Arc<Monitor>>,
) -> Result<HttpResponse> {
    let start = Instant::now();
    let metrics = monitor.get_metrics();
    let health = monitor.get_health_status();
    
    let mut checks = HashMap::new();
    
    // Check error rate
    let error_rate = if metrics.total_requests > 0 {
        (metrics.total_errors as f64 / metrics.total_requests as f64) * 100.0
    } else {
        0.0
    };
    
    let error_status = if error_rate < 10.0 { "up" } else { "down" };
    checks.insert("error_rate".to_string(), ComponentHealth {
        status: error_status.to_string(),
        message: Some(format!("{:.2}%", error_rate)),
        latency_ms: 0,
    });
    
    // Check active requests (queue depth)
    let queue_status = if health.active_requests < 1000 { "up" } else { "degraded" };
    checks.insert("queue_depth".to_string(), ComponentHealth {
        status: queue_status.to_string(),
        message: Some(format!("{} active", health.active_requests)),
        latency_ms: 0,
    });
    
    // Check response time
    let latency_status = if health.response_time_ms < 5000.0 { "up" } else { "degraded" };
    checks.insert("response_time".to_string(), ComponentHealth {
        status: latency_status.to_string(),
        message: Some(format!("{:.2}ms avg", health.response_time_ms)),
        latency_ms: start.elapsed().as_millis() as u64,
    });
    
    // Overall status - must be ALL up for ready
    let overall_status = if checks.values().all(|c| c.status == "up") {
        "ready"
    } else {
        "not_ready"
    };
    
    let response = HealthCheck {
        status: overall_status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        uptime_seconds: health.uptime_seconds,
        checks,
    };
    
    if overall_status == "ready" {
        Ok(HttpResponse::Ok().json(response))
    } else {
        Ok(HttpResponse::ServiceUnavailable().json(response))
    }
}

/// Detailed health check with component status
pub async fn health(
    monitor: web::Data<Arc<Monitor>>,
) -> Result<HttpResponse> {
    let start = Instant::now();
    let metrics = monitor.get_metrics();
    let health = monitor.get_health_status();
    
    let mut checks = HashMap::new();
    
    // Monitor health
    checks.insert("monitor".to_string(), ComponentHealth {
        status: "up".to_string(),
        message: Some(format!("{} total requests", metrics.total_requests)),
        latency_ms: 0,
    });
    
    // Error rate
    let error_rate = if metrics.total_requests > 0 {
        (metrics.total_errors as f64 / metrics.total_requests as f64) * 100.0
    } else {
        0.0
    };
    
    let error_status = if error_rate < 5.0 { 
        "up" 
    } else if error_rate < 10.0 { 
        "degraded" 
    } else { 
        "down" 
    };
    
    checks.insert("error_rate".to_string(), ComponentHealth {
        status: error_status.to_string(),
        message: Some(format!("{:.2}%", error_rate)),
        latency_ms: 0,
    });
    
    // Performance
    let perf_status = if health.response_time_ms < 1000.0 {
        "up"
    } else if health.response_time_ms < 5000.0 {
        "degraded"
    } else {
        "down"
    };
    
    checks.insert("performance".to_string(), ComponentHealth {
        status: perf_status.to_string(),
        message: Some(format!("{:.2}ms avg latency", health.response_time_ms)),
        latency_ms: start.elapsed().as_millis() as u64,
    });
    
    // Capacity
    let capacity_status = if health.active_requests < 500 {
        "up"
    } else if health.active_requests < 1000 {
        "degraded"
    } else {
        "down"
    };
    
    checks.insert("capacity".to_string(), ComponentHealth {
        status: capacity_status.to_string(),
        message: Some(format!("{} active requests", health.active_requests)),
        latency_ms: 0,
    });
    
    // Overall status
    let overall_status = if checks.values().all(|c| c.status == "up") {
        "healthy"
    } else if checks.values().any(|c| c.status == "down") {
        "unhealthy"
    } else {
        "degraded"
    };
    
    let response = HealthCheck {
        status: overall_status.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        uptime_seconds: health.uptime_seconds,
        checks,
    };
    
    Ok(HttpResponse::Ok().json(response))
}
