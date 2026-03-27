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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_health_check_struct_construction() {
        let hc = HealthCheck {
            status: "healthy".to_string(),
            version: "1.0.0".to_string(),
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            uptime_seconds: 3600,
            checks: HashMap::new(),
        };
        assert_eq!(hc.status, "healthy");
        assert_eq!(hc.version, "1.0.0");
        assert_eq!(hc.uptime_seconds, 3600);
        assert!(hc.checks.is_empty());
    }

    #[test]
    fn test_component_health_with_message() {
        let ch = ComponentHealth {
            status: "up".to_string(),
            message: Some("all good".to_string()),
            latency_ms: 42,
        };
        assert_eq!(ch.status, "up");
        assert_eq!(ch.message, Some("all good".to_string()));
        assert_eq!(ch.latency_ms, 42);
    }

    #[test]
    fn test_component_health_no_message() {
        let ch = ComponentHealth {
            status: "down".to_string(),
            message: None,
            latency_ms: 0,
        };
        assert_eq!(ch.status, "down");
        assert!(ch.message.is_none());
    }

    #[test]
    fn test_health_check_serde_roundtrip() {
        let mut checks = HashMap::new();
        checks.insert("db".to_string(), ComponentHealth {
            status: "up".to_string(),
            message: None,
            latency_ms: 5,
        });
        let hc = HealthCheck {
            status: "healthy".to_string(),
            version: "2.0.0".to_string(),
            timestamp: "2026-03-27T00:00:00Z".to_string(),
            uptime_seconds: 100,
            checks,
        };
        let json = serde_json::to_string(&hc).expect("serialization failed");
        let deserialized: HealthCheck = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(deserialized.status, "healthy");
        assert_eq!(deserialized.uptime_seconds, 100);
        assert!(deserialized.checks.contains_key("db"));
        assert_eq!(deserialized.checks["db"].status, "up");
        assert_eq!(deserialized.checks["db"].latency_ms, 5);
    }

    #[test]
    fn test_component_health_serde_roundtrip() {
        let ch = ComponentHealth {
            status: "degraded".to_string(),
            message: Some("50% error rate".to_string()),
            latency_ms: 1234,
        };
        let json = serde_json::to_string(&ch).expect("serialization failed");
        let deserialized: ComponentHealth =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(deserialized.status, "degraded");
        assert_eq!(deserialized.message, Some("50% error rate".to_string()));
        assert_eq!(deserialized.latency_ms, 1234);
    }

    #[test]
    fn test_health_check_various_statuses() {
        for status in &["healthy", "degraded", "unhealthy", "ready", "not_ready"] {
            let hc = HealthCheck {
                status: status.to_string(),
                version: "1.0.0".to_string(),
                timestamp: "2026-01-01T00:00:00Z".to_string(),
                uptime_seconds: 0,
                checks: HashMap::new(),
            };
            let json = serde_json::to_string(&hc).unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed["status"], *status);
        }
    }

    #[test]
    fn test_health_check_multiple_component_checks() {
        let mut checks = HashMap::new();
        for (name, status) in &[
            ("error_rate", "up"),
            ("queue_depth", "degraded"),
            ("response_time", "up"),
        ] {
            checks.insert(
                name.to_string(),
                ComponentHealth {
                    status: status.to_string(),
                    message: Some(format!("{} check", name)),
                    latency_ms: 1,
                },
            );
        }
        // Not all are "up" — queue_depth is degraded
        let all_up = checks.values().all(|c| c.status == "up");
        assert!(!all_up);
        // None are "down"
        let any_down = checks.values().any(|c| c.status == "down");
        assert!(!any_down);
    }
}
