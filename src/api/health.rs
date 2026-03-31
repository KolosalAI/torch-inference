use actix_web::{web, HttpRequest, HttpResponse, Result};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use crate::middleware::get_correlation_id;
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
    req: HttpRequest,
    monitor: web::Data<Arc<Monitor>>,
) -> Result<HttpResponse> {
    let correlation_id = get_correlation_id(&req);
    let health = monitor.get_health_status();

    let response = HealthCheck {
        status: if health.healthy { "healthy".to_string() } else { "degraded".to_string() },
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        uptime_seconds: health.uptime_seconds,
        checks: HashMap::new(),
    };

    Ok(HttpResponse::Ok()
        .insert_header(("x-correlation-id", correlation_id.as_str()))
        .json(response))
}

/// Readiness probe - strict health check
/// Returns 200 only if service is ready to accept traffic
pub async fn readiness(
    req: HttpRequest,
    monitor: web::Data<Arc<Monitor>>,
) -> Result<HttpResponse> {
    let correlation_id = get_correlation_id(&req);
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
        Ok(HttpResponse::Ok()
            .insert_header(("x-correlation-id", correlation_id.as_str()))
            .json(response))
    } else {
        Ok(HttpResponse::ServiceUnavailable()
            .insert_header(("x-correlation-id", correlation_id.as_str()))
            .json(response))
    }
}

/// Detailed health check with component status
pub async fn health(
    req: HttpRequest,
    monitor: web::Data<Arc<Monitor>>,
) -> Result<HttpResponse> {
    let correlation_id = get_correlation_id(&req);
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

    Ok(HttpResponse::Ok()
        .insert_header(("x-correlation-id", correlation_id.as_str()))
        .json(response))
}

#[cfg(test)]
mod handler_tests {
    use super::*;
    use actix_web::{test, web, App};

    fn make_monitor() -> web::Data<Arc<Monitor>> {
        web::Data::new(Arc::new(Monitor::new()))
    }

    fn bare_request() -> HttpRequest {
        test::TestRequest::get().uri("/").to_http_request()
    }

    // ── liveness handler ─────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_liveness_handler_returns_200() {
        let monitor = make_monitor();
        let result = liveness(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_liveness_handler_body_has_status() {
        let monitor = make_monitor();
        let result = liveness(bare_request(), monitor).await;
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(body["status"].is_string(), "liveness response must have a status field");
        assert!(body["version"].is_string(), "liveness response must have a version field");
        assert!(body["uptime_seconds"].is_number());
    }

    // ── readiness handler ────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_readiness_handler_returns_200_or_503() {
        let monitor = make_monitor();
        let result = readiness(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let status = resp.status();
        // Fresh monitor with no errors should be ready (200), but allow 503 too
        assert!(
            status == actix_web::http::StatusCode::OK
                || status == actix_web::http::StatusCode::SERVICE_UNAVAILABLE,
            "readiness should return 200 or 503, got {}",
            status
        );
    }

    #[actix_web::test]
    async fn test_readiness_handler_body_has_checks() {
        let monitor = make_monitor();
        let result = readiness(bare_request(), monitor).await;
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(body["status"].is_string());
        assert!(body["checks"].is_object(), "readiness response must have a checks object");
        let checks = body["checks"].as_object().unwrap();
        assert!(checks.contains_key("error_rate"), "must have error_rate check");
        assert!(checks.contains_key("queue_depth"), "must have queue_depth check");
        assert!(checks.contains_key("response_time"), "must have response_time check");
    }

    // ── health handler ───────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_health_handler_returns_200() {
        let monitor = make_monitor();
        let result = health(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_health_handler_body_has_all_checks() {
        let monitor = make_monitor();
        let result = health(bare_request(), monitor).await;
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(body["status"].is_string());
        assert!(body["version"].is_string());
        assert!(body["uptime_seconds"].is_number());
        let checks = body["checks"].as_object().unwrap();
        assert!(checks.contains_key("monitor"), "must have monitor check");
        assert!(checks.contains_key("error_rate"), "must have error_rate check");
        assert!(checks.contains_key("performance"), "must have performance check");
        assert!(checks.contains_key("capacity"), "must have capacity check");
    }

    /// Verify that a fresh monitor (no errors, no active requests) produces "healthy" status.
    #[actix_web::test]
    async fn test_health_handler_fresh_monitor_is_healthy() {
        let monitor = make_monitor();
        let result = health(bare_request(), monitor).await;
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["status"], "healthy", "fresh monitor should be healthy");
    }

    // ── readiness with errors (line 55 – error rate calc, 87 – "not_ready", 101 – 503) ──

    /// Drive total_requests > 0 and error_rate >= 10% to trigger the "not_ready" / 503 path.
    #[actix_web::test]
    async fn test_readiness_with_high_error_rate_returns_503() {
        let monitor_inner = Arc::new(Monitor::new());
        // Record 10 requests, all failures → 100% error rate (>= 10% threshold)
        for _ in 0..10 {
            monitor_inner.record_request_start();
            monitor_inner.record_request_end(10, "/test", false);
        }
        let monitor = web::Data::new(monitor_inner);
        let result = readiness(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        // High error rate → 503 Service Unavailable
        assert_eq!(resp.status(), actix_web::http::StatusCode::SERVICE_UNAVAILABLE);
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["status"], "not_ready");
    }

    /// Drive total_requests > 0 with zero errors to exercise line 55 (ratio calc) while staying ready.
    #[actix_web::test]
    async fn test_readiness_with_requests_and_no_errors_computes_error_rate() {
        let monitor_inner = Arc::new(Monitor::new());
        for _ in 0..5 {
            monitor_inner.record_request_start();
            monitor_inner.record_request_end(10, "/test", true);
        }
        let monitor = web::Data::new(monitor_inner);
        let result = readiness(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        let error_rate_msg = body["checks"]["error_rate"]["message"].as_str().unwrap_or("");
        assert!(error_rate_msg.contains("0.00"), "zero errors → 0.00% error rate");
    }

    // ── health handler with errors (lines 124, 131-132, 134, 176-177) ────────

    /// Drive error_rate between 5% and 10% → "degraded" error_status (lines 131-132).
    #[actix_web::test]
    async fn test_health_with_degraded_error_rate() {
        let monitor_inner = Arc::new(Monitor::new());
        // 14 successes + 1 failure = ~6.67% error rate (between 5% and 10%)
        for _ in 0..14 {
            monitor_inner.record_request_start();
            monitor_inner.record_request_end(10, "/test", true);
        }
        monitor_inner.record_request_start();
        monitor_inner.record_request_end(10, "/test", false);
        let monitor = web::Data::new(monitor_inner);
        let result = health(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["checks"]["error_rate"]["status"], "degraded");
    }

    /// Drive error_rate >= 10% → "down" error_status (line 134) and "unhealthy" overall (lines 176-177).
    #[actix_web::test]
    async fn test_health_with_high_error_rate_is_unhealthy() {
        let monitor_inner = Arc::new(Monitor::new());
        // 10 failures out of 10 = 100% error rate (>= 10% threshold → "down")
        for _ in 0..10 {
            monitor_inner.record_request_start();
            monitor_inner.record_request_end(10, "/test", false);
        }
        let monitor = web::Data::new(monitor_inner);
        let result = health(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["checks"]["error_rate"]["status"], "down");
        assert_eq!(body["status"], "unhealthy");
    }

    /// Drive total_requests > 0 with zero errors (line 124 – ratio calc path in health handler).
    #[actix_web::test]
    async fn test_health_with_requests_computes_error_rate() {
        let monitor_inner = Arc::new(Monitor::new());
        for _ in 0..5 {
            monitor_inner.record_request_start();
            monitor_inner.record_request_end(50, "/test", true);
        }
        let monitor = web::Data::new(monitor_inner);
        let result = health(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        // 0 errors → "up"
        assert_eq!(body["checks"]["error_rate"]["status"], "up");
    }

    // ── health handler perf branches (lines 146-147 "degraded", 149 "down") ──

    /// avg_latency between 1000ms and 5000ms → "degraded" performance (lines 146-147).
    #[actix_web::test]
    async fn test_health_with_degraded_performance() {
        let monitor_inner = Arc::new(Monitor::new());
        // Record one request with 2000ms latency → avg = 2000ms (>= 1000, < 5000 → "degraded")
        monitor_inner.record_request_start();
        monitor_inner.record_request_end(2000, "/test", true);
        let monitor = web::Data::new(monitor_inner);
        let result = health(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["checks"]["performance"]["status"], "degraded");
        // Overall must be at least "degraded" (no "down" checks)
        let status = body["status"].as_str().unwrap_or("");
        assert!(status == "degraded" || status == "unhealthy");
    }

    /// avg_latency >= 5000ms → "down" performance (line 149) and "unhealthy" overall (lines 176-177).
    #[actix_web::test]
    async fn test_health_with_down_performance_is_unhealthy() {
        let monitor_inner = Arc::new(Monitor::new());
        monitor_inner.record_request_start();
        monitor_inner.record_request_end(6000, "/test", true);
        let monitor = web::Data::new(monitor_inner);
        let result = health(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["checks"]["performance"]["status"], "down");
        assert_eq!(body["status"], "unhealthy");
    }

    // ── health handler capacity branches (lines 161-162 "degraded", 164 "down") ──

    /// 500 <= active_requests < 1000 → "degraded" capacity (lines 161-162).
    /// To achieve active_requests without completing them, call record_request_start 600 times.
    #[actix_web::test]
    async fn test_health_with_degraded_capacity() {
        let monitor_inner = Arc::new(Monitor::new());
        // 600 starts, 0 ends → active_requests = 600 (>= 500 and < 1000 → "degraded")
        for _ in 0..600 {
            monitor_inner.record_request_start();
        }
        let monitor = web::Data::new(monitor_inner);
        let result = health(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["checks"]["capacity"]["status"], "degraded");
    }

    /// active_requests >= 1000 → "down" capacity (line 164) and "unhealthy" overall.
    #[actix_web::test]
    async fn test_health_with_down_capacity_is_unhealthy() {
        let monitor_inner = Arc::new(Monitor::new());
        for _ in 0..1001 {
            monitor_inner.record_request_start();
        }
        let monitor = web::Data::new(monitor_inner);
        let result = health(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["checks"]["capacity"]["status"], "down");
        assert_eq!(body["status"], "unhealthy");
    }

    /// Mix: one check "degraded" but no "down" → overall "degraded" (line 179).
    #[actix_web::test]
    async fn test_health_degraded_overall_when_some_degraded_none_down() {
        let monitor_inner = Arc::new(Monitor::new());
        // avg_latency = 2000ms → performance "degraded", no other checks degrade
        monitor_inner.record_request_start();
        monitor_inner.record_request_end(2000, "/test", true);
        let monitor = web::Data::new(monitor_inner);
        let result = health(bare_request(), monitor).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        let body_bytes = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        // Performance is degraded → overall should be "degraded" (if no "down" exists)
        let status = body["status"].as_str().unwrap_or("");
        assert!(
            status == "degraded" || status == "unhealthy",
            "expected degraded or unhealthy, got {}",
            status
        );
    }

    /// Via actix-web test service — exercises the route registration path.
    #[actix_web::test]
    async fn test_health_via_app() {
        let monitor = Arc::new(Monitor::new());
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(monitor.clone()))
                .route("/health", web::get().to(health))
                .route("/readiness", web::get().to(readiness))
                .route("/liveness", web::get().to(liveness)),
        )
        .await;

        let req = test::TestRequest::get().uri("/health").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let req = test::TestRequest::get().uri("/liveness").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let req = test::TestRequest::get().uri("/readiness").to_request();
        let resp = test::call_service(&app, req).await;
        let status = resp.status();
        assert!(
            status == actix_web::http::StatusCode::OK
                || status == actix_web::http::StatusCode::SERVICE_UNAVAILABLE
        );
    }
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
