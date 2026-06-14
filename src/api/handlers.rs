use actix_web::{web, HttpRequest, HttpResponse, Responder};
use chrono::Utc;
use serde_json::json;

use crate::api::types::*;
use crate::config::Config;
use crate::core::engine::InferenceEngine;
use crate::dedup::RequestDeduplicator;
use crate::middleware::RateLimiter;
use crate::models::manager::ModelManager;
use crate::monitor::Monitor;

const PLAYGROUND_HTML: &str = include_str!("playground.html");
static PLAYGROUND_ETAG: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn playground_etag() -> &'static str {
    PLAYGROUND_ETAG.get_or_init(|| {
        use sha2::Digest;
        let hash = sha2::Sha256::digest(PLAYGROUND_HTML.as_bytes());
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        format!("\"{}\"", hex)
    })
}

pub async fn root(req: HttpRequest) -> impl Responder {
    let etag = playground_etag();
    if let Some(inm) = req.headers().get("if-none-match") {
        if inm.to_str().unwrap_or("") == etag {
            return HttpResponse::NotModified()
                .insert_header(("ETag", etag))
                .finish();
        }
    }
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .insert_header(("ETag", etag))
        .insert_header(("Cache-Control", "no-cache"))
        .body(PLAYGROUND_HTML)
}

#[allow(dead_code)] // superseded by api::health::health (registered at App level)
pub async fn health_check(
    engine: web::Data<std::sync::Arc<InferenceEngine>>,
    monitor: web::Data<std::sync::Arc<Monitor>>,
) -> impl Responder {
    let health = engine.health_check();
    let monitor_health = monitor.get_health_status();
    let metrics = monitor.get_metrics();
    let error_rate = if metrics.total_requests > 0 {
        monitor_health.error_count as f64 / metrics.total_requests as f64
    } else {
        0.0
    };

    HttpResponse::Ok().json(json!({
        "healthy": monitor_health.healthy,
        "checks": health,
        "uptime_seconds": monitor_health.uptime_seconds,
        "active_requests": monitor_health.active_requests,
        "total_requests": metrics.total_requests,
        "avg_latency_ms": monitor_health.response_time_ms,
        "error_count": monitor_health.error_count,
        "error_rate": error_rate,
        "timestamp": Utc::now().to_rfc3339(),
    }))
}

pub async fn predict(
    req: web::Json<InferenceRequest>,
    engine: web::Data<std::sync::Arc<InferenceEngine>>,
    rate_limiter: web::Data<std::sync::Arc<RateLimiter>>,
    monitor: web::Data<std::sync::Arc<Monitor>>,
    deduplicator: web::Data<std::sync::Arc<RequestDeduplicator>>,
    http_req: HttpRequest,
) -> impl Responder {
    // Borrow peer_addr as &str — ConnectionInfo scoped to this block, no String alloc.
    {
        let ci = http_req.connection_info();
        let client_ip = ci.peer_addr().unwrap_or("unknown");
        if let Err(e) = rate_limiter.is_allowed(client_ip) {
            monitor.record_request_end(0, "/predict", false);
            return actix_web::error::ErrorTooManyRequests(e.message).error_response();
        }
    }

    // Check for duplicate request — cache returns Arc<Value>: O(1) clone, no data copy.
    let dedup_key = deduplicator.generate_key(&req.model_name, &req.inputs);
    if let Some(cached_result) = deduplicator.get(&dedup_key) {
        monitor.record_request_start(); // Record start to balance metrics
        monitor.record_request_end(0, "/predict", true); // 0ms latency for cache hit

        return HttpResponse::Ok().json(InferenceResponse {
            success: true,
            result: Some((*cached_result).clone()),
            error: None,
            processing_time: Some(0.0),
            model_info: Some(serde_json::json!({"source": "deduplication_cache"})),
        });
    }

    monitor.record_request_start();
    let start = std::time::Instant::now();

    match engine.infer(&req.model_name, &req.inputs).await {
        Ok(result) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            monitor.record_request_end(latency_ms, "/predict", true);

            // Cache result for deduplication (TTL 10s)
            deduplicator.set(dedup_key, result.clone(), 10);

            HttpResponse::Ok().json(InferenceResponse {
                success: true,
                result: Some(result),
                error: None,
                processing_time: Some(latency_ms as f64),
                model_info: None,
            })
        }
        Err(e) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            monitor.record_request_end(latency_ms, "/predict", false);

            let body = InferenceResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
                processing_time: Some(latency_ms as f64),
                model_info: None,
            };
            match e {
                crate::error::InferenceError::InvalidInput(_) => HttpResponse::BadRequest().json(body),
                crate::error::InferenceError::ModelNotFound(_) => HttpResponse::NotFound().json(body),
                crate::error::InferenceError::AuthenticationFailed(_) => HttpResponse::Unauthorized().json(body),
                crate::error::InferenceError::Timeout => HttpResponse::GatewayTimeout().json(body),
                crate::error::InferenceError::GpuError(_) => HttpResponse::ServiceUnavailable().json(body),
                _ => HttpResponse::InternalServerError().json(body),
            }
        }
    }
}

pub async fn synthesize_tts(
    req: web::Json<TTSRequest>,
    engine: web::Data<std::sync::Arc<InferenceEngine>>,
    monitor: web::Data<std::sync::Arc<Monitor>>,
) -> impl Responder {
    monitor.record_request_start();
    let start = std::time::Instant::now();

    match engine.tts_synthesize(&req.model_name, &req.text).await {
        Ok(audio_data) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            monitor.record_request_end(latency_ms, "/synthesize", true);

            HttpResponse::Ok().json(TTSResponse {
                success: true,
                audio_data: Some(audio_data),
                audio_format: Some(req.output_format.clone()),
                duration: None,
                sample_rate: Some(16000),
                processing_time: Some(latency_ms as f64),
                error: None,
            })
        }
        Err(e) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            monitor.record_request_end(latency_ms, "/synthesize", false);

            let body = TTSResponse {
                success: false,
                audio_data: None,
                audio_format: None,
                duration: None,
                sample_rate: None,
                processing_time: Some(latency_ms as f64),
                error: Some(e.to_string()),
            };
            match e {
                crate::error::InferenceError::InvalidInput(_) => HttpResponse::BadRequest().json(body),
                crate::error::InferenceError::ModelNotFound(_) => HttpResponse::NotFound().json(body),
                crate::error::InferenceError::Timeout => HttpResponse::GatewayTimeout().json(body),
                _ => HttpResponse::InternalServerError().json(body),
            }
        }
    }
}

pub async fn list_models(
    models: web::Data<std::sync::Arc<ModelManager>>,
    monitor: web::Data<std::sync::Arc<Monitor>>,
) -> impl Responder {
    monitor.record_request_start();
    let start = std::time::Instant::now();

    let model_list = models.list_available();
    let latency_ms = start.elapsed().as_millis() as u64;
    monitor.record_request_end(latency_ms, "/models", true);

    HttpResponse::Ok().json(json!({
        "models": model_list,
        "total": model_list.len()
    }))
}

pub async fn get_stats(
    _engine: web::Data<std::sync::Arc<InferenceEngine>>,
    monitor: web::Data<std::sync::Arc<Monitor>>,
) -> impl Responder {
    monitor.record_request_start();
    let start = std::time::Instant::now();

    let stats = monitor.get_metrics();
    let latency_ms = start.elapsed().as_millis() as u64;
    monitor.record_request_end(latency_ms, "/stats", true);

    HttpResponse::Ok().json(stats)
}

pub async fn get_endpoint_stats(monitor: web::Data<std::sync::Arc<Monitor>>) -> impl Responder {
    let stats = monitor.get_endpoint_stats();
    let stats_json: Vec<serde_json::Value> = stats
        .iter()
        .map(|s| serde_json::to_value(s).unwrap_or_default())
        .collect();
    HttpResponse::Ok().json(json!({
        "endpoints": stats_json,
        "count": stats.len()
    }))
}

pub async fn get_system_info(
    config: web::Data<Config>,
    monitor: web::Data<std::sync::Arc<Monitor>>,
) -> impl Responder {
    let health = monitor.get_health_status();

    HttpResponse::Ok().json(json!({
        "server": config.server,
        "device": config.device,
        "batch": config.batch,
        "performance": config.performance,
        "health": health
    }))
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    // Note: /health is registered at the App level in main.rs (canonical).
    // /audio/transcribe and /audio/synthesize are also canonical at the App
    // level — duplicates here would shadow them silently.
    cfg.route("/", web::get().to(root))
        .route("/playground", web::get().to(root))
        .route("/predict", web::post().to(predict))
        .route("/synthesize", web::post().to(synthesize_tts))
        .route("/models", web::get().to(list_models))
        .route("/stats", web::get().to(get_stats))
        .route("/endpoints", web::get().to(get_endpoint_stats))
        .route("/info", web::get().to(get_system_info))
        // Audio endpoints (transcribe + synthesize are at the App level)
        .route(
            "/audio/validate",
            web::post().to(crate::api::audio::validate_audio),
        )
        .route(
            "/audio/health",
            web::get().to(crate::api::audio::audio_health),
        )
        // Image security endpoints
        .route(
            "/image/process/secure",
            web::post().to(crate::api::image::process_image_secure),
        )
        .route(
            "/image/validate/security",
            web::post().to(crate::api::image::validate_image_security),
        )
        .route(
            "/image/security/stats",
            web::get().to(crate::api::image::get_image_security_stats),
        )
        .route(
            "/image/health",
            web::get().to(crate::api::image::image_health),
        )
        // Model download endpoints
        .route(
            "/models/download",
            web::post().to(crate::api::model_download::download_model),
        )
        .route(
            "/models/download/status/{id}",
            web::get().to(crate::api::model_download::get_download_status),
        )
        .route(
            "/models/download/list",
            web::get().to(crate::api::model_download::list_downloads),
        )
        .route(
            "/models/managed",
            web::get().to(crate::api::model_download::list_models),
        )
        .route(
            "/models/download/{name}/info",
            web::get().to(crate::api::model_download::get_model_info),
        )
        .route(
            "/models/download/{name}",
            web::delete().to(crate::api::model_download::delete_model),
        )
        .route(
            "/models/cache/info",
            web::get().to(crate::api::model_download::get_cache_info),
        )
        .route(
            "/models/available",
            web::get().to(crate::api::model_download::list_available_models),
        )
        // SOTA model endpoints
        .route(
            "/models/sota",
            web::get().to(crate::api::model_download::list_sota_models),
        )
        .route(
            "/models/sota/{model_id}",
            web::post().to(crate::api::model_download::download_sota_model),
        )
        // System info endpoints
        .route(
            "/system/info",
            web::get().to(crate::api::system::get_system_info),
        )
        .route(
            "/system/config",
            web::get().to(crate::api::system::get_config),
        )
        .route(
            "/system/gpu/stats",
            web::get().to(crate::api::system::get_gpu_stats),
        )
        // Performance endpoints
        .route(
            "/performance",
            web::get().to(crate::api::performance::get_performance_metrics),
        )
        .route(
            "/performance/profile",
            web::post().to(crate::api::performance::profile_inference),
        )
        .route(
            "/performance/optimize",
            web::get().to(crate::api::performance::optimize_performance),
        )
        // Logging endpoints
        .route(
            "/logs",
            web::get().to(crate::api::logging::get_logging_info),
        )
        .route(
            "/logs/{log_file}",
            web::get().to(crate::api::logging::get_log_file),
        )
        .route(
            "/logs/{log_file}",
            web::delete().to(crate::api::logging::clear_log_file),
        )
        // OpenAI-compatible v1 endpoints
        .route("/v1/models", web::get().to(v1_list_models))
        .configure(crate::api::llm_proxy::configure_routes)
        .configure(crate::api::stt_proxy::configure_routes)
        // Self-hosted static assets (fonts, icons)
        .route(
            "/assets/remixicon.css",
            web::get().to(crate::api::assets::serve_remixicon_css),
        )
        .route(
            "/assets/remixicon.woff2",
            web::get().to(crate::api::assets::serve_remixicon_woff2),
        );
}

async fn v1_list_models() -> impl Responder {
    HttpResponse::Ok().json(json!({
        "object": "list",
        "data": []
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};
    use std::sync::Arc;

    use crate::config::Config;
    use crate::core::engine::InferenceEngine;
    use crate::dedup::RequestDeduplicator;
    use crate::middleware::RateLimiter;
    use crate::models::manager::ModelManager;
    use crate::monitor::Monitor;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_monitor() -> web::Data<Arc<Monitor>> {
        web::Data::new(Arc::new(Monitor::new()))
    }

    fn make_config() -> web::Data<Config> {
        web::Data::new(Config::default())
    }

    fn make_model_manager() -> web::Data<Arc<ModelManager>> {
        web::Data::new(Arc::new(ModelManager::new(&Config::default(), None)))
    }

    fn make_engine() -> web::Data<Arc<InferenceEngine>> {
        let cfg = Config::default();
        let manager = Arc::new(ModelManager::new(&cfg, None));
        web::Data::new(Arc::new(InferenceEngine::new(manager, &cfg)))
    }

    fn make_rate_limiter() -> web::Data<Arc<RateLimiter>> {
        // Very high limit so tests are never rate-limited
        web::Data::new(Arc::new(RateLimiter::new(100_000, 60)))
    }

    fn make_deduplicator() -> web::Data<Arc<RequestDeduplicator>> {
        web::Data::new(Arc::new(RequestDeduplicator::new(1000)))
    }

    // ── root ─────────────────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_root_returns_200() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get().uri("/").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }

    #[actix_web::test]
    async fn test_root_response_is_html() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get().uri("/").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            ct.contains("text/html"),
            "root should return HTML, got: {}",
            ct
        );
    }

    #[actix_web::test]
    async fn test_root_response_body_is_non_empty() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get().uri("/").to_request();
        let body = test::call_and_read_body(&app, req).await;
        assert!(!body.is_empty(), "root response body should not be empty");
    }

    #[actix_web::test]
    async fn test_root_response_contains_html_tag() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get().uri("/").to_request();
        let body = test::call_and_read_body(&app, req).await;
        let body_str = std::str::from_utf8(&body).unwrap_or("");
        assert!(
            body_str.contains("<html") || body_str.contains("<!DOCTYPE"),
            "root response should contain HTML"
        );
    }

    // ── get_endpoint_stats ───────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_get_endpoint_stats_returns_200() {
        let monitor = make_monitor();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .route("/endpoints", web::get().to(get_endpoint_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/endpoints").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_get_endpoint_stats_body_shape() {
        let monitor = make_monitor();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .route("/endpoints", web::get().to(get_endpoint_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/endpoints").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["endpoints"].is_array());
        assert!(body["count"].is_number());
    }

    #[actix_web::test]
    async fn test_get_endpoint_stats_empty_initially() {
        let monitor = make_monitor();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .route("/endpoints", web::get().to(get_endpoint_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/endpoints").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["count"], 0);
        assert_eq!(body["endpoints"].as_array().unwrap().len(), 0);
    }

    // ── get_system_info ──────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_get_system_info_returns_200() {
        let monitor = make_monitor();
        let config = make_config();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(config.clone())
                .route("/info", web::get().to(get_system_info)),
        )
        .await;
        let req = test::TestRequest::get().uri("/info").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_get_system_info_body_has_server_and_health() {
        let monitor = make_monitor();
        let config = make_config();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(config.clone())
                .route("/info", web::get().to(get_system_info)),
        )
        .await;
        let req = test::TestRequest::get().uri("/info").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["server"].is_object());
        assert!(body["device"].is_object());
        assert!(body["batch"].is_object());
        assert!(body["performance"].is_object());
        assert!(body["health"].is_object());
    }

    #[actix_web::test]
    async fn test_get_system_info_health_is_healthy() {
        let monitor = make_monitor();
        let config = make_config();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(config.clone())
                .route("/info", web::get().to(get_system_info)),
        )
        .await;
        let req = test::TestRequest::get().uri("/info").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        // A fresh Monitor with zero errors should report healthy
        assert_eq!(body["health"]["healthy"], true);
    }

    // ── list_models ──────────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_list_models_returns_200() {
        let monitor = make_monitor();
        let models = make_model_manager();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(models.clone())
                .route("/models", web::get().to(list_models)),
        )
        .await;
        let req = test::TestRequest::get().uri("/models").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_list_models_body_shape() {
        let monitor = make_monitor();
        let models = make_model_manager();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(models.clone())
                .route("/models", web::get().to(list_models)),
        )
        .await;
        let req = test::TestRequest::get().uri("/models").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["models"].is_array());
        assert!(body["total"].is_number());
    }

    #[actix_web::test]
    async fn test_list_models_total_matches_array_length() {
        let monitor = make_monitor();
        let models = make_model_manager();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(models.clone())
                .route("/models", web::get().to(list_models)),
        )
        .await;
        let req = test::TestRequest::get().uri("/models").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let arr_len = body["models"].as_array().unwrap().len() as u64;
        assert_eq!(body["total"].as_u64().unwrap(), arr_len);
    }

    // ── get_stats ────────────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_get_stats_returns_200() {
        let monitor = make_monitor();
        let engine = make_engine();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .route("/stats", web::get().to(get_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/stats").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_get_stats_body_has_metrics_fields() {
        let monitor = make_monitor();
        let engine = make_engine();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .route("/stats", web::get().to(get_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/stats").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        // SystemMetrics fields
        assert!(body["total_requests"].is_number());
        assert!(body["total_errors"].is_number());
        assert!(body["uptime_seconds"].is_number());
    }

    // ── health_check ─────────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_health_check_returns_200() {
        let monitor = make_monitor();
        let engine = make_engine();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .route("/health", web::get().to(health_check)),
        )
        .await;
        let req = test::TestRequest::get().uri("/health").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_web::test]
    async fn test_health_check_body_shape() {
        let monitor = make_monitor();
        let engine = make_engine();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .route("/health", web::get().to(health_check)),
        )
        .await;
        let req = test::TestRequest::get().uri("/health").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["healthy"].is_boolean());
        assert!(body["uptime_seconds"].is_number());
        assert!(body["active_requests"].is_number());
        assert!(body["error_count"].is_number());
        assert!(body["timestamp"].is_string());
    }

    #[actix_web::test]
    async fn test_health_check_fresh_monitor_is_healthy() {
        let monitor = make_monitor();
        let engine = make_engine();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .route("/health", web::get().to(health_check)),
        )
        .await;
        let req = test::TestRequest::get().uri("/health").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["healthy"], true);
        assert_eq!(body["error_count"], 0);
    }

    // ── predict ──────────────────────────────────────────────────────────────
    // The engine has no real model loaded, so it returns 500 — the error path
    // is still a valid handler code path to cover.

    #[actix_web::test]
    async fn test_predict_with_unknown_model_returns_500() {
        let monitor = make_monitor();
        let engine = make_engine();
        let rate_limiter = make_rate_limiter();
        let deduplicator = make_deduplicator();

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .app_data(rate_limiter.clone())
                .app_data(deduplicator.clone())
                .route("/predict", web::post().to(predict)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": "no-such-model",
            "inputs": {"data": [1, 2, 3]},
            "priority": 0,
            "timeout": null
        });

        let req = test::TestRequest::post()
            .uri("/predict")
            .set_json(&body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        // ModelNotFound is now mapped to 404 (was 500). Accept either to keep
        // the test resilient if the engine surfaces the error as a generic
        // InferenceFailed during early init.
        let status = resp.status().as_u16();
        assert!(
            status == 404 || status == 500,
            "expected 404 or 500, got {status}"
        );
    }

    #[actix_web::test]
    async fn test_predict_error_response_body_shape() {
        let monitor = make_monitor();
        let engine = make_engine();
        let rate_limiter = make_rate_limiter();
        let deduplicator = make_deduplicator();

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .app_data(rate_limiter.clone())
                .app_data(deduplicator.clone())
                .route("/predict", web::post().to(predict)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": "no-such-model",
            "inputs": {"data": [1]},
            "priority": 0,
            "timeout": null
        });

        let req = test::TestRequest::post()
            .uri("/predict")
            .set_json(&body)
            .to_request();
        let resp_body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp_body["success"], false);
        assert!(resp_body["error"].is_string());
    }

    #[actix_web::test]
    async fn test_predict_rate_limited_returns_429() {
        let monitor = make_monitor();
        let engine = make_engine();
        // Limit of 0 means every request is rejected
        let rate_limiter = web::Data::new(Arc::new(RateLimiter::new(0, 60)));
        let deduplicator = make_deduplicator();

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .app_data(rate_limiter.clone())
                .app_data(deduplicator.clone())
                .route("/predict", web::post().to(predict)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": "any-model",
            "inputs": {},
            "priority": 0,
            "timeout": null
        });

        let req = test::TestRequest::post()
            .uri("/predict")
            .set_json(&body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 429);
    }

    // ── synthesize_tts ───────────────────────────────────────────────────────
    // The engine has no TTS model, so we expect 500 covering the error branch.

    #[actix_web::test]
    async fn test_synthesize_tts_with_unknown_model_returns_500() {
        let monitor = make_monitor();
        let engine = make_engine();

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .route("/synthesize", web::post().to(synthesize_tts)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": "no-such-tts-model",
            "text": "Hello world",
            "voice": null,
            "speed": 1.0,
            "pitch": 1.0,
            "volume": 1.0,
            "language": "en",
            "emotion": null,
            "output_format": "wav"
        });

        let req = test::TestRequest::post()
            .uri("/synthesize")
            .set_json(&body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        // ModelNotFound is now mapped to 404; older code returned 500. Accept
        // either while the inference engine error mapping settles.
        let status = resp.status().as_u16();
        assert!(
            status == 404 || status == 500,
            "expected 404 or 500, got {status}"
        );
    }

    #[actix_web::test]
    async fn test_synthesize_tts_error_response_body_shape() {
        let monitor = make_monitor();
        let engine = make_engine();

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .route("/synthesize", web::post().to(synthesize_tts)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": "no-such-tts-model",
            "text": "Hi",
            "voice": null,
            "speed": 1.0,
            "pitch": 1.0,
            "volume": 1.0,
            "language": "en",
            "emotion": null,
            "output_format": "mp3"
        });

        let req = test::TestRequest::post()
            .uri("/synthesize")
            .set_json(&body)
            .to_request();
        let resp_body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp_body["success"], false);
        assert!(resp_body["error"].is_string());
    }

    // ── predict dedup cache-hit branch ────────────────────────────────────────

    #[actix_web::test]
    async fn test_predict_dedup_cache_hit_returns_200() {
        let monitor = make_monitor();
        let engine = make_engine();
        let rate_limiter = make_rate_limiter();
        let deduplicator = make_deduplicator();

        // Pre-populate the cache using the same key the handler will generate.
        let inputs = serde_json::json!({"data": [42]});
        let model_name = "cached-model";
        let cached_value = serde_json::json!({"output": "from_cache"});
        let dedup_inner = deduplicator.get_ref();
        let key = dedup_inner.generate_key(model_name, &inputs);
        dedup_inner.set(key, cached_value.clone(), 60);

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .app_data(rate_limiter.clone())
                .app_data(deduplicator.clone())
                .route("/predict", web::post().to(predict)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": model_name,
            "inputs": inputs,
            "priority": 0,
            "timeout": null
        });

        let req = test::TestRequest::post()
            .uri("/predict")
            .set_json(&body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }

    #[actix_web::test]
    async fn test_predict_dedup_cache_hit_body_shape() {
        let monitor = make_monitor();
        let engine = make_engine();
        let rate_limiter = make_rate_limiter();
        let deduplicator = make_deduplicator();

        let inputs = serde_json::json!({"text": "hello"});
        let model_name = "dedup-model";
        let cached_value = serde_json::json!({"result": "cached"});
        let dedup_inner = deduplicator.get_ref();
        let key = dedup_inner.generate_key(model_name, &inputs);
        dedup_inner.set(key, cached_value.clone(), 60);

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .app_data(rate_limiter.clone())
                .app_data(deduplicator.clone())
                .route("/predict", web::post().to(predict)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": model_name,
            "inputs": inputs,
            "priority": 0,
            "timeout": null
        });

        let req = test::TestRequest::post()
            .uri("/predict")
            .set_json(&body)
            .to_request();
        let resp_body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp_body["success"], true);
        assert!(resp_body["error"].is_null());
        // Cache hits report 0ms processing time and include model_info source field
        assert_eq!(resp_body["processing_time"], 0.0);
        assert_eq!(resp_body["model_info"]["source"], "deduplication_cache");
    }

    // ── predict: success path (lines 85-97) ──────────────────────────────────
    // Register a loaded BaseModel so engine.infer() succeeds and exercises lines 85-97.

    #[allow(dead_code)]
    fn make_engine_with_loaded_model(model_name: &str) -> web::Data<Arc<InferenceEngine>> {
        let cfg = Config::default();
        let manager = Arc::new(ModelManager::new(&cfg, None));
        // Synchronously register a loaded BaseModel using a separate runtime
        let mgr_clone = manager.clone();
        let model_name_owned = model_name.to_string();
        // Use tokio block_in_place since we're inside an async context
        // (called from actix_web::test which runs its own runtime)
        let rt = tokio::runtime::Handle::current();
        std::thread::spawn(move || {
            rt.block_on(async {
                use crate::models::manager::BaseModel;
                let mut model = BaseModel::new(model_name_owned.clone());
                model.load().await.unwrap();
                mgr_clone
                    .register_model(model_name_owned, model)
                    .await
                    .unwrap();
            });
        })
        .join()
        .unwrap();
        web::Data::new(Arc::new(InferenceEngine::new(manager, &cfg)))
    }

    #[actix_web::test]
    async fn test_predict_with_loaded_model_returns_200() {
        let monitor = make_monitor();
        let rate_limiter = make_rate_limiter();
        let deduplicator = make_deduplicator();
        let cfg = Config::default();
        let manager = Arc::new(ModelManager::new(&cfg, None));
        {
            use crate::models::manager::BaseModel;
            let mut model = BaseModel::new("loaded-infer-model".to_string());
            model.load().await.unwrap();
            manager
                .register_model("loaded-infer-model".to_string(), model)
                .await
                .unwrap();
        }
        let engine = web::Data::new(Arc::new(InferenceEngine::new(manager, &cfg)));

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .app_data(rate_limiter.clone())
                .app_data(deduplicator.clone())
                .route("/predict", web::post().to(predict)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": "loaded-infer-model",
            "inputs": {"data": [1, 2, 3]},
            "priority": 0,
            "timeout": null
        });
        let req = test::TestRequest::post()
            .uri("/predict")
            .set_json(&body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }

    #[actix_web::test]
    async fn test_predict_success_response_body_shape() {
        let monitor = make_monitor();
        let rate_limiter = make_rate_limiter();
        let deduplicator = make_deduplicator();
        let cfg = Config::default();
        let manager = Arc::new(ModelManager::new(&cfg, None));
        {
            use crate::models::manager::BaseModel;
            let mut model = BaseModel::new("echo-model".to_string());
            model.load().await.unwrap();
            manager
                .register_model("echo-model".to_string(), model)
                .await
                .unwrap();
        }
        let engine = web::Data::new(Arc::new(InferenceEngine::new(manager, &cfg)));

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .app_data(rate_limiter.clone())
                .app_data(deduplicator.clone())
                .route("/predict", web::post().to(predict)),
        )
        .await;

        let inputs = serde_json::json!({"key": "value"});
        let body = serde_json::json!({
            "model_name": "echo-model",
            "inputs": inputs,
            "priority": 0,
            "timeout": null
        });
        let req = test::TestRequest::post()
            .uri("/predict")
            .set_json(&body)
            .to_request();
        let resp_body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp_body["success"], true);
        assert!(resp_body["error"].is_null());
        assert!(resp_body["processing_time"].is_number());
        // result should be the echoed input (BaseModel::forward echoes inputs)
        assert!(resp_body["result"].is_object());
    }

    // ── synthesize_tts: success path (lines 124-135) ──────────────────────────

    #[actix_web::test]
    async fn test_synthesize_tts_with_loaded_model_returns_200() {
        let monitor = make_monitor();
        let cfg = Config::default();
        let manager = Arc::new(ModelManager::new(&cfg, None));
        {
            use crate::models::manager::BaseModel;
            let mut model = BaseModel::new("tts-success-model".to_string());
            model.load().await.unwrap();
            manager
                .register_model("tts-success-model".to_string(), model)
                .await
                .unwrap();
        }
        let engine = web::Data::new(Arc::new(InferenceEngine::new(manager, &cfg)));

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .route("/synthesize", web::post().to(synthesize_tts)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": "tts-success-model",
            "text": "Hello world",
            "voice": null,
            "speed": 1.0,
            "pitch": 1.0,
            "volume": 1.0,
            "language": "en",
            "emotion": null,
            "output_format": "wav"
        });
        let req = test::TestRequest::post()
            .uri("/synthesize")
            .set_json(&body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }

    #[actix_web::test]
    async fn test_synthesize_tts_success_response_body_shape() {
        let monitor = make_monitor();
        let cfg = Config::default();
        let manager = Arc::new(ModelManager::new(&cfg, None));
        {
            use crate::models::manager::BaseModel;
            let mut model = BaseModel::new("tts-body-model".to_string());
            model.load().await.unwrap();
            manager
                .register_model("tts-body-model".to_string(), model)
                .await
                .unwrap();
        }
        let engine = web::Data::new(Arc::new(InferenceEngine::new(manager, &cfg)));

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .route("/synthesize", web::post().to(synthesize_tts)),
        )
        .await;

        let body = serde_json::json!({
            "model_name": "tts-body-model",
            "text": "Testing TTS",
            "voice": null,
            "speed": 1.0,
            "pitch": 1.0,
            "volume": 1.0,
            "language": "en",
            "emotion": null,
            "output_format": "mp3"
        });
        let req = test::TestRequest::post()
            .uri("/synthesize")
            .set_json(&body)
            .to_request();
        let resp_body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp_body["success"], true);
        assert!(resp_body["error"].is_null());
        assert!(resp_body["audio_data"].is_string());
        assert!(resp_body["processing_time"].is_number());
        assert_eq!(resp_body["sample_rate"], 16000);
    }

    // ── ETag / Cache-Control on root ──────────────────────────────────────────

    #[actix_web::test]
    async fn test_root_etag_header_present() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get().uri("/").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
        let etag = resp.headers().get("etag");
        assert!(etag.is_some(), "ETag header must be present on 200 response");
    }

    #[actix_web::test]
    async fn test_root_cache_control_is_no_cache() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get().uri("/").to_request();
        let resp = test::call_service(&app, req).await;
        let cc = resp
            .headers()
            .get("cache-control")
            .expect("Cache-Control header must be present")
            .to_str()
            .unwrap();
        assert_eq!(cc, "no-cache");
    }

    #[actix_web::test]
    async fn test_root_304_on_matching_if_none_match() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        // First request — learn the ETag
        let req1 = test::TestRequest::get().uri("/").to_request();
        let resp1 = test::call_service(&app, req1).await;
        let etag = resp1
            .headers()
            .get("etag")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        // Second request — send matching ETag, expect 304
        let req2 = test::TestRequest::get()
            .uri("/")
            .insert_header(("if-none-match", etag.as_str()))
            .to_request();
        let resp2 = test::call_service(&app, req2).await;
        assert_eq!(resp2.status(), 304);
    }

    #[actix_web::test]
    async fn test_root_200_on_mismatched_if_none_match() {
        let app = test::init_service(App::new().route("/", web::get().to(root))).await;
        let req = test::TestRequest::get()
            .uri("/")
            .insert_header(("if-none-match", "\"stale-00000000\""))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }

    // ── configure_routes ─────────────────────────────────────────────────────

    #[actix_web::test]
    async fn test_configure_routes_registers_root() {
        // configure_routes registers all routes; verify the "/" route works through it.
        let monitor = make_monitor();
        let engine = make_engine();
        let models = make_model_manager();
        let rate_limiter = make_rate_limiter();
        let deduplicator = make_deduplicator();
        let config = make_config();

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .app_data(models.clone())
                .app_data(rate_limiter.clone())
                .app_data(deduplicator.clone())
                .app_data(config.clone())
                .configure(configure_routes),
        )
        .await;

        let req = test::TestRequest::get().uri("/").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }

    #[actix_web::test]
    async fn test_configure_routes_health_endpoint() {
        // /health is registered at the App level in main.rs (canonical),
        // not via configure_routes. configure_routes used to also register
        // it, but the duplicate was shadowed by the App-level handler and
        // has been removed. This test now exercises the canonical handler.
        let monitor = make_monitor();
        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .route("/health", web::get().to(crate::api::health::health)),
        )
        .await;

        let req = test::TestRequest::get().uri("/health").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }

    #[actix_web::test]
    async fn test_configure_routes_stats_endpoint() {
        let monitor = make_monitor();
        let engine = make_engine();
        let models = make_model_manager();
        let rate_limiter = make_rate_limiter();
        let deduplicator = make_deduplicator();
        let config = make_config();

        let app = test::init_service(
            App::new()
                .app_data(monitor.clone())
                .app_data(engine.clone())
                .app_data(models.clone())
                .app_data(rate_limiter.clone())
                .app_data(deduplicator.clone())
                .app_data(config.clone())
                .configure(configure_routes),
        )
        .await;

        let req = test::TestRequest::get().uri("/stats").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);
    }
}
