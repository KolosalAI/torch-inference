use actix_web::{web, HttpResponse, Responder, HttpRequest};
use serde_json::json;
use chrono::Utc;

use crate::api::types::*;
use crate::config::Config;
use crate::core::engine::InferenceEngine;
use crate::models::manager::ModelManager;
use crate::monitor::Monitor;
use crate::middleware::RateLimiter;
use crate::dedup::RequestDeduplicator;

pub async fn root() -> impl Responder {
    HttpResponse::Ok().json(json!({
        "message": "PyTorch Inference Framework API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": Utc::now().to_rfc3339(),
        "endpoints": {
            "inference": "/predict",
            "tts": "/synthesize",
            "health": "/health",
            "stats": "/stats",
            "models": "/models"
        }
    }))
}

pub async fn health_check(
    engine: web::Data<std::sync::Arc<InferenceEngine>>,
    monitor: web::Data<std::sync::Arc<Monitor>>,
) -> impl Responder {
    let health = engine.health_check();
    let monitor_health = monitor.get_health_status();
    
    HttpResponse::Ok().json(json!({
        "healthy": monitor_health.healthy,
        "checks": health,
        "uptime_seconds": monitor_health.uptime_seconds,
        "active_requests": monitor_health.active_requests,
        "error_count": monitor_health.error_count,
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
    let client_ip = http_req
        .connection_info()
        .peer_addr()
        .unwrap_or("unknown")
        .to_string();

    // Check rate limit
    if let Err(e) = rate_limiter.is_allowed(&client_ip) {
        monitor.record_request_end(0, "/predict", false);
        return actix_web::error::ErrorTooManyRequests(e.message).error_response();
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
            
            HttpResponse::InternalServerError().json(InferenceResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
                processing_time: Some(latency_ms as f64),
                model_info: None,
            })
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
            
            HttpResponse::InternalServerError().json(TTSResponse {
                success: false,
                audio_data: None,
                audio_format: None,
                duration: None,
                sample_rate: None,
                processing_time: Some(latency_ms as f64),
                error: Some(e.to_string()),
            })
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

pub async fn get_endpoint_stats(
    monitor: web::Data<std::sync::Arc<Monitor>>,
) -> impl Responder {
    let stats = monitor.get_endpoint_stats();
    let stats_json: Vec<serde_json::Value> = stats.iter().map(|s| serde_json::to_value(s).unwrap_or_default()).collect();
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
    cfg.route("/", web::get().to(root))
        .route("/health", web::get().to(health_check))
        .route("/predict", web::post().to(predict))
        .route("/synthesize", web::post().to(synthesize_tts))
        .route("/models", web::get().to(list_models))
        .route("/stats", web::get().to(get_stats))
        .route("/endpoints", web::get().to(get_endpoint_stats))
        .route("/info", web::get().to(get_system_info))
        // Audio endpoints
        .route("/audio/synthesize", web::post().to(crate::api::audio::synthesize_speech))
        .route("/audio/transcribe", web::post().to(crate::api::audio::transcribe_audio))
        .route("/audio/validate", web::post().to(crate::api::audio::validate_audio))
        .route("/audio/health", web::get().to(crate::api::audio::audio_health))
        // Image security endpoints
        .route("/image/process/secure", web::post().to(crate::api::image::process_image_secure))
        .route("/image/validate/security", web::post().to(crate::api::image::validate_image_security))
        .route("/image/security/stats", web::get().to(crate::api::image::get_image_security_stats))
        .route("/image/health", web::get().to(crate::api::image::image_health))
        // Model download endpoints
        .route("/models/download", web::post().to(crate::api::model_download::download_model))
        .route("/models/download/status/{id}", web::get().to(crate::api::model_download::get_download_status))
        .route("/models/download/list", web::get().to(crate::api::model_download::list_downloads))
        .route("/models/managed", web::get().to(crate::api::model_download::list_models))
        .route("/models/download/{name}/info", web::get().to(crate::api::model_download::get_model_info))
        .route("/models/download/{name}", web::delete().to(crate::api::model_download::delete_model))
        .route("/models/cache/info", web::get().to(crate::api::model_download::get_cache_info))
        .route("/models/available", web::get().to(crate::api::model_download::list_available_models))
        // SOTA model endpoints
        .route("/models/sota", web::get().to(crate::api::model_download::list_sota_models))
        .route("/models/sota/{model_id}", web::post().to(crate::api::model_download::download_sota_model))
        // System info endpoints
        .route("/system/info", web::get().to(crate::api::system::get_system_info))
        .route("/system/config", web::get().to(crate::api::system::get_config))
        .route("/system/gpu/stats", web::get().to(crate::api::system::get_gpu_stats))
        // Performance endpoints
        .route("/performance", web::get().to(crate::api::performance::get_performance_metrics))
        .route("/performance/profile", web::post().to(crate::api::performance::profile_inference))
        .route("/performance/optimize", web::get().to(crate::api::performance::optimize_performance))
        // Logging endpoints
        .route("/logs", web::get().to(crate::api::logging::get_logging_info))
        .route("/logs/{log_file}", web::get().to(crate::api::logging::get_log_file))
        .route("/logs/{log_file}", web::delete().to(crate::api::logging::clear_log_file));
}


