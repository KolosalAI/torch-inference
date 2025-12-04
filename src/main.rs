mod api;
mod auth;
mod batch;
mod cache;
mod config;
mod core;
mod dedup;
mod error;
mod middleware;
mod models;
mod monitor;
mod resilience;
mod security;
mod telemetry;

use actix_web::{web, App, HttpServer, middleware as actix_middleware};
use log::info;
use std::sync::Arc;

use crate::api::handlers;
use crate::config::Config;
use crate::core::engine::InferenceEngine;
use crate::core::gpu::GpuManager;
use crate::models::manager::ModelManager;
use crate::models::download::ModelDownloadManager;
use crate::middleware::RateLimiter;
use crate::monitor::Monitor;
use crate::resilience::{CircuitBreaker, CircuitBreakerConfig, Bulkhead, BulkheadConfig};
use crate::cache::Cache;
use crate::dedup::RequestDeduplicator;
use crate::telemetry::logger::setup_logging;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    setup_logging();
    
    let config = Config::load().expect("Failed to load configuration");
    info!("Loaded configuration");
    
    log_system_info();
    
    // Initialize components
    let model_manager = Arc::new(ModelManager::new(&config));
    let inference_engine = Arc::new(InferenceEngine::new(model_manager.clone(), &config));
    let monitor = Arc::new(Monitor::new());
    let rate_limiter = Arc::new(RateLimiter::default());
    let circuit_breaker = Arc::new(CircuitBreaker::new(CircuitBreakerConfig::default()));
    let bulkhead = Arc::new(Bulkhead::new(BulkheadConfig::default()));
    let cache = Arc::new(Cache::new(5000));
    let deduplicator = Arc::new(RequestDeduplicator::new(5000));
    
    // Initialize GPU manager
    let gpu_manager = Arc::new(GpuManager::new());
    info!("GPU Manager initialized. CUDA available: {}", GpuManager::is_cuda_available());
    
    // Initialize model download manager
    let cache_dir = std::env::var("MODEL_CACHE_DIR")
        .unwrap_or_else(|_| "./models_cache".to_string());
    let download_manager = Arc::new(
        ModelDownloadManager::new(&cache_dir)
            .expect("Failed to create model download manager")
    );
    download_manager.initialize().await.expect("Failed to initialize download manager");
    info!("Model download manager initialized at {}", cache_dir);
    
    // Initialize audio model manager
    let audio_model_dir = std::env::var("AUDIO_MODEL_DIR")
        .unwrap_or_else(|_| "./models/audio".to_string());
    let audio_model_manager = Arc::new(crate::core::audio_models::AudioModelManager::new(&audio_model_dir));
    audio_model_manager.initialize_default_models().await.ok(); // Don't fail if no models exist yet
    info!("Audio model manager initialized at {}", audio_model_dir);
    
    let start_time = std::time::Instant::now();
    
    let config_cloned = config.clone();
    inference_engine.warmup(&config_cloned).await.expect("Warmup failed");
    
    info!("Server starting on {}:{}", config.server.host, config.server.port);
    
    let addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = std::net::TcpListener::bind(&addr)?;
    
    let config_data = web::Data::new(config);
    let model_mgr = web::Data::new(model_manager);
    let infer_engine = web::Data::new(inference_engine);
    let monitor_data = web::Data::new(monitor.clone());
    let rate_limiter_data = web::Data::new(rate_limiter);
    let circuit_breaker_data = web::Data::new(circuit_breaker);
    let bulkhead_data = web::Data::new(bulkhead);
    let cache_data = web::Data::new(cache);
    let deduplicator_data = web::Data::new(deduplicator);
    
    // New feature data
    let download_state = web::Data::new(crate::api::model_download::ModelDownloadState {
        manager: download_manager,
    });
    let system_state = web::Data::new(crate::api::system::SystemInfoState {
        gpu_manager,
        start_time,
    });
    let audio_state = web::Data::new(crate::api::audio::AudioState {
        model_manager: audio_model_manager,
    });
    let performance_state = web::Data::new(crate::api::performance::PerformanceState {
        monitor,  // Use monitor here (not cloned)
        start_time,
    });
    
    HttpServer::new(move || {
        App::new()
            .app_data(config_data.clone())
            .app_data(model_mgr.clone())
            .app_data(infer_engine.clone())
            .app_data(monitor_data.clone())
            .app_data(rate_limiter_data.clone())
            .app_data(circuit_breaker_data.clone())
            .app_data(bulkhead_data.clone())
            .app_data(cache_data.clone())
            .app_data(deduplicator_data.clone())
            .app_data(download_state.clone())
            .app_data(system_state.clone())
            .app_data(audio_state.clone())
            .app_data(performance_state.clone())
            .wrap(actix_middleware::Logger::default())
            .configure(handlers::configure_routes)
    })
    .listen(listener)?
    .run()
    .await
}

fn log_system_info() {
    info!("╔══════════════════════════════════════════╗");
    info!("║     PyTorch Inference Framework - Rust   ║");
    info!("║             System Information           ║");
    info!("╚══════════════════════════════════════════╝");
    info!("CPU Cores: {}", num_cpus::get());
    info!("CUDA Support: {}", if cfg!(feature = "cuda") { "Enabled" } else { "Disabled" });
    info!("ONNX Support: {}", if cfg!(feature = "onnx") { "Enabled" } else { "Disabled" });
    info!("Audio Processing: Enabled");
    info!("Image Security: Enabled");
}



