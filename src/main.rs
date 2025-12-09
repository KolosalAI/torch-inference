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
    
    println!("\n{}", "═".repeat(80));
    println!("  🚀 PyTorch Inference Framework v1.0.0");
    println!("{}\n", "═".repeat(80));
    
    let config = Config::load().expect("Failed to load configuration");
    info!("✅ Configuration loaded successfully");
    
    log_system_info();
    
    // Initialize PyTorch auto-detection (if torch feature enabled)
    #[cfg(feature = "torch")]
    {
        info!("🔧 Initializing PyTorch environment...");
        match crate::core::torch_autodetect::initialize_torch().await {
            Ok(torch_config) => {
                info!("✅ PyTorch initialized successfully");
                info!("   ├─ Backend: {:?}", torch_config.backend);
                info!("   ├─ Path: {:?}", torch_config.libtorch_path);
                info!("   └─ Version: {}", torch_config.version);
            }
            Err(e) => {
                log::warn!("⚠️  PyTorch initialization failed: {}", e);
                log::warn!("   └─ ML inference features will be limited");
            }
        }
    }
    
    // Initialize components
    info!("🔧 Initializing core components...");
    let model_manager = Arc::new(ModelManager::new(&config));
    let inference_engine = Arc::new(InferenceEngine::new(model_manager.clone(), &config));
    let monitor = Arc::new(Monitor::new());
    let rate_limiter = Arc::new(RateLimiter::default());
    let circuit_breaker = Arc::new(CircuitBreaker::new(CircuitBreakerConfig::default()));
    let bulkhead = Arc::new(Bulkhead::new(BulkheadConfig::default()));
    let cache = Arc::new(Cache::new(5000));
    let deduplicator = Arc::new(RequestDeduplicator::new(5000));
    info!("✅ Core components initialized");
    
    // Initialize GPU manager
    info!("🎮 Initializing GPU manager...");
    let gpu_manager = Arc::new(GpuManager::new());
    if GpuManager::is_cuda_runtime_available() {
        match gpu_manager.get_info() {
            Ok(info) => {
                info!("✅ GPU Manager initialized - {} GPU(s) detected", info.count);
                for device in &info.devices {
                    info!("   └─ GPU {}: {} ({:.2} GB)", 
                        device.id, 
                        device.name, 
                        device.total_memory as f64 / 1024.0 / 1024.0 / 1024.0
                    );
                }
            }
            Err(e) => {
                log::warn!("⚠️  GPU Manager initialized but failed to get GPU info: {}", e);
            }
        }
    } else {
        info!("ℹ️  GPU Manager initialized (CUDA runtime not available, using CPU)");
    }
    
    // Initialize model download manager
    info!("📦 Initializing model download manager...");
    let cache_dir = std::env::var("MODEL_CACHE_DIR")
        .unwrap_or_else(|_| "./models".to_string());
    let download_manager = Arc::new(
        ModelDownloadManager::new(&cache_dir)
            .expect("Failed to create model download manager")
    );
    download_manager.initialize().await.expect("Failed to initialize download manager");
    info!("✅ Model download manager ready at: {}", cache_dir);
    
    // Initialize audio model manager (legacy)
    info!("🎵 Initializing audio model manager...");
    let audio_model_dir = std::env::var("AUDIO_MODEL_DIR")
        .unwrap_or_else(|_| "./models/audio".to_string());
    let audio_model_manager = Arc::new(crate::core::audio_models::AudioModelManager::new(&audio_model_dir));
    audio_model_manager.initialize_default_models().await.ok();
    info!("✅ Audio model manager ready at: {}", audio_model_dir);
    
    // Initialize modern TTS manager
    info!("🎙️  Initializing TTS engines...");
    let tts_config = crate::core::tts_manager::TTSManagerConfig::default();
    let tts_manager = Arc::new(crate::core::tts_manager::TTSManager::new(tts_config));
    tts_manager.initialize_defaults().await.expect("Failed to initialize TTS manager");
    let tts_stats = tts_manager.get_stats();
    info!("✅ TTS Manager ready - {} engine(s) loaded", tts_stats.total_engines);
    for engine_id in &tts_stats.engine_ids {
        info!("   └─ {}", engine_id);
    }
    
    let start_time = std::time::Instant::now();
    
    info!("🔥 Warming up inference engine...");
    let config_cloned = config.clone();
    inference_engine.warmup(&config_cloned).await.expect("Warmup failed");
    info!("✅ Inference engine ready");
    
    let addr = format!("{}:{}", config.server.host, config.server.port);
    info!("🌐 Starting HTTP server on {}...", addr);
    
    let listener = std::net::TcpListener::bind(&addr)?;
    
    let config_data = web::Data::new(config.clone());
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
    let tts_state = web::Data::new(crate::api::tts::TTSState {
        manager: tts_manager,
    });
    let performance_state = web::Data::new(crate::api::performance::PerformanceState {
        monitor,
        start_time,
    });
    
    println!("\n{}", "═".repeat(80));
    println!("  ✅ Server Ready!");
    println!("  🌐 Listening on: http://{}", addr);
    println!("  📚 API Docs: http://{}/health", addr);
    println!("{}\n", "═".repeat(80));
    
    info!("✅ Server started successfully - Workers: {}", config.server.workers);
    
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
            .app_data(tts_state.clone())
            .app_data(performance_state.clone())
            .wrap(actix_middleware::Logger::new(
                r#"%a "%r" %s %b "%{Referer}i" %T"#
            ))
            .configure(handlers::configure_routes)
            .configure(crate::api::tts::configure_routes)
            .configure(crate::api::registry::configure)
            .configure(api::models::configure)
    })
    .workers(config.server.workers)
    .listen(listener)?
    .run()
    .await
}

fn log_system_info() {
    println!("\n  📊 System Information:");
    println!("  {}", "─".repeat(78));
    
    // CPU Information
    let cpu_count = num_cpus::get();
    info!("💻 CPU: {} cores available", cpu_count);
    
    // Memory Information
    if let Ok(sys_info) = sys_info::mem_info() {
        let total_gb = sys_info.total as f64 / 1024.0 / 1024.0;
        let avail_gb = sys_info.avail as f64 / 1024.0 / 1024.0;
        info!("🧠 RAM: {:.2} GB total, {:.2} GB available", total_gb, avail_gb);
    }
    
    // CUDA information
    if cfg!(feature = "cuda") {
        info!("🎮 CUDA: Enabled");
        if GpuManager::is_cuda_runtime_available() {
            if let Some(cuda_info) = GpuManager::get_cuda_info() {
                info!("   └─ {}", cuda_info);
            }
        } else {
            log::warn!("   └─ CUDA runtime not detected");
        }
    } else {
        info!("🎮 CUDA: Disabled (compile with --features cuda to enable)");
    }
    
    // ONNX support
    if cfg!(feature = "onnx") {
        info!("🤖 ONNX: Enabled");
    } else {
        info!("🤖 ONNX: Disabled");
    }
    
    // Audio processing
    #[cfg(feature = "audio")]
    info!("🎵 Audio Processing: Enabled");
    
    #[cfg(not(feature = "audio"))]
    info!("🎵 Audio Processing: Disabled");
    
    // Image security
    #[cfg(feature = "image-security")]
    info!("🔒 Image Security: Enabled");
    
    #[cfg(not(feature = "image-security"))]
    info!("🔒 Image Security: Disabled");
    
    // OS Information
    info!("🖥️  OS: {} {}", std::env::consts::OS, std::env::consts::ARCH);
    
    println!("  {}\n", "─".repeat(78));
}


