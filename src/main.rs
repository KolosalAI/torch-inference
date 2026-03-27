mod api;
mod auth;
mod batch;
mod cache;
mod compression;
mod config;
mod core;
mod dedup;
mod error;
mod guard;
mod inflight_batch;
mod middleware;
mod model_pool;
mod models;
mod monitor;
mod resilience;
mod security;
mod telemetry;
mod tensor_pool;
mod worker_pool;

#[cfg(feature = "torch")]
mod torch_optimization;

use actix_web::{web, App, HttpServer, middleware as actix_middleware};
use log::info;
use std::sync::Arc;

use crate::api::handlers;
use crate::config::Config;
use crate::core::engine::InferenceEngine;
use crate::core::gpu::GpuManager;
use crate::models::manager::ModelManager;
use crate::models::download::ModelDownloadManager;
use crate::middleware::{RateLimiter, CorrelationIdMiddleware};
use crate::monitor::Monitor;
use crate::resilience::{CircuitBreaker, CircuitBreakerConfig, Bulkhead, BulkheadConfig};
use crate::cache::Cache;
use crate::dedup::RequestDeduplicator;
use crate::telemetry::init_structured_logging;
use crate::worker_pool::WorkerPool;
use crate::security::sanitizer::Sanitizer;

#[cfg(feature = "metrics")]
use crate::telemetry::prometheus;

// Use jemalloc for better memory performance (Unix platforms only, when feature is enabled)
#[cfg(all(feature = "jemalloc", not(target_env = "msvc"), not(target_os = "macos")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize structured logging
    let use_json = std::env::var("LOG_JSON").unwrap_or_else(|_| "false".to_string()) == "true";
    let log_dir = std::env::var("LOG_DIR").ok();
    init_structured_logging(log_dir.as_deref(), use_json);
    
    // Initialize Prometheus metrics (if enabled)
    #[cfg(feature = "metrics")]
    {
        if let Err(e) = prometheus::init_metrics() {
            log::warn!("[WARN] Failed to initialize Prometheus metrics: {}", e);
        } else {
            info!("[OK] Prometheus metrics initialized");
        }
    }
    
    println!("\n{}", "═".repeat(80));
    println!("  [START] PyTorch Inference Framework v1.0.0");
    println!("{}\n", "═".repeat(80));
    
    let mut config = Config::load().expect("Failed to load configuration");
    info!("[OK] Configuration loaded successfully");

    // Auto-detect device if set to "auto"
    if config.device.device_type == "auto" {
        info!("[INIT] Auto-detecting compute device...");
        let temp_gpu_manager = GpuManager::new();
        match temp_gpu_manager.get_info() {
            Ok(info) => {
                match info.backend {
                    crate::core::gpu::GpuBackend::Cuda => {
                        config.device.device_type = "cuda".to_string();
                        info!("[AUTO] Detected CUDA - Setting device_type to 'cuda'");
                        
                        // Auto-configure device IDs if not set
                        if config.device.device_ids.is_none() && info.count > 0 {
                            let ids: Vec<usize> = (0..info.count).collect();
                            info!("[AUTO] Detected {} CUDA devices - Setting device_ids to {:?}", info.count, ids);
                            config.device.device_ids = Some(ids);
                        }
                    },
                    crate::core::gpu::GpuBackend::Metal => {
                        config.device.device_type = "mps".to_string();
                        info!("[AUTO] Detected Metal - Setting device_type to 'mps'");
                        
                        // Enable Metal-specific optimizations
                        if config.device.metal_optimize_for_apple_silicon {
                            info!("[METAL] Apple Silicon optimizations enabled");
                            
                            // For unified memory, disable FP16 if not explicitly set (can cause issues on some models)
                            if !config.device.use_fp16 {
                                info!("[METAL] Using FP32 for maximum compatibility on unified memory");
                            }
                            
                            // Set optimal thread count for Apple Silicon (efficiency + performance cores)
                            let optimal_threads = (num_cpus::get() * 3) / 4; // Use 75% of cores for best performance
                            config.device.num_threads = optimal_threads;
                            info!("[METAL] Set optimal thread count to {} for Apple Silicon", optimal_threads);
                        }
                        
                        if config.device.metal_cache_shaders {
                            info!("[METAL] Metal shader caching enabled for faster startup");
                        }
                        
                        // Metal usually just uses device 0 (the unified memory GPU)
                        if config.device.device_ids.is_none() {
                             config.device.device_ids = Some(vec![0]);
                        }
                    },
                    crate::core::gpu::GpuBackend::Cpu => {
                        config.device.device_type = "cpu".to_string();
                        info!("[AUTO] No GPU detected - Setting device_type to 'cpu'");
                    }
                }
            },
            Err(e) => {
                log::warn!("[WARN] Failed to detect GPU info: {}. Defaulting to CPU.", e);
                config.device.device_type = "cpu".to_string();
            }
        }
    }
    
    log_system_info();
    
    // Initialize PyTorch auto-detection (if torch feature enabled)
    #[cfg(feature = "torch")]
    {
        info!("[INIT] Initializing PyTorch environment...");
        
        // Initialize tch for thread safety
        tch::maybe_init_cuda();
        
        match crate::core::torch_autodetect::initialize_torch().await {
            Ok(torch_config) => {
                info!("[OK] PyTorch initialized successfully");
                info!("   ├─ Backend: {:?}", torch_config.backend);
                info!("   ├─ Path: {:?}", torch_config.libtorch_path);
                info!("   └─ Version: {}", torch_config.version);
            }
            Err(e) => {
                log::warn!("[WARN]  PyTorch initialization failed: {}", e);
                log::warn!("   └─ ML inference features will be limited");
            }
        }
    }
    
    // Initialize tensor pool for memory optimization
    let tensor_pool = if config.performance.enable_tensor_pooling {
        info!("[OPT] Tensor pooling enabled (max: {} tensors)", config.performance.max_pooled_tensors);
        Some(Arc::new(crate::tensor_pool::TensorPool::new(config.performance.max_pooled_tensors)))
    } else {
        None
    };

    // Initialize components
    info!("[INIT] Initializing core components...");
    let model_manager = Arc::new(ModelManager::new(&config, tensor_pool.clone()));
    let inference_engine = Arc::new(InferenceEngine::new(model_manager.clone(), &config));
    let monitor = Arc::new(Monitor::new());
    let rate_limiter = Arc::new(RateLimiter::default());
    let circuit_breaker = Arc::new(CircuitBreaker::new(CircuitBreakerConfig::default()));
    let bulkhead = Arc::new(Bulkhead::new(BulkheadConfig::default()));
    
    let worker_pool_config = crate::worker_pool::WorkerPoolConfig {
        min_workers: config.performance.min_workers,
        max_workers: config.performance.max_workers,
        enable_auto_scaling: config.performance.enable_auto_scaling,
        enable_zero_scaling: config.performance.enable_zero_scaling,
        ..Default::default()
    };
    let worker_pool = crate::worker_pool::WorkerPool::new(worker_pool_config);
    worker_pool.initialize().await.expect("Failed to initialize worker pool");
    
    // Initialize cache with LRU eviction
    let cache_size = (config.performance.cache_size_mb * 1024 * 1024) / 1024; // Estimate entries
    let cache = Arc::new(Cache::new(cache_size.max(1000)));
    let deduplicator = Arc::new(RequestDeduplicator::new(5000));
    
    // Initialize compression service
    let compression = if config.performance.enable_result_compression {
        info!("[OPT] Result compression enabled (level: {})", config.performance.compression_level);
        Some(Arc::new(crate::compression::CompressionService::new(config.performance.compression_level)))
    } else {
        None
    };
    
    info!("[OK] Core components initialized");
    
    // Initialize GPU manager
    info!("[GPU] Initializing GPU manager...");
    let gpu_manager = Arc::new(GpuManager::new());
    
    // Try to detect GPUs
    match gpu_manager.get_info() {
        Ok(info) => {
            if info.available {
                match info.backend {
                    crate::core::gpu::GpuBackend::Cuda => {
                        info!("[OK] GPU Manager initialized - {} CUDA GPU(s) detected", info.count);
                    }
                    crate::core::gpu::GpuBackend::Metal => {
                        info!("[OK] GPU Manager initialized - Metal GPU detected (Apple Silicon)");
                    }
                    crate::core::gpu::GpuBackend::Cpu => {
                        info!("[INFO]  GPU Manager initialized (No GPU detected, using CPU)");
                    }
                }
                
                for device in &info.devices {
                    info!("   └─ GPU {}: {} ({:.2} GB)", 
                        device.id, 
                        device.name, 
                        device.total_memory as f64 / 1024.0 / 1024.0 / 1024.0
                    );
                }
            } else {
                info!("[INFO]  GPU Manager initialized (No GPU available, using CPU)");
            }
        }
        Err(e) => {
            log::warn!("[WARN]  GPU Manager initialized but failed to get GPU info: {}", e);
            info!("[INFO]  Falling back to CPU mode");
        }
    }
    
    // Initialize model download manager
    info!("[DOWNLOAD] Initializing model download manager...");
    let cache_dir = std::env::var("MODEL_CACHE_DIR")
        .unwrap_or_else(|_| "./models".to_string());
    let download_manager = Arc::new(
        ModelDownloadManager::new(&cache_dir)
            .expect("Failed to create model download manager")
    );
    download_manager.initialize().await.expect("Failed to initialize download manager");
    info!("[OK] Model download manager ready at: {}", cache_dir);
    
    // Initialize audio model manager (legacy)
    info!("[AUDIO] Initializing audio model manager...");
    let audio_model_dir = std::env::var("AUDIO_MODEL_DIR")
        .unwrap_or_else(|_| "./models/audio".to_string());
    let audio_model_manager = Arc::new(crate::core::audio_models::AudioModelManager::new(&audio_model_dir));
    audio_model_manager.initialize_default_models().await.ok();
    info!("[OK] Audio model manager ready at: {}", audio_model_dir);
    
    // Initialize modern TTS manager
    info!("[TTS]  Initializing TTS engines...");
    let tts_config = crate::core::tts_manager::TTSManagerConfig::default();
    let tts_manager = Arc::new(crate::core::tts_manager::TTSManager::new(tts_config));
    tts_manager.initialize_defaults().await.expect("Failed to initialize TTS manager");
    let tts_stats = tts_manager.get_stats();
    info!("[OK] TTS Manager ready - {} engine(s) loaded", tts_stats.total_engines);
    for engine_id in &tts_stats.engine_ids {
        info!("   └─ {}", engine_id);
    }
    
    let start_time = std::time::Instant::now();
    
    if config.performance.preload_models_on_startup {
        info!("[WARMUP] Pre-loading models on startup...");
        for model_name in &config.models.auto_load {
            if let Ok(_) = model_manager.get_model(model_name) {
                info!("   └─ Pre-loaded: {}", model_name);
            }
        }
    }
    
    // Warmup runs in the background so the HTTP server becomes reachable
    // (and /health returns 200) immediately rather than waiting for the first
    // inference pass to complete.
    info!("[WARMUP] Warming up inference engine (background)...");
    {
        let warmup_engine = inference_engine.clone();
        let config_cloned = config.clone();
        tokio::spawn(async move {
            match warmup_engine.warmup(&config_cloned).await {
                Ok(()) => info!("[OK] Inference engine warmup complete"),
                Err(e) => log::warn!("[WARN] Inference engine warmup failed: {}", e),
            }
        });
    }
    
    let addr = format!("{}:{}", config.server.host, config.server.port);
    info!("[SERVER] Starting HTTP server on {}...", addr);
    
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
        sanitizer: Sanitizer::new(config.sanitizer.clone()),
    });
    let tts_state = web::Data::new(crate::api::tts::TTSState {
        manager: tts_manager,
    });
    let performance_state = web::Data::new(crate::api::performance::PerformanceState {
        monitor,
        start_time,
    });
    
    println!("\n{}", "═".repeat(80));
    println!("  [OK] Server Ready!");
    println!("  [SERVER] Listening on: http://{}", addr);
    println!("  [HEALTH] Health Check: http://{}/health", addr);
    println!("  [HEALTH] Liveness Probe: http://{}/health/live", addr);
    println!("  [HEALTH] Readiness Probe: http://{}/health/ready", addr);
    #[cfg(feature = "metrics")]
    println!("  [METRICS] Prometheus Metrics: http://{}/metrics", addr);
    if config.performance.enable_tensor_pooling {
        println!("  [OPT] ✓ Tensor pooling enabled");
    }
    if config.performance.enable_result_compression {
        println!("  [OPT] ✓ Result compression enabled");
    }
    if config.performance.adaptive_batch_timeout {
        println!("  [OPT] ✓ Adaptive batching enabled");
    }
    if config.performance.enable_caching {
        println!("  [OPT] ✓ LRU caching enabled ({} MB)", config.performance.cache_size_mb);
    }
    println!("{}\n", "═".repeat(80));
    
    info!("[OK] Server started successfully - Workers: {}", config.server.workers);
    
    let server = HttpServer::new(move || {
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
            // Add correlation ID middleware first
            .wrap(CorrelationIdMiddleware)
            .wrap(actix_middleware::Logger::new(
                r#"%a "%r" %s %b "%{Referer}i" %T %{X-Correlation-ID}o"#
            ))
            // Health check endpoints
            .route("/health", web::get().to(crate::api::health::health))
            .route("/health/live", web::get().to(crate::api::health::liveness))
            .route("/health/ready", web::get().to(crate::api::health::readiness))
            // Metrics endpoint (Prometheus)
            .route("/metrics", web::get().to(crate::api::metrics_endpoint::metrics_handler))
            .configure(handlers::configure_routes)
            .configure(crate::api::tts::configure_routes)
            .configure(crate::api::registry::configure)
            .configure(api::models::configure)
    })
    .workers(config.server.workers)
    .shutdown_timeout(30)  // 30s graceful shutdown
    .listen(listener)?
    .run();

    // Handle graceful shutdown
    let server_handle = server.handle();
    tokio::spawn(async move {
        shutdown_signal().await;
        info!("[SHUTDOWN] Shutdown signal received, draining requests...");
        server_handle.stop(true).await;
    });

    server.await
}

async fn shutdown_signal() {
    use tokio::signal;
    
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("[SHUTDOWN] Received Ctrl+C signal");
        },
        _ = terminate => {
            info!("[SHUTDOWN] Received SIGTERM signal");
        },
    }
}

fn log_system_info() {
    println!("\n  [SYSTEM] System Information:");
    println!("  {}", "─".repeat(78));
    
    // CPU Information
    let cpu_count = num_cpus::get();
    info!("[CPU] CPU: {} cores available", cpu_count);
    
    // Memory Information
    if let Ok(sys_info) = sys_info::mem_info() {
        let total_gb = sys_info.total as f64 / 1024.0 / 1024.0;
        let avail_gb = sys_info.avail as f64 / 1024.0 / 1024.0;
        info!("[RAM] RAM: {:.2} GB total, {:.2} GB available", total_gb, avail_gb);
    }
    
    // GPU Backend information
    #[cfg(target_os = "macos")]
    {
        if GpuManager::is_metal_available() {
            let gpu_manager = GpuManager::new();
            if let Some(metal_info) = gpu_manager.get_metal_info_string() {
                info!("[METAL] Metal GPU: {}", metal_info);
            } else {
                info!("[METAL] Metal: Available (Apple Silicon GPU)");
            }
        } else {
            info!("[METAL] Metal: Not available");
        }
    }
    
    // CUDA information
    if cfg!(feature = "cuda") {
        info!("[GPU] CUDA: Enabled");
        if GpuManager::is_cuda_runtime_available() {
            if let Some(cuda_info) = GpuManager::get_cuda_info() {
                info!("   └─ {}", cuda_info);
            }
        } else {
            log::warn!("   └─ CUDA runtime not detected");
        }
    } else {
        #[cfg(not(target_os = "macos"))]
        info!("[GPU] CUDA: Disabled (compile with --features cuda to enable)");
        
        #[cfg(target_os = "macos")]
        {
            if !GpuManager::is_metal_available() {
                info!("[GPU] GPU: Not available (Metal not detected)");
            }
        }
    }
    
    // ONNX support
    if cfg!(feature = "onnx") {
        info!("[ONNX] ONNX: Enabled");
    } else {
        info!("[ONNX] ONNX: Disabled");
    }
    
    // Audio processing
    #[cfg(feature = "audio")]
    info!("[AUDIO] Audio Processing: Enabled");
    
    #[cfg(not(feature = "audio"))]
    info!("[AUDIO] Audio Processing: Disabled");
    
    // Image security
    #[cfg(feature = "image-security")]
    info!("[SECURITY] Image Security: Enabled");
    
    #[cfg(not(feature = "image-security"))]
    info!("[SECURITY] Image Security: Disabled");
    
    // OS Information
    info!("[OS]  OS: {} {}", std::env::consts::OS, std::env::consts::ARCH);
    
    println!("  {}\n", "─".repeat(78));
}


