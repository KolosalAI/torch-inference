#![allow(unexpected_cfgs)]
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
mod postprocess;
mod resilience;
mod security;
mod telemetry;
mod tensor_pool;
mod worker_pool;

#[cfg(feature = "torch")]
mod torch_optimization;

use actix_web::{web, App, HttpServer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(feature = "profiling")]
use std::sync::Mutex;
use std::time::Duration;

use crate::api::handlers;
use crate::cache::Cache;
use crate::config::Config;
use crate::core::engine::InferenceEngine;
use crate::core::gpu::GpuManager;
use crate::dedup::RequestDeduplicator;
use crate::middleware::{CorrelationIdMiddleware, RateLimiter, RequestLogger};
use crate::models::download::ModelDownloadManager;
use crate::models::manager::ModelManager;
use crate::monitor::Monitor;
use crate::resilience::{Bulkhead, BulkheadConfig, CircuitBreaker, CircuitBreakerConfig};
use crate::security::sanitizer::Sanitizer;
use crate::telemetry::init_structured_logging;

// ── No-op stub backends (replaced at runtime when a real model is loaded) ────

/// Stub classification backend — returns a "not configured" error.
struct NoOpClassificationBackend;

#[async_trait::async_trait]
impl crate::api::classify::ClassificationBackend for NoOpClassificationBackend {
    async fn classify_nchw(
        &self,
        _batch: ndarray::Array4<f32>,
        _top_k: usize,
    ) -> anyhow::Result<Vec<Vec<crate::api::classify::Prediction>>> {
        anyhow::bail!("no classification model loaded")
    }
}

/// Stub LLM backend — returns an empty model list and a "not configured" error.
struct NoOpLlmBackend;

#[async_trait::async_trait]
impl crate::api::llm::LlmBackend for NoOpLlmBackend {
    fn list_models(&self) -> Vec<crate::api::llm::ModelInfo> {
        vec![]
    }
    async fn complete(
        &self,
        _model: &str,
        _prompt: &str,
        _params: crate::core::llm::SamplingParams,
    ) -> anyhow::Result<(String, usize)> {
        anyhow::bail!("no LLM model loaded")
    }
}

#[cfg(feature = "profiling")]
use pprof::ProfilerGuardBuilder;

#[cfg(feature = "metrics")]
use crate::telemetry::prometheus;

// Use jemalloc for better memory performance when feature is enabled.
// Enabled on all non-MSVC platforms (including macOS) — jemalloc's arena-per-thread
// allocator significantly outperforms libmalloc for ML workloads with many short-lived tensors.
#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Resolve the number of Actix-web worker threads.
/// When `configured == 0`, defaults to the logical CPU count.
fn resolve_worker_count(configured: usize) -> usize {
    if configured == 0 {
        num_cpus::get()
    } else {
        configured
    }
}

/// Auto-detect ONNX Runtime, CUDA, and Metal library paths and set the
/// relevant environment variables before any runtime loads them.
///
/// * `ORT_DYLIB_PATH`  — path to `libonnxruntime.{dylib,so,dll}`
///
/// CUDA and Metal are detected by the existing `GpuManager`; this function
/// only handles the ORT library path which must be set before the first
/// ORT session is created.
fn auto_detect_backends() {
    // ── ORT dynamic library ─────────────────────────────────────────────────
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        // Already set by the user — respect it.
        return;
    }

    #[cfg(target_os = "macos")]
    let candidates: &[&str] = &[
        "/opt/homebrew/lib/libonnxruntime.dylib", // Homebrew (Apple Silicon)
        "/usr/local/lib/libonnxruntime.dylib",    // Homebrew (Intel Mac)
        "/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib",
    ];

    #[cfg(target_os = "linux")]
    let candidates: &[&str] = &[
        "/usr/lib/libonnxruntime.so",
        "/usr/local/lib/libonnxruntime.so",
        "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
        "/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
    ];

    #[cfg(target_os = "windows")]
    let candidates: &[&str] = &[
        r"C:\Program Files\onnxruntime\lib\onnxruntime.dll",
        r"C:\onnxruntime\lib\onnxruntime.dll",
    ];

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let candidates: &[&str] = &[];

    for path in candidates {
        if std::path::Path::new(path).exists() {
            // SAFETY: called before any threads that read this var are spawned.
            unsafe { std::env::set_var("ORT_DYLIB_PATH", path) };
            // Will be logged after the logging system is ready (see main).
            // Store it in a way main() can read:
            unsafe { std::env::set_var("_ORT_AUTODETECTED", path) };
            return;
        }
    }
}

/// Build a Tokio multi-thread runtime that:
/// 1. Spawns exactly `num_cpus` worker threads.
/// 2. Uses a 4 MiB stack per thread (ML call stacks are deep).
/// 3. Pins each worker thread to its own CPU core on startup, eliminating
///    scheduler migration overhead and stabilising SIMD latency.
fn build_runtime() -> tokio::runtime::Runtime {
    let n_workers = num_cpus::get();
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();

    // Each Tokio worker picks the next available core ID atomically.
    static NEXT_CORE: AtomicUsize = AtomicUsize::new(0);

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder
        .worker_threads(n_workers)
        .thread_stack_size(4 * 1024 * 1024) // 4 MiB — headroom for deep inference stacks
        .thread_name("torch-worker");

    if !core_ids.is_empty() {
        builder.on_thread_start(move || {
            let idx = NEXT_CORE.fetch_add(1, Ordering::Relaxed);
            let core_id = core_ids[idx % core_ids.len()];
            core_affinity::set_for_current(core_id);
        });
    }

    builder
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
}

fn main() -> std::io::Result<()> {
    build_runtime().block_on(async_main())
}

async fn async_main() -> std::io::Result<()> {
    // Auto-detect backend library paths BEFORE logging or ORT initialises.
    auto_detect_backends();

    // Configure Rayon's global thread pool to match the machine's logical CPU count.
    // ndarray parallel ops, image preprocessing, and any data-parallel work all draw
    // from this pool.  Setting it explicitly prevents rayon from spawning extra threads
    // that would compete with Tokio and ORT worker pools.
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .thread_name(|i| format!("rayon-{i}"))
        .build_global()
        .unwrap_or_else(|e| tracing::warn!(error = %e, "rayon global pool init failed — using default"));

    #[cfg(feature = "profiling")]
    let profiler_guard: Arc<Mutex<Option<pprof::ProfilerGuard<'static>>>> = {
        let guard = ProfilerGuardBuilder::default()
            .frequency(100)
            .build()
            .expect("failed to start pprof profiler");
        Arc::new(Mutex::new(Some(guard)))
    };

    // Initialize structured logging
    let use_json = std::env::var("LOG_JSON").unwrap_or_else(|_| "false".to_string()) == "true";
    let log_dir = std::env::var("LOG_DIR").ok();
    init_structured_logging(log_dir.as_deref(), use_json);
    #[cfg(feature = "profiling")]
    tracing::info!("pprof profiling active at 100 Hz; flamegraph.svg written on shutdown");

    // Initialize Prometheus metrics (if enabled)
    #[cfg(feature = "metrics")]
    {
        if let Err(e) = prometheus::init_metrics() {
            tracing::warn!(error = %e, "prometheus metrics initialization failed");
        } else {
            tracing::info!("prometheus metrics initialized");
        }
    }

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "torch-inference starting"
    );

    let mut config = {
        let _span = tracing::info_span!("config_load").entered();
        let cfg = Config::load().expect("Failed to load configuration");
        tracing::info!(
            device_type = %cfg.device.device_type,
            host        = %cfg.server.host,
            port        = cfg.server.port,
            "config loaded"
        );
        cfg
    };

    {
        let _span = tracing::info_span!("device_detect").entered();
        if config.device.device_type == "auto" {
            tracing::info!("auto-detecting compute device");
            let temp_gpu_manager = GpuManager::new();
            match temp_gpu_manager.get_info() {
                Ok(info) => match info.backend {
                    crate::core::gpu::GpuBackend::Cuda => {
                        config.device.device_type = "cuda".to_string();
                        tracing::info!(backend = "cuda", "cuda detected");
                        if config.device.device_ids.is_none() && info.count > 0 {
                            let ids: Vec<usize> = (0..info.count).collect();
                            config.device.device_ids = Some(ids.clone());
                            tracing::info!(
                                device_count = info.count,
                                device_ids   = ?ids,
                                "cuda devices configured"
                            );
                        }
                    }
                    crate::core::gpu::GpuBackend::Metal => {
                        config.device.device_type = "mps".to_string();
                        tracing::info!(backend = "mps", "metal detected");
                        if config.device.metal_optimize_for_apple_silicon {
                            let optimal_threads = (num_cpus::get() * 3) / 4;
                            config.device.num_threads = optimal_threads;
                            tracing::info!(
                                threads = optimal_threads,
                                fp16 = config.device.use_fp16,
                                shader_caching = config.device.metal_cache_shaders,
                                "apple silicon configured"
                            );
                        }
                        if config.device.device_ids.is_none() {
                            config.device.device_ids = Some(vec![0]);
                        }
                    }
                    crate::core::gpu::GpuBackend::Cpu => {
                        config.device.device_type = "cpu".to_string();
                        tracing::info!(backend = "cpu", "no gpu detected, using cpu");
                    }
                },
                Err(e) => {
                    tracing::warn!(error = %e, backend = "cpu", "gpu detection failed, defaulting to cpu");
                    config.device.device_type = "cpu".to_string();
                }
            }
            tracing::info!(backend = %config.device.device_type, "device detection complete");
        } else {
            tracing::info!(backend = %config.device.device_type, "device configured from config");
        }
    }

    {
        let _span = tracing::info_span!("system_info").entered();
        log_system_info();
    }

    if let Ok(detected) = std::env::var("_ORT_AUTODETECTED") {
        tracing::info!(ort_path = %detected, source = "auto-detected", "ort library found");
    } else if let Ok(explicit) = std::env::var("ORT_DYLIB_PATH") {
        tracing::info!(ort_path = %explicit, source = "environment", "ort library configured");
    } else {
        tracing::warn!("ort library not found; set ORT_DYLIB_PATH if onnx inference is needed");
    }

    #[cfg(feature = "torch")]
    {
        {
            let _span = tracing::info_span!("pytorch_init").entered();
            tch::maybe_init_cuda();
            tracing::info!("pytorch initializing");
            drop(_span);
        }
        match crate::core::torch_autodetect::initialize_torch().await {
            Ok(torch_config) => {
                tracing::info!(
                    backend = ?torch_config.backend,
                    path    = ?torch_config.libtorch_path,
                    version = %torch_config.version,
                    "pytorch initialized"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "pytorch initialization failed; ml inference limited");
            }
        }
    }

    // Initialize tensor pool for memory optimization
    let tensor_pool = if config.performance.enable_tensor_pooling {
        tracing::info!(
            max_tensors = config.performance.max_pooled_tensors,
            "tensor pooling enabled"
        );
        Some(Arc::new(crate::tensor_pool::TensorPool::new(
            config.performance.max_pooled_tensors,
        )))
    } else {
        None
    };

    // Initialize components
    let _comp_span = tracing::info_span!("components_init").entered();
    tracing::info!("initializing core components");
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

    tracing::info!(
        workers_min = config.performance.min_workers,
        workers_max = config.performance.max_workers,
        cache_size_mb = config.performance.cache_size_mb,
        "core components initialized"
    );
    drop(_comp_span);

    worker_pool
        .initialize()
        .await
        .expect("Failed to initialize worker pool");

    // Initialize cache with LRU eviction
    let cache_size = (config.performance.cache_size_mb * 1024 * 1024) / 1024; // Estimate entries
    let cache = Arc::new(Cache::new(cache_size.max(1000)));
    let deduplicator = Arc::new(RequestDeduplicator::new(5000));

    // Initialize compression service
    let _compression = if config.performance.enable_result_compression {
        tracing::info!(
            level = config.performance.compression_level,
            "result compression enabled"
        );
        Some(Arc::new(crate::compression::CompressionService::new(
            config.performance.compression_level,
        )))
    } else {
        None
    };

    // Initialize GPU manager
    let gpu_manager = Arc::new(GpuManager::new());
    {
        let _span = tracing::info_span!("gpu_init").entered();
        match gpu_manager.get_info() {
            Ok(info) if info.available => {
                let device_names: Vec<String> = info
                    .devices
                    .iter()
                    .map(|d| format!("{}:{}", d.name, d.id))
                    .collect();
                tracing::info!(
                    backend      = ?info.backend,
                    gpu_count    = info.count,
                    device_names = %device_names.join(", "),
                    "gpu init complete"
                );
            }
            Ok(_) => {
                tracing::info!(backend = "cpu", "gpu init complete, no gpu available");
            }
            Err(e) => {
                tracing::warn!(error = %e, "gpu detection failed, falling back to cpu");
            }
        }
    }

    // Initialize model download manager
    let download_manager = {
        let cache_dir = std::env::var("MODEL_CACHE_DIR").unwrap_or_else(|_| "./models".to_string());
        let dm = Arc::new(
            ModelDownloadManager::new(&cache_dir).expect("Failed to create model download manager"),
        );
        {
            let _span = tracing::info_span!("download_init").entered();
            tracing::info!(cache_dir = %cache_dir, "download manager initializing");
            drop(_span);
        }
        dm.initialize()
            .await
            .expect("Failed to initialize download manager");
        tracing::info!(cache_dir = %cache_dir, "download manager ready");
        dm
    };

    // Initialize audio model manager (legacy)
    let audio_model_manager = {
        let audio_model_dir =
            std::env::var("AUDIO_MODEL_DIR").unwrap_or_else(|_| "./models/audio".to_string());
        let am = Arc::new(crate::core::audio_models::AudioModelManager::new(
            &audio_model_dir,
        ));
        {
            let _span = tracing::info_span!("audio_init").entered();
            tracing::info!(model_dir = %audio_model_dir, "audio manager initializing");
            drop(_span);
        }
        am.initialize_default_models().await.ok();
        tracing::info!(model_dir = %audio_model_dir, "audio manager ready");
        am
    };

    // Initialize modern TTS manager
    let tts_manager = {
        let tts_config = crate::core::tts_manager::TTSManagerConfig::default();
        let tm = Arc::new(crate::core::tts_manager::TTSManager::new(tts_config));
        {
            let _span = tracing::info_span!("tts_init").entered();
            tracing::info!("tts manager initializing");
            drop(_span);
        }
        tm.initialize_defaults()
            .await
            .expect("Failed to initialize TTS manager");
        let tts_stats = tm.get_stats();
        tracing::info!(
            engine_count = tts_stats.total_engines,
            engine_ids   = %tts_stats.engine_ids.join(", "),
            "tts manager ready"
        );
        tm
    };

    let start_time = std::time::Instant::now();

    if config.performance.preload_models_on_startup {
        let _span = tracing::info_span!("model_preload").entered();
        tracing::info!(
            model_count = config.models.auto_load.len(),
            "preloading models on startup"
        );
        for model_name in &config.models.auto_load {
            if let Ok(_) = model_manager.get_model(model_name) {
                tracing::info!(model = %model_name, status = "loaded", "model preloaded");
            } else {
                tracing::warn!(model = %model_name, status = "not_found", "model preload skipped");
            }
        }
    }

    // Warmup runs in the background so the HTTP server becomes reachable
    // (and /health returns 200) immediately rather than waiting for the first
    // inference pass to complete.
    tracing::info!("starting background warmup");
    {
        use tracing::Instrument;
        let warmup_engine = inference_engine.clone();
        let config_cloned = config.clone();
        tokio::spawn(
            async move {
                match warmup_engine.warmup(&config_cloned).await {
                    Ok(()) => tracing::info!(status = "ok", "warmup complete"),
                    Err(e) => tracing::warn!(error = %e, status = "failed", "warmup failed"),
                }
            }
            .instrument(tracing::info_span!("warmup")),
        );
    }

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let display_addr = format!("localhost:{}", config.server.port);
    let worker_count = resolve_worker_count(config.server.workers);
    tracing::info!(addr = %addr, workers = worker_count, "binding http server");

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
    let classify_state = web::Data::new(crate::api::classify::ClassifyState {
        backend: std::sync::Arc::new(NoOpClassificationBackend),
    });
    let llm_state = web::Data::new(crate::api::llm::LlmState {
        #[cfg(feature = "candle")]
        // TODO: call backend.load_model(model_id, model_dir) here for models
        // configured in config.models.auto_load before accepting requests.
        backend: std::sync::Arc::new(crate::core::llm::CandleLlmBackend::new()),
        #[cfg(not(feature = "candle"))]
        backend: std::sync::Arc::new(NoOpLlmBackend),
    });
    let yolo_state = web::Data::new(crate::api::yolo::YoloState {
        models_dir: config.models.cache_dir.clone(),
    });
    let nn_state = web::Data::new(crate::api::inference::NeuralNetworkState {
        networks: Arc::new(dashmap::DashMap::new()),
    });

    tracing::info!(
        server_url    = %format!("http://{}", display_addr),
        health_url    = %format!("http://{}/health", display_addr),
        liveness_url  = %format!("http://{}/health/live", display_addr),
        readiness_url = %format!("http://{}/health/ready", display_addr),
        workers       = worker_count,
        "server ready"
    );
    eprintln!(
        "\n  Server:  http://{}\n  Health:  http://{}/health\n",
        display_addr, display_addr
    );
    tracing::info!(
        tensor_pooling = config.performance.enable_tensor_pooling,
        compression = config.performance.enable_result_compression,
        adaptive_batch = config.performance.adaptive_batch_timeout,
        lru_caching = config.performance.enable_caching,
        cache_size_mb = config.performance.cache_size_mb,
        "server features"
    );
    #[cfg(feature = "metrics")]
    tracing::info!(
        metrics_url = %format!("http://{}/metrics", display_addr),
        "prometheus metrics available"
    );
    tracing::info!(workers = worker_count, "server started successfully");
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
            .app_data(classify_state.clone())
            .app_data(llm_state.clone())
            .app_data(yolo_state.clone())
            .app_data(nn_state.clone())
            // CorrelationIdMiddleware runs first (innermost), RequestLogger runs last (outermost)
            // so it captures total wall time for each request
            .wrap(CorrelationIdMiddleware)
            .wrap(RequestLogger)
            // Health check endpoints
            .route("/health", web::get().to(crate::api::health::health))
            .route("/health/live", web::get().to(crate::api::health::liveness))
            .route(
                "/health/ready",
                web::get().to(crate::api::health::readiness),
            )
            // Metrics endpoint (Prometheus)
            .route(
                "/metrics",
                web::get().to(crate::api::metrics_endpoint::metrics_handler),
            )
            // Dashboard SSE stream
            .route(
                "/dashboard/stream",
                web::get().to(crate::api::dashboard::dashboard_stream),
            )
            .configure(handlers::configure_routes)
            .configure(crate::api::tts::configure_routes)
            .configure(crate::api::classify::configure_routes)
            .configure(crate::api::llm::configure_routes)
            .configure(crate::api::registry::configure)
            .configure(api::models::configure)
            .configure(crate::api::yolo::configure)
            .configure(crate::api::inference::configure)
    })
    .workers(worker_count)
    .keep_alive(Duration::from_secs(75))
    .client_request_timeout(Duration::from_secs(5))
    .client_disconnect_timeout(Duration::from_secs(1))
    .shutdown_timeout(30) // 30s graceful shutdown
    .listen(listener)?
    .run();

    // Graceful shutdown.
    //
    // ORT and Metal (macOS) hold C++ global state whose destructors race with
    // Tokio's thread-pool teardown and produce:
    //   libc++abi: terminating … mutex lock failed: EINVAL
    //
    // The abort fires *inside* server.await (during Actix worker cleanup),
    // so placing process::exit after server.await is too late.  Instead we
    // call process::exit(0) from the signal handler, after telling Actix to
    // stop accepting new connections but before waiting for workers to drop.
    // In-flight requests get a short grace window then we exit cleanly.
    let server_handle = server.handle();
    #[cfg(feature = "profiling")]
    let profiler_guard_for_signal = profiler_guard.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        tracing::info!("shutdown signal received, draining requests");
        // Tell Actix to close the listening socket and finish current requests.
        server_handle.stop(true).await;
        // Brief window for in-flight work to complete before we bypass C++ dtors.
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        tracing::info!("drain window elapsed, exiting");
        #[cfg(feature = "profiling")]
        {
            if let Ok(mut guard) = profiler_guard_for_signal.lock() {
                if let Some(g) = guard.take() {
                    if let Ok(report) = g.report().build() {
                        if let Ok(file) = std::fs::File::create("flamegraph.svg") {
                            report.flamegraph(file).ok();
                            tracing::info!("Flamegraph written to flamegraph.svg");
                        }
                    }
                }
            }
        }
        std::process::exit(0);
    });

    server.await?;
    Ok(())
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
            tracing::info!(signal = "ctrl_c", "shutdown signal received");
        },
        _ = terminate => {
            tracing::info!(signal = "sigterm", "shutdown signal received");
        },
    }
}

fn log_system_info() {
    let cpu_count = num_cpus::get();
    tracing::info!(cpu_cores = cpu_count, "cpu info");

    if let Ok(sys_info) = sys_info::mem_info() {
        let total_gb = sys_info.total as f64 / 1024.0 / 1024.0;
        let avail_gb = sys_info.avail as f64 / 1024.0 / 1024.0;
        tracing::info!(
            ram_total_gb = total_gb,
            ram_avail_gb = avail_gb,
            "memory info"
        );
    }

    #[cfg(target_os = "macos")]
    {
        if GpuManager::is_metal_available() {
            let gpu_manager = GpuManager::new();
            if let Some(metal_info) = gpu_manager.get_metal_info_string() {
                tracing::info!(metal_gpu = %metal_info, "metal available");
            } else {
                tracing::info!(metal_gpu = "apple silicon", "metal available");
            }
        } else {
            tracing::info!(metal = false, "metal not available");
        }
    }

    if cfg!(feature = "cuda") {
        if GpuManager::is_cuda_runtime_available() {
            if let Some(cuda_info) = GpuManager::get_cuda_info() {
                tracing::info!(cuda = true, cuda_info = %cuda_info, "cuda runtime available");
            } else {
                tracing::info!(cuda = true, "cuda runtime available");
            }
        } else {
            tracing::warn!(
                cuda_feature = true,
                cuda_runtime = false,
                "cuda feature enabled but runtime not detected"
            );
        }
    } else {
        #[cfg(not(target_os = "macos"))]
        tracing::info!(cuda = false, "cuda disabled");
    }

    if cfg!(feature = "onnx") {
        tracing::info!(onnx = true, "onnx enabled");
    } else {
        tracing::info!(onnx = false, "onnx disabled");
    }

    #[cfg(feature = "audio")]
    tracing::info!(audio = true, "audio processing enabled");
    #[cfg(not(feature = "audio"))]
    tracing::info!(audio = false, "audio processing disabled");

    #[cfg(feature = "image-security")]
    tracing::info!(image_security = true, "image security enabled");
    #[cfg(not(feature = "image-security"))]
    tracing::info!(image_security = false, "image security disabled");

    tracing::info!(
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        "os info"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises log_system_info() to cover lines 431-508.
    /// The function prints system information and logs CPU/GPU/memory details.
    /// It does not require any external runtime (ORT, torch, etc.).
    #[test]
    fn test_log_system_info_does_not_panic() {
        // Should complete without panicking on any platform
        log_system_info();
    }

    /// Calls log_system_info() multiple times to ensure it's idempotent.
    #[test]
    fn test_log_system_info_is_idempotent() {
        log_system_info();
        log_system_info();
    }

    #[test]
    fn resolve_worker_count_uses_num_cpus_when_zero() {
        assert_eq!(resolve_worker_count(4), 4);
        assert_eq!(resolve_worker_count(1), 1);
        let dynamic = resolve_worker_count(0);
        assert!(dynamic >= 1, "num_cpus should return at least 1");
    }
}
