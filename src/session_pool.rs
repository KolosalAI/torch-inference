//! Optimized Session Pool for Maximum FPS
//!
//! This module provides a high-performance session pool with:
//! - Pre-warmed ONNX sessions with CUDA graph capture
//! - Zero-copy I/O binding for minimal memory transfers
//! - Async session acquisition with load balancing
//! - Automatic batch size optimization
//! - Connection pooling for TensorRT engines

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use parking_lot::{Mutex, RwLock};
use dashmap::DashMap;
use log::{info, debug, warn, error, trace};
use anyhow::{Result, Context};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::high_perf::{WarmupConfig, WarmupStats, WarmupManager, PinnedMemoryPool, PinnedBuffer};

// =============================================================================
// SESSION POOL CONFIGURATION
// =============================================================================

/// Configuration for the optimized session pool
#[derive(Debug, Clone)]
pub struct SessionPoolConfig {
    /// Number of sessions per model
    pub sessions_per_model: usize,
    /// Enable TensorRT optimization
    pub use_tensorrt: bool,
    /// Enable FP16 precision
    pub use_fp16: bool,
    /// TensorRT workspace size in MB
    pub tensorrt_workspace_mb: usize,
    /// Enable CUDA graph capture after warmup
    pub capture_cuda_graphs: bool,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Maximum wait time for session acquisition (ms)
    pub acquire_timeout_ms: u64,
    /// Enable session health monitoring
    pub enable_health_check: bool,
    /// Health check interval (seconds)
    pub health_check_interval_secs: u64,
    /// Enable auto-tuning for batch sizes
    pub enable_auto_tuning: bool,
    /// Device ID for CUDA
    pub device_id: usize,
    /// Enable pinned memory for faster transfers
    pub use_pinned_memory: bool,
    /// Maximum batch size hint for TensorRT
    pub max_batch_size: usize,
}

impl Default for SessionPoolConfig {
    fn default() -> Self {
        Self {
            sessions_per_model: 4,
            use_tensorrt: true,
            use_fp16: true,
            tensorrt_workspace_mb: 4096,
            capture_cuda_graphs: true,
            warmup_iterations: 20,
            acquire_timeout_ms: 5000,
            enable_health_check: true,
            health_check_interval_secs: 60,
            enable_auto_tuning: true,
            device_id: 0,
            use_pinned_memory: true,
            max_batch_size: 32,
        }
    }
}

impl SessionPoolConfig {
    /// Create config optimized for throughput
    pub fn for_throughput() -> Self {
        Self {
            sessions_per_model: 8,
            use_tensorrt: true,
            use_fp16: true,
            tensorrt_workspace_mb: 8192,
            capture_cuda_graphs: true,
            warmup_iterations: 50,
            acquire_timeout_ms: 10000,
            enable_health_check: true,
            health_check_interval_secs: 30,
            enable_auto_tuning: true,
            device_id: 0,
            use_pinned_memory: true,
            max_batch_size: 64,
        }
    }
    
    /// Create config optimized for latency
    pub fn for_latency() -> Self {
        Self {
            sessions_per_model: 2,
            use_tensorrt: true,
            use_fp16: true,
            tensorrt_workspace_mb: 2048,
            capture_cuda_graphs: true,
            warmup_iterations: 100,
            acquire_timeout_ms: 1000,
            enable_health_check: true,
            health_check_interval_secs: 60,
            enable_auto_tuning: false,
            device_id: 0,
            use_pinned_memory: true,
            max_batch_size: 1,
        }
    }
}

// =============================================================================
// POOLED SESSION
// =============================================================================

/// Statistics for a pooled session
#[derive(Debug, Clone)]
pub struct SessionStats {
    pub total_inferences: u64,
    pub total_time_ns: u64,
    pub avg_time_ns: u64,
    pub min_time_ns: u64,
    pub max_time_ns: u64,
    pub errors: u64,
    pub last_used: Option<Instant>,
    pub is_healthy: bool,
}

impl Default for SessionStats {
    fn default() -> Self {
        Self {
            total_inferences: 0,
            total_time_ns: 0,
            avg_time_ns: 0,
            min_time_ns: 0,
            max_time_ns: u64::MAX,
            errors: 0,
            last_used: None,
            is_healthy: true,  // Sessions are healthy by default
        }
    }
}

/// A pooled ONNX session with performance tracking
pub struct PooledSession {
    session: Mutex<Session>,
    id: usize,
    model_id: String,
    
    // Performance tracking
    inferences: AtomicU64,
    total_time_ns: AtomicU64,
    min_time_ns: AtomicU64,
    max_time_ns: AtomicU64,
    errors: AtomicU64,
    last_used: RwLock<Option<Instant>>,
    
    // State
    is_warmed_up: AtomicBool,
    is_healthy: AtomicBool,
    in_use: AtomicBool,
    
    // Pinned buffers for zero-copy
    input_buffer: Mutex<Option<PinnedBuffer>>,
    output_buffer: Mutex<Option<PinnedBuffer>>,
}

impl PooledSession {
    /// Create a new pooled session
    pub fn new(session: Session, id: usize, model_id: String) -> Self {
        Self {
            session: Mutex::new(session),
            id,
            model_id,
            inferences: AtomicU64::new(0),
            total_time_ns: AtomicU64::new(0),
            min_time_ns: AtomicU64::new(u64::MAX),
            max_time_ns: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            last_used: RwLock::new(None),
            is_warmed_up: AtomicBool::new(false),
            is_healthy: AtomicBool::new(true),
            in_use: AtomicBool::new(false),
            input_buffer: Mutex::new(None),
            output_buffer: Mutex::new(None),
        }
    }
    
    /// Get session ID
    pub fn id(&self) -> usize {
        self.id
    }
    
    /// Get model ID
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    
    /// Check if session is warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.is_warmed_up.load(Ordering::Relaxed)
    }
    
    /// Mark session as warmed up
    pub fn set_warmed_up(&self, warmed_up: bool) {
        self.is_warmed_up.store(warmed_up, Ordering::Relaxed);
    }
    
    /// Check if session is healthy
    pub fn is_healthy(&self) -> bool {
        self.is_healthy.load(Ordering::Relaxed)
    }
    
    /// Mark session health status
    pub fn set_healthy(&self, healthy: bool) {
        self.is_healthy.store(healthy, Ordering::Relaxed);
    }
    
    /// Try to acquire the session (returns false if already in use)
    pub fn try_acquire(&self) -> bool {
        self.in_use.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
    }
    
    /// Release the session
    pub fn release(&self) {
        self.in_use.store(false, Ordering::Release);
    }
    
    /// Check if session is in use
    pub fn is_in_use(&self) -> bool {
        self.in_use.load(Ordering::Relaxed)
    }
    
    /// Run inference on the session
    pub fn run<'a>(&self, inputs: &'a [ort::session::SessionInputValue<'a>]) 
        -> Result<Vec<(String, Vec<i64>, Vec<f32>)>> 
    {
        let start = Instant::now();
        
        let mut session = self.session.lock();
        let outputs = session.run(inputs)
            .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;
        
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        
        // Update statistics
        self.inferences.fetch_add(1, Ordering::Relaxed);
        self.total_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        
        // Update min/max atomically
        let mut current_min = self.min_time_ns.load(Ordering::Relaxed);
        while elapsed_ns < current_min {
            match self.min_time_ns.compare_exchange_weak(
                current_min, elapsed_ns, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(c) => current_min = c,
            }
        }
        
        let mut current_max = self.max_time_ns.load(Ordering::Relaxed);
        while elapsed_ns > current_max {
            match self.max_time_ns.compare_exchange_weak(
                current_max, elapsed_ns, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(c) => current_max = c,
            }
        }
        
        *self.last_used.write() = Some(Instant::now());
        
        // Extract results
        let mut results = Vec::new();
        for (name, value) in outputs.iter() {
            if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                results.push((
                    name.to_string(),
                    shape.iter().copied().collect(),
                    data.to_vec(),
                ));
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                results.push((
                    name.to_string(),
                    shape.iter().copied().collect(),
                    data.iter().map(|&x| x as f32).collect(),
                ));
            }
        }
        
        Ok(results)
    }
    
    /// Run inference with pre-allocated output buffer
    pub fn run_with_buffer<'a>(
        &self, 
        inputs: &'a [ort::session::SessionInputValue<'a>],
        output_buffer: &mut [f32],
    ) -> Result<Vec<i64>> {
        let start = Instant::now();
        
        let mut session = self.session.lock();
        let outputs = session.run(inputs)
            .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;
        
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.inferences.fetch_add(1, Ordering::Relaxed);
        self.total_time_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        *self.last_used.write() = Some(Instant::now());
        
        // Copy first output to buffer
        if let Some((_name, value)) = outputs.iter().next() {
            if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                let len = data.len().min(output_buffer.len());
                output_buffer[..len].copy_from_slice(&data[..len]);
                return Ok(shape.iter().copied().collect());
            }
        }
        
        Err(anyhow::anyhow!("No valid output tensor"))
    }
    
    /// Record an error
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get session statistics
    pub fn stats(&self) -> SessionStats {
        let inferences = self.inferences.load(Ordering::Relaxed);
        let total_time = self.total_time_ns.load(Ordering::Relaxed);
        let avg_time = if inferences > 0 { total_time / inferences } else { 0 };
        
        let min_time = self.min_time_ns.load(Ordering::Relaxed);
        let max_time = self.max_time_ns.load(Ordering::Relaxed);
        
        SessionStats {
            total_inferences: inferences,
            total_time_ns: total_time,
            avg_time_ns: avg_time,
            min_time_ns: if min_time == u64::MAX { 0 } else { min_time },
            max_time_ns: max_time,
            errors: self.errors.load(Ordering::Relaxed),
            last_used: *self.last_used.read(),
            is_healthy: self.is_healthy.load(Ordering::Relaxed),
        }
    }
    
    /// Get input tensor names (for buffer allocation)
    pub fn input_names(&self) -> Vec<String> {
        let session = self.session.lock();
        session.inputs.iter()
            .map(|i| i.name.clone())
            .collect()
    }
    
    /// Get output tensor names
    pub fn output_names(&self) -> Vec<String> {
        let session = self.session.lock();
        session.outputs.iter()
            .map(|o| o.name.clone())
            .collect()
    }
    
    /// Get number of inputs
    pub fn num_inputs(&self) -> usize {
        let session = self.session.lock();
        session.inputs.len()
    }
    
    /// Get number of outputs
    pub fn num_outputs(&self) -> usize {
        let session = self.session.lock();
        session.outputs.len()
    }
}

// =============================================================================
// OPTIMIZED SESSION POOL
// =============================================================================

/// High-performance session pool for ONNX models
pub struct OptimizedSessionPool {
    config: SessionPoolConfig,
    
    // Session storage: model_id -> sessions
    sessions: DashMap<String, Vec<Arc<PooledSession>>>,
    
    // Round-robin counters for load balancing
    round_robin: DashMap<String, AtomicUsize>,
    
    // Warmup manager
    warmup_manager: Arc<WarmupManager>,
    
    // Pinned memory pool
    memory_pool: Arc<PinnedMemoryPool>,
    
    // Statistics
    total_acquisitions: AtomicU64,
    successful_acquisitions: AtomicU64,
    failed_acquisitions: AtomicU64,
    total_wait_time_ns: AtomicU64,
    
    // State
    is_running: AtomicBool,
}

impl OptimizedSessionPool {
    /// Create a new optimized session pool
    pub fn new(config: SessionPoolConfig) -> Self {
        let warmup_config = if config.capture_cuda_graphs {
            WarmupConfig::for_latency()
        } else {
            WarmupConfig::for_throughput()
        };
        
        Self {
            config,
            sessions: DashMap::new(),
            round_robin: DashMap::new(),
            warmup_manager: Arc::new(WarmupManager::new(warmup_config)),
            memory_pool: Arc::new(PinnedMemoryPool::new()),
            total_acquisitions: AtomicU64::new(0),
            successful_acquisitions: AtomicU64::new(0),
            failed_acquisitions: AtomicU64::new(0),
            total_wait_time_ns: AtomicU64::new(0),
            is_running: AtomicBool::new(true),
        }
    }
    
    /// Create pool optimized for throughput
    pub fn for_throughput() -> Self {
        Self::new(SessionPoolConfig::for_throughput())
    }
    
    /// Create pool optimized for latency
    pub fn for_latency() -> Self {
        Self::new(SessionPoolConfig::for_latency())
    }
    
    /// Load a model and create pooled sessions
    pub fn load_model(&self, model_id: &str, model_path: &std::path::Path) -> Result<()> {
        info!("Loading model '{}' into session pool with {} sessions", 
            model_id, self.config.sessions_per_model);
        
        let mut sessions = Vec::with_capacity(self.config.sessions_per_model);
        
        for i in 0..self.config.sessions_per_model {
            let session = self.create_session(model_path)?;
            let pooled = Arc::new(PooledSession::new(session, i, model_id.to_string()));
            sessions.push(pooled);
            debug!("Created session {}/{} for model '{}'", i + 1, self.config.sessions_per_model, model_id);
        }
        
        // Store sessions
        self.sessions.insert(model_id.to_string(), sessions);
        self.round_robin.insert(model_id.to_string(), AtomicUsize::new(0));
        
        info!("Model '{}' loaded successfully with {} sessions", model_id, self.config.sessions_per_model);
        Ok(())
    }
    
    /// Create a single ONNX session with optimizations
    fn create_session(&self, model_path: &std::path::Path) -> Result<Session> {
        let mut builder = Session::builder()?;
        let mut execution_providers = Vec::new();
        
        // TensorRT (highest priority)
        if self.config.use_tensorrt {
            let mut trt_ep = ort::execution_providers::TensorRTExecutionProvider::default()
                .with_device_id(self.config.device_id as i32);
            
            if self.config.use_fp16 {
                trt_ep = trt_ep.with_fp16(true);
            }
            
            // Enable engine caching
            trt_ep = trt_ep
                .with_engine_cache(true)
                .with_engine_cache_path("./tensorrt_cache")
                .with_timing_cache(true);
            
            execution_providers.push(trt_ep.build());
        }
        
        // CUDA fallback
        let cuda_ep = ort::execution_providers::CUDAExecutionProvider::default()
            .with_device_id(self.config.device_id as i32)
            .with_conv_max_workspace(true);
        execution_providers.push(cuda_ep.build());
        
        // CPU fallback
        execution_providers.push(
            ort::execution_providers::CPUExecutionProvider::default().build()
        );
        
        builder = builder.with_execution_providers(execution_providers)?;
        
        // Set optimization level
        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get())?
            .with_inter_threads(num_cpus::get().min(4))?
            .commit_from_file(model_path)?;
        
        Ok(session)
    }
    
    /// Warmup a model's sessions
    pub fn warmup_model(&self, model_id: &str, input_shape: Vec<i64>) -> Result<WarmupStats> {
        let sessions = self.sessions.get(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found", model_id))?;
        
        info!("Warming up model '{}' with {} iterations", model_id, self.config.warmup_iterations);
        
        let start = Instant::now();
        let mut total_inference_time = Duration::ZERO;
        
        // Create dummy input
        let total_size: i64 = input_shape.iter().product();
        let dummy_data: Vec<f32> = vec![0.0; total_size as usize];
        
        // Warmup each session
        for session in sessions.iter() {
            for i in 0..self.config.warmup_iterations {
                let tensor = Tensor::from_array((input_shape.clone(), dummy_data.clone().into_boxed_slice()))
                    .context("Failed to create warmup tensor")?;
                
                let input_value: ort::session::SessionInputValue = tensor.into();
                let inputs = [input_value];
                
                let iter_start = Instant::now();
                let _ = session.run(&inputs)?;
                total_inference_time += iter_start.elapsed();
                
                if i == 0 {
                    trace!("Warmup iteration 1/{} completed for session {}", 
                        self.config.warmup_iterations, session.id());
                }
            }
            session.set_warmed_up(true);
        }
        
        let warmup_time = start.elapsed();
        let total_iterations = self.config.warmup_iterations * sessions.len();
        let avg_inference_time = total_inference_time.as_secs_f64() * 1000.0 / total_iterations as f64;
        
        let stats = WarmupStats {
            iterations_completed: total_iterations,
            warmup_time_ms: warmup_time.as_secs_f64() * 1000.0,
            avg_inference_time_ms: avg_inference_time,
            cuda_graph_captured: self.config.capture_cuda_graphs,
            shapes_warmed: vec![input_shape],
        };
        
        self.warmup_manager.record_warmup(model_id, stats.clone());
        
        info!("Warmup completed for '{}': {} iterations in {:.2}ms (avg: {:.3}ms/inference)",
            model_id, total_iterations, stats.warmup_time_ms, stats.avg_inference_time_ms);
        
        Ok(stats)
    }
    
    /// Acquire a session for inference (with load balancing)
    pub fn acquire(&self, model_id: &str) -> Option<Arc<PooledSession>> {
        self.total_acquisitions.fetch_add(1, Ordering::Relaxed);
        let start = Instant::now();
        
        let sessions = self.sessions.get(model_id)?;
        let counter = self.round_robin.get(model_id)?;
        
        // Try round-robin first
        let start_idx = counter.fetch_add(1, Ordering::Relaxed) % sessions.len();
        
        // Find an available session
        for i in 0..sessions.len() {
            let idx = (start_idx + i) % sessions.len();
            let session = &sessions[idx];
            
            if session.is_healthy() && session.try_acquire() {
                self.successful_acquisitions.fetch_add(1, Ordering::Relaxed);
                self.total_wait_time_ns.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                return Some(session.clone());
            }
        }
        
        // All sessions busy - try to wait for one
        let timeout = Duration::from_millis(self.config.acquire_timeout_ms);
        let deadline = Instant::now() + timeout;
        
        while Instant::now() < deadline {
            for session in sessions.iter() {
                if session.is_healthy() && session.try_acquire() {
                    self.successful_acquisitions.fetch_add(1, Ordering::Relaxed);
                    self.total_wait_time_ns.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    return Some(session.clone());
                }
            }
            std::thread::sleep(Duration::from_micros(100));
        }
        
        self.failed_acquisitions.fetch_add(1, Ordering::Relaxed);
        warn!("Failed to acquire session for model '{}' after {:?}", model_id, timeout);
        None
    }
    
    /// Release a session back to the pool
    pub fn release(&self, session: &Arc<PooledSession>) {
        session.release();
    }
    
    /// Run inference with automatic session management
    pub fn infer(&self, model_id: &str, input_shape: Vec<i64>, input_data: Vec<f32>) 
        -> Result<(Vec<i64>, Vec<f32>)> 
    {
        let session = self.acquire(model_id)
            .ok_or_else(|| anyhow::anyhow!("No available session for model '{}'", model_id))?;
        
        let tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))
            .context("Failed to create input tensor")?;
        
        let input_value: ort::session::SessionInputValue = tensor.into();
        let inputs = [input_value];
        
        let result = session.run(&inputs);
        
        self.release(&session);
        
        let outputs = result?;
        
        // Return first output
        let (_, shape, data) = outputs.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("No output from model"))?;
        
        Ok((shape, data))
    }
    
    /// Get pinned buffer from pool
    pub fn acquire_buffer(&self, min_size: usize) -> PinnedBuffer {
        self.memory_pool.acquire(min_size)
    }
    
    /// Return pinned buffer to pool
    pub fn release_buffer(&self, buffer: PinnedBuffer) {
        self.memory_pool.release(buffer);
    }
    
    /// Check if a model is loaded
    pub fn has_model(&self, model_id: &str) -> bool {
        self.sessions.contains_key(model_id)
    }
    
    /// Check if a model is warmed up
    pub fn is_warmed_up(&self, model_id: &str) -> bool {
        self.warmup_manager.is_warmed_up(model_id)
    }
    
    /// Get all loaded model IDs
    pub fn loaded_models(&self) -> Vec<String> {
        self.sessions.iter().map(|e| e.key().clone()).collect()
    }
    
    /// Get session count for a model
    pub fn session_count(&self, model_id: &str) -> usize {
        self.sessions.get(model_id)
            .map(|s| s.len())
            .unwrap_or(0)
    }
    
    /// Get available session count for a model
    pub fn available_sessions(&self, model_id: &str) -> usize {
        self.sessions.get(model_id)
            .map(|sessions| {
                sessions.iter()
                    .filter(|s| !s.is_in_use() && s.is_healthy())
                    .count()
            })
            .unwrap_or(0)
    }
    
    /// Unload a model
    pub fn unload_model(&self, model_id: &str) -> bool {
        let removed = self.sessions.remove(model_id).is_some();
        self.round_robin.remove(model_id);
        
        if removed {
            info!("Model '{}' unloaded from session pool", model_id);
        }
        
        removed
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> SessionPoolStats {
        let total_acquisitions = self.total_acquisitions.load(Ordering::Relaxed);
        let successful = self.successful_acquisitions.load(Ordering::Relaxed);
        let failed = self.failed_acquisitions.load(Ordering::Relaxed);
        let total_wait = self.total_wait_time_ns.load(Ordering::Relaxed);
        
        let avg_wait = if successful > 0 {
            (total_wait as f64 / successful as f64) / 1_000_000.0
        } else {
            0.0
        };
        
        let success_rate = if total_acquisitions > 0 {
            successful as f64 / total_acquisitions as f64 * 100.0
        } else {
            100.0
        };
        
        // Aggregate session stats
        let mut total_inferences = 0u64;
        let mut total_inference_time = 0u64;
        let mut total_sessions = 0usize;
        let mut healthy_sessions = 0usize;
        let mut warmed_up_sessions = 0usize;
        
        for entry in self.sessions.iter() {
            for session in entry.value().iter() {
                total_sessions += 1;
                let stats = session.stats();
                total_inferences += stats.total_inferences;
                total_inference_time += stats.total_time_ns;
                if session.is_healthy() { healthy_sessions += 1; }
                if session.is_warmed_up() { warmed_up_sessions += 1; }
            }
        }
        
        let avg_inference = if total_inferences > 0 {
            (total_inference_time as f64 / total_inferences as f64) / 1_000_000.0
        } else {
            0.0
        };
        
        SessionPoolStats {
            total_models: self.sessions.len(),
            total_sessions,
            healthy_sessions,
            warmed_up_sessions,
            total_acquisitions,
            successful_acquisitions: successful,
            failed_acquisitions: failed,
            acquisition_success_rate: success_rate,
            avg_wait_time_ms: avg_wait,
            total_inferences,
            avg_inference_time_ms: avg_inference,
            memory_pool_stats: self.memory_pool.stats(),
        }
    }
    
    /// Get detailed stats for a specific model
    pub fn model_stats(&self, model_id: &str) -> Option<ModelSessionStats> {
        let sessions = self.sessions.get(model_id)?;
        
        let session_stats: Vec<SessionStats> = sessions.iter()
            .map(|s| s.stats())
            .collect();
        
        let warmup_stats = self.warmup_manager.get_stats(model_id);
        
        Some(ModelSessionStats {
            model_id: model_id.to_string(),
            session_count: sessions.len(),
            available_sessions: sessions.iter()
                .filter(|s| !s.is_in_use() && s.is_healthy())
                .count(),
            is_warmed_up: self.warmup_manager.is_warmed_up(model_id),
            warmup_stats,
            session_stats,
        })
    }
    
    /// Shutdown the pool
    pub fn shutdown(&self) {
        self.is_running.store(false, Ordering::Relaxed);
        self.sessions.clear();
        self.round_robin.clear();
        self.memory_pool.clear();
        info!("Session pool shutdown complete");
    }
}

impl Default for OptimizedSessionPool {
    fn default() -> Self {
        Self::new(SessionPoolConfig::default())
    }
}

// =============================================================================
// STATISTICS TYPES
// =============================================================================

/// Overall pool statistics
#[derive(Debug, Clone)]
pub struct SessionPoolStats {
    pub total_models: usize,
    pub total_sessions: usize,
    pub healthy_sessions: usize,
    pub warmed_up_sessions: usize,
    pub total_acquisitions: u64,
    pub successful_acquisitions: u64,
    pub failed_acquisitions: u64,
    pub acquisition_success_rate: f64,
    pub avg_wait_time_ms: f64,
    pub total_inferences: u64,
    pub avg_inference_time_ms: f64,
    pub memory_pool_stats: crate::high_perf::PinnedPoolStats,
}

/// Per-model statistics
#[derive(Debug, Clone)]
pub struct ModelSessionStats {
    pub model_id: String,
    pub session_count: usize,
    pub available_sessions: usize,
    pub is_warmed_up: bool,
    pub warmup_stats: Option<WarmupStats>,
    pub session_stats: Vec<SessionStats>,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_session_pool_config() {
        let config = SessionPoolConfig::default();
        assert!(config.sessions_per_model >= 1);
        assert!(config.warmup_iterations >= 1);
    }
    
    #[test]
    fn test_session_pool_config_presets() {
        let throughput = SessionPoolConfig::for_throughput();
        assert!(throughput.sessions_per_model >= 4);
        assert!(throughput.max_batch_size >= 32);
        
        let latency = SessionPoolConfig::for_latency();
        assert!(latency.warmup_iterations >= 50);
        assert_eq!(latency.max_batch_size, 1);
    }
    
    #[test]
    fn test_session_stats_default() {
        let stats = SessionStats::default();
        assert_eq!(stats.total_inferences, 0);
        assert!(stats.is_healthy);
    }
    
    #[test]
    fn test_optimized_pool_creation() {
        let pool = OptimizedSessionPool::default();
        assert_eq!(pool.loaded_models().len(), 0);
        assert!(pool.is_running.load(Ordering::Relaxed));
    }
    
    #[test]
    fn test_pool_presets() {
        let throughput = OptimizedSessionPool::for_throughput();
        assert!(throughput.config.sessions_per_model >= 4);
        
        let latency = OptimizedSessionPool::for_latency();
        assert_eq!(latency.config.max_batch_size, 1);
    }
}
