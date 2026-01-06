use std::path::Path;
use anyhow::{Result, Context};
use log::{info, warn, debug};
use serde_json::Value;
use std::sync::Arc;

use ort::session::Session;
use ort::session::builder::{SessionBuilder, GraphOptimizationLevel};

use crate::models::registry::{ModelMetadata, PreprocessingConfig, PostprocessingConfig};
use crate::tensor_pool::TensorPool;

/// TensorRT optimization settings
#[derive(Debug, Clone)]
pub struct TensorRTConfig {
    pub enabled: bool,
    pub precision: TensorRTPrecision,
    pub workspace_size_mb: usize,
    pub max_batch_size: usize,
    pub optimization_level: u32,
    pub cache_dir: Option<String>,
    pub use_dynamic_shapes: bool,
}

impl Default for TensorRTConfig {
    fn default() -> Self {
        Self {
            enabled: true,  // Enable TensorRT by default
            precision: TensorRTPrecision::FP16,
            workspace_size_mb: 2048, // 2GB workspace
            max_batch_size: 32,
            optimization_level: 5,
            cache_dir: Some("./tensorrt_cache".to_string()),
            use_dynamic_shapes: true,
        }
    }
}

/// TensorRT precision modes
#[derive(Debug, Clone, PartialEq)]
pub enum TensorRTPrecision {
    FP32,
    FP16,
    INT8,
}

impl TensorRTPrecision {
    pub fn as_str(&self) -> &str {
        match self {
            TensorRTPrecision::FP32 => "fp32",
            TensorRTPrecision::FP16 => "fp16",
            TensorRTPrecision::INT8 => "int8",
        }
    }
}

/// ONNX model loader with TensorRT and CUDA optimizations
#[allow(dead_code)]
pub struct OnnxModelLoader {
    use_tensorrt: bool,
    device_id: usize,
    tensor_pool: Option<Arc<TensorPool>>,
    tensorrt_config: TensorRTConfig,
    use_fp16: bool,
    intra_threads: usize,
}

impl OnnxModelLoader {
    pub fn new(use_tensorrt: bool, device_id: usize, tensor_pool: Option<Arc<TensorPool>>) -> Self {
        // Parse TensorRT config from environment if available
        let tensorrt_precision = std::env::var("TENSORRT_PRECISION")
            .ok()
            .and_then(|p| match p.to_lowercase().as_str() {
                "fp32" => Some(TensorRTPrecision::FP32),
                "fp16" => Some(TensorRTPrecision::FP16),
                "int8" => Some(TensorRTPrecision::INT8),
                _ => None,
            })
            .unwrap_or(TensorRTPrecision::FP16);
        
        let workspace_size_mb = std::env::var("TENSORRT_WORKSPACE_SIZE_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048);
        
        let max_batch_size = std::env::var("TENSORRT_MAX_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        
        let use_fp16 = std::env::var("USE_FP16")
            .ok()
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(true);
        
        let intra_threads = num_cpus::get();
        
        let tensorrt_config = TensorRTConfig {
            enabled: use_tensorrt,
            precision: tensorrt_precision,
            workspace_size_mb,
            max_batch_size,
            optimization_level: 5,
            cache_dir: Some("./tensorrt_cache".to_string()),
            use_dynamic_shapes: true,
        };
        
        if use_tensorrt {
            info!("ONNX loader initialized with TensorRT optimization:");
            info!("  ✓ Precision: {:?}", tensorrt_config.precision);
            info!("  ✓ Workspace: {} MB", tensorrt_config.workspace_size_mb);
            info!("  ✓ Max batch size: {}", tensorrt_config.max_batch_size);
            info!("  ✓ Device: CUDA:{}", device_id);
        } else {
            info!("ONNX loader initialized (TensorRT disabled)");
        }
        
        Self {
            use_tensorrt,
            device_id,
            tensor_pool,
            tensorrt_config,
            use_fp16,
            intra_threads,
        }
    }
    
    /// Create a new loader with custom TensorRT configuration
    pub fn with_tensorrt_config(
        device_id: usize, 
        tensorrt_config: TensorRTConfig,
        tensor_pool: Option<Arc<TensorPool>>
    ) -> Self {
        let use_tensorrt = tensorrt_config.enabled;
        
        Self {
            use_tensorrt,
            device_id,
            tensor_pool,
            tensorrt_config,
            use_fp16: true,
            intra_threads: num_cpus::get(),
        }
    }

    pub fn load_model(&self, path: &Path, device_id: Option<usize>) -> Result<LoadedOnnxModel> {
        info!("Loading ONNX model from: {:?} on device: {:?}", path, device_id);

        if !path.exists() {
            return Err(anyhow::anyhow!("Model file not found: {:?}", path));
        }

        let target_device = device_id.unwrap_or(self.device_id);
        let mut builder = Session::builder()?;
        
        // Build execution providers list based on availability and configuration
        let mut execution_providers = Vec::new();

        // 1. TensorRT (highest priority if enabled)
        if self.use_tensorrt {
            info!("Configuring TensorRT execution provider on device {}...", target_device);
            
            // Create TensorRT cache directory if it doesn't exist
            if let Some(cache_dir) = &self.tensorrt_config.cache_dir {
                if let Err(e) = std::fs::create_dir_all(cache_dir) {
                    warn!("Failed to create TensorRT cache directory: {}", e);
                }
            }
            
            let mut trt_ep = ort::execution_providers::TensorRTExecutionProvider::default()
                .with_device_id(target_device as i32);
            
            // Configure TensorRT precision
            match self.tensorrt_config.precision {
                TensorRTPrecision::FP16 => {
                    trt_ep = trt_ep.with_fp16(true);
                    info!("  ✓ TensorRT FP16 mode enabled");
                }
                TensorRTPrecision::INT8 => {
                    trt_ep = trt_ep.with_int8(true);
                    info!("  ✓ TensorRT INT8 mode enabled (requires calibration)");
                }
                TensorRTPrecision::FP32 => {
                    info!("  ✓ TensorRT FP32 mode (baseline precision)");
                }
            }
            
            // Set engine cache path for faster subsequent loads
            if let Some(cache_dir) = &self.tensorrt_config.cache_dir {
                trt_ep = trt_ep.with_engine_cache(true)
                    .with_engine_cache_path(cache_dir);
                info!("  ✓ TensorRT engine cache: {}", cache_dir);
            }
            
            // Set timing cache for faster optimization
            trt_ep = trt_ep.with_timing_cache(true);
            
            execution_providers.push(trt_ep.build());
            info!("  ✓ TensorRT execution provider configured");
        }

        // 2. CUDA (NVIDIA GPUs) - fallback if TensorRT is not available
        {
            let cuda_ep = ort::execution_providers::CUDAExecutionProvider::default()
                .with_device_id(target_device as i32);
            
            // Note: Additional CUDA EP options can be configured here
            // .with_arena_extend_strategy(ArenaExtendStrategy::NextPowerOfTwo)
            // .with_cudnn_conv_algo_search(CuDNNConvAlgoSearch::Exhaustive)
            
            execution_providers.push(cuda_ep.build());
            debug!("  ✓ CUDA execution provider configured as fallback");
        }

        // 3. CoreML (Apple Silicon / macOS)
        #[cfg(target_os = "macos")]
        {
            info!("Enabling CoreML execution provider");
            execution_providers.push(
                ort::execution_providers::CoreMLExecutionProvider::default()
                    .with_subgraphs(true)
                    .build()
            );
        }

        // 4. CPU (Always fallback)
        execution_providers.push(
            ort::execution_providers::CPUExecutionProvider::default().build()
        );

        // Register providers
        builder = builder.with_execution_providers(execution_providers)?;

        // Set graph optimization level (Level3 = all optimizations)
        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads)?
            .commit_from_file(path)?;

        info!("Successfully loaded ONNX model with optimized execution providers");

        Ok(LoadedOnnxModel {
            session,
            use_tensorrt: self.use_tensorrt,
            precision: self.tensorrt_config.precision.clone(),
        })
    }

    pub fn infer(
        &self,
        model: &LoadedOnnxModel,
        input: &Value,
        metadata: &ModelMetadata,
    ) -> Result<Value> {
        info!("Running ONNX inference for model: {}", metadata.name);

        // TODO: Implement proper input tensor conversion
        // For now, this is a placeholder that requires implementation based on input schema
        
        // Example of running inference (simplified)
        // let inputs = inputs![...];
        // let outputs = model.session.run(inputs)?;
        
        // For now, just return the input as a mock result
        Ok(input.clone())
    }
}

#[allow(dead_code)]
pub struct LoadedOnnxModel {
    session: Session,
    use_tensorrt: bool,
    precision: TensorRTPrecision,
}

impl LoadedOnnxModel {
    /// Check if this model is using TensorRT acceleration
    pub fn is_tensorrt_accelerated(&self) -> bool {
        self.use_tensorrt
    }
    
    /// Get the precision mode being used
    pub fn precision(&self) -> &TensorRTPrecision {
        &self.precision
    }
}

// SAFETY: Session is thread-safe
unsafe impl Send for LoadedOnnxModel {}
unsafe impl Sync for LoadedOnnxModel {}
