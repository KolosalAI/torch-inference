#![allow(dead_code)]
use anyhow::Result;
use log::{debug, info, warn};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::models::registry::ModelMetadata;
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
            enabled: false,
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
        tensor_pool: Option<Arc<TensorPool>>,
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
        info!(
            "Loading ONNX model from: {:?} on device: {:?}",
            path, device_id
        );

        if !path.exists() {
            return Err(anyhow::anyhow!("Model file not found: {:?}", path));
        }

        let target_device = device_id.unwrap_or(self.device_id);
        let mut builder = Session::builder()?;

        // Build execution providers list based on availability and configuration
        let mut execution_providers = Vec::new();

        // 1. TensorRT (highest priority if enabled)
        if self.use_tensorrt {
            info!(
                "Configuring TensorRT execution provider on device {}...",
                target_device
            );

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
                trt_ep = trt_ep
                    .with_engine_cache(true)
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
                    .build(),
            );
        }

        // 4. CPU (Always fallback)
        execution_providers.push(ort::execution_providers::CPUExecutionProvider::default().build());

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
        _model: &LoadedOnnxModel,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ===== TensorRTPrecision tests =====

    #[test]
    fn test_tensorrt_precision_fp32_as_str() {
        assert_eq!(TensorRTPrecision::FP32.as_str(), "fp32");
    }

    #[test]
    fn test_tensorrt_precision_fp16_as_str() {
        assert_eq!(TensorRTPrecision::FP16.as_str(), "fp16");
    }

    #[test]
    fn test_tensorrt_precision_int8_as_str() {
        assert_eq!(TensorRTPrecision::INT8.as_str(), "int8");
    }

    #[test]
    fn test_tensorrt_precision_debug() {
        let precisions = [
            TensorRTPrecision::FP32,
            TensorRTPrecision::FP16,
            TensorRTPrecision::INT8,
        ];
        for p in &precisions {
            let s = format!("{:?}", p);
            assert!(!s.is_empty());
        }
        assert!(format!("{:?}", TensorRTPrecision::FP32).contains("FP32"));
        assert!(format!("{:?}", TensorRTPrecision::FP16).contains("FP16"));
        assert!(format!("{:?}", TensorRTPrecision::INT8).contains("INT8"));
    }

    #[test]
    fn test_tensorrt_precision_clone() {
        let original = TensorRTPrecision::FP16;
        let cloned = original.clone();
        assert_eq!(original, cloned);

        let int8 = TensorRTPrecision::INT8;
        assert_eq!(int8.clone(), TensorRTPrecision::INT8);
    }

    #[test]
    fn test_tensorrt_precision_partial_eq() {
        assert_eq!(TensorRTPrecision::FP32, TensorRTPrecision::FP32);
        assert_eq!(TensorRTPrecision::FP16, TensorRTPrecision::FP16);
        assert_eq!(TensorRTPrecision::INT8, TensorRTPrecision::INT8);
        assert_ne!(TensorRTPrecision::FP32, TensorRTPrecision::FP16);
        assert_ne!(TensorRTPrecision::FP16, TensorRTPrecision::INT8);
        assert_ne!(TensorRTPrecision::FP32, TensorRTPrecision::INT8);
    }

    // ===== TensorRTConfig tests =====

    #[test]
    fn test_tensorrt_config_default() {
        let config = TensorRTConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.precision, TensorRTPrecision::FP16);
        assert_eq!(config.workspace_size_mb, 2048);
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.optimization_level, 5);
        assert!(config.use_dynamic_shapes);
        assert_eq!(config.cache_dir.as_deref(), Some("./tensorrt_cache"));
    }

    #[test]
    fn test_tensorrt_config_custom_construction() {
        let config = TensorRTConfig {
            enabled: true,
            precision: TensorRTPrecision::INT8,
            workspace_size_mb: 4096,
            max_batch_size: 64,
            optimization_level: 3,
            cache_dir: Some("/tmp/trt_cache".to_string()),
            use_dynamic_shapes: false,
        };
        assert!(config.enabled);
        assert_eq!(config.precision, TensorRTPrecision::INT8);
        assert_eq!(config.workspace_size_mb, 4096);
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.optimization_level, 3);
        assert!(!config.use_dynamic_shapes);
        assert_eq!(config.cache_dir.as_deref(), Some("/tmp/trt_cache"));
    }

    #[test]
    fn test_tensorrt_config_no_cache_dir() {
        let config = TensorRTConfig {
            enabled: false,
            precision: TensorRTPrecision::FP32,
            workspace_size_mb: 1024,
            max_batch_size: 8,
            optimization_level: 1,
            cache_dir: None,
            use_dynamic_shapes: true,
        };
        assert!(config.cache_dir.is_none());
    }

    #[test]
    fn test_tensorrt_config_debug() {
        let config = TensorRTConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("TensorRTConfig"));
    }

    #[test]
    fn test_tensorrt_config_clone() {
        let config = TensorRTConfig {
            enabled: true,
            precision: TensorRTPrecision::FP32,
            workspace_size_mb: 512,
            max_batch_size: 16,
            optimization_level: 2,
            cache_dir: Some("/cache".to_string()),
            use_dynamic_shapes: false,
        };
        let cloned = config.clone();
        assert_eq!(cloned.enabled, config.enabled);
        assert_eq!(cloned.precision, config.precision);
        assert_eq!(cloned.workspace_size_mb, config.workspace_size_mb);
        assert_eq!(cloned.max_batch_size, config.max_batch_size);
        assert_eq!(cloned.optimization_level, config.optimization_level);
        assert_eq!(cloned.cache_dir, config.cache_dir);
        assert_eq!(cloned.use_dynamic_shapes, config.use_dynamic_shapes);
    }

    // ===== OnnxModelLoader::new() tests =====

    #[test]
    fn test_onnx_loader_new_no_tensorrt() {
        let loader = OnnxModelLoader::new(false, 0, None);
        assert!(!loader.use_tensorrt);
        assert_eq!(loader.device_id, 0);
        assert!(loader.tensor_pool.is_none());
        assert!(!loader.tensorrt_config.enabled);
    }

    #[test]
    fn test_onnx_loader_new_with_tensorrt() {
        let loader = OnnxModelLoader::new(true, 1, None);
        assert!(loader.use_tensorrt);
        assert_eq!(loader.device_id, 1);
        assert!(loader.tensorrt_config.enabled);
    }

    #[test]
    fn test_onnx_loader_new_device_id_zero() {
        let loader = OnnxModelLoader::new(false, 0, None);
        assert_eq!(loader.device_id, 0);
    }

    #[test]
    fn test_onnx_loader_new_intra_threads_nonzero() {
        let loader = OnnxModelLoader::new(false, 0, None);
        // Should be >= 1 (based on num_cpus)
        assert!(loader.intra_threads >= 1);
    }

    #[test]
    fn test_onnx_loader_new_env_fp32_precision() {
        // Set env var for precision
        std::env::set_var("TENSORRT_PRECISION", "fp32");
        let loader = OnnxModelLoader::new(true, 0, None);
        assert_eq!(loader.tensorrt_config.precision, TensorRTPrecision::FP32);
        std::env::remove_var("TENSORRT_PRECISION");
    }

    #[test]
    fn test_onnx_loader_new_env_int8_precision() {
        std::env::set_var("TENSORRT_PRECISION", "int8");
        let loader = OnnxModelLoader::new(true, 0, None);
        assert_eq!(loader.tensorrt_config.precision, TensorRTPrecision::INT8);
        std::env::remove_var("TENSORRT_PRECISION");
    }

    #[test]
    fn test_onnx_loader_new_env_fp16_precision_default() {
        // Unknown value should default to FP16
        std::env::set_var("TENSORRT_PRECISION", "unknown_value");
        let loader = OnnxModelLoader::new(false, 0, None);
        assert_eq!(loader.tensorrt_config.precision, TensorRTPrecision::FP16);
        std::env::remove_var("TENSORRT_PRECISION");
    }

    #[test]
    fn test_onnx_loader_new_env_workspace_size() {
        std::env::set_var("TENSORRT_WORKSPACE_SIZE_MB", "8192");
        let loader = OnnxModelLoader::new(false, 0, None);
        assert_eq!(loader.tensorrt_config.workspace_size_mb, 8192);
        std::env::remove_var("TENSORRT_WORKSPACE_SIZE_MB");
    }

    #[test]
    fn test_onnx_loader_new_env_max_batch_size() {
        std::env::set_var("TENSORRT_MAX_BATCH_SIZE", "128");
        let loader = OnnxModelLoader::new(false, 0, None);
        assert_eq!(loader.tensorrt_config.max_batch_size, 128);
        std::env::remove_var("TENSORRT_MAX_BATCH_SIZE");
    }

    #[test]
    fn test_onnx_loader_new_env_use_fp16_true() {
        std::env::set_var("USE_FP16", "1");
        let loader = OnnxModelLoader::new(false, 0, None);
        assert!(loader.use_fp16);
        std::env::remove_var("USE_FP16");
    }

    #[test]
    fn test_onnx_loader_new_env_use_fp16_false() {
        std::env::set_var("USE_FP16", "0");
        let loader = OnnxModelLoader::new(false, 0, None);
        assert!(!loader.use_fp16);
        std::env::remove_var("USE_FP16");
    }

    #[test]
    fn test_onnx_loader_new_env_use_fp16_true_string() {
        std::env::set_var("USE_FP16", "true");
        let loader = OnnxModelLoader::new(false, 0, None);
        assert!(loader.use_fp16);
        std::env::remove_var("USE_FP16");
    }

    // ===== OnnxModelLoader::with_tensorrt_config() tests =====

    #[test]
    fn test_onnx_loader_with_tensorrt_config_enabled() {
        let config = TensorRTConfig {
            enabled: true,
            precision: TensorRTPrecision::FP32,
            workspace_size_mb: 1024,
            max_batch_size: 16,
            optimization_level: 3,
            cache_dir: None,
            use_dynamic_shapes: false,
        };
        let loader = OnnxModelLoader::with_tensorrt_config(2, config, None);
        assert!(loader.use_tensorrt);
        assert_eq!(loader.device_id, 2);
        assert_eq!(loader.tensorrt_config.precision, TensorRTPrecision::FP32);
    }

    #[test]
    fn test_onnx_loader_with_tensorrt_config_disabled() {
        let config = TensorRTConfig {
            enabled: false,
            ..TensorRTConfig::default()
        };
        let loader = OnnxModelLoader::with_tensorrt_config(0, config, None);
        assert!(!loader.use_tensorrt);
        assert!(!loader.tensorrt_config.enabled);
    }

    #[test]
    fn test_onnx_loader_with_tensorrt_config_device_id() {
        let config = TensorRTConfig::default();
        let loader = OnnxModelLoader::with_tensorrt_config(5, config, None);
        assert_eq!(loader.device_id, 5);
    }

    // ===== OnnxModelLoader::load_model() error path =====

    #[test]
    fn test_onnx_loader_load_model_nonexistent_path() {
        let loader = OnnxModelLoader::new(false, 0, None);
        let path = std::path::Path::new("/nonexistent/model_xyz.onnx");
        let result = loader.load_model(path, None);
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("not found") || err.contains("nonexistent") || !err.is_empty());
    }

    // ===== load_model with existing-but-invalid file (covers EP setup paths) =====

    #[test]
    #[ignore = "requires libonnxruntime.dylib"]
    fn test_onnx_loader_load_model_invalid_file_cpu_only() {
        // Create a temp file with garbage content (not a valid ONNX model).
        // load_model will proceed past the `!path.exists()` check, execute all
        // execution-provider setup code (lines 153-232), and then fail at
        // commit_from_file (line 238) because the bytes are not valid ONNX.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"not a valid onnx model").unwrap();

        let loader = OnnxModelLoader::new(false, 0, None);
        let result = loader.load_model(tmp.path(), None);
        // Must fail — the content is not valid ONNX
        assert!(result.is_err(), "loading invalid ONNX bytes should fail");
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib"]
    fn test_onnx_loader_load_model_invalid_file_with_explicit_device_id() {
        // Exercise `device_id.unwrap_or(self.device_id)` branch (line 153) by
        // passing Some(device_id) explicitly.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"garbage").unwrap();

        let loader = OnnxModelLoader::new(false, 0, None);
        let result = loader.load_model(tmp.path(), Some(1));
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib"]
    fn test_onnx_loader_load_model_invalid_file_with_tensorrt() {
        // Exercise the TensorRT EP setup branch (lines 160-199) by creating
        // a loader with use_tensorrt = true. The TensorRT EP itself will not
        // be available in CI, but all configuration code still runs before
        // commit_from_file fails on the invalid bytes.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"invalid onnx bytes").unwrap();

        let loader = OnnxModelLoader::new(true, 0, None);
        let result = loader.load_model(tmp.path(), None);
        // Expect failure — invalid content
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib"]
    fn test_onnx_loader_load_model_tensorrt_fp32_precision() {
        // Exercise TensorRT FP32 branch (line 184)
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"fake").unwrap();

        std::env::set_var("TENSORRT_PRECISION", "fp32");
        let loader = OnnxModelLoader::new(true, 0, None);
        std::env::remove_var("TENSORRT_PRECISION");

        assert_eq!(loader.tensorrt_config.precision, TensorRTPrecision::FP32);
        let result = loader.load_model(tmp.path(), None);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib"]
    fn test_onnx_loader_load_model_tensorrt_int8_precision() {
        // Exercise TensorRT INT8 branch (lines 180-181)
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"fake").unwrap();

        std::env::set_var("TENSORRT_PRECISION", "int8");
        let loader = OnnxModelLoader::new(true, 0, None);
        std::env::remove_var("TENSORRT_PRECISION");

        assert_eq!(loader.tensorrt_config.precision, TensorRTPrecision::INT8);
        let result = loader.load_model(tmp.path(), None);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib"]
    fn test_onnx_loader_load_model_tensorrt_no_cache_dir() {
        // Exercise TensorRT with cache_dir = None (skip engine cache path)
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"fake").unwrap();

        let config = TensorRTConfig {
            enabled: true,
            precision: TensorRTPrecision::FP16,
            workspace_size_mb: 512,
            max_batch_size: 4,
            optimization_level: 3,
            cache_dir: None,
            use_dynamic_shapes: false,
        };
        let loader = OnnxModelLoader::with_tensorrt_config(0, config, None);
        let result = loader.load_model(tmp.path(), None);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib"]
    fn test_onnx_loader_load_model_device_id_none_uses_self_device() {
        // When device_id arg is None, target_device = self.device_id
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"garbage").unwrap();

        let loader = OnnxModelLoader::new(false, 2, None);
        assert_eq!(loader.device_id, 2);
        // Fails at commit_from_file, but device setup code ran
        let result = loader.load_model(tmp.path(), None);
        assert!(result.is_err());
    }
}
