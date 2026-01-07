use std::path::Path;
use std::sync::Mutex;
use anyhow::{Result, Context};
use log::{info, warn, debug};
use serde_json::Value;
use std::sync::Arc;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::models::registry::{ModelMetadata, PreprocessingConfig, PostprocessingConfig};
use crate::tensor_pool::TensorPool;
use crate::core::cuda_optimizer::{CudaOptimizer, CudaOptimizerConfig, ComputePrecision};

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
    /// Enable builder optimization level (0-5, higher = more optimization)
    pub builder_optimization_level: u32,
    /// Number of auxiliary streams for TensorRT
    pub auxiliary_streams: usize,
    /// Enable layer normalization plugin
    pub enable_layer_norm_plugin: bool,
    /// Enable GELU plugin for transformer models
    pub enable_gelu_plugin: bool,
    /// Enable sparse weights for applicable models
    pub enable_sparse_weights: bool,
}

impl Default for TensorRTConfig {
    fn default() -> Self {
        Self {
            enabled: true,  // Enable TensorRT by default
            precision: TensorRTPrecision::FP16,
            workspace_size_mb: 4096, // 4GB workspace for larger models
            max_batch_size: 64,      // Increased for better throughput
            optimization_level: 5,    // Maximum optimization
            cache_dir: Some("./tensorrt_cache".to_string()),
            use_dynamic_shapes: true,
            builder_optimization_level: 5,
            auxiliary_streams: 4,
            enable_layer_norm_plugin: true,
            enable_gelu_plugin: true,
            enable_sparse_weights: false, // Disable by default (requires sparse model)
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
    inter_threads: usize,
    cuda_optimizer: Option<CudaOptimizer>,
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
            .unwrap_or(4096); // Increased default workspace
        
        let max_batch_size = std::env::var("TENSORRT_MAX_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64); // Increased default batch size
        
        let use_fp16 = std::env::var("USE_FP16")
            .ok()
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(true);
        
        // Optimize thread counts based on CPU
        let num_cpus = num_cpus::get();
        let intra_threads = num_cpus;
        let inter_threads = (num_cpus / 2).max(1);
        
        let tensorrt_config = TensorRTConfig {
            enabled: use_tensorrt,
            precision: tensorrt_precision,
            workspace_size_mb,
            max_batch_size,
            optimization_level: 5,
            cache_dir: Some("./tensorrt_cache".to_string()),
            use_dynamic_shapes: true,
            builder_optimization_level: 5,
            auxiliary_streams: 4,
            enable_layer_norm_plugin: true,
            enable_gelu_plugin: true,
            enable_sparse_weights: false,
        };
        
        // Create CUDA optimizer for maximum performance
        let cuda_optimizer = if use_tensorrt {
            Some(CudaOptimizer::for_throughput())
        } else {
            Some(CudaOptimizer::for_latency())
        };
        
        if use_tensorrt {
            info!("ONNX loader initialized with TensorRT optimization:");
            info!("  ✓ Precision: {:?}", tensorrt_config.precision);
            info!("  ✓ Workspace: {} MB", tensorrt_config.workspace_size_mb);
            info!("  ✓ Max batch size: {}", tensorrt_config.max_batch_size);
            info!("  ✓ Builder optimization level: {}", tensorrt_config.builder_optimization_level);
            info!("  ✓ Auxiliary streams: {}", tensorrt_config.auxiliary_streams);
            info!("  ✓ Device: CUDA:{}", device_id);
            info!("  ✓ Intra-op threads: {}", intra_threads);
            info!("  ✓ Inter-op threads: {}", inter_threads);
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
            inter_threads,
            cuda_optimizer,
        }
    }
    
    /// Create a new loader with custom TensorRT configuration
    pub fn with_tensorrt_config(
        device_id: usize, 
        tensorrt_config: TensorRTConfig,
        tensor_pool: Option<Arc<TensorPool>>
    ) -> Self {
        let use_tensorrt = tensorrt_config.enabled;
        let num_cpus = num_cpus::get();
        
        let cuda_optimizer = if use_tensorrt {
            Some(CudaOptimizer::for_throughput())
        } else {
            Some(CudaOptimizer::for_latency())
        };
        
        Self {
            use_tensorrt,
            device_id,
            tensor_pool,
            tensorrt_config,
            use_fp16: true,
            intra_threads: num_cpus,
            inter_threads: (num_cpus / 2).max(1),
            cuda_optimizer,
        }
    }
    
    /// Create a loader optimized for maximum throughput
    pub fn for_throughput(device_id: usize) -> Self {
        let config = TensorRTConfig {
            enabled: true,
            precision: TensorRTPrecision::FP16,
            workspace_size_mb: 8192, // 8GB workspace
            max_batch_size: 128,
            optimization_level: 5,
            cache_dir: Some("./tensorrt_cache".to_string()),
            use_dynamic_shapes: true,
            builder_optimization_level: 5,
            auxiliary_streams: 8, // More streams for parallel execution
            enable_layer_norm_plugin: true,
            enable_gelu_plugin: true,
            enable_sparse_weights: false,
        };
        Self::with_tensorrt_config(device_id, config, None)
    }
    
    /// Create a loader optimized for minimum latency
    pub fn for_latency(device_id: usize) -> Self {
        let config = TensorRTConfig {
            enabled: true,
            precision: TensorRTPrecision::FP16,
            workspace_size_mb: 2048, // Smaller workspace, faster optimization
            max_batch_size: 1,       // Single sample for lowest latency
            optimization_level: 5,
            cache_dir: Some("./tensorrt_cache".to_string()),
            use_dynamic_shapes: false, // Fixed shapes for best latency
            builder_optimization_level: 5,
            auxiliary_streams: 2,
            enable_layer_norm_plugin: true,
            enable_gelu_plugin: true,
            enable_sparse_weights: false,
        };
        Self::with_tensorrt_config(device_id, config, None)
    }
    
    /// Create a loader optimized for INT8 inference (requires calibration data)
    pub fn for_int8(device_id: usize) -> Self {
        let config = TensorRTConfig {
            enabled: true,
            precision: TensorRTPrecision::INT8,
            workspace_size_mb: 4096,
            max_batch_size: 64,
            optimization_level: 5,
            cache_dir: Some("./tensorrt_cache".to_string()),
            use_dynamic_shapes: true,
            builder_optimization_level: 5,
            auxiliary_streams: 4,
            enable_layer_norm_plugin: true,
            enable_gelu_plugin: true,
            enable_sparse_weights: false,
        };
        Self::with_tensorrt_config(device_id, config, None)
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
                .with_device_id(target_device as i32)
                .with_conv_max_workspace(true);  // Use max workspace for cuDNN algorithms
            
            execution_providers.push(cuda_ep.build());
            debug!("  ✓ CUDA execution provider configured with optimized settings");
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
        // Configure thread pools for optimal CPU utilization
        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(self.intra_threads)?
            .with_inter_threads(self.inter_threads)?
            .commit_from_file(path)?;

        info!("Successfully loaded ONNX model with optimized execution providers");

        Ok(LoadedOnnxModel {
            session: Mutex::new(session),
            use_tensorrt: self.use_tensorrt,
            precision: self.tensorrt_config.precision.clone(),
        })
    }
    
    /// Get the CUDA optimizer instance
    pub fn cuda_optimizer(&self) -> Option<&CudaOptimizer> {
        self.cuda_optimizer.as_ref()
    }
    
    /// Get the TensorRT configuration
    pub fn tensorrt_config(&self) -> &TensorRTConfig {
        &self.tensorrt_config
    }

    pub fn infer(
        &self,
        model: &LoadedOnnxModel,
        input: &Value,
        metadata: &ModelMetadata,
    ) -> Result<Value> {
        info!("Running ONNX inference for model: {}", metadata.name);
        
        // Convert JSON input to ONNX tensor(s)
        let input_tensors = self.json_to_tensors(input, model)?;
        
        // Run inference and extract results (handles locking)
        let outputs = model.run_and_extract(input_tensors.as_slice())
            .context("Failed to run ONNX inference")?;
        
        // Convert outputs to JSON
        let output_json = self.extracted_to_json(&outputs)?;
        
        Ok(output_json)
    }
    
    /// Convert JSON value to ONNX input tensors
    fn json_to_tensors<'a>(&self, input: &Value, model: &LoadedOnnxModel) -> Result<Vec<ort::session::SessionInputValue<'a>>> {
        let mut tensors = Vec::new();
        
        // Lock session to get input info - extract as owned strings
        let input_names: Vec<String> = {
            let session = model.session.lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock session: {}", e))?;
            session.inputs.iter().map(|i| i.name.clone()).collect()
        };
        
        match input {
            // Handle array input (single input model)
            Value::Array(arr) => {
                if input_names.is_empty() {
                    return Err(anyhow::anyhow!("Model has no inputs defined"));
                }
                
                let tensor = self.json_array_to_tensor(arr)?;
                tensors.push(tensor);
            }
            // Handle object input (named inputs for multi-input models)
            Value::Object(obj) => {
                for input_name in &input_names {
                    if let Some(value) = obj.get(input_name) {
                        if let Value::Array(arr) = value {
                            let tensor = self.json_array_to_tensor(arr)?;
                            tensors.push(tensor);
                        } else {
                            return Err(anyhow::anyhow!(
                                "Input '{}' must be an array", input_name
                            ));
                        }
                    } else {
                        return Err(anyhow::anyhow!(
                            "Missing required input: {}", input_name
                        ));
                    }
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Input must be a JSON array or object with named inputs"
                ));
            }
        }
        
        Ok(tensors)
    }
    
    /// Convert a JSON array to an ONNX tensor
    fn json_array_to_tensor<'a>(&self, arr: &[Value]) -> Result<ort::session::SessionInputValue<'a>> {
        // Infer shape and flatten the array
        let (shape, flat_data) = self.flatten_json_array(arr)?;
        
        // Convert shape to i64 for ort
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        
        // Create tensor using (shape, data) tuple - doesn't require ndarray feature
        let tensor = Tensor::from_array((shape_i64, flat_data.into_boxed_slice()))
            .context("Failed to create ONNX tensor")?;
        
        Ok(tensor.into())
    }
    
    /// Recursively flatten a JSON array and infer its shape
    fn flatten_json_array(&self, arr: &[Value]) -> Result<(Vec<usize>, Vec<f32>)> {
        let mut shape = vec![arr.len()];
        let mut flat_data = Vec::new();
        
        if arr.is_empty() {
            return Ok((shape, flat_data));
        }
        
        match &arr[0] {
            Value::Number(_) => {
                // Base case: array of numbers
                for val in arr {
                    let num = val.as_f64()
                        .ok_or_else(|| anyhow::anyhow!("Invalid number in array"))?;
                    flat_data.push(num as f32);
                }
            }
            Value::Array(inner) => {
                // Recursive case: nested array
                let (inner_shape, _) = self.flatten_json_array(inner)?;
                shape.extend(inner_shape);
                
                for val in arr {
                    if let Value::Array(inner_arr) = val {
                        let (_, inner_flat) = self.flatten_json_array(inner_arr)?;
                        flat_data.extend(inner_flat);
                    } else {
                        return Err(anyhow::anyhow!("Inconsistent array structure"));
                    }
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Array elements must be numbers or nested arrays"
                ));
            }
        }
        
        Ok((shape, flat_data))
    }
    
    /// Convert extracted tensor outputs to JSON
    fn extracted_to_json(&self, outputs: &[(String, Vec<i64>, Vec<f32>)]) -> Result<Value> {
        let mut results = serde_json::Map::new();
        
        for (name, shape, data) in outputs {
            let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            let json_array = self.tensor_data_to_json(&shape_vec, data);
            results.insert(name.clone(), json_array);
        }
        
        // If single output, return it directly; otherwise return object
        if results.len() == 1 {
            Ok(results.into_iter().next().unwrap().1)
        } else {
            Ok(Value::Object(results))
        }
    }
    
    /// Convert flat tensor data back to nested JSON array based on shape
    fn tensor_data_to_json(&self, shape: &[usize], data: &[f32]) -> Value {
        if shape.is_empty() {
            return Value::Array(vec![]);
        }
        
        if shape.len() == 1 {
            // 1D array
            return Value::Array(
                data.iter()
                    .map(|&x| Value::Number(serde_json::Number::from_f64(x as f64).unwrap_or(serde_json::Number::from(0))))
                    .collect()
            );
        }
        
        // Multi-dimensional: recursively build nested arrays
        let inner_size: usize = shape[1..].iter().product();
        let outer_size = shape[0];
        
        let mut result = Vec::with_capacity(outer_size);
        for i in 0..outer_size {
            let start = i * inner_size;
            let end = start + inner_size;
            let inner_data = &data[start..end];
            let inner_json = self.tensor_data_to_json(&shape[1..], inner_data);
            result.push(inner_json);
        }
        
        Value::Array(result)
    }
    
    /// Run inference with raw tensor input (more efficient for benchmarks)
    /// Returns (shape, data) tuple
    pub fn infer_tensor(
        &self,
        model: &LoadedOnnxModel,
        shape: Vec<i64>,
        data: Vec<f32>,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let tensor = Tensor::from_array((shape, data.into_boxed_slice()))
            .context("Failed to create ONNX tensor")?;
        
        let input_value: ort::session::SessionInputValue = tensor.into();
        let inputs = [input_value];
        
        let outputs = model.run_and_extract(&inputs[..])
            .context("Failed to run ONNX inference")?;
        
        // Return first output
        let (_, out_shape, out_data) = outputs.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("No outputs from model"))?;
        
        Ok((out_shape, out_data))
    }
}

#[allow(dead_code)]
pub struct LoadedOnnxModel {
    session: Mutex<Session>,
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
    
    /// Run inference and return extracted tensor data
    /// Returns a Vec of (output_name, shape, data) tuples
    pub fn run_and_extract<'a>(&self, inputs: &'a [ort::session::SessionInputValue<'a>]) 
        -> Result<Vec<(String, Vec<i64>, Vec<f32>)>> 
    {
        let mut session = self.session.lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock session: {}", e))?;
        
        let outputs = session.run(inputs)
            .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;
        
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
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
                results.push((
                    name.to_string(),
                    shape.iter().copied().collect(),
                    data.iter().map(|&x| x as f32).collect(),
                ));
            }
        }
        
        Ok(results)
    }
}

// SAFETY: Session wrapped in Mutex is thread-safe
unsafe impl Send for LoadedOnnxModel {}
unsafe impl Sync for LoadedOnnxModel {}
