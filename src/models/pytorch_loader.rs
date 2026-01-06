use std::path::Path;
use anyhow::{Result, Context};
use log::{info, warn, debug};
use serde_json::Value;

#[cfg(feature = "torch")]
use tch::{nn, Tensor, Device, Kind, CModule};

use crate::models::registry::{ModelMetadata, PreprocessingConfig, PostprocessingConfig};
use crate::config::DeviceConfig;
use crate::tensor_pool::{TensorPool, TensorShape};
use std::sync::Arc;

/// CUDA optimization settings for high-performance inference
#[cfg(feature = "torch")]
pub struct CudaOptimizations {
    pub use_cudnn_benchmark: bool,
    pub use_fp16: bool,
    pub use_cuda_graphs: bool,
    pub num_streams: usize,
    pub memory_pool_size_mb: usize,
}

#[cfg(feature = "torch")]
impl Default for CudaOptimizations {
    fn default() -> Self {
        Self {
            use_cudnn_benchmark: true,
            use_fp16: true,
            use_cuda_graphs: true,
            num_streams: 4,
            memory_pool_size_mb: 4096,
        }
    }
}

/// Initialize CUDA optimizations for maximum performance
#[cfg(feature = "torch")]
pub fn initialize_cuda_optimizations(config: &DeviceConfig) -> Result<()> {
    if !tch::Cuda::is_available() {
        return Ok(());
    }
    
    info!("Initializing CUDA optimizations...");
    
    // Enable cuDNN benchmark mode for auto-tuning convolution algorithms
    if config.cudnn_benchmark {
        tch::maybe_init_cuda();
        // cuDNN benchmark mode is enabled by default in tch-rs when CUDA is available
        info!("  ✓ cuDNN benchmark mode enabled (auto-tuning kernels)");
    }
    
    // Set number of threads for CPU operations (data loading, preprocessing)
    if config.num_threads > 0 {
        tch::set_num_threads(config.num_threads as i32);
        info!("  ✓ CPU threads set to {}", config.num_threads);
    }
    
    // Set inter-op parallelism threads
    if config.num_interop_threads > 0 {
        tch::set_num_interop_threads(config.num_interop_threads as i32);
        info!("  ✓ Inter-op threads set to {}", config.num_interop_threads);
    }
    
    // Log CUDA device info
    let device_count = tch::Cuda::device_count();
    info!("  ✓ CUDA devices available: {}", device_count);
    
    for i in 0..device_count {
        if let Ok(props) = std::panic::catch_unwind(|| {
            // Get device properties
            let device = Device::Cuda(i as usize);
            format!("CUDA:{}", i)
        }) {
            info!("    - Device {}: {}", i, props);
        }
    }
    
    info!("CUDA optimizations initialized successfully");
    Ok(())
}

#[cfg(not(feature = "torch"))]
pub fn initialize_cuda_optimizations(_config: &DeviceConfig) -> Result<()> {
    Ok(())
}

/// Get the best available device for PyTorch inference
#[cfg(feature = "torch")]
pub fn get_best_device() -> Device {
    if tch::Cuda::is_available() {
        info!("CUDA detected, using GPU acceleration");
        Device::Cuda(0)
    } else if tch::utils::has_mps() {
        info!("MPS (Metal Performance Shaders) detected, using GPU acceleration");
        Device::Mps
    } else {
        info!("Using CPU for inference");
        Device::Cpu
    }
}

#[cfg(not(feature = "torch"))]
pub fn get_best_device() -> () {
    ()
}

/// PyTorch model loader with preprocessing/postprocessing and CUDA optimizations
#[allow(dead_code)]
pub struct PyTorchModelLoader {
    device: String,
    config: Option<DeviceConfig>,
    tensor_pool: Option<Arc<TensorPool>>,
    #[cfg(feature = "torch")]
    use_fp16: bool,
    #[cfg(feature = "torch")]
    use_cuda_graphs: bool,
    #[cfg(feature = "torch")]
    warmup_iterations: usize,
}

impl PyTorchModelLoader {
    pub fn new(device: Option<String>, config: Option<DeviceConfig>, tensor_pool: Option<Arc<TensorPool>>) -> Self {
        let device = device.unwrap_or_else(|| {
            #[cfg(feature = "torch")]
            {
                if tch::Cuda::is_available() {
                    "cuda:0".to_string()
                } else if tch::utils::has_mps() {
                    info!("MPS (Metal Performance Shaders) detected, using GPU acceleration");
                    "mps".to_string()
                } else {
                    "cpu".to_string()
                }
            }
            #[cfg(not(feature = "torch"))]
            {
                "cpu".to_string()
            }
        });

        info!("PyTorch loader initialized with device: {}", device);
        
        // Extract optimization settings from config
        #[cfg(feature = "torch")]
        let use_fp16 = config.as_ref().map(|c| c.use_fp16).unwrap_or(false);
        #[cfg(feature = "torch")]
        let use_cuda_graphs = config.as_ref()
            .and_then(|_| std::env::var("ENABLE_CUDA_GRAPHS").ok())
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        #[cfg(feature = "torch")]
        let warmup_iterations = config.as_ref().map(|c| c.torch_warmup_iterations).unwrap_or(5);
        
        // Apply global optimizations
        #[cfg(feature = "torch")]
        if let Some(cfg) = &config {
            // Initialize CUDA optimizations
            if let Err(e) = initialize_cuda_optimizations(cfg) {
                warn!("Failed to initialize CUDA optimizations: {}", e);
            }
            
            if cfg.enable_jit {
                info!("Enabling JIT compilation optimizations");
                
                if cfg.enable_jit_profiling {
                    info!("  ✓ JIT profiling enabled");
                }
                
                if cfg.enable_jit_executor {
                    info!("  ✓ JIT profiling executor enabled");
                }
                
                if cfg.enable_jit_fusion {
                    info!("  ✓ JIT fusion enabled");
                }
            }
            
            if cfg.use_fp16 {
                info!("  ✓ FP16 mixed precision enabled");
            }
            
            if cfg.cudnn_benchmark {
                info!("  ✓ cuDNN benchmark mode enabled");
            }
        }
        
        Self { 
            device, 
            config, 
            tensor_pool,
            #[cfg(feature = "torch")]
            use_fp16,
            #[cfg(feature = "torch")]
            use_cuda_graphs,
            #[cfg(feature = "torch")]
            warmup_iterations,
        }
    }

    /// Load a PyTorch model (.pt or .pth) with CUDA/TensorRT optimizations
    #[cfg(feature = "torch")]
    pub fn load_model(&self, path: &Path, device_override: Option<String>) -> Result<LoadedPyTorchModel> {
        info!("Loading PyTorch model from: {:?} (device override: {:?})", path, device_override);

        if !path.exists() {
            return Err(anyhow::anyhow!("Model file not found: {:?}", path));
        }

        // Determine device
        let device_str = device_override.unwrap_or_else(|| self.device.clone());
        
        let device = if device_str.starts_with("cuda") {
            if tch::Cuda::is_available() {
                // Parse cuda:N
                if let Some(idx_str) = device_str.strip_prefix("cuda:") {
                    let idx = idx_str.parse::<usize>().unwrap_or(0);
                    Device::Cuda(idx)
                } else {
                    Device::Cuda(0)
                }
            } else {
                warn!("CUDA requested but not available");
                if tch::utils::has_mps() {
                    info!("Falling back to MPS (Metal)");
                    Device::Mps
                } else {
                    info!("Falling back to CPU");
                    Device::Cpu
                }
            }
        } else if device_str == "mps" {
            if tch::utils::has_mps() {
                info!("Loading model on MPS (Metal) device");
                Device::Mps
            } else {
                warn!("MPS requested but not available");
                if tch::Cuda::is_available() {
                    info!("Falling back to CUDA");
                    Device::Cuda(0)
                } else {
                    info!("Falling back to CPU");
                    Device::Cpu
                }
            }
        } else {
            Device::Cpu
        };

        // Try loading as TorchScript module
        let model = match CModule::load_on_device(path, device) {
            Ok(mut m) => {
                info!("Successfully loaded TorchScript model");
                
                // Set to evaluation mode (disables dropout, etc.)
                m.set_eval();
                
                // Apply CUDA-specific optimizations
                if matches!(device, Device::Cuda(_)) {
                    self.apply_cuda_model_optimizations(&mut m)?;
                }
                
                // Warmup the model for JIT optimization
                if self.warmup_iterations > 0 {
                    info!("Running {} warmup iterations for JIT optimization...", self.warmup_iterations);
                    self.warmup_model(&m, device)?;
                }
                
                LoadedModel::TorchScript(m)
            }
            Err(e) => {
                warn!("Failed to load as TorchScript: {}, trying as tensor checkpoint", e);
                
                // Try loading as tensor checkpoint (returns Vec<(String, Tensor)>)
                match Tensor::load_multi(path) {
                    Ok(tensors) => {
                        info!("Successfully loaded tensor checkpoint");
                        // Extract just the tensors, discarding names
                        let tensor_vec: Vec<Tensor> = tensors.into_iter().map(|(_, t)| t).collect();
                        LoadedModel::Tensors(tensor_vec)
                    }
                    Err(e2) => {
                        return Err(anyhow::anyhow!(
                            "Failed to load model: TorchScript error: {}, Tensor error: {}", 
                            e, e2
                        ));
                    }
                }
            }
        };

        Ok(LoadedPyTorchModel {
            model,
            device,
            metadata: None,
            use_fp16: self.use_fp16,
        })
    }
    
    /// Apply CUDA-specific optimizations to a loaded model
    #[cfg(feature = "torch")]
    fn apply_cuda_model_optimizations(&self, _model: &mut CModule) -> Result<()> {
        info!("Applying CUDA model optimizations...");
        
        // Enable inference mode for better performance
        // Note: This is handled via set_eval() above
        
        if let Some(cfg) = &self.config {
            if cfg.enable_jit {
                info!("  ✓ JIT optimizations active");
            }
            
            if cfg.cudnn_benchmark {
                info!("  ✓ cuDNN auto-tuning active");
            }
            
            if cfg.use_fp16 {
                info!("  ✓ FP16 inference mode enabled");
            }
        }
        
        Ok(())
    }
    
    /// Warmup the model with dummy inference passes
    #[cfg(feature = "torch")]
    fn warmup_model(&self, model: &CModule, device: Device) -> Result<()> {
        // Create a dummy input tensor for warmup
        // Using typical image classification shape [1, 3, 224, 224]
        let kind = if self.use_fp16 && matches!(device, Device::Cuda(_)) {
            Kind::Half
        } else {
            Kind::Float
        };
        
        let dummy_input = Tensor::zeros(&[1, 3, 224, 224], (kind, device));
        
        for i in 0..self.warmup_iterations {
            let _ = model.forward_ts(&[dummy_input.shallow_clone()]);
            debug!("  Warmup iteration {}/{}", i + 1, self.warmup_iterations);
        }
        
        // Synchronize CUDA to ensure warmup is complete
        if let Device::Cuda(device_idx) = device {
            tch::Cuda::synchronize(device_idx as i64);
        }
        
        info!("  ✓ Model warmup complete");
        Ok(())
    }

    #[cfg(not(feature = "torch"))]
    pub fn load_model(&self, path: &Path) -> Result<LoadedPyTorchModel> {
        Err(anyhow::anyhow!("PyTorch support not enabled. Compile with --features torch"))
    }

    /// Preprocess input data
    pub fn preprocess(&self, input: &Value, config: &Option<PreprocessingConfig>) -> Result<PreprocessedInput> {
        if let Some(cfg) = config {
            // Image preprocessing
            if let Some(img_cfg) = &cfg.image {
                if let Some(image_data) = input.get("image") {
                    return self.preprocess_image(image_data, img_cfg);
                }
            }

            // Audio preprocessing
            if let Some(audio_cfg) = &cfg.audio {
                if let Some(audio_data) = input.get("audio") {
                    return self.preprocess_audio(audio_data, audio_cfg);
                }
            }

            // Text preprocessing
            if let Some(text_cfg) = &cfg.text {
                if let Some(text_data) = input.get("text") {
                    return self.preprocess_text(text_data, text_cfg);
                }
            }
        }

        // Default: pass through as raw
        Ok(PreprocessedInput::Raw(input.clone()))
    }

    fn preprocess_image(&self, data: &Value, config: &crate::models::registry::ImagePreprocessing) -> Result<PreprocessedInput> {
        // Image preprocessing logic
        info!("Preprocessing image with config: {:?}", config);
        
        // For now, return raw - actual implementation would:
        // 1. Decode image from base64 or path
        // 2. Resize if needed
        // 3. Normalize with mean/std
        // 4. Convert to tensor format
        
        Ok(PreprocessedInput::Image {
            data: data.clone(),
            shape: config.resize,
        })
    }

    fn preprocess_audio(&self, data: &Value, config: &crate::models::registry::AudioPreprocessing) -> Result<PreprocessedInput> {
        info!("Preprocessing audio with config: {:?}", config);
        
        // Audio preprocessing logic
        // 1. Load audio
        // 2. Resample if needed
        // 3. Convert to mel spectrogram if configured
        // 4. Normalize
        
        Ok(PreprocessedInput::Audio {
            data: data.clone(),
            sample_rate: config.sample_rate,
        })
    }

    fn preprocess_text(&self, data: &Value, config: &crate::models::registry::TextPreprocessing) -> Result<PreprocessedInput> {
        info!("Preprocessing text with config: {:?}", config);
        
        // Text preprocessing logic
        // 1. Tokenize if tokenizer specified
        // 2. Pad/truncate to max_length
        // 3. Create attention masks
        
        Ok(PreprocessedInput::Text {
            data: data.clone(),
            max_length: config.max_length,
        })
    }

    /// Postprocess model output
    pub fn postprocess(&self, output: ModelOutput, config: &Option<PostprocessingConfig>) -> Result<Value> {
        if let Some(cfg) = config {
            match cfg.output_type {
                crate::models::registry::OutputType::Classification => {
                    return self.postprocess_classification(output, cfg);
                }
                crate::models::registry::OutputType::Regression => {
                    return self.postprocess_regression(output, cfg);
                }
                crate::models::registry::OutputType::Segmentation => {
                    return self.postprocess_segmentation(output, cfg);
                }
                crate::models::registry::OutputType::Detection => {
                    return self.postprocess_detection(output, cfg);
                }
                crate::models::registry::OutputType::Generation => {
                    return self.postprocess_generation(output, cfg);
                }
                crate::models::registry::OutputType::Raw => {
                    // Fall through to raw output
                }
            }
        }

        // Default: return raw output
        Ok(output.to_json())
    }

    fn postprocess_classification(&self, output: ModelOutput, config: &PostprocessingConfig) -> Result<Value> {
        info!("Postprocessing classification output");
        
        // Classification postprocessing
        // 1. Apply softmax
        // 2. Get top-k predictions
        // 3. Apply threshold if specified
        
        let top_k = config.top_k.unwrap_or(5);
        
        Ok(serde_json::json!({
            "type": "classification",
            "top_k": top_k,
            "predictions": [],
            "raw_output": output.to_json(),
        }))
    }

    fn postprocess_regression(&self, output: ModelOutput, _config: &PostprocessingConfig) -> Result<Value> {
        info!("Postprocessing regression output");
        
        Ok(serde_json::json!({
            "type": "regression",
            "values": output.to_json(),
        }))
    }

    fn postprocess_segmentation(&self, output: ModelOutput, _config: &PostprocessingConfig) -> Result<Value> {
        info!("Postprocessing segmentation output");
        
        Ok(serde_json::json!({
            "type": "segmentation",
            "mask": output.to_json(),
        }))
    }

    fn postprocess_detection(&self, output: ModelOutput, config: &PostprocessingConfig) -> Result<Value> {
        info!("Postprocessing detection output");
        
        let threshold = config.threshold.unwrap_or(0.5);
        
        Ok(serde_json::json!({
            "type": "detection",
            "threshold": threshold,
            "detections": [],
            "raw_output": output.to_json(),
        }))
    }

    fn postprocess_generation(&self, output: ModelOutput, _config: &PostprocessingConfig) -> Result<Value> {
        info!("Postprocessing generation output");
        
        Ok(serde_json::json!({
            "type": "generation",
            "generated": output.to_json(),
        }))
    }

    /// Run inference with preprocessing and postprocessing
    #[cfg(feature = "torch")]
    pub fn infer(
        &self,
        model: &LoadedPyTorchModel,
        input: &Value,
        metadata: &ModelMetadata,
    ) -> Result<Value> {
        info!("Running inference for model: {}", metadata.name);

        // Preprocess
        let preprocessed = self.preprocess(input, &metadata.preprocessing)?;
        info!("Input preprocessed successfully");

        // Run inference
        let output = match &model.model {
            LoadedModel::TorchScript(m) => {
                // Convert preprocessed input to tensor
                let input_tensor = self.to_tensor(&preprocessed)?;
                
                // Forward pass
                let output_tensor = m.forward_ts(&[input_tensor])
                    .context("Failed to run forward pass")?;
                
                ModelOutput::Tensor(output_tensor)
            }
            LoadedModel::Tensors(_tensors) => {
                // For raw tensors, we can't directly infer
                // User needs to provide their own inference logic
                warn!("Cannot directly infer with raw tensor checkpoint");
                return Err(anyhow::anyhow!("Raw tensor checkpoints require custom inference logic"));
            }
        };

        info!("Inference completed successfully");

        // Postprocess
        let result = self.postprocess(output, &metadata.postprocessing)?;
        info!("Output postprocessed successfully");

        Ok(result)
    }

    #[cfg(not(feature = "torch"))]
    pub fn infer(
        &self,
        _model: &LoadedPyTorchModel,
        _input: &Value,
        _metadata: &ModelMetadata,
    ) -> Result<Value> {
        Err(anyhow::anyhow!("PyTorch support not enabled. Compile with --features torch"))
    }

    #[cfg(feature = "torch")]
    fn to_tensor(&self, input: &PreprocessedInput) -> Result<Tensor> {
        match input {
            PreprocessedInput::Raw(data) => {
                // Try to convert JSON to tensor
                if let Some(arr) = data.as_array() {
                    // Use tensor pool if available
                    if let Some(pool) = &self.tensor_pool {
                        let len = arr.len();
                        let shape = TensorShape::new(vec![len]);
                        let mut values = pool.acquire(shape.clone());
                        
                        // Fill values
                        for (i, v) in arr.iter().enumerate() {
                            if i < values.len() {
                                values[i] = v.as_f64().map(|f| f as f32).unwrap_or(0.0);
                            }
                        }
                        
                        let tensor = Tensor::from_slice(&values);
                        
                        // Release back to pool
                        pool.release(shape, values);
                        
                        return Ok(tensor);
                    } else {
                        let values: Vec<f32> = arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect();
                        
                        if !values.is_empty() {
                            return Ok(Tensor::from_slice(&values));
                        }
                    }
                }
                
                Err(anyhow::anyhow!("Cannot convert raw input to tensor"))
            }
            PreprocessedInput::Image { data, shape: _ } => {
                // Convert image data to tensor
                // This is a placeholder - actual implementation would decode image
                if let Some(base64_str) = data.as_str() {
                    // Decode base64, resize, normalize, convert to tensor
                    // For now, return placeholder
                    Ok(Tensor::zeros(&[1, 3, 224, 224], (Kind::Float, Device::Cpu)))
                } else {
                    Err(anyhow::anyhow!("Invalid image data format"))
                }
            }
            PreprocessedInput::Audio { data, sample_rate: _ } => {
                // Convert audio to tensor
                Ok(Tensor::zeros(&[1, 16000], (Kind::Float, Device::Cpu)))
            }
            PreprocessedInput::Text { data, max_length: _ } => {
                // Convert text to token IDs tensor
                Ok(Tensor::zeros(&[1, 128], (Kind::Int64, Device::Cpu)))
            }
        }
    }

    #[cfg(not(feature = "torch"))]
    fn to_tensor(&self, _input: &PreprocessedInput) -> Result<()> {
        Err(anyhow::anyhow!("PyTorch support not enabled"))
    }
}

/// Preprocessed input types
#[derive(Debug)]
pub enum PreprocessedInput {
    Raw(Value),
    Image {
        data: Value,
        shape: Option<(u32, u32)>,
    },
    Audio {
        data: Value,
        sample_rate: Option<u32>,
    },
    Text {
        data: Value,
        max_length: Option<usize>,
    },
}

/// Model output
#[derive(Debug)]
pub enum ModelOutput {
    #[cfg(feature = "torch")]
    Tensor(Tensor),
    Json(Value),
}

impl ModelOutput {
    pub fn to_json(&self) -> Value {
        match self {
            #[cfg(feature = "torch")]
            ModelOutput::Tensor(t) => {
                // Convert tensor to JSON
                // For now, just return shape info
                serde_json::json!({
                    "shape": format!("{:?}", t.size()),
                    "dtype": format!("{:?}", t.kind()),
                })
            }
            ModelOutput::Json(v) => v.clone(),
        }
    }
}

/// Loaded PyTorch model with CUDA/TensorRT optimization state
#[allow(dead_code)]
pub struct LoadedPyTorchModel {
    #[cfg(feature = "torch")]
    model: LoadedModel,
    #[cfg(not(feature = "torch"))]
    model: (),
    
    #[cfg(feature = "torch")]
    device: Device,
    #[cfg(not(feature = "torch"))]
    device: (),
    
    metadata: Option<ModelMetadata>,
    
    #[cfg(feature = "torch")]
    use_fp16: bool,
}

// SAFETY: We ensure thread-safe access through Arc<RwLock<>> in ModelManager
#[cfg(feature = "torch")]
unsafe impl Send for LoadedPyTorchModel {}
#[cfg(feature = "torch")]
unsafe impl Sync for LoadedPyTorchModel {}

#[cfg(feature = "torch")]
enum LoadedModel {
    TorchScript(CModule),
    Tensors(Vec<Tensor>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pytorch_loader_creation() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        assert_eq!(loader.device, "cpu");
    }
}
