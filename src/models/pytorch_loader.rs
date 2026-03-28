use std::path::Path;
use anyhow::{Result, Context};
use log::{info, warn};
use serde_json::Value;

#[cfg(feature = "torch")]
use tch::{nn, Tensor, Device, Kind, CModule};

use crate::models::registry::{ModelMetadata, PreprocessingConfig, PostprocessingConfig};
use crate::config::DeviceConfig;
use crate::tensor_pool::{TensorPool, TensorShape};
use std::sync::Arc;

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

/// PyTorch model loader with preprocessing/postprocessing
pub struct PyTorchModelLoader {
    device: String,
    config: Option<DeviceConfig>,
    tensor_pool: Option<Arc<TensorPool>>,
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
        
        // Apply global JIT settings if config is provided
        #[cfg(feature = "torch")]
        if let Some(cfg) = &config {
            if cfg.enable_jit {
                info!("Enabling JIT compilation optimizations");
                // Note: tch-rs might not expose all JIT flags directly in safe Rust
                // We can try to set some if available, or just rely on CModule loading
                
                if cfg.enable_jit_profiling {
                    // tch::jit::set_profiling_mode(true); // Hypothetical API
                    info!("JIT profiling enabled (if supported)");
                }
                
                if cfg.enable_jit_executor {
                    // tch::jit::set_profiling_executor(true); // Hypothetical API
                    info!("JIT profiling executor enabled (if supported)");
                }
            }
        }
        
        Self { device, config, tensor_pool }
    }

    /// Load a PyTorch model (.pt or .pth)
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
                
                // Apply JIT optimizations
                m.set_eval();
                
                // If JIT is enabled in config, we can try to optimize further
                if let Some(cfg) = &self.config {
                    if cfg.enable_jit {
                        // Trigger JIT compilation/optimization
                        // Some models might benefit from a dummy forward pass to warm up the JIT
                        info!("JIT enabled: Model set to eval mode for optimization");
                    }
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
        })
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

/// Loaded PyTorch model
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

    // ── Line 57: default device detection when device=None and torch disabled ──
    // When `device` is None the `unwrap_or_else` closure executes.  Without the
    // `torch` feature, the #[cfg(not(feature = "torch"))] block at line 55-58
    // runs and returns the hard-coded "cpu" string (line 57).

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_pytorch_loader_none_device_defaults_to_cpu() {
        // Passing None triggers the unwrap_or_else closure → executes line 57.
        let loader = PyTorchModelLoader::new(None, None, None);
        assert_eq!(loader.device, "cpu", "default device without torch feature must be 'cpu'");
    }

    // ── Lines 462-463: #[cfg(not(feature = "torch"))] to_tensor returns Err ──

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_to_tensor_returns_err_without_torch_raw() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = PreprocessedInput::Raw(serde_json::json!([1.0, 2.0, 3.0]));
        let result = loader.to_tensor(&input);
        assert!(result.is_err(), "to_tensor should error without torch feature");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("PyTorch") || msg.contains("torch") || msg.contains("enabled"),
            "{msg}"
        );
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_to_tensor_returns_err_without_torch_image() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = PreprocessedInput::Image {
            data: serde_json::json!("base64data"),
            shape: Some((224, 224)),
        };
        let result = loader.to_tensor(&input);
        assert!(result.is_err());
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_to_tensor_returns_err_without_torch_audio() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = PreprocessedInput::Audio {
            data: serde_json::json!([0.1, 0.2]),
            sample_rate: Some(16000),
        };
        let result = loader.to_tensor(&input);
        assert!(result.is_err());
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_to_tensor_returns_err_without_torch_text() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = PreprocessedInput::Text {
            data: serde_json::json!("hello"),
            max_length: Some(128),
        };
        let result = loader.to_tensor(&input);
        assert!(result.is_err());
    }

    // ── Helper ──────────────────────────────────────────────────────────────

    fn make_post_config(output_type: crate::models::registry::OutputType) -> PostprocessingConfig {
        PostprocessingConfig {
            output_type,
            threshold: None,
            top_k: None,
            custom: None,
        }
    }

    // ── preprocess tests ────────────────────────────────────────────────────

    #[test]
    fn test_preprocess_none_config_returns_raw() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = serde_json::json!({"key": "value"});
        let result = loader.preprocess(&input, &None).unwrap();
        match result {
            PreprocessedInput::Raw(v) => assert_eq!(v, input),
            other => panic!("expected Raw, got {:?}", other),
        }
    }

    #[test]
    fn test_preprocess_image_config_with_image_field_returns_image() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = serde_json::json!({"image": "base64data"});
        let cfg = Some(crate::models::registry::PreprocessingConfig {
            image: Some(crate::models::registry::ImagePreprocessing {
                resize: Some((224, 224)),
                normalize: None,
                to_tensor: false,
                color_space: None,
            }),
            audio: None,
            text: None,
            custom: None,
        });
        let result = loader.preprocess(&input, &cfg).unwrap();
        match result {
            PreprocessedInput::Image { .. } => {}
            other => panic!("expected Image, got {:?}", other),
        }
    }

    #[test]
    fn test_preprocess_image_config_without_image_field_falls_through_to_raw() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = serde_json::json!({"data": "something"});
        let cfg = Some(crate::models::registry::PreprocessingConfig {
            image: Some(crate::models::registry::ImagePreprocessing {
                resize: None,
                normalize: None,
                to_tensor: false,
                color_space: None,
            }),
            audio: None,
            text: None,
            custom: None,
        });
        let result = loader.preprocess(&input, &cfg).unwrap();
        match result {
            PreprocessedInput::Raw(_) => {}
            other => panic!("expected Raw, got {:?}", other),
        }
    }

    #[test]
    fn test_preprocess_audio_config_with_audio_field_returns_audio() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = serde_json::json!({"audio": [0.1, 0.2, 0.3]});
        let cfg = Some(crate::models::registry::PreprocessingConfig {
            image: None,
            audio: Some(crate::models::registry::AudioPreprocessing {
                sample_rate: Some(16000),
                channels: None,
                normalize: false,
                to_mel: None,
            }),
            text: None,
            custom: None,
        });
        let result = loader.preprocess(&input, &cfg).unwrap();
        match result {
            PreprocessedInput::Audio { .. } => {}
            other => panic!("expected Audio, got {:?}", other),
        }
    }

    #[test]
    fn test_preprocess_audio_config_without_audio_field_falls_through() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = serde_json::json!({"data": "something"});
        let cfg = Some(crate::models::registry::PreprocessingConfig {
            image: None,
            audio: Some(crate::models::registry::AudioPreprocessing {
                sample_rate: None,
                channels: None,
                normalize: false,
                to_mel: None,
            }),
            text: None,
            custom: None,
        });
        let result = loader.preprocess(&input, &cfg).unwrap();
        match result {
            PreprocessedInput::Raw(_) => {}
            other => panic!("expected Raw, got {:?}", other),
        }
    }

    #[test]
    fn test_preprocess_text_config_with_text_field_returns_text() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = serde_json::json!({"text": "hello world"});
        let cfg = Some(crate::models::registry::PreprocessingConfig {
            image: None,
            audio: None,
            text: Some(crate::models::registry::TextPreprocessing {
                tokenizer: None,
                max_length: Some(128),
                padding: false,
                truncation: false,
            }),
            custom: None,
        });
        let result = loader.preprocess(&input, &cfg).unwrap();
        match result {
            PreprocessedInput::Text { .. } => {}
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn test_preprocess_text_config_without_text_field_falls_through() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let input = serde_json::json!({"data": "something"});
        let cfg = Some(crate::models::registry::PreprocessingConfig {
            image: None,
            audio: None,
            text: Some(crate::models::registry::TextPreprocessing {
                tokenizer: None,
                max_length: None,
                padding: false,
                truncation: false,
            }),
            custom: None,
        });
        let result = loader.preprocess(&input, &cfg).unwrap();
        match result {
            PreprocessedInput::Raw(_) => {}
            other => panic!("expected Raw, got {:?}", other),
        }
    }

    // ── postprocess tests ───────────────────────────────────────────────────

    #[test]
    fn test_postprocess_none_config_returns_raw_json() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let output = ModelOutput::Json(serde_json::json!({"score": 0.9}));
        let result = loader.postprocess(output, &None).unwrap();
        assert_eq!(result, serde_json::json!({"score": 0.9}));
    }

    #[test]
    fn test_postprocess_classification_has_type_field() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let output = ModelOutput::Json(serde_json::json!([]));
        let cfg = Some(make_post_config(crate::models::registry::OutputType::Classification));
        let result = loader.postprocess(output, &cfg).unwrap();
        assert_eq!(result["type"], "classification");
    }

    #[test]
    fn test_postprocess_classification_top_k_override() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let output = ModelOutput::Json(serde_json::json!([]));
        let cfg = Some(PostprocessingConfig {
            output_type: crate::models::registry::OutputType::Classification,
            threshold: None,
            top_k: Some(10),
            custom: None,
        });
        let result = loader.postprocess(output, &cfg).unwrap();
        assert_eq!(result["top_k"], 10);
    }

    #[test]
    fn test_postprocess_regression_has_type_field() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let output = ModelOutput::Json(serde_json::json!(42.0));
        let cfg = Some(make_post_config(crate::models::registry::OutputType::Regression));
        let result = loader.postprocess(output, &cfg).unwrap();
        assert_eq!(result["type"], "regression");
    }

    #[test]
    fn test_postprocess_segmentation_has_type_field() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let output = ModelOutput::Json(serde_json::json!([]));
        let cfg = Some(make_post_config(crate::models::registry::OutputType::Segmentation));
        let result = loader.postprocess(output, &cfg).unwrap();
        assert_eq!(result["type"], "segmentation");
    }

    #[test]
    fn test_postprocess_detection_has_type_field() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let output = ModelOutput::Json(serde_json::json!([]));
        let cfg = Some(make_post_config(crate::models::registry::OutputType::Detection));
        let result = loader.postprocess(output, &cfg).unwrap();
        assert_eq!(result["type"], "detection");
    }

    #[test]
    fn test_postprocess_detection_threshold_override() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let output = ModelOutput::Json(serde_json::json!([]));
        let cfg = Some(PostprocessingConfig {
            output_type: crate::models::registry::OutputType::Detection,
            threshold: Some(0.7),
            top_k: None,
            custom: None,
        });
        let result = loader.postprocess(output, &cfg).unwrap();
        // threshold is stored as f32; compare as f64 with tolerance
        let threshold = result["threshold"].as_f64().expect("threshold should be a number");
        assert!((threshold - 0.7).abs() < 1e-5, "expected threshold ~0.7, got {}", threshold);
    }

    #[test]
    fn test_postprocess_generation_has_type_field() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let output = ModelOutput::Json(serde_json::json!("generated text"));
        let cfg = Some(make_post_config(crate::models::registry::OutputType::Generation));
        let result = loader.postprocess(output, &cfg).unwrap();
        assert_eq!(result["type"], "generation");
    }

    #[test]
    fn test_postprocess_raw_output_type_falls_through_to_raw_json() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let inner = serde_json::json!({"raw": true});
        let output = ModelOutput::Json(inner.clone());
        let cfg = Some(make_post_config(crate::models::registry::OutputType::Raw));
        let result = loader.postprocess(output, &cfg).unwrap();
        assert_eq!(result, inner);
    }

    // ── ModelOutput::to_json tests ───────────────────────────────────────────

    #[test]
    fn test_model_output_to_json_returns_inner_value() {
        let inner = serde_json::json!({"answer": 42});
        let output = ModelOutput::Json(inner.clone());
        assert_eq!(output.to_json(), inner);
    }

    // ── non-torch stub tests ────────────────────────────────────────────────

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_get_best_device_non_torch_returns_unit() {
        let result = get_best_device();
        assert_eq!(result, ());
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_load_model_non_torch_returns_err() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let result = loader.load_model(std::path::Path::new("/tmp/nonexistent.pt"));
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("PyTorch") || msg.contains("torch") || msg.contains("enabled"),
            "unexpected error message: {msg}"
        );
    }

    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_infer_non_torch_returns_err() {
        let loader = PyTorchModelLoader::new(Some("cpu".to_string()), None, None);
        let model = LoadedPyTorchModel {
            model: (),
            device: (),
            metadata: None,
        };
        let metadata = crate::models::registry::ModelMetadata {
            id: "test".to_string(),
            name: "test".to_string(),
            format: crate::models::registry::ModelFormat::PyTorch,
            path: std::path::PathBuf::from("/tmp/test.pt"),
            version: "1.0".to_string(),
            description: None,
            tags: vec![],
            input_schema: None,
            output_schema: None,
            preprocessing: None,
            postprocessing: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            file_size: 0,
            checksum: None,
        };
        let input = serde_json::json!({});
        let result = loader.infer(&model, &input, &metadata);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("PyTorch") || msg.contains("torch") || msg.contains("enabled"),
            "unexpected error message: {msg}"
        );
    }
}
