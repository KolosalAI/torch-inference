#![allow(dead_code)]
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "torch")]
use tch::{Device, Tensor};

#[cfg(not(feature = "torch"))]
type Device = ();

#[cfg(feature = "torch")]
use crate::models::pytorch_loader::get_best_device;

#[cfg(feature = "torch")]
use crate::torch_optimization::{OptimizedTorchModel, TorchOptimizationConfig,
    TorchOptimizationConfigBuilder};

use crate::core::model_cache::{ModelCache, CacheStats};
#[cfg(feature = "torch")]
use crate::core::model_cache::cache_key;

/// Generic neural network for inference
pub struct NeuralNetwork {
    #[cfg(feature = "torch")]
    optimizer: OptimizedTorchModel,

    #[cfg(not(feature = "torch"))]
    _optimizer: (),

    input_shapes: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
    metadata: NetworkMetadata,
    cache: ModelCache,
    model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetadata {
    pub name: String,
    pub task: String,
    pub framework: String,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub outputs: HashMap<String, Vec<f32>>,
    pub inference_time_ms: f64,
    pub device: String,
}

impl NeuralNetwork {
    /// Create a new neural network from a model file
    #[cfg(feature = "torch")]
    pub fn new(
        model_path: &Path,
        device: Option<Device>,
        metadata: Option<NetworkMetadata>,
        cache_capacity: Option<usize>,
        opt_config: Option<TorchOptimizationConfig>,
    ) -> Result<Self> {
        let device = device.unwrap_or_else(|| get_best_device());
        let model_id = model_path.to_string_lossy().to_string();

        let config = opt_config.unwrap_or_else(|| {
            let is_accel = device != Device::Cpu;
            TorchOptimizationConfigBuilder::new()
                .device(device)
                .cudnn_benchmark(true)
                .warmup_iterations(3)
                .autocast(is_accel)
                .fp16(is_accel)
                .build()
        });

        let metadata = metadata.unwrap_or(NetworkMetadata {
            name: "custom_model".to_string(),
            task: "unknown".to_string(),
            framework: "pytorch".to_string(),
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            description: None,
        });

        log::info!("Loading neural network model from {:?}", model_path);
        let model_path_str = model_path.to_str()
            .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
        let mut optimizer = OptimizedTorchModel::new(model_path_str, config)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        optimizer.warmup().map_err(|e| anyhow::anyhow!("{}", e))?;

        log::info!("Neural network loaded successfully: {}", metadata.name);
        Ok(Self {
            optimizer,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            metadata,
            cache: ModelCache::new(cache_capacity.unwrap_or(512)),
            model_id,
        })
    }

    #[cfg(not(feature = "torch"))]
    pub fn new(
        _model_path: &Path,
        _device: Option<()>,
        _metadata: Option<NetworkMetadata>,
        _cache_capacity: Option<usize>,
        _opt_config: Option<()>,
    ) -> Result<Self> {
        bail!("PyTorch feature not enabled. Compile with --features torch");
    }

    /// Run inference with a single tensor input
    #[cfg(feature = "torch")]
    pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
        log::debug!("Running inference with input shape: {:?}", input.size());
        let start = std::time::Instant::now();
        let input = input.to_device(self.optimizer.device());
        let output = self.optimizer.infer(&input)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        log::debug!("Inference completed in {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
        Ok(output)
    }

    #[cfg(not(feature = "torch"))]
    pub fn predict(&self, _input: &()) -> Result<()> {
        bail!("PyTorch feature not enabled");
    }

    /// Run inference with multiple tensor inputs
    #[cfg(feature = "torch")]
    pub fn predict_multi(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        log::debug!("Running inference with {} inputs", inputs.len());
        if inputs.is_empty() {
            bail!("predict_multi requires at least one input tensor");
        }
        let start = std::time::Instant::now();
        let inputs: Vec<Tensor> = inputs.iter()
            .map(|t| t.to_device(self.optimizer.device()))
            .collect();
        let outputs = self.optimizer.infer(&inputs[0])
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        log::debug!("Multi-input inference completed in {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0);
        Ok(vec![outputs])
    }

    #[cfg(not(feature = "torch"))]
    pub fn predict_multi(&self, _inputs: &[()]) -> Result<Vec<()>> {
        bail!("PyTorch feature not enabled");
    }

    /// Run inference from raw float data
    #[cfg(feature = "torch")]
    pub fn predict_from_slice(&self, data: &[f32], shape: &[i64]) -> Result<InferenceResult> {
        // Build params bytes from shape.
        let mut params = Vec::with_capacity(shape.len() * 8);
        for &dim in shape {
            params.extend_from_slice(&dim.to_le_bytes());
        }
        // Cast data to byte slice for hashing only (f32 is plain-old-data).
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        let key = cache_key(&self.model_id, data_bytes, &params);

        self.cache.get_or_run(key, || {
            let start = std::time::Instant::now();
            let input = Tensor::from_slice(data)
                .reshape(shape)
                .to_device(self.optimizer.device());
            let output = self.optimizer.infer(&input)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let output_vec: Vec<f32> = output.flatten(0, -1).try_into()?;
            let mut outputs = HashMap::new();
            outputs.insert(
                self.metadata.output_names.first()
                    .cloned()
                    .unwrap_or_else(|| "output".to_string()),
                output_vec,
            );
            Ok(InferenceResult {
                outputs,
                inference_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                device: format!("{:?}", self.optimizer.device()),
            })
        })
    }

    #[cfg(not(feature = "torch"))]
    pub fn predict_from_slice(&self, _data: &[f32], _shape: &[i64]) -> Result<InferenceResult> {
        bail!("PyTorch feature not enabled");
    }

    /// Get model metadata
    pub fn metadata(&self) -> &NetworkMetadata {
        &self.metadata
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    #[cfg(feature = "torch")]
    pub fn device(&self) -> Device {
        self.optimizer.device()
    }

    #[cfg(not(feature = "torch"))]
    pub fn device(&self) -> () {
        ()
    }

    /// Construct a stub network for tests (no model loaded)
    // Stub only available without torch feature: OptimizedTorchModel requires a real .pt file.
    #[cfg(all(test, not(feature = "torch")))]
    pub fn new_stub(metadata: Option<NetworkMetadata>) -> Self {
        Self {
            _optimizer: (),
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            metadata: metadata.unwrap_or(NetworkMetadata {
                name: "stub".to_string(),
                task: "test".to_string(),
                framework: "none".to_string(),
                input_names: vec!["input".to_string()],
                output_names: vec!["output".to_string()],
                description: None,
            }),
            cache: ModelCache::new(512),
            model_id: "stub".to_string(),
        }
    }

    /// Set input shapes (for validation)
    pub fn set_input_shapes(&mut self, shapes: Vec<Vec<i64>>) {
        self.input_shapes = shapes;
    }

    /// Set output shapes (for validation)
    pub fn set_output_shapes(&mut self, shapes: Vec<Vec<i64>>) {
        self.output_shapes = shapes;
    }
}

/// Helper to create common neural network architectures
pub mod architectures {

    /// MLP (Multi-Layer Perceptron) configuration
    #[derive(Debug, Clone)]
    pub struct MLPConfig {
        pub input_size: i64,
        pub hidden_sizes: Vec<i64>,
        pub output_size: i64,
        pub activation: Activation,
        pub dropout: Option<f64>,
    }

    #[derive(Debug, Clone)]
    pub enum Activation {
        ReLU,
        Sigmoid,
        Tanh,
        LeakyReLU(f64),
    }

    /// CNN (Convolutional Neural Network) configuration
    #[derive(Debug, Clone)]
    pub struct CNNConfig {
        pub input_channels: i64,
        pub conv_layers: Vec<ConvLayerConfig>,
        pub fc_layers: Vec<i64>,
        pub output_size: i64,
    }

    #[derive(Debug, Clone)]
    pub struct ConvLayerConfig {
        pub out_channels: i64,
        pub kernel_size: i64,
        pub stride: i64,
        pub padding: i64,
    }

    /// RNN/LSTM configuration
    #[derive(Debug, Clone)]
    pub struct RNNConfig {
        pub input_size: i64,
        pub hidden_size: i64,
        pub num_layers: i64,
        pub output_size: i64,
        pub bidirectional: bool,
        pub rnn_type: RNNType,
    }

    #[derive(Debug, Clone)]
    pub enum RNNType {
        LSTM,
        GRU,
        RNN,
    }
}

/// Batch inference for multiple inputs
pub struct BatchInference {
    network: NeuralNetwork,
    batch_size: usize,
}

impl BatchInference {
    pub fn new(network: NeuralNetwork, batch_size: usize) -> Self {
        Self { network, batch_size }
    }

    /// Process multiple inputs in batches
    #[cfg(feature = "torch")]
    pub fn predict_batch(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let mut results = Vec::new();

        for chunk in inputs.chunks(self.batch_size) {
            // Stack inputs into a batch
            let batch = Tensor::stack(chunk, 0);

            // Run inference
            let output = self.network.predict(&batch)?;

            // Split outputs
            for i in 0..chunk.len() {
                results.push(output.get(i as i64));
            }
        }

        Ok(results)
    }

    #[cfg(not(feature = "torch"))]
    pub fn predict_batch(&self, _inputs: Vec<()>) -> Result<Vec<()>> {
        bail!("PyTorch feature not enabled");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_neural_network_cache_stats_initially_zero() {
        let nn = NeuralNetwork::new_stub(None);
        let stats = nn.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_metadata_creation() {
        let metadata = NetworkMetadata {
            name: "test_model".to_string(),
            task: "classification".to_string(),
            framework: "pytorch".to_string(),
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            description: Some("Test model".to_string()),
        };

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.task, "classification");
    }

    // ── NetworkMetadata extended ──────────────────────────────────────────────

    #[test]
    fn test_network_metadata_serialize_deserialize() {
        let meta = NetworkMetadata {
            name: "my_net".to_string(),
            task: "regression".to_string(),
            framework: "pytorch".to_string(),
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            description: None,
        };
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("my_net"));
        let back: NetworkMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "my_net");
        assert_eq!(back.task, "regression");
        assert!(back.description.is_none());
    }

    #[test]
    fn test_network_metadata_clone_debug() {
        let meta = NetworkMetadata {
            name: "net".to_string(),
            task: "detection".to_string(),
            framework: "pytorch".to_string(),
            input_names: vec!["in1".to_string(), "in2".to_string()],
            output_names: vec!["out1".to_string()],
            description: Some("desc".to_string()),
        };
        let cloned = meta.clone();
        assert_eq!(cloned.input_names.len(), 2);
        let dbg = format!("{:?}", cloned);
        assert!(dbg.contains("NetworkMetadata"));
    }

    #[test]
    fn test_network_metadata_description_some() {
        let meta = NetworkMetadata {
            name: "n".to_string(),
            task: "t".to_string(),
            framework: "f".to_string(),
            input_names: vec![],
            output_names: vec![],
            description: Some("hello".to_string()),
        };
        assert_eq!(meta.description.as_deref(), Some("hello"));
    }

    // ── InferenceResult ───────────────────────────────────────────────────────

    #[test]
    fn test_inference_result_construction() {
        let mut outputs = HashMap::new();
        outputs.insert("logits".to_string(), vec![0.1_f32, 0.9]);
        let result = InferenceResult {
            outputs,
            inference_time_ms: 12.5,
            device: "cpu".to_string(),
        };
        assert_eq!(result.device, "cpu");
        assert!((result.inference_time_ms - 12.5).abs() < 1e-6);
        let logits = result.outputs.get("logits").unwrap();
        assert_eq!(logits.len(), 2);
    }

    #[test]
    fn test_inference_result_serialize_deserialize() {
        let mut outputs = HashMap::new();
        outputs.insert("out".to_string(), vec![1.0_f32]);
        let ir = InferenceResult {
            outputs,
            inference_time_ms: 5.0,
            device: "cuda:0".to_string(),
        };
        let json = serde_json::to_string(&ir).unwrap();
        assert!(json.contains("cuda:0"));
        let back: InferenceResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.device, "cuda:0");
        assert!((back.inference_time_ms - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_inference_result_clone_debug() {
        let mut outputs = HashMap::new();
        outputs.insert("k".to_string(), vec![0.5_f32]);
        let ir = InferenceResult {
            outputs,
            inference_time_ms: 1.0,
            device: "cpu".to_string(),
        };
        let cloned = ir.clone();
        assert_eq!(cloned.device, "cpu");
        let dbg = format!("{:?}", cloned);
        assert!(dbg.contains("InferenceResult"));
    }

    // ── NeuralNetwork without torch feature ──────────────────────────────────

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_neural_network_new_without_torch_returns_error() {
        let path = std::path::Path::new("/nonexistent/model.pt");
        let result = NeuralNetwork::new(path, None, None, None, None);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("PyTorch") || msg.contains("torch"), "unexpected: {}", msg);
    }

    // Helper: constructs NeuralNetwork directly (only valid without `torch` feature).
    #[cfg(not(feature = "torch"))]
    fn make_test_network() -> NeuralNetwork {
        NeuralNetwork {
            _optimizer: (),
            input_shapes: vec![],
            output_shapes: vec![],
            metadata: NetworkMetadata {
                name: "test".to_string(),
                task: "test".to_string(),
                framework: "none".to_string(),
                input_names: vec!["in".to_string()],
                output_names: vec!["out".to_string()],
                description: None,
            },
            cache: ModelCache::new(512),
            model_id: "test".to_string(),
        }
    }

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_neural_network_predict_without_torch_returns_error() {
        let nn = make_test_network();
        let result = nn.predict(&());
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("PyTorch") || msg.contains("torch"), "unexpected: {}", msg);
    }

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_neural_network_predict_multi_without_torch_returns_error() {
        let nn = make_test_network();
        let result = nn.predict_multi(&[]);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_neural_network_predict_from_slice_without_torch_returns_error() {
        let nn = make_test_network();
        let result = nn.predict_from_slice(&[1.0_f32], &[1]);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_neural_network_metadata_accessor() {
        let nn = make_test_network();
        let meta = nn.metadata();
        assert_eq!(meta.name, "test");
        assert_eq!(meta.framework, "none");
    }

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_neural_network_device_accessor() {
        let nn = make_test_network();
        let _d: () = nn.device();
    }

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_neural_network_set_input_shapes() {
        let mut nn = make_test_network();
        nn.set_input_shapes(vec![vec![1, 128], vec![1, 64]]);
        assert_eq!(nn.input_shapes.len(), 2);
    }

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_neural_network_set_output_shapes() {
        let mut nn = make_test_network();
        nn.set_output_shapes(vec![vec![1, 10]]);
        assert_eq!(nn.output_shapes.len(), 1);
    }

    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_batch_inference_new_and_predict_batch_error() {
        let nn = make_test_network();
        let bi = BatchInference::new(nn, 4);
        assert_eq!(bi.batch_size, 4);
        let result = bi.predict_batch(vec![]);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("PyTorch") || msg.contains("torch"), "unexpected: {}", msg);
    }

    // ── architectures module ──────────────────────────────────────────────────

    #[test]
    fn test_mlp_config_construction() {
        use super::architectures::{MLPConfig, Activation};
        let cfg = MLPConfig {
            input_size: 128,
            hidden_sizes: vec![256, 128],
            output_size: 10,
            activation: Activation::ReLU,
            dropout: Some(0.5),
        };
        assert_eq!(cfg.input_size, 128);
        assert_eq!(cfg.hidden_sizes.len(), 2);
        assert_eq!(cfg.output_size, 10);
        assert!(cfg.dropout.is_some());
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("MLPConfig"));
    }

    #[test]
    fn test_activation_variants_debug() {
        use super::architectures::Activation;
        assert_eq!(format!("{:?}", Activation::ReLU), "ReLU");
        assert_eq!(format!("{:?}", Activation::Sigmoid), "Sigmoid");
        assert_eq!(format!("{:?}", Activation::Tanh), "Tanh");
        assert!(format!("{:?}", Activation::LeakyReLU(0.01)).contains("LeakyReLU"));
    }

    #[test]
    fn test_activation_clone() {
        use super::architectures::Activation;
        let a = Activation::LeakyReLU(0.2);
        let b = a.clone();
        assert!(format!("{:?}", b).contains("0.2"));
    }

    #[test]
    fn test_cnn_config_construction() {
        use super::architectures::{CNNConfig, ConvLayerConfig};
        let cfg = CNNConfig {
            input_channels: 3,
            conv_layers: vec![
                ConvLayerConfig { out_channels: 32, kernel_size: 3, stride: 1, padding: 1 },
            ],
            fc_layers: vec![512, 256],
            output_size: 10,
        };
        assert_eq!(cfg.input_channels, 3);
        assert_eq!(cfg.conv_layers.len(), 1);
        assert_eq!(cfg.fc_layers.len(), 2);
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("CNNConfig"));
    }

    #[test]
    fn test_conv_layer_config_clone() {
        use super::architectures::ConvLayerConfig;
        let cl = ConvLayerConfig { out_channels: 64, kernel_size: 5, stride: 2, padding: 2 };
        let cloned = cl.clone();
        assert_eq!(cloned.out_channels, 64);
        assert_eq!(cloned.kernel_size, 5);
    }

    #[test]
    fn test_rnn_config_construction() {
        use super::architectures::{RNNConfig, RNNType};
        let cfg = RNNConfig {
            input_size: 256,
            hidden_size: 512,
            num_layers: 2,
            output_size: 10,
            bidirectional: true,
            rnn_type: RNNType::LSTM,
        };
        assert_eq!(cfg.input_size, 256);
        assert_eq!(cfg.hidden_size, 512);
        assert!(cfg.bidirectional);
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("RNNConfig"));
    }

    #[test]
    fn test_rnn_type_variants_debug() {
        use super::architectures::RNNType;
        assert_eq!(format!("{:?}", RNNType::LSTM), "LSTM");
        assert_eq!(format!("{:?}", RNNType::GRU), "GRU");
        assert_eq!(format!("{:?}", RNNType::RNN), "RNN");
    }

    #[test]
    fn test_rnn_type_clone() {
        use super::architectures::RNNType;
        let t = RNNType::GRU;
        let u = t.clone();
        assert_eq!(format!("{:?}", u), "GRU");
    }
}
