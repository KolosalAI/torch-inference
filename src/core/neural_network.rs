use anyhow::{Result, Context, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashMap;

#[cfg(feature = "torch")]
use tch::{nn, Tensor, Device, Kind, CModule};

#[cfg(not(feature = "torch"))]
type Device = ();

/// Generic neural network for inference
pub struct NeuralNetwork {
    #[cfg(feature = "torch")]
    model: CModule,
    #[cfg(not(feature = "torch"))]
    model: (),
    
    #[cfg(feature = "torch")]
    device: Device,
    #[cfg(not(feature = "torch"))]
    device: (),
    
    input_shapes: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
    metadata: NetworkMetadata,
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
    ) -> Result<Self> {
        let device = device.unwrap_or(Device::Cpu);
        
        log::info!("Loading neural network model from {:?}", model_path);
        
        let model = CModule::load_on_device(model_path, device)
            .context("Failed to load neural network model")?;
        
        let metadata = metadata.unwrap_or(NetworkMetadata {
            name: "custom_model".to_string(),
            task: "unknown".to_string(),
            framework: "pytorch".to_string(),
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            description: None,
        });
        
        log::info!("Neural network loaded successfully: {}", metadata.name);
        
        Ok(Self {
            model,
            device,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            metadata,
        })
    }
    
    #[cfg(not(feature = "torch"))]
    pub fn new(
        _model_path: &Path,
        _device: Option<()>,
        metadata: Option<NetworkMetadata>,
    ) -> Result<Self> {
        bail!("PyTorch feature not enabled. Compile with --features torch");
    }
    
    /// Run inference with a single tensor input
    #[cfg(feature = "torch")]
    pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
        log::debug!("Running inference with input shape: {:?}", input.size());
        
        let start = std::time::Instant::now();
        
        // Move input to device
        let input = input.to_device(self.device);
        
        // Run forward pass
        let output = self.model.forward_ts(&[input])
            .context("Forward pass failed")?;
        
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        log::debug!("Inference completed in {:.2}ms", elapsed);
        
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
        
        let start = std::time::Instant::now();
        
        // Move inputs to device
        let inputs: Vec<Tensor> = inputs.iter()
            .map(|t| t.to_device(self.device))
            .collect();
        
        // Run forward pass
        let outputs = self.model.forward_ts(&inputs)
            .context("Forward pass failed")?;
        
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        log::debug!("Multi-input inference completed in {:.2}ms", elapsed);
        
        // Handle both single and multiple outputs
        Ok(vec![outputs])
    }
    
    #[cfg(not(feature = "torch"))]
    pub fn predict_multi(&self, _inputs: &[()]) -> Result<Vec<()>> {
        bail!("PyTorch feature not enabled");
    }
    
    /// Run inference from raw float data
    #[cfg(feature = "torch")]
    pub fn predict_from_slice(&self, data: &[f32], shape: &[i64]) -> Result<InferenceResult> {
        let start = std::time::Instant::now();
        
        // Create tensor from slice
        let input = Tensor::of_slice(data)
            .reshape(shape)
            .to_device(self.device);
        
        // Run inference
        let output = self.predict(&input)?;
        
        // Convert output to Vec<f32>
        let output_vec: Vec<f32> = output.flatten(0, -1).try_into()?;
        
        let mut outputs = HashMap::new();
        outputs.insert(
            self.metadata.output_names.get(0)
                .unwrap_or(&"output".to_string())
                .clone(),
            output_vec,
        );
        
        let inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(InferenceResult {
            outputs,
            inference_time_ms,
            device: format!("{:?}", self.device),
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
    
    /// Get device
    #[cfg(feature = "torch")]
    pub fn device(&self) -> Device {
        self.device
    }
    
    #[cfg(not(feature = "torch"))]
    pub fn device(&self) -> () {
        ()
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
    use super::*;
    
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

/// Model quantization utilities
pub mod quantization {
    use super::*;
    
    #[derive(Debug, Clone)]
    pub enum QuantizationType {
        Dynamic,
        Static,
        QAT, // Quantization Aware Training
    }
    
    #[derive(Debug, Clone)]
    pub struct QuantizationConfig {
        pub qtype: QuantizationType,
        pub dtype: String, // "qint8", "quint8", etc.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
}
