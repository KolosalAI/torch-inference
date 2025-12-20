/// StyleTTS2 Model Implementation in Rust
/// Native implementation of StyleTTS2 architecture using tch-rs
#[cfg(feature = "torch")]
use tch::{nn, Device, Tensor, Kind, CModule};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct StyleTTS2Config {
    pub dim_in: i64,
    pub hidden_dim: i64,
    pub max_conv_dim: i64,
    pub n_layer: i64,
    pub n_mels: i64,
    pub n_token: i64,
    pub style_dim: i64,
    pub dropout: f64,
    pub max_dur: i64,
    pub multispeaker: bool,
    pub text_encoder_kernel_size: i64,
}

impl Default for StyleTTS2Config {
    fn default() -> Self {
        Self {
            dim_in: 64,
            hidden_dim: 512,
            max_conv_dim: 512,
            n_layer: 3,
            n_mels: 80,
            n_token: 178,
            style_dim: 128,
            dropout: 0.2,
            max_dur: 50,
            multispeaker: true,
            text_encoder_kernel_size: 5,
        }
    }
}

#[cfg(feature = "torch")]
pub struct StyleTTS2Model {
    model: CModule,
    config: StyleTTS2Config,
    device: Device,
}

#[cfg(feature = "torch")]
impl StyleTTS2Model {
    /// Load StyleTTS2 model from file
    pub fn load(model_path: &std::path::Path, device: Device) -> Result<Self> {
        log::info!("Loading StyleTTS2 model from {:?}", model_path);
        
        // Load the TorchScript model
        let model = CModule::load_on_device(model_path, device)
            .context("Failed to load StyleTTS2 model")?;
        
        log::info!("StyleTTS2 model loaded successfully");
        
        // Load config (use default for now)
        let config = StyleTTS2Config::default();
        
        Ok(Self {
            model,
            config,
            device,
        })
    }
    
    /// Run inference on phoneme tokens
    pub fn synthesize(&self, phoneme_tokens: &[i64], speaker_id: Option<i64>) -> Result<Tensor> {
        log::debug!("StyleTTS2 inference: {} tokens", phoneme_tokens.len());
        
        // Convert tokens to tensor [batch=1, seq_len]
        let tokens = Tensor::from_slice(phoneme_tokens)
            .to_device(self.device)
            .unsqueeze(0); // Add batch dimension
        
        log::debug!("Token tensor shape: {:?}", tokens.size());
        
        // Prepare inputs
        let mut inputs = vec![tokens];
        
        // Add speaker embedding if multispeaker
        if self.config.multispeaker {
            let speaker = speaker_id.unwrap_or(0);
            let speaker_tensor = Tensor::from_slice(&[speaker])
                .to_device(self.device);
            inputs.push(speaker_tensor);
        }
        
        // Run model inference
        let output = self.model.forward_ts(&inputs)
            .context("Failed to run StyleTTS2 model forward pass")?;
        
        log::debug!("Model output shape: {:?}", output.size());
        
        Ok(output)
    }
    
    /// Get mel-spectrogram from model output
    pub fn get_mel_spectrogram(&self, model_output: &Tensor) -> Result<Tensor> {
        // Extract mel-spectrogram from model output
        // StyleTTS2 output format: [batch, n_mels, time]
        Ok(model_output.shallow_clone())
    }
}

/// Simplified inference wrapper
#[cfg(feature = "torch")]
pub struct StyleTTS2Inference {
    model: StyleTTS2Model,
}

#[cfg(feature = "torch")]
impl StyleTTS2Inference {
    pub fn new(model_path: &std::path::Path, device: Device) -> Result<Self> {
        let model = StyleTTS2Model::load(model_path, device)?;
        Ok(Self { model })
    }
    
    /// Full synthesis pipeline: tokens -> mel-spectrogram
    pub fn tokens_to_mel(&self, tokens: &[i64], speaker: Option<i64>) -> Result<Vec<Vec<f32>>> {
        // Run model
        let output = self.model.synthesize(tokens, speaker)?;
        
        // Get mel-spectrogram
        let mel = self.model.get_mel_spectrogram(&output)?;
        
        // Convert to Vec<Vec<f32>> [n_mels, time]
        let mel_data = self.tensor_to_mel(&mel)?;
        
        log::info!("Generated mel-spectrogram: {} mels x {} frames", 
                   mel_data.len(), 
                   mel_data.get(0).map(|v| v.len()).unwrap_or(0));
        
        Ok(mel_data)
    }
    
    /// Convert tensor to mel-spectrogram array
    fn tensor_to_mel(&self, mel_tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
        // Remove batch dimension if present
        let mel = if mel_tensor.size().len() == 3 {
            mel_tensor.squeeze_dim(0)
        } else {
            mel_tensor.shallow_clone()
        };
        
        // Shape should be [n_mels, time]
        let shape = mel.size();
        let n_mels = shape[0] as usize;
        let n_frames = shape[1] as usize;
        
        // Convert to CPU and extract data
        let mel_cpu = mel.to_device(Device::Cpu);
        let mel_vec: Vec<f32> = mel_cpu.view([-1])
            .try_into()
            .context("Failed to convert mel tensor to Vec")?;
        
        // Reshape to [n_mels, n_frames]
        let mut result = vec![vec![0.0; n_frames]; n_mels];
        for i in 0..n_mels {
            for j in 0..n_frames {
                result[i][j] = mel_vec[i * n_frames + j];
            }
        }
        
        Ok(result)
    }
}

#[cfg(not(feature = "torch"))]
pub struct StyleTTS2Inference;

#[cfg(not(feature = "torch"))]
impl StyleTTS2Inference {
    pub fn new(_model_path: &std::path::Path, _device: i32) -> Result<Self> {
        anyhow::bail!("StyleTTS2 requires torch feature to be enabled")
    }
}
