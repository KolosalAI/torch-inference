/// Piper TTS Engine - Neural TTS using ONNX Runtime
use anyhow::{Result, Context, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use std::collections::HashMap;

use super::tts_engine::{TTSEngine, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality, SynthesisParams};
use super::audio::AudioData;

/// Piper TTS Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiperConfig {
    pub model_path: PathBuf,
    pub config_path: PathBuf,
    pub sample_rate: u32,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

/// Piper TTS Engine - High-quality neural TTS
#[allow(dead_code)]
pub struct PiperTTSEngine {
    config: PiperConfig,
    capabilities: EngineCapabilities,
    has_onnx: bool,
}

impl PiperTTSEngine {
    pub fn new(cfg: &serde_json::Value) -> Result<Self> {
        let model_path = cfg.get("model_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing model_path"))?;
        
        let config_path = cfg.get("config_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing config_path"))?;
        
        let model_path = PathBuf::from(model_path);
        let config_path = PathBuf::from(config_path);
        
        // Verify files exist
        if !model_path.exists() {
            bail!("Model file not found: {:?}", model_path);
        }
        
        // Load config if it exists
        let (sample_rate, noise_scale, length_scale, noise_w) = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
            
            let sample_rate = config_json.get("audio")
                .and_then(|a| a.get("sample_rate"))
                .and_then(|s| s.as_u64())
                .unwrap_or(22050) as u32;
            
            let noise_scale = config_json.get("inference")
                .and_then(|i| i.get("noise_scale"))
                .and_then(|n| n.as_f64())
                .unwrap_or(0.667) as f32;
            
            let length_scale = config_json.get("inference")
                .and_then(|i| i.get("length_scale"))
                .and_then(|l| l.as_f64())
                .unwrap_or(1.0) as f32;
            
            let noise_w = config_json.get("inference")
                .and_then(|i| i.get("noise_w"))
                .and_then(|n| n.as_f64())
                .unwrap_or(0.8) as f32;
            
            (sample_rate, noise_scale, length_scale, noise_w)
        } else {
            (22050, 0.667, 1.0, 0.8)
        };
        
        let config = PiperConfig {
            model_path: model_path.clone(),
            config_path,
            sample_rate,
            noise_scale,
            length_scale,
            noise_w,
        };
        
        let capabilities = EngineCapabilities {
            name: "Piper Neural TTS".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "en_US_lessac".to_string(),
                    name: "English US (Lessac)".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::High,
                },
            ],
            max_text_length: 10000,
            sample_rate,
            supports_ssml: false,
            supports_streaming: false,
        };
        
        let capabilities = EngineCapabilities {
            name: "Piper Neural TTS".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "lessac".to_string(),
                    name: "Lessac (High Quality)".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
            ],
            max_text_length: 5000,
            sample_rate,
            supports_ssml: false,
            supports_streaming: false,
        };
        
        // Check if ONNX model exists
        let has_onnx = model_path.exists();
        if has_onnx {
            log::info!("[OK] Piper ONNX model found at: {:?}", model_path);
            #[cfg(not(feature = "onnx"))]
            {
                bail!("ONNX feature not enabled. Compile with --features onnx to use Piper TTS");
            }
        } else {
            bail!("Piper model not found at {:?}", model_path);
        }
        
        Ok(Self {
            config,
            capabilities,
            has_onnx,
        })
    }
    

    fn synthesize_fallback(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        log::warn!("ORT not available, using fallback synthesis");
        
        // Simple fallback - generate tone based on text length
        let duration = (text.len() as f32 * 0.05 / params.speed).max(0.5);
        let sample_count = (self.config.sample_rate as f32 * duration) as usize;
        
        let mut samples = Vec::with_capacity(sample_count);
        let frequency = 440.0 * params.pitch;
        
        for i in 0..sample_count {
            let t = i as f32 / self.config.sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3;
            samples.push(sample);
        }
        
        Ok(AudioData {
            samples,
            sample_rate: self.config.sample_rate,
            channels: 1,
        })
    }
}

#[async_trait]
impl TTSEngine for PiperTTSEngine {
    fn name(&self) -> &str {
        "piper"
    }
    
    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }
    
    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        self.validate_text(text)?;
        bail!("Piper TTS requires ONNX Runtime. Please use Windows SAPI engine or compile with --features onnx")
    }
    
    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.capabilities.supported_voices.clone()
    }
}
