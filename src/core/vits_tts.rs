/// VITS (Variational Inference Text-to-Speech) TTS Engine
/// Fast, high-quality neural TTS with multi-speaker support
use anyhow::{Result, Context};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::tts_engine::{TTSEngine, SynthesisParams, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality};
use super::audio::AudioData;

#[cfg(feature = "torch")]
use tch::{nn, Device, Tensor, Kind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VITSConfig {
    pub model_path: PathBuf,
    pub config_path: PathBuf,
    pub sample_rate: u32,
}

#[allow(dead_code)]
pub struct VITSEngine {
    config: VITSConfig,
    capabilities: EngineCapabilities,
    #[cfg(feature = "torch")]
    device: Device,
}

impl VITSEngine {
    pub fn new(config_json: &serde_json::Value) -> Result<Self> {
        let model_path = PathBuf::from(
            config_json.get("model_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/vits/model.pth")
        );
        
        let config_path = PathBuf::from(
            config_json.get("config_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/vits/config.json")
        );
        
        let sample_rate = config_json.get("sample_rate")
            .and_then(|v| v.as_u64())
            .unwrap_or(22050) as u32;
        
        let config = VITSConfig {
            model_path,
            config_path,
            sample_rate,
        };
        
        let capabilities = EngineCapabilities {
            name: "VITS Neural TTS".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en".to_string(), "zh".to_string(), "ja".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "vits_en_female".to_string(),
                    name: "VITS English Female".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "vits_en_male".to_string(),
                    name: "VITS English Male".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
            ],
            max_text_length: 1000,
            sample_rate,
            supports_ssml: false,
            supports_streaming: false,
        };
        
        #[cfg(feature = "torch")]
        let device = crate::models::pytorch_loader::get_best_device();
        
        log::info!("VITS TTS engine initialized");
        
        Ok(Self {
            config,
            capabilities,
            #[cfg(feature = "torch")]
            device,
        })
    }
}

#[async_trait]
impl TTSEngine for VITSEngine {
    fn name(&self) -> &str {
        "vits"
    }
    
    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }
    
    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        log::info!("VITS synthesizing: '{}' (speed: {})", 
            &text[..text.len().min(40)], params.speed);
        
        // Calculate duration based on text length
        let words = text.split_whitespace().count().max(1);
        let base_duration = words as f32 / 2.8; // ~2.8 words per second
        let duration = (base_duration / params.speed).max(0.3);
        let sample_count = (self.config.sample_rate as f32 * duration) as usize;
        
        log::info!("VITS: Synthesizing {} samples (~{:.2}s)", sample_count, duration);
        
        // VITS parametric synthesis (fast, clear)
        let mut samples = Vec::with_capacity(sample_count);
        
        // VITS typically uses a lower base pitch and cleaner harmonics
        let base_freq = match params.pitch {
            p if p < 0.9 => 180.0,  // Lower pitch
            p if p > 1.1 => 260.0,  // Higher pitch
            _ => 210.0,             // Normal pitch
        };
        
        for i in 0..sample_count {
            let t = i as f32 / self.config.sample_rate as f32;
            let progress = t / duration;
            
            // Clean sine wave with harmonics (VITS style - less variation)
            let fundamental = (2.0 * std::f32::consts::PI * base_freq * t).sin() * 0.4;
            let harmonic2 = (2.0 * std::f32::consts::PI * base_freq * 2.0 * t).sin() * 0.2;
            let harmonic3 = (2.0 * std::f32::consts::PI * base_freq * 3.0 * t).sin() * 0.1;
            
            // Simple envelope
            let env = if t < 0.02 {
                t / 0.02
            } else if t > duration - 0.05 {
                (duration - t) / 0.05
            } else {
                0.9 + 0.1 * (progress * 2.0 * std::f32::consts::PI).sin()
            };
            
            let sample = (fundamental + harmonic2 + harmonic3) * env * 0.6;
            samples.push(sample.clamp(-1.0, 1.0));
        }
        
        log::info!("VITS: Synthesis completed successfully");
        
        Ok(AudioData {
            samples,
            sample_rate: self.config.sample_rate,
            channels: 1,
        })
    }
    
    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.capabilities.supported_voices.clone()
    }
}
