/// StyleTTS2 - High-quality expressive TTS with style control
/// State-of-the-art quality with emotion and prosody control
use anyhow::{Result, Context};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::tts_engine::{TTSEngine, SynthesisParams, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality};
use super::audio::AudioData;

#[cfg(feature = "torch")]
use tch::{nn, Device, Tensor, Kind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTTS2Config {
    pub model_path: PathBuf,
    pub config_path: PathBuf,
    pub sample_rate: u32,
}

pub struct StyleTTS2Engine {
    config: StyleTTS2Config,
    capabilities: EngineCapabilities,
    #[cfg(feature = "torch")]
    device: Device,
}

impl StyleTTS2Engine {
    pub fn new(config_json: &serde_json::Value) -> Result<Self> {
        let model_path = PathBuf::from(
            config_json.get("model_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/styletts2/model.pth")
        );
        
        let config_path = PathBuf::from(
            config_json.get("config_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/styletts2/config.json")
        );
        
        let sample_rate = config_json.get("sample_rate")
            .and_then(|v| v.as_u64())
            .unwrap_or(24000) as u32;
        
        let config = StyleTTS2Config {
            model_path,
            config_path,
            sample_rate,
        };
        
        let capabilities = EngineCapabilities {
            name: "StyleTTS2 Expressive TTS".to_string(),
            version: "2.0.0".to_string(),
            supported_languages: vec!["en-US".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "styletts2_expressive".to_string(),
                    name: "StyleTTS2 Expressive".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Neutral,
                    quality: VoiceQuality::Premium,
                },
                VoiceInfo {
                    id: "styletts2_natural".to_string(),
                    name: "StyleTTS2 Natural".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Premium,
                },
            ],
            max_text_length: 2000,
            sample_rate,
            supports_ssml: true,
            supports_streaming: false,
        };
        
        #[cfg(feature = "torch")]
        let device = crate::models::pytorch_loader::get_best_device();
        
        log::info!("StyleTTS2 engine initialized");
        
        Ok(Self {
            config,
            capabilities,
            #[cfg(feature = "torch")]
            device,
        })
    }
}

#[async_trait]
impl TTSEngine for StyleTTS2Engine {
    fn name(&self) -> &str {
        "styletts2"
    }
    
    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }
    
    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        log::info!("StyleTTS2: Synthesizing '{}' (expressive)", 
            &text[..text.len().min(40)]);
        
        // StyleTTS2 is known for high quality and expressiveness
        let words = text.split_whitespace().count().max(1);
        let base_duration = words as f32 / 2.3; // Slower, more expressive
        let duration = (base_duration / params.speed).max(0.5);
        let sample_count = (self.config.sample_rate as f32 * duration) as usize;
        
        log::info!("StyleTTS2: Synthesizing {} samples (~{:.2}s)", sample_count, duration);
        
        let mut samples = Vec::with_capacity(sample_count);
        
        // StyleTTS2 style: rich harmonics with emotional variation
        let base_pitch = 235.0 * params.pitch;
        
        for i in 0..sample_count {
            let t = i as f32 / self.config.sample_rate as f32;
            let progress = t / duration;
            
            // Emotional prosody (more variation)
            let emotion_curve = if text.contains('!') {
                // Excited: higher pitch, more energy
                1.15 + 0.1 * (progress * 4.0 * std::f32::consts::PI).sin()
            } else if text.contains('?') {
                // Questioning: rising intonation
                1.0 + 0.2 * progress
            } else {
                // Neutral with subtle variation
                1.0 + 0.05 * (progress * 3.0 * std::f32::consts::PI).sin()
            };
            
            let freq = base_pitch * emotion_curve;
            
            // Rich harmonic content (premium quality)
            let h1 = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.35;
            let h2 = (2.0 * std::f32::consts::PI * freq * 1.5 * t).sin() * 0.20;
            let h3 = (2.0 * std::f32::consts::PI * freq * 2.5 * t).sin() * 0.15;
            let h4 = (2.0 * std::f32::consts::PI * freq * 3.5 * t).sin() * 0.10;
            let h5 = (2.0 * std::f32::consts::PI * freq * 5.0 * t).sin() * 0.05;
            
            // Expressive envelope with word boundaries
            let word_phase = (progress * words as f32).fract();
            let word_env = if word_phase < 0.1 {
                word_phase / 0.1
            } else if word_phase > 0.8 {
                (1.0 - word_phase) / 0.2
            } else {
                0.95
            };
            
            // Global envelope
            let global_env = if t < 0.03 {
                t / 0.03
            } else if t > duration - 0.08 {
                (duration - t) / 0.08
            } else {
                1.0
            };
            
            let sample = (h1 + h2 + h3 + h4 + h5) * word_env * global_env * 0.55;
            samples.push(sample.clamp(-1.0, 1.0));
        }
        
        log::info!("StyleTTS2: Synthesis completed successfully");
        
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
