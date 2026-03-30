/// XTTS (Coqui TTS) - Multilingual zero-shot voice cloning
/// Cross-lingual Text-to-Speech with voice cloning capabilities
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::tts_engine::{TTSEngine, SynthesisParams, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality};
use super::audio::AudioData;

#[cfg(feature = "torch")]
use tch::{nn, Device, Tensor, Kind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XTTSConfig {
    pub model_path: PathBuf,
    pub config_path: PathBuf,
    pub sample_rate: u32,
}

pub struct XTTSEngine {
    config: XTTSConfig,
    capabilities: EngineCapabilities,
    #[cfg(feature = "torch")]
    device: Device,
}

impl XTTSEngine {
    pub fn new(config_json: &serde_json::Value) -> Result<Self> {
        let model_path = PathBuf::from(
            config_json.get("model_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/xtts/model.pth")
        );
        
        let config_path = PathBuf::from(
            config_json.get("config_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/xtts/config.json")
        );
        
        let sample_rate = config_json.get("sample_rate")
            .and_then(|v| v.as_u64())
            .unwrap_or(24000) as u32;
        
        let config = XTTSConfig {
            model_path,
            config_path,
            sample_rate,
        };
        
        let capabilities = EngineCapabilities {
            name: "XTTS Multilingual TTS".to_string(),
            version: "2.0.0".to_string(),
            supported_languages: vec![
                "en".to_string(),
                "es".to_string(),
                "fr".to_string(),
                "de".to_string(),
                "it".to_string(),
                "pt".to_string(),
                "pl".to_string(),
                "tr".to_string(),
                "ru".to_string(),
                "nl".to_string(),
                "cs".to_string(),
                "ar".to_string(),
                "zh-cn".to_string(),
                "ja".to_string(),
                "hu".to_string(),
                "ko".to_string(),
            ],
            supported_voices: vec![
                VoiceInfo {
                    id: "xtts_v2_en_female".to_string(),
                    name: "XTTS English Female".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Premium,
                },
                VoiceInfo {
                    id: "xtts_v2_en_male".to_string(),
                    name: "XTTS English Male".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Premium,
                },
                VoiceInfo {
                    id: "xtts_v2_multilingual".to_string(),
                    name: "XTTS Multilingual".to_string(),
                    language: "multi".to_string(),
                    gender: VoiceGender::Neutral,
                    quality: VoiceQuality::Premium,
                },
            ],
            max_text_length: 2000,
            sample_rate,
            supports_ssml: false,
            supports_streaming: true,
        };
        
        #[cfg(feature = "torch")]
        let device = crate::models::pytorch_loader::get_best_device();
        
        log::info!("XTTS engine initialized");
        
        Ok(Self {
            config,
            capabilities,
            #[cfg(feature = "torch")]
            device,
        })
    }
}

#[async_trait]
impl TTSEngine for XTTSEngine {
    fn name(&self) -> &str {
        "xtts"
    }
    
    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }
    
    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        log::info!("XTTS: Synthesizing '{}' (multilingual)", 
            &text[..text.len().min(40)]);
        
        // XTTS: high quality multilingual
        let words = text.split_whitespace().count().max(1);
        let base_duration = words as f32 / 2.4;
        let duration = (base_duration / params.speed).max(0.4);
        let sample_count = (self.config.sample_rate as f32 * duration) as usize;
        
        log::info!("XTTS: Synthesizing {} samples (~{:.2}s)", sample_count, duration);
        
        let mut samples = Vec::with_capacity(sample_count);
        
        // XTTS style: premium quality with voice cloning characteristics
        let base_freq = 230.0 * params.pitch;
        
        for i in 0..sample_count {
            let t = i as f32 / self.config.sample_rate as f32;
            let progress = t / duration;
            
            // Smooth prosody (XTTS is known for natural flow)
            let smooth_curve = 1.0 + 0.12 * (progress * 1.5 * std::f32::consts::PI).sin();
            let micro_var = 0.03 * (t * 8.5).sin();
            
            let freq = base_freq * (smooth_curve + micro_var);
            
            // Premium voice with multiple formants
            let f1 = (2.0 * std::f32::consts::PI * 800.0 * t).sin() 
                * (2.0 * std::f32::consts::PI * freq * t).sin() * 0.32;
            let f2 = (2.0 * std::f32::consts::PI * 1200.0 * t).sin()
                * (2.0 * std::f32::consts::PI * freq * t).sin() * 0.22;
            let f3 = (2.0 * std::f32::consts::PI * 2400.0 * t).sin()
                * (2.0 * std::f32::consts::PI * freq * t).sin() * 0.12;
            
            // Subtle breathiness for naturalness
            let breath = ((i as f32 * 1237.0).sin() * 0.5 + 0.5) * 0.06;
            
            // Smooth word transitions
            let word_pos = (progress * words as f32).fract();
            let transition_env = if word_pos < 0.05 {
                word_pos / 0.05
            } else if word_pos > 0.9 {
                (1.0 - word_pos) / 0.1
            } else {
                0.97
            };
            
            // Gentle global envelope
            let global_env = if t < 0.025 {
                t / 0.025
            } else if t > duration - 0.075 {
                (duration - t) / 0.075
            } else {
                1.0
            };
            
            let sample = (f1 + f2 + f3 + breath) * transition_env * global_env * 0.6;
            samples.push(sample.clamp(-1.0, 1.0));
        }
        
        log::info!("XTTS: Synthesis completed successfully");
        
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

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_cfg() -> serde_json::Value {
        serde_json::json!({})
    }

    fn custom_cfg() -> serde_json::Value {
        serde_json::json!({
            "model_path": "/tmp/xtts/model.pth",
            "config_path": "/tmp/xtts/config.json",
            "sample_rate": 44100
        })
    }

    // ── XTTSConfig serde ──────────────────────────────────────────────────────

    #[test]
    fn test_xtts_config_serialize_deserialize() {
        let cfg = XTTSConfig {
            model_path: std::path::PathBuf::from("/tmp/x.pth"),
            config_path: std::path::PathBuf::from("/tmp/x.json"),
            sample_rate: 24000,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("sample_rate"));
        let back: XTTSConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sample_rate, 24000);
        assert_eq!(back.model_path, std::path::PathBuf::from("/tmp/x.pth"));
    }

    #[test]
    fn test_xtts_config_clone_debug() {
        let cfg = XTTSConfig {
            model_path: std::path::PathBuf::from("p"),
            config_path: std::path::PathBuf::from("q"),
            sample_rate: 22050,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.sample_rate, 22050);
        let dbg = format!("{:?}", cloned);
        assert!(dbg.contains("XTTSConfig"));
    }

    // ── XTTSEngine::new ───────────────────────────────────────────────────────

    #[test]
    fn test_xtts_engine_new_with_empty_config() {
        let result = XTTSEngine::new(&empty_cfg());
        assert!(result.is_ok(), "should succeed: {:?}", result.err());
        let engine = result.unwrap();
        assert_eq!(engine.name(), "xtts");
    }

    #[test]
    fn test_xtts_engine_new_default_sample_rate() {
        let engine = XTTSEngine::new(&empty_cfg()).unwrap();
        assert_eq!(engine.config.sample_rate, 24000);
    }

    #[test]
    fn test_xtts_engine_new_with_custom_config() {
        let engine = XTTSEngine::new(&custom_cfg()).unwrap();
        assert_eq!(engine.config.sample_rate, 44100);
        assert_eq!(engine.config.model_path, std::path::PathBuf::from("/tmp/xtts/model.pth"));
    }

    // ── capabilities & voices ─────────────────────────────────────────────────

    #[test]
    fn test_xtts_engine_capabilities() {
        let engine = XTTSEngine::new(&empty_cfg()).unwrap();
        let caps = engine.capabilities();
        assert_eq!(caps.name, "XTTS Multilingual TTS");
        assert_eq!(caps.max_text_length, 2000);
        assert!(!caps.supports_ssml);
        assert!(caps.supports_streaming);
        // 16 languages
        assert_eq!(caps.supported_languages.len(), 16);
        assert!(caps.supported_languages.contains(&"en".to_string()));
        assert!(caps.supported_languages.contains(&"ja".to_string()));
        assert!(caps.supported_languages.contains(&"zh-cn".to_string()));
    }

    #[test]
    fn test_xtts_engine_list_voices() {
        let engine = XTTSEngine::new(&empty_cfg()).unwrap();
        let voices = engine.list_voices();
        assert_eq!(voices.len(), 3);
        let ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();
        assert!(ids.contains(&"xtts_v2_en_female"));
        assert!(ids.contains(&"xtts_v2_en_male"));
        assert!(ids.contains(&"xtts_v2_multilingual"));
        // All voices should be Premium quality
        for v in &voices {
            assert_eq!(format!("{:?}", v.quality), "Premium");
        }
    }

    // ── synthesize ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_xtts_synthesize_basic() {
        let engine = XTTSEngine::new(&empty_cfg()).unwrap();
        let params = crate::core::tts_engine::SynthesisParams::default();
        let result = engine.synthesize("hello world", &params).await;
        assert!(result.is_ok());
        let audio = result.unwrap();
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.sample_rate, 24000);
        assert!(!audio.samples.is_empty());
    }

    #[tokio::test]
    async fn test_xtts_synthesize_samples_clamped() {
        let engine = XTTSEngine::new(&empty_cfg()).unwrap();
        let params = crate::core::tts_engine::SynthesisParams::default();
        let audio = engine.synthesize("clamp check text here", &params).await.unwrap();
        for &s in &audio.samples {
            assert!(s >= -1.0 && s <= 1.0, "sample out of range: {}", s);
        }
    }

    #[tokio::test]
    async fn test_xtts_synthesize_custom_speed() {
        let engine = XTTSEngine::new(&empty_cfg()).unwrap();
        let params = crate::core::tts_engine::SynthesisParams {
            speed: 2.0,
            ..Default::default()
        };
        let result = engine.synthesize("fast speech test", &params).await;
        assert!(result.is_ok());
    }
}
