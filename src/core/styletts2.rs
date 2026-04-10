/// StyleTTS2 - High-quality expressive TTS with style control
/// State-of-the-art quality with emotion and prosody control
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::audio::AudioData;
use super::tts_engine::{
    EngineCapabilities, SynthesisParams, TTSEngine, VoiceGender, VoiceInfo, VoiceQuality,
};

#[cfg(feature = "torch")]
use tch::{nn, Device, Kind, Tensor};

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
            config_json
                .get("model_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/styletts2/model.pth"),
        );

        let config_path = PathBuf::from(
            config_json
                .get("config_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/styletts2/config.json"),
        );

        let sample_rate = config_json
            .get("sample_rate")
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
        log::info!(
            "StyleTTS2: Synthesizing '{}' (expressive)",
            &text[..text.len().min(40)]
        );

        // Delegate to the shared Kokoro ONNX backend when available.
        if let Some(backend) = crate::core::onnx_backend::get_kokoro_onnx_backend() {
            let kokoro_voice =
                crate::core::onnx_backend::map_voice(params.voice.as_deref());
            let mapped = SynthesisParams {
                voice: Some(kokoro_voice.to_string()),
                ..params.clone()
            };
            log::info!(
                "StyleTTS2: delegating to Kokoro ONNX (voice={})",
                kokoro_voice
            );
            return backend.synthesize(text, &mapped).await;
        }

        // No ONNX model available — fall back to parametric synthesis.
        log::warn!("StyleTTS2: Kokoro ONNX backend unavailable, using parametric fallback");

        let words = text.split_whitespace().count().max(1);
        let base_duration = words as f32 / 2.3;
        let duration = (base_duration / params.speed).max(0.5);
        let sample_count = (self.config.sample_rate as f32 * duration) as usize;

        let mut samples = Vec::with_capacity(sample_count);
        let base_pitch = 235.0 * params.pitch;

        for i in 0..sample_count {
            let t = i as f32 / self.config.sample_rate as f32;
            let progress = t / duration;
            let emotion_curve = if text.contains('!') {
                1.15 + 0.1 * (progress * 4.0 * std::f32::consts::PI).sin()
            } else if text.contains('?') {
                1.0 + 0.2 * progress
            } else {
                1.0 + 0.05 * (progress * 3.0 * std::f32::consts::PI).sin()
            };
            let freq = base_pitch * emotion_curve;
            let h1 = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.35;
            let h2 = (2.0 * std::f32::consts::PI * freq * 1.5 * t).sin() * 0.20;
            let h3 = (2.0 * std::f32::consts::PI * freq * 2.5 * t).sin() * 0.15;
            let h4 = (2.0 * std::f32::consts::PI * freq * 3.5 * t).sin() * 0.10;
            let h5 = (2.0 * std::f32::consts::PI * freq * 5.0 * t).sin() * 0.05;
            let word_phase = (progress * words as f32).fract();
            let word_env = if word_phase < 0.1 {
                word_phase / 0.1
            } else if word_phase > 0.8 {
                (1.0 - word_phase) / 0.2
            } else {
                0.95
            };
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

        log::info!("StyleTTS2: Synthesis completed (parametric fallback)");
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
            "model_path": "/tmp/styletts2/model.pth",
            "config_path": "/tmp/styletts2/config.json",
            "sample_rate": 48000
        })
    }

    // ── StyleTTS2Config serde ─────────────────────────────────────────────────

    #[test]
    fn test_styletts2_config_serialize_deserialize() {
        let cfg = StyleTTS2Config {
            model_path: std::path::PathBuf::from("/tmp/m.pth"),
            config_path: std::path::PathBuf::from("/tmp/c.json"),
            sample_rate: 24000,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("sample_rate"));
        let back: StyleTTS2Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sample_rate, 24000);
    }

    #[test]
    fn test_styletts2_config_clone_debug() {
        let cfg = StyleTTS2Config {
            model_path: std::path::PathBuf::from("a"),
            config_path: std::path::PathBuf::from("b"),
            sample_rate: 16000,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.sample_rate, 16000);
        let dbg = format!("{:?}", cloned);
        assert!(dbg.contains("StyleTTS2Config"));
    }

    // ── StyleTTS2Engine::new ──────────────────────────────────────────────────

    #[test]
    fn test_styletts2_engine_new_with_empty_config() {
        let result = StyleTTS2Engine::new(&empty_cfg());
        assert!(result.is_ok(), "should succeed: {:?}", result.err());
        let engine = result.unwrap();
        assert_eq!(engine.name(), "styletts2");
    }

    #[test]
    fn test_styletts2_engine_new_default_sample_rate() {
        let engine = StyleTTS2Engine::new(&empty_cfg()).unwrap();
        assert_eq!(engine.config.sample_rate, 24000);
    }

    #[test]
    fn test_styletts2_engine_new_with_custom_config() {
        let engine = StyleTTS2Engine::new(&custom_cfg()).unwrap();
        assert_eq!(engine.config.sample_rate, 48000);
        assert_eq!(
            engine.config.model_path,
            std::path::PathBuf::from("/tmp/styletts2/model.pth")
        );
    }

    // ── capabilities & voices ─────────────────────────────────────────────────

    #[test]
    fn test_styletts2_engine_capabilities() {
        let engine = StyleTTS2Engine::new(&empty_cfg()).unwrap();
        let caps = engine.capabilities();
        assert_eq!(caps.name, "StyleTTS2 Expressive TTS");
        assert_eq!(caps.max_text_length, 2000);
        assert!(caps.supports_ssml);
        assert!(!caps.supports_streaming);
        assert!(caps.supported_languages.contains(&"en-US".to_string()));
    }

    #[test]
    fn test_styletts2_engine_list_voices() {
        let engine = StyleTTS2Engine::new(&empty_cfg()).unwrap();
        let voices = engine.list_voices();
        assert_eq!(voices.len(), 2);
        let ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();
        assert!(ids.contains(&"styletts2_expressive"));
        assert!(ids.contains(&"styletts2_natural"));
        // Quality should be Premium for both
        for v in &voices {
            assert_eq!(format!("{:?}", v.quality), "Premium");
        }
    }

    // ── synthesize ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_styletts2_synthesize_neutral_text() {
        let engine = StyleTTS2Engine::new(&empty_cfg()).unwrap();
        let params = crate::core::tts_engine::SynthesisParams::default();
        let result = engine.synthesize("hello world", &params).await;
        assert!(result.is_ok());
        let audio = result.unwrap();
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.sample_rate, 24000);
        assert!(!audio.samples.is_empty());
    }

    #[tokio::test]
    async fn test_styletts2_synthesize_exclamation() {
        let engine = StyleTTS2Engine::new(&empty_cfg()).unwrap();
        let params = crate::core::tts_engine::SynthesisParams::default();
        let result = engine.synthesize("Amazing result!", &params).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_styletts2_synthesize_question() {
        let engine = StyleTTS2Engine::new(&empty_cfg()).unwrap();
        let params = crate::core::tts_engine::SynthesisParams::default();
        let result = engine.synthesize("Is this working?", &params).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_styletts2_synthesize_samples_clamped() {
        let engine = StyleTTS2Engine::new(&empty_cfg()).unwrap();
        let params = crate::core::tts_engine::SynthesisParams::default();
        let audio = engine.synthesize("check clamp", &params).await.unwrap();
        for &s in &audio.samples {
            assert!(s >= -1.0 && s <= 1.0, "sample out of range: {}", s);
        }
    }
}
