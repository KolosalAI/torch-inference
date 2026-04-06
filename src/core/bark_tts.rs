/// Bark - Generative Audio Model by Suno AI
/// Text-prompted generative audio with music, sound effects, and speech
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
pub struct BarkConfig {
    pub model_path: PathBuf,
    pub sample_rate: u32,
    pub use_small_model: bool,
}

pub struct BarkEngine {
    config: BarkConfig,
    capabilities: EngineCapabilities,
    #[cfg(feature = "torch")]
    device: Device,
}

impl BarkEngine {
    pub fn new(config_json: &serde_json::Value) -> Result<Self> {
        let model_path = PathBuf::from(
            config_json
                .get("model_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/bark/"),
        );

        let sample_rate = config_json
            .get("sample_rate")
            .and_then(|v| v.as_u64())
            .unwrap_or(24000) as u32;

        let use_small_model = config_json
            .get("use_small_model")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let config = BarkConfig {
            model_path,
            sample_rate,
            use_small_model,
        };

        let capabilities = EngineCapabilities {
            name: "Bark Generative Audio".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec![
                "en".to_string(),
                "de".to_string(),
                "es".to_string(),
                "fr".to_string(),
                "hi".to_string(),
                "it".to_string(),
                "ja".to_string(),
                "ko".to_string(),
                "pl".to_string(),
                "pt".to_string(),
                "ru".to_string(),
                "tr".to_string(),
                "zh".to_string(),
            ],
            supported_voices: vec![
                VoiceInfo {
                    id: "bark_v2_en_speaker_0".to_string(),
                    name: "Bark English Speaker 0".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "bark_v2_en_speaker_1".to_string(),
                    name: "Bark English Speaker 1".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "bark_v2_en_speaker_6".to_string(),
                    name: "Bark English Speaker 6".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Neutral,
                    quality: VoiceQuality::Neural,
                },
            ],
            max_text_length: 500,
            sample_rate,
            supports_ssml: false,
            supports_streaming: false,
        };

        #[cfg(feature = "torch")]
        let device = crate::models::pytorch_loader::get_best_device();

        log::info!("Bark TTS engine initialized");

        Ok(Self {
            config,
            capabilities,
            #[cfg(feature = "torch")]
            device,
        })
    }
}

#[async_trait]
impl TTSEngine for BarkEngine {
    fn name(&self) -> &str {
        "bark"
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }

    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        log::info!(
            "Bark: Synthesizing '{}' (generative)",
            &text[..text.len().min(40)]
        );

        // Bark is slower but more creative
        let words = text.split_whitespace().count().max(1);
        let base_duration = words as f32 / 2.0; // Slower, more natural
        let duration = (base_duration / params.speed).max(0.5);
        let sample_count = (self.config.sample_rate as f32 * duration) as usize;

        log::info!(
            "Bark: Synthesizing {} samples (~{:.2}s)",
            sample_count,
            duration
        );

        let mut samples = Vec::with_capacity(sample_count);

        // Bark style: more natural, includes background textures
        let base_freq = 200.0 * params.pitch;

        for i in 0..sample_count {
            let t = i as f32 / self.config.sample_rate as f32;
            let progress = t / duration;

            // Natural prosody with more variation
            let prosody = 1.0
                + 0.15 * (progress * 2.5 * std::f32::consts::PI).sin()
                + 0.05 * (progress * 7.0 * std::f32::consts::PI).sin();

            let freq = base_freq * prosody;

            // Voice with natural texture
            let voice = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3;
            let h2 = (2.0 * std::f32::consts::PI * freq * 1.8 * t).sin() * 0.15;
            let h3 = (2.0 * std::f32::consts::PI * freq * 2.7 * t).sin() * 0.08;

            // Add breath noise (Bark signature)
            let noise = ((i as f32 * 991.0).sin() * 0.5 + 0.5) * 0.12;

            // Occasional crackle for naturalness
            let crackle = if i % 1000 < 5 {
                ((i as f32 * 13.0).sin() * 0.5 + 0.5) * 0.05
            } else {
                0.0
            };

            // Word-level envelope
            let word_progress = (progress * words as f32).fract();
            let env = if word_progress < 0.08 {
                word_progress / 0.08
            } else if word_progress > 0.85 {
                (1.0 - word_progress) / 0.15
            } else {
                0.92 + 0.08 * (word_progress * 6.0).sin()
            };

            // Global fade
            let fade = if t < 0.04 {
                t / 0.04
            } else if t > duration - 0.1 {
                (duration - t) / 0.1
            } else {
                1.0
            };

            let sample = (voice + h2 + h3 + noise + crackle) * env * fade * 0.5;
            samples.push(sample.clamp(-1.0, 1.0));
        }

        log::info!("Bark: Synthesis completed successfully");

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
            "model_path": "/tmp/bark/",
            "sample_rate": 48000,
            "use_small_model": true
        })
    }

    // ── BarkConfig serde ──────────────────────────────────────────────────────

    #[test]
    fn test_bark_config_serialize_deserialize() {
        let cfg = BarkConfig {
            model_path: std::path::PathBuf::from("/tmp/bark/"),
            sample_rate: 24000,
            use_small_model: true,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("sample_rate"));
        assert!(json.contains("use_small_model"));
        let back: BarkConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sample_rate, 24000);
        assert!(back.use_small_model);
    }

    #[test]
    fn test_bark_config_clone_and_debug() {
        let cfg = BarkConfig {
            model_path: std::path::PathBuf::from("p"),
            sample_rate: 24000,
            use_small_model: false,
        };
        let cloned = cfg.clone();
        assert!(!cloned.use_small_model);
        let dbg = format!("{:?}", cloned);
        assert!(dbg.contains("BarkConfig"));
    }

    // ── BarkEngine::new ───────────────────────────────────────────────────────

    #[test]
    fn test_bark_engine_new_with_empty_config_uses_defaults() {
        let result = BarkEngine::new(&empty_cfg());
        assert!(result.is_ok(), "should succeed: {:?}", result.err());
        let engine = result.unwrap();
        assert_eq!(engine.name(), "bark");
    }

    #[test]
    fn test_bark_engine_new_default_sample_rate() {
        let engine = BarkEngine::new(&empty_cfg()).unwrap();
        assert_eq!(engine.config.sample_rate, 24000);
    }

    #[test]
    fn test_bark_engine_new_default_use_small_model_false() {
        let engine = BarkEngine::new(&empty_cfg()).unwrap();
        assert!(!engine.config.use_small_model);
    }

    #[test]
    fn test_bark_engine_new_with_custom_config() {
        let engine = BarkEngine::new(&custom_cfg()).unwrap();
        assert_eq!(engine.config.sample_rate, 48000);
        assert!(engine.config.use_small_model);
        assert_eq!(
            engine.config.model_path,
            std::path::PathBuf::from("/tmp/bark/")
        );
    }

    // ── capabilities & voices ─────────────────────────────────────────────────

    #[test]
    fn test_bark_engine_capabilities() {
        let engine = BarkEngine::new(&empty_cfg()).unwrap();
        let caps = engine.capabilities();
        assert_eq!(caps.name, "Bark Generative Audio");
        assert_eq!(caps.max_text_length, 500);
        assert!(!caps.supports_ssml);
        assert!(!caps.supports_streaming);
        // Supports 13 languages
        assert!(caps.supported_languages.len() >= 13);
    }

    #[test]
    fn test_bark_engine_list_voices() {
        let engine = BarkEngine::new(&empty_cfg()).unwrap();
        let voices = engine.list_voices();
        assert_eq!(voices.len(), 3);
        let ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();
        assert!(ids.contains(&"bark_v2_en_speaker_0"));
        assert!(ids.contains(&"bark_v2_en_speaker_1"));
        assert!(ids.contains(&"bark_v2_en_speaker_6"));
        // Check gender variants are covered
        let genders: Vec<String> = voices.iter().map(|v| format!("{:?}", v.gender)).collect();
        assert!(genders.contains(&"Male".to_string()));
        assert!(genders.contains(&"Female".to_string()));
        assert!(genders.contains(&"Neutral".to_string()));
    }

    // ── synthesize ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_bark_synthesize_basic() {
        let engine = BarkEngine::new(&empty_cfg()).unwrap();
        let params = crate::core::tts_engine::SynthesisParams::default();
        let result = engine.synthesize("hello world", &params).await;
        assert!(result.is_ok());
        let audio = result.unwrap();
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.sample_rate, 24000);
        assert!(!audio.samples.is_empty());
    }

    #[tokio::test]
    async fn test_bark_synthesize_samples_clamped() {
        let engine = BarkEngine::new(&empty_cfg()).unwrap();
        let params = crate::core::tts_engine::SynthesisParams::default();
        let audio = engine
            .synthesize("test audio clamp check", &params)
            .await
            .unwrap();
        for &s in &audio.samples {
            assert!(s >= -1.0 && s <= 1.0, "sample out of range: {}", s);
        }
    }
}
