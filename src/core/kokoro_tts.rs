#![allow(dead_code)]
/// Kokoro TTS Engine - High-quality neural TTS
/// Uses the Kokoro v1.0 82M parameter model
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::audio::AudioData;
use super::python_tts_bridge::KokoroPythonBridge;
use super::tts_engine::{
    EngineCapabilities, SynthesisParams, TTSEngine, VoiceGender, VoiceInfo, VoiceQuality,
};

#[cfg(feature = "torch")]
use super::g2p_misaki::MisakiG2P;
#[cfg(feature = "torch")]
use super::istftnet_vocoder::ISTFTNetVocoder;
#[cfg(feature = "torch")]
use super::styletts2_model::StyleTTS2Inference;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroConfig {
    pub model_path: PathBuf,
    pub sample_rate: u32,
}

pub struct KokoroEngine {
    config: KokoroConfig,
    capabilities: EngineCapabilities,
    python_bridge: Option<KokoroPythonBridge>,
    #[cfg(feature = "torch")]
    native_inference: Option<NativeInference>,
    #[cfg(feature = "torch")]
    g2p: MisakiG2P,
    #[cfg(feature = "torch")]
    device: tch::Device,
}

#[cfg(feature = "torch")]
struct NativeInference {
    model: StyleTTS2Inference,
    vocoder: ISTFTNetVocoder,
    device: tch::Device,
}

impl KokoroEngine {
    pub fn new(config_json: &serde_json::Value) -> Result<Self> {
        let model_path = PathBuf::from(
            config_json
                .get("model_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/kokoro-82m/kokoro-v1_0.pth"),
        );

        let sample_rate = config_json
            .get("sample_rate")
            .and_then(|v| v.as_u64())
            .unwrap_or(24000) as u32;

        let config = KokoroConfig {
            model_path: model_path.clone(),
            sample_rate,
        };

        let capabilities = EngineCapabilities {
            name: "Kokoro TTS v1.0".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en-US".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "af".to_string(),
                    name: "American Female".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "am".to_string(),
                    name: "American Male".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "bf".to_string(),
                    name: "British Female".to_string(),
                    language: "en-GB".to_string(),
                    gender: VoiceGender::Female,
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

        #[cfg(feature = "torch")]
        let g2p = MisakiG2P::new()?;

        // Try to initialize native Rust inference
        #[cfg(feature = "torch")]
        let native_inference = Self::init_native_inference(&model_path, device);

        // Try to initialize Python bridge as fallback
        let python_bridge = match KokoroPythonBridge::new() {
            Ok(bridge) => {
                log::info!("Kokoro Python bridge initialized successfully");
                Some(bridge)
            }
            Err(e) => {
                log::warn!("Kokoro Python bridge not available: {}", e);
                log::warn!("Install with: pip install kokoro soundfile");
                None
            }
        };

        log::info!("Kokoro TTS engine initialized");

        Ok(Self {
            config,
            capabilities,
            python_bridge,
            #[cfg(feature = "torch")]
            native_inference,
            #[cfg(feature = "torch")]
            g2p,
            #[cfg(feature = "torch")]
            device,
        })
    }

    #[cfg(feature = "torch")]
    fn init_native_inference(
        model_path: &std::path::Path,
        device: tch::Device,
    ) -> Option<NativeInference> {
        let safetensors = model_path.with_extension("safetensors");
        if !safetensors.exists() {
            log::warn!(
                "Kokoro: {:?} not found. Run: python convert_kokoro.py",
                safetensors
            );
            return None;
        }
        let model = match StyleTTS2Inference::new(&safetensors, device) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("Kokoro model load failed: {}", e);
                return None;
            }
        };
        let vocoder = match ISTFTNetVocoder::new(Some(&safetensors), device, 24000) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("Kokoro vocoder load failed: {}", e);
                return None;
            }
        };
        log::info!("Kokoro native inference ready");
        Some(NativeInference {
            model,
            vocoder,
            device,
        })
    }

    #[cfg(feature = "torch")]
    fn synthesize_with_native(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        let inf = self
            .native_inference
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Native inference not initialized"))?;

        let tokens = self.g2p.text_to_tokens(text)?;
        let speaker_id = self.voice_to_speaker_id(params.voice.as_deref());
        let mel = inf.model.tokens_to_mel(&tokens, Some(speaker_id))?;
        let samples = inf.vocoder.mel_to_audio(&mel, inf.device)?;

        Ok(AudioData {
            samples,
            sample_rate: self.config.sample_rate,
            channels: 1,
        })
    }

    #[cfg(feature = "torch")]
    fn voice_to_speaker_id(&self, voice: Option<&str>) -> i64 {
        match voice {
            Some("af_heart") => 0,
            Some("af_bella") => 1,
            Some("af_sarah") => 2,
            Some("af_nicole") => 3,
            Some("am_adam") => 4,
            Some("am_michael") => 5,
            Some("bf_emma") => 6,
            Some("bf_isabella") => 7,
            Some("bm_george") => 8,
            Some("bm_lewis") => 9,
            _ => 0, // Default to af_heart
        }
    }
}

#[async_trait]
impl TTSEngine for KokoroEngine {
    fn name(&self) -> &str {
        "kokoro"
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }

    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        self.validate_text(text)?;

        #[cfg(feature = "torch")]
        if self.native_inference.is_some() {
            return self.synthesize_with_native(text, params);
        }

        if let Some(ref bridge) = self.python_bridge {
            return bridge.synthesize(text, params.voice.as_deref(), params.speed);
        }

        // Last resort: delegate to the shared Kokoro ONNX engine (same model, different runtime).
        if let Some(backend) = crate::core::onnx_backend::get_kokoro_onnx_backend() {
            let kokoro_voice =
                crate::core::onnx_backend::map_voice(params.voice.as_deref());
            let mapped = super::tts_engine::SynthesisParams {
                voice: Some(kokoro_voice.to_string()),
                ..params.clone()
            };
            log::info!(
                "Kokoro: Python bridge unavailable, delegating to ONNX engine (voice={})",
                kokoro_voice
            );
            return backend.synthesize(text, &mapped).await;
        }

        anyhow::bail!(
            "Kokoro TTS unavailable. Run: python convert_kokoro.py  \
             then: cargo build --features torch"
        )
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.capabilities.supported_voices.clone()
    }

    fn is_ready(&self) -> bool {
        // Ready if native inference OR Python bridge is available
        #[cfg(feature = "torch")]
        {
            if self.native_inference.is_some() {
                return true;
            }
        }

        if self.python_bridge.is_some() {
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tts_engine::{SynthesisParams, TTSEngine};

    fn make_engine(model_path: &str, sample_rate: u64) -> KokoroEngine {
        let config = serde_json::json!({
            "model_path": model_path,
            "sample_rate": sample_rate
        });
        KokoroEngine::new(&config).unwrap()
    }

    #[tokio::test]
    async fn test_kokoro_returns_error_when_model_absent() {
        let engine = make_engine("/nonexistent/kokoro-v1_0.safetensors", 24000);
        let params = SynthesisParams::default();
        let result = engine.synthesize("hello", &params).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("unavailable")
                || msg.contains("not found")
                || msg.contains("failed")
                || msg.contains("Kokoro")
                || msg.contains("TTS"),
            "Unexpected error: {}",
            msg
        );
    }

    #[test]
    fn test_kokoro_engine_name() {
        let engine = make_engine("/nonexistent/model.pth", 24000);
        assert_eq!(engine.name(), "kokoro");
    }

    #[test]
    fn test_kokoro_engine_capabilities_sample_rate() {
        let engine = make_engine("/nonexistent/model.pth", 22050);
        assert_eq!(engine.capabilities().sample_rate, 22050);
    }

    #[test]
    fn test_kokoro_engine_capabilities_name() {
        let engine = make_engine("/nonexistent/model.pth", 24000);
        let caps = engine.capabilities();
        assert!(caps.name.contains("Kokoro"));
    }

    #[test]
    fn test_kokoro_engine_default_sample_rate() {
        // When no sample_rate is provided, should default to 24000
        let config = serde_json::json!({ "model_path": "/nonexistent/model.pth" });
        let engine = KokoroEngine::new(&config).unwrap();
        assert_eq!(engine.capabilities().sample_rate, 24000);
    }

    #[test]
    fn test_kokoro_engine_default_model_path() {
        // When no model_path is provided, should use the built-in default
        let config = serde_json::json!({});
        let engine = KokoroEngine::new(&config).unwrap();
        assert!(engine
            .config
            .model_path
            .to_string_lossy()
            .contains("kokoro"));
    }

    #[test]
    fn test_list_voices_not_empty() {
        let engine = make_engine("/nonexistent/model.pth", 24000);
        let voices = engine.list_voices();
        assert!(
            !voices.is_empty(),
            "list_voices should return at least one voice"
        );
    }

    #[test]
    fn test_list_voices_contains_expected_ids() {
        let engine = make_engine("/nonexistent/model.pth", 24000);
        let voices = engine.list_voices();
        let ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();
        assert!(ids.contains(&"af"), "expected 'af' voice: {:?}", ids);
        assert!(ids.contains(&"am"), "expected 'am' voice: {:?}", ids);
    }

    #[test]
    fn test_is_ready_false_when_no_model() {
        let engine = make_engine("/nonexistent/model.pth", 24000);
        // No model file exists, no Python bridge expected in test env
        // is_ready() should return false
        assert!(!engine.is_ready());
    }

    #[tokio::test]
    async fn test_synthesize_empty_text_returns_error() {
        let engine = make_engine("/nonexistent/model.pth", 24000);
        let params = SynthesisParams::default();
        let result = engine.synthesize("", &params).await;
        assert!(result.is_err(), "empty text should fail");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("empty") || msg.contains("Text"),
            "Unexpected error for empty text: {}",
            msg
        );
    }

    #[tokio::test]
    async fn test_synthesize_too_long_text_returns_error() {
        let engine = make_engine("/nonexistent/model.pth", 24000);
        let params = SynthesisParams::default();
        // max_text_length is 1000; build a string of 1001 'a's
        let long_text = "a".repeat(1001);
        let result = engine.synthesize(&long_text, &params).await;
        assert!(result.is_err(), "too-long text should fail");
    }

    #[test]
    fn test_kokoro_config_debug_clone() {
        let config = KokoroConfig {
            model_path: std::path::PathBuf::from("/tmp/model.pth"),
            sample_rate: 24000,
        };
        let cloned = config.clone();
        let _ = format!("{:?}", cloned);
        let json = serde_json::to_string(&config).unwrap();
        let back: KokoroConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sample_rate, 24000);
    }

    /// Cover line 225: `is_ready()` returns `true` when python_bridge is Some.
    /// Constructed directly to inject a stub KokoroPythonBridge (unit struct
    /// when python feature is disabled).
    #[test]
    #[cfg(not(feature = "torch"))]
    fn test_is_ready_true_when_python_bridge_present() {
        use crate::core::python_tts_bridge::KokoroPythonBridge;
        use crate::core::tts_engine::EngineCapabilities;

        let capabilities = EngineCapabilities {
            name: "Kokoro TTS v1.0".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en-US".to_string()],
            supported_voices: vec![],
            max_text_length: 1000,
            sample_rate: 24000,
            supports_ssml: false,
            supports_streaming: false,
        };

        let engine = KokoroEngine {
            config: KokoroConfig {
                model_path: std::path::PathBuf::from("/nonexistent/model.pth"),
                sample_rate: 24000,
            },
            capabilities,
            python_bridge: Some(KokoroPythonBridge),
        };

        // With a Some(bridge), is_ready() hits the `return true` at line 225.
        assert!(
            engine.is_ready(),
            "is_ready() should be true when python_bridge is Some"
        );
    }
}
