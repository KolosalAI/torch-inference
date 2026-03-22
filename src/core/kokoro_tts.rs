/// Kokoro TTS Engine - High-quality neural TTS
/// Uses the Kokoro v1.0 82M parameter model
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::tts_engine::{TTSEngine, SynthesisParams, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality};
use super::audio::AudioData;
use super::python_tts_bridge::KokoroPythonBridge;

#[cfg(feature = "torch")]
use super::g2p_misaki::MisakiG2P;
#[cfg(feature = "torch")]
use super::styletts2_model::StyleTTS2Inference;
#[cfg(feature = "torch")]
use super::istftnet_vocoder::ISTFTNetVocoder;

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
    model:   StyleTTS2Inference,
    vocoder: ISTFTNetVocoder,
    device:  tch::Device,
}

impl KokoroEngine {
    pub fn new(config_json: &serde_json::Value) -> Result<Self> {
        let model_path = PathBuf::from(
            config_json.get("model_path")
                .and_then(|v| v.as_str())
                .unwrap_or("models/kokoro-82m/kokoro-v1_0.pth")
        );
        
        let sample_rate = config_json.get("sample_rate")
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
    fn init_native_inference(model_path: &std::path::Path, device: tch::Device) -> Option<NativeInference> {
        let safetensors = model_path.with_extension("safetensors");
        if !safetensors.exists() {
            log::warn!("Kokoro: {:?} not found. Run: python convert_kokoro.py", safetensors);
            return None;
        }
        let model = match StyleTTS2Inference::new(&safetensors, device) {
            Ok(m) => m,
            Err(e) => { log::warn!("Kokoro model load failed: {}", e); return None; }
        };
        let vocoder = match ISTFTNetVocoder::new(Some(&safetensors), device, 24000) {
            Ok(v) => v,
            Err(e) => { log::warn!("Kokoro vocoder load failed: {}", e); return None; }
        };
        log::info!("Kokoro native inference ready");
        Some(NativeInference { model, vocoder, device })
    }
    
    #[cfg(feature = "torch")]
    fn synthesize_with_native(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        let inf = self.native_inference.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Native inference not initialized"))?;

        let tokens = self.g2p.text_to_tokens(text)?;
        let speaker_id = self.voice_to_speaker_id(params.voice.as_deref());
        let mel = inf.model.tokens_to_mel(&tokens, Some(speaker_id))?;
        let samples = inf.vocoder.mel_to_audio(&mel, inf.device)?;

        Ok(AudioData { samples, sample_rate: self.config.sample_rate, channels: 1 })
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

    #[tokio::test]
    async fn test_kokoro_returns_error_when_model_absent() {
        let config = serde_json::json!({
            "model_path": "/nonexistent/kokoro-v1_0.safetensors",
            "sample_rate": 24000
        });
        let engine = KokoroEngine::new(&config).unwrap();
        // Engine constructs OK (model loading is lazy) but synthesize fails
        let params = crate::core::tts_engine::SynthesisParams::default();
        let result = engine.synthesize("hello", &params).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        // Must NOT silently produce audio — must return an error
        assert!(msg.contains("unavailable") || msg.contains("not found") || msg.contains("failed")
                || msg.contains("Kokoro") || msg.contains("TTS"),
                "Unexpected error: {}", msg);
    }
}
