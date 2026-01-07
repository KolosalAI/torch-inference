/// Kokoro TTS Engine - High-quality neural TTS
/// Uses the Kokoro v1.0 82M parameter model
use anyhow::{Result, Context};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::tts_engine::{TTSEngine, SynthesisParams, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality};
use super::audio::AudioData;
use super::python_tts_bridge::KokoroPythonBridge;

#[cfg(feature = "torch")]
use super::phoneme_converter::EnhancedG2P;
#[cfg(feature = "torch")]
use super::g2p_misaki::MisakiG2P;
#[cfg(feature = "torch")]
use super::styletts2_model::StyleTTS2Inference;
#[cfg(feature = "torch")]
use super::istftnet_vocoder::ISTFTNetVocoder;

#[cfg(feature = "torch")]
use tch::{nn, Device, Tensor, Kind, CModule};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroConfig {
    pub model_path: PathBuf,
    pub sample_rate: u32,
}

#[allow(dead_code)]
pub struct KokoroEngine {
    config: KokoroConfig,
    capabilities: EngineCapabilities,
    python_bridge: Option<KokoroPythonBridge>,
    #[cfg(feature = "torch")]
    native_inference: Option<NativeInference>,
    #[cfg(feature = "torch")]
    device: Device,
}

#[cfg(feature = "torch")]
struct NativeInference {
    g2p_basic: EnhancedG2P,
    g2p_misaki: MisakiG2P,
    model: StyleTTS2Inference,
    vocoder: ISTFTNetVocoder,
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
        
        // Try to initialize native Rust inference
        #[cfg(feature = "torch")]
        let native_inference = Self::init_native_inference(&model_path, device, sample_rate);
        
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
            device,
        })
    }
    
    #[cfg(feature = "torch")]
    fn init_native_inference(model_path: &PathBuf, device: Device, sample_rate: u32) -> Option<NativeInference> {
        log::info!("Initializing native Kokoro inference pipeline");
        
        // Initialize basic G2P
        let g2p_basic = match EnhancedG2P::new() {
            Ok(g) => {
                log::info!("✅ Basic G2P converter initialized");
                g
            }
            Err(e) => {
                log::warn!("Failed to initialize basic G2P: {}", e);
                return None;
            }
        };
        
        // Initialize Misaki-style G2P (better quality)
        let g2p_misaki = match MisakiG2P::new() {
            Ok(g) => {
                log::info!("✅ Misaki G2P converter initialized (500+ word dictionary)");
                g
            }
            Err(e) => {
                log::warn!("Failed to initialize Misaki G2P: {}", e);
                return None;
            }
        };
        
        // Initialize model
        let model = match StyleTTS2Inference::new(model_path, device) {
            Ok(m) => {
                log::info!("✅ StyleTTS2 model loaded");
                m
            }
            Err(e) => {
                log::warn!("Failed to load StyleTTS2 model: {}", e);
                return None;
            }
        };
        
        // Initialize vocoder
        let vocoder = match ISTFTNetVocoder::new(None, device, sample_rate as i64) {
            Ok(v) => {
                log::info!("✅ ISTFTNet vocoder initialized");
                v
            }
            Err(e) => {
                log::warn!("Failed to initialize vocoder: {}", e);
                return None;
            }
        };
        
        log::info!("🎉 Native Kokoro inference fully initialized!");
        
        Some(NativeInference {
            g2p_basic,
            g2p_misaki,
            model,
            vocoder,
        })
    }
    
    #[cfg(feature = "torch")]
    fn synthesize_with_native(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        let inference = self.native_inference.as_ref()
            .context("Native inference not initialized")?;
        
        log::info!("🎙️ Native Rust synthesis: '{}'", &text[..text.len().min(50)]);
        
        // Step 1: Text to phonemes (use Misaki G2P for better quality)
        let phoneme_tokens = inference.g2p_misaki.text_to_tokens(text)
            .context("Failed to convert text to phonemes")?;
        
        log::debug!("Phoneme tokens: {} tokens (Misaki G2P)", phoneme_tokens.len());
        
        // Step 2: Phonemes to mel-spectrogram
        let speaker_id = self.voice_to_speaker_id(params.voice.as_deref());
        let mel = inference.model.tokens_to_mel(&phoneme_tokens, Some(speaker_id))
            .context("Failed to generate mel-spectrogram")?;
        
        log::debug!("Mel-spectrogram: {} x {}", mel.len(), mel[0].len());
        
        // Step 3: Mel to waveform
        let samples = inference.vocoder.mel_to_audio(&mel)
            .context("Failed to convert mel to audio")?;
        
        log::info!("✅ Generated {} samples ({:.2}s)", 
                   samples.len(), 
                   samples.len() as f32 / self.config.sample_rate as f32);
        
        Ok(AudioData {
            samples,
            sample_rate: self.config.sample_rate,
            channels: 1,
        })
    }
    
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
        // Try native Rust inference first (fastest, no Python overhead)
        #[cfg(feature = "torch")]
        {
            if self.native_inference.is_some() {
                log::info!("🦀 Using native Rust Kokoro inference");
                match self.synthesize_with_native(text, params) {
                    Ok(audio) => return Ok(audio),
                    Err(e) => {
                        log::warn!("Native inference failed: {}", e);
                        log::info!("Falling back to Python bridge...");
                    }
                }
            }
        }
        
        // Fallback to Python bridge
        if let Some(ref bridge) = self.python_bridge {
            log::info!("🐍 Using Kokoro Python bridge");
            return bridge.synthesize(text, params.voice.as_deref(), params.speed);
        }
        
        // No working TTS method available
        anyhow::bail!(
            "Kokoro TTS not available. Options:\n\
             1. Native Rust (requires proper model format)\n\
             2. Python bridge: pip install kokoro soundfile"
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
