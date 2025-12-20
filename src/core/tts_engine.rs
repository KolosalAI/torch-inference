/// Generic TTS Engine Trait - Production-grade modular architecture
/// This allows any TTS implementation to be plugged in
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::audio::AudioData;

/// TTS synthesis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisParams {
    pub speed: f32,
    pub pitch: f32,
    pub voice: Option<String>,
    pub language: Option<String>,
}

impl Default for SynthesisParams {
    fn default() -> Self {
        Self {
            speed: 1.0,
            pitch: 1.0,
            voice: None,
            language: None,
        }
    }
}

/// TTS engine capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCapabilities {
    pub name: String,
    pub version: String,
    pub supported_languages: Vec<String>,
    pub supported_voices: Vec<VoiceInfo>,
    pub max_text_length: usize,
    pub sample_rate: u32,
    pub supports_ssml: bool,
    pub supports_streaming: bool,
}

/// Voice information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub language: String,
    pub gender: VoiceGender,
    pub quality: VoiceQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceGender {
    Male,
    Female,
    Neutral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceQuality {
    Standard,
    High,
    Premium,
    Neural,
}

/// Generic TTS Engine trait
#[async_trait]
pub trait TTSEngine: Send + Sync {
    /// Get engine name
    fn name(&self) -> &str;
    
    /// Get engine capabilities
    fn capabilities(&self) -> &EngineCapabilities;
    
    /// Synthesize text to speech
    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData>;
    
    /// Get available voices
    fn list_voices(&self) -> Vec<VoiceInfo>;
    
    /// Check if engine is ready
    fn is_ready(&self) -> bool {
        true
    }
    
    /// Warmup the engine
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
    
    /// Validate input text
    fn validate_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            anyhow::bail!("Text cannot be empty");
        }
        
        let max_len = self.capabilities().max_text_length;
        if text.len() > max_len {
            anyhow::bail!("Text too long: {} > {}", text.len(), max_len);
        }
        
        Ok(())
    }
}

/// TTS Engine factory for creating engines
pub struct TTSEngineFactory;

impl TTSEngineFactory {
    /// Create a production TTS engine by name
    pub fn create(engine_type: &str, config: &serde_json::Value) -> Result<Arc<dyn TTSEngine>> {
        match engine_type {
            // Production engines only
            "kokoro" => Ok(Arc::new(crate::core::kokoro_tts::KokoroEngine::new(config)?)),
            "kokoro-onnx" => Ok(Arc::new(crate::core::kokoro_onnx::KokoroOnnxEngine::new(config)?)),
            "windows-sapi" | "sapi" => {
                #[cfg(target_os = "windows")]
                {
                    Ok(Arc::new(crate::core::windows_sapi_tts::WindowsSAPIEngine::new()?))
                }
                #[cfg(not(target_os = "windows"))]
                {
                    anyhow::bail!("Windows SAPI engine is only available on Windows")
                }
            },
            "piper" => Ok(Arc::new(crate::core::piper_tts::PiperTTSEngine::new(config)?)),
            "vits" => Ok(Arc::new(crate::core::vits_tts::VITSEngine::new(config)?)),
            "styletts2" => Ok(Arc::new(crate::core::styletts2::StyleTTS2Engine::new(config)?)),
            "bark" => Ok(Arc::new(crate::core::bark_tts::BarkEngine::new(config)?)),
            "xtts" => Ok(Arc::new(crate::core::xtts::XTTSEngine::new(config)?)),
            "torch" => Ok(Arc::new(TorchTTSEngine::new(config)?)),
            _ => anyhow::bail!(
                "Unknown engine type: '{}'. Available production engines: kokoro, kokoro-onnx, windows-sapi, piper, vits, styletts2, bark, xtts, torch", 
                engine_type
            ),
        }
    }
    
    /// List available production engine types
    pub fn available_engines() -> Vec<&'static str> {
        vec!["kokoro", "kokoro-onnx", "windows-sapi", "piper", "vits", "styletts2", "bark", "xtts", "torch"]
    }
}

/// PyTorch-based TTS Engine (placeholder for future ML-based TTS)
pub struct TorchTTSEngine {
    capabilities: EngineCapabilities,
}

impl TorchTTSEngine {
    pub fn new(_config: &serde_json::Value) -> Result<Self> {
        let capabilities = EngineCapabilities {
            name: "PyTorch TTS Engine".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec![],
            max_text_length: 10000,
            sample_rate: 24000,
            supports_ssml: false,
            supports_streaming: false,
        };
        
        Ok(Self { capabilities })
    }
}

#[async_trait]
impl TTSEngine for TorchTTSEngine {
    fn name(&self) -> &str {
        "torch"
    }
    
    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }
    
    async fn synthesize(&self, _text: &str, _params: &SynthesisParams) -> Result<AudioData> {
        anyhow::bail!("PyTorch TTS engine requires a trained model. Please configure a model path.")
    }
    
    fn list_voices(&self) -> Vec<VoiceInfo> {
        vec![]
    }
}
