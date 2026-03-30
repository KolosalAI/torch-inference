#![allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;

    // ──────────────────────────── SynthesisParams ────────────────────────────

    #[test]
    fn test_synthesis_params_default_values() {
        let p = SynthesisParams::default();
        assert_eq!(p.speed, 1.0);
        assert_eq!(p.pitch, 1.0);
        assert!(p.voice.is_none());
        assert!(p.language.is_none());
    }

    #[test]
    fn test_synthesis_params_custom_values() {
        let p = SynthesisParams {
            speed: 1.5,
            pitch: 0.8,
            voice: Some("en-US-Wavenet-A".to_string()),
            language: Some("en-US".to_string()),
        };
        assert_eq!(p.speed, 1.5);
        assert_eq!(p.pitch, 0.8);
        assert_eq!(p.voice.as_deref(), Some("en-US-Wavenet-A"));
        assert_eq!(p.language.as_deref(), Some("en-US"));
    }

    #[test]
    fn test_synthesis_params_clone() {
        let p = SynthesisParams {
            speed: 2.0,
            pitch: 1.2,
            voice: Some("v1".to_string()),
            language: Some("fr".to_string()),
        };
        let q = p.clone();
        assert_eq!(q.speed, p.speed);
        assert_eq!(q.pitch, p.pitch);
        assert_eq!(q.voice, p.voice);
        assert_eq!(q.language, p.language);
    }

    // ──────────────────────────── VoiceGender ────────────────────────────────

    #[test]
    fn test_voice_gender_variants_debug() {
        assert_eq!(format!("{:?}", VoiceGender::Male),    "Male");
        assert_eq!(format!("{:?}", VoiceGender::Female),  "Female");
        assert_eq!(format!("{:?}", VoiceGender::Neutral), "Neutral");
    }

    #[test]
    fn test_voice_gender_clone() {
        let g = VoiceGender::Female;
        let h = g.clone();
        assert_eq!(format!("{:?}", h), "Female");
    }

    // ──────────────────────────── VoiceQuality ───────────────────────────────

    #[test]
    fn test_voice_quality_variants_debug() {
        assert_eq!(format!("{:?}", VoiceQuality::Standard), "Standard");
        assert_eq!(format!("{:?}", VoiceQuality::High),     "High");
        assert_eq!(format!("{:?}", VoiceQuality::Premium),  "Premium");
        assert_eq!(format!("{:?}", VoiceQuality::Neural),   "Neural");
    }

    #[test]
    fn test_voice_quality_clone() {
        let q = VoiceQuality::Neural;
        let r = q.clone();
        assert_eq!(format!("{:?}", r), "Neural");
    }

    // ──────────────────────────── VoiceInfo ──────────────────────────────────

    #[test]
    fn test_voice_info_construction() {
        let v = VoiceInfo {
            id: "af_heart".to_string(),
            name: "Heart".to_string(),
            language: "en-US".to_string(),
            gender: VoiceGender::Female,
            quality: VoiceQuality::Neural,
        };
        assert_eq!(v.id, "af_heart");
        assert_eq!(v.name, "Heart");
        assert_eq!(v.language, "en-US");
        assert_eq!(format!("{:?}", v.gender),  "Female");
        assert_eq!(format!("{:?}", v.quality), "Neural");
    }

    #[test]
    fn test_voice_info_clone() {
        let v = VoiceInfo {
            id: "id1".to_string(),
            name: "Name1".to_string(),
            language: "de".to_string(),
            gender: VoiceGender::Male,
            quality: VoiceQuality::High,
        };
        let w = v.clone();
        assert_eq!(w.id, v.id);
        assert_eq!(w.language, v.language);
    }

    // ──────────────────────────── EngineCapabilities ─────────────────────────

    #[test]
    fn test_engine_capabilities_construction() {
        let caps = EngineCapabilities {
            name: "test-engine".to_string(),
            version: "2.0".to_string(),
            supported_languages: vec!["en".to_string(), "fr".to_string()],
            supported_voices: vec![],
            max_text_length: 5000,
            sample_rate: 22050,
            supports_ssml: true,
            supports_streaming: false,
        };
        assert_eq!(caps.name, "test-engine");
        assert_eq!(caps.version, "2.0");
        assert_eq!(caps.supported_languages.len(), 2);
        assert_eq!(caps.max_text_length, 5000);
        assert_eq!(caps.sample_rate, 22050);
        assert!(caps.supports_ssml);
        assert!(!caps.supports_streaming);
    }

    #[test]
    fn test_engine_capabilities_clone() {
        let caps = EngineCapabilities {
            name: "c1".to_string(),
            version: "1".to_string(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec![],
            max_text_length: 1000,
            sample_rate: 24000,
            supports_ssml: false,
            supports_streaming: true,
        };
        let c2 = caps.clone();
        assert_eq!(c2.name, caps.name);
        assert_eq!(c2.sample_rate, caps.sample_rate);
        assert!(c2.supports_streaming);
    }

    // ──────────────────────────── TTSEngineFactory ───────────────────────────

    #[test]
    fn test_factory_available_engines_is_non_empty() {
        let engines = TTSEngineFactory::available_engines();
        assert!(!engines.is_empty());
        assert!(engines.contains(&"torch"));
        assert!(engines.contains(&"kokoro-onnx"));
    }

    #[test]
    fn test_factory_create_unknown_engine_returns_err() {
        let cfg = serde_json::json!({});
        let result = TTSEngineFactory::create("nonexistent-engine-xyz", &cfg);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("Unknown engine type"), "error message should mention unknown type: {msg}");
    }

    // ──────────────────────────── TorchTTSEngine ─────────────────────────────

    #[test]
    fn test_torch_engine_new_succeeds() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_torch_engine_name() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        assert_eq!(engine.name(), "torch");
    }

    #[test]
    fn test_torch_engine_capabilities() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        let caps = engine.capabilities();
        assert_eq!(caps.sample_rate, 24000);
        assert_eq!(caps.max_text_length, 10000);
        assert!(!caps.supports_ssml);
        assert!(!caps.supports_streaming);
        assert!(caps.supported_voices.is_empty());
    }

    #[test]
    fn test_torch_engine_list_voices_is_empty() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        assert!(engine.list_voices().is_empty());
    }

    #[test]
    fn test_torch_engine_is_ready_default_true() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        assert!(engine.is_ready());
    }

    #[tokio::test]
    async fn test_torch_engine_synthesize_returns_err() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        let params = SynthesisParams::default();
        let result = engine.synthesize("Hello", &params).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("trained model"), "error should mention trained model: {msg}");
    }

    #[tokio::test]
    async fn test_torch_engine_warmup_ok() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        let result = engine.warmup().await;
        assert!(result.is_ok());
    }

    // ──────────────────────── TTSEngine::validate_text ───────────────────────

    #[test]
    fn test_validate_text_empty_returns_err() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        let result = engine.validate_text("");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("empty"), "error should mention empty: {msg}");
    }

    #[test]
    fn test_validate_text_too_long_returns_err() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        // max_text_length is 10000
        let long_text = "a".repeat(10001);
        let result = engine.validate_text(&long_text);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("too long"), "error should mention too long: {msg}");
    }

    #[test]
    fn test_validate_text_within_limit_ok() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        let result = engine.validate_text("Hello, world!");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_text_exactly_at_limit_ok() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        let at_limit = "a".repeat(10000);
        let result = engine.validate_text(&at_limit);
        assert!(result.is_ok());
    }

    // ── Line 90: default TTSEngine::warmup returns Ok(()) ────────────────────

    #[tokio::test]
    async fn test_default_warmup_returns_ok_unit() {
        let cfg = serde_json::json!({});
        let engine = TorchTTSEngine::new(&cfg).unwrap();
        // TorchTTSEngine doesn't override warmup(), calling the default (line 90)
        let result = engine.warmup().await;
        assert!(result.is_ok(), "default warmup must return Ok: {:?}", result);
    }

    // ── Line 125: Windows-SAPI bail on non-Windows ────────────────────────────

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_factory_create_windows_sapi_errors_on_non_windows() {
        let cfg = serde_json::json!({});
        let result = TTSEngineFactory::create("windows-sapi", &cfg);
        assert!(result.is_err(), "windows-sapi should fail on non-Windows");
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("Windows") || msg.contains("windows"), "{msg}");
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_factory_create_sapi_alias_errors_on_non_windows() {
        let cfg = serde_json::json!({});
        let result = TTSEngineFactory::create("sapi", &cfg);
        assert!(result.is_err(), "sapi should fail on non-Windows");
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("Windows") || msg.contains("windows"), "{msg}");
    }
}
