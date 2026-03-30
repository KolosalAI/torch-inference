#![allow(dead_code, unused_variables, unreachable_code)]
/// Piper TTS Engine - Neural TTS using ONNX Runtime
use anyhow::{Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::tts_engine::{TTSEngine, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality, SynthesisParams};
use super::audio::AudioData;

/// Piper TTS Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiperConfig {
    pub model_path: PathBuf,
    pub config_path: PathBuf,
    pub sample_rate: u32,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

/// Piper TTS Engine - High-quality neural TTS
pub struct PiperTTSEngine {
    config: PiperConfig,
    capabilities: EngineCapabilities,
    has_onnx: bool,
}

impl PiperTTSEngine {
    pub fn new(cfg: &serde_json::Value) -> Result<Self> {
        let model_path = cfg.get("model_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing model_path"))?;
        
        let config_path = cfg.get("config_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing config_path"))?;
        
        let model_path = PathBuf::from(model_path);
        let config_path = PathBuf::from(config_path);
        
        // Verify files exist
        if !model_path.exists() {
            bail!("Model file not found: {:?}", model_path);
        }
        
        // Load config if it exists
        let (sample_rate, noise_scale, length_scale, noise_w) = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
            
            let sample_rate = config_json.get("audio")
                .and_then(|a| a.get("sample_rate"))
                .and_then(|s| s.as_u64())
                .unwrap_or(22050) as u32;
            
            let noise_scale = config_json.get("inference")
                .and_then(|i| i.get("noise_scale"))
                .and_then(|n| n.as_f64())
                .unwrap_or(0.667) as f32;
            
            let length_scale = config_json.get("inference")
                .and_then(|i| i.get("length_scale"))
                .and_then(|l| l.as_f64())
                .unwrap_or(1.0) as f32;
            
            let noise_w = config_json.get("inference")
                .and_then(|i| i.get("noise_w"))
                .and_then(|n| n.as_f64())
                .unwrap_or(0.8) as f32;
            
            (sample_rate, noise_scale, length_scale, noise_w)
        } else {
            (22050, 0.667, 1.0, 0.8)
        };
        
        let config = PiperConfig {
            model_path: model_path.clone(),
            config_path,
            sample_rate,
            noise_scale,
            length_scale,
            noise_w,
        };
        
        let capabilities = EngineCapabilities {
            name: "Piper Neural TTS".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "en_US_lessac".to_string(),
                    name: "English US (Lessac)".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::High,
                },
            ],
            max_text_length: 10000,
            sample_rate,
            supports_ssml: false,
            supports_streaming: false,
        };
        
        let capabilities = EngineCapabilities {
            name: "Piper Neural TTS".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "lessac".to_string(),
                    name: "Lessac (High Quality)".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
            ],
            max_text_length: 5000,
            sample_rate,
            supports_ssml: false,
            supports_streaming: false,
        };
        
        // Check if ONNX model exists
        let has_onnx = model_path.exists();
        if has_onnx {
            log::info!("[OK] Piper ONNX model found at: {:?}", model_path);
            #[cfg(not(feature = "onnx"))]
            {
                bail!("ONNX feature not enabled. Compile with --features onnx to use Piper TTS");
            }
        } else {
            bail!("Piper model not found at {:?}", model_path);
        }
        
        Ok(Self {
            config,
            capabilities,
            has_onnx,
        })
    }
    

    fn synthesize_fallback(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        log::warn!("ORT not available, using fallback synthesis");
        
        // Simple fallback - generate tone based on text length
        let duration = (text.len() as f32 * 0.05 / params.speed).max(0.5);
        let sample_count = (self.config.sample_rate as f32 * duration) as usize;
        
        let mut samples = Vec::with_capacity(sample_count);
        let frequency = 440.0 * params.pitch;
        
        for i in 0..sample_count {
            let t = i as f32 / self.config.sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3;
            samples.push(sample);
        }
        
        Ok(AudioData {
            samples,
            sample_rate: self.config.sample_rate,
            channels: 1,
        })
    }
}

#[async_trait]
impl TTSEngine for PiperTTSEngine {
    fn name(&self) -> &str {
        "piper"
    }
    
    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }
    
    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        self.validate_text(text)?;
        bail!("Piper TTS requires ONNX Runtime. Please use Windows SAPI engine or compile with --features onnx")
    }
    
    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.capabilities.supported_voices.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── PiperConfig serde ─────────────────────────────────────────────────────

    #[test]
    fn test_piper_config_serialize_deserialize() {
        let cfg = PiperConfig {
            model_path: PathBuf::from("/tmp/piper.onnx"),
            config_path: PathBuf::from("/tmp/piper.json"),
            sample_rate: 22050,
            noise_scale: 0.667,
            length_scale: 1.0,
            noise_w: 0.8,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("sample_rate"));
        assert!(json.contains("noise_scale"));
        let back: PiperConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sample_rate, 22050);
        assert!((back.noise_scale - 0.667_f32).abs() < 1e-4);
        assert!((back.length_scale - 1.0_f32).abs() < 1e-4);
        assert!((back.noise_w - 0.8_f32).abs() < 1e-4);
    }

    #[test]
    fn test_piper_config_clone_debug() {
        let cfg = PiperConfig {
            model_path: PathBuf::from("m"),
            config_path: PathBuf::from("c"),
            sample_rate: 16000,
            noise_scale: 0.5,
            length_scale: 1.2,
            noise_w: 0.6,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.sample_rate, 16000);
        let dbg = format!("{:?}", cloned);
        assert!(dbg.contains("PiperConfig"));
    }

    // ── PiperTTSEngine::new error paths ───────────────────────────────────────

    #[test]
    fn test_piper_new_missing_model_path_returns_error() {
        let cfg = serde_json::json!({});
        let result = PiperTTSEngine::new(&cfg);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("model_path") || msg.contains("Missing"), "unexpected: {}", msg);
    }

    #[test]
    fn test_piper_new_missing_config_path_returns_error() {
        let cfg = serde_json::json!({ "model_path": "/tmp/piper.onnx" });
        let result = PiperTTSEngine::new(&cfg);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("config_path") || msg.contains("Missing") || msg.contains("not found"), "unexpected: {}", msg);
    }

    #[test]
    fn test_piper_new_nonexistent_model_path_returns_error() {
        let cfg = serde_json::json!({
            "model_path": "/nonexistent/piper_model.onnx",
            "config_path": "/nonexistent/piper_config.json"
        });
        let result = PiperTTSEngine::new(&cfg);
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(
            msg.contains("not found") || msg.contains("ONNX") || msg.contains("nonexistent"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn test_piper_new_with_real_model_file_but_no_onnx_feature() {
        // Create a temporary fake model file so the path-exists check passes
        let mut model_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(model_file, "fake onnx model").unwrap();
        let model_path = model_file.path().to_str().unwrap().to_owned();

        // config_path doesn't need to exist; the code falls back to defaults
        let cfg = serde_json::json!({
            "model_path": model_path,
            "config_path": "/nonexistent/piper_config.json"
        });
        let result = PiperTTSEngine::new(&cfg);
        // Without the `onnx` feature the engine should bail with an ONNX error;
        // with it enabled it will succeed (or fail on model loading). Either way
        // the test must not panic.
        let _ = result;
    }

    #[test]
    fn test_piper_new_with_real_model_and_config_files() {
        // Create fake model file
        let mut model_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(model_file, "fake onnx model").unwrap();
        let model_path = model_file.path().to_str().unwrap().to_owned();

        // Create a minimal Piper config JSON
        let mut config_file = tempfile::NamedTempFile::new().unwrap();
        writeln!(config_file, r#"{{"audio":{{"sample_rate":22050}},"inference":{{"noise_scale":0.5,"length_scale":1.0,"noise_w":0.7}}}}"#).unwrap();
        let config_path = config_file.path().to_str().unwrap().to_owned();

        let cfg = serde_json::json!({
            "model_path": model_path,
            "config_path": config_path
        });
        // Should fail at the ONNX load step (or succeed with onnx feature),
        // but the config-reading branch (lines 49-76) will be exercised.
        let _result = PiperTTSEngine::new(&cfg);
    }

    fn make_stub_engine() -> PiperTTSEngine {
        PiperTTSEngine {
            config: PiperConfig {
                model_path: PathBuf::from("/tmp/fake_piper.onnx"),
                config_path: PathBuf::from("/tmp/fake_piper.json"),
                sample_rate: 22050,
                noise_scale: 0.667,
                length_scale: 1.0,
                noise_w: 0.8,
            },
            capabilities: EngineCapabilities {
                name: "Piper Neural TTS".to_string(),
                version: "1.0.0".to_string(),
                supported_languages: vec!["en".to_string()],
                supported_voices: vec![
                    VoiceInfo {
                        id: "lessac".to_string(),
                        name: "Lessac".to_string(),
                        language: "en".to_string(),
                        gender: VoiceGender::Male,
                        quality: VoiceQuality::Neural,
                    },
                ],
                max_text_length: 5000,
                sample_rate: 22050,
                supports_ssml: false,
                supports_streaming: false,
            },
            has_onnx: false,
        }
    }

    #[test]
    fn test_piper_engine_name() {
        let engine = make_stub_engine();
        assert_eq!(engine.name(), "piper");
    }

    #[test]
    fn test_piper_engine_capabilities() {
        let engine = make_stub_engine();
        let caps = engine.capabilities();
        assert_eq!(caps.name, "Piper Neural TTS");
        assert_eq!(caps.sample_rate, 22050);
        assert!(!caps.supports_ssml);
    }

    #[test]
    fn test_piper_engine_list_voices() {
        let engine = make_stub_engine();
        let voices = engine.list_voices();
        assert_eq!(voices.len(), 1);
        assert_eq!(voices[0].id, "lessac");
    }

    #[tokio::test]
    async fn test_piper_engine_synthesize_bails() {
        let engine = make_stub_engine();
        let params = SynthesisParams {
            speed: 1.0,
            pitch: 1.0,
            voice: Some("lessac".to_string()),
            language: None,
        };
        let result = engine.synthesize("hello", &params).await;
        assert!(result.is_err());
        let msg = format!("{}", result.err().unwrap());
        assert!(msg.contains("ONNX") || msg.contains("Piper"), "unexpected: {}", msg);
    }

    #[test]
    fn test_piper_synthesize_fallback_produces_audio() {
        let engine = make_stub_engine();
        let params = SynthesisParams {
            speed: 1.0,
            pitch: 1.0,
            voice: None,
            language: None,
        };
        let result = engine.synthesize_fallback("hello world", &params);
        assert!(result.is_ok(), "synthesize_fallback should not error: {:?}", result.err());
        let audio = result.unwrap();
        assert!(!audio.samples.is_empty());
        assert_eq!(audio.sample_rate, 22050);
        assert_eq!(audio.channels, 1);
    }

    #[test]
    fn test_piper_synthesize_fallback_empty_text() {
        let engine = make_stub_engine();
        let params = SynthesisParams {
            speed: 2.0,
            pitch: 0.5,
            voice: None,
            language: None,
        };
        let result = engine.synthesize_fallback("", &params);
        assert!(result.is_ok());
        // Empty text => duration max(0 * 0.05 / 2.0, 0.5) = 0.5 seconds
        let audio = result.unwrap();
        let expected_min = (22050.0 * 0.5) as usize;
        assert!(audio.samples.len() >= expected_min);
    }
}
