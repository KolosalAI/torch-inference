#![allow(dead_code)]
/// Windows SAPI TTS Engine - Real speech using Windows Speech API
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::process::Command;
use tempfile::NamedTempFile;

use super::audio::{AudioData, AudioProcessor};
use super::tts_engine::{
    EngineCapabilities, SynthesisParams, TTSEngine, VoiceGender, VoiceInfo, VoiceQuality,
};

/// Windows SAPI TTS Engine
pub struct WindowsSAPIEngine {
    capabilities: EngineCapabilities,
    audio_processor: AudioProcessor,
}

impl WindowsSAPIEngine {
    pub fn new() -> Result<Self> {
        let capabilities = EngineCapabilities {
            name: "Windows SAPI".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "david".to_string(),
                    name: "Microsoft David (Real Speech)".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "zira".to_string(),
                    name: "Microsoft Zira (Real Speech)".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
            ],
            max_text_length: 10000,
            sample_rate: 22050,
            supports_ssml: true,
            supports_streaming: false,
        };

        Ok(Self {
            capabilities,
            audio_processor: AudioProcessor::new(),
        })
    }

    fn generate_sapi_speech(&self, text: &str, rate: i32) -> Result<Vec<u8>> {
        // Create temporary file for output
        let temp_file = NamedTempFile::new()?.into_temp_path();
        let wav_path = temp_file.to_str().unwrap();

        log::debug!("Generating SAPI speech to: {}", wav_path);

        // Call PowerShell script
        let script_path = std::env::current_dir()?.join("sapi_tts.ps1");

        let output = Command::new("powershell")
            .args([
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                script_path.to_str().unwrap(),
                "-Text",
                text,
                "-OutputFile",
                wav_path,
                "-Rate",
                &rate.to_string(),
            ])
            .output()
            .context("Failed to execute PowerShell SAPI script")?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            log::error!("SAPI error: {}", error);
            return Err(anyhow::anyhow!("SAPI TTS failed: {}", error));
        }

        // Wait for file to be written
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Verify file exists
        if !std::path::Path::new(wav_path).exists() {
            return Err(anyhow::anyhow!("SAPI did not create output file"));
        }

        // Read the generated WAV file
        let wav_data = std::fs::read(wav_path).context("Failed to read generated WAV file")?;

        if wav_data.len() < 44 {
            return Err(anyhow::anyhow!(
                "Generated WAV file is invalid or empty ({} bytes)",
                wav_data.len()
            ));
        }

        log::debug!("[OK] SAPI generated {} bytes of audio", wav_data.len());
        Ok(wav_data)
    }
}

#[async_trait]
impl TTSEngine for WindowsSAPIEngine {
    fn name(&self) -> &str {
        "windows-sapi"
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }

    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        self.validate_text(text)?;

        log::info!("🎤 Generating REAL SPEECH with Windows SAPI: '{}'", text);

        // Convert speed to SAPI rate (-10 to 10)
        let rate = ((params.speed - 1.0) * 10.0).clamp(-10.0, 10.0) as i32;

        // Generate speech using SAPI
        let wav_data = self
            .generate_sapi_speech(text, rate)
            .context("SAPI speech generation failed")?;

        // Load the WAV file
        let audio = self
            .audio_processor
            .load_audio(&wav_data)
            .context("Failed to load generated audio")?;

        log::info!(
            "[OK] Generated REAL SPEECH: {:.2}s",
            audio.samples.len() as f32 / audio.sample_rate as f32
        );

        Ok(audio)
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.capabilities.supported_voices.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── WindowsSAPIEngine::new ────────────────────────────────────────────────

    #[test]
    fn test_windows_sapi_engine_new_succeeds() {
        let result = WindowsSAPIEngine::new();
        assert!(
            result.is_ok(),
            "WindowsSAPIEngine::new should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_windows_sapi_engine_name() {
        let engine = WindowsSAPIEngine::new().unwrap();
        assert_eq!(engine.name(), "windows-sapi");
    }

    // ── capabilities ──────────────────────────────────────────────────────────

    #[test]
    fn test_windows_sapi_capabilities() {
        let engine = WindowsSAPIEngine::new().unwrap();
        let caps = engine.capabilities();
        assert_eq!(caps.name, "Windows SAPI");
        assert_eq!(caps.version, "1.0.0");
        assert_eq!(caps.sample_rate, 22050);
        assert_eq!(caps.max_text_length, 10000);
        assert!(caps.supports_ssml);
        assert!(!caps.supports_streaming);
        assert!(caps.supported_languages.contains(&"en".to_string()));
    }

    // ── voices ────────────────────────────────────────────────────────────────

    #[test]
    fn test_windows_sapi_list_voices() {
        let engine = WindowsSAPIEngine::new().unwrap();
        let voices = engine.list_voices();
        assert_eq!(voices.len(), 2);
        let ids: Vec<&str> = voices.iter().map(|v| v.id.as_str()).collect();
        assert!(ids.contains(&"david"));
        assert!(ids.contains(&"zira"));
    }

    #[test]
    fn test_windows_sapi_voice_genders() {
        let engine = WindowsSAPIEngine::new().unwrap();
        let voices = engine.list_voices();
        let david = voices.iter().find(|v| v.id == "david").unwrap();
        let zira = voices.iter().find(|v| v.id == "zira").unwrap();
        assert_eq!(format!("{:?}", david.gender), "Male");
        assert_eq!(format!("{:?}", zira.gender), "Female");
    }

    #[test]
    fn test_windows_sapi_voice_quality_neural() {
        let engine = WindowsSAPIEngine::new().unwrap();
        let voices = engine.list_voices();
        for v in &voices {
            assert_eq!(format!("{:?}", v.quality), "Neural");
        }
    }

    #[test]
    fn test_windows_sapi_voice_language() {
        let engine = WindowsSAPIEngine::new().unwrap();
        let voices = engine.list_voices();
        for v in &voices {
            assert_eq!(v.language, "en-US");
        }
    }

    // ── is_ready (default impl) ───────────────────────────────────────────────

    #[test]
    fn test_windows_sapi_is_ready() {
        let engine = WindowsSAPIEngine::new().unwrap();
        assert!(engine.is_ready());
    }

    // ── synthesize error path (PowerShell not available on non-Windows) ────────

    #[tokio::test]
    async fn test_windows_sapi_synthesize_fails_gracefully_on_non_windows() {
        let engine = WindowsSAPIEngine::new().unwrap();
        let params = crate::core::tts_engine::SynthesisParams::default();
        // On non-Windows platforms powershell is unavailable, so this must err.
        // On Windows it would try to run the script (may also err without the ps1 file).
        let result = engine.synthesize("hello", &params).await;
        // We just assert it doesn't panic; on non-Windows it will be Err.
        #[cfg(not(target_os = "windows"))]
        assert!(result.is_err());
    }
}
