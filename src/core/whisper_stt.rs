/// Whisper Speech-to-Text Engine
/// For validating TTS output quality by transcribing generated audio
use anyhow::{Result, Context};
use std::path::Path;

use super::audio::{AudioData, AudioProcessor};

#[cfg(feature = "torch")]
use tch::{nn, Device, Tensor, Kind, CModule};

pub struct WhisperConfig {
    pub model_path: String,
    pub language: Option<String>,
    pub task: String, // "transcribe" or "translate"
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_path: "models/whisper/whisper-base.pt".to_string(),
            language: Some("en".to_string()),
            task: "transcribe".to_string(),
        }
    }
}

pub struct WhisperEngine {
    config: WhisperConfig,
    #[cfg(feature = "torch")]
    model: Option<CModule>,
    #[cfg(feature = "torch")]
    device: Device,
}

impl WhisperEngine {
    pub fn new(config: WhisperConfig) -> Result<Self> {
        #[cfg(feature = "torch")]
        let device = crate::models::pytorch_loader::get_best_device();
        
        #[cfg(feature = "torch")]
        let model = if Path::new(&config.model_path).exists() {
            log::info!("Loading Whisper model from {}", config.model_path);
            match CModule::load_on_device(&config.model_path, device) {
                Ok(m) => {
                    log::info!("Whisper model loaded successfully");
                    Some(m)
                }
                Err(e) => {
                    log::warn!("Failed to load Whisper model: {}, using fallback", e);
                    None
                }
            }
        } else {
            log::warn!("Whisper model not found at {}", config.model_path);
            None
        };
        
        Ok(Self {
            config,
            #[cfg(feature = "torch")]
            model,
            #[cfg(feature = "torch")]
            device,
        })
    }
    
    /// Transcribe audio data to text
    pub fn transcribe(&self, audio: &AudioData) -> Result<String> {
        #[cfg(feature = "torch")]
        {
            if let Some(ref _model) = self.model {
                // TODO: Implement actual Whisper inference
                // For now, use simple pattern matching as fallback
                self.transcribe_fallback(audio)
            } else {
                self.transcribe_fallback(audio)
            }
        }
        
        #[cfg(not(feature = "torch"))]
        {
            self.transcribe_fallback(audio)
        }
    }
    
    /// Transcribe from WAV file
    pub fn transcribe_file<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let processor = AudioProcessor::new();
        let data = std::fs::read(path)?;
        let audio = processor.load_audio(&data)?;
        self.transcribe(&audio)
    }
    
    /// Fallback transcription using pattern analysis
    /// This analyzes audio characteristics to make educated guesses
    fn transcribe_fallback(&self, audio: &AudioData) -> Result<String> {
        log::info!("Using Whisper fallback transcription");
        
        // Analyze audio characteristics
        let duration = audio.samples.len() as f32 / audio.sample_rate as f32;
        let avg_amplitude = audio.samples.iter().map(|s| s.abs()).sum::<f32>() / audio.samples.len() as f32;
        
        // Count zero crossings (rough estimate of pitch/content)
        let mut zero_crossings = 0;
        for i in 1..audio.samples.len() {
            if (audio.samples[i-1] >= 0.0 && audio.samples[i] < 0.0) ||
               (audio.samples[i-1] < 0.0 && audio.samples[i] >= 0.0) {
                zero_crossings += 1;
            }
        }
        
        let avg_frequency = (zero_crossings as f32) / (2.0 * duration);
        
        // Estimate word count based on duration (rough: 2.5-3 words per second)
        let estimated_words = (duration * 2.7).round() as usize;
        
        log::info!("Audio analysis: duration={:.2}s, freq={:.1}Hz, est_words={}", 
            duration, avg_frequency, estimated_words);
        
        // For testing purposes, return a placeholder that indicates the audio was processed
        // In production with real Whisper model, this would return actual transcription
        Ok(format!("[FALLBACK] Detected speech: {:.2}s duration, ~{} words, {:.1}Hz avg frequency",
            duration, estimated_words, avg_frequency))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine_audio(sample_rate: u32, duration_secs: f32, freq: f32) -> AudioData {
        let n = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
            })
            .collect();
        AudioData { samples, sample_rate, channels: 1 }
    }

    #[test]
    fn test_whisper_engine_creation() {
        let config = WhisperConfig::default();
        let engine = WhisperEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_whisper_config_default_fields() {
        let config = WhisperConfig::default();
        assert_eq!(config.task, "transcribe");
        assert_eq!(config.language.as_deref(), Some("en"));
        assert!(!config.model_path.is_empty());
    }

    #[test]
    fn test_fallback_transcription() {
        let config = WhisperConfig::default();
        let engine = WhisperEngine::new(config).unwrap();

        let audio = make_sine_audio(16000, 2.0, 200.0);

        let result = engine.transcribe(&audio);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("duration"));
    }

    #[test]
    fn test_transcribe_contains_duration_info() {
        let config = WhisperConfig::default();
        let engine = WhisperEngine::new(config).unwrap();
        let audio = make_sine_audio(16000, 1.0, 440.0);
        let text = engine.transcribe(&audio).unwrap();
        assert!(text.contains("1.00") || text.contains("1.0") || text.contains("duration"),
                "Expected duration in output: {}", text);
    }

    #[test]
    fn test_transcribe_contains_words_estimate() {
        let config = WhisperConfig::default();
        let engine = WhisperEngine::new(config).unwrap();
        let audio = make_sine_audio(16000, 3.0, 200.0);
        let text = engine.transcribe(&audio).unwrap();
        assert!(text.contains("words") || text.contains("~"),
                "Expected word count in output: {}", text);
    }

    #[test]
    fn test_transcribe_silence() {
        let config = WhisperConfig::default();
        let engine = WhisperEngine::new(config).unwrap();
        // All-zero samples (silence)
        let audio = AudioData {
            samples: vec![0.0f32; 8000],
            sample_rate: 8000,
            channels: 1,
        };
        let result = engine.transcribe(&audio);
        assert!(result.is_ok(), "transcribe failed on silence: {:?}", result);
    }

    #[test]
    fn test_transcribe_single_sample() {
        let config = WhisperConfig::default();
        let engine = WhisperEngine::new(config).unwrap();
        let audio = AudioData {
            samples: vec![0.5f32],
            sample_rate: 16000,
            channels: 1,
        };
        // Should not panic/crash on tiny audio
        let result = engine.transcribe(&audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transcribe_returns_fallback_marker() {
        let config = WhisperConfig::default();
        let engine = WhisperEngine::new(config).unwrap();
        let audio = make_sine_audio(22050, 0.5, 300.0);
        let text = engine.transcribe(&audio).unwrap();
        // Fallback always prepends "[FALLBACK]"
        assert!(text.contains("[FALLBACK]"),
                "Expected [FALLBACK] marker: {}", text);
    }

    /// Exercises `transcribe_file` (lines 87-91) by writing a minimal WAV to a
    /// temporary file and calling the method on it.
    #[test]
    fn test_transcribe_file_with_valid_wav() {
        // Build a minimal WAV in memory using hound.
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
            for _ in 0..160 { // 10ms of silence
                writer.write_sample(0i16).unwrap();
            }
            writer.finalize().unwrap();
        }
        let wav_bytes = buf.into_inner();

        // Write to a temp file.
        let tmp_path = std::env::temp_dir().join("whisper_test_transcribe_file.wav");
        std::fs::write(&tmp_path, &wav_bytes).expect("Failed to write temp WAV");

        let config = WhisperConfig::default();
        let engine = WhisperEngine::new(config).unwrap();
        let result = engine.transcribe_file(&tmp_path);

        // Clean up before assertion.
        let _ = std::fs::remove_file(&tmp_path);

        assert!(result.is_ok(), "transcribe_file should succeed on a valid WAV: {:?}", result);
        let text = result.unwrap();
        assert!(text.contains("[FALLBACK]"), "Expected fallback marker in: {}", text);
    }

    /// Exercises `transcribe_file` error path (line 89) by passing a nonexistent
    /// file path — `std::fs::read` should return an error.
    #[test]
    fn test_transcribe_file_missing_path_returns_error() {
        let config = WhisperConfig::default();
        let engine = WhisperEngine::new(config).unwrap();
        let result = engine.transcribe_file("/tmp/definitely_does_not_exist_xyz.wav");
        assert!(result.is_err(), "transcribe_file with missing path must error");
    }

    #[test]
    fn test_custom_config() {
        let config = WhisperConfig {
            model_path: "/tmp/fake_model.pt".to_string(),
            language: None,
            task: "translate".to_string(),
        };
        let engine = WhisperEngine::new(config).unwrap();
        // Model file won't exist, but engine construction must succeed
        let audio = make_sine_audio(16000, 1.0, 440.0);
        let result = engine.transcribe(&audio);
        assert!(result.is_ok());
    }
}
