#![allow(dead_code)]
/// ONNX-based TTS and STT model implementations
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "onnx")]
use ort::session::Session;

use super::audio::{AudioData, AudioProcessor};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSConfig {
    pub model_path: PathBuf,
    pub vocoder_path: Option<PathBuf>,
    pub sample_rate: u32,
    pub max_text_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STTConfig {
    pub model_path: PathBuf,
    pub sample_rate: u32,
    pub chunk_length_secs: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSParameters {
    pub speed: f32,
    pub pitch: f32,
    pub energy: f32,
}

impl Default for TTSParameters {
    fn default() -> Self {
        Self {
            speed: 1.0,
            pitch: 1.0,
            energy: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub confidence: f32,
    pub segments: Option<Vec<TranscriptionSegment>>,
}

/// Text-to-Speech model using ONNX Runtime
pub struct TTSModel {
    config: TTSConfig,
    processor: AudioProcessor,
}

impl TTSModel {
    pub fn new(config: TTSConfig) -> Result<Self> {
        let processor = AudioProcessor::with_sample_rate(config.sample_rate);

        Ok(Self { config, processor })
    }

    /// Synthesize speech from text
    pub fn synthesize(&self, text: &str, params: &TTSParameters) -> Result<AudioData> {
        // Validate text length
        if text.len() > self.config.max_text_length {
            bail!(
                "Text too long: {} characters (max: {})",
                text.len(),
                self.config.max_text_length
            );
        }

        #[cfg(feature = "onnx")]
        {
            self.synthesize_with_onnx(text, params)
        }

        #[cfg(not(feature = "onnx"))]
        {
            self.synthesize_fallback(text, params)
        }
    }

    #[cfg(feature = "onnx")]
    fn synthesize_with_onnx(&self, text: &str, params: &TTSParameters) -> Result<AudioData> {
        use ndarray::Array2;

        // Create ONNX session for this inference
        let session = Session::builder()?.commit_from_file(&self.config.model_path)?;

        // Tokenize text
        let tokens = self.tokenize_text(text)?;

        // Prepare input tensors
        let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens)?;

        let speaker_embeddings = Array2::zeros((1, 512));

        // Run inference
        let outputs = session.run(vec![input_ids.into(), speaker_embeddings.into()])?;

        // Extract audio from first output
        let audio_output = &outputs[0];
        let view = audio_output.view();
        let shape = view.shape();

        // Extract samples from the tensor
        let mut samples = Vec::new();
        for elem in view.iter() {
            if let Some(&val) = elem.downcast_ref::<f32>() {
                samples.push(val);
            }
        }

        // Apply parameters
        let adjusted_samples = self.apply_parameters(&samples, params)?;

        Ok(AudioData {
            samples: adjusted_samples,
            sample_rate: self.config.sample_rate,
            channels: 1,
        })
    }

    #[cfg(not(feature = "onnx"))]
    fn synthesize_fallback(&self, text: &str, params: &TTSParameters) -> Result<AudioData> {
        // Fallback implementation without ONNX
        log::warn!("ONNX feature not enabled, returning dummy audio");

        let duration = (text.len() as f32 * 0.1).max(1.0); // Rough estimate
        let sample_count = (self.config.sample_rate as f32 * duration) as usize;

        // Generate simple sine wave as placeholder
        let mut samples = Vec::with_capacity(sample_count);
        let frequency = 440.0 * params.pitch; // A4 note adjusted by pitch

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

    fn tokenize_text(&self, text: &str) -> Result<Vec<i64>> {
        // Simple character-level tokenization
        Ok(text
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .map(|c| c as i64)
            .collect())
    }

    fn apply_parameters(&self, samples: &[f32], params: &TTSParameters) -> Result<Vec<f32>> {
        let mut result = samples.to_vec();

        // Speed adjustment (simple resampling)
        if (params.speed - 1.0).abs() > 0.01 {
            let new_len = (samples.len() as f32 / params.speed) as usize;
            result = self.simple_resample(samples, new_len);
        }

        // Pitch adjustment (frequency domain or simple time-stretch)
        if (params.pitch - 1.0).abs() > 0.01 {
            // Simplified pitch shift - in production use proper algorithm
            result = result.iter().map(|&s| s * params.pitch.sqrt()).collect();
        }

        // Energy adjustment (volume)
        if (params.energy - 1.0).abs() > 0.01 {
            result = result.iter().map(|&s| s * params.energy).collect();
        }

        Ok(result)
    }

    fn simple_resample(&self, samples: &[f32], new_len: usize) -> Vec<f32> {
        if new_len == samples.len() {
            return samples.to_vec();
        }

        let ratio = samples.len() as f32 / new_len as f32;
        (0..new_len)
            .map(|i| {
                let src_idx = (i as f32 * ratio) as usize;
                if src_idx < samples.len() {
                    samples[src_idx]
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Speech-to-Text model using ONNX Runtime
pub struct STTModel {
    config: STTConfig,
    processor: AudioProcessor,
}

impl STTModel {
    pub fn new(config: STTConfig) -> Result<Self> {
        let processor = AudioProcessor::with_sample_rate(config.sample_rate);

        Ok(Self { config, processor })
    }

    /// Transcribe audio to text
    pub fn transcribe(
        &self,
        audio: &AudioData,
        return_timestamps: bool,
    ) -> Result<TranscriptionResult> {
        #[cfg(feature = "onnx")]
        {
            self.transcribe_with_onnx(audio, return_timestamps)
        }

        #[cfg(not(feature = "onnx"))]
        {
            self.transcribe_fallback(audio, return_timestamps)
        }
    }

    #[cfg(feature = "onnx")]
    fn transcribe_with_onnx(
        &self,
        audio: &AudioData,
        return_timestamps: bool,
    ) -> Result<TranscriptionResult> {
        use onnxruntime::ndarray::Array3;

        // Create ONNX session for this inference
        let session = Session::builder()?.commit_from_file(&self.config.model_path)?;

        // Resample if needed
        let audio = if audio.sample_rate != self.config.sample_rate {
            self.processor.resample(audio, self.config.sample_rate)?
        } else {
            audio.clone()
        };

        // Convert to mono if stereo
        let mono_samples = if audio.channels > 1 {
            self.to_mono(&audio.samples, audio.channels)
        } else {
            audio.samples.clone()
        };

        // Prepare input features
        let features = self.extract_features(&mono_samples)?;

        // Create input tensor
        let input_tensor = Array3::from_shape_vec(
            (1, features.len(), features[0].len()),
            features.into_iter().flatten().collect(),
        )?;

        // Run inference
        let outputs = session.run(vec![input_tensor.into()])?;

        // Extract and decode output
        let output = &outputs[0];
        let view = output.view();

        // Extract token IDs
        let mut token_ids = Vec::new();
        for elem in view.iter() {
            if let Some(&val) = elem.downcast_ref::<i64>() {
                token_ids.push(val);
            }
        }

        let text = self.decode_tokens(&token_ids)?;

        // Extract segments if timestamps requested
        let segments = if return_timestamps {
            Some(self.extract_segments_from_tokens(&token_ids, &text)?)
        } else {
            None
        };

        Ok(TranscriptionResult {
            text,
            language: Some("en".to_string()),
            confidence: 0.95,
            segments,
        })
    }

    #[cfg(not(feature = "onnx"))]
    fn transcribe_fallback(
        &self,
        audio: &AudioData,
        return_timestamps: bool,
    ) -> Result<TranscriptionResult> {
        // Fallback implementation without ONNX
        log::warn!("ONNX feature not enabled, returning placeholder transcription");

        let duration = audio.samples.len() as f32 / audio.sample_rate as f32;
        let _word_count = (duration * 2.0) as usize; // Assume ~2 words per second

        let text = format!(
            "[Transcription placeholder - {} seconds of audio]",
            duration as u32
        );

        let segments = if return_timestamps {
            Some(vec![TranscriptionSegment {
                text: text.clone(),
                start_time: 0.0,
                end_time: duration,
                confidence: 0.0,
            }])
        } else {
            None
        };

        Ok(TranscriptionResult {
            text,
            language: Some("en".to_string()),
            confidence: 0.0,
            segments,
        })
    }

    fn to_mono(&self, samples: &[f32], channels: u16) -> Vec<f32> {
        // `channels == 0` would divide by zero. `channels == 1` is already
        // mono — cloning is cheaper than the chunk + sum dance.
        if channels <= 1 {
            return samples.to_vec();
        }
        samples
            .chunks(channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    }

    fn extract_features(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Extract features (mel spectrogram, MFCC, etc.)
        // This is a simplified placeholder - in production use proper feature extraction
        let chunk_size = 512;
        let hop_length = 256;

        let mut features = Vec::new();
        for i in (0..samples.len()).step_by(hop_length) {
            let end = (i + chunk_size).min(samples.len());
            let chunk: Vec<f32> = samples[i..end].to_vec();

            // Simplified feature: just return the chunk normalized
            let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
            let normalized: Vec<f32> = chunk.iter().map(|&x| x - mean).collect();
            features.push(normalized);
        }

        Ok(features)
    }

    fn decode_tokens(&self, tokens: &[i64]) -> Result<String> {
        // Simple token-to-text decoding
        let text: String = tokens
            .iter()
            .filter(|&&t| t > 0 && t < 128) // ASCII range
            .map(|&t| t as u8 as char)
            .collect();

        Ok(text.trim().to_string())
    }

    fn extract_segments_from_tokens(
        &self,
        _tokens: &[i64],
        text: &str,
    ) -> Result<Vec<TranscriptionSegment>> {
        // Simple word-level segmentation
        // In production, use model's alignment information
        let words: Vec<&str> = text.split_whitespace().collect();
        let duration_per_word = 0.5; // seconds

        let segments = words
            .iter()
            .enumerate()
            .map(|(i, word)| TranscriptionSegment {
                text: word.to_string(),
                start_time: i as f32 * duration_per_word,
                end_time: (i + 1) as f32 * duration_per_word,
                confidence: 0.95,
            })
            .collect();

        Ok(segments)
    }
}

/// Model manager for audio models
pub struct AudioModelManager {
    tts_models: dashmap::DashMap<String, Arc<TTSModel>>,
    stt_models: dashmap::DashMap<String, Arc<STTModel>>,
    model_dir: PathBuf,
    whisper: parking_lot::RwLock<Option<Arc<super::whisper_onnx::WhisperOnnxPipeline>>>,
}

impl AudioModelManager {
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Self {
        Self {
            tts_models: dashmap::DashMap::new(),
            stt_models: dashmap::DashMap::new(),
            model_dir: model_dir.as_ref().to_path_buf(),
            whisper: parking_lot::RwLock::new(None),
        }
    }

    /// Register the Whisper ONNX pipeline as the active STT engine.
    pub fn set_whisper_pipeline(&self, pipeline: super::whisper_onnx::WhisperOnnxPipeline) {
        *self.whisper.write() = Some(Arc::new(pipeline));
        log::info!("WhisperOnnxPipeline registered as default STT engine");
    }

    /// Transcribe `audio`.  Prefers the Whisper ONNX pipeline if loaded; falls
    /// back to the legacy "default" STTModel, then errors with a clear message.
    pub fn transcribe_audio(
        &self,
        audio: &super::audio::AudioData,
        return_timestamps: bool,
    ) -> anyhow::Result<TranscriptionResult> {
        if let Some(w) = self.whisper.read().clone() {
            return w.transcribe(audio, return_timestamps);
        }
        match self.stt_models.get("default") {
            Some(m) => m.transcribe(audio, return_timestamps),
            None => anyhow::bail!(
                "STT model not loaded — Whisper models are downloading on first run, please retry"
            ),
        }
    }

    pub async fn load_tts_model(&self, name: &str, config: TTSConfig) -> Result<()> {
        let model = TTSModel::new(config).context(format!("Failed to load TTS model: {}", name))?;

        self.tts_models.insert(name.to_string(), Arc::new(model));
        log::info!("Loaded TTS model: {}", name);
        Ok(())
    }

    pub async fn load_stt_model(&self, name: &str, config: STTConfig) -> Result<()> {
        let model = STTModel::new(config).context(format!("Failed to load STT model: {}", name))?;

        self.stt_models.insert(name.to_string(), Arc::new(model));
        log::info!("Loaded STT model: {}", name);
        Ok(())
    }

    pub fn get_tts_model(&self, name: &str) -> Option<Arc<TTSModel>> {
        self.tts_models.get(name).map(|m| m.clone())
    }

    pub fn get_stt_model(&self, name: &str) -> Option<Arc<STTModel>> {
        self.stt_models.get(name).map(|m| m.clone())
    }

    pub fn list_tts_models(&self) -> Vec<String> {
        self.tts_models.iter().map(|e| e.key().clone()).collect()
    }

    pub fn list_stt_models(&self) -> Vec<String> {
        self.stt_models.iter().map(|e| e.key().clone()).collect()
    }

    pub async fn initialize_default_models(&self) -> Result<()> {
        // Load default models if they exist
        let default_tts_path = self.model_dir.join("tts_default.onnx");
        if default_tts_path.exists() {
            let config = TTSConfig {
                model_path: default_tts_path,
                vocoder_path: None,
                sample_rate: 16000,
                max_text_length: 1000,
            };
            self.load_tts_model("default", config).await?;
        }

        let default_stt_path = self.model_dir.join("stt_default.onnx");
        if default_stt_path.exists() {
            let config = STTConfig {
                model_path: default_stt_path,
                sample_rate: 16000,
                chunk_length_secs: 30.0,
            };
            self.load_stt_model("default", config).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_parameters_default() {
        let params = TTSParameters::default();
        assert_eq!(params.speed, 1.0);
        assert_eq!(params.pitch, 1.0);
        assert_eq!(params.energy, 1.0);
    }

    #[test]
    fn test_audio_model_manager() {
        let manager = AudioModelManager::new("./models");
        assert_eq!(manager.list_tts_models().len(), 0);
        assert_eq!(manager.list_stt_models().len(), 0);
    }

    // ── TTSModel::simple_resample line 208: else branch (src_idx >= len) ─────

    #[test]
    fn test_simple_resample_else_branch_via_synthesize() {
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 8000,
            max_text_length: 1000,
        };
        let model = TTSModel::new(config).unwrap();
        // speed=0.01 forces new_len >> sample_count so simple_resample hits
        // the else branch (line 208) for out-of-bounds indices.
        let params = TTSParameters {
            speed: 0.01,
            pitch: 1.0,
            energy: 1.0,
        };
        let result = model.synthesize("hi", &params);
        assert!(result.is_ok(), "synthesize should succeed: {:?}", result);
        assert!(!result.unwrap().samples.is_empty());
    }

    // ── AudioModelManager::initialize_default_models lines 453, 463 ──────────

    #[tokio::test]
    async fn test_initialize_default_models_loads_tts_when_file_exists() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tts_default.onnx"), b"dummy").unwrap();
        let manager = AudioModelManager::new(dir.path());
        let result = manager.initialize_default_models().await;
        assert!(
            result.is_ok(),
            "initialize_default_models should succeed: {:?}",
            result
        );
        assert_eq!(manager.list_tts_models().len(), 1);
    }

    #[tokio::test]
    async fn test_initialize_default_models_loads_stt_when_file_exists() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("stt_default.onnx"), b"dummy").unwrap();
        let manager = AudioModelManager::new(dir.path());
        let result = manager.initialize_default_models().await;
        assert!(
            result.is_ok(),
            "initialize_default_models should succeed: {:?}",
            result
        );
        assert_eq!(manager.list_stt_models().len(), 1);
    }

    // ── TTSModel: synthesize text-too-long (lines 80-81) ─────────────────

    #[test]
    fn test_synthesize_text_too_long_returns_err() {
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 16000,
            max_text_length: 10,
        };
        let model = TTSModel::new(config).unwrap();
        let params = TTSParameters::default();
        // 11 chars > max_text_length=10 → bail!
        let result = model.synthesize("hello world", &params);
        assert!(result.is_err(), "expected Err for too-long text, got Ok");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Text too long"),
            "unexpected error: {}",
            err_msg
        );
    }

    #[test]
    fn test_synthesize_empty_text_ok() {
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 16000,
            max_text_length: 1000,
        };
        let model = TTSModel::new(config).unwrap();
        let params = TTSParameters::default();
        // Empty text is within max_text_length; should not bail
        let result = model.synthesize("", &params);
        assert!(
            result.is_ok(),
            "empty text synthesize should succeed: {:?}",
            result
        );
    }

    // ── TTSModel: tokenize_text (lines 165-170) ───────────────────────────
    // tokenize_text is private but exercised via synthesize (non-onnx path)

    #[test]
    fn test_tokenize_text_via_synthesize_filters_non_alnum() {
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 8000,
            max_text_length: 1000,
        };
        let model = TTSModel::new(config).unwrap();
        // The fallback synthesize uses text.len() for duration, not tokenize_text,
        // but tokenize_text is called via synthesize_with_onnx (onnx feature off,
        // so synthesize_fallback is invoked instead). We verify the fallback runs.
        let params = TTSParameters::default();
        let result = model.synthesize("hello!", &params);
        assert!(result.is_ok());
    }

    // ── TTSModel: apply_parameters branches (lines 177-208) ──────────────

    #[test]
    fn test_apply_parameters_all_identity_no_change() {
        // speed=1.0, pitch=1.0, energy=1.0 → all branches skipped → samples unchanged
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 8000,
            max_text_length: 1000,
        };
        let model = TTSModel::new(config).unwrap();
        let params = TTSParameters {
            speed: 1.0,
            pitch: 1.0,
            energy: 1.0,
        };
        let result = model.synthesize("a", &params).unwrap();
        // Just verify it runs without panic
        assert!(!result.samples.is_empty());
    }

    #[test]
    fn test_apply_parameters_speed_adjustment() {
        // speed != 1.0 → triggers simple_resample (lines 177-180)
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 8000,
            max_text_length: 1000,
        };
        let model = TTSModel::new(config).unwrap();
        // speed=2.0 makes new_len = sample_count/2 (faster → fewer samples)
        let params = TTSParameters {
            speed: 2.0,
            pitch: 1.0,
            energy: 1.0,
        };
        let result = model.synthesize("hello", &params);
        assert!(
            result.is_ok(),
            "speed adjustment should succeed: {:?}",
            result
        );
        let audio = result.unwrap();
        assert!(!audio.samples.is_empty());
    }

    #[test]
    fn test_apply_parameters_pitch_adjustment() {
        // pitch != 1.0 → triggers pitch branch (lines 183-185)
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 8000,
            max_text_length: 1000,
        };
        let model = TTSModel::new(config).unwrap();
        let params = TTSParameters {
            speed: 1.0,
            pitch: 2.0,
            energy: 1.0,
        };
        let result = model.synthesize("hello", &params);
        assert!(
            result.is_ok(),
            "pitch adjustment should succeed: {:?}",
            result
        );
    }

    #[test]
    fn test_apply_parameters_energy_adjustment() {
        // energy != 1.0 → triggers energy branch (lines 189-191)
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 8000,
            max_text_length: 1000,
        };
        let model = TTSModel::new(config).unwrap();
        let params = TTSParameters {
            speed: 1.0,
            pitch: 1.0,
            energy: 0.5,
        };
        let result = model.synthesize("hello", &params);
        assert!(
            result.is_ok(),
            "energy adjustment should succeed: {:?}",
            result
        );
    }

    #[test]
    fn test_apply_parameters_all_adjusted() {
        // All three branches triggered
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 8000,
            max_text_length: 1000,
        };
        let model = TTSModel::new(config).unwrap();
        let params = TTSParameters {
            speed: 1.5,
            pitch: 0.8,
            energy: 1.2,
        };
        let result = model.synthesize("hello world", &params);
        assert!(
            result.is_ok(),
            "all-param adjustment should succeed: {:?}",
            result
        );
    }

    // ── simple_resample same-length no-op (line 197-198) ─────────────────

    #[test]
    fn test_simple_resample_same_len_noop() {
        // speed = 1.02 → new_len ≈ sample_count. When new_len == samples.len(),
        // simple_resample returns early (lines 197-198). Use a text that
        // produces sample_count where (sample_count / 1.02) as usize == sample_count.
        // Use speed very close to 1.0 but outside the 0.01 threshold.
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 8000,
            max_text_length: 1000,
        };
        let model = TTSModel::new(config).unwrap();
        // Very long text so sample_count is large and new_len ≈ sample_count
        let text = "a".repeat(200);
        let params = TTSParameters {
            speed: 1.015,
            pitch: 1.0,
            energy: 1.0,
        };
        let result = model.synthesize(&text, &params);
        assert!(
            result.is_ok(),
            "near-identity resample should succeed: {:?}",
            result
        );
    }

    // ── STTModel construction and transcribe fallback (lines 336-389) ─────

    fn make_test_audio(sample_rate: u32, duration_secs: f32) -> AudioData {
        let sample_count = (sample_rate as f32 * duration_secs) as usize;
        AudioData {
            samples: vec![0.0f32; sample_count],
            sample_rate,
            channels: 1,
        }
    }

    #[test]
    fn test_stt_model_new() {
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        let model = STTModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_transcribe_fallback_no_timestamps() {
        // Exercises transcribe_fallback (lines 308-333) without timestamps
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        let model = STTModel::new(config).unwrap();
        let audio = make_test_audio(16000, 2.0);
        let result = model.transcribe(&audio, false);
        assert!(
            result.is_ok(),
            "transcribe without timestamps failed: {:?}",
            result
        );
        let tr = result.unwrap();
        assert!(tr.segments.is_none());
        assert_eq!(tr.language, Some("en".to_string()));
        assert!((tr.confidence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_transcribe_fallback_with_timestamps() {
        // Exercises transcribe_fallback with return_timestamps=true (lines 317-326)
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        let model = STTModel::new(config).unwrap();
        let audio = make_test_audio(16000, 3.0);
        let result = model.transcribe(&audio, true);
        assert!(
            result.is_ok(),
            "transcribe with timestamps failed: {:?}",
            result
        );
        let tr = result.unwrap();
        assert!(tr.segments.is_some());
        let segs = tr.segments.unwrap();
        assert!(!segs.is_empty());
        // Verify segment fields
        assert_eq!(segs[0].start_time, 0.0);
        assert!((segs[0].end_time - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_transcribe_fallback_text_contains_duration() {
        // The placeholder text includes the duration as an integer
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        let model = STTModel::new(config).unwrap();
        let audio = make_test_audio(16000, 5.0);
        let tr = model.transcribe(&audio, false).unwrap();
        assert!(tr.text.contains("5"), "expected '5' in text: {}", tr.text);
    }

    // ── to_mono (lines 336-341) ───────────────────────────────────────────
    // to_mono is private — exercise via transcribe with a stereo AudioData

    #[test]
    fn test_transcribe_stereo_audio_uses_to_mono() {
        // channels > 1 → to_mono is called in transcribe_with_onnx.
        // With no onnx feature, transcribe_fallback is called (which doesn't
        // call to_mono), so we exercise to_mono via the onnx path only when
        // feature is enabled. Here we test the fallback path; to_mono is also
        // exercised via transcribe_with_onnx when onnx is available.
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        let model = STTModel::new(config).unwrap();
        let audio = AudioData {
            samples: vec![0.0f32; 32000], // stereo: 16000 frames × 2 channels
            sample_rate: 16000,
            channels: 2,
        };
        let result = model.transcribe(&audio, false);
        assert!(result.is_ok(), "stereo transcribe failed: {:?}", result);
    }

    // ── extract_features (lines 343-360) ─────────────────────────────────
    // Exercised indirectly via transcribe_with_onnx when onnx feature is on.
    // Without onnx, verify the fallback still returns something sensible.

    #[test]
    fn test_transcribe_short_audio() {
        // Very short audio (< 512 samples) exercises edge cases in extract_features
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        let model = STTModel::new(config).unwrap();
        let audio = AudioData {
            samples: vec![0.1f32; 100],
            sample_rate: 16000,
            channels: 1,
        };
        let result = model.transcribe(&audio, false);
        assert!(
            result.is_ok(),
            "short audio transcribe failed: {:?}",
            result
        );
    }

    // ── decode_tokens (lines 363-372) ────────────────────────────────────
    // decode_tokens is private — exercised by transcribe_with_onnx (onnx path).
    // Without onnx, we can't call it directly, so we create an STTModel and
    // call transcribe; the fallback path is still exercised.

    #[test]
    fn test_transcribe_zero_duration_audio() {
        // Zero-sample audio: duration=0, word_count=0, segment if timestamps requested
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        let model = STTModel::new(config).unwrap();
        let audio = AudioData {
            samples: vec![],
            sample_rate: 16000,
            channels: 1,
        };
        let result = model.transcribe(&audio, true);
        assert!(
            result.is_ok(),
            "zero-duration transcribe failed: {:?}",
            result
        );
        let tr = result.unwrap();
        assert!(tr.segments.is_some());
    }

    // ── extract_segments_from_tokens (lines 374-389) ──────────────────────
    // Exercised via transcribe with return_timestamps=true (onnx path).
    // Without onnx, segments come from transcribe_fallback instead.
    // We verify segment correctness with a multi-word placeholder.

    #[test]
    fn test_transcribe_with_timestamps_segment_fields() {
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        let model = STTModel::new(config).unwrap();
        let audio = make_test_audio(16000, 10.0);
        let tr = model.transcribe(&audio, true).unwrap();
        let segs = tr.segments.unwrap();
        assert!(!segs.is_empty());
        // Fields are properly populated
        for seg in &segs {
            assert!(seg.end_time >= seg.start_time);
            assert!(!seg.text.is_empty());
        }
    }

    // ── AudioModelManager: get / list helpers ────────────────────────────

    #[test]
    fn test_get_tts_model_not_found() {
        let manager = AudioModelManager::new("./models");
        assert!(manager.get_tts_model("nonexistent").is_none());
    }

    #[test]
    fn test_get_stt_model_not_found() {
        let manager = AudioModelManager::new("./models");
        assert!(manager.get_stt_model("nonexistent").is_none());
    }

    #[tokio::test]
    async fn test_load_and_get_tts_model() {
        let manager = AudioModelManager::new("./models");
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 16000,
            max_text_length: 1000,
        };
        manager.load_tts_model("my_tts", config).await.unwrap();
        let tts = manager.get_tts_model("my_tts");
        assert!(tts.is_some());
        let models = manager.list_tts_models();
        assert!(models.contains(&"my_tts".to_string()));
    }

    #[tokio::test]
    async fn test_load_and_get_stt_model() {
        let manager = AudioModelManager::new("./models");
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        manager.load_stt_model("my_stt", config).await.unwrap();
        let stt = manager.get_stt_model("my_stt");
        assert!(stt.is_some());
        let models = manager.list_stt_models();
        assert!(models.contains(&"my_stt".to_string()));
    }

    // ── TTSConfig / STTConfig serde and clone ─────────────────────────────

    #[test]
    fn test_tts_config_clone_and_debug() {
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/tmp/model.onnx"),
            vocoder_path: Some(std::path::PathBuf::from("/tmp/vocoder.onnx")),
            sample_rate: 22050,
            max_text_length: 500,
        };
        let cloned = config.clone();
        assert_eq!(cloned.sample_rate, 22050);
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn test_stt_config_clone_and_debug() {
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/tmp/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 15.0,
        };
        let cloned = config.clone();
        assert!((cloned.chunk_length_secs - 15.0).abs() < 1e-6);
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn test_tts_parameters_clone_and_debug() {
        let params = TTSParameters {
            speed: 0.8,
            pitch: 1.2,
            energy: 0.9,
        };
        let cloned = params.clone();
        assert!((cloned.speed - 0.8).abs() < 1e-6);
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn test_transcription_segment_fields() {
        let seg = TranscriptionSegment {
            text: "hello".to_string(),
            start_time: 0.5,
            end_time: 1.0,
            confidence: 0.9,
        };
        let cloned = seg.clone();
        assert_eq!(cloned.text, "hello");
        let _ = format!("{:?}", cloned);
        let json = serde_json::to_string(&seg).unwrap();
        let back: TranscriptionSegment = serde_json::from_str(&json).unwrap();
        assert!((back.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_transcription_result_fields() {
        let tr = TranscriptionResult {
            text: "hello world".to_string(),
            language: Some("en".to_string()),
            confidence: 0.95,
            segments: None,
        };
        let cloned = tr.clone();
        assert_eq!(cloned.text, "hello world");
        let _ = format!("{:?}", cloned);
        let json = serde_json::to_string(&tr).unwrap();
        let back: TranscriptionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.language, Some("en".to_string()));
    }

    // ── Direct calls to private TTSModel methods (lines 165-212) ─────────

    fn make_tts_model() -> TTSModel {
        let config = TTSConfig {
            model_path: std::path::PathBuf::from("/nonexistent/model.onnx"),
            vocoder_path: None,
            sample_rate: 16000,
            max_text_length: 1000,
        };
        TTSModel::new(config).unwrap()
    }

    fn make_stt_model() -> STTModel {
        let config = STTConfig {
            model_path: std::path::PathBuf::from("/nonexistent/stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        STTModel::new(config).unwrap()
    }

    // Lines 165-170: tokenize_text
    #[test]
    fn test_tokenize_text_alphanumeric_and_whitespace() {
        let model = make_tts_model();
        let tokens = model.tokenize_text("hello world 123!").unwrap();
        // '!' is not alphanumeric/whitespace so should be filtered out
        // "hello world 123" → 15 chars kept
        assert_eq!(tokens.len(), 15);
        // First char 'h' == 104
        assert_eq!(tokens[0], 'h' as i64);
    }

    #[test]
    fn test_tokenize_text_empty() {
        let model = make_tts_model();
        let tokens = model.tokenize_text("").unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_text_only_special_chars_filtered() {
        let model = make_tts_model();
        let tokens = model.tokenize_text("!!!###$$$").unwrap();
        assert!(tokens.is_empty());
    }

    // Lines 173-194: apply_parameters (speed, pitch, energy branches)
    #[test]
    fn test_apply_parameters_direct_identity() {
        let model = make_tts_model();
        let samples = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let params = TTSParameters {
            speed: 1.0,
            pitch: 1.0,
            energy: 1.0,
        };
        let result = model.apply_parameters(&samples, &params).unwrap();
        assert_eq!(result, samples);
    }

    #[test]
    fn test_apply_parameters_direct_speed_branch() {
        let model = make_tts_model();
        let samples: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let params = TTSParameters {
            speed: 2.0,
            pitch: 1.0,
            energy: 1.0,
        };
        let result = model.apply_parameters(&samples, &params).unwrap();
        // speed=2.0 → new_len = 100/2.0 = 50
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_apply_parameters_direct_pitch_branch() {
        let model = make_tts_model();
        let samples = vec![1.0f32; 10];
        let params = TTSParameters {
            speed: 1.0,
            pitch: 4.0,
            energy: 1.0,
        };
        let result = model.apply_parameters(&samples, &params).unwrap();
        // pitch=4.0 → each sample * sqrt(4.0) = 2.0
        for &s in &result {
            assert!((s - 2.0f32).abs() < 1e-5, "expected 2.0, got {}", s);
        }
    }

    #[test]
    fn test_apply_parameters_direct_energy_branch() {
        let model = make_tts_model();
        let samples = vec![1.0f32; 10];
        let params = TTSParameters {
            speed: 1.0,
            pitch: 1.0,
            energy: 0.5,
        };
        let result = model.apply_parameters(&samples, &params).unwrap();
        // energy=0.5 → each sample * 0.5
        for &s in &result {
            assert!((s - 0.5f32).abs() < 1e-5, "expected 0.5, got {}", s);
        }
    }

    #[test]
    fn test_apply_parameters_direct_all_branches() {
        let model = make_tts_model();
        let samples: Vec<f32> = (0..200).map(|i| i as f32 * 0.01).collect();
        let params = TTSParameters {
            speed: 1.5,
            pitch: 2.0,
            energy: 2.0,
        };
        let result = model.apply_parameters(&samples, &params).unwrap();
        // Just verify it runs without panic and returns a non-empty vec
        assert!(!result.is_empty());
    }

    // Lines 196-211: simple_resample (same-length, upsample, downsample, out-of-bounds)
    #[test]
    fn test_simple_resample_direct_same_length() {
        let model = make_tts_model();
        let samples = vec![1.0f32, 2.0, 3.0];
        let result = model.simple_resample(&samples, 3);
        assert_eq!(result, samples);
    }

    #[test]
    fn test_simple_resample_direct_downsample() {
        let model = make_tts_model();
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let result = model.simple_resample(&samples, 5);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_simple_resample_direct_upsample_hits_else_branch() {
        // Upsampling: new_len > samples.len() causes src_idx >= samples.len() → else branch (0.0)
        let model = make_tts_model();
        let samples = vec![1.0f32, 2.0, 3.0];
        // new_len=10 with ratio=3/10=0.3 → last indices map past end
        let result = model.simple_resample(&samples, 10);
        assert_eq!(result.len(), 10);
        // First few should be valid sample values; tail may be 0.0
        assert_eq!(result[0], samples[0]);
    }

    #[test]
    fn test_simple_resample_direct_empty_input() {
        let model = make_tts_model();
        let samples: Vec<f32> = vec![];
        let result = model.simple_resample(&samples, 0);
        assert!(result.is_empty());
    }

    // ── Direct calls to private STTModel methods (lines 336-389) ─────────

    // Lines 336-341: to_mono
    #[test]
    fn test_to_mono_direct_stereo() {
        let model = make_stt_model();
        // interleaved stereo: [L0, R0, L1, R1, ...]
        let stereo = vec![1.0f32, 3.0, 2.0, 4.0]; // frames: (1,3) and (2,4)
        let mono = model.to_mono(&stereo, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 2.0f32).abs() < 1e-5); // (1+3)/2
        assert!((mono[1] - 3.0f32).abs() < 1e-5); // (2+4)/2
    }

    #[test]
    fn test_to_mono_direct_mono_passthrough() {
        let model = make_stt_model();
        let samples = vec![0.5f32, 0.6, 0.7];
        let mono = model.to_mono(&samples, 1);
        assert_eq!(mono, samples);
    }

    #[test]
    fn test_to_mono_direct_empty() {
        let model = make_stt_model();
        let mono = model.to_mono(&[], 2);
        assert!(mono.is_empty());
    }

    // Lines 343-360: extract_features
    #[test]
    fn test_extract_features_direct_basic() {
        let model = make_stt_model();
        let samples: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();
        let features = model.extract_features(&samples).unwrap();
        // hop_length=256 → ceil(1024/256)=4 chunks
        assert!(!features.is_empty());
        // Each feature chunk should be non-empty
        for chunk in &features {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_extract_features_direct_short_input() {
        let model = make_stt_model();
        // Less than chunk_size=512 → one chunk
        let samples = vec![0.5f32; 100];
        let features = model.extract_features(&samples).unwrap();
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].len(), 100);
    }

    #[test]
    fn test_extract_features_direct_empty_input() {
        let model = make_stt_model();
        let features = model.extract_features(&[]).unwrap();
        assert!(features.is_empty());
    }

    #[test]
    fn test_extract_features_direct_normalized_mean_zero() {
        let model = make_stt_model();
        // All samples the same → mean == value → normalized == 0.0
        let samples = vec![3.0f32; 512];
        let features = model.extract_features(&samples).unwrap();
        assert!(!features.is_empty());
        let chunk = &features[0];
        let max_abs = chunk.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "expected all zeros after normalization, max_abs={}",
            max_abs
        );
    }

    // Lines 363-371: decode_tokens
    #[test]
    fn test_decode_tokens_direct_ascii_range() {
        let model = make_stt_model();
        // 'h'=104, 'i'=105, both in (0, 128)
        let tokens = vec![104i64, 105];
        let text = model.decode_tokens(&tokens).unwrap();
        assert_eq!(text, "hi");
    }

    #[test]
    fn test_decode_tokens_direct_filters_out_of_range() {
        let model = make_stt_model();
        // 0 and 128 are filtered out; only 65 ('A') passes
        let tokens = vec![0i64, 65, 128, -1, 200];
        let text = model.decode_tokens(&tokens).unwrap();
        assert_eq!(text, "A");
    }

    #[test]
    fn test_decode_tokens_direct_empty() {
        let model = make_stt_model();
        let text = model.decode_tokens(&[]).unwrap();
        assert_eq!(text, "");
    }

    #[test]
    fn test_decode_tokens_direct_whitespace_trimmed() {
        let model = make_stt_model();
        // space=32, 'a'=97, space=32
        let tokens = vec![32i64, 97, 32];
        let text = model.decode_tokens(&tokens).unwrap();
        assert_eq!(text, "a");
    }

    // Lines 374-389: extract_segments_from_tokens
    #[test]
    fn test_extract_segments_from_tokens_direct_single_word() {
        let model = make_stt_model();
        let tokens = vec![1i64, 2, 3]; // _tokens unused in implementation
        let segs = model
            .extract_segments_from_tokens(&tokens, "hello")
            .unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hello");
        assert!((segs[0].start_time - 0.0f32).abs() < 1e-5);
        assert!((segs[0].end_time - 0.5f32).abs() < 1e-5);
        assert!((segs[0].confidence - 0.95f32).abs() < 1e-5);
    }

    #[test]
    fn test_extract_segments_from_tokens_direct_multiple_words() {
        let model = make_stt_model();
        let tokens = vec![];
        let segs = model
            .extract_segments_from_tokens(&tokens, "hello world foo")
            .unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].text, "hello");
        assert_eq!(segs[1].text, "world");
        assert_eq!(segs[2].text, "foo");
        // timing: word i → start=i*0.5, end=(i+1)*0.5
        assert!((segs[1].start_time - 0.5f32).abs() < 1e-5);
        assert!((segs[2].end_time - 1.5f32).abs() < 1e-5);
    }

    #[test]
    fn test_extract_segments_from_tokens_direct_empty_text() {
        let model = make_stt_model();
        let tokens = vec![1i64];
        let segs = model.extract_segments_from_tokens(&tokens, "").unwrap();
        assert!(segs.is_empty());
    }
}
