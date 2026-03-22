/// Kokoro ONNX TTS Engine - Real ort 2.0.0-rc.10 inference
/// Uses ONNX Runtime for cross-platform neural TTS inference
use anyhow::{Result, Context};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use super::tts_engine::{TTSEngine, SynthesisParams, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality};
use super::audio::AudioData;

const VOICE_STYLE_DIM: usize = 256;
/// Number of style vectors per voice (one per possible phoneme-sequence length, up to 510)
const VOICE_PACK_SIZE: usize = 510;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroOnnxConfig {
    pub model_dir: PathBuf,
    pub sample_rate: u32,
}

pub struct KokoroOnnxEngine {
    session: Mutex<Session>,
    voice_styles: HashMap<String, Vec<f32>>,  // voice_id -> flat f32[VOICE_PACK_SIZE * VOICE_STYLE_DIM]
    config: KokoroOnnxConfig,
    capabilities: EngineCapabilities,
}

// Session is Send+Sync as documented by ort
unsafe impl Send for KokoroOnnxEngine {}
unsafe impl Sync for KokoroOnnxEngine {}

impl std::fmt::Debug for KokoroOnnxEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KokoroOnnxEngine")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl KokoroOnnxEngine {
    pub fn new(cfg: &serde_json::Value) -> Result<Self> {
        let model_dir = PathBuf::from(
            cfg.get("model_dir").and_then(|v| v.as_str()).unwrap_or("models/kokoro-82m")
        );
        // Accept several filename conventions
        let model_path = ["kokoro-v1.0.int8.onnx", "kokoro-v1.0.onnx", "kokoro-v1_0.onnx"]
            .iter()
            .map(|name| model_dir.join(name))
            .find(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!(
                "Kokoro ONNX model not found in {:?}. \
                 Download from: https://github.com/thewh1teagle/kokoro-onnx/releases",
                model_dir
            ))?;

        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;
        let session = builder.commit_from_file(&model_path)?;

        // Log I/O shapes for diagnostics (fields, not methods)
        for inp in &session.inputs  { log::info!("ONNX input:  {:?}", inp); }
        for out in &session.outputs { log::info!("ONNX output: {:?}", out); }

        // Preload all voice style .bin files
        let voices_dir = model_dir.join("voices");
        let mut voice_styles = HashMap::new();
        let default_voices = [
            "af_heart", "af_bella", "af_sarah", "af_nicole",
            "am_adam",  "am_michael",
            "bf_emma",  "bf_isabella",
            "bm_george","bm_lewis",
        ];
        for voice_id in default_voices {
            let bin_path = voices_dir.join(format!("{}.bin", voice_id));
            match Self::load_voice_style(&bin_path) {
                Ok(style) => { voice_styles.insert(voice_id.to_string(), style); }
                Err(e) => log::warn!("Voice {:?} not loaded: {}", voice_id, e),
            }
        }
        if !voice_styles.contains_key("af_heart") {
            log::warn!("Default voice af_heart not found in {:?}", voices_dir);
        }

        let sample_rate = cfg.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(24000) as u32;
        let config = KokoroOnnxConfig { model_dir, sample_rate };
        let capabilities = Self::build_capabilities(sample_rate);

        Ok(Self { session: Mutex::new(session), voice_styles, config, capabilities })
    }

    fn load_voice_style(path: &std::path::Path) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        let floats: Vec<f32> = bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let expected = VOICE_PACK_SIZE * VOICE_STYLE_DIM;
        anyhow::ensure!(floats.len() == expected,
            "Voice style {:?}: expected {} floats ({}x{}), got {}",
            path, expected, VOICE_PACK_SIZE, VOICE_STYLE_DIM, floats.len());
        Ok(floats)
    }

    fn build_capabilities(sample_rate: u32) -> EngineCapabilities {
        EngineCapabilities {
            name: "Kokoro ONNX TTS".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en-US".to_string(), "en-GB".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "af_heart".to_string(),
                    name: "Heart (American Female)".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "af_bella".to_string(),
                    name: "Bella (American Female)".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "af_sarah".to_string(),
                    name: "Sarah (American Female)".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "af_nicole".to_string(),
                    name: "Nicole (American Female)".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "am_adam".to_string(),
                    name: "Adam (American Male)".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "am_michael".to_string(),
                    name: "Michael (American Male)".to_string(),
                    language: "en-US".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "bf_emma".to_string(),
                    name: "Emma (British Female)".to_string(),
                    language: "en-GB".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "bf_isabella".to_string(),
                    name: "Isabella (British Female)".to_string(),
                    language: "en-GB".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "bm_george".to_string(),
                    name: "George (British Male)".to_string(),
                    language: "en-GB".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
                VoiceInfo {
                    id: "bm_lewis".to_string(),
                    name: "Lewis (British Male)".to_string(),
                    language: "en-GB".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
            ],
            max_text_length: 1000,
            sample_rate,
            supports_ssml: false,
            supports_streaming: false,
        }
    }

    fn synthesize_with_onnx(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        use crate::core::g2p_misaki::MisakiG2P;

        let g2p = MisakiG2P::new()?;
        let phoneme_tokens = g2p.text_to_tokens(text)?;
        anyhow::ensure!(!phoneme_tokens.is_empty(), "G2P produced no tokens for: {:?}", text);

        let phoneme_count = phoneme_tokens.len();

        // style: f32 [1, VOICE_STYLE_DIM]
        // Indexed by phoneme_count-1 (without BOS/EOS), matching Python: pack[len(ps)-1]
        let voice_id = params.voice.as_deref().unwrap_or("af_heart");
        let pack = self.voice_styles.get(voice_id)
            .or_else(|| self.voice_styles.get("af_heart"))
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; VOICE_PACK_SIZE * VOICE_STYLE_DIM]);
        let style_row = (phoneme_count.saturating_sub(1)).min(VOICE_PACK_SIZE - 1);
        let style_vec: Vec<f32> = pack[style_row * VOICE_STYLE_DIM..(style_row + 1) * VOICE_STYLE_DIM].to_vec();
        let style_tensor = Tensor::<f32>::from_array(([1usize, VOICE_STYLE_DIM], style_vec))?;

        // tokens: int64 [1, seq_len] with BOS(0) and EOS(0) matching PyTorch: [[0, *input_ids, 0]]
        let tokens_with_bos_eos: Vec<i64> = std::iter::once(0i64)
            .chain(phoneme_tokens.iter().copied())
            .chain(std::iter::once(0i64))
            .collect();
        let seq_len = tokens_with_bos_eos.len();
        let tokens_tensor = Tensor::<i64>::from_array(([1usize, seq_len], tokens_with_bos_eos))?;

        // speed: f32 [1]
        let speed_tensor = Tensor::<f32>::from_array(([1usize], vec![params.speed]))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "tokens" => tokens_tensor,
            "style"  => style_tensor,
            "speed"  => speed_tensor
        ])?;

        let (_shape, audio_slice) = outputs["audio"].try_extract_tensor::<f32>()?;
        let samples: Vec<f32> = audio_slice.iter().copied().collect();

        log::info!("Kokoro ONNX: {} phonemes ({} tokens w/ BOS/EOS) -> {} samples ({:.2}s)",
            phoneme_count, seq_len, samples.len(),
            samples.len() as f32 / self.config.sample_rate as f32);

        Ok(AudioData { samples, sample_rate: self.config.sample_rate, channels: 1 })
    }
}

#[async_trait]
impl TTSEngine for KokoroOnnxEngine {
    fn name(&self) -> &str {
        "kokoro-onnx"
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }

    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        self.validate_text(text)?;
        self.synthesize_with_onnx(text, params)
            .with_context(|| format!("Kokoro ONNX synthesis failed for: {:?}", &text[..text.len().min(40)]))
    }

    async fn warmup(&self) -> Result<()> {
        log::info!("Kokoro ONNX engine ready (model loaded at startup)");
        Ok(())
    }

    fn validate_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            anyhow::bail!("Text cannot be empty");
        }
        if text.len() > self.capabilities.max_text_length {
            anyhow::bail!("Text too long (max {} characters)", self.capabilities.max_text_length);
        }
        Ok(())
    }

    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.capabilities.supported_voices.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_engine_errors_when_model_absent() {
        let config = serde_json::json!({
            "model_dir": "/nonexistent/kokoro-onnx",
            "sample_rate": 24000
        });
        let result = KokoroOnnxEngine::new(&config);
        assert!(result.is_err(), "Expected Err when model file missing");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not found") || msg.contains("No such file") || msg.contains("model"),
                "Unexpected error: {}", msg);
    }

    #[test]
    fn test_load_voice_style_binary() {
        use std::io::Write;

        // Write 510*256 little-endian f32 values (all 0.5) to a temp file
        let n = VOICE_PACK_SIZE * VOICE_STYLE_DIM;
        let mut path = std::env::temp_dir();
        path.push("test_voice_style.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        for _ in 0..n {
            f.write_all(&0.5f32.to_le_bytes()).unwrap();
        }

        let loaded = super::KokoroOnnxEngine::load_voice_style(&path).unwrap();
        assert_eq!(loaded.len(), n);
        assert!((loaded[0] - 0.5).abs() < 1e-6);
        std::fs::remove_file(&path).ok();
    }
}
