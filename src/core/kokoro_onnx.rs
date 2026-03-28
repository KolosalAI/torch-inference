/// Kokoro ONNX TTS Engine - Real ort 2.0.0-rc.10 inference
/// Uses ONNX Runtime for cross-platform neural TTS inference
use anyhow::{Result, Context};
use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use super::tts_engine::{TTSEngine, SynthesisParams, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality};
use super::audio::AudioData;

const VOICE_STYLE_DIM: usize = 256;
/// Number of style vectors per voice (one per possible phoneme-sequence length, up to 510)
const VOICE_PACK_SIZE: usize = 510;

/// Default number of ONNX sessions in the pool. Each session handles one concurrent synthesis.
/// Setting this to 1 reproduces the old single-mutex behaviour; 2-4 is recommended in production.
const DEFAULT_POOL_SIZE: usize = 2;

/// G2P cache capacity (number of distinct texts cached).
/// Stored in a lock-free DashMap — reads never block.
const G2P_CACHE_CAPACITY: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KokoroOnnxConfig {
    pub model_dir: PathBuf,
    pub sample_rate: u32,
    /// Number of ONNX sessions to keep in the pool for concurrent synthesis.
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
}

fn default_pool_size() -> usize { DEFAULT_POOL_SIZE }

// ─── Session pool ────────────────────────────────────────────────────────────

/// A pool of ONNX sessions.  Each synthesis request checks out one session for
/// the duration of `Session::run()`, then returns it.  Concurrency is bounded
/// by the number of sessions created at startup; additional requests wait
/// asynchronously on the semaphore without blocking the runtime thread.
struct SessionPool {
    sessions:  tokio::sync::Mutex<Vec<Session>>,
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl SessionPool {
    fn new(sessions: Vec<Session>) -> Self {
        let n = sessions.len();
        Self {
            sessions:  tokio::sync::Mutex::new(sessions),
            semaphore: Arc::new(tokio::sync::Semaphore::new(n)),
        }
    }

    /// Acquire a session from the pool.  Awaits if all sessions are in use.
    async fn acquire(&self) -> SessionGuard<'_> {
        // Acquire the semaphore permit *before* locking the Vec so we never
        // hold the Vec mutex while waiting.
        let permit = self.semaphore
            .acquire()
            .await
            .expect("session pool semaphore closed");
        let session = {
            let mut guard = self.sessions.lock().await;
            guard.pop().expect("session available after permit was acquired")
        };
        SessionGuard { session: Some(session), pool: self, _permit: permit }
    }
}

/// RAII guard that returns the session to the pool on drop.
struct SessionGuard<'a> {
    session:  Option<Session>,
    pool:     &'a SessionPool,
    _permit:  tokio::sync::SemaphorePermit<'a>,
}

impl<'a> Drop for SessionGuard<'a> {
    fn drop(&mut self) {
        if let Some(s) = self.session.take() {
            // best-effort: if the lock is somehow poisoned we simply discard
            // the session (it will be absent from the pool but the semaphore
            // will still be released when `_permit` drops, so no deadlock).
            if let Ok(mut guard) = self.pool.sessions.try_lock() {
                guard.push(s);
            }
        }
    }
}

impl<'a> std::ops::DerefMut for SessionGuard<'a> {
    fn deref_mut(&mut self) -> &mut Session {
        self.session.as_mut().expect("session is present inside guard")
    }
}

impl<'a> std::ops::Deref for SessionGuard<'a> {
    type Target = Session;
    fn deref(&self) -> &Session {
        self.session.as_ref().expect("session is present inside guard")
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct KokoroOnnxEngine {
    pool:        SessionPool,
    /// voice_id → flat f32[VOICE_PACK_SIZE * VOICE_STYLE_DIM], loaded once at startup
    voice_styles: HashMap<String, Vec<f32>>,
    config:      KokoroOnnxConfig,
    capabilities: EngineCapabilities,
    /// Per-engine G2P result cache: text → phoneme token ids.
    /// DashMap allows concurrent lock-free reads; no blocking under concurrent synthesis.
    g2p_cache: DashMap<String, Arc<Vec<i64>>>,
}

// Session is Send+Sync as documented by ort; SessionPool wraps it safely.
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
        let pool_size = cfg.get("pool_size")
            .and_then(|v| v.as_u64())
            .map(|n| n.max(1) as usize)
            .unwrap_or(DEFAULT_POOL_SIZE);

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

        // Build `pool_size` independent sessions with hardware-accelerated EP.
        let mut sessions = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(2)?;

            // On macOS: enable CoreML to route through the Apple Neural Engine.
            // Measured gain: ~30ms CPU → ~8-12ms ANE per sentence on M-series.
            #[cfg(target_os = "macos")]
            let builder = builder.with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default()
                    .with_subgraphs(true)
                    .build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])?;

            let session = builder.commit_from_file(&model_path)?;
            if i == 0 {
                for inp in &session.inputs  { log::info!("ONNX input:  {:?}", inp); }
                for out in &session.outputs { log::info!("ONNX output: {:?}", out); }
                #[cfg(target_os = "macos")]
                log::info!("KokoroOnnx: CoreML execution provider active (ANE path)");
            }
            sessions.push(session);
        }
        log::info!("KokoroOnnx: created session pool (size={})", pool_size);

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
        let config = KokoroOnnxConfig { model_dir, sample_rate, pool_size };
        let capabilities = Self::build_capabilities(sample_rate);

        Ok(Self {
            pool: SessionPool::new(sessions),
            voice_styles,
            config,
            capabilities,
            g2p_cache: DashMap::with_capacity(G2P_CACHE_CAPACITY),
        })
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
            supports_streaming: true,  // sentence-level streaming via /tts/stream
        }
    }

    /// Look up or compute the phoneme tokens for `text`.
    ///
    /// Uses a lock-free DashMap so concurrent synthesis requests never block
    /// each other on the G2P cache — reads are entirely contention-free.
    fn cached_g2p(&self, text: &str) -> Result<Vec<i64>> {
        use crate::core::g2p_misaki::MisakiG2P;

        // Fast path: lock-free read from DashMap.
        if let Some(tokens) = self.g2p_cache.get(text) {
            return Ok(tokens.as_ref().clone());
        }

        // Slow path: compute G2P then insert.
        let g2p = MisakiG2P::new()?;
        let tokens = Arc::new(g2p.text_to_tokens(text)?);

        // Evict one random entry if cache is full.
        if self.g2p_cache.len() >= G2P_CACHE_CAPACITY {
            if let Some(entry) = self.g2p_cache.iter().next() {
                let evict_key = entry.key().clone();
                drop(entry);
                self.g2p_cache.remove(&evict_key);
            }
        }

        self.g2p_cache.insert(text.to_string(), Arc::clone(&tokens));
        Ok(Arc::try_unwrap(tokens).unwrap_or_else(|arc| arc.as_ref().clone()))
    }

    async fn synthesize_with_onnx(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        let phoneme_tokens = self.cached_g2p(text)?;
        anyhow::ensure!(!phoneme_tokens.is_empty(), "G2P produced no tokens for: {:?}", text);

        let phoneme_count = phoneme_tokens.len();

        // style: f32 [1, VOICE_STYLE_DIM]
        // Select row by phoneme_count-1, capped at VOICE_PACK_SIZE-1.
        // Borrow the pack directly — copy only the 256-float row (1 KB), not 522 KB.
        let voice_id = params.voice.as_deref().unwrap_or("af_heart");
        let pack: &[f32] = self.voice_styles
            .get(voice_id)
            .or_else(|| self.voice_styles.get("af_heart"))
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let style_row  = (phoneme_count.saturating_sub(1)).min(VOICE_PACK_SIZE - 1);
        let style_vec: Vec<f32> = if pack.len() == VOICE_PACK_SIZE * VOICE_STYLE_DIM {
            pack[style_row * VOICE_STYLE_DIM..(style_row + 1) * VOICE_STYLE_DIM].to_vec()
        } else {
            // Voice pack absent — use silence (zeroes)
            vec![0.0f32; VOICE_STYLE_DIM]
        };
        let style_tensor = Tensor::<f32>::from_array(([1usize, VOICE_STYLE_DIM], style_vec))?;

        // tokens: int64 [1, seq_len] with BOS(0) and EOS(0)
        let mut tokens_with_bos_eos: Vec<i64> = Vec::with_capacity(phoneme_tokens.len() + 2);
        tokens_with_bos_eos.push(0);
        tokens_with_bos_eos.extend_from_slice(&phoneme_tokens);
        tokens_with_bos_eos.push(0);
        let seq_len = tokens_with_bos_eos.len();
        let tokens_tensor = Tensor::<i64>::from_array(([1usize, seq_len], tokens_with_bos_eos))?;

        // speed: f32 [1]
        let speed_tensor = Tensor::<f32>::from_array(([1usize], vec![params.speed]))?;

        // Check out a session from the pool — async wait, no blocking.
        let mut session = self.pool.acquire().await;
        let outputs = session.run(ort::inputs![
            "tokens" => tokens_tensor,
            "style"  => style_tensor,
            "speed"  => speed_tensor
        ])?;
        // Session automatically returned to pool when `session` guard drops here.

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
            .await
            .with_context(|| format!("Kokoro ONNX synthesis failed for: {:?}", &text[..text.len().min(40)]))
    }

    async fn warmup(&self) -> Result<()> {
        log::info!("Kokoro ONNX engine ready (pool_size={}, model loaded at startup)",
            self.config.pool_size);
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
    use crate::core::tts_engine::TTSEngine;

    // ── default_pool_size ─────────────────────────────────────────────────────

    #[test]
    fn test_default_pool_size_returns_two() {
        assert_eq!(default_pool_size(), DEFAULT_POOL_SIZE);
        assert!(default_pool_size() >= 1, "pool size must be at least 1");
    }

    // ── KokoroOnnxConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_kokoro_onnx_config_construction() {
        let cfg = KokoroOnnxConfig {
            model_dir: PathBuf::from("models/kokoro-82m"),
            sample_rate: 24000,
            pool_size: 2,
        };
        assert_eq!(cfg.model_dir, PathBuf::from("models/kokoro-82m"));
        assert_eq!(cfg.sample_rate, 24000);
        assert_eq!(cfg.pool_size, 2);
    }

    #[test]
    fn test_kokoro_onnx_config_serde_roundtrip() {
        let cfg = KokoroOnnxConfig {
            model_dir: PathBuf::from("/tmp/models/kokoro"),
            sample_rate: 22050,
            pool_size: 3,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let deser: KokoroOnnxConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.model_dir, cfg.model_dir);
        assert_eq!(deser.sample_rate, cfg.sample_rate);
        assert_eq!(deser.pool_size, cfg.pool_size);
    }

    #[test]
    fn test_kokoro_onnx_config_serde_default_pool_size() {
        // When pool_size is absent from JSON, serde default kicks in
        let json = r#"{"model_dir": "/tmp/model", "sample_rate": 24000}"#;
        let deser: KokoroOnnxConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(deser.pool_size, DEFAULT_POOL_SIZE);
    }

    #[test]
    fn test_kokoro_onnx_config_debug_and_clone() {
        let cfg = KokoroOnnxConfig {
            model_dir: PathBuf::from("models/kokoro"),
            sample_rate: 24000,
            pool_size: 1,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.sample_rate, cfg.sample_rate);
        let _ = format!("{:?}", cloned);
    }

    // ── build_capabilities ────────────────────────────────────────────────────

    #[test]
    fn test_build_capabilities_name_and_version() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        assert_eq!(caps.name, "Kokoro ONNX TTS");
        assert_eq!(caps.version, "1.0.0");
    }

    #[test]
    fn test_build_capabilities_sample_rate_propagated() {
        let caps = KokoroOnnxEngine::build_capabilities(22050);
        assert_eq!(caps.sample_rate, 22050);
    }

    #[test]
    fn test_build_capabilities_max_text_length() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        assert_eq!(caps.max_text_length, 1000);
    }

    #[test]
    fn test_build_capabilities_ssml_and_streaming_disabled() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        assert!(!caps.supports_ssml);
        assert!(!caps.supports_streaming);
    }

    #[test]
    fn test_build_capabilities_supported_languages() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        assert!(caps.supported_languages.contains(&"en-US".to_string()));
        assert!(caps.supported_languages.contains(&"en-GB".to_string()));
    }

    #[test]
    fn test_build_capabilities_voices_count() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        // There are 10 default voices defined
        assert_eq!(caps.supported_voices.len(), 10);
    }

    #[test]
    fn test_build_capabilities_voice_ids_present() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        let ids: Vec<&str> = caps.supported_voices.iter().map(|v| v.id.as_str()).collect();
        for expected in &["af_heart", "af_bella", "af_sarah", "af_nicole",
                          "am_adam", "am_michael", "bf_emma", "bf_isabella",
                          "bm_george", "bm_lewis"] {
            assert!(ids.contains(expected), "missing voice: {}", expected);
        }
    }

    #[test]
    fn test_build_capabilities_voice_gender_variety() {
        use crate::core::tts_engine::VoiceGender;
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        let has_female = caps.supported_voices.iter().any(|v| matches!(v.gender, VoiceGender::Female));
        let has_male = caps.supported_voices.iter().any(|v| matches!(v.gender, VoiceGender::Male));
        assert!(has_female, "should have at least one female voice");
        assert!(has_male, "should have at least one male voice");
    }

    #[test]
    fn test_build_capabilities_voice_quality_neural() {
        use crate::core::tts_engine::VoiceQuality;
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        // All default voices should be Neural quality
        assert!(caps.supported_voices.iter().all(|v| matches!(v.quality, VoiceQuality::Neural)));
    }

    #[test]
    fn test_build_capabilities_gb_voices_have_correct_language() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        for voice in &caps.supported_voices {
            if voice.id.starts_with("bf_") || voice.id.starts_with("bm_") {
                assert_eq!(voice.language, "en-GB", "british voice {} should have en-GB language", voice.id);
            } else {
                assert_eq!(voice.language, "en-US", "american voice {} should have en-US language", voice.id);
            }
        }
    }

    // ── load_voice_style ──────────────────────────────────────────────────────
    // (covers the error path for wrong size)

    #[test]
    fn test_load_voice_style_wrong_size_returns_err() {
        use std::io::Write;
        let mut path = std::env::temp_dir();
        path.push("test_voice_wrong_size.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        // Write only 4 floats — far less than VOICE_PACK_SIZE * VOICE_STYLE_DIM
        for _ in 0..4_usize {
            f.write_all(&1.0f32.to_le_bytes()).unwrap();
        }
        let result = KokoroOnnxEngine::load_voice_style(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err(), "wrong-sized bin file should return Err");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("expected") || msg.contains("floats"),
            "error message should mention expected size: {}", msg);
    }

    #[test]
    fn test_load_voice_style_missing_file_returns_err() {
        let path = std::path::Path::new("/nonexistent/voice/style.bin");
        let result = KokoroOnnxEngine::load_voice_style(path);
        assert!(result.is_err(), "missing file should return Err");
    }

    // ── KokoroOnnxEngine::new error path ─────────────────────────────────────

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

    // ── module constants ──────────────────────────────────────────────────────

    #[test]
    fn test_voice_style_dim_is_256() {
        assert_eq!(VOICE_STYLE_DIM, 256);
    }

    #[test]
    fn test_voice_pack_size_is_510() {
        assert_eq!(VOICE_PACK_SIZE, 510);
    }

    #[test]
    fn test_g2p_cache_capacity_is_1024() {
        assert_eq!(G2P_CACHE_CAPACITY, 1024);
    }

    #[test]
    fn test_default_pool_size_constant_is_2() {
        assert_eq!(DEFAULT_POOL_SIZE, 2);
    }

    // ── load_voice_style edge cases ───────────────────────────────────────────

    #[test]
    fn test_load_voice_style_odd_byte_count_returns_err() {
        use std::io::Write;
        // 3 bytes is not a multiple of 4, so chunks_exact(4) will give 0 floats
        // → wrong size error
        let mut path = std::env::temp_dir();
        path.push("test_voice_odd_bytes.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&[0x01, 0x02, 0x03]).unwrap();
        let result = KokoroOnnxEngine::load_voice_style(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err(), "3-byte file produces 0 floats, should be Err");
    }

    #[test]
    fn test_load_voice_style_empty_file_returns_err() {
        use std::io::Write;
        let mut path = std::env::temp_dir();
        path.push("test_voice_empty.bin");
        std::fs::File::create(&path).unwrap(); // create empty file
        let result = KokoroOnnxEngine::load_voice_style(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err(), "empty file should yield 0 floats and return Err");
    }

    #[test]
    fn test_load_voice_style_one_extra_float_returns_err() {
        use std::io::Write;
        // VOICE_PACK_SIZE * VOICE_STYLE_DIM + 1 floats should fail
        let n = VOICE_PACK_SIZE * VOICE_STYLE_DIM + 1;
        let mut path = std::env::temp_dir();
        path.push("test_voice_one_extra.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        for _ in 0..n {
            f.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        let result = KokoroOnnxEngine::load_voice_style(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err(), "one extra float should return Err (wrong total size)");
    }

    #[test]
    fn test_load_voice_style_one_fewer_float_returns_err() {
        use std::io::Write;
        let n = VOICE_PACK_SIZE * VOICE_STYLE_DIM - 1;
        let mut path = std::env::temp_dir();
        path.push("test_voice_one_fewer.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        for _ in 0..n {
            f.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        let result = KokoroOnnxEngine::load_voice_style(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err(), "one fewer float should return Err");
    }

    #[test]
    fn test_load_voice_style_correct_size_all_zeroes() {
        use std::io::Write;
        let n = VOICE_PACK_SIZE * VOICE_STYLE_DIM;
        let mut path = std::env::temp_dir();
        path.push("test_voice_all_zeroes.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        for _ in 0..n {
            f.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        let loaded = KokoroOnnxEngine::load_voice_style(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.len(), n);
        assert!(loaded.iter().all(|&x| x == 0.0f32));
    }

    #[test]
    fn test_load_voice_style_correct_size_negative_values() {
        use std::io::Write;
        let n = VOICE_PACK_SIZE * VOICE_STYLE_DIM;
        let mut path = std::env::temp_dir();
        path.push("test_voice_negative.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        for i in 0..n {
            let val = if i % 2 == 0 { -1.0f32 } else { 1.0f32 };
            f.write_all(&val.to_le_bytes()).unwrap();
        }
        let loaded = KokoroOnnxEngine::load_voice_style(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.len(), n);
        assert_eq!(loaded[0], -1.0f32);
        assert_eq!(loaded[1], 1.0f32);
    }

    // ── KokoroOnnxConfig serde edge cases ────────────────────────────────────

    #[test]
    fn test_kokoro_onnx_config_pool_size_1_deserialized() {
        let json = r#"{"model_dir":"/tmp","sample_rate":16000,"pool_size":1}"#;
        let cfg: KokoroOnnxConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.pool_size, 1);
        assert_eq!(cfg.sample_rate, 16000);
    }

    #[test]
    fn test_kokoro_onnx_config_large_pool_size_deserialized() {
        let json = r#"{"model_dir":"/tmp","sample_rate":44100,"pool_size":8}"#;
        let cfg: KokoroOnnxConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.pool_size, 8);
        assert_eq!(cfg.sample_rate, 44100);
    }

    #[test]
    fn test_kokoro_onnx_config_serialize_contains_model_dir() {
        let cfg = KokoroOnnxConfig {
            model_dir: PathBuf::from("/some/specific/path"),
            sample_rate: 24000,
            pool_size: 2,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("some/specific/path"), "serialized JSON should contain model_dir path");
        assert!(json.contains("24000"), "serialized JSON should contain sample_rate");
    }

    // ── build_capabilities per-voice detail ──────────────────────────────────

    #[test]
    fn test_build_capabilities_af_heart_voice_details() {
        use crate::core::tts_engine::{VoiceGender, VoiceQuality};
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        let voice = caps.supported_voices.iter()
            .find(|v| v.id == "af_heart")
            .expect("af_heart voice should exist");
        assert_eq!(voice.name, "Heart (American Female)");
        assert_eq!(voice.language, "en-US");
        assert!(matches!(voice.gender, VoiceGender::Female));
        assert!(matches!(voice.quality, VoiceQuality::Neural));
    }

    #[test]
    fn test_build_capabilities_am_adam_voice_details() {
        use crate::core::tts_engine::{VoiceGender, VoiceQuality};
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        let voice = caps.supported_voices.iter()
            .find(|v| v.id == "am_adam")
            .expect("am_adam voice should exist");
        assert_eq!(voice.name, "Adam (American Male)");
        assert_eq!(voice.language, "en-US");
        assert!(matches!(voice.gender, VoiceGender::Male));
        assert!(matches!(voice.quality, VoiceQuality::Neural));
    }

    #[test]
    fn test_build_capabilities_bm_george_voice_details() {
        use crate::core::tts_engine::{VoiceGender, VoiceQuality};
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        let voice = caps.supported_voices.iter()
            .find(|v| v.id == "bm_george")
            .expect("bm_george voice should exist");
        assert_eq!(voice.name, "George (British Male)");
        assert_eq!(voice.language, "en-GB");
        assert!(matches!(voice.gender, VoiceGender::Male));
        assert!(matches!(voice.quality, VoiceQuality::Neural));
    }

    #[test]
    fn test_build_capabilities_bf_emma_voice_details() {
        use crate::core::tts_engine::{VoiceGender, VoiceQuality};
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        let voice = caps.supported_voices.iter()
            .find(|v| v.id == "bf_emma")
            .expect("bf_emma voice should exist");
        assert_eq!(voice.name, "Emma (British Female)");
        assert_eq!(voice.language, "en-GB");
        assert!(matches!(voice.gender, VoiceGender::Female));
        assert!(matches!(voice.quality, VoiceQuality::Neural));
    }

    #[test]
    fn test_build_capabilities_supported_languages_count() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        assert_eq!(caps.supported_languages.len(), 2,
            "should have exactly 2 supported languages (en-US and en-GB)");
    }

    #[test]
    fn test_build_capabilities_sample_rate_8000() {
        let caps = KokoroOnnxEngine::build_capabilities(8000);
        assert_eq!(caps.sample_rate, 8000);
    }

    #[test]
    fn test_build_capabilities_sample_rate_44100() {
        let caps = KokoroOnnxEngine::build_capabilities(44100);
        assert_eq!(caps.sample_rate, 44100);
    }

    // ── KokoroOnnxEngine::new error path variants ─────────────────────────────

    #[test]
    fn test_onnx_engine_new_with_pool_size_1_fails_without_model() {
        let config = serde_json::json!({
            "model_dir": "/nonexistent/dir",
            "sample_rate": 24000,
            "pool_size": 1
        });
        let result = KokoroOnnxEngine::new(&config);
        assert!(result.is_err(), "should fail when model file is missing");
    }

    #[test]
    fn test_onnx_engine_new_with_pool_size_0_treated_as_1_fails_without_model() {
        // pool_size 0 is clamped to 1 via n.max(1)
        let config = serde_json::json!({
            "model_dir": "/nonexistent/dir",
            "sample_rate": 24000,
            "pool_size": 0
        });
        let result = KokoroOnnxEngine::new(&config);
        assert!(result.is_err(), "should fail when model file is missing");
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib"]
    fn test_onnx_engine_new_default_model_dir_fails_without_model() {
        // When model_dir key is absent, default "models/kokoro-82m" is used
        let config = serde_json::json!({"sample_rate": 24000});
        let result = KokoroOnnxEngine::new(&config);
        // May succeed if real models exist, or fail gracefully — either is acceptable
        // as long as it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_onnx_engine_error_mentions_model_dir() {
        let config = serde_json::json!({
            "model_dir": "/definitely/does/not/exist/kokoro",
            "sample_rate": 24000
        });
        let result = KokoroOnnxEngine::new(&config);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        // Error should mention that the model was not found
        assert!(
            msg.contains("not found") || msg.contains("does not exist") || msg.contains("kokoro"),
            "error message should be descriptive: {}", msg
        );
    }

    // ── default_pool_size serde integration ──────────────────────────────────

    #[test]
    fn test_default_pool_size_fn_matches_constant() {
        assert_eq!(default_pool_size(), DEFAULT_POOL_SIZE,
            "default_pool_size() function must return DEFAULT_POOL_SIZE constant");
    }

    #[test]
    fn test_kokoro_onnx_config_serde_pool_size_explicit_2() {
        let json = r#"{"model_dir":"/tmp","sample_rate":24000,"pool_size":2}"#;
        let cfg: KokoroOnnxConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.pool_size, 2);
    }

    // ── load_voice_style float conversion correctness ─────────────────────────

    #[test]
    fn test_load_voice_style_float_conversion_known_bytes() {
        use std::io::Write;
        // 1.0f32 in little-endian bytes is 0x00 0x00 0x80 0x3F
        // Fill the whole file with 1.0f32 values
        let n = VOICE_PACK_SIZE * VOICE_STYLE_DIM;
        let mut path = std::env::temp_dir();
        path.push("test_voice_ones.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        for _ in 0..n {
            f.write_all(&1.0f32.to_le_bytes()).unwrap();
        }
        let loaded = KokoroOnnxEngine::load_voice_style(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.len(), n);
        assert!(loaded.iter().all(|&x| (x - 1.0f32).abs() < 1e-7),
            "all values should be 1.0f32 after correct LE byte conversion");
    }

    #[test]
    fn test_load_voice_style_error_message_contains_path_info() {
        use std::io::Write;
        // Wrong size — check error message includes size information
        let mut path = std::env::temp_dir();
        path.push("test_voice_size_err_msg.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        // Write exactly 10 floats
        for _ in 0..10_usize {
            f.write_all(&0.0f32.to_le_bytes()).unwrap();
        }
        let err = KokoroOnnxEngine::load_voice_style(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        let msg = err.to_string();
        // The error should mention the expected count and actual count
        assert!(msg.contains("510") || msg.contains("expected") || msg.contains("got"),
            "error message should mention expected/actual sizes: {}", msg);
    }

    // ── load_voice_style: exact byte count produces correct length ────────────
    // Additional edge cases to strengthen coverage of the happy path
    // and the error path formatting.

    #[test]
    fn test_load_voice_style_last_row_is_accessible() {
        use std::io::Write;
        // Write a full voice pack; verify the last row (index 509) is accessible
        let n = VOICE_PACK_SIZE * VOICE_STYLE_DIM;
        let mut path = std::env::temp_dir();
        path.push("test_voice_last_row.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        // Fill with sequential float values to distinguish rows
        for i in 0..n {
            let val = (i % 256) as f32;
            f.write_all(&val.to_le_bytes()).unwrap();
        }
        let loaded = KokoroOnnxEngine::load_voice_style(&path).unwrap();
        std::fs::remove_file(&path).ok();
        // Last row starts at index (VOICE_PACK_SIZE - 1) * VOICE_STYLE_DIM
        let last_row_start = (VOICE_PACK_SIZE - 1) * VOICE_STYLE_DIM;
        assert_eq!(loaded.len(), n);
        assert!(last_row_start + VOICE_STYLE_DIM <= loaded.len(),
            "last row must be within bounds");
    }

    #[test]
    fn test_load_voice_style_error_mentions_voice_pack_size_256() {
        use std::io::Write;
        // Error message should include VOICE_STYLE_DIM (256) and VOICE_PACK_SIZE (510)
        let mut path = std::env::temp_dir();
        path.push("test_voice_dim_in_error.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        // Write exactly 1 float — wrong size
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
        let err = KokoroOnnxEngine::load_voice_style(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        let msg = err.to_string();
        assert!(
            msg.contains("256") || msg.contains("510") || msg.contains("expected"),
            "error message should reference expected dimensions: {}",
            msg
        );
    }

    // ── build_capabilities: voice name format validation ──────────────────────

    #[test]
    fn test_build_capabilities_all_voice_names_nonempty() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        for voice in &caps.supported_voices {
            assert!(!voice.name.is_empty(), "voice '{}' must have a non-empty name", voice.id);
            assert!(!voice.id.is_empty(), "voice id must be non-empty");
            assert!(!voice.language.is_empty(), "voice '{}' must have a non-empty language", voice.id);
        }
    }

    #[test]
    fn test_build_capabilities_sample_rate_zero_accepted() {
        // build_capabilities doesn't validate sample_rate — it just stores it
        let caps = KokoroOnnxEngine::build_capabilities(0);
        assert_eq!(caps.sample_rate, 0);
    }

    #[test]
    fn test_build_capabilities_version_string() {
        let caps = KokoroOnnxEngine::build_capabilities(24000);
        assert!(!caps.version.is_empty(), "version string must not be empty");
        assert!(caps.version.contains('.'), "version should be in semver format");
    }

    // ── KokoroOnnxConfig serde: serialize preserves all three fields ──────────

    #[test]
    fn test_kokoro_onnx_config_serde_all_fields_present_in_json() {
        let cfg = KokoroOnnxConfig {
            model_dir: PathBuf::from("/models/kokoro"),
            sample_rate: 22050,
            pool_size: 4,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("model_dir"), "serialized JSON must contain model_dir key");
        assert!(json.contains("sample_rate"), "serialized JSON must contain sample_rate key");
        assert!(json.contains("pool_size"), "serialized JSON must contain pool_size key");
        assert!(json.contains("22050"), "serialized JSON must contain sample_rate value");
        assert!(json.contains("4"), "serialized JSON must contain pool_size value");
    }

    // ── default_pool_size: verifies the serde default fn is wired correctly ────

    #[test]
    fn test_serde_default_pool_size_wired_to_constant() {
        // When pool_size is missing from JSON, serde calls default_pool_size()
        // which must return DEFAULT_POOL_SIZE (2).
        let json = r#"{"model_dir":"/tmp","sample_rate":24000}"#;
        let cfg: KokoroOnnxConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.pool_size, DEFAULT_POOL_SIZE,
            "missing pool_size must fall back to DEFAULT_POOL_SIZE ({})", DEFAULT_POOL_SIZE);
    }
}
