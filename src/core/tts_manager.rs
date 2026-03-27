/// Production-grade TTS Manager
/// Manages multiple TTS engines and provides a unified interface
use anyhow::{Result, Context};
use dashmap::DashMap;
use lru::LruCache;
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::path::PathBuf;

use super::tts_engine::{TTSEngine, TTSEngineFactory, SynthesisParams, EngineCapabilities};
use super::audio::AudioData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSManagerConfig {
    pub default_engine: String,
    pub cache_dir: PathBuf,
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
    #[serde(default = "default_synthesis_cache_capacity")]
    pub synthesis_cache_capacity: usize,
}

fn default_max_concurrent() -> usize { 10 }
fn default_synthesis_cache_capacity() -> usize { 128 }

impl Default for TTSManagerConfig {
    fn default() -> Self {
        Self {
            default_engine: "kokoro-onnx".to_string(),
            cache_dir: PathBuf::from("./cache/tts"),
            max_concurrent_requests: 10,
            synthesis_cache_capacity: 128,
        }
    }
}

/// TTS Manager - coordinates multiple TTS engines
pub struct TTSManager {
    config: TTSManagerConfig,
    engines: DashMap<String, Arc<dyn TTSEngine>>,
    /// Content-addressed cache: key = hash(text + engine_id + params),
    /// value = Arc<AudioData> so cache hits are O(1) pointer clones.
    synthesis_cache: tokio::sync::Mutex<LruCache<u64, Arc<AudioData>>>,
}

impl TTSManager {
    pub fn new(config: TTSManagerConfig) -> Self {
        let cap = NonZeroUsize::new(config.synthesis_cache_capacity.max(1))
            .expect("cache capacity is non-zero");
        Self {
            synthesis_cache: tokio::sync::Mutex::new(LruCache::new(cap)),
            config,
            engines: DashMap::new(),
        }
    }

    /// FNV-1a 64-bit hash. Unlike `DefaultHasher`, this is stable across Rust
    /// versions and process restarts — safe for use as a persistent cache key.
    fn fnv1a_u64(data: &[u8]) -> u64 {
        const OFFSET_BASIS: u64 = 14695981039346656037;
        const PRIME: u64 = 1099511628211;
        let mut hash = OFFSET_BASIS;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(PRIME);
        }
        hash
    }

    fn synthesis_cache_key(text: &str, engine_id: &str, params: &SynthesisParams) -> u64 {
        // NUL bytes separate fields to prevent collisions like ("ab","c") == ("a","bc")
        let mut buf = Vec::with_capacity(
            text.len() + engine_id.len()
                + params.voice.as_deref().map_or(0, |s| s.len())
                + params.language.as_deref().map_or(0, |s| s.len())
                + 4   // field separators
                + 4   // speed f32 bits (u32)
                + 4,  // pitch f32 bits (u32)
        );
        buf.extend_from_slice(text.as_bytes());
        buf.push(0);
        buf.extend_from_slice(engine_id.as_bytes());
        buf.push(0);
        buf.extend_from_slice(params.voice.as_deref().unwrap_or("").as_bytes());
        buf.push(0);
        buf.extend_from_slice(params.language.as_deref().unwrap_or("").as_bytes());
        buf.push(0);
        buf.extend_from_slice(&params.speed.to_bits().to_le_bytes());
        buf.extend_from_slice(&params.pitch.to_bits().to_le_bytes());
        Self::fnv1a_u64(&buf)
    }
    
    /// Register a TTS engine
    pub fn register_engine(&self, id: String, engine: Arc<dyn TTSEngine>) -> Result<()> {
        if self.engines.contains_key(&id) {
            anyhow::bail!("Engine with id '{}' already registered", id);
        }
        
        self.engines.insert(id.clone(), engine);
        log::info!("Registered TTS engine: {}", id);
        Ok(())
    }
    
    /// Load an engine from configuration
    pub async fn load_engine(&self, id: String, engine_type: &str, config: serde_json::Value) -> Result<()> {
        let engine = TTSEngineFactory::create(engine_type, &config)
            .context(format!("Failed to create engine: {}", engine_type))?;
        
        // Warmup the engine
        engine.warmup().await
            .context("Engine warmup failed")?;
        
        self.register_engine(id, engine)?;
        Ok(())
    }
    
    /// Get an engine by ID
    pub fn get_engine(&self, id: &str) -> Option<Arc<dyn TTSEngine>> {
        self.engines.get(id).map(|e| e.clone())
    }

    /// Call a closure with a borrowed reference to an engine, avoiding an Arc
    /// clone for callers that only need to invoke one synchronous method.
    /// Do **not** use this when the result must be held across an `.await`
    /// boundary — use `get_engine` in that case.
    /// Do **not** call back into this `TTSManager` from inside the closure:
    /// the DashMap shard lock is held for the duration of `f`, and re-entrant
    /// calls on the same shard will deadlock.
    pub fn with_engine<F, R>(&self, id: &str, f: F) -> Option<R>
    where
        F: FnOnce(&dyn TTSEngine) -> R,
    {
        self.engines.get(id).map(|e| f(e.as_ref()))
    }

    /// Get the default engine
    pub fn get_default_engine(&self) -> Option<Arc<dyn TTSEngine>> {
        self.get_engine(&self.config.default_engine)
    }

    /// List all registered engines
    pub fn list_engines(&self) -> Vec<String> {
        self.engines.iter().map(|e| e.key().clone()).collect()
    }

    /// Get engine capabilities
    pub fn get_capabilities(&self, engine_id: &str) -> Option<EngineCapabilities> {
        self.with_engine(engine_id, |e| e.capabilities().clone())
    }
    
    /// Synthesize speech using a specific engine.
    ///
    /// Results are cached by content hash — repeated phrases with the same
    /// engine/voice/speed/pitch are returned from memory, bypassing G2P and
    /// ONNX inference entirely.  Concurrency is governed by each engine's own
    /// internal session pool; the manager adds no extra serialization.
    pub async fn synthesize(&self,
        text: &str,
        engine_id: Option<&str>,
        params: SynthesisParams
    ) -> Result<AudioData> {
        let engine_id = engine_id.unwrap_or(&self.config.default_engine);
        let cache_key = Self::synthesis_cache_key(text, engine_id, &params);

        // Fast path: return cached AudioData (O(1) Arc clone, no inference).
        {
            let mut cache = self.synthesis_cache.lock().await;
            if let Some(cached) = cache.get(&cache_key) {
                log::debug!("TTS cache hit ({} chars, engine '{}')", text.len(), engine_id);
                return Ok((**cached).clone());
            }
        }

        let engine = self.get_engine(engine_id)
            .ok_or_else(|| anyhow::anyhow!("Engine '{}' not found", engine_id))?;

        engine.validate_text(text)?;

        log::info!("Synthesizing with engine '{}': {} characters", engine_id, text.len());
        let audio = engine.synthesize(text, &params).await
            .context("Synthesis failed")?;

        log::info!("Synthesis complete: {:.2}s audio generated",
            audio.samples.len() as f32 / audio.sample_rate as f32);

        // Store result; evicts LRU entry automatically when at capacity.
        {
            let mut cache = self.synthesis_cache.lock().await;
            cache.put(cache_key, Arc::new(audio.clone()));
        }

        Ok(audio)
    }
    
    /// Initialize production TTS engines only
    pub async fn initialize_defaults(&self) -> Result<()> {
        log::info!("Initializing production TTS engines...");
        
        // Load Kokoro ONNX TTS as PRIMARY engine (high-quality neural TTS with parametric fallback)
        log::info!("Loading Kokoro ONNX TTS engine...");
        let kokoro_onnx_config = serde_json::json!({
            "model_dir": "models/kokoro-82m",
            "sample_rate": 24000
        });
        match self.load_engine("kokoro-onnx".to_string(), "kokoro-onnx", kokoro_onnx_config).await {
            Ok(_) => log::info!("[OK] Kokoro ONNX TTS engine loaded successfully"),
            Err(e) => log::warn!("[WARN] Failed to load Kokoro ONNX engine: {}", e),
        }
        
        // Load original Kokoro TTS as SECONDARY engine
        log::info!("Loading Kokoro neural TTS engine...");
        let kokoro_config = serde_json::json!({
            "model_path": "models/kokoro-82m/kokoro-v1_0.pth",
            "sample_rate": 24000
        });
        match self.load_engine("kokoro".to_string(), "kokoro", kokoro_config).await {
            Ok(_) => log::info!("[OK] Kokoro TTS engine loaded successfully"),
            Err(e) => log::warn!("[WARN] Failed to load Kokoro engine: {}", e),
        }
        
        // Load Windows SAPI as secondary engine (REAL SPEECH on Windows)
        #[cfg(target_os = "windows")]
        {
            log::info!("Loading Windows SAPI engine (production-grade speech)...");
            let sapi_config = serde_json::json!({});
            
            match self.load_engine("windows-sapi".to_string(), "windows-sapi", sapi_config).await {
                Ok(_) => log::info!("[OK] Windows SAPI engine loaded successfully"),
                Err(e) => log::warn!("[WARN]  Failed to load Windows SAPI engine: {}", e),
            }
        }
        
        // Load Piper neural TTS only if ONNX feature is enabled
        #[cfg(feature = "onnx")]
        {
            let piper_model_path = PathBuf::from("models/tts/piper_lessac/model.onnx");
            let piper_config_path = PathBuf::from("models/tts/piper_lessac/config.json");
            
            if piper_model_path.exists() {
                log::info!("Loading Piper neural TTS engine...");
                let piper_config = serde_json::json!({
                    "model_path": piper_model_path.to_string_lossy(),
                    "config_path": piper_config_path.to_string_lossy()
                });
                
                match self.load_engine("piper".to_string(), "piper", piper_config).await {
                    Ok(_) => log::info!("[OK] Piper neural TTS engine loaded successfully"),
                    Err(e) => log::warn!("[WARN]  Failed to load Piper engine: {}", e),
                }
            } else {
                log::info!("[WARN]  Piper model not found. Skipping Piper TTS.");
            }
        }
        
        #[cfg(not(feature = "onnx"))]
        {
            log::info!("[WARN]  ONNX feature not enabled. Piper TTS requires ONNX runtime (compile with --features onnx)");
        }
        
        // Load VITS TTS engine
        log::info!("Loading VITS neural TTS engine...");
        let vits_config = serde_json::json!({
            "model_path": "models/vits/model.pth",
            "config_path": "models/vits/config.json",
            "sample_rate": 22050
        });
        match self.load_engine("vits".to_string(), "vits", vits_config).await {
            Ok(_) => log::info!("[OK] VITS TTS engine loaded successfully"),
            Err(e) => log::warn!("[WARN]  Failed to load VITS engine: {}", e),
        }
        
        // Load StyleTTS2 expressive TTS engine
        log::info!("Loading StyleTTS2 expressive TTS engine...");
        let styletts2_config = serde_json::json!({
            "model_path": "models/styletts2/model.pth",
            "config_path": "models/styletts2/config.json",
            "sample_rate": 24000
        });
        match self.load_engine("styletts2".to_string(), "styletts2", styletts2_config).await {
            Ok(_) => log::info!("[OK] StyleTTS2 engine loaded successfully"),
            Err(e) => log::warn!("[WARN]  Failed to load StyleTTS2 engine: {}", e),
        }
        
        // Load Bark generative audio engine
        log::info!("Loading Bark generative audio engine...");
        let bark_config = serde_json::json!({
            "model_path": "models/bark/",
            "sample_rate": 24000,
            "use_small_model": false
        });
        match self.load_engine("bark".to_string(), "bark", bark_config).await {
            Ok(_) => log::info!("[OK] Bark engine loaded successfully"),
            Err(e) => log::warn!("[WARN]  Failed to load Bark engine: {}", e),
        }
        
        // Load XTTS multilingual TTS engine
        log::info!("Loading XTTS multilingual TTS engine...");
        let xtts_config = serde_json::json!({
            "model_path": "models/xtts/model.pth",
            "config_path": "models/xtts/config.json",
            "sample_rate": 24000
        });
        match self.load_engine("xtts".to_string(), "xtts", xtts_config).await {
            Ok(_) => log::info!("[OK] XTTS engine loaded successfully"),
            Err(e) => log::warn!("[WARN]  Failed to load XTTS engine: {}", e),
        }
        
        let engine_count = self.engines.len();
        if engine_count == 0 {
            log::warn!("[WARN]  No TTS engines loaded. Install models to enable TTS functionality.");
        } else {
            log::info!("[OK] {} production TTS engine(s) initialized", engine_count);
        }
        
        Ok(())
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> TTSManagerStats {
        let (cache_size, cache_capacity) = self.synthesis_cache
            .try_lock()
            .map(|c| (c.len(), c.cap().get()))
            .unwrap_or((0, self.config.synthesis_cache_capacity));

        TTSManagerStats {
            total_engines: self.engines.len(),
            engine_ids: self.list_engines(),
            cache_size,
            cache_capacity,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSManagerStats {
    pub total_engines: usize,
    pub engine_ids: Vec<String>,
    pub cache_size: usize,
    pub cache_capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tts_manager_creation() {
        let config = TTSManagerConfig::default();
        let manager = TTSManager::new(config);

        assert_eq!(manager.list_engines().len(), 0);
    }

    #[tokio::test]
    async fn test_load_demo_engine() {
        let config = TTSManagerConfig::default();
        let manager = TTSManager::new(config);

        // Test that loading an unknown engine fails gracefully
        let demo_config = serde_json::json!({ "sample_rate": 24000 });
        let result = manager.load_engine("test".to_string(), "demo", demo_config).await;

        assert!(result.is_err());
        assert_eq!(manager.list_engines().len(), 0);
    }

    #[tokio::test]
    async fn test_synthesis() {
        let config = TTSManagerConfig::default();
        let manager = TTSManager::new(config);

        let params = SynthesisParams::default();
        // Test that synthesis without an engine fails gracefully
        let result = manager.synthesize("Hello, world!", Some("nonexistent"), params).await;

        assert!(result.is_err());
    }

    #[test]
    fn test_synthesis_cache_key_is_stable_across_calls() {
        let params = SynthesisParams {
            speed: 1.0,
            pitch: 1.0,
            voice: Some("af_heart".to_string()),
            language: Some("en-US".to_string()),
        };

        let k1 = TTSManager::synthesis_cache_key("Hello world", "kokoro-onnx", &params);
        let k2 = TTSManager::synthesis_cache_key("Hello world", "kokoro-onnx", &params);
        assert_eq!(k1, k2, "same inputs must produce identical keys");

        assert_eq!(k1, TTSManager::synthesis_cache_key("Hello world", "kokoro-onnx", &params),
            "key must be deterministic");

        // This value is computed by FNV-1a over the canonical buffer.
        // If it ever changes, the synthesis cache is invalidated across deploys —
        // update this constant intentionally, never accidentally.
        assert_eq!(k1, 0xf7923ea9548525f4_u64, "hash must be stable across runs");
    }

    #[test]
    fn test_synthesis_cache_key_distinguishes_inputs() {
        let params = SynthesisParams::default();

        let k_text_a = TTSManager::synthesis_cache_key("Hello", "kokoro-onnx", &params);
        let k_text_b = TTSManager::synthesis_cache_key("World", "kokoro-onnx", &params);
        assert_ne!(k_text_a, k_text_b, "different text must produce different keys");

        let k_engine_a = TTSManager::synthesis_cache_key("Hello", "kokoro-onnx", &params);
        let k_engine_b = TTSManager::synthesis_cache_key("Hello", "piper", &params);
        assert_ne!(k_engine_a, k_engine_b, "different engine must produce different keys");

        let params_fast = SynthesisParams { speed: 2.0, ..SynthesisParams::default() };
        let k_speed_a = TTSManager::synthesis_cache_key("Hello", "kokoro-onnx", &params);
        let k_speed_b = TTSManager::synthesis_cache_key("Hello", "kokoro-onnx", &params_fast);
        assert_ne!(k_speed_a, k_speed_b, "different speed must produce different keys");
    }

    #[test]
    fn test_with_engine_calls_closure_and_returns_result() {
        use std::sync::Arc;
        use crate::core::tts_engine::{EngineCapabilities, VoiceInfo};

        struct MockEngine;

        #[async_trait::async_trait]
        impl crate::core::tts_engine::TTSEngine for MockEngine {
            fn name(&self) -> &str { "mock" }
            fn capabilities(&self) -> &EngineCapabilities {
                // Return a static reference via Box::leak — acceptable in tests
                Box::leak(Box::new(EngineCapabilities {
                    name: "mock".to_string(),
                    version: "0.1".to_string(),
                    supported_languages: vec!["en".to_string()],
                    supported_voices: vec![],
                    max_text_length: 1000,
                    sample_rate: 24000,
                    supports_ssml: false,
                    supports_streaming: false,
                }))
            }
            async fn synthesize(
                &self,
                _text: &str,
                _params: &crate::core::tts_engine::SynthesisParams,
            ) -> anyhow::Result<crate::core::audio::AudioData> {
                anyhow::bail!("not implemented")
            }
            fn list_voices(&self) -> Vec<VoiceInfo> { vec![] }
        }

        let manager = TTSManager::new(TTSManagerConfig::default());
        manager.engines.insert("mock".to_string(), Arc::new(MockEngine));

        let result: Option<String> = manager.with_engine("mock", |e| e.name().to_string());
        assert_eq!(result, Some("mock".to_string()), "closure should fire and return engine name");
    }

    #[test]
    fn test_with_engine_returns_none_for_unknown_engine() {
        let manager = TTSManager::new(TTSManagerConfig::default());
        let result: Option<String> = manager.with_engine("nonexistent", |_e| "found".to_string());
        assert!(result.is_none(), "with_engine must return None for an unknown engine id");
    }

    // ──────────────────────────── shared mock ────────────────────────────────

    fn make_mock_engine() -> Arc<MockEngine> {
        Arc::new(MockEngine)
    }

    struct MockEngine;

    #[async_trait::async_trait]
    impl crate::core::tts_engine::TTSEngine for MockEngine {
        fn name(&self) -> &str { "mock" }
        fn capabilities(&self) -> &crate::core::tts_engine::EngineCapabilities {
            Box::leak(Box::new(crate::core::tts_engine::EngineCapabilities {
                name: "mock".to_string(),
                version: "0.1".to_string(),
                supported_languages: vec!["en".to_string()],
                supported_voices: vec![],
                max_text_length: 1000,
                sample_rate: 24000,
                supports_ssml: false,
                supports_streaming: false,
            }))
        }
        async fn synthesize(
            &self,
            text: &str,
            _params: &crate::core::tts_engine::SynthesisParams,
        ) -> anyhow::Result<crate::core::audio::AudioData> {
            // Return a simple sine-wave-ish audio just to give back a real value
            let _ = text;
            Ok(crate::core::audio::AudioData {
                samples: vec![0.0_f32; 240],
                sample_rate: 24000,
                channels: 1,
            })
        }
        fn list_voices(&self) -> Vec<crate::core::tts_engine::VoiceInfo> { vec![] }
    }

    fn make_manager_with_mock(engine_id: &str) -> TTSManager {
        let manager = TTSManager::new(TTSManagerConfig {
            default_engine: engine_id.to_string(),
            ..TTSManagerConfig::default()
        });
        manager.engines.insert(engine_id.to_string(), make_mock_engine());
        manager
    }

    // ──────────────────────────── register_engine ────────────────────────────

    #[test]
    fn test_register_engine_success() {
        let manager = TTSManager::new(TTSManagerConfig::default());
        let result = manager.register_engine("new-engine".to_string(), make_mock_engine());
        assert!(result.is_ok());
        assert!(manager.get_engine("new-engine").is_some());
    }

    #[test]
    fn test_register_engine_duplicate_returns_err() {
        let manager = TTSManager::new(TTSManagerConfig::default());
        manager.register_engine("dup".to_string(), make_mock_engine()).unwrap();
        let result = manager.register_engine("dup".to_string(), make_mock_engine());
        assert!(result.is_err(), "registering the same id twice should fail");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("already registered"), "error should mention 'already registered': {msg}");
    }

    // ──────────────────────────── list_engines ───────────────────────────────

    #[test]
    fn test_list_engines_empty() {
        let manager = TTSManager::new(TTSManagerConfig::default());
        assert!(manager.list_engines().is_empty());
    }

    #[test]
    fn test_list_engines_returns_registered_names() {
        let manager = TTSManager::new(TTSManagerConfig::default());
        manager.register_engine("engine-a".to_string(), make_mock_engine()).unwrap();
        manager.register_engine("engine-b".to_string(), make_mock_engine()).unwrap();
        let mut names = manager.list_engines();
        names.sort();
        assert_eq!(names, vec!["engine-a", "engine-b"]);
    }

    // ──────────────────────────── get_default_engine ─────────────────────────

    #[test]
    fn test_get_default_engine_none_when_not_registered() {
        let config = TTSManagerConfig {
            default_engine: "not-there".to_string(),
            ..TTSManagerConfig::default()
        };
        let manager = TTSManager::new(config);
        assert!(manager.get_default_engine().is_none());
    }

    #[test]
    fn test_get_default_engine_some_after_registration() {
        let manager = TTSManager::new(TTSManagerConfig {
            default_engine: "the-default".to_string(),
            ..TTSManagerConfig::default()
        });
        manager.register_engine("the-default".to_string(), make_mock_engine()).unwrap();
        let engine = manager.get_default_engine();
        assert!(engine.is_some());
        assert_eq!(engine.unwrap().name(), "mock");
    }

    // ──────────────────────────── set_default round-trip ─────────────────────

    #[test]
    fn test_default_engine_round_trip_via_config() {
        // TTSManager config sets the default; verify get_default_engine respects it
        let manager = TTSManager::new(TTSManagerConfig {
            default_engine: "alpha".to_string(),
            ..TTSManagerConfig::default()
        });
        manager.register_engine("alpha".to_string(), make_mock_engine()).unwrap();
        manager.register_engine("beta".to_string(), make_mock_engine()).unwrap();

        // default should be alpha
        assert_eq!(manager.get_default_engine().map(|e| e.name().to_string()),
                   Some("mock".to_string()));

        // alpha engine can be retrieved directly
        assert!(manager.get_engine("alpha").is_some());
    }

    // ──────────────────────────── get_capabilities ───────────────────────────

    #[test]
    fn test_get_capabilities_returns_some_for_known_engine() {
        let manager = make_manager_with_mock("cap-test");
        let caps = manager.get_capabilities("cap-test");
        assert!(caps.is_some());
        let caps = caps.unwrap();
        assert_eq!(caps.sample_rate, 24000);
        assert_eq!(caps.max_text_length, 1000);
    }

    #[test]
    fn test_get_capabilities_returns_none_for_unknown_engine() {
        let manager = TTSManager::new(TTSManagerConfig::default());
        assert!(manager.get_capabilities("ghost").is_none());
    }

    // ──────────────────────────── synthesize ─────────────────────────────────

    #[tokio::test]
    async fn test_synthesize_engine_not_found_returns_err() {
        let manager = TTSManager::new(TTSManagerConfig::default());
        let result = manager
            .synthesize("Hello", Some("missing-engine"), SynthesisParams::default())
            .await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("not found"), "error should mention 'not found': {msg}");
    }

    #[tokio::test]
    async fn test_synthesize_uses_default_engine_when_none_given() {
        let manager = make_manager_with_mock("mock");
        let result = manager
            .synthesize("Hello world", None, SynthesisParams::default())
            .await;
        assert!(result.is_ok(), "synthesis with default engine should succeed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_synthesize_returns_audio() {
        let manager = make_manager_with_mock("mock");
        let audio = manager
            .synthesize("Test", Some("mock"), SynthesisParams::default())
            .await
            .expect("synthesis should succeed");
        assert_eq!(audio.sample_rate, 24000);
        assert_eq!(audio.channels, 1);
        assert!(!audio.samples.is_empty());
    }

    #[tokio::test]
    async fn test_synthesize_cache_hit_on_second_call() {
        let manager = make_manager_with_mock("mock");
        let params = SynthesisParams {
            speed: 1.0,
            pitch: 1.0,
            voice: Some("test-voice".to_string()),
            language: None,
        };

        let audio1 = manager
            .synthesize("Cache me", Some("mock"), params.clone())
            .await
            .unwrap();
        let audio2 = manager
            .synthesize("Cache me", Some("mock"), params)
            .await
            .unwrap();

        // Both calls should return identical data
        assert_eq!(audio1.samples, audio2.samples);
        assert_eq!(audio1.sample_rate, audio2.sample_rate);
    }

    #[tokio::test]
    async fn test_synthesize_empty_text_returns_err() {
        let manager = make_manager_with_mock("mock");
        let result = manager
            .synthesize("", Some("mock"), SynthesisParams::default())
            .await;
        assert!(result.is_err(), "empty text should fail validation");
    }

    // ──────────────────────────── TTSManagerStats ────────────────────────────

    #[test]
    fn test_tts_manager_stats_fields() {
        let stats = TTSManagerStats {
            total_engines: 3,
            engine_ids: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            cache_size: 10,
            cache_capacity: 128,
        };
        assert_eq!(stats.total_engines, 3);
        assert_eq!(stats.engine_ids.len(), 3);
        assert_eq!(stats.cache_size, 10);
        assert_eq!(stats.cache_capacity, 128);
    }

    #[test]
    fn test_get_stats_empty_manager() {
        let manager = TTSManager::new(TTSManagerConfig::default());
        let stats = manager.get_stats();
        assert_eq!(stats.total_engines, 0);
        assert!(stats.engine_ids.is_empty());
        assert_eq!(stats.cache_size, 0);
        assert_eq!(stats.cache_capacity, 128);
    }

    #[test]
    fn test_get_stats_after_registration() {
        let manager = TTSManager::new(TTSManagerConfig::default());
        manager.register_engine("e1".to_string(), make_mock_engine()).unwrap();
        manager.register_engine("e2".to_string(), make_mock_engine()).unwrap();
        let stats = manager.get_stats();
        assert_eq!(stats.total_engines, 2);
        assert_eq!(stats.engine_ids.len(), 2);
        assert_eq!(stats.cache_capacity, 128);
    }

    // ──────────────────────────── TTSManagerConfig ───────────────────────────

    #[test]
    fn test_tts_manager_config_default_values() {
        let cfg = TTSManagerConfig::default();
        assert_eq!(cfg.default_engine, "kokoro-onnx");
        assert_eq!(cfg.max_concurrent_requests, 10);
        assert_eq!(cfg.synthesis_cache_capacity, 128);
    }

    #[test]
    fn test_cache_key_voice_none_vs_some_differ() {
        let p_none = SynthesisParams { voice: None, ..SynthesisParams::default() };
        let p_some = SynthesisParams { voice: Some("v".to_string()), ..SynthesisParams::default() };
        let k1 = TTSManager::synthesis_cache_key("hi", "eng", &p_none);
        let k2 = TTSManager::synthesis_cache_key("hi", "eng", &p_some);
        assert_ne!(k1, k2, "voice=None vs voice=Some must produce different keys");
    }

    #[test]
    fn test_cache_key_language_none_vs_some_differ() {
        let p_none = SynthesisParams { language: None, ..SynthesisParams::default() };
        let p_some = SynthesisParams { language: Some("fr".to_string()), ..SynthesisParams::default() };
        let k1 = TTSManager::synthesis_cache_key("hi", "eng", &p_none);
        let k2 = TTSManager::synthesis_cache_key("hi", "eng", &p_some);
        assert_ne!(k1, k2, "language=None vs language=Some must produce different keys");
    }

    #[test]
    fn test_cache_key_pitch_affects_key() {
        let p1 = SynthesisParams { pitch: 1.0, ..SynthesisParams::default() };
        let p2 = SynthesisParams { pitch: 0.5, ..SynthesisParams::default() };
        let k1 = TTSManager::synthesis_cache_key("hi", "eng", &p1);
        let k2 = TTSManager::synthesis_cache_key("hi", "eng", &p2);
        assert_ne!(k1, k2, "different pitch must produce different keys");
    }

    // ──────────────────────────── TTSManagerConfig deserialization ───────────

    #[test]
    fn test_tts_manager_config_serde_uses_defaults() {
        // Deserialize from a minimal JSON object — serde defaults should fill in
        let json = r#"{"default_engine":"custom","cache_dir":"/tmp/tts"}"#;
        let cfg: TTSManagerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.default_engine, "custom");
        assert_eq!(cfg.max_concurrent_requests, 10);
        assert_eq!(cfg.synthesis_cache_capacity, 128);
    }

    #[test]
    fn test_tts_manager_config_serde_custom_values() {
        let json = r#"{"default_engine":"piper","cache_dir":"/tmp","max_concurrent_requests":5,"synthesis_cache_capacity":64}"#;
        let cfg: TTSManagerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.default_engine, "piper");
        assert_eq!(cfg.max_concurrent_requests, 5);
        assert_eq!(cfg.synthesis_cache_capacity, 64);
    }

    #[test]
    fn test_tts_manager_config_debug_and_clone() {
        let cfg = TTSManagerConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.default_engine, cfg.default_engine);
        // Should not panic
        let _ = format!("{:?}", cloned);
    }

    // ──────────────────────────── TTSManager custom capacity ─────────────────

    #[test]
    fn test_tts_manager_capacity_1_clamped() {
        // Capacity of 1 should not panic (NonZeroUsize)
        let cfg = TTSManagerConfig {
            synthesis_cache_capacity: 1,
            ..TTSManagerConfig::default()
        };
        let manager = TTSManager::new(cfg);
        let stats = manager.get_stats();
        assert_eq!(stats.cache_capacity, 1);
    }

    #[test]
    fn test_tts_manager_capacity_0_clamped_to_1() {
        // Capacity 0 is clamped to 1 via .max(1)
        let cfg = TTSManagerConfig {
            synthesis_cache_capacity: 0,
            ..TTSManagerConfig::default()
        };
        let manager = TTSManager::new(cfg);
        let stats = manager.get_stats();
        assert_eq!(stats.cache_capacity, 1);
    }

    // ──────────────────────────── cache eviction at capacity ─────────────────

    #[tokio::test]
    async fn test_synthesize_cache_evicts_lru_at_capacity() {
        // Capacity 1 means the second unique synthesis evicts the first
        let cfg = TTSManagerConfig {
            default_engine: "mock".to_string(),
            synthesis_cache_capacity: 1,
            ..TTSManagerConfig::default()
        };
        let manager = TTSManager::new(cfg);
        manager.engines.insert("mock".to_string(), make_mock_engine());

        let params = SynthesisParams::default();
        // First synthesis — fills the single cache slot
        manager.synthesize("first text", Some("mock"), params.clone()).await.unwrap();
        // Second synthesis — evicts "first text" and fills with "second text"
        manager.synthesize("second text", Some("mock"), params.clone()).await.unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.cache_size, 1, "LRU eviction should keep only 1 entry");
    }

    #[tokio::test]
    async fn test_synthesize_cache_grows_up_to_capacity() {
        let cfg = TTSManagerConfig {
            default_engine: "mock".to_string(),
            synthesis_cache_capacity: 4,
            ..TTSManagerConfig::default()
        };
        let manager = TTSManager::new(cfg);
        manager.engines.insert("mock".to_string(), make_mock_engine());

        let params = SynthesisParams::default();
        for i in 0..4 {
            let text = format!("unique text number {}", i);
            manager.synthesize(&text, Some("mock"), params.clone()).await.unwrap();
        }

        let stats = manager.get_stats();
        assert_eq!(stats.cache_size, 4, "cache should hold exactly 4 unique entries");
    }

    // ──────────────────────────── get_stats with cache populated ─────────────

    #[tokio::test]
    async fn test_get_stats_reflects_populated_cache() {
        let manager = make_manager_with_mock("mock");
        let params = SynthesisParams::default();

        manager.synthesize("hello", Some("mock"), params.clone()).await.unwrap();
        manager.synthesize("world", Some("mock"), params.clone()).await.unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.cache_size, 2);
        assert_eq!(stats.total_engines, 1);
    }

    // ──────────────────────────── text length validation ─────────────────────

    #[tokio::test]
    async fn test_synthesize_text_too_long_returns_err() {
        // MockEngine declares max_text_length = 1000
        let manager = make_manager_with_mock("mock");
        let long_text = "a".repeat(1001);
        let result = manager
            .synthesize(&long_text, Some("mock"), SynthesisParams::default())
            .await;
        assert!(result.is_err(), "text exceeding max_text_length should fail");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("1001") || msg.contains("too long") || msg.contains("1000"),
            "error should mention text length: {msg}");
    }

    // ──────────────────────────── list_voices via engine ─────────────────────

    #[test]
    fn test_with_engine_list_voices_returns_empty_for_mock() {
        let manager = make_manager_with_mock("mock");
        let voices = manager.with_engine("mock", |e| e.list_voices());
        assert!(voices.is_some());
        assert!(voices.unwrap().is_empty(), "mock engine has no voices");
    }

    #[test]
    fn test_with_engine_is_ready_returns_true_for_mock() {
        let manager = make_manager_with_mock("mock");
        let ready = manager.with_engine("mock", |e| e.is_ready());
        assert_eq!(ready, Some(true));
    }

    // ──────────────────────────── fnv1a hash edge cases ──────────────────────

    #[test]
    fn test_synthesis_cache_key_empty_text_and_engine() {
        let params = SynthesisParams::default();
        // Should not panic, and be deterministic
        let k1 = TTSManager::synthesis_cache_key("", "", &params);
        let k2 = TTSManager::synthesis_cache_key("", "", &params);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_synthesis_cache_key_empty_vs_nonempty_text() {
        let params = SynthesisParams::default();
        let k_empty = TTSManager::synthesis_cache_key("", "eng", &params);
        let k_nonempty = TTSManager::synthesis_cache_key("a", "eng", &params);
        assert_ne!(k_empty, k_nonempty);
    }

    #[test]
    fn test_synthesis_cache_key_separator_prevents_concatenation_collision() {
        // ("ab", "c") must differ from ("a", "bc") — the NUL separator prevents merging
        let params = SynthesisParams::default();
        let k1 = TTSManager::synthesis_cache_key("ab", "c", &params);
        let k2 = TTSManager::synthesis_cache_key("a", "bc", &params);
        assert_ne!(k1, k2, "NUL separator must prevent field-concatenation collisions");
    }

    // ──────────────────────────── TTSManagerStats serde ──────────────────────

    #[test]
    fn test_tts_manager_stats_serde_roundtrip() {
        let stats = TTSManagerStats {
            total_engines: 2,
            engine_ids: vec!["kokoro-onnx".to_string(), "piper".to_string()],
            cache_size: 5,
            cache_capacity: 64,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deser: TTSManagerStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.total_engines, 2);
        assert_eq!(deser.engine_ids, stats.engine_ids);
        assert_eq!(deser.cache_size, 5);
        assert_eq!(deser.cache_capacity, 64);
    }

    #[test]
    fn test_tts_manager_stats_debug_and_clone() {
        let stats = TTSManagerStats {
            total_engines: 1,
            engine_ids: vec!["mock".to_string()],
            cache_size: 0,
            cache_capacity: 128,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_engines, stats.total_engines);
        let _ = format!("{:?}", cloned);
    }

    // ──────────────────────────── synthesize: same params different engine ───

    #[tokio::test]
    async fn test_synthesize_different_engines_have_independent_caches() {
        let manager = TTSManager::new(TTSManagerConfig {
            default_engine: "mock1".to_string(),
            ..TTSManagerConfig::default()
        });
        manager.engines.insert("mock1".to_string(), make_mock_engine());
        manager.engines.insert("mock2".to_string(), make_mock_engine());

        let params = SynthesisParams::default();
        let a1 = manager.synthesize("hello", Some("mock1"), params.clone()).await.unwrap();
        let a2 = manager.synthesize("hello", Some("mock2"), params.clone()).await.unwrap();

        // Both return audio (same mock output), but they are cached separately
        let stats = manager.get_stats();
        assert_eq!(stats.cache_size, 2, "same text on two engines = 2 distinct cache entries");
        assert_eq!(a1.sample_rate, a2.sample_rate);
    }

    // ──────────────────────────── synthesize: None engine falls back ─────────

    #[tokio::test]
    async fn test_synthesize_none_engine_uses_config_default() {
        let manager = TTSManager::new(TTSManagerConfig {
            default_engine: "mock".to_string(),
            ..TTSManagerConfig::default()
        });
        manager.engines.insert("mock".to_string(), make_mock_engine());

        // Pass None as engine_id — should use "mock" from config
        let result = manager
            .synthesize("hello default", None, SynthesisParams::default())
            .await;
        assert!(result.is_ok(), "None engine should use default: {:?}", result.err());
    }

    // ──────────────────────────── get_capabilities ──────────────────────────

    #[test]
    fn test_get_capabilities_sample_rate_and_max_length() {
        let manager = make_manager_with_mock("cap-engine");
        let caps = manager.get_capabilities("cap-engine").unwrap();
        assert_eq!(caps.sample_rate, 24000);
        assert_eq!(caps.max_text_length, 1000);
        assert!(!caps.supports_ssml);
        assert!(!caps.supports_streaming);
    }
}
