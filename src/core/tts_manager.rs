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
}
