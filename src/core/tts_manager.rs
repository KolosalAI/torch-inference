/// Production-grade TTS Manager
/// Manages multiple TTS engines and provides a unified interface
use anyhow::{Result, Context};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::PathBuf;

use super::tts_engine::{TTSEngine, TTSEngineFactory, SynthesisParams, EngineCapabilities};
use super::audio::AudioData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSManagerConfig {
    pub default_engine: String,
    pub cache_dir: PathBuf,
    pub max_concurrent_requests: usize,
}

impl Default for TTSManagerConfig {
    fn default() -> Self {
        Self {
            default_engine: "kokoro-onnx".to_string(),
            cache_dir: PathBuf::from("./cache/tts"),
            max_concurrent_requests: 10,
        }
    }
}

/// TTS Manager - coordinates multiple TTS engines
pub struct TTSManager {
    config: TTSManagerConfig,
    engines: DashMap<String, Arc<dyn TTSEngine>>,
    request_semaphore: tokio::sync::Semaphore,
}

impl TTSManager {
    pub fn new(config: TTSManagerConfig) -> Self {
        let max_concurrent = config.max_concurrent_requests;
        Self {
            config,
            engines: DashMap::new(),
            request_semaphore: tokio::sync::Semaphore::new(max_concurrent),
        }
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
        self.get_engine(engine_id).map(|e| e.capabilities().clone())
    }
    
    /// Synthesize speech using a specific engine
    pub async fn synthesize(&self, 
        text: &str, 
        engine_id: Option<&str>, 
        params: SynthesisParams
    ) -> Result<AudioData> {
        // Acquire semaphore permit for rate limiting
        let _permit = self.request_semaphore.acquire().await
            .context("Failed to acquire request permit")?;
        
        // Get the engine
        let engine_id = engine_id.unwrap_or(&self.config.default_engine);
        let engine = self.get_engine(engine_id)
            .ok_or_else(|| anyhow::anyhow!("Engine '{}' not found", engine_id))?;
        
        // Validate input
        engine.validate_text(text)?;
        
        // Synthesize
        log::info!("Synthesizing with engine '{}': {} characters", engine_id, text.len());
        let audio = engine.synthesize(text, &params).await
            .context("Synthesis failed")?;
        
        log::info!("Synthesis complete: {:.2}s audio generated", 
            audio.samples.len() as f32 / audio.sample_rate as f32);
        
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
        TTSManagerStats {
            total_engines: self.engines.len(),
            engine_ids: self.list_engines(),
            available_permits: self.request_semaphore.available_permits(),
            max_concurrent: self.config.max_concurrent_requests,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSManagerStats {
    pub total_engines: usize,
    pub engine_ids: Vec<String>,
    pub available_permits: usize,
    pub max_concurrent: usize,
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
}
