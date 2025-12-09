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
            default_engine: "windows-sapi".to_string(), // Use Windows SAPI for REAL speech
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
        
        // Load Windows SAPI as primary engine (REAL SPEECH on Windows)
        #[cfg(target_os = "windows")]
        {
            log::info!("Loading Windows SAPI engine (production-grade speech)...");
            let sapi_config = serde_json::json!({});
            
            match self.load_engine("windows-sapi".to_string(), "windows-sapi", sapi_config).await {
                Ok(_) => log::info!("✅ Windows SAPI engine loaded successfully"),
                Err(e) => log::warn!("⚠️  Failed to load Windows SAPI engine: {}", e),
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
                    Ok(_) => log::info!("✅ Piper neural TTS engine loaded successfully"),
                    Err(e) => log::warn!("⚠️  Failed to load Piper engine: {}", e),
                }
            } else {
                log::info!("⚠️  Piper model not found. Skipping Piper TTS.");
            }
        }
        
        #[cfg(not(feature = "onnx"))]
        {
            log::info!("⚠️  ONNX feature not enabled. Piper TTS requires ONNX runtime (compile with --features onnx)");
        }
        
        let engine_count = self.engines.len();
        if engine_count == 0 {
            log::warn!("⚠️  No TTS engines loaded. Install models to enable TTS functionality.");
        } else {
            log::info!("✅ {} production TTS engine(s) initialized", engine_count);
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
        
        let demo_config = serde_json::json!({ "sample_rate": 24000 });
        manager.load_engine("test".to_string(), "demo", demo_config).await.unwrap();
        
        assert_eq!(manager.list_engines().len(), 1);
        assert!(manager.get_engine("test").is_some());
    }
    
    #[tokio::test]
    async fn test_synthesis() {
        let config = TTSManagerConfig::default();
        let manager = TTSManager::new(config);
        
        manager.initialize_defaults().await.unwrap();
        
        let params = SynthesisParams::default();
        let audio = manager.synthesize("Hello, world!", Some("demo"), params).await.unwrap();
        
        assert!(audio.samples.len() > 0);
        assert_eq!(audio.sample_rate, 24000);
    }
}
