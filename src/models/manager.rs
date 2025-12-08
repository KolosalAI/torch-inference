use dashmap::DashMap;
use serde_json::json;
use log::{info, warn, error};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::error::{InferenceError, Result};
use crate::models::registry::{ModelRegistry, ModelMetadata, ModelFormat};
use crate::models::pytorch_loader::{PyTorchModelLoader, LoadedPyTorchModel};

#[derive(Clone)]
pub struct BaseModel {
    pub name: String,
    pub device: String,
    pub is_loaded: bool,
}

impl BaseModel {
    pub fn new(name: String) -> Self {
        Self {
            name,
            device: "cpu".to_string(),
            is_loaded: false,
        }
    }
    
    pub async fn load(&mut self) -> Result<()> {
        info!("Loading model: {}", self.name);
        self.is_loaded = true;
        Ok(())
    }
    
    pub async fn forward(&self, inputs: &serde_json::Value) -> Result<serde_json::Value> {
        if !self.is_loaded {
            return Err(InferenceError::ModelLoadError("Model not loaded".to_string()));
        }
        Ok(inputs.clone())
    }
    
    pub fn model_info(&self) -> serde_json::Value {
        json!({
            "name": self.name,
            "device": self.device,
            "loaded": self.is_loaded
        })
    }
}

/// Enhanced Model Manager with registry support
pub struct ModelManager {
    models: DashMap<String, BaseModel>,
    registry: Arc<ModelRegistry>,
    pytorch_loader: Arc<PyTorchModelLoader>,
    loaded_pytorch_models: Arc<RwLock<DashMap<String, LoadedPyTorchModel>>>,
    config: Config,
}

impl ModelManager {
    pub fn new(config: &Config) -> Self {
        let model_path = PathBuf::from(&config.models.cache_dir);
        let device_type = config.device.device_type.clone();
        
        Self {
            models: DashMap::new(),
            registry: Arc::new(ModelRegistry::new(model_path)),
            pytorch_loader: Arc::new(PyTorchModelLoader::new(Some(device_type))),
            loaded_pytorch_models: Arc::new(RwLock::new(DashMap::new())),
            config: config.clone(),
        }
    }
    
    /// Register a model from file path
    pub async fn register_model_from_path(&self, path: &Path, name: Option<String>) -> Result<String> {
        info!("Registering model from path: {:?}", path);
        let model_id = self.registry.register_from_path(path, name).await
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        info!("Model registered with ID: {}", model_id);
        Ok(model_id)
    }
    
    /// Scan directory and register all models
    pub async fn scan_and_register(&self, dir_path: &Path) -> Result<Vec<String>> {
        info!("Scanning directory for models: {:?}", dir_path);
        self.registry.scan_directory(dir_path).await
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))
    }
    
    /// Load a PyTorch model
    #[cfg(feature = "torch")]
    pub async fn load_pytorch_model(&self, model_id: &str) -> Result<()> {
        info!("Loading PyTorch model: {}", model_id);
        
        // Get metadata from registry
        let metadata = self.registry.get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        // Check format
        if metadata.format != ModelFormat::PyTorch {
            return Err(InferenceError::ModelLoadError(
                format!("Model {} is not a PyTorch model", model_id)
            ));
        }
        
        // Load model
        let loaded_model = self.pytorch_loader.load_model(&metadata.path)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        // Store in cache
        let mut models = self.loaded_pytorch_models.write().await;
        models.insert(model_id.to_string(), loaded_model);
        drop(models);
        
        // Mark as loaded in registry
        self.registry.mark_loaded(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        info!("PyTorch model loaded successfully: {}", model_id);
        Ok(())
    }
    
    #[cfg(not(feature = "torch"))]
    pub async fn load_pytorch_model(&self, model_id: &str) -> Result<()> {
        Err(InferenceError::ModelLoadError(
            "PyTorch support not enabled. Compile with --features torch".to_string()
        ))
    }
    
    /// Run inference on a PyTorch model
    #[cfg(feature = "torch")]
    pub async fn infer_pytorch(&self, model_id: &str, input: &serde_json::Value) -> Result<serde_json::Value> {
        info!("Running inference on model: {}", model_id);
        
        // Get metadata
        let metadata = self.registry.get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        // Get loaded model
        let models = self.loaded_pytorch_models.read().await;
        let model = models.get(model_id)
            .ok_or_else(|| InferenceError::ModelNotFound(
                format!("Model {} not loaded", model_id)
            ))?;
        
        // Run inference with preprocessing/postprocessing
        let result = self.pytorch_loader.infer(model, input, &metadata)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        // Mark as used
        self.registry.mark_used(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        info!("Inference completed for model: {}", model_id);
        Ok(result)
    }
    
    #[cfg(not(feature = "torch"))]
    pub async fn infer_pytorch(&self, _model_id: &str, _input: &serde_json::Value) -> Result<serde_json::Value> {
        Err(InferenceError::ModelLoadError(
            "PyTorch support not enabled. Compile with --features torch".to_string()
        ))
    }
    
    /// Get model metadata from registry
    pub fn get_model_metadata(&self, model_id: &str) -> Result<ModelMetadata> {
        self.registry.get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))
    }
    
    /// List all registered models
    pub fn list_registered_models(&self) -> Vec<ModelMetadata> {
        self.registry.list_models()
    }
    
    /// List models by format
    pub fn list_by_format(&self, format: ModelFormat) -> Vec<ModelMetadata> {
        self.registry.list_by_format(format)
    }
    
    /// Get registry statistics
    pub fn get_registry_stats(&self) -> serde_json::Value {
        let stats = self.registry.get_stats();
        json!({
            "total_models": stats.total_models,
            "loaded_models": stats.loaded_models,
            "format_counts": stats.format_counts,
            "total_size_mb": stats.total_size_bytes as f64 / 1024.0 / 1024.0,
        })
    }
    
    /// Export registry
    pub fn export_registry(&self) -> serde_json::Value {
        self.registry.export_registry()
    }
    
    // Legacy methods for backward compatibility
    
    pub async fn register_model(&self, name: String, model: BaseModel) -> Result<()> {
        info!("Registering legacy model: {}", name);
        self.models.insert(name, model);
        Ok(())
    }
    
    pub async fn load_model(&self, name: &str) -> Result<()> {
        if let Some(mut entry) = self.models.get_mut(name) {
            entry.load().await?;
            Ok(())
        } else {
            Err(InferenceError::ModelNotFound(name.to_string()))
        }
    }
    
    pub fn get_model(&self, name: &str) -> Result<BaseModel> {
        self.models
            .get(name)
            .map(|m| m.clone())
            .ok_or_else(|| InferenceError::ModelNotFound(name.to_string()))
    }
    
    pub fn list_available(&self) -> Vec<String> {
        let mut models: Vec<String> = self.models.iter().map(|m| m.key().clone()).collect();
        let registered: Vec<String> = self.registry.list_models()
            .iter()
            .map(|m| m.id.clone())
            .collect();
        models.extend(registered);
        models
    }
    
    pub async fn initialize_default_models(&self) -> Result<()> {
        info!("Initializing default models");
        
        // Initialize legacy example model
        let example_model = BaseModel::new("example".to_string());
        self.register_model("example".to_string(), example_model).await?;
        
        // Scan default model directories
        let model_dirs = vec![
            PathBuf::from(&self.config.models.cache_dir),
            PathBuf::from("./models"),
            PathBuf::from("./models/audio"),
        ];
        
        for dir in model_dirs {
            if dir.exists() && dir.is_dir() {
                match self.scan_and_register(&dir).await {
                    Ok(models) => {
                        info!("Registered {} models from {:?}", models.len(), dir);
                    }
                    Err(e) => {
                        warn!("Failed to scan directory {:?}: {}", dir, e);
                    }
                }
            }
        }
        
        // Load auto-load models from config
        for model_name in &self.config.models.auto_load {
            // Try legacy model first
            if let Ok(model) = self.get_model(model_name) {
                let mut m = model;
                if let Err(e) = m.load().await {
                    warn!("Failed to load legacy model {}: {}", model_name, e);
                }
            } else {
                // Try PyTorch model from registry
                #[cfg(feature = "torch")]
                {
                    if let Err(e) = self.load_pytorch_model(model_name).await {
                        warn!("Failed to load PyTorch model {}: {}", model_name, e);
                    }
                }
            }
        }
        
        info!("Model initialization complete");
        Ok(())
    }
}
