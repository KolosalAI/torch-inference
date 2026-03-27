use dashmap::DashMap;
use serde_json::json;
use log::{info, warn, error};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use rand;

use crate::config::Config;
use crate::error::{InferenceError, Result};
use crate::models::registry::{ModelRegistry, ModelMetadata, ModelFormat};
use crate::models::pytorch_loader::{PyTorchModelLoader, LoadedPyTorchModel};
use crate::models::onnx_loader::{OnnxModelLoader, LoadedOnnxModel};
use crate::tensor_pool::TensorPool;

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
    loaded_pytorch_models: Arc<DashMap<String, Vec<LoadedPyTorchModel>>>,
    onnx_loader: Arc<OnnxModelLoader>,
    loaded_onnx_models: Arc<DashMap<String, Vec<LoadedOnnxModel>>>,
    config: Config,
    tensor_pool: Option<Arc<TensorPool>>,
}

impl ModelManager {
    pub fn new(config: &Config, tensor_pool: Option<Arc<TensorPool>>) -> Self {
        let model_path = PathBuf::from(&config.models.cache_dir);
        let device_type = config.device.device_type.clone();
        
        Self {
            models: DashMap::new(),
            registry: Arc::new(ModelRegistry::new(model_path)),
            pytorch_loader: Arc::new(PyTorchModelLoader::new(
                Some(device_type.clone()), 
                Some(config.device.clone()),
                tensor_pool.clone()
            )),
            loaded_pytorch_models: Arc::new(DashMap::new()),
            onnx_loader: Arc::new(OnnxModelLoader::new(
                config.device.use_tensorrt, 
                config.device.device_id,
                tensor_pool.clone()
            )),
            loaded_onnx_models: Arc::new(DashMap::new()),
            config: config.clone(),
            tensor_pool,
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
        // Determine devices to load on
        let devices = if let Some(ids) = &self.config.device.device_ids {
            ids.clone()
        } else {
            vec![self.config.device.device_id]
        };
        
        let mut loaded_models = Vec::new();
        
        for device_id in devices {
            let device_str = if self.config.device.device_type == "cuda" || self.config.device.device_type == "auto" {
                format!("cuda:{}", device_id)
            } else {
                self.config.device.device_type.clone()
            };
            
            info!("Loading PyTorch model {} on device {}", model_id, device_str);
            let loaded_model = self.pytorch_loader.load_model(&metadata.path, Some(device_str))
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            loaded_models.push(loaded_model);
        }
        
        // Store in cache
        self.loaded_pytorch_models.insert(model_id.to_string(), loaded_models);
        
        // Mark as loaded in registry
        self.registry.mark_loaded(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        info!("PyTorch model loaded successfully: {}", model_id);
        Ok(())
    }

    /// Load an ONNX model
    pub async fn load_onnx_model(&self, model_id: &str) -> Result<()> {
        info!("Loading ONNX model: {}", model_id);
        
        // Get metadata from registry
        let metadata = self.registry.get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        // Check format
        if metadata.format != ModelFormat::ONNX {
            return Err(InferenceError::ModelLoadError(
                format!("Model {} is not an ONNX model", model_id)
            ));
        }
        
        // Determine devices to load on
        let devices = if let Some(ids) = &self.config.device.device_ids {
            ids.clone()
        } else {
            vec![self.config.device.device_id]
        };
        
        let mut loaded_models = Vec::new();
        
        for device_id in devices {
            info!("Loading model {} on device {}", model_id, device_id);
            let loaded_model = self.onnx_loader.load_model(&metadata.path, Some(device_id))
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            loaded_models.push(loaded_model);
        }
        
        // Store in cache
        self.loaded_onnx_models.insert(model_id.to_string(), loaded_models);
        
        // Mark as loaded in registry
        self.registry.mark_loaded(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        info!("ONNX model loaded successfully: {}", model_id);
        Ok(())
    }

    /// Load a PyTorch model (fallback when torch feature is disabled)
    #[cfg(not(feature = "torch"))]
    pub async fn load_pytorch_model(&self, _model_id: &str) -> Result<()> {
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
        let models = self.loaded_pytorch_models.get(model_id)
            .ok_or_else(|| InferenceError::ModelNotFound(
                format!("Model {} not loaded", model_id)
            ))?;
            
        if models.is_empty() {
             return Err(InferenceError::ModelLoadError("No model instances loaded".to_string()));
        }
        
        // Select a replica
        let idx = rand::random::<usize>() % models.len();
        let model = &models[idx];
        
        // Run inference with preprocessing/postprocessing
        let result = self.pytorch_loader.infer(model, input, &metadata)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        // Mark as used
        self.registry.mark_used(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        info!("Inference completed for model: {}", model_id);
        Ok(result)
    }

    /// Run inference on an ONNX model
    pub async fn infer_onnx(&self, model_id: &str, input: &serde_json::Value) -> Result<serde_json::Value> {
        info!("Running inference on ONNX model: {}", model_id);
        
        // Get metadata
        let metadata = self.registry.get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        
        // Get loaded models
        let models = self.loaded_onnx_models.get(model_id)
            .ok_or_else(|| InferenceError::ModelNotFound(
                format!("Model {} not loaded", model_id)
            ))?;
            
        if models.is_empty() {
             return Err(InferenceError::ModelLoadError("No model instances loaded".to_string()));
        }
        
        // Select a replica (simple random load balancing)
        let idx = rand::random::<usize>() % models.len();
        let model = &models[idx];
        
        // Run inference
        let result = self.onnx_loader.infer(model, input, &metadata)
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
    
    /// Run inference on any registered model
    pub async fn infer_registered(&self, model_id: &str, input: &serde_json::Value) -> Result<serde_json::Value> {
        let metadata = self.registry.get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            
        match metadata.format {
            ModelFormat::PyTorch => {
                #[cfg(feature = "torch")]
                return self.infer_pytorch(model_id, input).await;
                #[cfg(not(feature = "torch"))]
                return Err(InferenceError::ModelLoadError("PyTorch support disabled".to_string()));
            }
            ModelFormat::ONNX => {
                return self.infer_onnx(model_id, input).await;
            }
            _ => {
                return Err(InferenceError::ModelLoadError(format!("Unsupported format: {:?}", metadata.format)));
            }
        }
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
                    if let Ok(metadata) = self.registry.get_model(model_name) {
                        if metadata.format == ModelFormat::PyTorch {
                            if let Err(e) = self.load_pytorch_model(model_name).await {
                                warn!("Failed to load PyTorch model {}: {}", model_name, e);
                            }
                        }
                    }
                }

                // Try ONNX model from registry
                {
                    if let Ok(metadata) = self.registry.get_model(model_name) {
                        if metadata.format == ModelFormat::ONNX {
                            if let Err(e) = self.load_onnx_model(model_name).await {
                                warn!("Failed to load ONNX model {}: {}", model_name, e);
                            }
                        }
                    }
                }
            }
        }
        
        info!("Model initialization complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn default_manager() -> ModelManager {
        ModelManager::new(&Config::default(), None)
    }

    // ── BaseModel ─────────────────────────────────────────────────────────────

    #[test]
    fn test_base_model_new() {
        let model = BaseModel::new("my-model".to_string());
        assert_eq!(model.name, "my-model");
        assert_eq!(model.device, "cpu");
        assert!(!model.is_loaded);
    }

    #[tokio::test]
    async fn test_base_model_load() {
        let mut model = BaseModel::new("m".to_string());
        assert!(!model.is_loaded);
        model.load().await.unwrap();
        assert!(model.is_loaded);
    }

    #[tokio::test]
    async fn test_base_model_forward_not_loaded() {
        let model = BaseModel::new("m".to_string());
        let result = model.forward(&serde_json::json!({"x": 1})).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("not loaded"));
    }

    #[tokio::test]
    async fn test_base_model_forward_loaded() {
        let mut model = BaseModel::new("m".to_string());
        model.load().await.unwrap();
        let input = serde_json::json!({"data": [1, 2, 3]});
        let output = model.forward(&input).await.unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_base_model_model_info() {
        let model = BaseModel::new("info-test".to_string());
        let info = model.model_info();
        assert_eq!(info["name"], "info-test");
        assert_eq!(info["device"], "cpu");
        assert_eq!(info["loaded"], false);
    }

    #[test]
    fn test_base_model_clone() {
        let model = BaseModel::new("clone-test".to_string());
        let cloned = model.clone();
        assert_eq!(cloned.name, model.name);
        assert_eq!(cloned.is_loaded, model.is_loaded);
    }

    // ── ModelManager construction ─────────────────────────────────────────────

    #[test]
    fn test_manager_new() {
        let manager = default_manager();
        // Freshly created manager has no models
        assert!(manager.list_available().is_empty());
        assert!(manager.list_registered_models().is_empty());
    }

    // ── list_available / list_registered_models ───────────────────────────────

    #[tokio::test]
    async fn test_list_available_includes_legacy_and_registered() {
        let manager = default_manager();

        // Register a legacy BaseModel
        let base = BaseModel::new("legacy-a".to_string());
        manager
            .register_model("legacy-a".to_string(), base)
            .await
            .unwrap();

        let available = manager.list_available();
        assert!(available.contains(&"legacy-a".to_string()));
    }

    // ── get_model – legacy path ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_model_not_found_legacy() {
        let manager = default_manager();
        let result = manager.get_model("ghost");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_model_found_legacy() {
        let manager = default_manager();
        manager
            .register_model("alpha".to_string(), BaseModel::new("alpha".to_string()))
            .await
            .unwrap();
        let model = manager.get_model("alpha").unwrap();
        assert_eq!(model.name, "alpha");
    }

    // ── register_model (legacy) ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_register_model_legacy() {
        let manager = default_manager();
        let base = BaseModel::new("beta".to_string());
        manager.register_model("beta".to_string(), base).await.unwrap();
        assert!(manager.get_model("beta").is_ok());
    }

    // ── load_model (legacy) ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_load_model_legacy_not_found() {
        let manager = default_manager();
        let result = manager.load_model("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_load_model_legacy_success() {
        let manager = default_manager();
        manager
            .register_model("gamma".to_string(), BaseModel::new("gamma".to_string()))
            .await
            .unwrap();
        manager.load_model("gamma").await.unwrap();
        let model = manager.get_model("gamma").unwrap();
        assert!(model.is_loaded);
    }

    // ── get_model_metadata – registry path ───────────────────────────────────

    #[test]
    fn test_get_model_metadata_not_found() {
        let manager = default_manager();
        let result = manager.get_model_metadata("missing");
        assert!(result.is_err());
    }

    // ── list_by_format ────────────────────────────────────────────────────────

    #[test]
    fn test_list_by_format_empty() {
        let manager = default_manager();
        let results = manager.list_by_format(ModelFormat::ONNX);
        assert!(results.is_empty());
    }

    // ── get_registry_stats ────────────────────────────────────────────────────

    #[test]
    fn test_get_registry_stats_empty() {
        let manager = default_manager();
        let stats = manager.get_registry_stats();
        assert_eq!(stats["total_models"], 0);
        assert_eq!(stats["loaded_models"], 0);
    }

    // ── export_registry ───────────────────────────────────────────────────────

    #[test]
    fn test_export_registry_empty() {
        let manager = default_manager();
        let exported = manager.export_registry();
        assert_eq!(exported["total"], 0);
    }

    // ── register_model_from_path – error path ────────────────────────────────

    #[tokio::test]
    async fn test_register_model_from_path_nonexistent() {
        let manager = default_manager();
        let result = manager
            .register_model_from_path(Path::new("/no/such/model.onnx"), None)
            .await;
        assert!(result.is_err());
    }

    // ── scan_and_register – error path ───────────────────────────────────────

    #[tokio::test]
    async fn test_scan_and_register_nonexistent_dir() {
        let manager = default_manager();
        let result = manager
            .scan_and_register(Path::new("/no/such/directory"))
            .await;
        assert!(result.is_err());
    }

    // ── scan_and_register – empty directory ──────────────────────────────────

    #[tokio::test]
    async fn test_scan_and_register_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let manager = default_manager();
        let registered = manager.scan_and_register(dir.path()).await.unwrap();
        assert!(registered.is_empty());
    }

    // ── scan_and_register + list_registered_models ───────────────────────────

    #[tokio::test]
    async fn test_scan_and_register_with_models() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("net.onnx"), b"onnx data").unwrap();

        let manager = default_manager();
        let registered = manager.scan_and_register(dir.path()).await.unwrap();
        assert_eq!(registered.len(), 1);
        assert!(registered.contains(&"net".to_string()));

        let listed = manager.list_registered_models();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "net");
    }

    // ── register_model_from_path – success + metadata lookup ─────────────────

    #[tokio::test]
    async fn test_register_model_from_path_success() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scored.pt");
        std::fs::write(&path, b"fake pt data").unwrap();

        let manager = default_manager();
        let model_id = manager
            .register_model_from_path(&path, Some("scored-model".to_string()))
            .await
            .unwrap();
        assert_eq!(model_id, "scored-model");

        let meta = manager.get_model_metadata("scored-model").unwrap();
        assert_eq!(meta.format, ModelFormat::PyTorch);
    }

    // ── get_registry_stats after registering a model ─────────────────────────

    #[tokio::test]
    async fn test_get_registry_stats_after_registration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stat.onnx");
        std::fs::write(&path, b"bytes").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, None)
            .await
            .unwrap();

        let stats = manager.get_registry_stats();
        assert_eq!(stats["total_models"], 1);
        assert_eq!(stats["loaded_models"], 0);
    }

    // ── list_available combines legacy + registry ─────────────────────────────

    #[tokio::test]
    async fn test_list_available_combined() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reg.onnx");
        std::fs::write(&path, b"data").unwrap();

        let manager = default_manager();
        // Register one legacy, one registry model
        manager
            .register_model("legacy".to_string(), BaseModel::new("legacy".to_string()))
            .await
            .unwrap();
        manager.register_model_from_path(&path, None).await.unwrap();

        let available = manager.list_available();
        assert!(available.contains(&"legacy".to_string()));
        assert!(available.contains(&"reg".to_string()));
    }

    // ── infer_onnx / infer_registered – model not loaded errors ──────────────

    #[tokio::test]
    async fn test_infer_onnx_model_not_in_registry() {
        let manager = default_manager();
        let result = manager
            .infer_onnx("ghost", &serde_json::json!({}))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_infer_registered_model_not_found() {
        let manager = default_manager();
        let result = manager
            .infer_registered("missing", &serde_json::json!({}))
            .await;
        assert!(result.is_err());
    }

    // ── load_pytorch_model disabled (no torch feature) ───────────────────────

    #[cfg(not(feature = "torch"))]
    #[tokio::test]
    async fn test_load_pytorch_model_disabled() {
        let manager = default_manager();
        let result = manager.load_pytorch_model("any").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("PyTorch"));
    }

    #[cfg(not(feature = "torch"))]
    #[tokio::test]
    async fn test_infer_pytorch_disabled() {
        let manager = default_manager();
        let result = manager
            .infer_pytorch("any", &serde_json::json!({}))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("PyTorch"));
    }
}
