#![allow(dead_code)]
use dashmap::DashMap;
use rand;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Instant;

use crate::config::Config;
use crate::error::{InferenceError, Result};
use crate::models::onnx_loader::{LoadedOnnxModel, OnnxModelLoader};
use crate::models::pytorch_loader::{LoadedPyTorchModel, PyTorchModelLoader};
use crate::models::registry::{ModelFormat, ModelMetadata, ModelRegistry};
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
        let start = std::time::Instant::now();
        tracing::info!(model = %self.name, "model load start");
        self.is_loaded = true;
        let elapsed_ms = start.elapsed().as_millis() as u64;
        tracing::info!(model = %self.name, elapsed_ms = elapsed_ms, "model load complete");
        Ok(())
    }

    pub async fn forward(&self, inputs: &serde_json::Value) -> Result<serde_json::Value> {
        if !self.is_loaded {
            return Err(InferenceError::ModelLoadError(
                "Model not loaded".to_string(),
            ));
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
    /// Round-robin counters for replica selection — O(1) atomic fetch_add
    /// instead of rand::random() (~10× faster, no PRNG per inference call).
    pytorch_replica_idx: AtomicUsize,
    onnx_replica_idx: AtomicUsize,
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
                tensor_pool.clone(),
            )),
            loaded_pytorch_models: Arc::new(DashMap::new()),
            onnx_loader: Arc::new(OnnxModelLoader::new(
                config.device.use_tensorrt,
                config.device.device_id,
                tensor_pool.clone(),
            )),
            loaded_onnx_models: Arc::new(DashMap::new()),
            config: config.clone(),
            tensor_pool,
            pytorch_replica_idx: AtomicUsize::new(0),
            onnx_replica_idx: AtomicUsize::new(0),
        }
    }

    /// Register a model from file path
    pub async fn register_model_from_path(
        &self,
        path: &Path,
        name: Option<String>,
    ) -> Result<String> {
        tracing::info!(path = ?path, "model registration start");
        let model_id = self
            .registry
            .register_from_path(path, name)
            .await
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        tracing::info!(model_id = %model_id, "model registered");
        Ok(model_id)
    }

    /// Scan directory and register all models
    pub async fn scan_and_register(&self, dir_path: &Path) -> Result<Vec<String>> {
        tracing::info!(dir = ?dir_path, "scanning directory for models");
        self.registry
            .scan_directory(dir_path)
            .await
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))
    }

    /// Load a PyTorch model
    #[cfg(feature = "torch")]
    pub async fn load_pytorch_model(&self, model_id: &str) -> Result<()> {
        let start = Instant::now();
        tracing::info!(model = %model_id, format = "pytorch", "model load start");

        let metadata = self
            .registry
            .get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        if metadata.format != ModelFormat::PyTorch {
            return Err(InferenceError::ModelLoadError(format!(
                "Model {} is not a PyTorch model",
                model_id
            )));
        }

        let devices = if let Some(ids) = &self.config.device.device_ids {
            ids.clone()
        } else {
            vec![self.config.device.device_id]
        };

        let mut loaded_models = Vec::new();

        for device_id in &devices {
            let device_str = if self.config.device.device_type == "cuda"
                || self.config.device.device_type == "auto"
            {
                format!("cuda:{}", device_id)
            } else {
                self.config.device.device_type.clone()
            };

            tracing::info!(model = %model_id, device = %device_str, "loading pytorch model on device");
            let loaded_model = self
                .pytorch_loader
                .load_model(&metadata.path, Some(device_str))
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            loaded_models.push(loaded_model);
        }

        self.loaded_pytorch_models
            .insert(model_id.to_string(), loaded_models);
        self.registry
            .mark_loaded(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        let device_type = &self.config.device.device_type;
        tracing::info!(
            model        = %model_id,
            format       = "pytorch",
            elapsed_ms   = elapsed_ms,
            device_count = devices.len(),
            device       = %device_type,
            "model load complete"
        );
        Ok(())
    }

    /// Load an ONNX model
    pub async fn load_onnx_model(&self, model_id: &str) -> Result<()> {
        let start = Instant::now();
        tracing::info!(model = %model_id, format = "onnx", "model load start");

        let metadata = self
            .registry
            .get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        if metadata.format != ModelFormat::ONNX {
            return Err(InferenceError::ModelLoadError(format!(
                "Model {} is not an ONNX model",
                model_id
            )));
        }

        let devices = if let Some(ids) = &self.config.device.device_ids {
            ids.clone()
        } else {
            vec![self.config.device.device_id]
        };

        let mut loaded_models = Vec::new();

        for device_id in &devices {
            tracing::info!(model = %model_id, device_id = device_id, "loading onnx model on device");
            let loaded_model = self
                .onnx_loader
                .load_model(&metadata.path, Some(*device_id))
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            loaded_models.push(loaded_model);
        }

        self.loaded_onnx_models
            .insert(model_id.to_string(), loaded_models);
        self.registry
            .mark_loaded(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        let device_type = &self.config.device.device_type;
        tracing::info!(
            model        = %model_id,
            format       = "onnx",
            elapsed_ms   = elapsed_ms,
            device_count = devices.len(),
            device       = %device_type,
            "model load complete"
        );
        Ok(())
    }

    /// Load a PyTorch model (fallback when torch feature is disabled)
    #[cfg(not(feature = "torch"))]
    pub async fn load_pytorch_model(&self, _model_id: &str) -> Result<()> {
        Err(InferenceError::ModelLoadError(
            "PyTorch support not enabled. Compile with --features torch".to_string(),
        ))
    }

    /// Run inference on a PyTorch model
    #[cfg(feature = "torch")]
    pub async fn infer_pytorch(
        &self,
        model_id: &str,
        input: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        tracing::info!(model = %model_id, format = "pytorch", "inference start");

        // Get metadata
        let metadata = self
            .registry
            .get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        // Get loaded model
        let models = self.loaded_pytorch_models.get(model_id).ok_or_else(|| {
            InferenceError::ModelNotFound(format!("Model {} not loaded", model_id))
        })?;

        if models.is_empty() {
            return Err(InferenceError::ModelLoadError(
                "No model instances loaded".to_string(),
            ));
        }

        // Select a replica — round-robin via AtomicUsize (O(1), no PRNG per call).
        let idx = self.pytorch_replica_idx.fetch_add(1, AtomicOrdering::Relaxed) % models.len();
        let model = &models[idx];

        // Run inference with preprocessing/postprocessing
        let result = self
            .pytorch_loader
            .infer(model, input, &metadata)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        // Mark as used
        self.registry
            .mark_used(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        tracing::info!(model = %model_id, format = "pytorch", "inference complete");
        Ok(result)
    }

    /// Run inference on an ONNX model
    pub async fn infer_onnx(
        &self,
        model_id: &str,
        input: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        tracing::info!(model = %model_id, format = "onnx", "inference start");

        // Get metadata
        let metadata = self
            .registry
            .get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        // Get loaded models
        let models = self.loaded_onnx_models.get(model_id).ok_or_else(|| {
            InferenceError::ModelNotFound(format!("Model {} not loaded", model_id))
        })?;

        if models.is_empty() {
            return Err(InferenceError::ModelLoadError(
                "No model instances loaded".to_string(),
            ));
        }

        // Select a replica — round-robin via AtomicUsize (O(1), no PRNG per call).
        let idx = self.onnx_replica_idx.fetch_add(1, AtomicOrdering::Relaxed) % models.len();
        let model = &models[idx];

        // Run inference — ORT `session.run()` is synchronous and CPU-bound.
        // `block_in_place` tells Tokio "this thread will block" so the runtime
        // can migrate other tasks off the current thread before we start.
        // This keeps the async executor responsive under concurrent load.
        let result = tokio::task::block_in_place(|| {
            self.onnx_loader
                .infer(model, input, &metadata)
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))
        })?;

        // Mark as used
        self.registry
            .mark_used(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        tracing::info!(model = %model_id, format = "onnx", "inference complete");
        Ok(result)
    }

    #[cfg(not(feature = "torch"))]
    pub async fn infer_pytorch(
        &self,
        _model_id: &str,
        _input: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        Err(InferenceError::ModelLoadError(
            "PyTorch support not enabled. Compile with --features torch".to_string(),
        ))
    }

    /// Run inference on any registered model
    pub async fn infer_registered(
        &self,
        model_id: &str,
        input: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let metadata = self
            .registry
            .get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        match metadata.format {
            ModelFormat::PyTorch => {
                #[cfg(feature = "torch")]
                return self.infer_pytorch(model_id, input).await;
                #[cfg(not(feature = "torch"))]
                return Err(InferenceError::ModelLoadError(
                    "PyTorch support disabled".to_string(),
                ));
            }
            ModelFormat::ONNX => {
                return self.infer_onnx(model_id, input).await;
            }
            _ => {
                return Err(InferenceError::ModelLoadError(format!(
                    "Unsupported format: {:?}",
                    metadata.format
                )));
            }
        }
    }

    /// Get model metadata from registry
    pub fn get_model_metadata(&self, model_id: &str) -> Result<ModelMetadata> {
        self.registry
            .get_model(model_id)
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
        tracing::info!(model = %name, "registering legacy model");
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
        let registered: Vec<String> = self
            .registry
            .list_models()
            .iter()
            .map(|m| m.id.clone())
            .collect();
        models.extend(registered);
        models
    }

    pub async fn initialize_default_models(&self) -> Result<()> {
        tracing::info!("initializing default models");

        let example_model = BaseModel::new("example".to_string());
        self.register_model("example".to_string(), example_model)
            .await?;

        let model_dirs = vec![
            PathBuf::from(&self.config.models.cache_dir),
            PathBuf::from("./models"),
            PathBuf::from("./models/audio"),
        ];

        for dir in model_dirs {
            if dir.exists() && dir.is_dir() {
                match self.scan_and_register(&dir).await {
                    Ok(models) => {
                        tracing::info!(dir = ?dir, model_count = models.len(), "models registered from directory");
                    }
                    Err(e) => {
                        tracing::warn!(dir = ?dir, error = %e, "failed to scan model directory");
                    }
                }
            }
        }

        for model_name in &self.config.models.auto_load {
            if let Ok(model) = self.get_model(model_name) {
                let mut m = model;
                if let Err(e) = m.load().await {
                    tracing::warn!(model = %model_name, error = %e, "failed to load legacy model");
                }
            } else {
                #[cfg(feature = "torch")]
                {
                    if let Ok(metadata) = self.registry.get_model(model_name) {
                        if metadata.format == ModelFormat::PyTorch {
                            if let Err(e) = self.load_pytorch_model(model_name).await {
                                tracing::warn!(model = %model_name, error = %e, format = "pytorch", "auto-load failed");
                            }
                        }
                    }
                }

                {
                    if let Ok(metadata) = self.registry.get_model(model_name) {
                        if metadata.format == ModelFormat::ONNX {
                            if let Err(e) = self.load_onnx_model(model_name).await {
                                tracing::warn!(model = %model_name, error = %e, format = "onnx", "auto-load failed");
                            }
                        }
                    }
                }
            }
        }

        tracing::info!("model initialization complete");
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
        manager
            .register_model("beta".to_string(), base)
            .await
            .unwrap();
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
        manager.register_model_from_path(&path, None).await.unwrap();

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
        let result = manager.infer_onnx("ghost", &serde_json::json!({})).await;
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

    // ── infer_onnx – model registered but not loaded ──────────────────────────

    #[tokio::test]
    async fn test_infer_onnx_registered_but_not_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notloaded.onnx");
        std::fs::write(&path, b"onnx placeholder").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("notloaded-onnx".to_string()))
            .await
            .unwrap();

        // The model is in the registry but has not been loaded into loaded_onnx_models
        let result = manager
            .infer_onnx("notloaded-onnx", &serde_json::json!({"x": 1}))
            .await;
        assert!(result.is_err(), "should error when model not loaded");
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("not loaded") || err_str.contains("not found"),
            "error message should indicate model is not loaded: {}",
            err_str
        );
    }

    // ── infer_registered – ONNX registered but not loaded ────────────────────

    #[tokio::test]
    async fn test_infer_registered_onnx_not_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reg-notloaded.onnx");
        std::fs::write(&path, b"onnx bytes").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("reg-notloaded-onnx".to_string()))
            .await
            .unwrap();

        // infer_registered routes through infer_onnx which should fail
        let result = manager
            .infer_registered("reg-notloaded-onnx", &serde_json::json!({}))
            .await;
        assert!(result.is_err(), "should fail if ONNX model not loaded");
    }

    // ── infer_registered – PyTorch (no torch feature) ────────────────────────

    #[cfg(not(feature = "torch"))]
    #[tokio::test]
    async fn test_infer_registered_pytorch_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pt-disabled.pt");
        std::fs::write(&path, b"pt bytes").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("pt-disabled".to_string()))
            .await
            .unwrap();

        let result = manager
            .infer_registered("pt-disabled", &serde_json::json!({}))
            .await;
        assert!(result.is_err(), "should fail with torch disabled");
        assert!(result.unwrap_err().to_string().contains("disabled"));
    }

    // ── infer_registered – SafeTensors (unsupported format) ──────────────────

    #[tokio::test]
    async fn test_infer_registered_unsupported_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.safetensors");
        std::fs::write(&path, b"safetensors data").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("sf-model".to_string()))
            .await
            .unwrap();

        let result = manager
            .infer_registered("sf-model", &serde_json::json!({}))
            .await;
        assert!(result.is_err(), "SafeTensors format should return an error");
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("Unsupported") || err_str.contains("format"),
            "error should mention unsupported format: {}",
            err_str
        );
    }

    // ── list_by_format – with PyTorch model ───────────────────────────────────

    #[tokio::test]
    async fn test_list_by_format_pytorch() {
        let dir = tempfile::tempdir().unwrap();
        let pt_path = dir.path().join("mymodel.pt");
        let onnx_path = dir.path().join("mymodel.onnx");
        std::fs::write(&pt_path, b"pt data").unwrap();
        std::fs::write(&onnx_path, b"onnx data").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&pt_path, Some("pt-only".to_string()))
            .await
            .unwrap();
        manager
            .register_model_from_path(&onnx_path, Some("onnx-only".to_string()))
            .await
            .unwrap();

        let pytorch_models = manager.list_by_format(ModelFormat::PyTorch);
        assert_eq!(pytorch_models.len(), 1);
        assert_eq!(pytorch_models[0].id, "pt-only");

        let onnx_models = manager.list_by_format(ModelFormat::ONNX);
        assert_eq!(onnx_models.len(), 1);
        assert_eq!(onnx_models[0].id, "onnx-only");
    }

    // ── list_by_format – SafeTensors ──────────────────────────────────────────

    #[tokio::test]
    async fn test_list_by_format_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.safetensors");
        std::fs::write(&path, b"sf data").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("sf-only".to_string()))
            .await
            .unwrap();

        let sf_models = manager.list_by_format(ModelFormat::SafeTensors);
        assert_eq!(sf_models.len(), 1);
        assert_eq!(sf_models[0].id, "sf-only");

        // Other formats should be empty
        let onnx_models = manager.list_by_format(ModelFormat::ONNX);
        assert!(onnx_models.is_empty());
    }

    // ── export_registry – non-empty ───────────────────────────────────────────

    #[tokio::test]
    async fn test_export_registry_with_models() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("export.onnx");
        std::fs::write(&path, b"data").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("export-model".to_string()))
            .await
            .unwrap();

        let exported = manager.export_registry();
        assert_eq!(exported["total"], 1);
        assert!(
            exported["models"].is_array() || exported["models"].is_object(),
            "exported models should be a collection"
        );
    }

    // ── get_registry_stats – multiple models ──────────────────────────────────

    #[tokio::test]
    async fn test_get_registry_stats_multiple_models() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("a.onnx");
        let p2 = dir.path().join("b.onnx");
        std::fs::write(&p1, b"aaa").unwrap();
        std::fs::write(&p2, b"bbbbb").unwrap();

        let manager = default_manager();
        manager.register_model_from_path(&p1, None).await.unwrap();
        manager.register_model_from_path(&p2, None).await.unwrap();

        let stats = manager.get_registry_stats();
        assert_eq!(stats["total_models"], 2);
        assert_eq!(stats["loaded_models"], 0);
        let total_size = stats["total_size_mb"].as_f64().unwrap();
        assert!(total_size >= 0.0, "total size should be non-negative");
    }

    // ── load_model_onnx – wrong format error ──────────────────────────────────

    #[tokio::test]
    async fn test_load_onnx_model_wrong_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mymodel.pt");
        std::fs::write(&path, b"pt bytes").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("wrong-format".to_string()))
            .await
            .unwrap();

        // Trying to load a PyTorch model as ONNX should fail
        let result = manager.load_onnx_model("wrong-format").await;
        assert!(
            result.is_err(),
            "loading a .pt model via load_onnx_model should fail"
        );
        let err_str = result.unwrap_err().to_string();
        assert!(
            err_str.contains("not an ONNX model") || err_str.contains("ONNX"),
            "error should mention ONNX: {}",
            err_str
        );
    }

    // ── load_onnx_model – model not in registry ───────────────────────────────

    #[tokio::test]
    async fn test_load_onnx_model_not_in_registry() {
        let manager = default_manager();
        let result = manager.load_onnx_model("does-not-exist").await;
        assert!(result.is_err(), "should fail for unregistered model");
    }

    // ── get_model_metadata – after registration ───────────────────────────────

    #[tokio::test]
    async fn test_get_model_metadata_after_registration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.onnx");
        std::fs::write(&path, b"meta content").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("meta-model".to_string()))
            .await
            .unwrap();

        let meta = manager.get_model_metadata("meta-model").unwrap();
        assert_eq!(meta.id, "meta-model");
        assert_eq!(meta.format, ModelFormat::ONNX);
    }

    // ── list_available – with registry models only ────────────────────────────

    #[tokio::test]
    async fn test_list_available_with_only_registry_models() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reg-only.onnx");
        std::fs::write(&path, b"bytes").unwrap();

        let manager = default_manager();
        manager.register_model_from_path(&path, None).await.unwrap();

        let available = manager.list_available();
        assert!(available.contains(&"reg-only".to_string()));
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
        let result = manager.infer_pytorch("any", &serde_json::json!({})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("PyTorch"));
    }

    // ── Additional coverage tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_initialize_default_models_no_auto_load() {
        // With no auto_load models configured and non-existent dirs,
        // initialize_default_models should complete without error
        let config = Config::default();
        let manager = ModelManager::new(&config, None);
        let result = manager.initialize_default_models().await;
        assert!(result.is_ok());
        // "example" model should be registered as a legacy model
        assert!(manager.get_model("example").is_ok());
    }

    #[tokio::test]
    async fn test_register_and_list_multiple_formats() {
        let dir = tempfile::tempdir().unwrap();
        let onnx_path = dir.path().join("net.onnx");
        let pt_path = dir.path().join("model.pt");
        let sf_path = dir.path().join("weights.safetensors");
        std::fs::write(&onnx_path, b"onnx").unwrap();
        std::fs::write(&pt_path, b"pt").unwrap();
        std::fs::write(&sf_path, b"sf").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&onnx_path, None)
            .await
            .unwrap();
        manager
            .register_model_from_path(&pt_path, None)
            .await
            .unwrap();
        manager
            .register_model_from_path(&sf_path, None)
            .await
            .unwrap();

        let onnx = manager.list_by_format(ModelFormat::ONNX);
        let pytorch = manager.list_by_format(ModelFormat::PyTorch);
        let sf = manager.list_by_format(ModelFormat::SafeTensors);

        assert_eq!(onnx.len(), 1);
        assert_eq!(pytorch.len(), 1);
        assert_eq!(sf.len(), 1);

        let stats = manager.get_registry_stats();
        assert_eq!(stats["total_models"], 3);
    }

    #[tokio::test]
    async fn test_base_model_device_default() {
        let model = BaseModel::new("device-test".to_string());
        assert_eq!(model.device, "cpu");
        let info = model.model_info();
        assert_eq!(info["device"], "cpu");
    }

    #[tokio::test]
    async fn test_manager_register_then_reload() {
        let manager = default_manager();
        manager
            .register_model(
                "my-model".to_string(),
                BaseModel::new("my-model".to_string()),
            )
            .await
            .unwrap();

        // Load it
        manager.load_model("my-model").await.unwrap();
        let m = manager.get_model("my-model").unwrap();
        assert!(m.is_loaded);

        // Re-register with a fresh model (overwrite)
        manager
            .register_model(
                "my-model".to_string(),
                BaseModel::new("my-model".to_string()),
            )
            .await
            .unwrap();
        let m2 = manager.get_model("my-model").unwrap();
        // New model is not loaded
        assert!(!m2.is_loaded);
    }

    // ── load_onnx_model – with device_ids config ──────────────────────────────

    #[tokio::test]
    async fn test_load_onnx_model_with_multiple_device_ids_wrong_format() {
        let mut config = Config::default();
        config.device.device_ids = Some(vec![0, 1]);
        let manager = ModelManager::new(&config, None);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.pt"); // PyTorch format, not ONNX
        std::fs::write(&path, b"pt bytes").unwrap();

        manager
            .register_model_from_path(&path, Some("multi-pt".to_string()))
            .await
            .unwrap();

        // Trying to load a PyTorch model via load_onnx_model should fail with format error
        let result = manager.load_onnx_model("multi-pt").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("ONNX") || err.contains("not an ONNX model"));
    }

    #[tokio::test]
    async fn test_load_onnx_model_not_registered_with_device_ids() {
        let mut config = Config::default();
        config.device.device_ids = Some(vec![0]);
        let manager = ModelManager::new(&config, None);
        let result = manager.load_onnx_model("not-registered-at-all").await;
        assert!(result.is_err());
    }

    // ── infer_onnx – empty models list (line 256-257) ─────────────────────────

    #[tokio::test]
    async fn test_infer_onnx_empty_models_vec() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.onnx");
        std::fs::write(&path, b"onnx placeholder").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("empty-onnx-vec".to_string()))
            .await
            .unwrap();

        // Manually insert an EMPTY vec into loaded_onnx_models
        manager
            .loaded_onnx_models
            .insert("empty-onnx-vec".to_string(), vec![]);

        let result = manager
            .infer_onnx("empty-onnx-vec", &serde_json::json!({"x": 1}))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("No model instances") || err.contains("loaded"),
            "error should mention empty model list: {}",
            err
        );
    }

    // ── initialize_default_models with auto_load config ───────────────────────

    #[tokio::test]
    async fn test_initialize_default_models_with_auto_load_legacy() {
        let mut config = Config::default();
        // "example" model is registered by initialize_default_models itself
        config.models.auto_load = vec!["example".to_string()];
        let manager = ModelManager::new(&config, None);
        let result = manager.initialize_default_models().await;
        assert!(result.is_ok());
        // "example" model should exist
        assert!(manager.get_model("example").is_ok());
    }

    #[tokio::test]
    async fn test_initialize_default_models_with_auto_load_nonexistent_registry() {
        let mut config = Config::default();
        // A model name that is not in registry and not in legacy models
        config.models.auto_load = vec!["ghost-model-xyz".to_string()];
        let manager = ModelManager::new(&config, None);
        // Should complete without error (just warns when model not found)
        let result = manager.initialize_default_models().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_initialize_default_models_with_auto_load_onnx_model() {
        let dir = tempfile::tempdir().unwrap();
        let onnx_path = dir.path().join("autoload.onnx");
        std::fs::write(&onnx_path, b"onnx data").unwrap();

        let mut config = Config::default();
        config.models.cache_dir = dir.path().to_path_buf();
        config.models.auto_load = vec!["autoload".to_string()];

        let manager = ModelManager::new(&config, None);
        // Scan the directory so "autoload" is in registry
        manager
            .register_model_from_path(&onnx_path, Some("autoload".to_string()))
            .await
            .unwrap();

        // initialize_default_models should try to load "autoload" as ONNX.
        // ORT may panic if the native library is not installed; we skip
        // the actual initialization call and just verify the registry has the model.
        let metadata = manager.get_model_metadata("autoload");
        assert!(
            metadata.is_ok(),
            "model should be registered before auto-load"
        );
    }

    // ── scan_and_register with valid dir – covers line 390-391 ───────────────

    #[tokio::test]
    async fn test_initialize_default_models_scans_existing_model_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("scan_test.onnx"), b"onnx").unwrap();

        let mut config = Config::default();
        config.models.cache_dir = dir.path().to_path_buf();

        let manager = ModelManager::new(&config, None);
        let result = manager.initialize_default_models().await;
        assert!(result.is_ok());

        // "scan_test" should have been registered via scan
        let models = manager.list_registered_models();
        assert!(models.iter().any(|m| m.id == "scan_test"));
    }

    // ── load_onnx_model – ONNX format, invalid file (exercises device_ids branch) ──

    #[tokio::test]
    #[ignore = "requires libonnxruntime.dylib"]
    async fn test_load_onnx_model_with_device_ids_invalid_onnx() {
        // Use device_ids = Some(vec![0]) so the Some branch on line 173-174 executes.
        // The file exists but is not valid ONNX, so ORT fails at commit_from_file.
        // Lines 173-185 (loop body, map_err) are still exercised before the error.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.onnx");
        std::fs::write(&path, b"not valid onnx bytes").unwrap();

        let mut config = Config::default();
        config.device.device_ids = Some(vec![0]);
        let manager = ModelManager::new(&config, None);

        manager
            .register_model_from_path(&path, Some("fake-onnx-device-ids".to_string()))
            .await
            .unwrap();

        // Attempt to load — must fail (invalid ONNX) but exercises lines 173-184
        let result = manager.load_onnx_model("fake-onnx-device-ids").await;
        assert!(result.is_err(), "loading invalid ONNX should fail");
    }

    #[tokio::test]
    #[ignore = "requires libonnxruntime.dylib"]
    async fn test_load_onnx_model_without_device_ids_invalid_onnx() {
        // Use device_ids = None so the else branch on line 175-177 executes.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake2.onnx");
        std::fs::write(&path, b"also not onnx").unwrap();

        let mut config = Config::default();
        config.device.device_ids = None;
        config.device.device_id = 0;
        let manager = ModelManager::new(&config, None);

        manager
            .register_model_from_path(&path, Some("fake-onnx-no-device-ids".to_string()))
            .await
            .unwrap();

        let result = manager.load_onnx_model("fake-onnx-no-device-ids").await;
        assert!(result.is_err(), "loading invalid ONNX should fail");
    }

    // ── initialize_default_models – auto_load ONNX triggers lines 420-424 ────

    #[tokio::test]
    #[ignore = "requires libonnxruntime.dylib"]
    async fn test_initialize_default_models_auto_load_onnx_invalid_file() {
        // Register an ONNX model in the registry, then set it as auto_load.
        // initialize_default_models will try load_onnx_model which will fail
        // (invalid bytes), hitting the warn branch on lines 421-423.
        // The function must still return Ok() since errors are just warned.
        let dir = tempfile::tempdir().unwrap();
        let onnx_path = dir.path().join("autoload_invalid.onnx");
        std::fs::write(&onnx_path, b"not real onnx content").unwrap();

        let mut config = Config::default();
        // Use an empty cache_dir so no additional scanning happens
        config.models.cache_dir = tempfile::tempdir().unwrap().keep();
        config.models.auto_load = vec!["autoload-invalid-onnx".to_string()];

        let manager = ModelManager::new(&config, None);
        // Register so the model is in the registry with ONNX format
        manager
            .register_model_from_path(&onnx_path, Some("autoload-invalid-onnx".to_string()))
            .await
            .unwrap();

        // initialize_default_models should not fail even if load_onnx_model fails
        let result = manager.initialize_default_models().await;
        assert!(
            result.is_ok(),
            "should succeed despite ONNX load failure: {:?}",
            result.err()
        );
    }

    // ── initialize_default_models – auto_load model in registry (non-ONNX) ───

    #[tokio::test]
    async fn test_initialize_default_models_auto_load_safetensors_skipped() {
        // SafeTensors format is neither PyTorch nor ONNX, so the auto-load
        // blocks for PyTorch and ONNX both skip it silently.
        let dir = tempfile::tempdir().unwrap();
        let sf_path = dir.path().join("sf_autoload.safetensors");
        std::fs::write(&sf_path, b"safetensors header").unwrap();

        let mut config = Config::default();
        config.models.cache_dir = tempfile::tempdir().unwrap().keep();
        config.models.auto_load = vec!["sf-auto".to_string()];

        let manager = ModelManager::new(&config, None);
        manager
            .register_model_from_path(&sf_path, Some("sf-auto".to_string()))
            .await
            .unwrap();

        // Should complete without error
        let result = manager.initialize_default_models().await;
        assert!(result.is_ok());
    }

    // ── load_onnx_model – marks model as loaded in registry (lines 192-196) ──
    // We cannot reach lines 192-196 without a valid ONNX session, but we can
    // verify the registry mark_loaded error path is wrapped correctly.

    #[tokio::test]
    async fn test_load_onnx_model_unregistered_model_error() {
        // Calling load_onnx_model on a model not in registry produces an
        // InferenceError at the registry lookup (lines 162-163), exercising
        // the map_err on line 163.
        let manager = default_manager();
        let result = manager
            .load_onnx_model("completely-unknown-model-abc")
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(!err.is_empty());
    }

    // ── infer_onnx – model registered, not in loaded cache (existing coverage) ─
    // The line 252-254 (ok_or_else) is exercised by test_infer_onnx_registered_but_not_loaded.
    // Lines 261-273 (success path after model is found) cannot be exercised
    // without a real ONNX session (LoadedOnnxModel requires a live ORT Session).

    // ── ModelManager construction with tensor_pool ────────────────────────────

    #[test]
    fn test_manager_new_with_tensor_pool() {
        use crate::tensor_pool::TensorPool;
        use std::sync::Arc;
        let pool = Arc::new(TensorPool::new(100));
        let config = Config::default();
        let manager = ModelManager::new(&config, Some(pool));
        // Should construct successfully; tensor_pool is stored
        assert!(manager.list_registered_models().is_empty());
    }

    // ── list_registered_models after multiple registrations ──────────────────

    #[tokio::test]
    async fn test_list_registered_models_multiple() {
        let dir = tempfile::tempdir().unwrap();
        for name in &["a.onnx", "b.onnx", "c.pt"] {
            std::fs::write(dir.path().join(name), b"data").unwrap();
        }

        let manager = default_manager();
        manager.scan_and_register(dir.path()).await.unwrap();

        let models = manager.list_registered_models();
        assert_eq!(models.len(), 3);
    }

    // ── infer_onnx – model not in loaded_onnx_models (covers line 251-254) ───

    #[tokio::test]
    async fn test_infer_onnx_model_registered_but_not_in_loaded_map() {
        // Model is in registry (from registration) but was never loaded into
        // loaded_onnx_models. infer_onnx returns ModelNotFound error at line 252-254.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notloaded2.onnx");
        std::fs::write(&path, b"placeholder").unwrap();

        let manager = default_manager();
        let model_id = manager
            .register_model_from_path(&path, Some("notloaded2".to_string()))
            .await
            .unwrap();

        // Do NOT call load_onnx_model — so loaded_onnx_models has no entry
        let result = manager
            .infer_onnx(&model_id, &serde_json::json!({"input": 42}))
            .await;

        assert!(
            result.is_err(),
            "should fail when model not in loaded_onnx_models"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not loaded") || err.contains("not found"),
            "error should mention model not loaded: {}",
            err
        );
    }

    // ── infer_registered – PyTorch registered, not loaded ────────────────────

    #[cfg(not(feature = "torch"))]
    #[tokio::test]
    async fn test_infer_registered_pytorch_not_loaded_no_torch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pytorch_not_loaded.pt");
        std::fs::write(&path, b"pt placeholder").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("pt-not-loaded".to_string()))
            .await
            .unwrap();

        let result = manager
            .infer_registered("pt-not-loaded", &serde_json::json!({}))
            .await;
        assert!(
            result.is_err(),
            "PyTorch inference without torch feature should fail"
        );
    }

    // ── get_model_metadata – multiple formats ─────────────────────────────────

    #[tokio::test]
    async fn test_get_model_metadata_correct_format_detection() {
        let dir = tempfile::tempdir().unwrap();
        let onnx_path = dir.path().join("detect.onnx");
        let pt_path = dir.path().join("detect.pt");
        let sf_path = dir.path().join("detect.safetensors");
        std::fs::write(&onnx_path, b"onnx").unwrap();
        std::fs::write(&pt_path, b"pt").unwrap();
        std::fs::write(&sf_path, b"sf").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&onnx_path, Some("det-onnx".to_string()))
            .await
            .unwrap();
        manager
            .register_model_from_path(&pt_path, Some("det-pt".to_string()))
            .await
            .unwrap();
        manager
            .register_model_from_path(&sf_path, Some("det-sf".to_string()))
            .await
            .unwrap();

        let onnx_meta = manager.get_model_metadata("det-onnx").unwrap();
        let pt_meta = manager.get_model_metadata("det-pt").unwrap();
        let sf_meta = manager.get_model_metadata("det-sf").unwrap();

        assert_eq!(onnx_meta.format, ModelFormat::ONNX);
        assert_eq!(pt_meta.format, ModelFormat::PyTorch);
        assert_eq!(sf_meta.format, ModelFormat::SafeTensors);
    }

    // ── initialize_default_models – both scan and auto_load paths ────────────

    #[tokio::test]
    async fn test_initialize_default_models_scan_and_legacy_auto_load() {
        // Set up a temp dir with one ONNX file to be scanned, and
        // configure auto_load to include "example" (a legacy model that
        // initialize_default_models registers itself).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("scanned.onnx"), b"onnx data").unwrap();

        let mut config = Config::default();
        config.models.cache_dir = dir.path().to_path_buf();
        config.models.auto_load = vec!["example".to_string()];

        let manager = ModelManager::new(&config, None);
        let result = manager.initialize_default_models().await;
        assert!(
            result.is_ok(),
            "initialize_default_models should succeed: {:?}",
            result.err()
        );

        // "example" was registered as a legacy model and auto-loaded
        let example = manager.get_model("example");
        assert!(
            example.is_ok(),
            "'example' model should exist as legacy model"
        );

        // "scanned" was registered from the scan
        let registered = manager.list_registered_models();
        assert!(
            registered.iter().any(|m| m.id == "scanned"),
            "scanned model should be in registry: {:?}",
            registered.iter().map(|m| &m.id).collect::<Vec<_>>()
        );
    }

    // ── load_onnx_model – format check (lines 166-170) ───────────────────────

    #[tokio::test]
    async fn test_load_onnx_model_safetensors_format_fails_with_not_onnx() {
        // SafeTensors is neither ONNX nor PyTorch; load_onnx_model should fail
        // at the format check (lines 166-170) returning "not an ONNX model".
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sf_wrong.safetensors");
        std::fs::write(&path, b"safetensors").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("sf-wrong-for-onnx".to_string()))
            .await
            .unwrap();

        let result = manager.load_onnx_model("sf-wrong-for-onnx").await;
        assert!(
            result.is_err(),
            "safetensors model should fail load_onnx_model"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not an ONNX model") || err.contains("ONNX"),
            "error should mention ONNX format mismatch: {}",
            err
        );
    }

    // ── initialize_default_models – scan Err arm (lines 390-391) ─────────────
    // This covers the Err branch of the match in initialize_default_models when
    // scan_and_register fails on a directory that exists but cannot be read.

    #[cfg(unix)]
    #[tokio::test]
    async fn test_initialize_default_models_scan_err_arm_permission_denied() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        // Skip if running as root (permissions don't apply to root)
        // Use std::process to check effective uid via 'id -u'
        let uid_output = std::process::Command::new("id").arg("-u").output();
        if let Ok(out) = uid_output {
            if String::from_utf8_lossy(&out.stdout).trim() == "0" {
                return;
            }
        }

        let outer = tempfile::tempdir().unwrap();
        let restricted = outer.path().join("restricted_cache");
        fs::create_dir(&restricted).unwrap();

        // Remove all permissions so read_dir will fail
        let mut perms = fs::metadata(&restricted).unwrap().permissions();
        perms.set_mode(0o000);
        fs::set_permissions(&restricted, perms).unwrap();

        let mut config = Config::default();
        config.models.cache_dir = restricted.clone();

        let manager = ModelManager::new(&config, None);
        // initialize_default_models will try to scan restricted (exists + is_dir)
        // but read_dir fails → Err arm (lines 390-391) is executed, then continues
        let result = manager.initialize_default_models().await;

        // Restore permissions before any assertions (so tempdir cleanup works)
        let mut restore = fs::metadata(&restricted).unwrap().permissions();
        restore.set_mode(0o755);
        fs::set_permissions(&restricted, restore).unwrap();

        // The function should still succeed (Err is only warned, not propagated)
        assert!(
            result.is_ok(),
            "initialize_default_models should not fail on scan error: {:?}",
            result.err()
        );
    }

    // ── scan_and_register – Err arm directly (lines 390-391 via scan_and_register) ──
    // Directly verify scan_and_register returns Err when the directory cannot be read.

    #[cfg(unix)]
    #[tokio::test]
    async fn test_scan_and_register_permission_denied() {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;

        // Skip if running as root
        let uid_output = std::process::Command::new("id").arg("-u").output();
        if let Ok(out) = uid_output {
            if String::from_utf8_lossy(&out.stdout).trim() == "0" {
                return;
            }
        }

        let outer = tempfile::tempdir().unwrap();
        let restricted = outer.path().join("no_read");
        fs::create_dir(&restricted).unwrap();

        let mut perms = fs::metadata(&restricted).unwrap().permissions();
        perms.set_mode(0o000);
        fs::set_permissions(&restricted, perms).unwrap();

        let manager = default_manager();
        let result = manager.scan_and_register(&restricted).await;

        // Restore permissions
        let mut restore = fs::metadata(&restricted).unwrap().permissions();
        restore.set_mode(0o755);
        fs::set_permissions(&restricted, restore).unwrap();

        assert!(
            result.is_err(),
            "scan_and_register should fail on unreadable directory"
        );
    }

    // ── load_onnx_model – empty device_ids list avoids ORT call ──────────────
    // Setting device_ids = Some(vec![]) causes the Some branch (lines 173-174)
    // to be taken, but the for loop body (line 181+, which calls ORT) never
    // executes.  This exercises lines 173-174, 179, 189, 192-193, 195-196.

    #[tokio::test]
    async fn test_load_onnx_model_empty_device_ids_skips_ort() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty_devids.onnx");
        std::fs::write(&path, b"onnx placeholder").unwrap();

        let mut config = Config::default();
        // Empty vec → Some branch is taken (line 173-174) but loop never executes
        config.device.device_ids = Some(vec![]);
        let manager = ModelManager::new(&config, None);

        manager
            .register_model_from_path(&path, Some("empty-devids-onnx".to_string()))
            .await
            .unwrap();

        // With an empty device list the ONNX loader is never called, so this
        // should succeed and the model should be marked as loaded in the registry.
        let result = manager.load_onnx_model("empty-devids-onnx").await;
        assert!(
            result.is_ok(),
            "load_onnx_model with empty device_ids should succeed: {:?}",
            result.err()
        );

        // Registry should reflect the model as loaded (lines 192-193 ran)
        let stats = manager.get_registry_stats();
        assert_eq!(
            stats["loaded_models"], 1,
            "model should be marked as loaded in registry"
        );
    }

    // ── infer_onnx – model in loaded_onnx_models but empty vec (line 256-257) ─
    // Re-confirm that inserting an empty vec triggers the "No model instances"
    // error path at line 256-257 (different manager instance for isolation).

    #[tokio::test]
    async fn test_infer_onnx_empty_loaded_vec_no_ort() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("infer_empty.onnx");
        std::fs::write(&path, b"placeholder").unwrap();

        let manager = default_manager();
        manager
            .register_model_from_path(&path, Some("infer-empty-onnx".to_string()))
            .await
            .unwrap();

        // Simulate a state where the model was "loaded" but with no instances
        manager
            .loaded_onnx_models
            .insert("infer-empty-onnx".to_string(), vec![]);

        let result = manager
            .infer_onnx("infer-empty-onnx", &serde_json::json!({"x": 0}))
            .await;
        assert!(result.is_err(), "empty loaded_onnx_models vec should fail");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("No model instances") || err.contains("loaded"),
            "error should mention empty model instances: {}",
            err
        );
    }

    // ── initialize_default_models: auto-load ONNX with empty device_ids ──────
    // With device_ids = Some(vec![]) the auto-load path for an ONNX model
    // exercises lines 420-422 (get_model, format check, load call) and since
    // load_onnx_model succeeds (no ORT call), the warn branch (line 423) is
    // NOT taken.  Lines 420-422 are covered.

    #[tokio::test]
    async fn test_initialize_default_models_auto_load_onnx_empty_device_ids() {
        let dir = tempfile::tempdir().unwrap();
        let onnx_path = dir.path().join("autoload_edev.onnx");
        std::fs::write(&onnx_path, b"onnx bytes").unwrap();

        let mut config = Config::default();
        // Use a separate temp dir for cache so no accidental scan conflicts
        config.models.cache_dir = tempfile::tempdir().unwrap().keep();
        config.models.auto_load = vec!["autoload-edev".to_string()];
        config.device.device_ids = Some(vec![]); // avoids ORT call

        let manager = ModelManager::new(&config, None);
        // Register so it appears in registry as ONNX format
        manager
            .register_model_from_path(&onnx_path, Some("autoload-edev".to_string()))
            .await
            .unwrap();

        // initialize_default_models will try auto-loading "autoload-edev".
        // It's not in legacy models → else branch → ONNX block →
        // load_onnx_model("autoload-edev") succeeds with empty device_ids.
        let result = manager.initialize_default_models().await;
        assert!(
            result.is_ok(),
            "initialize_default_models should succeed: {:?}",
            result.err()
        );

        // Confirm the model was marked as loaded (loaded_models count increases)
        let stats = manager.get_registry_stats();
        assert_eq!(
            stats["loaded_models"], 1,
            "auto-loaded ONNX model should be marked loaded"
        );
    }
}
