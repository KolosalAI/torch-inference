use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use log::{info, warn, error};
use anyhow::{Result, Context};
use chrono::{DateTime, Utc};

use crate::error::InferenceError;

/// Model format types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    PyTorch,
    ONNX,
    Candle,
    SafeTensors,
}

impl ModelFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "pt" | "pth" => Some(ModelFormat::PyTorch),
            "onnx" => Some(ModelFormat::ONNX),
            "safetensors" => Some(ModelFormat::SafeTensors),
            _ => None,
        }
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub format: ModelFormat,
    pub path: PathBuf,
    pub version: String,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub input_schema: Option<serde_json::Value>,
    pub output_schema: Option<serde_json::Value>,
    pub preprocessing: Option<PreprocessingConfig>,
    pub postprocessing: Option<PostprocessingConfig>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub file_size: u64,
    pub checksum: Option<String>,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub image: Option<ImagePreprocessing>,
    pub audio: Option<AudioPreprocessing>,
    pub text: Option<TextPreprocessing>,
    pub custom: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePreprocessing {
    pub resize: Option<(u32, u32)>,
    pub normalize: Option<NormalizationParams>,
    pub to_tensor: bool,
    pub color_space: Option<String>, // RGB, BGR, Grayscale
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPreprocessing {
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub normalize: bool,
    pub to_mel: Option<MelSpectrogramParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MelSpectrogramParams {
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPreprocessing {
    pub tokenizer: Option<String>,
    pub max_length: Option<usize>,
    pub padding: bool,
    pub truncation: bool,
}

/// Postprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessingConfig {
    pub output_type: OutputType,
    pub threshold: Option<f32>,
    pub top_k: Option<usize>,
    pub custom: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    Classification,
    Regression,
    Segmentation,
    Detection,
    Generation,
    Raw,
}

/// Model registry entry
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub metadata: ModelMetadata,
    pub is_loaded: bool,
    pub load_count: u64,
    pub last_used: Option<DateTime<Utc>>,
}

/// Model Registry
pub struct ModelRegistry {
    models: DashMap<String, ModelEntry>,
    base_path: PathBuf,
    config_cache: Arc<RwLock<HashMap<String, serde_json::Value>>>,
}

impl ModelRegistry {
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            models: DashMap::new(),
            base_path,
            config_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a model from a file path
    pub async fn register_from_path(&self, path: &Path, model_name: Option<String>) -> Result<String> {
        info!("Registering model from path: {:?}", path);

        if !path.exists() {
            return Err(anyhow::anyhow!("Model file does not exist: {:?}", path));
        }

        // Determine format from extension
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid file extension"))?;

        let format = ModelFormat::from_extension(extension)
            .ok_or_else(|| anyhow::anyhow!("Unsupported model format: {}", extension))?;

        // Generate model ID
        let model_id = model_name.clone().unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        // Get file metadata
        let metadata = tokio::fs::metadata(path).await
            .context("Failed to read file metadata")?;
        let file_size = metadata.len();

        // Look for config file
        let config_path = path.with_extension("json");
        let (preprocessing, postprocessing, input_schema, output_schema) = 
            if config_path.exists() {
                self.load_model_config(&config_path).await?
            } else {
                (None, None, None, None)
            };

        // Create metadata
        let model_metadata = ModelMetadata {
            id: model_id.clone(),
            name: model_name.unwrap_or_else(|| model_id.clone()),
            format,
            path: path.to_path_buf(),
            version: "1.0.0".to_string(),
            description: None,
            tags: vec![],
            input_schema,
            output_schema,
            preprocessing,
            postprocessing,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            file_size,
            checksum: None,
        };

        // Create entry
        let entry = ModelEntry {
            metadata: model_metadata,
            is_loaded: false,
            load_count: 0,
            last_used: None,
        };

        // Store in registry
        self.models.insert(model_id.clone(), entry);

        info!("Model registered successfully: {}", model_id);
        Ok(model_id)
    }

    /// Load model configuration from JSON file
    async fn load_model_config(&self, config_path: &Path) -> Result<(
        Option<PreprocessingConfig>,
        Option<PostprocessingConfig>,
        Option<serde_json::Value>,
        Option<serde_json::Value>,
    )> {
        let content = tokio::fs::read_to_string(config_path).await
            .context("Failed to read config file")?;

        let config: serde_json::Value = serde_json::from_str(&content)
            .context("Failed to parse config JSON")?;

        let preprocessing = config.get("preprocessing")
            .and_then(|p| serde_json::from_value(p.clone()).ok());

        let postprocessing = config.get("postprocessing")
            .and_then(|p| serde_json::from_value(p.clone()).ok());

        let input_schema = config.get("input_schema").cloned();
        let output_schema = config.get("output_schema").cloned();

        Ok((preprocessing, postprocessing, input_schema, output_schema))
    }

    /// Scan directory and register all models
    pub async fn scan_directory(&self, dir_path: &Path) -> Result<Vec<String>> {
        info!("Scanning directory for models: {:?}", dir_path);

        if !dir_path.exists() || !dir_path.is_dir() {
            return Err(anyhow::anyhow!("Invalid directory path: {:?}", dir_path));
        }

        let mut registered = Vec::new();
        let mut entries = tokio::fs::read_dir(dir_path).await
            .context("Failed to read directory")?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if ModelFormat::from_extension(ext).is_some() {
                        match self.register_from_path(&path, None).await {
                            Ok(model_id) => {
                                info!("Registered model: {}", model_id);
                                registered.push(model_id);
                            }
                            Err(e) => {
                                warn!("Failed to register model {:?}: {}", path, e);
                            }
                        }
                    }
                }
            }
        }

        info!("Registered {} models from directory", registered.len());
        Ok(registered)
    }

    /// Get model metadata
    pub fn get_model(&self, model_id: &str) -> Result<ModelMetadata> {
        self.models
            .get(model_id)
            .map(|entry| entry.metadata.clone())
            .ok_or_else(|| InferenceError::ModelNotFound(model_id.to_string()).into())
    }

    /// Update model metadata
    pub fn update_model(&self, model_id: &str, update_fn: impl FnOnce(&mut ModelEntry)) -> Result<()> {
        self.models
            .get_mut(model_id)
            .map(|mut entry| update_fn(&mut entry))
            .ok_or_else(|| InferenceError::ModelNotFound(model_id.to_string()).into())
    }

    /// Mark model as loaded
    pub fn mark_loaded(&self, model_id: &str) -> Result<()> {
        self.update_model(model_id, |entry| {
            entry.is_loaded = true;
            entry.load_count += 1;
        })
    }

    /// Mark model as used
    pub fn mark_used(&self, model_id: &str) -> Result<()> {
        self.update_model(model_id, |entry| {
            entry.last_used = Some(Utc::now());
        })
    }

    /// List all registered models
    pub fn list_models(&self) -> Vec<ModelMetadata> {
        self.models
            .iter()
            .map(|entry| entry.metadata.clone())
            .collect()
    }

    /// List models by format
    pub fn list_by_format(&self, format: ModelFormat) -> Vec<ModelMetadata> {
        self.models
            .iter()
            .filter(|entry| entry.metadata.format == format)
            .map(|entry| entry.metadata.clone())
            .collect()
    }

    /// Search models by tags
    pub fn search_by_tags(&self, tags: &[String]) -> Vec<ModelMetadata> {
        self.models
            .iter()
            .filter(|entry| {
                tags.iter().any(|tag| entry.metadata.tags.contains(tag))
            })
            .map(|entry| entry.metadata.clone())
            .collect()
    }

    /// Remove model from registry
    pub fn unregister(&self, model_id: &str) -> Result<()> {
        self.models
            .remove(model_id)
            .map(|_| ())
            .ok_or_else(|| InferenceError::ModelNotFound(model_id.to_string()).into())
    }

    /// Get registry statistics
    pub fn get_stats(&self) -> RegistryStats {
        let total_models = self.models.len();
        let loaded_models = self.models.iter().filter(|e| e.is_loaded).count();
        
        let mut format_counts = HashMap::new();
        for entry in self.models.iter() {
            *format_counts.entry(entry.metadata.format.clone()).or_insert(0) += 1;
        }

        let total_size: u64 = self.models.iter()
            .map(|e| e.metadata.file_size)
            .sum();

        RegistryStats {
            total_models,
            loaded_models,
            format_counts,
            total_size_bytes: total_size,
        }
    }

    /// Export registry to JSON
    pub fn export_registry(&self) -> serde_json::Value {
        let models: Vec<_> = self.models.iter()
            .map(|entry| serde_json::json!({
                "id": entry.metadata.id,
                "name": entry.metadata.name,
                "format": entry.metadata.format,
                "path": entry.metadata.path,
                "version": entry.metadata.version,
                "is_loaded": entry.is_loaded,
                "load_count": entry.load_count,
                "last_used": entry.last_used,
            }))
            .collect();

        serde_json::json!({
            "models": models,
            "total": models.len(),
            "exported_at": Utc::now(),
        })
    }
}

#[derive(Debug, Serialize)]
pub struct RegistryStats {
    pub total_models: usize,
    pub loaded_models: usize,
    pub format_counts: HashMap<ModelFormat, usize>,
    pub total_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── ModelFormat ──────────────────────────────────────────────────────────

    #[test]
    fn test_model_format_from_extension_pytorch() {
        assert_eq!(ModelFormat::from_extension("pt"), Some(ModelFormat::PyTorch));
        assert_eq!(ModelFormat::from_extension("pth"), Some(ModelFormat::PyTorch));
        assert_eq!(ModelFormat::from_extension("PT"), Some(ModelFormat::PyTorch));
        assert_eq!(ModelFormat::from_extension("PTH"), Some(ModelFormat::PyTorch));
    }

    #[test]
    fn test_model_format_from_extension_onnx() {
        assert_eq!(ModelFormat::from_extension("onnx"), Some(ModelFormat::ONNX));
        assert_eq!(ModelFormat::from_extension("ONNX"), Some(ModelFormat::ONNX));
    }

    #[test]
    fn test_model_format_from_extension_safetensors() {
        assert_eq!(ModelFormat::from_extension("safetensors"), Some(ModelFormat::SafeTensors));
        assert_eq!(ModelFormat::from_extension("SAFETENSORS"), Some(ModelFormat::SafeTensors));
    }

    #[test]
    fn test_model_format_from_extension_unknown() {
        assert_eq!(ModelFormat::from_extension("unknown"), None);
        assert_eq!(ModelFormat::from_extension("bin"), None);
        assert_eq!(ModelFormat::from_extension(""), None);
        assert_eq!(ModelFormat::from_extension("txt"), None);
    }

    #[test]
    fn test_model_format_equality() {
        assert_eq!(ModelFormat::PyTorch, ModelFormat::PyTorch);
        assert_ne!(ModelFormat::PyTorch, ModelFormat::ONNX);
        assert_ne!(ModelFormat::ONNX, ModelFormat::SafeTensors);
        assert_ne!(ModelFormat::SafeTensors, ModelFormat::Candle);
    }

    #[test]
    fn test_model_format_clone() {
        let fmt = ModelFormat::ONNX;
        let cloned = fmt.clone();
        assert_eq!(fmt, cloned);
    }

    #[test]
    fn test_model_format_debug() {
        let fmt = ModelFormat::PyTorch;
        let dbg = format!("{:?}", fmt);
        assert!(dbg.contains("PyTorch"));
    }

    #[test]
    fn test_model_format_serde_roundtrip() {
        let fmt = ModelFormat::ONNX;
        let json = serde_json::to_string(&fmt).unwrap();
        let decoded: ModelFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(fmt, decoded);
    }

    #[test]
    fn test_model_format_hash() {
        use std::collections::HashMap;
        let mut map: HashMap<ModelFormat, usize> = HashMap::new();
        map.insert(ModelFormat::PyTorch, 1);
        map.insert(ModelFormat::ONNX, 2);
        assert_eq!(map[&ModelFormat::PyTorch], 1);
        assert_eq!(map[&ModelFormat::ONNX], 2);
    }

    // ── ModelRegistry::new ───────────────────────────────────────────────────

    #[test]
    fn test_registry_new() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        // Empty registry should list no models
        assert!(registry.list_models().is_empty());
    }

    // ── get_model – model not found ──────────────────────────────────────────

    #[test]
    fn test_get_model_not_found() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let result = registry.get_model("nonexistent");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nonexistent"));
    }

    // ── unregister ───────────────────────────────────────────────────────────

    #[test]
    fn test_unregister_not_found() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let result = registry.unregister("ghost");
        assert!(result.is_err());
    }

    // ── mark_loaded / mark_used on missing model ─────────────────────────────

    #[test]
    fn test_mark_loaded_missing() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        assert!(registry.mark_loaded("nope").is_err());
    }

    #[test]
    fn test_mark_used_missing() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        assert!(registry.mark_used("nope").is_err());
    }

    // ── update_model on missing model ────────────────────────────────────────

    #[test]
    fn test_update_model_missing() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let result = registry.update_model("ghost", |_e| {});
        assert!(result.is_err());
    }

    // ── get_stats on empty registry ──────────────────────────────────────────

    #[test]
    fn test_get_stats_empty() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let stats = registry.get_stats();
        assert_eq!(stats.total_models, 0);
        assert_eq!(stats.loaded_models, 0);
        assert_eq!(stats.total_size_bytes, 0);
        assert!(stats.format_counts.is_empty());
    }

    // ── export_registry on empty registry ────────────────────────────────────

    #[test]
    fn test_export_registry_empty() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let exported = registry.export_registry();
        assert_eq!(exported["total"], 0);
        assert!(exported["models"].as_array().unwrap().is_empty());
    }

    // ── list_by_format / search_by_tags on empty registry ────────────────────

    #[test]
    fn test_list_by_format_empty() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let results = registry.list_by_format(ModelFormat::ONNX);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_by_tags_empty() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let results = registry.search_by_tags(&["audio".to_string()]);
        assert!(results.is_empty());
    }

    // ── register_from_path ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_register_from_path_nonexistent() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let result = registry.register_from_path(
            Path::new("/nonexistent/path/model.onnx"),
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_register_from_path_unsupported_extension() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"dummy").unwrap();
        // Rename to give it an unsupported extension
        let path = tmp.path().with_extension("bin");
        std::fs::copy(tmp.path(), &path).unwrap();

        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let result = registry.register_from_path(&path, None).await;
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_register_from_path_no_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("modelfile");
        std::fs::write(&path, b"dummy").unwrap();

        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let result = registry.register_from_path(&path, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_register_from_path_success_onnx() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mymodel.onnx");
        std::fs::write(&path, b"dummy onnx data").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let result = registry.register_from_path(&path, None).await;
        assert!(result.is_ok());
        let model_id = result.unwrap();
        assert_eq!(model_id, "mymodel");

        // Model should now be gettable
        let meta = registry.get_model(&model_id).unwrap();
        assert_eq!(meta.id, "mymodel");
        assert_eq!(meta.format, ModelFormat::ONNX);
        assert_eq!(meta.file_size, 15);
    }

    #[tokio::test]
    async fn test_register_from_path_with_explicit_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.pt");
        std::fs::write(&path, b"fake pytorch model").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let result = registry
            .register_from_path(&path, Some("my-custom-name".to_string()))
            .await;
        assert!(result.is_ok());
        let model_id = result.unwrap();
        assert_eq!(model_id, "my-custom-name");

        let meta = registry.get_model("my-custom-name").unwrap();
        assert_eq!(meta.format, ModelFormat::PyTorch);
        assert_eq!(meta.name, "my-custom-name");
    }

    // ── mark_loaded / mark_used on registered model ──────────────────────────

    #[tokio::test]
    async fn test_mark_loaded_success() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.onnx");
        std::fs::write(&path, b"data").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let model_id = registry.register_from_path(&path, None).await.unwrap();

        // Initially not loaded
        {
            let entry = registry.models.get(&model_id).unwrap();
            assert!(!entry.is_loaded);
            assert_eq!(entry.load_count, 0);
        }

        registry.mark_loaded(&model_id).unwrap();

        {
            let entry = registry.models.get(&model_id).unwrap();
            assert!(entry.is_loaded);
            assert_eq!(entry.load_count, 1);
        }
    }

    #[tokio::test]
    async fn test_mark_used_success() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.onnx");
        std::fs::write(&path, b"data").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let model_id = registry.register_from_path(&path, None).await.unwrap();

        {
            let entry = registry.models.get(&model_id).unwrap();
            assert!(entry.last_used.is_none());
        }

        registry.mark_used(&model_id).unwrap();

        {
            let entry = registry.models.get(&model_id).unwrap();
            assert!(entry.last_used.is_some());
        }
    }

    // ── list_models / list_by_format after registration ──────────────────────

    #[tokio::test]
    async fn test_list_models_after_registration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.onnx");
        std::fs::write(&path, b"data").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        registry.register_from_path(&path, None).await.unwrap();

        let models = registry.list_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "x");
    }

    #[tokio::test]
    async fn test_list_by_format_after_registration() {
        let dir = tempfile::tempdir().unwrap();
        let onnx_path = dir.path().join("m1.onnx");
        let pt_path = dir.path().join("m2.pt");
        std::fs::write(&onnx_path, b"onnx").unwrap();
        std::fs::write(&pt_path, b"pytorch").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        registry.register_from_path(&onnx_path, None).await.unwrap();
        registry.register_from_path(&pt_path, None).await.unwrap();

        let onnx_models = registry.list_by_format(ModelFormat::ONNX);
        assert_eq!(onnx_models.len(), 1);
        assert_eq!(onnx_models[0].format, ModelFormat::ONNX);

        let pt_models = registry.list_by_format(ModelFormat::PyTorch);
        assert_eq!(pt_models.len(), 1);
        assert_eq!(pt_models[0].format, ModelFormat::PyTorch);
    }

    // ── search_by_tags ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_search_by_tags() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tagged.onnx");
        std::fs::write(&path, b"data").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let model_id = registry.register_from_path(&path, None).await.unwrap();

        // Manually add tags via update_model
        registry
            .update_model(&model_id, |entry| {
                entry.metadata.tags = vec!["audio".to_string(), "tts".to_string()];
            })
            .unwrap();

        let found = registry.search_by_tags(&["audio".to_string()]);
        assert_eq!(found.len(), 1);

        let not_found = registry.search_by_tags(&["vision".to_string()]);
        assert!(not_found.is_empty());
    }

    // ── unregister ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_unregister_success() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("del.onnx");
        std::fs::write(&path, b"data").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let model_id = registry.register_from_path(&path, None).await.unwrap();

        assert_eq!(registry.list_models().len(), 1);
        registry.unregister(&model_id).unwrap();
        assert!(registry.list_models().is_empty());
        assert!(registry.get_model(&model_id).is_err());
    }

    // ── get_stats after registration ──────────────────────────────────────────

    #[tokio::test]
    async fn test_get_stats_after_registration() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("s1.onnx");
        let p2 = dir.path().join("s2.pt");
        std::fs::write(&p1, b"12345").unwrap();
        std::fs::write(&p2, b"123").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let id1 = registry.register_from_path(&p1, None).await.unwrap();
        registry.register_from_path(&p2, None).await.unwrap();

        let stats = registry.get_stats();
        assert_eq!(stats.total_models, 2);
        assert_eq!(stats.loaded_models, 0);
        assert_eq!(stats.total_size_bytes, 5 + 3);
        assert_eq!(stats.format_counts[&ModelFormat::ONNX], 1);
        assert_eq!(stats.format_counts[&ModelFormat::PyTorch], 1);

        // Load one, check loaded_models increments
        registry.mark_loaded(&id1).unwrap();
        let stats2 = registry.get_stats();
        assert_eq!(stats2.loaded_models, 1);
    }

    // ── export_registry after registration ────────────────────────────────────

    #[tokio::test]
    async fn test_export_registry_with_models() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("exp.onnx");
        std::fs::write(&path, b"data").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        registry.register_from_path(&path, None).await.unwrap();

        let exported = registry.export_registry();
        assert_eq!(exported["total"], 1);
        let arr = exported["models"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], "exp");
    }

    // ── scan_directory ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_scan_directory_not_found() {
        let registry = ModelRegistry::new(PathBuf::from("/tmp/models"));
        let result = registry
            .scan_directory(Path::new("/no/such/directory"))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_scan_directory_empty() {
        let dir = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let registered = registry.scan_directory(dir.path()).await.unwrap();
        assert!(registered.is_empty());
    }

    #[tokio::test]
    async fn test_scan_directory_with_models() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.onnx"), b"onnx1").unwrap();
        std::fs::write(dir.path().join("b.pt"), b"pytorch1").unwrap();
        std::fs::write(dir.path().join("c.txt"), b"ignored").unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let mut registered = registry.scan_directory(dir.path()).await.unwrap();
        registered.sort();

        assert_eq!(registered.len(), 2);
        assert!(registered.contains(&"a".to_string()));
        assert!(registered.contains(&"b".to_string()));
    }

    // ── register_from_path with JSON config ──────────────────────────────────

    #[tokio::test]
    async fn test_register_with_config_json() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("cfg_model.onnx");
        std::fs::write(&model_path, b"onnx bytes").unwrap();

        // Write a companion config JSON
        let config_path = dir.path().join("cfg_model.json");
        let config_json = serde_json::json!({
            "input_schema": {"type": "array"},
            "output_schema": {"type": "number"}
        });
        std::fs::write(&config_path, config_json.to_string()).unwrap();

        let registry = ModelRegistry::new(dir.path().to_path_buf());
        let model_id = registry
            .register_from_path(&model_path, None)
            .await
            .unwrap();

        let meta = registry.get_model(&model_id).unwrap();
        assert!(meta.input_schema.is_some());
        assert!(meta.output_schema.is_some());
    }

    // ── ModelMetadata struct fields ───────────────────────────────────────────

    #[test]
    fn test_model_metadata_serde() {
        let now = Utc::now();
        let meta = ModelMetadata {
            id: "m1".to_string(),
            name: "Test Model".to_string(),
            format: ModelFormat::ONNX,
            path: PathBuf::from("/tmp/m1.onnx"),
            version: "1.0.0".to_string(),
            description: Some("A test".to_string()),
            tags: vec!["tag1".to_string()],
            input_schema: None,
            output_schema: None,
            preprocessing: None,
            postprocessing: None,
            created_at: now,
            updated_at: now,
            file_size: 100,
            checksum: Some("abc123".to_string()),
        };

        let json = serde_json::to_string(&meta).unwrap();
        let decoded: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "m1");
        assert_eq!(decoded.format, ModelFormat::ONNX);
        assert_eq!(decoded.file_size, 100);
        assert_eq!(decoded.checksum, Some("abc123".to_string()));
    }

    // ── RegistryStats serialization ───────────────────────────────────────────

    #[test]
    fn test_registry_stats_serialize() {
        let mut format_counts = HashMap::new();
        format_counts.insert(ModelFormat::ONNX, 3usize);
        let stats = RegistryStats {
            total_models: 3,
            loaded_models: 1,
            format_counts,
            total_size_bytes: 9999,
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("total_models"));
        assert!(json.contains("9999"));
    }

    // ── OutputType serde ──────────────────────────────────────────────────────

    #[test]
    fn test_output_type_serde() {
        for variant in [
            OutputType::Classification,
            OutputType::Regression,
            OutputType::Segmentation,
            OutputType::Detection,
            OutputType::Generation,
            OutputType::Raw,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let decoded: OutputType = serde_json::from_str(&json).unwrap();
            assert_eq!(
                format!("{:?}", variant),
                format!("{:?}", decoded)
            );
        }
    }

    // ── PostprocessingConfig / PreprocessingConfig structs ────────────────────

    #[test]
    fn test_postprocessing_config_serde() {
        let cfg = PostprocessingConfig {
            output_type: OutputType::Classification,
            threshold: Some(0.5),
            top_k: Some(5),
            custom: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: PostprocessingConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(decoded.output_type, OutputType::Classification));
        assert_eq!(decoded.threshold, Some(0.5));
        assert_eq!(decoded.top_k, Some(5));
    }

    #[test]
    fn test_preprocessing_config_serde() {
        let cfg = PreprocessingConfig {
            image: Some(ImagePreprocessing {
                resize: Some((224, 224)),
                normalize: Some(NormalizationParams {
                    mean: vec![0.485, 0.456, 0.406],
                    std: vec![0.229, 0.224, 0.225],
                }),
                to_tensor: true,
                color_space: Some("RGB".to_string()),
            }),
            audio: Some(AudioPreprocessing {
                sample_rate: Some(22050),
                channels: Some(1),
                normalize: true,
                to_mel: Some(MelSpectrogramParams {
                    n_fft: 1024,
                    hop_length: 256,
                    n_mels: 80,
                }),
            }),
            text: Some(TextPreprocessing {
                tokenizer: Some("bert-base".to_string()),
                max_length: Some(512),
                padding: true,
                truncation: true,
            }),
            custom: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: PreprocessingConfig = serde_json::from_str(&json).unwrap();
        let img = decoded.image.unwrap();
        assert_eq!(img.resize, Some((224, 224)));
        assert!(img.to_tensor);
        let audio = decoded.audio.unwrap();
        assert_eq!(audio.sample_rate, Some(22050));
        let text = decoded.text.unwrap();
        assert_eq!(text.max_length, Some(512));
        assert!(text.padding);
    }

    // ── ModelEntry struct ─────────────────────────────────────────────────────

    #[test]
    fn test_model_entry_debug() {
        let now = Utc::now();
        let entry = ModelEntry {
            metadata: ModelMetadata {
                id: "e1".to_string(),
                name: "entry1".to_string(),
                format: ModelFormat::SafeTensors,
                path: PathBuf::from("/tmp/e1.safetensors"),
                version: "1.0".to_string(),
                description: None,
                tags: vec![],
                input_schema: None,
                output_schema: None,
                preprocessing: None,
                postprocessing: None,
                created_at: now,
                updated_at: now,
                file_size: 0,
                checksum: None,
            },
            is_loaded: false,
            load_count: 0,
            last_used: None,
        };
        let dbg = format!("{:?}", entry);
        assert!(dbg.contains("e1"));
    }
}
