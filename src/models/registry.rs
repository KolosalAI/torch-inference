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

    #[test]
    fn test_model_format_from_extension() {
        assert_eq!(ModelFormat::from_extension("pt"), Some(ModelFormat::PyTorch));
        assert_eq!(ModelFormat::from_extension("pth"), Some(ModelFormat::PyTorch));
        assert_eq!(ModelFormat::from_extension("onnx"), Some(ModelFormat::ONNX));
        assert_eq!(ModelFormat::from_extension("safetensors"), Some(ModelFormat::SafeTensors));
        assert_eq!(ModelFormat::from_extension("unknown"), None);
    }
}
