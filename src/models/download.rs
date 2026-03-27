use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{Result, Context, bail};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use dashmap::DashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadTask {
    pub id: String,
    pub model_name: String,
    pub source: ModelSource,
    pub status: DownloadStatus,
    pub progress: f32,
    pub total_size: Option<u64>,
    pub downloaded_size: u64,
    pub error: Option<String>,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSource {
    HuggingFace { 
        repo_id: String, 
        revision: Option<String>,
    },
    TorchHub { repo: String, model: String },
    Url { url: String },
    Local { path: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DownloadStatus {
    Pending,
    Downloading,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub source: ModelSource,
    pub local_path: PathBuf,
    pub size_bytes: u64,
    pub downloaded_at: chrono::DateTime<chrono::Utc>,
    pub metadata: ModelMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMetadata {
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub framework: Option<String>,
    pub task: Option<String>,
    pub license: Option<String>,
}

pub struct ModelDownloadManager {
    cache_dir: PathBuf,
    tasks: Arc<DashMap<String, DownloadTask>>,
    models: Arc<DashMap<String, ModelInfo>>,
    max_concurrent_downloads: usize,
}

impl ModelDownloadManager {
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        
        Ok(Self {
            cache_dir,
            tasks: Arc::new(DashMap::new()),
            models: Arc::new(DashMap::new()),
            max_concurrent_downloads: 3,
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        // Create cache directory
        fs::create_dir_all(&self.cache_dir).await
            .context("Failed to create cache directory")?;

        // Load existing models
        self.scan_cache().await?;

        Ok(())
    }

    async fn scan_cache(&self) -> Result<()> {
        if !self.cache_dir.exists() {
            return Ok(());
        }

        let mut entries = fs::read_dir(&self.cache_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_dir() {
                let model_name = path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Check for model metadata
                let metadata_path = path.join("metadata.json");
                let metadata = if metadata_path.exists() {
                    let content = fs::read_to_string(&metadata_path).await?;
                    serde_json::from_str(&content).unwrap_or_default()
                } else {
                    ModelMetadata::default()
                };

                // Calculate directory size
                let size = self.calculate_dir_size(&path).await?;

                let model_info = ModelInfo {
                    name: model_name.clone(),
                    source: ModelSource::Local {
                        path: path.to_string_lossy().to_string(),
                    },
                    local_path: path,
                    size_bytes: size,
                    downloaded_at: chrono::Utc::now(),
                    metadata,
                };

                self.models.insert(model_name, model_info);
            }
        }

        Ok(())
    }

    async fn calculate_dir_size(&self, path: &Path) -> Result<u64> {
        let mut total = 0u64;
        let mut entries = fs::read_dir(path).await?;

        while let Some(entry) = entries.next_entry().await? {
            let metadata = entry.metadata().await?;
            if metadata.is_file() {
                total += metadata.len();
            } else if metadata.is_dir() {
                total += Box::pin(self.calculate_dir_size(&entry.path())).await?;
            }
        }

        Ok(total)
    }

    pub async fn download_model(&self, model_name: String, source: ModelSource) -> Result<String> {
        let task_id = Uuid::new_v4().to_string();
        let task_id_clone = task_id.clone();
        
        let task = DownloadTask {
            id: task_id.clone(),
            model_name: model_name.clone(),
            source: source.clone(),
            status: DownloadStatus::Pending,
            progress: 0.0,
            total_size: None,
            downloaded_size: 0,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
        };

        self.tasks.insert(task_id.clone(), task.clone());

        // Spawn download task
        let manager = self.clone();
        tokio::spawn(async move {
            if let Err(e) = manager.execute_download(task_id_clone.clone()).await {
                manager.update_task_error(&task_id_clone, &e.to_string());
            }
        });

        Ok(task_id)
    }

    async fn execute_download(&self, task_id: String) -> Result<()> {
        let task = self.tasks.get(&task_id)
            .context("Task not found")?
            .clone();

        self.update_task_status(&task_id, DownloadStatus::Downloading);

        let model_dir = self.cache_dir.join(&task.model_name);
        fs::create_dir_all(&model_dir).await?;

        match &task.source {
            ModelSource::HuggingFace { repo_id, revision } => {
                self.download_from_huggingface(
                    &task_id, 
                    repo_id, 
                    revision.as_deref(),
                    &model_dir
                ).await?;
            }
            ModelSource::Url { url } => {
                self.download_from_url(&task_id, url, &model_dir).await?;
            }
            ModelSource::TorchHub { repo: _, model: _ } => {
                bail!("TorchHub downloads not yet implemented");
            }
            ModelSource::Local { path: _ } => {
                bail!("Local models don't need downloading");
            }
        }

        self.update_task_status(&task_id, DownloadStatus::Completed);

        // Save model info
        let size = self.calculate_dir_size(&model_dir).await?;
        let model_info = ModelInfo {
            name: task.model_name.clone(),
            source: task.source.clone(),
            local_path: model_dir,
            size_bytes: size,
            downloaded_at: chrono::Utc::now(),
            metadata: ModelMetadata::default(),
        };

        self.models.insert(task.model_name, model_info);

        Ok(())
    }

    async fn download_from_huggingface(
        &self,
        task_id: &str,
        repo_id: &str,
        revision: Option<&str>,
        target_dir: &Path,
    ) -> Result<()> {
        let revision = revision.unwrap_or("main");
        
        log::info!("Downloading from HuggingFace: {} (revision: {})", repo_id, revision);
        
        // HuggingFace API endpoint
        let api_url = format!("https://huggingface.co/api/models/{}", repo_id);
        
        let client = reqwest::Client::new();
        let response = client.get(&api_url)
            .send()
            .await
            .context("Failed to fetch model info")?;

        if !response.status().is_success() {
            bail!("Model not found: {}", repo_id);
        }

        let _model_info: serde_json::Value = response.json().await?;
        
        // Get list of files
        let files_url = format!(
            "https://huggingface.co/api/models/{}/tree/{}",
            repo_id, revision
        );
        
        let files_response = client.get(&files_url)
            .send()
            .await
            .context("Failed to fetch file list")?;

        let files: Vec<serde_json::Value> = files_response.json().await?;

        log::info!("Found {} files in repository", files.len());

        // Download all files (no filtering)
        let files_to_download: Vec<&str> = files.iter()
            .filter_map(|f| f["path"].as_str())
            .filter(|path| {
                // Skip directories
                !path.ends_with('/')
            })
            .collect();

        log::info!("Downloading {} files", files_to_download.len());

        // Download all files
        let total_files = files_to_download.len();
        for (idx, path) in files_to_download.iter().enumerate() {
            log::info!("Downloading file {}/{}: {}", idx + 1, total_files, path);
            
            let file_url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                repo_id, revision, path
            );

            let file_path = target_dir.join(path);
            
            // Create parent directories
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).await?;
            }

            // Download file
            let mut response = client.get(&file_url)
                .send()
                .await
                .context(format!("Failed to download file: {}", path))?;

            if !response.status().is_success() {
                log::warn!("Failed to download {}: {}", path, response.status());
                continue;
            }

            let total_size = response.content_length();
            let mut downloaded = 0u64;

            let mut file = fs::File::create(&file_path).await?;

            while let Some(chunk) = response.chunk().await? {
                file.write_all(&chunk).await?;
                downloaded += chunk.len() as u64;

                // Update progress
                let file_progress = idx as f32 / total_files as f32 * 100.0;
                if let Some(total) = total_size {
                    let chunk_progress = (downloaded as f32 / total as f32) * (1.0 / total_files as f32) * 100.0;
                    self.update_task_progress(task_id, file_progress + chunk_progress, downloaded, total_size);
                }
            }

            log::info!("Downloaded: {} ({} bytes)", path, downloaded);
        }

        Ok(())
    }

    async fn download_from_url(&self, task_id: &str, url: &str, target_dir: &Path) -> Result<()> {
        let client = reqwest::Client::new();
        let response = client.get(url).send().await?;

        if !response.status().is_success() {
            bail!("Failed to download from URL: {}", url);
        }

        let total_size = response.content_length();
        let mut downloaded = 0u64;

        // Extract filename from URL
        let filename = url.rsplit('/').next().unwrap_or("model.bin");
        let file_path = target_dir.join(filename);

        let mut file = fs::File::create(&file_path).await?;
        let mut stream = response.bytes_stream();

        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if let Some(total) = total_size {
                let progress = (downloaded as f32 / total as f32) * 100.0;
                self.update_task_progress(task_id, progress, downloaded, Some(total));
            }
        }

        Ok(())
    }

    fn update_task_status(&self, task_id: &str, status: DownloadStatus) {
        if let Some(mut task) = self.tasks.get_mut(task_id) {
            task.status = status.clone();
            if status == DownloadStatus::Completed {
                task.completed_at = Some(chrono::Utc::now());
            }
        }
    }

    fn update_task_progress(&self, task_id: &str, progress: f32, downloaded: u64, total: Option<u64>) {
        if let Some(mut task) = self.tasks.get_mut(task_id) {
            task.progress = progress;
            task.downloaded_size = downloaded;
            task.total_size = total;
        }
    }

    fn update_task_error(&self, task_id: &str, error: &str) {
        if let Some(mut task) = self.tasks.get_mut(task_id) {
            task.status = DownloadStatus::Failed;
            task.error = Some(error.to_string());
            task.completed_at = Some(chrono::Utc::now());
        }
    }

    pub fn get_task_status(&self, task_id: &str) -> Option<DownloadTask> {
        self.tasks.get(task_id).map(|t| t.clone())
    }

    pub fn list_tasks(&self) -> Vec<DownloadTask> {
        self.tasks.iter().map(|e| e.value().clone()).collect()
    }

    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.models.iter().map(|e| e.value().clone()).collect()
    }

    pub fn get_model(&self, name: &str) -> Option<ModelInfo> {
        self.models.get(name).map(|m| m.clone())
    }

    pub async fn delete_model(&self, name: &str) -> Result<()> {
        if let Some((_, model)) = self.models.remove(name) {
            if model.local_path.exists() {
                fs::remove_dir_all(&model.local_path).await
                    .context("Failed to delete model directory")?;
            }
            Ok(())
        } else {
            bail!("Model not found: {}", name)
        }
    }

    pub fn get_cache_info(&self) -> CacheInfo {
        let total_size: u64 = self.models.iter().map(|e| e.value().size_bytes).sum();
        let model_count = self.models.len();

        CacheInfo {
            cache_dir: self.cache_dir.clone(),
            total_size_bytes: total_size,
            model_count,
            models: self.list_models(),
        }
    }
}

impl Clone for ModelDownloadManager {
    fn clone(&self) -> Self {
        Self {
            cache_dir: self.cache_dir.clone(),
            tasks: Arc::clone(&self.tasks),
            models: Arc::clone(&self.models),
            max_concurrent_downloads: self.max_concurrent_downloads,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo {
    pub cache_dir: PathBuf,
    pub total_size_bytes: u64,
    pub model_count: usize,
    pub models: Vec<ModelInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== DownloadStatus Tests =====

    #[test]
    fn test_download_status_equality() {
        assert_eq!(DownloadStatus::Pending, DownloadStatus::Pending);
        assert_eq!(DownloadStatus::Downloading, DownloadStatus::Downloading);
        assert_eq!(DownloadStatus::Completed, DownloadStatus::Completed);
        assert_eq!(DownloadStatus::Failed, DownloadStatus::Failed);
        assert_eq!(DownloadStatus::Cancelled, DownloadStatus::Cancelled);
    }

    #[test]
    fn test_download_status_inequality() {
        assert_ne!(DownloadStatus::Pending, DownloadStatus::Completed);
        assert_ne!(DownloadStatus::Downloading, DownloadStatus::Failed);
        assert_ne!(DownloadStatus::Completed, DownloadStatus::Cancelled);
        assert_ne!(DownloadStatus::Failed, DownloadStatus::Pending);
        assert_ne!(DownloadStatus::Cancelled, DownloadStatus::Downloading);
    }

    #[test]
    fn test_download_status_clone() {
        let status = DownloadStatus::Downloading;
        let cloned = status.clone();
        assert_eq!(status, cloned);
    }

    #[test]
    fn test_download_status_debug() {
        let statuses = [
            DownloadStatus::Pending,
            DownloadStatus::Downloading,
            DownloadStatus::Completed,
            DownloadStatus::Failed,
            DownloadStatus::Cancelled,
        ];
        for status in &statuses {
            let debug_str = format!("{:?}", status);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_download_status_serde_roundtrip() {
        let statuses = [
            DownloadStatus::Pending,
            DownloadStatus::Downloading,
            DownloadStatus::Completed,
            DownloadStatus::Failed,
            DownloadStatus::Cancelled,
        ];
        for status in &statuses {
            let json = serde_json::to_string(status).unwrap();
            let deserialized: DownloadStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(*status, deserialized);
        }
    }

    // ===== ModelSource Tests =====

    #[test]
    fn test_model_source_huggingface() {
        let source = ModelSource::HuggingFace {
            repo_id: "bert-base-uncased".to_string(),
            revision: Some("main".to_string()),
        };
        match &source {
            ModelSource::HuggingFace { repo_id, revision } => {
                assert_eq!(repo_id, "bert-base-uncased");
                assert_eq!(revision.as_deref(), Some("main"));
            }
            _ => panic!("Expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_model_source_huggingface_no_revision() {
        let source = ModelSource::HuggingFace {
            repo_id: "gpt2".to_string(),
            revision: None,
        };
        match &source {
            ModelSource::HuggingFace { repo_id, revision } => {
                assert_eq!(repo_id, "gpt2");
                assert!(revision.is_none());
            }
            _ => panic!("Expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_model_source_torchhub() {
        let source = ModelSource::TorchHub {
            repo: "pytorch/vision".to_string(),
            model: "resnet50".to_string(),
        };
        match &source {
            ModelSource::TorchHub { repo, model } => {
                assert_eq!(repo, "pytorch/vision");
                assert_eq!(model, "resnet50");
            }
            _ => panic!("Expected TorchHub variant"),
        }
    }

    #[test]
    fn test_model_source_url() {
        let source = ModelSource::Url {
            url: "https://example.com/model.bin".to_string(),
        };
        match &source {
            ModelSource::Url { url } => {
                assert_eq!(url, "https://example.com/model.bin");
            }
            _ => panic!("Expected Url variant"),
        }
    }

    #[test]
    fn test_model_source_local() {
        let source = ModelSource::Local {
            path: "/tmp/my_model".to_string(),
        };
        match &source {
            ModelSource::Local { path } => {
                assert_eq!(path, "/tmp/my_model");
            }
            _ => panic!("Expected Local variant"),
        }
    }

    #[test]
    fn test_model_source_clone() {
        let source = ModelSource::HuggingFace {
            repo_id: "test-model".to_string(),
            revision: None,
        };
        let cloned = source.clone();
        match cloned {
            ModelSource::HuggingFace { repo_id, .. } => assert_eq!(repo_id, "test-model"),
            _ => panic!("Expected HuggingFace variant"),
        }
    }

    #[test]
    fn test_model_source_debug() {
        let source = ModelSource::Local { path: "/tmp/model".to_string() };
        let debug_str = format!("{:?}", source);
        assert!(debug_str.contains("Local"));
    }

    #[test]
    fn test_model_source_serde_roundtrip_huggingface() {
        let source = ModelSource::HuggingFace {
            repo_id: "bert-base".to_string(),
            revision: Some("v1".to_string()),
        };
        let json = serde_json::to_string(&source).unwrap();
        let deserialized: ModelSource = serde_json::from_str(&json).unwrap();
        match deserialized {
            ModelSource::HuggingFace { repo_id, revision } => {
                assert_eq!(repo_id, "bert-base");
                assert_eq!(revision.as_deref(), Some("v1"));
            }
            _ => panic!("Expected HuggingFace"),
        }
    }

    #[test]
    fn test_model_source_serde_roundtrip_url() {
        let source = ModelSource::Url { url: "http://example.com/m.bin".to_string() };
        let json = serde_json::to_string(&source).unwrap();
        let deserialized: ModelSource = serde_json::from_str(&json).unwrap();
        match deserialized {
            ModelSource::Url { url } => assert_eq!(url, "http://example.com/m.bin"),
            _ => panic!("Expected Url"),
        }
    }

    #[test]
    fn test_model_source_serde_roundtrip_local() {
        let source = ModelSource::Local { path: "/some/path".to_string() };
        let json = serde_json::to_string(&source).unwrap();
        let deserialized: ModelSource = serde_json::from_str(&json).unwrap();
        match deserialized {
            ModelSource::Local { path } => assert_eq!(path, "/some/path"),
            _ => panic!("Expected Local"),
        }
    }

    #[test]
    fn test_model_source_serde_roundtrip_torchhub() {
        let source = ModelSource::TorchHub {
            repo: "pytorch/hub".to_string(),
            model: "vgg16".to_string(),
        };
        let json = serde_json::to_string(&source).unwrap();
        let deserialized: ModelSource = serde_json::from_str(&json).unwrap();
        match deserialized {
            ModelSource::TorchHub { repo, model } => {
                assert_eq!(repo, "pytorch/hub");
                assert_eq!(model, "vgg16");
            }
            _ => panic!("Expected TorchHub"),
        }
    }

    // ===== ModelMetadata Tests =====

    #[test]
    fn test_model_metadata_default() {
        let meta = ModelMetadata::default();
        assert!(meta.description.is_none());
        assert!(meta.tags.is_empty());
        assert!(meta.framework.is_none());
        assert!(meta.task.is_none());
        assert!(meta.license.is_none());
    }

    #[test]
    fn test_model_metadata_with_values() {
        let meta = ModelMetadata {
            description: Some("A language model".to_string()),
            tags: vec!["nlp".to_string(), "bert".to_string()],
            framework: Some("pytorch".to_string()),
            task: Some("text-classification".to_string()),
            license: Some("MIT".to_string()),
        };
        assert_eq!(meta.description.as_deref(), Some("A language model"));
        assert_eq!(meta.tags.len(), 2);
        assert_eq!(meta.framework.as_deref(), Some("pytorch"));
        assert_eq!(meta.task.as_deref(), Some("text-classification"));
        assert_eq!(meta.license.as_deref(), Some("MIT"));
    }

    #[test]
    fn test_model_metadata_clone() {
        let meta = ModelMetadata {
            description: Some("test".to_string()),
            tags: vec!["tag1".to_string()],
            framework: None,
            task: None,
            license: None,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.description, meta.description);
        assert_eq!(cloned.tags, meta.tags);
    }

    #[test]
    fn test_model_metadata_serde_roundtrip() {
        let meta = ModelMetadata {
            description: Some("test model".to_string()),
            tags: vec!["vision".to_string()],
            framework: Some("onnx".to_string()),
            task: Some("image-classification".to_string()),
            license: Some("Apache-2.0".to_string()),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.description, meta.description);
        assert_eq!(deserialized.tags, meta.tags);
        assert_eq!(deserialized.framework, meta.framework);
        assert_eq!(deserialized.task, meta.task);
        assert_eq!(deserialized.license, meta.license);
    }

    #[test]
    fn test_model_metadata_serde_with_explicit_tags() {
        // ModelMetadata requires `tags` to be present when deserializing from JSON
        let json = r#"{"tags":[]}"#;
        let meta: ModelMetadata = serde_json::from_str(json).unwrap();
        assert!(meta.description.is_none());
        assert!(meta.tags.is_empty());
    }

    // ===== DownloadTask Tests =====

    fn make_download_task(status: DownloadStatus) -> DownloadTask {
        DownloadTask {
            id: "task-123".to_string(),
            model_name: "my_model".to_string(),
            source: ModelSource::HuggingFace {
                repo_id: "test/model".to_string(),
                revision: None,
            },
            status,
            progress: 0.0,
            total_size: None,
            downloaded_size: 0,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
        }
    }

    #[test]
    fn test_download_task_construction() {
        let task = make_download_task(DownloadStatus::Pending);
        assert_eq!(task.id, "task-123");
        assert_eq!(task.model_name, "my_model");
        assert_eq!(task.status, DownloadStatus::Pending);
        assert_eq!(task.progress, 0.0);
        assert!(task.total_size.is_none());
        assert_eq!(task.downloaded_size, 0);
        assert!(task.error.is_none());
        assert!(task.completed_at.is_none());
    }

    #[test]
    fn test_download_task_clone() {
        let task = make_download_task(DownloadStatus::Downloading);
        let cloned = task.clone();
        assert_eq!(task.id, cloned.id);
        assert_eq!(task.model_name, cloned.model_name);
        assert_eq!(task.status, cloned.status);
    }

    #[test]
    fn test_download_task_with_progress() {
        let mut task = make_download_task(DownloadStatus::Downloading);
        task.progress = 50.0;
        task.downloaded_size = 512;
        task.total_size = Some(1024);
        assert_eq!(task.progress, 50.0);
        assert_eq!(task.downloaded_size, 512);
        assert_eq!(task.total_size, Some(1024));
    }

    #[test]
    fn test_download_task_with_error() {
        let mut task = make_download_task(DownloadStatus::Failed);
        task.error = Some("connection refused".to_string());
        task.completed_at = Some(chrono::Utc::now());
        assert_eq!(task.error.as_deref(), Some("connection refused"));
        assert!(task.completed_at.is_some());
    }

    #[test]
    fn test_download_task_serde_roundtrip() {
        let task = DownloadTask {
            id: "roundtrip-task".to_string(),
            model_name: "roundtrip_model".to_string(),
            source: ModelSource::Url { url: "http://example.com/m.bin".to_string() },
            status: DownloadStatus::Completed,
            progress: 100.0,
            total_size: Some(2048),
            downloaded_size: 2048,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: Some(chrono::Utc::now()),
        };
        let json = serde_json::to_string(&task).unwrap();
        let deserialized: DownloadTask = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, task.id);
        assert_eq!(deserialized.model_name, task.model_name);
        assert_eq!(deserialized.status, DownloadStatus::Completed);
        assert_eq!(deserialized.progress, 100.0);
        assert_eq!(deserialized.total_size, Some(2048));
        assert_eq!(deserialized.downloaded_size, 2048);
    }

    #[test]
    fn test_download_task_debug() {
        let task = make_download_task(DownloadStatus::Pending);
        let debug_str = format!("{:?}", task);
        assert!(debug_str.contains("DownloadTask"));
        assert!(debug_str.contains("task-123"));
    }

    // ===== ModelInfo Tests =====

    #[test]
    fn test_model_info_construction() {
        let info = ModelInfo {
            name: "my_model".to_string(),
            source: ModelSource::Local { path: "/tmp/model".to_string() },
            local_path: PathBuf::from("/tmp/model"),
            size_bytes: 1024 * 1024,
            downloaded_at: chrono::Utc::now(),
            metadata: ModelMetadata::default(),
        };
        assert_eq!(info.name, "my_model");
        assert_eq!(info.size_bytes, 1024 * 1024);
        assert_eq!(info.local_path, PathBuf::from("/tmp/model"));
    }

    #[test]
    fn test_model_info_clone() {
        let info = ModelInfo {
            name: "cloned_model".to_string(),
            source: ModelSource::Local { path: "/tmp/model".to_string() },
            local_path: PathBuf::from("/tmp/model"),
            size_bytes: 512,
            downloaded_at: chrono::Utc::now(),
            metadata: ModelMetadata::default(),
        };
        let cloned = info.clone();
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.size_bytes, info.size_bytes);
    }

    #[test]
    fn test_model_info_debug() {
        let info = ModelInfo {
            name: "debug_model".to_string(),
            source: ModelSource::Local { path: "/tmp/model".to_string() },
            local_path: PathBuf::from("/tmp/model"),
            size_bytes: 0,
            downloaded_at: chrono::Utc::now(),
            metadata: ModelMetadata::default(),
        };
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("ModelInfo"));
    }

    #[test]
    fn test_model_info_serde_roundtrip() {
        let info = ModelInfo {
            name: "serde_model".to_string(),
            source: ModelSource::HuggingFace {
                repo_id: "org/model".to_string(),
                revision: Some("v2".to_string()),
            },
            local_path: PathBuf::from("/cache/org/model"),
            size_bytes: 4096,
            downloaded_at: chrono::Utc::now(),
            metadata: ModelMetadata {
                description: Some("serialized model".to_string()),
                tags: vec!["test".to_string()],
                framework: Some("pytorch".to_string()),
                task: None,
                license: None,
            },
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, info.name);
        assert_eq!(deserialized.size_bytes, info.size_bytes);
        assert_eq!(deserialized.local_path, info.local_path);
        assert_eq!(deserialized.metadata.description, info.metadata.description);
    }

    // ===== CacheInfo Tests =====

    #[test]
    fn test_cache_info_construction() {
        let cache_info = CacheInfo {
            cache_dir: PathBuf::from("/tmp/cache"),
            total_size_bytes: 8192,
            model_count: 2,
            models: vec![],
        };
        assert_eq!(cache_info.cache_dir, PathBuf::from("/tmp/cache"));
        assert_eq!(cache_info.total_size_bytes, 8192);
        assert_eq!(cache_info.model_count, 2);
        assert!(cache_info.models.is_empty());
    }

    #[test]
    fn test_cache_info_clone() {
        let cache_info = CacheInfo {
            cache_dir: PathBuf::from("/tmp/cache"),
            total_size_bytes: 100,
            model_count: 1,
            models: vec![],
        };
        let cloned = cache_info.clone();
        assert_eq!(cloned.cache_dir, cache_info.cache_dir);
        assert_eq!(cloned.total_size_bytes, cache_info.total_size_bytes);
    }

    #[test]
    fn test_cache_info_debug() {
        let cache_info = CacheInfo {
            cache_dir: PathBuf::from("/tmp"),
            total_size_bytes: 0,
            model_count: 0,
            models: vec![],
        };
        let debug_str = format!("{:?}", cache_info);
        assert!(debug_str.contains("CacheInfo"));
    }

    #[test]
    fn test_cache_info_serde_roundtrip() {
        let cache_info = CacheInfo {
            cache_dir: PathBuf::from("/models/cache"),
            total_size_bytes: 1024,
            model_count: 0,
            models: vec![],
        };
        let json = serde_json::to_string(&cache_info).unwrap();
        let deserialized: CacheInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.cache_dir, cache_info.cache_dir);
        assert_eq!(deserialized.total_size_bytes, cache_info.total_size_bytes);
        assert_eq!(deserialized.model_count, cache_info.model_count);
    }

    // ===== ModelDownloadManager Tests =====

    #[test]
    fn test_manager_new() {
        let manager = ModelDownloadManager::new("/tmp/test_cache").unwrap();
        assert_eq!(manager.cache_dir, PathBuf::from("/tmp/test_cache"));
        assert_eq!(manager.max_concurrent_downloads, 3);
    }

    #[test]
    fn test_manager_new_with_pathbuf() {
        let path = PathBuf::from("/tmp/model_cache");
        let manager = ModelDownloadManager::new(&path).unwrap();
        assert_eq!(manager.cache_dir, path);
    }

    #[test]
    fn test_manager_list_tasks_empty() {
        let manager = ModelDownloadManager::new("/tmp/cache_test_empty").unwrap();
        let tasks = manager.list_tasks();
        assert!(tasks.is_empty());
    }

    #[test]
    fn test_manager_list_models_empty() {
        let manager = ModelDownloadManager::new("/tmp/cache_test_models").unwrap();
        let models = manager.list_models();
        assert!(models.is_empty());
    }

    #[test]
    fn test_manager_get_task_not_found() {
        let manager = ModelDownloadManager::new("/tmp/cache_test_task").unwrap();
        let task = manager.get_task_status("nonexistent");
        assert!(task.is_none());
    }

    #[test]
    fn test_manager_get_model_not_found() {
        let manager = ModelDownloadManager::new("/tmp/cache_test_model_lookup").unwrap();
        let model = manager.get_model("nonexistent_model");
        assert!(model.is_none());
    }

    #[test]
    fn test_manager_get_cache_info_empty() {
        let manager = ModelDownloadManager::new("/tmp/cache_info_test").unwrap();
        let info = manager.get_cache_info();
        assert_eq!(info.cache_dir, PathBuf::from("/tmp/cache_info_test"));
        assert_eq!(info.total_size_bytes, 0);
        assert_eq!(info.model_count, 0);
        assert!(info.models.is_empty());
    }

    #[test]
    fn test_manager_clone_shares_state() {
        let manager = ModelDownloadManager::new("/tmp/cache_clone_test").unwrap();
        let cloned = manager.clone();
        assert_eq!(manager.cache_dir, cloned.cache_dir);
        assert_eq!(manager.max_concurrent_downloads, cloned.max_concurrent_downloads);
    }

    #[test]
    fn test_manager_update_task_status() {
        let manager = ModelDownloadManager::new("/tmp/cache_status_test").unwrap();

        // Insert a task manually
        let task = DownloadTask {
            id: "status-task".to_string(),
            model_name: "model".to_string(),
            source: ModelSource::Local { path: "/tmp".to_string() },
            status: DownloadStatus::Pending,
            progress: 0.0,
            total_size: None,
            downloaded_size: 0,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
        };
        manager.tasks.insert("status-task".to_string(), task);

        manager.update_task_status("status-task", DownloadStatus::Downloading);
        let updated = manager.get_task_status("status-task").unwrap();
        assert_eq!(updated.status, DownloadStatus::Downloading);
        assert!(updated.completed_at.is_none());
    }

    #[test]
    fn test_manager_update_task_status_completed_sets_completed_at() {
        let manager = ModelDownloadManager::new("/tmp/cache_completed_test").unwrap();

        let task = DownloadTask {
            id: "completed-task".to_string(),
            model_name: "model".to_string(),
            source: ModelSource::Local { path: "/tmp".to_string() },
            status: DownloadStatus::Downloading,
            progress: 99.0,
            total_size: Some(1024),
            downloaded_size: 1020,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
        };
        manager.tasks.insert("completed-task".to_string(), task);

        manager.update_task_status("completed-task", DownloadStatus::Completed);
        let updated = manager.get_task_status("completed-task").unwrap();
        assert_eq!(updated.status, DownloadStatus::Completed);
        assert!(updated.completed_at.is_some());
    }

    #[test]
    fn test_manager_update_task_progress() {
        let manager = ModelDownloadManager::new("/tmp/cache_progress_test").unwrap();

        let task = DownloadTask {
            id: "progress-task".to_string(),
            model_name: "model".to_string(),
            source: ModelSource::Local { path: "/tmp".to_string() },
            status: DownloadStatus::Downloading,
            progress: 0.0,
            total_size: None,
            downloaded_size: 0,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
        };
        manager.tasks.insert("progress-task".to_string(), task);

        manager.update_task_progress("progress-task", 75.5, 768, Some(1024));
        let updated = manager.get_task_status("progress-task").unwrap();
        assert_eq!(updated.progress, 75.5);
        assert_eq!(updated.downloaded_size, 768);
        assert_eq!(updated.total_size, Some(1024));
    }

    #[test]
    fn test_manager_update_task_error() {
        let manager = ModelDownloadManager::new("/tmp/cache_error_test").unwrap();

        let task = DownloadTask {
            id: "error-task".to_string(),
            model_name: "model".to_string(),
            source: ModelSource::Url { url: "http://bad.example/".to_string() },
            status: DownloadStatus::Downloading,
            progress: 10.0,
            total_size: None,
            downloaded_size: 100,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
        };
        manager.tasks.insert("error-task".to_string(), task);

        manager.update_task_error("error-task", "Connection timeout");
        let updated = manager.get_task_status("error-task").unwrap();
        assert_eq!(updated.status, DownloadStatus::Failed);
        assert_eq!(updated.error.as_deref(), Some("Connection timeout"));
        assert!(updated.completed_at.is_some());
    }

    #[test]
    fn test_manager_update_nonexistent_task_is_noop() {
        let manager = ModelDownloadManager::new("/tmp/cache_noop_test").unwrap();
        // These calls should not panic even if task doesn't exist
        manager.update_task_status("ghost", DownloadStatus::Failed);
        manager.update_task_progress("ghost", 50.0, 500, Some(1000));
        manager.update_task_error("ghost", "some error");
        assert!(manager.get_task_status("ghost").is_none());
    }

    #[test]
    fn test_manager_get_cache_info_with_models() {
        let manager = ModelDownloadManager::new("/tmp/cache_info_models_test").unwrap();

        let model1 = ModelInfo {
            name: "model_a".to_string(),
            source: ModelSource::Local { path: "/tmp/model_a".to_string() },
            local_path: PathBuf::from("/tmp/model_a"),
            size_bytes: 1000,
            downloaded_at: chrono::Utc::now(),
            metadata: ModelMetadata::default(),
        };
        let model2 = ModelInfo {
            name: "model_b".to_string(),
            source: ModelSource::Local { path: "/tmp/model_b".to_string() },
            local_path: PathBuf::from("/tmp/model_b"),
            size_bytes: 2000,
            downloaded_at: chrono::Utc::now(),
            metadata: ModelMetadata::default(),
        };

        manager.models.insert("model_a".to_string(), model1);
        manager.models.insert("model_b".to_string(), model2);

        let info = manager.get_cache_info();
        assert_eq!(info.model_count, 2);
        assert_eq!(info.total_size_bytes, 3000);
        assert_eq!(info.models.len(), 2);
    }

    #[test]
    fn test_manager_get_model_found() {
        let manager = ModelDownloadManager::new("/tmp/cache_get_model_test").unwrap();

        let model = ModelInfo {
            name: "found_model".to_string(),
            source: ModelSource::Local { path: "/tmp/found_model".to_string() },
            local_path: PathBuf::from("/tmp/found_model"),
            size_bytes: 500,
            downloaded_at: chrono::Utc::now(),
            metadata: ModelMetadata::default(),
        };
        manager.models.insert("found_model".to_string(), model);

        let retrieved = manager.get_model("found_model");
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.name, "found_model");
        assert_eq!(retrieved.size_bytes, 500);
    }

    #[test]
    fn test_manager_list_tasks_returns_all() {
        let manager = ModelDownloadManager::new("/tmp/cache_list_tasks_test").unwrap();

        for i in 0..3 {
            let task = DownloadTask {
                id: format!("task-{}", i),
                model_name: format!("model-{}", i),
                source: ModelSource::Local { path: "/tmp".to_string() },
                status: DownloadStatus::Pending,
                progress: 0.0,
                total_size: None,
                downloaded_size: 0,
                error: None,
                started_at: chrono::Utc::now(),
                completed_at: None,
            };
            manager.tasks.insert(format!("task-{}", i), task);
        }

        let tasks = manager.list_tasks();
        assert_eq!(tasks.len(), 3);
    }

    #[test]
    fn test_manager_list_models_returns_all() {
        let manager = ModelDownloadManager::new("/tmp/cache_list_models_test").unwrap();

        for i in 0..4 {
            let model = ModelInfo {
                name: format!("model_{}", i),
                source: ModelSource::Local { path: format!("/tmp/model_{}", i) },
                local_path: PathBuf::from(format!("/tmp/model_{}", i)),
                size_bytes: i as u64 * 100,
                downloaded_at: chrono::Utc::now(),
                metadata: ModelMetadata::default(),
            };
            manager.models.insert(format!("model_{}", i), model);
        }

        let models = manager.list_models();
        assert_eq!(models.len(), 4);
    }

    #[tokio::test]
    async fn test_manager_delete_model_not_found() {
        let manager = ModelDownloadManager::new("/tmp/cache_delete_test").unwrap();
        let result = manager.delete_model("nonexistent_model").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nonexistent_model"));
    }

    #[tokio::test]
    async fn test_manager_delete_model_without_local_dir() {
        let manager = ModelDownloadManager::new("/tmp/cache_delete_no_dir_test").unwrap();

        // Insert a model pointing to a path that doesn't actually exist on disk
        let model = ModelInfo {
            name: "ghost_model".to_string(),
            source: ModelSource::Local { path: "/tmp/nonexistent_ghost_model".to_string() },
            local_path: PathBuf::from("/tmp/nonexistent_ghost_model"),
            size_bytes: 0,
            downloaded_at: chrono::Utc::now(),
            metadata: ModelMetadata::default(),
        };
        manager.models.insert("ghost_model".to_string(), model);

        // Delete should succeed (path doesn't exist, so skip removal)
        let result = manager.delete_model("ghost_model").await;
        assert!(result.is_ok());
        // Model should be removed from manager
        assert!(manager.get_model("ghost_model").is_none());
    }

    #[tokio::test]
    async fn test_manager_initialize_creates_dir() {
        let tmp_dir = std::env::temp_dir().join("torch_inf_test_init");
        // Remove if already exists
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;

        let manager = ModelDownloadManager::new(&tmp_dir).unwrap();
        let result = manager.initialize().await;
        assert!(result.is_ok());
        assert!(tmp_dir.exists());

        // Clean up
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
    }

    #[tokio::test]
    async fn test_manager_download_model_returns_task_id() {
        let manager = ModelDownloadManager::new("/tmp/cache_dl_task_test").unwrap();

        // Use a Local source — the background task will fail, but we get a task ID immediately
        let task_id = manager.download_model(
            "test_model".to_string(),
            ModelSource::Local { path: "/nonexistent".to_string() },
        ).await.unwrap();

        assert!(!task_id.is_empty());

        // Task should exist in the manager right after scheduling
        let task = manager.get_task_status(&task_id);
        assert!(task.is_some());
        let task = task.unwrap();
        assert_eq!(task.model_name, "test_model");
        // Status is initially Pending (may transition in background)
        assert!(
            task.status == DownloadStatus::Pending || task.status == DownloadStatus::Downloading
        );
    }

    #[tokio::test]
    async fn test_manager_clone_shares_tasks_map() {
        let manager = ModelDownloadManager::new("/tmp/cache_clone_share_test").unwrap();
        let cloned = manager.clone();

        // Insert a task in manager, should be visible in cloned
        let task = DownloadTask {
            id: "shared-task".to_string(),
            model_name: "model".to_string(),
            source: ModelSource::Local { path: "/tmp".to_string() },
            status: DownloadStatus::Pending,
            progress: 0.0,
            total_size: None,
            downloaded_size: 0,
            error: None,
            started_at: chrono::Utc::now(),
            completed_at: None,
        };
        manager.tasks.insert("shared-task".to_string(), task);

        let found = cloned.get_task_status("shared-task");
        assert!(found.is_some());
    }
}
