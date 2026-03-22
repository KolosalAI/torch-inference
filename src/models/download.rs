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

    #[test]
    fn test_download_status() {
        assert_eq!(DownloadStatus::Pending, DownloadStatus::Pending);
        assert_ne!(DownloadStatus::Pending, DownloadStatus::Completed);
    }
}
