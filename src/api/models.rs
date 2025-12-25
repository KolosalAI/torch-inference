/// Model Registry API Module
/// Provides REST API endpoints for model management
use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub score: f32,
    pub rank: i32,
    pub size: String,
    pub url: String,
    pub architecture: String,
    pub voices: String,
    pub quality: String,
    pub status: String,
    pub note: Option<String>,
    #[serde(default)]
    pub model_type: String, // "tts", "image-classification", "neural-network"
    #[serde(default)]
    pub task: String, // "text-to-speech", "classification", "detection", etc.
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub version: String,
    pub updated: String,
    pub models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::from_file("model_registry.json").unwrap_or_else(|_| Self::default())
    }
    
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let registry: ModelRegistry = serde_json::from_str(&content)?;
        Ok(registry)
    }
    
    pub fn get_model(&self, model_id: &str) -> Option<&ModelInfo> {
        self.models.get(model_id)
    }
    
    pub fn list_models(&self) -> Vec<(&String, &ModelInfo)> {
        let mut models: Vec<_> = self.models.iter().collect();
        models.sort_by_key(|(_, info)| info.rank);
        models
    }
    
    pub fn get_downloaded_models(&self) -> Vec<(&String, &ModelInfo)> {
        self.models.iter()
            .filter(|(_, info)| info.status == "Downloaded" || info.status == "Active")
            .collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            updated: chrono::Utc::now().to_rfc3339(),
            models: HashMap::new(),
        }
    }
}

// API Endpoints

/// GET /api/models - List all models
pub async fn list_models() -> Result<HttpResponse> {
    let registry = ModelRegistry::new();
    Ok(HttpResponse::Ok().json(registry))
}

/// GET /api/models/{model_id} - Get specific model info
pub async fn get_model(path: web::Path<String>) -> Result<HttpResponse> {
    let model_id = path.into_inner();
    let registry = ModelRegistry::new();
    
    match registry.get_model(&model_id) {
        Some(model) => Ok(HttpResponse::Ok().json(model)),
        None => Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Model not found",
            "model_id": model_id
        })))
    }
}

/// GET /api/models/downloaded - List downloaded models
pub async fn list_downloaded_models() -> Result<HttpResponse> {
    let registry = ModelRegistry::new();
    let downloaded: HashMap<_, _> = registry.get_downloaded_models()
        .into_iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "count": downloaded.len(),
        "models": downloaded
    })))
}

#[derive(Debug, Deserialize)]
pub struct DownloadRequest {
    pub model_id: String,
}

/// POST /api/models/download - Download a model
pub async fn download_model(req: web::Json<DownloadRequest>) -> Result<HttpResponse> {
    let registry = ModelRegistry::new();
    
    match registry.get_model(&req.model_id) {
        Some(model) => {
            // Check if already downloaded
            if model.status == "Downloaded" || model.status == "Active" {
                return Ok(HttpResponse::Ok().json(serde_json::json!({
                    "status": "already_downloaded",
                    "model": model
                })));
            }
            
            // Spawn async download task
            let model_id = req.model_id.clone();
            let model_clone = model.clone();
            tokio::spawn(async move {
                if let Err(e) = download_model_async(&model_id, &model_clone).await {
                    log::error!("Failed to download {}: {}", model_id, e);
                }
            });
            
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "status": "download_initiated",
                "model": model,
                "message": format!("Download started for {}", model.name)
            })))
        },
        None => Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Model not found",
            "model_id": req.model_id
        })))
    }
}

async fn download_model_async(model_id: &str, model: &ModelInfo) -> anyhow::Result<()> {
    use tokio::fs;
    use tokio::io::AsyncWriteExt;
    
    // Determine cache directory based on model type
    let model_type = if model.model_type.is_empty() {
        "tts"  // default to tts for backward compatibility
    } else {
        model.model_type.as_str()
    };
    
    let cache_dir = match model_type {
        "tts" => std::path::Path::new("models/tts").join(model_id),
        "image-classification" => std::path::Path::new("models/classification").join(model_id),
        "object-detection" => std::path::Path::new("models/detection").join(model_id),
        "segmentation" => std::path::Path::new("models/segmentation").join(model_id),
        "neural-network" => std::path::Path::new("models/neural").join(model_id),
        _ => std::path::Path::new("models/other").join(model_id),
    };
    
    fs::create_dir_all(&cache_dir).await?;
    
    log::info!("Downloading {} ({}) from {}", model.name, model_type, model.url);
    
    if model.url == "Built-in" {
        log::info!("{} is built-in, no download needed", model.name);
        return Ok(());
    }
    
    // Check if URL is a direct file download (contains "resolve") or a repository
    if model.url.contains("resolve") {
        // Direct file download
        let client = reqwest::Client::new();
        let response = client.get(&model.url).send().await?;
        
        if !response.status().is_success() {
            anyhow::bail!("Failed to download: HTTP {}", response.status());
        }
        
        let extension = if model.url.ends_with(".pth") { "pth" } 
                       else if model.url.ends_with(".onnx") { "onnx" }
                       else { "bin" };
        
        let filename = format!("model.{}", extension);
        let filepath = cache_dir.join(&filename);
        
        let mut file = fs::File::create(&filepath).await?;
        let bytes = response.bytes().await?;
        file.write_all(&bytes).await?;
        
        log::info!("Downloaded {} ({} bytes) to {:?}", model.name, bytes.len(), filepath);
        
        // Download config if available  
        if let Some(config_url) = model.note.as_ref().and_then(|n| {
            if n.contains("http") { Some(n.as_str()) } else { None }
        }) {
            let config_response = client.get(config_url).send().await?;
            if config_response.status().is_success() {
                let config_bytes = config_response.bytes().await?;
                let config_path = cache_dir.join("config.json");
                let mut config_file = fs::File::create(&config_path).await?;
                config_file.write_all(&config_bytes).await?;
                log::info!("Downloaded config to {:?}", config_path);
            }
        }
    } else if model.url.starts_with("https://huggingface.co/") {
        // Repository download - extract repo_id
        let repo_id = model.url
            .trim_start_matches("https://huggingface.co/")
            .trim_end_matches('/');
        
        log::info!("Downloading repository {} from HuggingFace", repo_id);
        
        download_huggingface_repo(repo_id, &cache_dir).await?;
        
        log::info!("Successfully downloaded repository {} to {:?}", repo_id, cache_dir);
    } else {
        log::warn!("{} requires manual download from: {}", model.name, model.url);
    }
    
    Ok(())
}

async fn download_huggingface_repo(repo_id: &str, target_dir: &std::path::Path) -> anyhow::Result<()> {
    use tokio::fs;
    use tokio::io::AsyncWriteExt;
    
    let client = reqwest::Client::new();
    
    // Get list of files in the repository
    let files_url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
    
    log::info!("Fetching file list from {}", files_url);
    
    let response = client.get(&files_url).send().await?;
    
    if !response.status().is_success() {
        anyhow::bail!("Failed to fetch repository file list: HTTP {}", response.status());
    }
    
    let files: Vec<serde_json::Value> = response.json().await?;
    
    log::info!("Found {} items in repository", files.len());
    
    // Filter files to download (skip .git files, large binaries we don't need, etc.)
    let files_to_download: Vec<String> = files.iter()
        .filter_map(|f| {
            if f["type"] == "file" {
                f["path"].as_str().map(|s| s.to_string())
            } else {
                None
            }
        })
        .filter(|path| {
            // Skip unnecessary files
            !path.starts_with(".git") && 
            !path.ends_with(".md") &&
            !path.ends_with(".txt") &&
            !path.contains("test") &&
            !path.contains("example")
        })
        .collect();
    
    log::info!("Downloading {} files from repository", files_to_download.len());
    
    // Download each file
    for (idx, file_path) in files_to_download.iter().enumerate() {
        log::info!("Downloading {}/{}: {}", idx + 1, files_to_download.len(), file_path);
        
        let file_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, file_path
        );
        
        let local_path = target_dir.join(file_path);
        
        // Create parent directories
        if let Some(parent) = local_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        
        // Download file
        match client.get(&file_url).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    let bytes = response.bytes().await?;
                    let mut file = fs::File::create(&local_path).await?;
                    file.write_all(&bytes).await?;
                    log::info!("  ✓ Downloaded {} ({} bytes)", file_path, bytes.len());
                } else {
                    log::warn!("  ✗ Failed to download {}: HTTP {}", file_path, response.status());
                }
            }
            Err(e) => {
                log::warn!("  ✗ Error downloading {}: {}", file_path, e);
            }
        }
    }
    
    Ok(())
}

/// GET /api/models/comparison - Get model comparison
pub async fn get_model_comparison() -> Result<HttpResponse> {
    let registry = ModelRegistry::new();
    let models = registry.list_models();
    
    let comparison: Vec<_> = models.iter()
        .map(|(id, info)| {
            serde_json::json!({
                "id": id,
                "name": info.name,
                "score": info.score,
                "rank": info.rank,
                "size": info.size,
                "quality": info.quality,
                "status": info.status,
                "architecture": info.architecture
            })
        })
        .collect();
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "models": comparison,
        "total_count": comparison.len()
    })))
}

// Configure routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/models")
            .route("", web::get().to(list_models))
            .route("/downloaded", web::get().to(list_downloaded_models))
            .route("/comparison", web::get().to(get_model_comparison))
            .route("/download", web::post().to(download_model))
            .route("/{model_id}", web::get().to(get_model))
    );
}
