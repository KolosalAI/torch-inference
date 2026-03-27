/// Model Registry API Module
/// Provides REST API endpoints for model management
use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;
use tokio::io::AsyncWriteExt;

// ─── Global singletons ───────────────────────────────────────────────────────

/// The registry is built once and shared across all requests as a &'static
/// reference.  This eliminates the ~50+ HashMap insertions that were previously
/// executed on *every* API call.
static REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();

fn get_registry() -> &'static ModelRegistry {
    REGISTRY.get_or_init(ModelRegistry::build)
}

/// A single reqwest::Client with an internal connection pool, shared across
/// all download operations.  Creating one per request discards the connection
/// pool and forces a new TCP handshake on every download.
static HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn get_http_client() -> &'static reqwest::Client {
    HTTP_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .pool_max_idle_per_host(8)
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .expect("Failed to build global HTTP client")
    })
}

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
    /// Build the registry.  Only called once via `get_registry()`.
    fn build() -> Self {
        let mut models = HashMap::new();
        
        // Fish Speech v1.5
        models.insert("fish-speech-1.5".to_string(), ModelInfo {
            name: "Fish Speech v1.5".to_string(),
            score: 57.1,
            rank: 8,
            size: "~1 GB".to_string(),
            url: "https://huggingface.co/fishaudio/fish-speech-1.5".to_string(),
            architecture: "VQGAN + Llama".to_string(),
            voices: "Multi".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Requires git-lfs".to_string()),
            model_type: "tts".to_string(),
            task: "text-to-speech".to_string(),
        });
        
        // XTTS v2
        models.insert("xtts-v2".to_string(), ModelInfo {
            name: "XTTS v2".to_string(),
            score: 56.1,
            rank: 9,
            size: "~2 GB".to_string(),
            url: "https://huggingface.co/coqui/XTTS-v2".to_string(),
            architecture: "Transformer + HiFiGAN".to_string(),
            voices: "Voice Cloning".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Coqui TTS".to_string()),
            model_type: "tts".to_string(),
            task: "text-to-speech".to_string(),
        });
        
        // StyleTTS 2
        models.insert("styletts2".to_string(), ModelInfo {
            name: "StyleTTS 2".to_string(),
            score: 49.0,
            rank: 12,
            size: "~500 MB".to_string(),
            url: "https://huggingface.co/yl4579/StyleTTS2-LJSpeech".to_string(),
            architecture: "StyleTTS2".to_string(),
            voices: "Single".to_string(),
            quality: "Medium-High".to_string(),
            status: "Available".to_string(),
            note: Some("Expressive TTS with voice cloning".to_string()),
            model_type: "tts".to_string(),
            task: "text-to-speech".to_string(),
        });
        
        // MetaVoice
        models.insert("metavoice".to_string(), ModelInfo {
            name: "MetaVoice".to_string(),
            score: 49.1,
            rank: 11,
            size: "~1 GB".to_string(),
            url: "https://huggingface.co/metavoiceio/metavoice-1B-v0.1".to_string(),
            architecture: "Transformer".to_string(),
            voices: "Multi".to_string(),
            quality: "Medium-High".to_string(),
            status: "Available".to_string(),
            note: None,
            model_type: "tts".to_string(),
            task: "text-to-speech".to_string(),
        });
        
        // OpenVoice
        models.insert("openvoice".to_string(), ModelInfo {
            name: "OpenVoice".to_string(),
            score: 43.1,
            rank: 14,
            size: "~600 MB".to_string(),
            url: "https://huggingface.co/myshell-ai/OpenVoice".to_string(),
            architecture: "VITS".to_string(),
            voices: "Voice Cloning".to_string(),
            quality: "Medium".to_string(),
            status: "Available".to_string(),
            note: None,
            model_type: "tts".to_string(),
            task: "text-to-speech".to_string(),
        });
        
        // MeloTTS
        models.insert("melotts".to_string(), ModelInfo {
            name: "MeloTTS".to_string(),
            score: 41.3,
            rank: 15,
            size: "~200 MB".to_string(),
            url: "https://huggingface.co/myshell-ai/MeloTTS-English".to_string(),
            architecture: "VITS".to_string(),
            voices: "Multi-language".to_string(),
            quality: "Medium".to_string(),
            status: "Available".to_string(),
            note: None,
            model_type: "tts".to_string(),
            task: "text-to-speech".to_string(),
        });
        
        // Windows SAPI
        models.insert("windows-sapi".to_string(), ModelInfo {
            name: "Windows SAPI".to_string(),
            score: 0.0,
            rank: 0,
            size: "Built-in".to_string(),
            url: "Built-in".to_string(),
            architecture: "Neural TTS".to_string(),
            voices: "3".to_string(),
            quality: "High".to_string(),
            status: "Active".to_string(),
            note: Some("Currently used for real speech".to_string()),
            model_type: "tts".to_string(),
            task: "text-to-speech".to_string(),
        });
        
        // Piper Lessac voice
        models.insert("piper_lessac".to_string(), ModelInfo {
            name: "Piper (Lessac)".to_string(),
            score: 0.0,
            rank: 0,
            size: "60 MB".to_string(),
            url: "https://huggingface.co/rhasspy/piper-voices".to_string(),
            architecture: "VITS (ONNX)".to_string(),
            voices: "1".to_string(),
            quality: "Medium".to_string(),
            status: "Downloaded".to_string(),
            note: None,
            model_type: "tts".to_string(),
            task: "text-to-speech".to_string(),
        });
        
        // ============================================
        // IMAGE CLASSIFICATION MODELS
        // ============================================
        
        // ResNet-50
        models.insert("resnet50".to_string(), ModelInfo {
            name: "ResNet-50".to_string(),
            score: 0.0,
            rank: 100,
            size: "98 MB".to_string(),
            url: "https://download.pytorch.org/models/resnet50-0676ba61.pth".to_string(),
            architecture: "ResNet-50".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("ImageNet pre-trained, 1000 classes".to_string()),
            model_type: "image-classification".to_string(),
            task: "classification".to_string(),
        });
        
        // ResNet-18
        models.insert("resnet18".to_string(), ModelInfo {
            name: "ResNet-18".to_string(),
            score: 0.0,
            rank: 101,
            size: "45 MB".to_string(),
            url: "https://download.pytorch.org/models/resnet18-5c106cde.pth".to_string(),
            architecture: "ResNet-18".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("ImageNet pre-trained, lightweight".to_string()),
            model_type: "image-classification".to_string(),
            task: "classification".to_string(),
        });
        
        // MobileNet V2
        models.insert("mobilenet-v2".to_string(), ModelInfo {
            name: "MobileNet V2".to_string(),
            score: 0.0,
            rank: 102,
            size: "14 MB".to_string(),
            url: "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth".to_string(),
            architecture: "MobileNetV2".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Fast mobile inference, ImageNet".to_string()),
            model_type: "image-classification".to_string(),
            task: "classification".to_string(),
        });
        
        // EfficientNet B0
        models.insert("efficientnet-b0".to_string(), ModelInfo {
            name: "EfficientNet B0".to_string(),
            score: 0.0,
            rank: 103,
            size: "20 MB".to_string(),
            url: "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth".to_string(),
            architecture: "EfficientNet-B0".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Efficient architecture, ImageNet".to_string()),
            model_type: "image-classification".to_string(),
            task: "classification".to_string(),
        });
        
        // VGG16
        models.insert("vgg16".to_string(), ModelInfo {
            name: "VGG16".to_string(),
            score: 0.0,
            rank: 104,
            size: "528 MB".to_string(),
            url: "https://download.pytorch.org/models/vgg16-397923af.pth".to_string(),
            architecture: "VGG-16".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Classic CNN architecture, ImageNet".to_string()),
            model_type: "image-classification".to_string(),
            task: "classification".to_string(),
        });
        
        // DenseNet-121
        models.insert("densenet121".to_string(), ModelInfo {
            name: "DenseNet-121".to_string(),
            score: 0.0,
            rank: 105,
            size: "31 MB".to_string(),
            url: "https://download.pytorch.org/models/densenet121-a639ec97.pth".to_string(),
            architecture: "DenseNet-121".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Dense connections, ImageNet".to_string()),
            model_type: "image-classification".to_string(),
            task: "classification".to_string(),
        });
        
        // Inception V3
        models.insert("inception-v3".to_string(), ModelInfo {
            name: "Inception V3".to_string(),
            score: 0.0,
            rank: 106,
            size: "104 MB".to_string(),
            url: "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth".to_string(),
            architecture: "Inception-V3".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Multi-scale features, ImageNet".to_string()),
            model_type: "image-classification".to_string(),
            task: "classification".to_string(),
        });
        
        // ============================================
        // OBJECT DETECTION MODELS
        // ============================================
        
        // Faster R-CNN ResNet-50
        models.insert("faster-rcnn-resnet50".to_string(), ModelInfo {
            name: "Faster R-CNN ResNet-50".to_string(),
            score: 0.0,
            rank: 200,
            size: "160 MB".to_string(),
            url: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth".to_string(),
            architecture: "Faster R-CNN + ResNet-50".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Object detection, COCO dataset".to_string()),
            model_type: "object-detection".to_string(),
            task: "detection".to_string(),
        });
        
        // RetinaNet ResNet-50
        models.insert("retinanet-resnet50".to_string(), ModelInfo {
            name: "RetinaNet ResNet-50".to_string(),
            score: 0.0,
            rank: 201,
            size: "145 MB".to_string(),
            url: "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth".to_string(),
            architecture: "RetinaNet + ResNet-50".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Single-shot detector, COCO".to_string()),
            model_type: "object-detection".to_string(),
            task: "detection".to_string(),
        });
        
        // ============================================
        // SEGMENTATION MODELS
        // ============================================
        
        // DeepLabV3 ResNet-50
        models.insert("deeplabv3-resnet50".to_string(), ModelInfo {
            name: "DeepLabV3 ResNet-50".to_string(),
            score: 0.0,
            rank: 300,
            size: "163 MB".to_string(),
            url: "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth".to_string(),
            architecture: "DeepLabV3 + ResNet-50".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Semantic segmentation, COCO".to_string()),
            model_type: "segmentation".to_string(),
            task: "segmentation".to_string(),
        });
        
        // FCN ResNet-50
        models.insert("fcn-resnet50".to_string(), ModelInfo {
            name: "FCN ResNet-50".to_string(),
            score: 0.0,
            rank: 301,
            size: "126 MB".to_string(),
            url: "https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth".to_string(),
            architecture: "FCN + ResNet-50".to_string(),
            voices: "N/A".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Fully convolutional network, COCO".to_string()),
            model_type: "segmentation".to_string(),
            task: "segmentation".to_string(),
        });
        
        Self {
            version: "1.0".to_string(),
            updated: chrono::Utc::now().to_rfc3339(),
            models,
        }
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

// API Endpoints

/// GET /api/models - List all models
pub async fn list_models() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(get_registry()))
}

/// GET /api/models/{model_id} - Get specific model info
pub async fn get_model(path: web::Path<String>) -> Result<HttpResponse> {
    let model_id = path.into_inner();
    let registry = get_registry();

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
    let registry = get_registry();
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
    let registry = get_registry();

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

/// Download a single URL to disk using streaming, so the full file content
/// is never held in memory. Memory usage is bounded to the I/O read buffer
/// (~64 KB) regardless of file size.
async fn download_file_streaming(
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
) -> anyhow::Result<()> {
    use futures_util::TryStreamExt;
    use tokio_util::io::StreamReader;

    let response = client.get(url).send().await?;
    anyhow::ensure!(
        response.status().is_success(),
        "HTTP {} downloading {}",
        response.status(),
        url
    );

    let stream = response
        .bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
    let mut reader = StreamReader::new(stream);
    let mut file = tokio::fs::File::create(dest).await?;
    tokio::io::copy(&mut reader, &mut file).await?;
    file.flush().await?;
    Ok(())
}

async fn download_model_async(model_id: &str, model: &ModelInfo) -> anyhow::Result<()> {
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
    
    tokio::fs::create_dir_all(&cache_dir).await?;
    
    log::info!("Downloading {} ({}) from {}", model.name, model_type, model.url);
    
    if model.url == "Built-in" {
        log::info!("{} is built-in, no download needed", model.name);
        return Ok(());
    }
    
    // Check if URL is a direct file download (contains "resolve") or a repository
    if model.url.contains("resolve") {
        let extension = if model.url.ends_with(".pth") { "pth" }
                       else if model.url.ends_with(".onnx") { "onnx" }
                       else { "bin" };
        let filepath = cache_dir.join(format!("model.{}", extension));
        download_file_streaming(get_http_client(), &model.url, &filepath).await?;
        log::info!("Downloaded {} to {:?}", model.name, filepath);

        // Download config if available (small JSON — buffered path is fine)
        if let Some(config_url) = model.note.as_ref().and_then(|n| {
            if n.contains("http") { Some(n.as_str()) } else { None }
        }) {
            let config_response = get_http_client().get(config_url).send().await?;
            if config_response.status().is_success() {
                let config_bytes = config_response.bytes().await?;
                let config_path = cache_dir.join("config.json");
                tokio::fs::write(&config_path, &config_bytes).await?;
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

    // Reuse the global pooled client — avoids a new TCP handshake per repo file.
    let client = get_http_client();

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
    let registry = get_registry();
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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::method;

    #[tokio::test]
    async fn test_download_file_streaming_writes_content() {
        let server = MockServer::start().await;
        let content = b"hello streaming world 1234567890";

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(content.as_slice()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.bin");
        let client = reqwest::Client::new();

        download_file_streaming(&client, &server.uri(), &dest)
            .await
            .expect("streaming download should succeed");

        assert_eq!(std::fs::read(&dest).unwrap(), content);
    }

    #[tokio::test]
    async fn test_download_file_streaming_fails_on_http_error() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("missing.bin");
        let client = reqwest::Client::new();

        let result = download_file_streaming(&client, &server.uri(), &dest).await;
        assert!(result.is_err(), "should error on 404");
    }
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
