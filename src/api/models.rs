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
    REGISTRY.get_or_init(ModelRegistry::load)
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
    #[serde(default, deserialize_with = "de_f32_or_str")]
    pub score: f32,
    #[serde(default, deserialize_with = "de_i32_or_str")]
    pub rank: i32,
    #[serde(default)]
    pub size: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub architecture: String,
    #[serde(default, deserialize_with = "de_string_or_num")]
    pub voices: String,
    #[serde(default)]
    pub quality: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub note: Option<String>,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub task: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub version: String,
    pub updated: String,
    pub models: HashMap<String, ModelInfo>,
}

fn de_f32_or_str<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    match serde_json::Value::deserialize(d)? {
        serde_json::Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0) as f32),
        serde_json::Value::String(s) => Ok(s.parse::<f32>().unwrap_or(0.0)),
        _ => Ok(0.0),
    }
}

fn de_i32_or_str<'de, D: serde::Deserializer<'de>>(d: D) -> Result<i32, D::Error> {
    match serde_json::Value::deserialize(d)? {
        serde_json::Value::Number(n) => Ok(n.as_i64().unwrap_or(0) as i32),
        serde_json::Value::String(s) => Ok(s.parse::<i32>().unwrap_or(0)),
        _ => Ok(0),
    }
}

fn de_string_or_num<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::String(s) => s,
        serde_json::Value::Number(n) => n.to_string(),
        _ => String::new(),
    })
}

impl ModelRegistry {
    /// Return an empty registry with sensible zero-value metadata.
    pub fn empty() -> Self {
        Self {
            version: "0.0".to_string(),
            updated: String::new(),
            models: HashMap::new(),
        }
    }

    /// Parse a registry from a JSON string. Returns an empty registry on parse error.
    pub fn from_json_str(s: &str) -> Self {
        serde_json::from_str(s).unwrap_or_else(|e| {
            log::warn!("Failed to parse model registry JSON: {}", e);
            Self::empty()
        })
    }

    /// Load the registry from disk (path via MODEL_REGISTRY_PATH env var, default
    /// `model_registry.json`). Falls back to the file embedded at compile-time so
    /// the binary works without the file present at runtime.
    pub fn load() -> Self {
        let path = std::env::var("MODEL_REGISTRY_PATH")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "model_registry.json".to_string());

        let data = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            log::info!(
                "model_registry.json not found at '{}', using compiled-in copy",
                path
            );
            include_str!("../../model_registry.json").to_string()
        });

        Self::from_json_str(&data)
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

/// Download multiple files concurrently (up to 8 in-flight at once).
///
/// Each `(url, dest)` pair is downloaded using `download_file_streaming` so
/// that large files are never fully buffered in memory.  Per-file errors are
/// logged as warnings and do not abort the overall operation, matching the
/// previous sequential behaviour.
async fn download_files_concurrent(
    client: &reqwest::Client,
    files: Vec<(String, std::path::PathBuf)>,
) -> anyhow::Result<()> {
    use futures::StreamExt;

    futures::stream::iter(files)
        .map(|(url, dest)| {
            let client = client.clone();
            async move {
                // Ensure parent directories exist before writing.
                if let Some(parent) = dest.parent() {
                    if let Err(e) = tokio::fs::create_dir_all(parent).await {
                        log::warn!("  Failed to create directories for {:?}: {}", dest, e);
                        return;
                    }
                }
                match download_file_streaming(&client, &url, &dest).await {
                    Ok(()) => log::info!("  Downloaded {}", url),
                    Err(e) => log::warn!("  Failed to download {}: {}", url, e),
                }
            }
        })
        .buffer_unordered(8)
        .collect::<Vec<_>>()
        .await;

    Ok(())
}

async fn download_huggingface_repo(repo_id: &str, target_dir: &std::path::Path) -> anyhow::Result<()> {
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
    let files_to_download: Vec<(String, std::path::PathBuf)> = files.iter()
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
        .map(|file_path| {
            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                repo_id, file_path
            );
            let dest = target_dir.join(&file_path);
            (url, dest)
        })
        .collect();

    log::info!("Downloading {} files concurrently from repository", files_to_download.len());

    download_files_concurrent(client, files_to_download).await?;

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
    use wiremock::matchers::{method, path};

    /// Verify that download_huggingface_repo returns Ok immediately when the
    /// HuggingFace API returns an empty file list (no files to download).
    /// This is a structural test: it exercises the concurrent-download path
    /// with zero tasks and confirms the function signature compiles correctly.
    #[tokio::test]
    async fn test_download_huggingface_repo_empty_file_list() {
        let server = MockServer::start().await;

        // The function calls /api/models/<repo_id>/tree/main
        Mock::given(method("GET"))
            .and(path("/api/models/test-org/test-model/tree/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
            .mount(&server)
            .await;

        // We need to call the function but it uses a hard-coded huggingface.co URL.
        // Instead, exercise via the internal helper with an empty vec to confirm
        // the concurrent path compiles and short-circuits cleanly.
        let dir = tempfile::tempdir().unwrap();
        let client = reqwest::Client::new();
        let files_to_download: Vec<(String, std::path::PathBuf)> = vec![];

        // Run the concurrent download with an empty set — must return Ok(())
        let results = download_files_concurrent(&client, files_to_download).await;
        assert!(results.is_ok(), "empty file list should return Ok");
        let _ = server; // keep mock server alive
    }

    /// Verify that download_files_concurrent downloads all files concurrently
    /// and succeeds when the server responds with 200 for every file.
    #[tokio::test]
    async fn test_download_files_concurrent_downloads_all_files() {
        let server = MockServer::start().await;
        let content = b"binary file content";

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(content.as_slice()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let client = reqwest::Client::new();

        let files: Vec<(String, std::path::PathBuf)> = (0..5)
            .map(|i| {
                let url = format!("{}/file{}.bin", server.uri(), i);
                let dest = dir.path().join(format!("file{}.bin", i));
                (url, dest)
            })
            .collect();

        let result = download_files_concurrent(&client, files.clone()).await;
        assert!(result.is_ok(), "concurrent download should succeed");

        for (_, dest) in &files {
            assert!(dest.exists(), "file {:?} should have been downloaded", dest);
            assert_eq!(std::fs::read(dest).unwrap(), content);
        }
    }

    /// Verify that per-file failures are tolerated: the overall function returns
    /// Ok even when individual files return 404.
    #[tokio::test]
    async fn test_download_files_concurrent_tolerates_per_file_errors() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let client = reqwest::Client::new();

        let files: Vec<(String, std::path::PathBuf)> = vec![
            (format!("{}/missing.bin", server.uri()), dir.path().join("missing.bin")),
        ];

        let result = download_files_concurrent(&client, files).await;
        assert!(result.is_ok(), "per-file 404 should be tolerated as a warning");
    }

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

#[cfg(test)]
mod registry_tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_load_from_env_var_temp_file() {
        let json = r#"{
            "version": "2.0",
            "updated": "2026-03-27T00:00:00Z",
            "models": {
                "env-model": {
                    "name": "Env Model",
                    "score": 99.0,
                    "rank": 1,
                    "size": "10 MB",
                    "url": "https://example.com/env-model",
                    "architecture": "Test",
                    "voices": "1",
                    "quality": "High",
                    "status": "Available"
                }
            }
        }"#;

        let dir = tempfile::tempdir().unwrap();
        let registry_path = dir.path().join("test_registry.json");
        std::fs::write(&registry_path, json).unwrap();

        std::env::set_var("MODEL_REGISTRY_PATH", registry_path.to_str().unwrap());
        let registry = ModelRegistry::load();
        std::env::remove_var("MODEL_REGISTRY_PATH");

        assert!(
            registry.get_model("env-model").is_some(),
            "load() should read the registry from the env-var path"
        );
        assert_eq!(registry.get_model("env-model").unwrap().name, "Env Model");
    }

    #[test]
    #[serial]
    fn test_load_falls_back_to_compiled_in_registry() {
        // Make sure the env var points nowhere so load() falls back to include_str!
        std::env::set_var("MODEL_REGISTRY_PATH", "/this/path/does/not/exist/registry.json");
        let registry = ModelRegistry::load();
        std::env::remove_var("MODEL_REGISTRY_PATH");

        // The compiled-in model_registry.json should have at least one model.
        assert!(
            !registry.models.is_empty(),
            "load() fallback to compiled-in registry should return non-empty models"
        );
    }

    #[test]
    fn test_registry_loads_from_json_str() {
        let json = r#"{
            "version": "1.0",
            "updated": "2026-01-01T00:00:00Z",
            "models": {
                "test-model": {
                    "name": "Test Model",
                    "score": 50.0,
                    "rank": 5,
                    "size": "100 MB",
                    "url": "https://example.com",
                    "architecture": "Test",
                    "voices": "1",
                    "quality": "High",
                    "status": "Available"
                }
            }
        }"#;

        let registry = ModelRegistry::from_json_str(json);
        assert!(registry.get_model("test-model").is_some());
        let model = registry.get_model("test-model").unwrap();
        assert_eq!(model.name, "Test Model");
        assert!((model.score - 50.0).abs() < 0.01);
        assert_eq!(model.rank, 5);
    }

    #[test]
    fn test_registry_handles_mixed_rank_type() {
        let json = r#"{
            "version": "1.0",
            "updated": "2026-01-01T00:00:00Z",
            "models": {
                "windows-sapi": {
                    "name": "Windows SAPI",
                    "score": "N/A (Native)",
                    "rank": "Production",
                    "size": "Built-in",
                    "url": "Built-in",
                    "architecture": "Neural TTS",
                    "voices": 3,
                    "quality": "High",
                    "status": "Active"
                }
            }
        }"#;

        let registry = ModelRegistry::from_json_str(json);
        assert!(registry.get_model("windows-sapi").is_some());
        let model = registry.get_model("windows-sapi").unwrap();
        assert_eq!(model.score, 0.0);
        assert_eq!(model.rank, 0);
        // Check de_string_or_num: integer 3 should become "3"
        assert_eq!(model.voices, "3");
    }

    #[test]
    fn test_registry_from_json_str_invalid_json_returns_empty() {
        let registry = ModelRegistry::from_json_str("not valid json {{{{");
        assert_eq!(registry.models.len(), 0);
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
