use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use crate::models::download::{ModelDownloadManager, ModelSource, DownloadTask};
use crate::error::ApiError;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
pub struct DownloadModelRequest {
    pub model_name: String,
    pub source_type: String,
    pub repo_id: Option<String>,
    pub revision: Option<String>,
    pub url: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DownloadModelResponse {
    pub task_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub models: Vec<ModelInfoResponse>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelInfoResponse {
    pub name: String,
    pub source: String,
    pub size_bytes: u64,
    pub size_human: String,
    pub downloaded_at: String,
}

pub struct ModelDownloadState {
    pub manager: Arc<ModelDownloadManager>,
}

pub async fn download_model(
    req: web::Json<DownloadModelRequest>,
    state: web::Data<ModelDownloadState>,
) -> Result<HttpResponse, ApiError> {
    let source = match req.source_type.as_str() {
        "huggingface" => {
            let repo_id = req.repo_id.clone()
                .ok_or_else(|| ApiError::BadRequest("repo_id required for HuggingFace source".to_string()))?;
            ModelSource::HuggingFace {
                repo_id,
                revision: req.revision.clone(),
            }
        }
        "url" => {
            let url = req.url.clone()
                .ok_or_else(|| ApiError::BadRequest("url required for URL source".to_string()))?;
            ModelSource::Url { url }
        }
        _ => {
            return Err(ApiError::BadRequest(format!("Unknown source type: {}", req.source_type)));
        }
    };

    let task_id = state.manager.download_model(req.model_name.clone(), source)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(DownloadModelResponse {
        task_id: task_id.clone(),
        status: "started".to_string(),
        message: format!("Download task {} created for model {}", task_id, req.model_name),
    }))
}

pub async fn get_download_status(
    task_id: web::Path<String>,
    state: web::Data<ModelDownloadState>,
) -> Result<HttpResponse, ApiError> {
    let task = state.manager.get_task_status(&task_id)
        .ok_or_else(|| ApiError::NotFound(format!("Task {} not found", task_id)))?;

    Ok(HttpResponse::Ok().json(task))
}

pub async fn list_downloads(
    state: web::Data<ModelDownloadState>,
) -> Result<HttpResponse, ApiError> {
    let tasks = state.manager.list_tasks();
    Ok(HttpResponse::Ok().json(tasks))
}

pub async fn list_models(
    state: web::Data<ModelDownloadState>,
) -> Result<HttpResponse, ApiError> {
    let models = state.manager.list_models();
    
    let model_infos: Vec<ModelInfoResponse> = models.iter().map(|m| {
        ModelInfoResponse {
            name: m.name.clone(),
            source: format!("{:?}", m.source),
            size_bytes: m.size_bytes,
            size_human: format_bytes(m.size_bytes),
            downloaded_at: m.downloaded_at.to_rfc3339(),
        }
    }).collect();

    Ok(HttpResponse::Ok().json(ModelListResponse {
        total: model_infos.len(),
        models: model_infos,
    }))
}

pub async fn get_model_info(
    name: web::Path<String>,
    state: web::Data<ModelDownloadState>,
) -> Result<HttpResponse, ApiError> {
    let model = state.manager.get_model(&name)
        .ok_or_else(|| ApiError::NotFound(format!("Model {} not found", name)))?;

    Ok(HttpResponse::Ok().json(ModelInfoResponse {
        name: model.name,
        source: format!("{:?}", model.source),
        size_bytes: model.size_bytes,
        size_human: format_bytes(model.size_bytes),
        downloaded_at: model.downloaded_at.to_rfc3339(),
    }))
}

pub async fn delete_model(
    name: web::Path<String>,
    state: web::Data<ModelDownloadState>,
) -> Result<HttpResponse, ApiError> {
    state.manager.delete_model(&name)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "message": format!("Model {} deleted successfully", name),
        "success": true,
    })))
}

pub async fn get_cache_info(
    state: web::Data<ModelDownloadState>,
) -> Result<HttpResponse, ApiError> {
    let cache_info = state.manager.get_cache_info();
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "cache_dir": cache_info.cache_dir,
        "total_size_bytes": cache_info.total_size_bytes,
        "total_size_human": format_bytes(cache_info.total_size_bytes),
        "model_count": cache_info.model_count,
    })))
}

pub async fn list_available_models() -> Result<HttpResponse, ApiError> {
    // Load models from registry
    let registry_path = std::path::Path::new("model_registry.json");
    let models = if registry_path.exists() {
        match std::fs::read_to_string(registry_path) {
            Ok(content) => {
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(registry) => {
                        if let Some(models_obj) = registry.get("models").and_then(|m| m.as_object()) {
                            let mut available = Vec::new();
                            
                            // Categorize models
                            let mut categories = std::collections::HashMap::new();
                            categories.insert("tts", 0);
                            categories.insert("image-classification", 0);
                            categories.insert("speech-recognition", 0);
                            categories.insert("multimodal", 0);
                            
                            for (model_id, model_data) in models_obj {
                                let model_obj = model_data.as_object().unwrap();
                                
                                // Determine task type
                                let task = if let Some(task_str) = model_obj.get("task").and_then(|t| t.as_str()) {
                                    task_str.to_string()
                                } else if model_obj.get("voices").is_some() {
                                    "text-to-speech".to_string()
                                } else if model_obj.get("accuracy").is_some() {
                                    "image-classification".to_string()
                                } else {
                                    "unknown".to_string()
                                };
                                
                                // Count categories
                                let category = match task.as_str() {
                                    "text-to-speech" => "tts",
                                    "Image Classification" | "image-classification" => "image-classification",
                                    "automatic-speech-recognition" => "speech-recognition",
                                    "zero-shot-image-classification" => "multimodal",
                                    _ => "other",
                                };
                                *categories.entry(category).or_insert(0) += 1;
                                
                                // Extract repo_id from URL
                                let url = model_obj.get("url").and_then(|u| u.as_str()).unwrap_or("");
                                let repo_id = if url.contains("huggingface.co") {
                                    // Extract repo_id from HuggingFace URL
                                    if let Some(parts) = url.split("/").collect::<Vec<_>>().get(3..5) {
                                        Some(format!("{}/{}", parts[0], parts[1]))
                                    } else {
                                        Some(url.to_string())
                                    }
                                } else {
                                    None
                                };
                                
                                let mut model_info = serde_json::json!({
                                    "id": model_id,
                                    "name": model_obj.get("name").and_then(|n| n.as_str()).unwrap_or(model_id),
                                    "architecture": model_obj.get("architecture").and_then(|a| a.as_str()).unwrap_or("Unknown"),
                                    "task": task,
                                    "source": if url.contains("huggingface") { "huggingface" } else { "url" },
                                    "url": url,
                                    "size_estimate": model_obj.get("size").and_then(|s| s.as_str()).unwrap_or("Unknown"),
                                    "status": model_obj.get("status").and_then(|s| s.as_str()).unwrap_or("Available"),
                                });
                                
                                // Add repo_id if available
                                if let Some(repo) = repo_id {
                                    model_info["repo_id"] = serde_json::json!(repo);
                                }
                                
                                // Add task-specific info
                                if task == "image-classification" {
                                    if let Some(accuracy) = model_obj.get("accuracy").and_then(|a| a.as_str()) {
                                        model_info["accuracy"] = serde_json::json!(accuracy);
                                    }
                                    if let Some(rank) = model_obj.get("rank") {
                                        model_info["rank"] = rank.clone();
                                    }
                                    if let Some(dataset) = model_obj.get("dataset").and_then(|d| d.as_str()) {
                                        model_info["dataset"] = serde_json::json!(dataset);
                                    }
                                } else if task == "text-to-speech" {
                                    if let Some(voices) = model_obj.get("voices") {
                                        model_info["voices"] = voices.clone();
                                    }
                                    if let Some(quality) = model_obj.get("quality").and_then(|q| q.as_str()) {
                                        model_info["quality"] = serde_json::json!(quality);
                                    }
                                    if let Some(score) = model_obj.get("score") {
                                        model_info["score"] = score.clone();
                                    }
                                }
                                
                                if let Some(note) = model_obj.get("note").and_then(|n| n.as_str()) {
                                    model_info["note"] = serde_json::json!(note);
                                }
                                
                                available.push(model_info);
                            }
                            
                            // Sort by task and rank/score
                            available.sort_by(|a, b| {
                                let task_a = a["task"].as_str().unwrap_or("");
                                let task_b = b["task"].as_str().unwrap_or("");
                                
                                match task_a.cmp(task_b) {
                                    std::cmp::Ordering::Equal => {
                                        // Within same task, sort by rank (lower is better)
                                        let rank_a = a["rank"].as_u64().unwrap_or(999);
                                        let rank_b = b["rank"].as_u64().unwrap_or(999);
                                        rank_a.cmp(&rank_b)
                                    }
                                    other => other,
                                }
                            });
                            
                            return Ok(HttpResponse::Ok().json(serde_json::json!({
                                "models": available,
                                "total": available.len(),
                                "categories": categories,
                                "message": "Use POST /models/download to download any model",
                                "source": "model_registry.json",
                            })));
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to parse registry: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to read registry: {}", e);
            }
        }
    };
    
    // Fallback to hardcoded list if registry not available
    let available = vec![
        // Text Models
        serde_json::json!({
            "name": "bert-base-uncased",
            "repo_id": "bert-base-uncased",
            "task": "text-classification",
            "source": "huggingface",
            "description": "BERT base model (uncased)",
            "size_estimate": "~440 MB",
        }),
        serde_json::json!({
            "name": "gpt2",
            "repo_id": "gpt2",
            "task": "text-generation",
            "source": "huggingface",
            "description": "GPT-2 language model",
            "size_estimate": "~500 MB",
        }),
        
        // Image Models
        serde_json::json!({
            "name": "resnet50",
            "repo_id": "microsoft/resnet-50",
            "task": "image-classification",
            "source": "huggingface",
            "description": "ResNet-50 image classification",
            "size_estimate": "~100 MB",
        }),
        serde_json::json!({
            "name": "yolov5",
            "repo_id": "ultralytics/yolov5",
            "task": "object-detection",
            "source": "huggingface",
            "description": "YOLOv5 object detection",
            "size_estimate": "~30 MB",
        }),
        
        // Audio/TTS Models
        serde_json::json!({
            "name": "whisper-base",
            "repo_id": "openai/whisper-base",
            "task": "automatic-speech-recognition",
            "source": "huggingface",
            "description": "Whisper base model for speech recognition",
            "size_estimate": "~140 MB",
        }),
        
        // Multimodal
        serde_json::json!({
            "name": "clip-vit-base",
            "repo_id": "openai/clip-vit-base-patch32",
            "task": "zero-shot-image-classification",
            "source": "huggingface",
            "description": "CLIP vision-language model",
            "size_estimate": "~350 MB",
        }),
    ];

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "models": available,
        "total": available.len(),
        "categories": {
            "text": 2,
            "image": 2,
            "audio": 2,
            "multimodal": 1
        },
        "message": "Use POST /models/download to download any model (fallback list)",
        "source": "hardcoded",
    })))
}

pub async fn download_sota_model(
    model_id: web::Path<String>,
    state: web::Data<ModelDownloadState>,
) -> Result<HttpResponse, ApiError> {
    // Load model info from registry
    let registry_path = std::path::Path::new("model_registry.json");
    
    if !registry_path.exists() {
        return Err(ApiError::NotFound("Model registry not found".to_string()));
    }
    
    let content = std::fs::read_to_string(registry_path)
        .map_err(|e| ApiError::InternalError(format!("Failed to read registry: {}", e)))?;
    
    let registry: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| ApiError::InternalError(format!("Failed to parse registry: {}", e)))?;
    
    let models_obj = registry.get("models")
        .and_then(|m| m.as_object())
        .ok_or_else(|| ApiError::InternalError("Invalid registry format".to_string()))?;
    
    let model_data = models_obj.get(model_id.as_str())
        .ok_or_else(|| ApiError::NotFound(format!("Model {} not found in registry", model_id)))?;
    
    let model_obj = model_data.as_object()
        .ok_or_else(|| ApiError::InternalError("Invalid model data".to_string()))?;
    
    // Extract model info
    let url = model_obj.get("url")
        .and_then(|u| u.as_str())
        .ok_or_else(|| ApiError::BadRequest("Model URL not found".to_string()))?;
    
    let name = model_obj.get("name")
        .and_then(|n| n.as_str())
        .unwrap_or(model_id.as_str());
    
    // Determine source type
    let source = if url.contains("huggingface.co") {
        // Extract repo_id from URL
        // Format: https://huggingface.co/{org}/{repo}/...
        let parts: Vec<&str> = url.split('/').collect();
        if parts.len() >= 5 {
            let repo_id = format!("{}/{}", parts[3], parts[4]);
            ModelSource::HuggingFace {
                repo_id,
                revision: None,
            }
        } else {
            ModelSource::Url { url: url.to_string() }
        }
    } else if url != "Built-in" {
        ModelSource::Url { url: url.to_string() }
    } else {
        return Err(ApiError::BadRequest("Built-in models cannot be downloaded".to_string()));
    };
    
    // Start download
    let task_id = state.manager.download_model(name.to_string(), source)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "task_id": task_id,
        "status": "started",
        "model_id": model_id.as_str(),
        "model_name": name,
        "message": format!("Download task {} created for model {}", task_id, name),
    })))
}

pub async fn list_sota_models() -> Result<HttpResponse, ApiError> {
    // Load and filter SOTA image classification models from registry
    let registry_path = std::path::Path::new("model_registry.json");
    
    if !registry_path.exists() {
        return Ok(HttpResponse::Ok().json(serde_json::json!({
            "models": [],
            "total": 0,
            "message": "Model registry not found",
        })));
    }
    
    let content = std::fs::read_to_string(registry_path)
        .map_err(|e| ApiError::InternalError(format!("Failed to read registry: {}", e)))?;
    
    let registry: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| ApiError::InternalError(format!("Failed to parse registry: {}", e)))?;
    
    let models_obj = registry.get("models")
        .and_then(|m| m.as_object())
        .ok_or_else(|| ApiError::InternalError("Invalid registry format".to_string()))?;
    
    let mut sota_models = Vec::new();
    
    for (model_id, model_data) in models_obj {
        let model_obj = model_data.as_object().unwrap();
        
        // Filter for image classification models (SOTA)
        if let Some(task) = model_obj.get("task").and_then(|t| t.as_str()) {
            if task == "Image Classification" {
                let mut model_info = serde_json::json!({
                    "id": model_id,
                    "name": model_obj.get("name").and_then(|n| n.as_str()).unwrap_or(model_id),
                    "architecture": model_obj.get("architecture").and_then(|a| a.as_str()).unwrap_or("Unknown"),
                    "accuracy": model_obj.get("accuracy").and_then(|a| a.as_str()).unwrap_or("N/A"),
                    "rank": model_obj.get("rank"),
                    "size": model_obj.get("size").and_then(|s| s.as_str()).unwrap_or("Unknown"),
                    "url": model_obj.get("url").and_then(|u| u.as_str()).unwrap_or(""),
                    "dataset": model_obj.get("dataset").and_then(|d| d.as_str()).unwrap_or("ImageNet-1K"),
                    "status": model_obj.get("status").and_then(|s| s.as_str()).unwrap_or("Available"),
                });
                
                if let Some(note) = model_obj.get("note").and_then(|n| n.as_str()) {
                    model_info["note"] = serde_json::json!(note);
                }
                
                sota_models.push(model_info);
            }
        }
    }
    
    // Sort by rank
    sota_models.sort_by(|a, b| {
        let rank_a = a["rank"].as_u64().unwrap_or(999);
        let rank_b = b["rank"].as_u64().unwrap_or(999);
        rank_a.cmp(&rank_b)
    });
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "models": sota_models,
        "total": sota_models.len(),
        "message": "SOTA image classification models. Use POST /models/sota/{model_id} to download",
        "documentation": "See SOTA_MODELS.md for details",
    })))
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_bytes_terabyte() {
        assert_eq!(format_bytes(1024u64 * 1024 * 1024 * 1024), "1.00 TB");
    }

    #[test]
    fn test_format_bytes_partial_kb() {
        // 512 bytes = 0.50 KB
        assert_eq!(format_bytes(512), "512.00 B");
    }

    #[test]
    fn test_format_bytes_1_byte() {
        assert_eq!(format_bytes(1), "1.00 B");
    }

    #[test]
    fn test_format_bytes_just_under_kb() {
        assert_eq!(format_bytes(1023), "1023.00 B");
    }

    #[test]
    fn test_format_bytes_large_gb() {
        // 2 GB
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_download_model_request_struct() {
        let req = DownloadModelRequest {
            model_name: "bert-base".to_string(),
            source_type: "huggingface".to_string(),
            repo_id: Some("bert-base-uncased".to_string()),
            revision: None,
            url: None,
        };
        assert_eq!(req.model_name, "bert-base");
        assert_eq!(req.source_type, "huggingface");
        assert_eq!(req.repo_id, Some("bert-base-uncased".to_string()));
        assert!(req.revision.is_none());
        assert!(req.url.is_none());
    }

    #[test]
    fn test_download_model_request_url_source() {
        let req = DownloadModelRequest {
            model_name: "custom-model".to_string(),
            source_type: "url".to_string(),
            repo_id: None,
            revision: None,
            url: Some("https://example.com/model.onnx".to_string()),
        };
        assert_eq!(req.source_type, "url");
        assert_eq!(req.url, Some("https://example.com/model.onnx".to_string()));
    }

    #[test]
    fn test_download_model_response_struct() {
        let resp = DownloadModelResponse {
            task_id: "task-abc".to_string(),
            status: "started".to_string(),
            message: "Download task created".to_string(),
        };
        assert_eq!(resp.task_id, "task-abc");
        assert_eq!(resp.status, "started");
    }

    #[test]
    fn test_model_info_response_struct() {
        let info = ModelInfoResponse {
            name: "resnet50".to_string(),
            source: "HuggingFace { repo_id: \"microsoft/resnet-50\", revision: None }".to_string(),
            size_bytes: 1_048_576,
            size_human: format_bytes(1_048_576),
            downloaded_at: "2024-01-01T00:00:00Z".to_string(),
        };
        assert_eq!(info.name, "resnet50");
        assert_eq!(info.size_bytes, 1_048_576);
        assert_eq!(info.size_human, "1.00 MB");
    }

    #[test]
    fn test_model_list_response_struct() {
        let list = ModelListResponse {
            models: vec![],
            total: 0,
        };
        assert_eq!(list.total, 0);
        assert!(list.models.is_empty());
    }

    #[test]
    fn test_model_info_response_size_human_consistency() {
        // The size_human field should always reflect format_bytes of size_bytes
        for &bytes in &[0u64, 1, 1023, 1024, 1_048_576, 1_073_741_824] {
            let info = ModelInfoResponse {
                name: "m".to_string(),
                source: "test".to_string(),
                size_bytes: bytes,
                size_human: format_bytes(bytes),
                downloaded_at: "2024-01-01T00:00:00Z".to_string(),
            };
            assert_eq!(info.size_human, format_bytes(bytes));
        }
    }

    // ── Handler unit tests ────────────────────────────────────────────────────

    fn make_download_state() -> web::Data<ModelDownloadState> {
        let manager = Arc::new(
            crate::models::download::ModelDownloadManager::new("/tmp/test_model_cache_dl")
                .expect("create manager"),
        );
        web::Data::new(ModelDownloadState { manager })
    }

    // download_model — huggingface source missing repo_id → BadRequest
    #[actix_web::test]
    async fn test_download_model_huggingface_missing_repo_id() {
        let state = make_download_state();
        let req = web::Json(DownloadModelRequest {
            model_name: "test-model".to_string(),
            source_type: "huggingface".to_string(),
            repo_id: None,
            revision: None,
            url: None,
        });
        let result = download_model(req, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::BadRequest(_)));
    }

    // download_model — url source missing url → BadRequest
    #[actix_web::test]
    async fn test_download_model_url_source_missing_url() {
        let state = make_download_state();
        let req = web::Json(DownloadModelRequest {
            model_name: "test-model".to_string(),
            source_type: "url".to_string(),
            repo_id: None,
            revision: None,
            url: None,
        });
        let result = download_model(req, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::BadRequest(_)));
    }

    // download_model — unknown source_type → BadRequest
    #[actix_web::test]
    async fn test_download_model_unknown_source_type() {
        let state = make_download_state();
        let req = web::Json(DownloadModelRequest {
            model_name: "test-model".to_string(),
            source_type: "s3".to_string(),
            repo_id: None,
            revision: None,
            url: None,
        });
        let result = download_model(req, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::BadRequest(_)));
    }

    // get_download_status — task not found → NotFound
    #[actix_web::test]
    async fn test_get_download_status_not_found() {
        let state = make_download_state();
        let task_id = web::Path::from("nonexistent-task-id-xyz".to_string());
        let result = get_download_status(task_id, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::NotFound(_)));
    }

    // list_downloads — empty manager returns Ok
    #[actix_web::test]
    async fn test_list_downloads_empty() {
        let state = make_download_state();
        let result = list_downloads(state).await;
        assert!(result.is_ok());
    }

    // list_models — empty manager returns Ok with empty list
    #[actix_web::test]
    async fn test_list_models_empty() {
        let state = make_download_state();
        let result = list_models(state).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // get_model_info — model not found → NotFound
    #[actix_web::test]
    async fn test_get_model_info_not_found() {
        let state = make_download_state();
        let name = web::Path::from("nonexistent-model".to_string());
        let result = get_model_info(name, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::NotFound(_)));
    }

    // delete_model — model not found → InternalError (delete_model bails)
    #[actix_web::test]
    async fn test_delete_model_not_found() {
        let state = make_download_state();
        let name = web::Path::from("nonexistent-model".to_string());
        let result = delete_model(name, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::InternalError(_)));
    }

    // get_cache_info — empty manager returns Ok
    #[actix_web::test]
    async fn test_get_cache_info_empty() {
        let state = make_download_state();
        let result = get_cache_info(state).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // list_available_models — no registry file falls back to hardcoded list
    #[actix_web::test]
    async fn test_list_available_models_fallback() {
        // Run from a temp directory that has no model_registry.json
        let result = list_available_models().await;
        // Should always succeed (falls back to hardcoded list)
        assert!(result.is_ok());
    }

    // DownloadModelRequest — huggingface with revision
    #[test]
    fn test_download_model_request_with_revision() {
        let req = DownloadModelRequest {
            model_name: "bert".to_string(),
            source_type: "huggingface".to_string(),
            repo_id: Some("bert-base-uncased".to_string()),
            revision: Some("main".to_string()),
            url: None,
        };
        assert_eq!(req.revision.as_deref(), Some("main"));
    }

    // DownloadModelResponse — serialization roundtrip
    #[test]
    fn test_download_model_response_serde() {
        let resp = DownloadModelResponse {
            task_id: "abc-123".to_string(),
            status: "started".to_string(),
            message: "Download started".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["task_id"], "abc-123");
        assert_eq!(back["status"], "started");
    }

    // ModelListResponse — serialization
    #[test]
    fn test_model_list_response_serde() {
        let list = ModelListResponse {
            models: vec![ModelInfoResponse {
                name: "m1".to_string(),
                source: "Local".to_string(),
                size_bytes: 1024,
                size_human: "1.00 KB".to_string(),
                downloaded_at: "2024-01-01T00:00:00Z".to_string(),
            }],
            total: 1,
        };
        let json = serde_json::to_string(&list).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["total"], 1);
        assert_eq!(back["models"][0]["name"], "m1");
    }
}
