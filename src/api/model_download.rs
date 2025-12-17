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
    // Curated list of available models for download
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
        "message": "Use POST /models/download to download any model",
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
}
