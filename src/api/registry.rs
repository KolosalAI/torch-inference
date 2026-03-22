use actix_web::{web, HttpResponse, Result as ActixResult};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::PathBuf;
use log::{info, error};

use crate::core::engine::InferenceEngine;
use crate::models::registry::ModelFormat;

#[derive(Deserialize)]
pub struct RegisterModelRequest {
    pub path: String,
    pub name: Option<String>,
}

#[derive(Deserialize)]
pub struct ScanDirectoryRequest {
    pub path: String,
}

#[derive(Deserialize)]
pub struct InferenceRequest {
    pub model_id: String,
    pub input: serde_json::Value,
}

#[derive(Deserialize)]
pub struct ListByFormatQuery {
    pub format: Option<String>,
}

#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(error: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
        }
    }
}

/// Register a model from file path
pub async fn register_model(
    engine: web::Data<Arc<InferenceEngine>>,
    req: web::Json<RegisterModelRequest>,
) -> ActixResult<HttpResponse> {
    info!("API: Register model from path: {}", req.path);

    let path = PathBuf::from(&req.path);
    
    match engine.model_manager.register_model_from_path(&path, req.name.clone()).await {
        Ok(model_id) => {
            let metadata = engine.model_manager.get_model_metadata(&model_id)
                .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
            
            Ok(HttpResponse::Ok().json(ApiResponse::success(serde_json::json!({
                "model_id": model_id,
                "metadata": metadata,
            }))))
        }
        Err(e) => {
            error!("Failed to register model: {}", e);
            Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(e.to_string())))
        }
    }
}

/// Scan directory and register all models
pub async fn scan_directory(
    engine: web::Data<Arc<InferenceEngine>>,
    req: web::Json<ScanDirectoryRequest>,
) -> ActixResult<HttpResponse> {
    info!("API: Scan directory: {}", req.path);

    let path = PathBuf::from(&req.path);
    
    match engine.model_manager.scan_and_register(&path).await {
        Ok(model_ids) => {
            Ok(HttpResponse::Ok().json(ApiResponse::success(serde_json::json!({
                "registered": model_ids.len(),
                "model_ids": model_ids,
            }))))
        }
        Err(e) => {
            error!("Failed to scan directory: {}", e);
            Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(e.to_string())))
        }
    }
}

/// Get model metadata
pub async fn get_model_metadata(
    engine: web::Data<Arc<InferenceEngine>>,
    model_id: web::Path<String>,
) -> ActixResult<HttpResponse> {
    info!("API: Get model metadata: {}", model_id);

    match engine.model_manager.get_model_metadata(&model_id) {
        Ok(metadata) => {
            Ok(HttpResponse::Ok().json(ApiResponse::success(metadata)))
        }
        Err(e) => {
            error!("Model not found: {}", e);
            Ok(HttpResponse::NotFound().json(ApiResponse::<()>::error(e.to_string())))
        }
    }
}

/// List all registered models
pub async fn list_registered_models(
    engine: web::Data<Arc<InferenceEngine>>,
    query: web::Query<ListByFormatQuery>,
) -> ActixResult<HttpResponse> {
    info!("API: List registered models");

    let models = if let Some(format_str) = &query.format {
        let format = match format_str.to_lowercase().as_str() {
            "pytorch" | "pt" | "pth" => ModelFormat::PyTorch,
            "onnx" => ModelFormat::ONNX,
            "candle" => ModelFormat::Candle,
            "safetensors" => ModelFormat::SafeTensors,
            _ => {
                return Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
                    format!("Unknown format: {}", format_str)
                )));
            }
        };
        engine.model_manager.list_by_format(format)
    } else {
        engine.model_manager.list_registered_models()
    };

    Ok(HttpResponse::Ok().json(ApiResponse::success(serde_json::json!({
        "models": models,
        "count": models.len(),
    }))))
}

/// Load a PyTorch model
pub async fn load_pytorch_model(
    engine: web::Data<Arc<InferenceEngine>>,
    model_id: web::Path<String>,
) -> ActixResult<HttpResponse> {
    info!("API: Load PyTorch model: {}", model_id);

    #[cfg(feature = "torch")]
    {
        match engine.model_manager.load_pytorch_model(&model_id).await {
            Ok(_) => {
                Ok(HttpResponse::Ok().json(ApiResponse::success(serde_json::json!({
                    "model_id": model_id.as_str(),
                    "status": "loaded",
                }))))
            }
            Err(e) => {
                error!("Failed to load model: {}", e);
                Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(e.to_string())))
            }
        }
    }

    #[cfg(not(feature = "torch"))]
    {
        Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "PyTorch support not enabled. Compile with --features torch".to_string()
        )))
    }
}

/// Run inference on a PyTorch model
pub async fn infer_pytorch(
    engine: web::Data<Arc<InferenceEngine>>,
    req: web::Json<InferenceRequest>,
) -> ActixResult<HttpResponse> {
    info!("API: PyTorch inference on model: {}", req.model_id);

    #[cfg(feature = "torch")]
    {
        match engine.model_manager.infer_pytorch(&req.model_id, &req.input).await {
            Ok(output) => {
                Ok(HttpResponse::Ok().json(ApiResponse::success(serde_json::json!({
                    "model_id": req.model_id,
                    "output": output,
                }))))
            }
            Err(e) => {
                error!("Inference failed: {}", e);
                Ok(HttpResponse::InternalServerError().json(ApiResponse::<()>::error(e.to_string())))
            }
        }
    }

    #[cfg(not(feature = "torch"))]
    {
        Ok(HttpResponse::BadRequest().json(ApiResponse::<()>::error(
            "PyTorch support not enabled. Compile with --features torch".to_string()
        )))
    }
}

/// Get registry statistics
pub async fn get_registry_stats(
    engine: web::Data<Arc<InferenceEngine>>,
) -> ActixResult<HttpResponse> {
    info!("API: Get registry stats");

    let stats = engine.model_manager.get_registry_stats();
    Ok(HttpResponse::Ok().json(ApiResponse::success(stats)))
}

/// Export registry
pub async fn export_registry(
    engine: web::Data<Arc<InferenceEngine>>,
) -> ActixResult<HttpResponse> {
    info!("API: Export registry");

    let registry = engine.model_manager.export_registry();
    Ok(HttpResponse::Ok().json(ApiResponse::success(registry)))
}

/// Configure routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/registry")
            .route("/register", web::post().to(register_model))
            .route("/scan", web::post().to(scan_directory))
            .route("/models", web::get().to(list_registered_models))
            .route("/models/{model_id}", web::get().to(get_model_metadata))
            .route("/models/{model_id}/load", web::post().to(load_pytorch_model))
            .route("/infer", web::post().to(infer_pytorch))
            .route("/stats", web::get().to(get_registry_stats))
            .route("/export", web::get().to(export_registry))
    );
}
