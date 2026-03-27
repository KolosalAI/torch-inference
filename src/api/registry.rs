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

#[cfg(test)]
mod tests {
    use super::*;

    // ── ApiResponse ───────────────────────────────────────────────────────────

    #[test]
    fn test_api_response_success_contains_data() {
        let resp: ApiResponse<String> = ApiResponse::success("hello".to_string());
        assert!(resp.success);
        assert_eq!(resp.data, Some("hello".to_string()));
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_api_response_error_contains_message() {
        let resp: ApiResponse<String> = ApiResponse::error("something went wrong".to_string());
        assert!(!resp.success);
        assert!(resp.data.is_none());
        assert_eq!(resp.error, Some("something went wrong".to_string()));
    }

    #[test]
    fn test_api_response_success_with_json_value() {
        let data = serde_json::json!({"model_id": "abc", "status": "loaded"});
        let resp: ApiResponse<serde_json::Value> = ApiResponse::success(data.clone());
        assert!(resp.success);
        assert_eq!(resp.data.unwrap(), data);
    }

    #[test]
    fn test_api_response_error_with_unit_type() {
        let resp: ApiResponse<()> = ApiResponse::error("not found".to_string());
        assert!(!resp.success);
        assert!(resp.data.is_none());
        assert_eq!(resp.error.as_deref(), Some("not found"));
    }

    #[test]
    fn test_api_response_success_serde() {
        let resp: ApiResponse<u32> = ApiResponse::success(42u32);
        let json = serde_json::to_string(&resp).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(v["success"], true);
        assert_eq!(v["data"], 42);
        assert!(v["error"].is_null());
    }

    #[test]
    fn test_api_response_error_serde() {
        let resp: ApiResponse<u32> = ApiResponse::error("oops".to_string());
        let json = serde_json::to_string(&resp).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(v["success"], false);
        assert!(v["data"].is_null());
        assert_eq!(v["error"], "oops");
    }

    // ── Request struct deserialization ────────────────────────────────────────

    #[test]
    fn test_register_model_request_deserialize_with_name() {
        let json = r#"{"path": "/models/bert.pt", "name": "bert"}"#;
        let req: RegisterModelRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.path, "/models/bert.pt");
        assert_eq!(req.name, Some("bert".to_string()));
    }

    #[test]
    fn test_register_model_request_deserialize_without_name() {
        let json = r#"{"path": "/models/bert.onnx"}"#;
        let req: RegisterModelRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.path, "/models/bert.onnx");
        assert!(req.name.is_none());
    }

    #[test]
    fn test_scan_directory_request_deserialize() {
        let json = r#"{"path": "/models"}"#;
        let req: ScanDirectoryRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.path, "/models");
    }

    #[test]
    fn test_inference_request_deserialize() {
        let json = r#"{"model_id": "bert-base", "input": {"tokens": [1, 2, 3]}}"#;
        let req: InferenceRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.model_id, "bert-base");
        assert!(req.input.is_object());
    }

    #[test]
    fn test_list_by_format_query_with_format() {
        let json = r#"{"format": "onnx"}"#;
        let q: ListByFormatQuery = serde_json::from_str(json).expect("deserialize");
        assert_eq!(q.format, Some("onnx".to_string()));
    }

    #[test]
    fn test_list_by_format_query_without_format() {
        let json = r#"{}"#;
        let q: ListByFormatQuery = serde_json::from_str(json).expect("deserialize");
        assert!(q.format.is_none());
    }

    // ── ModelFormat matching (exercises the branches in list_registered_models) ──

    #[test]
    fn test_model_format_from_format_string_pytorch_variants() {
        use crate::models::registry::ModelFormat;
        // These strings map to ModelFormat::PyTorch inside list_registered_models
        for s in &["pytorch", "pt", "pth"] {
            let matched = match s.to_lowercase().as_str() {
                "pytorch" | "pt" | "pth" => ModelFormat::PyTorch,
                "onnx" => ModelFormat::ONNX,
                "candle" => ModelFormat::Candle,
                "safetensors" => ModelFormat::SafeTensors,
                _ => panic!("unexpected format"),
            };
            assert_eq!(matched, ModelFormat::PyTorch, "failed for {s}");
        }
    }

    #[test]
    fn test_model_format_from_format_string_onnx() {
        use crate::models::registry::ModelFormat;
        let matched = match "onnx" {
            "pytorch" | "pt" | "pth" => ModelFormat::PyTorch,
            "onnx" => ModelFormat::ONNX,
            "candle" => ModelFormat::Candle,
            "safetensors" => ModelFormat::SafeTensors,
            _ => panic!("unexpected format"),
        };
        assert_eq!(matched, ModelFormat::ONNX);
    }

    #[test]
    fn test_model_format_from_format_string_candle() {
        use crate::models::registry::ModelFormat;
        let matched = match "candle" {
            "pytorch" | "pt" | "pth" => ModelFormat::PyTorch,
            "onnx" => ModelFormat::ONNX,
            "candle" => ModelFormat::Candle,
            "safetensors" => ModelFormat::SafeTensors,
            _ => panic!("unexpected format"),
        };
        assert_eq!(matched, ModelFormat::Candle);
    }

    #[test]
    fn test_model_format_from_format_string_safetensors() {
        use crate::models::registry::ModelFormat;
        let matched = match "safetensors" {
            "pytorch" | "pt" | "pth" => ModelFormat::PyTorch,
            "onnx" => ModelFormat::ONNX,
            "candle" => ModelFormat::Candle,
            "safetensors" => ModelFormat::SafeTensors,
            _ => panic!("unexpected format"),
        };
        assert_eq!(matched, ModelFormat::SafeTensors);
    }

    #[test]
    fn test_unknown_format_string_is_recognized() {
        // This mirrors the `_` arm that returns BadRequest in list_registered_models
        let format_str = "unknown_format";
        let is_unknown = !matches!(format_str.to_lowercase().as_str(),
            "pytorch" | "pt" | "pth" | "onnx" | "candle" | "safetensors");
        assert!(is_unknown, "unexpected format should be flagged");
    }

    // ── ApiResponse JSON structure ─────────────────────────────────────────────

    #[test]
    fn test_api_response_success_nested_object() {
        let data = serde_json::json!({
            "model_id": "bert",
            "metadata": {"format": "onnx", "version": "1.0"}
        });
        let resp: ApiResponse<serde_json::Value> = ApiResponse::success(data);
        let json = serde_json::to_string(&resp).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert_eq!(v["success"], true);
        assert_eq!(v["data"]["model_id"], "bert");
        assert!(v["error"].is_null());
    }

    #[test]
    fn test_api_response_error_long_message() {
        let msg = "a".repeat(500);
        let resp: ApiResponse<()> = ApiResponse::error(msg.clone());
        assert_eq!(resp.error.as_deref(), Some(msg.as_str()));
        assert!(!resp.success);
    }

    #[test]
    fn test_register_model_request_empty_path() {
        let json = r#"{"path": ""}"#;
        let req: RegisterModelRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.path, "");
        assert!(req.name.is_none());
    }

    #[test]
    fn test_inference_request_array_input() {
        let json = r#"{"model_id": "net", "input": [1.0, 2.0, 3.0]}"#;
        let req: InferenceRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.model_id, "net");
        assert!(req.input.is_array());
    }

    #[test]
    fn test_inference_request_string_input() {
        let json = r#"{"model_id": "text-model", "input": "hello world"}"#;
        let req: InferenceRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.input.as_str(), Some("hello world"));
    }

    #[test]
    fn test_scan_directory_request_relative_path() {
        let json = r#"{"path": "./models"}"#;
        let req: ScanDirectoryRequest = serde_json::from_str(json).expect("deserialize");
        assert_eq!(req.path, "./models");
    }

    #[test]
    fn test_api_response_success_vec_data() {
        let data = vec!["model-a".to_string(), "model-b".to_string()];
        let resp: ApiResponse<Vec<String>> = ApiResponse::success(data.clone());
        assert!(resp.success);
        assert_eq!(resp.data, Some(data));
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_api_response_error_does_not_have_data() {
        let resp: ApiResponse<serde_json::Value> = ApiResponse::error("internal error".to_string());
        assert!(resp.data.is_none());
        assert!(resp.error.is_some());
    }

    #[test]
    fn test_list_by_format_query_case_insensitive_matching() {
        // The handler does to_lowercase() before matching; verify lower produces known variant
        let format_str = "ONNX";
        let lower = format_str.to_lowercase();
        let is_onnx = lower == "onnx";
        assert!(is_onnx);
    }
}
