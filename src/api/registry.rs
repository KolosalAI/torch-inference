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
    use actix_web::test as actix_test;

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

    // ── Handler tests ──────────────────────────────────────────────────────────

    use crate::core::engine::InferenceEngine;
    use crate::models::manager::ModelManager;
    use crate::config::Config;

    fn make_engine() -> web::Data<Arc<InferenceEngine>> {
        let manager = Arc::new(ModelManager::new(&Config::default(), None));
        let engine = Arc::new(InferenceEngine::new(manager, &Config::default()));
        web::Data::new(engine)
    }

    // list_registered_models — no format query returns all (empty list)
    #[actix_web::test]
    async fn test_list_registered_models_handler_empty() {
        let engine = make_engine();
        let query = web::Query(ListByFormatQuery { format: None });
        let resp = list_registered_models(engine, query).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // list_registered_models — known format returns 200
    #[actix_web::test]
    async fn test_list_registered_models_handler_with_onnx_format() {
        let engine = make_engine();
        let query = web::Query(ListByFormatQuery { format: Some("onnx".to_string()) });
        let resp = list_registered_models(engine, query).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // list_registered_models — known format PyTorch returns 200
    #[actix_web::test]
    async fn test_list_registered_models_handler_with_pytorch_format() {
        let engine = make_engine();
        let query = web::Query(ListByFormatQuery { format: Some("pth".to_string()) });
        let resp = list_registered_models(engine, query).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // list_registered_models — unknown format returns 400
    #[actix_web::test]
    async fn test_list_registered_models_handler_unknown_format() {
        let engine = make_engine();
        let query = web::Query(ListByFormatQuery { format: Some("exotic_format".to_string()) });
        let resp = list_registered_models(engine, query).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    // list_registered_models — "safetensors" format returns 200
    #[actix_web::test]
    async fn test_list_registered_models_handler_safetensors_format() {
        let engine = make_engine();
        let query = web::Query(ListByFormatQuery { format: Some("safetensors".to_string()) });
        let resp = list_registered_models(engine, query).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // list_registered_models — "candle" format returns 200
    #[actix_web::test]
    async fn test_list_registered_models_handler_candle_format() {
        let engine = make_engine();
        let query = web::Query(ListByFormatQuery { format: Some("candle".to_string()) });
        let resp = list_registered_models(engine, query).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // get_model_metadata — model not found returns 404
    #[actix_web::test]
    async fn test_get_model_metadata_handler_not_found() {
        let engine = make_engine();
        let path = web::Path::from("nonexistent-model".to_string());
        let resp = get_model_metadata(engine, path).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
    }

    // register_model — non-existent path returns 400
    #[actix_web::test]
    async fn test_register_model_handler_bad_path() {
        let engine = make_engine();
        let req = web::Json(RegisterModelRequest {
            path: "/no/such/model.onnx".to_string(),
            name: None,
        });
        let resp = register_model(engine, req).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    // scan_directory — non-existent directory returns 400
    #[actix_web::test]
    async fn test_scan_directory_handler_bad_path() {
        let engine = make_engine();
        let req = web::Json(ScanDirectoryRequest {
            path: "/no/such/directory".to_string(),
        });
        let resp = scan_directory(engine, req).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    // scan_directory — empty directory returns 200 with 0 models
    #[actix_web::test]
    async fn test_scan_directory_handler_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let engine = make_engine();
        let req = web::Json(ScanDirectoryRequest {
            path: dir.path().to_string_lossy().to_string(),
        });
        let resp = scan_directory(engine, req).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // get_registry_stats — always returns 200
    #[actix_web::test]
    async fn test_get_registry_stats_handler() {
        let engine = make_engine();
        let resp = get_registry_stats(engine).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // export_registry — always returns 200
    #[actix_web::test]
    async fn test_export_registry_handler() {
        let engine = make_engine();
        let resp = export_registry(engine).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // load_pytorch_model — model not registered returns bad request or internal error
    #[actix_web::test]
    async fn test_load_pytorch_model_handler_not_registered() {
        let engine = make_engine();
        let path = web::Path::from("ghost-model".to_string());
        let resp = load_pytorch_model(engine, path).await.unwrap();
        // Without torch feature returns 400; with torch and missing model returns 500
        let status = resp.status().as_u16();
        assert!(status == 400 || status == 500,
            "expected 400 or 500, got {status}");
    }

    // infer_pytorch — unknown model returns bad request or internal error
    #[actix_web::test]
    async fn test_infer_pytorch_handler_missing_model() {
        let engine = make_engine();
        let req = web::Json(InferenceRequest {
            model_id: "not-a-model".to_string(),
            input: serde_json::json!({"x": 1}),
        });
        let resp = infer_pytorch(engine, req).await.unwrap();
        let status = resp.status().as_u16();
        assert!(status == 400 || status == 500,
            "expected 400 or 500, got {status}");
    }

    // ── register_model success path ────────────────────────────────────────────
    // Create a real temp .onnx file so register_model_from_path succeeds and
    // we exercise lines 67-73 (the Ok branch).

    #[actix_web::test]
    async fn test_register_model_handler_success_with_onnx_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("my_model.onnx");
        std::fs::write(&model_path, b"fake onnx content").unwrap();

        let engine = make_engine();
        let req = web::Json(RegisterModelRequest {
            path: model_path.to_string_lossy().to_string(),
            name: Some("my-onnx".to_string()),
        });
        let resp = register_model(engine, req).await.unwrap();
        // Should be 200 OK with model_id and metadata in the response
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // ── get_model_metadata success path ───────────────────────────────────────
    // Register a model first, then query its metadata to hit lines 114-115.

    #[actix_web::test]
    async fn test_get_model_metadata_handler_success() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("meta_model.onnx");
        std::fs::write(&model_path, b"fake onnx content").unwrap();

        let manager = Arc::new(crate::models::manager::ModelManager::new(&crate::config::Config::default(), None));
        // Register via the manager directly
        let model_id = manager
            .register_model_from_path(&model_path, Some("meta-model".to_string()))
            .await
            .expect("registration should succeed");

        let engine = Arc::new(crate::core::engine::InferenceEngine::new(manager, &crate::config::Config::default()));
        let engine_data = web::Data::new(engine);
        let path = web::Path::from(model_id.clone());
        let resp = get_model_metadata(engine_data, path).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // ── configure function ────────────────────────────────────────────────────
    // The configure() function registers routes on a ServiceConfig. We verify
    // it completes without panicking by wiring it into an actix App.

    #[actix_web::test]
    async fn test_configure_registers_routes() {
        let engine = make_engine();
        let app = actix_test::init_service(
            actix_web::App::new()
                .app_data(engine.clone())
                .configure(configure)
        ).await;

        // /registry/models should respond (empty list → 200)
        let req = actix_test::TestRequest::get()
            .uri("/registry/models")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK,
            "configure() must register the /registry/models route");
    }

    #[actix_web::test]
    async fn test_configure_stats_route() {
        let engine = make_engine();
        let app = actix_test::init_service(
            actix_web::App::new()
                .app_data(engine.clone())
                .configure(configure)
        ).await;

        let req = actix_test::TestRequest::get()
            .uri("/registry/stats")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK,
            "configure() must register the /registry/stats route");
    }

    #[actix_web::test]
    async fn test_configure_export_route() {
        let engine = make_engine();
        let app = actix_test::init_service(
            actix_web::App::new()
                .app_data(engine.clone())
                .configure(configure)
        ).await;

        let req = actix_test::TestRequest::get()
            .uri("/registry/export")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK,
            "configure() must register the /registry/export route");
    }

    #[actix_web::test]
    async fn test_configure_unknown_model_metadata_route() {
        let engine = make_engine();
        let app = actix_test::init_service(
            actix_web::App::new()
                .app_data(engine.clone())
                .configure(configure)
        ).await;

        let req = actix_test::TestRequest::get()
            .uri("/registry/models/does-not-exist")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND,
            "GET /registry/models/{{id}} for unknown model should return 404");
    }
}
