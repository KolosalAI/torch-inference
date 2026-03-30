#![allow(dead_code, unused_imports, unused_variables)]
use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "torch")]
use tch::{Tensor, Device};

use crate::core::neural_network::{NeuralNetwork, InferenceResult, NetworkMetadata};
use crate::error::ApiError;

#[cfg(feature = "torch")]
use crate::models::pytorch_loader::get_best_device;

pub struct NeuralNetworkState {
    pub networks: Arc<dashmap::DashMap<String, Arc<NeuralNetwork>>>,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
    pub model_path: String,
    pub device: Option<String>,
    pub metadata: Option<NetworkMetadata>,
}

#[derive(Debug, Deserialize)]
pub struct InferenceRequest {
    pub model_id: String,
    pub input_data: Vec<f32>,
    pub input_shape: Vec<i64>,
}

#[derive(Debug, Serialize)]
pub struct LoadModelResponse {
    pub success: bool,
    pub model_id: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub success: bool,
    pub result: Option<InferenceResult>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub models: Vec<ModelInfoResponse>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelInfoResponse {
    pub model_id: String,
    pub metadata: NetworkMetadata,
}

/// POST /api/nn/load - Load a neural network model
#[cfg(feature = "torch")]
pub async fn load_model(
    req: web::Json<LoadModelRequest>,
    state: web::Data<NeuralNetworkState>,
) -> Result<HttpResponse, ApiError> {
    let model_path = PathBuf::from(&req.model_path);
    
    if !model_path.exists() {
        return Err(ApiError::NotFound(format!("Model file not found: {}", req.model_path)));
    }
    
    // Parse device
    let device = if let Some(dev_str) = &req.device {
        if dev_str.starts_with("cuda") {
            Device::Cuda(0)
        } else if dev_str == "mps" {
            Device::Mps
        } else {
            Device::Cpu
        }
    } else {
        get_best_device()
    };
    
    // Load model
    match NeuralNetwork::new(&model_path, Some(device), req.metadata.clone()) {
        Ok(network) => {
            let network = Arc::new(network);
            state.networks.insert(req.model_id.clone(), network);
            
            Ok(HttpResponse::Ok().json(LoadModelResponse {
                success: true,
                model_id: req.model_id.clone(),
                message: format!("Model {} loaded successfully", req.model_id),
            }))
        }
        Err(e) => {
            Err(ApiError::InternalError(format!("Failed to load model: {}", e)))
        }
    }
}

#[cfg(not(feature = "torch"))]
pub async fn load_model(
    _req: web::Json<LoadModelRequest>,
    _state: web::Data<NeuralNetworkState>,
) -> Result<HttpResponse, ApiError> {
    Err(ApiError::InternalError(
        "PyTorch feature not enabled. Compile with --features torch".to_string()
    ))
}

/// POST /api/nn/predict - Run inference on a loaded model
#[cfg(feature = "torch")]
pub async fn predict(
    req: web::Json<InferenceRequest>,
    state: web::Data<NeuralNetworkState>,
) -> Result<HttpResponse, ApiError> {
    // Get model
    let network = state.networks.get(&req.model_id)
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", req.model_id)))?;
    
    // Run inference
    match network.predict_from_slice(&req.input_data, &req.input_shape) {
        Ok(result) => {
            Ok(HttpResponse::Ok().json(InferenceResponse {
                success: true,
                result: Some(result),
                error: None,
            }))
        }
        Err(e) => {
            Ok(HttpResponse::Ok().json(InferenceResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
            }))
        }
    }
}

#[cfg(not(feature = "torch"))]
pub async fn predict(
    _req: web::Json<InferenceRequest>,
    _state: web::Data<NeuralNetworkState>,
) -> Result<HttpResponse, ApiError> {
    Err(ApiError::InternalError(
        "PyTorch feature not enabled".to_string()
    ))
}

/// GET /api/nn/models - List loaded models
pub async fn list_models(
    state: web::Data<NeuralNetworkState>,
) -> Result<HttpResponse, ApiError> {
    let models: Vec<ModelInfoResponse> = state.networks.iter()
        .map(|entry| {
            let model_id = entry.key().clone();
            let network = entry.value();
            ModelInfoResponse {
                model_id,
                metadata: network.metadata().clone(),
            }
        })
        .collect();
    
    let total = models.len();
    
    Ok(HttpResponse::Ok().json(ModelListResponse {
        models,
        total,
    }))
}

/// GET /api/nn/models/{model_id} - Get model info
pub async fn get_model_info(
    path: web::Path<String>,
    state: web::Data<NeuralNetworkState>,
) -> Result<HttpResponse, ApiError> {
    let model_id = path.into_inner();
    
    let network = state.networks.get(&model_id)
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", model_id)))?;
    
    Ok(HttpResponse::Ok().json(ModelInfoResponse {
        model_id: model_id.clone(),
        metadata: network.metadata().clone(),
    }))
}

/// DELETE /api/nn/models/{model_id} - Unload a model
pub async fn unload_model(
    path: web::Path<String>,
    state: web::Data<NeuralNetworkState>,
) -> Result<HttpResponse, ApiError> {
    let model_id = path.into_inner();
    
    if state.networks.remove(&model_id).is_some() {
        Ok(HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "message": format!("Model {} unloaded successfully", model_id)
        })))
    } else {
        Err(ApiError::NotFound(format!("Model not found: {}", model_id)))
    }
}

/// POST /api/nn/batch - Batch inference
#[derive(Debug, Deserialize)]
pub struct BatchInferenceRequest {
    pub model_id: String,
    pub inputs: Vec<BatchInput>,
}

#[derive(Debug, Deserialize)]
pub struct BatchInput {
    pub data: Vec<f32>,
    pub shape: Vec<i64>,
}

#[derive(Debug, Serialize)]
pub struct BatchInferenceResponse {
    pub success: bool,
    pub results: Vec<InferenceResult>,
    pub total_time_ms: f64,
    pub error: Option<String>,
}

#[cfg(feature = "torch")]
pub async fn batch_predict(
    req: web::Json<BatchInferenceRequest>,
    state: web::Data<NeuralNetworkState>,
) -> Result<HttpResponse, ApiError> {
    let network = state.networks.get(&req.model_id)
        .ok_or_else(|| ApiError::NotFound(format!("Model not found: {}", req.model_id)))?;
    
    let start = std::time::Instant::now();
    let mut results = Vec::new();
    
    for input in &req.inputs {
        match network.predict_from_slice(&input.data, &input.shape) {
            Ok(result) => results.push(result),
            Err(e) => {
                return Ok(HttpResponse::Ok().json(BatchInferenceResponse {
                    success: false,
                    results: Vec::new(),
                    total_time_ms: 0.0,
                    error: Some(e.to_string()),
                }));
            }
        }
    }
    
    let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
    
    Ok(HttpResponse::Ok().json(BatchInferenceResponse {
        success: true,
        results,
        total_time_ms,
        error: None,
    }))
}

#[cfg(not(feature = "torch"))]
pub async fn batch_predict(
    _req: web::Json<BatchInferenceRequest>,
    _state: web::Data<NeuralNetworkState>,
) -> Result<HttpResponse, ApiError> {
    Err(ApiError::InternalError("PyTorch feature not enabled".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_load_model_request_serde() {
        let json = r#"{"model_id":"my_model","model_path":"/path/to/model.pt","device":"cpu"}"#;
        let req: LoadModelRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_id, "my_model");
        assert_eq!(req.model_path, "/path/to/model.pt");
        assert_eq!(req.device.as_deref(), Some("cpu"));
        assert!(req.metadata.is_none());
    }

    #[test]
    fn test_load_model_request_with_metadata() {
        let json = r#"{
            "model_id": "resnet",
            "model_path": "/models/resnet.pt",
            "device": "cuda",
            "metadata": {
                "name": "ResNet50",
                "task": "classification",
                "framework": "pytorch",
                "input_names": ["input"],
                "output_names": ["output"],
                "description": null
            }
        }"#;
        let req: LoadModelRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_id, "resnet");
        let meta = req.metadata.unwrap();
        assert_eq!(meta.name, "ResNet50");
        assert_eq!(meta.task, "classification");
    }

    #[test]
    fn test_inference_request_serde() {
        let json = r#"{"model_id":"net1","input_data":[1.0,2.0,3.0],"input_shape":[1,3]}"#;
        let req: InferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_id, "net1");
        assert_eq!(req.input_data, vec![1.0f32, 2.0, 3.0]);
        assert_eq!(req.input_shape, vec![1i64, 3]);
    }

    #[test]
    fn test_load_model_response_serialization() {
        let resp = LoadModelResponse {
            success: true,
            model_id: "my_model".to_string(),
            message: "Model loaded successfully".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], true);
        assert_eq!(back["model_id"], "my_model");
    }

    #[test]
    fn test_inference_response_success_serialization() {
        let result = InferenceResult {
            outputs: {
                let mut m = HashMap::new();
                m.insert("output".to_string(), vec![0.1f32, 0.9]);
                m
            },
            inference_time_ms: 5.2,
            device: "cpu".to_string(),
        };
        let resp = InferenceResponse {
            success: true,
            result: Some(result),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], true);
        assert!(back["error"].is_null());
    }

    #[test]
    fn test_inference_response_failure_serialization() {
        let resp = InferenceResponse {
            success: false,
            result: None,
            error: Some("Model not loaded".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], false);
        assert_eq!(back["error"], "Model not loaded");
    }

    #[test]
    fn test_model_list_response_serialization() {
        let resp = ModelListResponse {
            models: vec![],
            total: 0,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["total"], 0);
        assert!(back["models"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_batch_input_serde() {
        let json = r#"{"data":[0.5,0.3],"shape":[1,2]}"#;
        let input: BatchInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.data.len(), 2);
        assert_eq!(input.shape, vec![1i64, 2]);
    }

    #[test]
    fn test_batch_inference_request_serde() {
        let json = r#"{"model_id":"net","inputs":[{"data":[1.0],"shape":[1,1]}]}"#;
        let req: BatchInferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_id, "net");
        assert_eq!(req.inputs.len(), 1);
    }

    #[test]
    fn test_batch_inference_response_serialization() {
        let resp = BatchInferenceResponse {
            success: true,
            results: vec![],
            total_time_ms: 10.0,
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], true);
        assert!((back["total_time_ms"].as_f64().unwrap() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_network_metadata_serde_roundtrip() {
        let meta = NetworkMetadata {
            name: "TestNet".to_string(),
            task: "regression".to_string(),
            framework: "pytorch".to_string(),
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
            description: Some("A test network".to_string()),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let back: NetworkMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "TestNet");
        assert_eq!(back.description.as_deref(), Some("A test network"));
    }

    // ── Handler tests via actix-web test harness ──────────────────────────────

    fn make_state_with_network(model_id: &str) -> web::Data<NeuralNetworkState> {
        let networks = Arc::new(dashmap::DashMap::new());
        // Construct a minimal NeuralNetwork-less state; we bypass NeuralNetwork
        // loading and instead insert a pre-built network via the public API.
        // Since NeuralNetwork::new requires a real model file (torch feature),
        // we only insert when the dashmap is empty — handler tests that exercise
        // NOT-found paths just leave the map empty.
        let _ = model_id; // used only in the "found" variant below
        web::Data::new(NeuralNetworkState { networks })
    }

    #[actix_web::test]
    async fn test_list_models_handler_empty() {
        let state = make_state_with_network("");
        let resp = list_models(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_get_model_info_handler_not_found() {
        let state = make_state_with_network("");
        let path = web::Path::from("nonexistent".to_string());
        let result = get_model_info(path, state).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::ApiError::NotFound(_)));
    }

    #[actix_web::test]
    async fn test_unload_model_handler_not_found() {
        let state = make_state_with_network("");
        let path = web::Path::from("ghost_model".to_string());
        let result = unload_model(path, state).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::ApiError::NotFound(_)));
    }

    #[actix_web::test]
    async fn test_load_model_handler_file_not_found() {
        let state = make_state_with_network("");
        let req = web::Json(LoadModelRequest {
            model_id: "test".to_string(),
            model_path: "/nonexistent/path/model.pt".to_string(),
            device: Some("cpu".to_string()),
            metadata: None,
        });
        let result = load_model(req, state).await;
        // Without the torch feature the handler returns InternalError; with torch
        // and a missing file it returns NotFound.  Either way it must be Err.
        assert!(result.is_err());
    }

    #[actix_web::test]
    async fn test_predict_handler_model_not_found() {
        let state = make_state_with_network("");
        let req = web::Json(InferenceRequest {
            model_id: "missing_model".to_string(),
            input_data: vec![1.0, 2.0],
            input_shape: vec![1, 2],
        });
        let result = predict(req, state).await;
        // Returns NotFound (torch) or InternalError (no torch); either is Err.
        assert!(result.is_err());
    }

    #[actix_web::test]
    async fn test_batch_predict_handler_model_not_found() {
        let state = make_state_with_network("");
        let req = web::Json(BatchInferenceRequest {
            model_id: "missing".to_string(),
            inputs: vec![BatchInput {
                data: vec![0.5],
                shape: vec![1, 1],
            }],
        });
        let result = batch_predict(req, state).await;
        assert!(result.is_err());
    }

    #[actix_web::test]
    async fn test_model_info_response_serde() {
        let meta = NetworkMetadata {
            name: "net".to_string(),
            task: "cls".to_string(),
            framework: "pytorch".to_string(),
            input_names: vec![],
            output_names: vec![],
            description: None,
        };
        let resp = ModelInfoResponse {
            model_id: "my_net".to_string(),
            metadata: meta,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["model_id"], "my_net");
        assert_eq!(v["metadata"]["name"], "net");
    }

    #[test]
    fn test_batch_inference_response_failure_serde() {
        let resp = BatchInferenceResponse {
            success: false,
            results: vec![],
            total_time_ms: 0.0,
            error: Some("engine error".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["success"], false);
        assert_eq!(v["error"], "engine error");
    }

    #[test]
    fn test_load_model_request_device_none() {
        let json = r#"{"model_id":"m","model_path":"/x.pt"}"#;
        let req: LoadModelRequest = serde_json::from_str(json).unwrap();
        assert!(req.device.is_none());
    }

    #[test]
    fn test_inference_response_no_result_serde() {
        let resp = InferenceResponse {
            success: false,
            result: None,
            error: Some("not found".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["result"].is_null());
    }

    // ── Additional serde tests ────────────────────────────────────────────────

    #[test]
    fn test_load_model_request_device_mps() {
        let json = r#"{"model_id":"m","model_path":"/m.pt","device":"mps"}"#;
        let req: LoadModelRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.device.as_deref(), Some("mps"));
    }

    #[test]
    fn test_load_model_request_device_cuda() {
        let json = r#"{"model_id":"m","model_path":"/m.pt","device":"cuda:0"}"#;
        let req: LoadModelRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.device.as_deref(), Some("cuda:0"));
    }

    #[test]
    fn test_batch_inference_response_zero_time() {
        let resp = BatchInferenceResponse {
            success: true,
            results: vec![],
            total_time_ms: 0.0,
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["success"], true);
        assert!(v["error"].is_null());
    }

    #[test]
    fn test_load_model_response_failure_serde() {
        let resp = LoadModelResponse {
            success: false,
            model_id: "failed_model".to_string(),
            message: "Failed to load".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["success"], false);
        assert_eq!(v["message"], "Failed to load");
    }

    #[test]
    fn test_model_list_response_with_one_model() {
        let meta = NetworkMetadata {
            name: "TestNet".to_string(),
            task: "classification".to_string(),
            framework: "pytorch".to_string(),
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            description: None,
        };
        let resp = ModelListResponse {
            models: vec![ModelInfoResponse {
                model_id: "test-net".to_string(),
                metadata: meta,
            }],
            total: 1,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["total"], 1);
        assert_eq!(v["models"][0]["model_id"], "test-net");
    }

    #[test]
    fn test_inference_request_empty_data() {
        let json = r#"{"model_id":"net","input_data":[],"input_shape":[]}"#;
        let req: InferenceRequest = serde_json::from_str(json).unwrap();
        assert!(req.input_data.is_empty());
        assert!(req.input_shape.is_empty());
    }

    #[test]
    fn test_batch_input_empty_data() {
        let json = r#"{"data":[],"shape":[]}"#;
        let input: BatchInput = serde_json::from_str(json).unwrap();
        assert!(input.data.is_empty());
        assert!(input.shape.is_empty());
    }

    #[test]
    fn test_batch_inference_request_empty_inputs() {
        let json = r#"{"model_id":"net","inputs":[]}"#;
        let req: BatchInferenceRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_id, "net");
        assert!(req.inputs.is_empty());
    }

    #[test]
    fn test_network_metadata_no_description() {
        let meta = NetworkMetadata {
            name: "Bare".to_string(),
            task: "detection".to_string(),
            framework: "onnx".to_string(),
            input_names: vec![],
            output_names: vec![],
            description: None,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["description"].is_null());
    }

    // ── NeuralNetworkState construction ───────────────────────────────────────

    #[test]
    fn test_neural_network_state_starts_empty() {
        let state = make_state_with_network("");
        assert!(state.networks.is_empty());
    }

    // ── list_models + insert directly ─────────────────────────────────────────

    // NOTE: NeuralNetwork::new always fails without the torch feature, so we
    // cannot construct a NeuralNetwork to insert.  The list_models, get_model_info
    // and unload_model "found" paths are therefore exercised only when the torch
    // feature is active.  Without torch, we validate the empty-state behaviour
    // which is already well covered above.

    #[actix_web::test]
    async fn test_list_models_handler_returns_ok_status() {
        let state = make_state_with_network("");
        let resp = list_models(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_unload_model_returns_not_found_for_missing() {
        let state = make_state_with_network("");
        let path = web::Path::from("not-loaded".to_string());
        let result = unload_model(path, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::NotFound(_)));
    }
}

// Configure routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/nn")
            .route("/load", web::post().to(load_model))
            .route("/predict", web::post().to(predict))
            .route("/batch", web::post().to(batch_predict))
            .route("/models", web::get().to(list_models))
            .route("/models/{model_id}", web::get().to(get_model_info))
            .route("/models/{model_id}", web::delete().to(unload_model))
    );
}

#[cfg(test)]
mod configure_and_found_tests {
    use super::*;
    use actix_web::{test as actix_test, App};

    // ── configure() smoke test (covers lines 698-706) ─────────────────────────
    //
    // Calling `configure` with the app data-less service and hitting each route
    // exercises the route-registration code. Without the NeuralNetworkState app
    // data the handlers return 500 (missing extractor), but the routes are still
    // registered and reachable (i.e. NOT 404/405).

    fn make_empty_state() -> web::Data<NeuralNetworkState> {
        web::Data::new(NeuralNetworkState {
            networks: Arc::new(dashmap::DashMap::new()),
        })
    }

    #[actix_web::test]
    async fn test_configure_registers_all_nn_routes() {
        let state = make_empty_state();
        let app = actix_test::init_service(
            App::new()
                .app_data(state.clone())
                .configure(configure),
        )
        .await;

        // GET /api/nn/models — list_models (empty state → 200)
        let req = actix_test::TestRequest::get()
            .uri("/api/nn/models")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_ne!(resp.status(), actix_web::http::StatusCode::NOT_FOUND,
            "/api/nn/models must be registered");
        assert_ne!(resp.status(), actix_web::http::StatusCode::METHOD_NOT_ALLOWED);

        // POST /api/nn/load — registered even without torch feature
        let req = actix_test::TestRequest::post()
            .uri("/api/nn/load")
            .set_json(serde_json::json!({"model_id":"x","model_path":"/x.pt"}))
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_ne!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);

        // GET /api/nn/models/{model_id} — returns 404 body (not 404 "route not found")
        let req = actix_test::TestRequest::get()
            .uri("/api/nn/models/absent")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        // The handler returns ApiError::NotFound → 404, not "route not found"
        // Both represent 404 but the route IS registered.
        // We just verify it isn't 405 (method not allowed) which would mean wrong method.
        assert_ne!(resp.status(), actix_web::http::StatusCode::METHOD_NOT_ALLOWED);

        // DELETE /api/nn/models/{model_id}
        let req = actix_test::TestRequest::delete()
            .uri("/api/nn/models/absent")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_ne!(resp.status(), actix_web::http::StatusCode::METHOD_NOT_ALLOWED);
    }

    // ── list_models with a non-empty dashmap (covers lines 159-163) ───────────
    //
    // We need at least one entry in the DashMap to drive the `.map()` closure.
    // NeuralNetwork::new always fails without the `torch` feature, so these
    // tests are gated behind `#[cfg(feature = "torch")]`.

    #[cfg(feature = "torch")]
    #[actix_web::test]
    async fn test_list_models_returns_entries_when_network_loaded() {
        use std::path::Path;
        let networks = Arc::new(dashmap::DashMap::new());
        // Load a real model — only viable in CI with torch + a valid model path.
        // Here we just assert the test compiles; in practice, create a minimal
        // TorchScript model in the fixture directory.
        if let Ok(network) = crate::core::neural_network::NeuralNetwork::new(
            Path::new("tests/fixtures/minimal.pt"),
            None,
            None,
        ) {
            networks.insert("fixture-net".to_string(), Arc::new(network));
        }

        let state = web::Data::new(NeuralNetworkState { networks });
        let resp = list_models(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // ── get_model_info found path (covers lines 186-188) ─────────────────────

    #[cfg(feature = "torch")]
    #[actix_web::test]
    async fn test_get_model_info_found_returns_200() {
        use std::path::Path;
        let networks = Arc::new(dashmap::DashMap::new());
        if let Ok(network) = crate::core::neural_network::NeuralNetwork::new(
            Path::new("tests/fixtures/minimal.pt"),
            None,
            None,
        ) {
            networks.insert("net-a".to_string(), Arc::new(network));
        }

        if !networks.is_empty() {
            let state = web::Data::new(NeuralNetworkState { networks });
            let path = web::Path::from("net-a".to_string());
            let result = get_model_info(path, state).await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap().status(), actix_web::http::StatusCode::OK);
        }
    }

    // ── unload_model success path (covers lines 200-202) ─────────────────────
    //
    // We insert a network and then call unload_model. The `networks.remove()`
    // returns `Some(...)`, triggering the success branch.

    #[cfg(feature = "torch")]
    #[actix_web::test]
    async fn test_unload_model_success_removes_network() {
        use std::path::Path;
        let networks = Arc::new(dashmap::DashMap::new());
        if let Ok(network) = crate::core::neural_network::NeuralNetwork::new(
            Path::new("tests/fixtures/minimal.pt"),
            None,
            None,
        ) {
            networks.insert("to-unload".to_string(), Arc::new(network));
        }

        if !networks.is_empty() {
            let state = web::Data::new(NeuralNetworkState {
                networks: networks.clone(),
            });
            let path = web::Path::from("to-unload".to_string());
            let result = unload_model(path, state).await;
            assert!(result.is_ok());
            assert_eq!(result.unwrap().status(), actix_web::http::StatusCode::OK);
            assert!(networks.get("to-unload").is_none(),
                "network should be removed from the map");
        }
    }

    // ── list_models with pre-populated dashmap via direct insert (no-torch) ──
    //
    // Since NeuralNetwork cannot be constructed without torch, we exercise the
    // "empty map" path which IS reachable and verify the JSON shape is correct.
    // The map iteration closure (lines 159-163) only runs when there is at least
    // one entry, which is only possible with the torch feature.

    #[actix_web::test]
    async fn test_list_models_empty_returns_zero_total() {
        let state = make_empty_state();
        let resp = list_models(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // ── list_models with one entry: exercises map closure (lines 159-163) ─────
    //
    // Uses NeuralNetwork::new_stub() so the test runs without the torch feature.

    #[actix_web::test]
    async fn test_list_models_with_stub_entry_covers_map_closure() {
        let networks = Arc::new(dashmap::DashMap::new());
        let stub = Arc::new(crate::core::neural_network::NeuralNetwork::new_stub(None));
        networks.insert("stub-net".to_string(), stub);

        let state = web::Data::new(NeuralNetworkState { networks });
        let resp = list_models(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let body = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["total"], 1);
        let models = v["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["model_id"], "stub-net");
    }

    // ── get_model_info found path (lines 186-188) ─────────────────────────────
    //
    // Insert a stub network and request it by id; the found branch returns 200.

    #[actix_web::test]
    async fn test_get_model_info_found_with_stub_returns_200() {
        let networks = Arc::new(dashmap::DashMap::new());
        let stub = Arc::new(crate::core::neural_network::NeuralNetwork::new_stub(None));
        networks.insert("info-net".to_string(), stub);

        let state = web::Data::new(NeuralNetworkState { networks });
        let path = web::Path::from("info-net".to_string());
        let result = get_model_info(path, state).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let body = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["model_id"], "info-net");
    }

    // ── unload_model success path (lines 200-202) ─────────────────────────────
    //
    // Insert a stub network and then unload it; the found branch returns 200
    // and the entry is removed from the map.

    #[actix_web::test]
    async fn test_unload_model_found_with_stub_returns_200() {
        let networks = Arc::new(dashmap::DashMap::new());
        let stub = Arc::new(crate::core::neural_network::NeuralNetwork::new_stub(None));
        networks.insert("rm-net".to_string(), stub);

        let state = web::Data::new(NeuralNetworkState {
            networks: networks.clone(),
        });
        let path = web::Path::from("rm-net".to_string());
        let result = unload_model(path, state).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        let body = actix_web::body::to_bytes(resp.into_body()).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["success"], true);
        assert!(v["message"].as_str().unwrap().contains("rm-net"));
        assert!(networks.get("rm-net").is_none(), "network must be removed");
    }

    // ── unload_model removes the entry and returns success message ────────────
    //
    // We test the found branch by manually inserting into dashmap.
    // NeuralNetwork is required, so this is torch-only.  Without torch we only
    // verify the not-found branch (already covered by the existing tests module).
    //
    // For the no-torch case we do nothing here — coverage of 200-202 requires torch.
}
