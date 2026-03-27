use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use std::path::PathBuf;

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
