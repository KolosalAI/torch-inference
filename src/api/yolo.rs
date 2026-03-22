// API endpoints for YOLO object detection

use actix_web::{web, HttpResponse, Result};
use actix_multipart::Multipart;
use serde::{Deserialize, Serialize};
use futures_util::StreamExt;
use std::sync::Arc;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::core::yolo::{YoloDetector, YoloVersion, YoloSize, YoloResults, load_coco_names};
use crate::error::ApiError;

/// YOLO detection request
#[derive(Debug, Deserialize)]
pub struct YoloDetectRequest {
    pub model_version: String,  // v5, v8, v10, v11, v12
    pub model_size: String,     // n, s, m, l, x
    #[serde(default = "default_conf_threshold")]
    pub conf_threshold: f32,
    #[serde(default = "default_iou_threshold")]
    pub iou_threshold: f32,
}

fn default_conf_threshold() -> f32 { 0.25 }
fn default_iou_threshold() -> f32 { 0.45 }

/// YOLO detection response
#[derive(Debug, Serialize)]
pub struct YoloDetectResponse {
    pub success: bool,
    pub results: Option<YoloResults>,
    pub error: Option<String>,
}

/// YOLO model info request
#[derive(Debug, Deserialize)]
pub struct YoloInfoRequest {
    pub model_version: String,
    pub model_size: String,
}

/// YOLO model info response
#[derive(Debug, Serialize)]
pub struct YoloInfoResponse {
    pub model_name: String,
    pub version: String,
    pub size: String,
    pub num_classes: usize,
    pub input_size: (i64, i64),
    pub conf_threshold: f32,
    pub iou_threshold: f32,
    pub available: bool,
}

/// Available YOLO models
#[derive(Debug, Serialize)]
pub struct YoloModelsResponse {
    pub versions: Vec<String>,
    pub sizes: Vec<String>,
    pub total_models: usize,
}

pub struct YoloState {
    pub models_dir: PathBuf,
}

/// Detect objects in an uploaded image
pub async fn detect_objects(
    mut payload: Multipart,
    query: web::Query<YoloDetectRequest>,
    state: web::Data<YoloState>,
) -> Result<HttpResponse, ApiError> {
    // Parse model version and size
    let version = YoloVersion::from_str(&query.model_version)
        .ok_or_else(|| ApiError::BadRequest(format!("Invalid YOLO version: {}", query.model_version)))?;
    
    let size = YoloSize::from_suffix(&query.model_size)
        .ok_or_else(|| ApiError::BadRequest(format!("Invalid model size: {}", query.model_size)))?;

    // Save uploaded image to temp file
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join(format!("yolo_input_{}.jpg", uuid::Uuid::new_v4()));

    // Extract image from multipart
    while let Some(Ok(mut field)) = payload.next().await {
        let mut file = fs::File::create(&temp_file).await
            .map_err(|e| ApiError::InternalError(e.to_string()))?;

        while let Some(chunk) = field.next().await {
            let data = chunk.map_err(|e| ApiError::InternalError(e.to_string()))?;
            file.write_all(&data).await
                .map_err(|e| ApiError::InternalError(e.to_string()))?;
        }
    }

    // Load model
    let model_name = format!("yolo{}{}",
        version.as_str().to_lowercase().replace("yolo", ""),
        size.suffix()
    );
    let model_path = state.models_dir.join(&model_name).join(format!("{}.pt", model_name));

    if !model_path.exists() {
        // Cleanup temp file
        let _ = fs::remove_file(&temp_file).await;
        
        return Err(ApiError::NotFound(format!(
            "Model not found: {}. Please download it first.", model_name
        )));
    }

    // Load COCO class names
    let class_names = load_coco_names();

    // Create detector
    #[cfg(feature = "torch")]
    let detector = {
        use tch::Device;
        let mut detector = YoloDetector::new(
            &model_path,
            version,
            size,
            class_names,
            Some(Device::Cpu),
        ).map_err(|e| ApiError::InternalError(e.to_string()))?;

        detector.set_conf_threshold(query.conf_threshold);
        detector.set_iou_threshold(query.iou_threshold);
        detector
    };

    #[cfg(not(feature = "torch"))]
    {
        let _ = fs::remove_file(&temp_file).await;
        return Err(ApiError::InternalError(
            "PyTorch feature not enabled. Compile with --features torch".to_string()
        ));
    }

    // Perform detection
    #[cfg(feature = "torch")]
    let results = detector.detect(&temp_file)
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    // Cleanup temp file
    let _ = fs::remove_file(&temp_file).await;

    #[cfg(feature = "torch")]
    return Ok(HttpResponse::Ok().json(YoloDetectResponse {
        success: true,
        results: Some(results),
        error: None,
    }));

    #[cfg(not(feature = "torch"))]
    Ok(HttpResponse::Ok().json(YoloDetectResponse {
        success: false,
        results: None,
        error: Some("PyTorch not enabled".to_string()),
    }))
}

/// Get information about a specific YOLO model
pub async fn get_model_info(
    query: web::Query<YoloInfoRequest>,
    state: web::Data<YoloState>,
) -> Result<HttpResponse, ApiError> {
    let version = YoloVersion::from_str(&query.model_version)
        .ok_or_else(|| ApiError::BadRequest(format!("Invalid YOLO version: {}", query.model_version)))?;
    
    let size = YoloSize::from_suffix(&query.model_size)
        .ok_or_else(|| ApiError::BadRequest(format!("Invalid model size: {}", query.model_size)))?;

    let model_name = format!("yolo{}{}",
        version.as_str().to_lowercase().replace("yolo", ""),
        size.suffix()
    );
    let model_path = state.models_dir.join(&model_name).join(format!("{}.pt", model_name));

    let response = YoloInfoResponse {
        model_name: model_name.clone(),
        version: version.as_str().to_string(),
        size: size.suffix().to_string(),
        num_classes: 80, // COCO dataset
        input_size: (640, 640),
        conf_threshold: 0.25,
        iou_threshold: 0.45,
        available: model_path.exists(),
    };

    Ok(HttpResponse::Ok().json(response))
}

/// List all available YOLO models
pub async fn list_models(_state: web::Data<YoloState>) -> Result<HttpResponse, ApiError> {
    let versions = vec!["v5", "v8", "v10", "v11", "v12"];
    let sizes = vec!["n", "s", "m", "l", "x"];
    
    let response = YoloModelsResponse {
        versions: versions.iter().map(|s| s.to_string()).collect(),
        sizes: sizes.iter().map(|s| s.to_string()).collect(),
        total_models: versions.len() * sizes.len(),
    };

    Ok(HttpResponse::Ok().json(response))
}

/// Download a YOLO model
#[derive(Debug, Deserialize)]
pub struct YoloDownloadRequest {
    pub model_version: String,
    pub model_size: String,
}

#[derive(Debug, Serialize)]
pub struct YoloDownloadResponse {
    pub success: bool,
    pub message: String,
    pub download_url: Option<String>,
}

pub async fn download_model(
    req: web::Json<YoloDownloadRequest>,
    _state: web::Data<YoloState>,
) -> Result<HttpResponse, ApiError> {
    let version = YoloVersion::from_str(&req.model_version)
        .ok_or_else(|| ApiError::BadRequest(format!("Invalid YOLO version: {}", req.model_version)))?;
    
    let size = YoloSize::from_suffix(&req.model_size)
        .ok_or_else(|| ApiError::BadRequest(format!("Invalid model size: {}", req.model_size)))?;

    // Generate download URL based on version
    let download_url = match version {
        YoloVersion::V5 => format!(
            "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5{}.pt",
            size.suffix()
        ),
        YoloVersion::V8 => format!(
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8{}.pt",
            size.suffix()
        ),
        YoloVersion::V10 => format!(
            "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{}.pt",
            size.suffix()
        ),
        YoloVersion::V11 => format!(
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11{}.pt",
            size.suffix()
        ),
        YoloVersion::V12 => {
            // YOLOv12 is the latest (as of Dec 2024)
            // Using placeholder URL - update with actual release URL
            format!(
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12{}.pt",
                size.suffix()
            )
        }
    };

    let response = YoloDownloadResponse {
        success: true,
        message: format!("Download {} {} from the provided URL", version.as_str(), size.suffix()),
        download_url: Some(download_url),
    };

    Ok(HttpResponse::Ok().json(response))
}

/// Configure YOLO routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/yolo")
            .route("/detect", web::post().to(detect_objects))
            .route("/info", web::get().to(get_model_info))
            .route("/models", web::get().to(list_models))
            .route("/download", web::post().to(download_model))
    );
}
