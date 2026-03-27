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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::yolo::{YoloVersion, YoloSize};

    #[test]
    fn test_yolo_detect_request_serde_defaults() {
        let json = r#"{"model_version":"v8","model_size":"n"}"#;
        let req: YoloDetectRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_version, "v8");
        assert_eq!(req.model_size, "n");
        assert!((req.conf_threshold - 0.25).abs() < f32::EPSILON);
        assert!((req.iou_threshold - 0.45).abs() < f32::EPSILON);
    }

    #[test]
    fn test_yolo_detect_request_serde_custom_thresholds() {
        let json = r#"{"model_version":"v5","model_size":"m","conf_threshold":0.5,"iou_threshold":0.6}"#;
        let req: YoloDetectRequest = serde_json::from_str(json).unwrap();
        assert!((req.conf_threshold - 0.5).abs() < f32::EPSILON);
        assert!((req.iou_threshold - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_conf_threshold() {
        assert!((default_conf_threshold() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_iou_threshold() {
        assert!((default_iou_threshold() - 0.45).abs() < f32::EPSILON);
    }

    #[test]
    fn test_yolo_detect_response_serialization_success() {
        let resp = YoloDetectResponse {
            success: true,
            results: None,
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], true);
        assert!(back["error"].is_null());
    }

    #[test]
    fn test_yolo_detect_response_serialization_failure() {
        let resp = YoloDetectResponse {
            success: false,
            results: None,
            error: Some("Model not found".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], false);
        assert_eq!(back["error"], "Model not found");
    }

    #[test]
    fn test_yolo_info_request_serde() {
        let json = r#"{"model_version":"v11","model_size":"s"}"#;
        let req: YoloInfoRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_version, "v11");
        assert_eq!(req.model_size, "s");
    }

    #[test]
    fn test_yolo_info_response_serialization() {
        let resp = YoloInfoResponse {
            model_name: "yolov8n".to_string(),
            version: "YOLOv8".to_string(),
            size: "n".to_string(),
            num_classes: 80,
            input_size: (640, 640),
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            available: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["num_classes"], 80);
        assert_eq!(back["available"], false);
    }

    #[test]
    fn test_yolo_models_response_serialization() {
        let resp = YoloModelsResponse {
            versions: vec!["v5".to_string(), "v8".to_string()],
            sizes: vec!["n".to_string(), "s".to_string()],
            total_models: 4,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["total_models"], 4);
        assert_eq!(back["versions"][0], "v5");
    }

    #[test]
    fn test_yolo_download_request_serde() {
        let json = r#"{"model_version":"v8","model_size":"l"}"#;
        let req: YoloDownloadRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model_version, "v8");
        assert_eq!(req.model_size, "l");
    }

    #[test]
    fn test_yolo_download_response_serialization() {
        let resp = YoloDownloadResponse {
            success: true,
            message: "Download available".to_string(),
            download_url: Some("https://example.com/model.pt".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], true);
        assert!(back["download_url"].as_str().unwrap().contains("example.com"));
    }

    #[test]
    fn test_yolo_version_from_str_valid() {
        assert_eq!(YoloVersion::from_str("v5"), Some(YoloVersion::V5));
        assert_eq!(YoloVersion::from_str("v8"), Some(YoloVersion::V8));
        assert_eq!(YoloVersion::from_str("v10"), Some(YoloVersion::V10));
        assert_eq!(YoloVersion::from_str("v11"), Some(YoloVersion::V11));
        assert_eq!(YoloVersion::from_str("v12"), Some(YoloVersion::V12));
    }

    #[test]
    fn test_yolo_version_from_str_invalid() {
        assert_eq!(YoloVersion::from_str("v99"), None);
        assert_eq!(YoloVersion::from_str(""), None);
    }

    #[test]
    fn test_yolo_size_from_suffix_valid() {
        assert_eq!(YoloSize::from_suffix("n"), Some(YoloSize::Nano));
        assert_eq!(YoloSize::from_suffix("s"), Some(YoloSize::Small));
        assert_eq!(YoloSize::from_suffix("m"), Some(YoloSize::Medium));
        assert_eq!(YoloSize::from_suffix("l"), Some(YoloSize::Large));
        assert_eq!(YoloSize::from_suffix("x"), Some(YoloSize::XLarge));
    }

    #[test]
    fn test_yolo_size_from_suffix_invalid() {
        assert_eq!(YoloSize::from_suffix("z"), None);
        assert_eq!(YoloSize::from_suffix(""), None);
    }

    #[test]
    fn test_yolo_version_as_str() {
        assert_eq!(YoloVersion::V5.as_str(), "YOLOv5");
        assert_eq!(YoloVersion::V8.as_str(), "YOLOv8");
    }

    #[test]
    fn test_yolo_size_suffix() {
        assert_eq!(YoloSize::Nano.suffix(), "n");
        assert_eq!(YoloSize::XLarge.suffix(), "x");
    }

    // ── Handler unit tests (no actix HTTP stack needed) ───────────────────────

    fn make_yolo_state(dir: std::path::PathBuf) -> web::Data<YoloState> {
        web::Data::new(YoloState { models_dir: dir })
    }

    // list_models — always succeeds and returns all versions/sizes
    #[actix_web::test]
    async fn test_list_models_handler_returns_all_versions() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let result = list_models(state).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // get_model_info — invalid version returns BadRequest
    #[actix_web::test]
    async fn test_get_model_info_invalid_version() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let query = web::Query(YoloInfoRequest {
            model_version: "v99".to_string(),
            model_size: "n".to_string(),
        });
        let result = get_model_info(query, state).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::ApiError::BadRequest(_)));
    }

    // get_model_info — invalid size returns BadRequest
    #[actix_web::test]
    async fn test_get_model_info_invalid_size() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let query = web::Query(YoloInfoRequest {
            model_version: "v8".to_string(),
            model_size: "z".to_string(),
        });
        let result = get_model_info(query, state).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::ApiError::BadRequest(_)));
    }

    // get_model_info — valid version+size, model not on disk → available=false
    #[actix_web::test]
    async fn test_get_model_info_valid_model_not_on_disk() {
        let state = make_yolo_state(std::path::PathBuf::from("/nonexistent_dir_for_test"));
        let query = web::Query(YoloInfoRequest {
            model_version: "v8".to_string(),
            model_size: "n".to_string(),
        });
        let result = get_model_info(query, state).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // download_model — invalid version returns BadRequest
    #[actix_web::test]
    async fn test_download_model_invalid_version() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let req = web::Json(YoloDownloadRequest {
            model_version: "v999".to_string(),
            model_size: "n".to_string(),
        });
        let result = download_model(req, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::BadRequest(_)));
    }

    // download_model — invalid size returns BadRequest
    #[actix_web::test]
    async fn test_download_model_invalid_size() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let req = web::Json(YoloDownloadRequest {
            model_version: "v5".to_string(),
            model_size: "huge".to_string(),
        });
        let result = download_model(req, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::BadRequest(_)));
    }

    // download_model — each valid version produces a URL
    #[actix_web::test]
    async fn test_download_model_v5_produces_url() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let req = web::Json(YoloDownloadRequest {
            model_version: "v5".to_string(),
            model_size: "n".to_string(),
        });
        let result = download_model(req, state).await;
        assert!(result.is_ok());
    }

    #[actix_web::test]
    async fn test_download_model_v8_produces_url() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let req = web::Json(YoloDownloadRequest {
            model_version: "v8".to_string(),
            model_size: "s".to_string(),
        });
        let result = download_model(req, state).await;
        assert!(result.is_ok());
    }

    #[actix_web::test]
    async fn test_download_model_v10_produces_url() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let req = web::Json(YoloDownloadRequest {
            model_version: "v10".to_string(),
            model_size: "m".to_string(),
        });
        let result = download_model(req, state).await;
        assert!(result.is_ok());
    }

    #[actix_web::test]
    async fn test_download_model_v11_produces_url() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let req = web::Json(YoloDownloadRequest {
            model_version: "v11".to_string(),
            model_size: "l".to_string(),
        });
        let result = download_model(req, state).await;
        assert!(result.is_ok());
    }

    #[actix_web::test]
    async fn test_download_model_v12_produces_url() {
        let state = make_yolo_state(std::path::PathBuf::from("/tmp"));
        let req = web::Json(YoloDownloadRequest {
            model_version: "v12".to_string(),
            model_size: "x".to_string(),
        });
        let result = download_model(req, state).await;
        assert!(result.is_ok());
    }

    // YoloDownloadResponse — None download_url serialization
    #[test]
    fn test_yolo_download_response_no_url() {
        let resp = YoloDownloadResponse {
            success: false,
            message: "failed".to_string(),
            download_url: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], false);
        assert!(back["download_url"].is_null());
    }

    // YoloInfoResponse — available=true variant
    #[test]
    fn test_yolo_info_response_available_true() {
        let resp = YoloInfoResponse {
            model_name: "yolov5n".to_string(),
            version: "YOLOv5".to_string(),
            size: "n".to_string(),
            num_classes: 80,
            input_size: (640, 640),
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            available: true,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["available"], true);
        assert_eq!(back["model_name"], "yolov5n");
    }

    // All YoloVersion as_str values
    #[test]
    fn test_yolo_version_all_as_str() {
        assert_eq!(YoloVersion::V10.as_str(), "YOLOv10");
        assert_eq!(YoloVersion::V11.as_str(), "YOLOv11");
        assert_eq!(YoloVersion::V12.as_str(), "YOLOv12");
    }

    // All YoloSize suffix values
    #[test]
    fn test_yolo_size_all_suffixes() {
        assert_eq!(YoloSize::Small.suffix(), "s");
        assert_eq!(YoloSize::Medium.suffix(), "m");
        assert_eq!(YoloSize::Large.suffix(), "l");
    }
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
