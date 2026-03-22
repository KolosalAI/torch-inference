use actix_web::{web, HttpResponse, Result};
use actix_multipart::Multipart;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::path::PathBuf;

use crate::core::image_classifier::{ImageClassifier, TopKResults};
use crate::error::ApiError;

pub struct ImageClassificationState {
    pub classifier: Arc<ImageClassifier>,
}

#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    pub image_path: Option<String>,
    pub top_k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ClassifyResponse {
    pub success: bool,
    pub results: Option<TopKResults>,
    pub error: Option<String>,
}

/// POST /api/classify/image - Classify an image file
pub async fn classify_image_file(
    req: web::Json<ClassifyRequest>,
    state: web::Data<ImageClassificationState>,
) -> Result<HttpResponse, ApiError> {
    let image_path = req.image_path.as_ref()
        .ok_or_else(|| ApiError::BadRequest("image_path is required".to_string()))?;
    
    let path = PathBuf::from(image_path);
    if !path.exists() {
        return Err(ApiError::NotFound(format!("Image file not found: {}", image_path)));
    }
    
    let top_k = req.top_k.unwrap_or(5);
    
    match state.classifier.classify(&path, top_k) {
        Ok(results) => {
            Ok(HttpResponse::Ok().json(ClassifyResponse {
                success: true,
                results: Some(results),
                error: None,
            }))
        }
        Err(e) => {
            Ok(HttpResponse::Ok().json(ClassifyResponse {
                success: false,
                results: None,
                error: Some(e.to_string()),
            }))
        }
    }
}

/// POST /api/classify/upload - Upload and classify an image
pub async fn classify_image_upload(
    mut payload: Multipart,
    state: web::Data<ImageClassificationState>,
) -> Result<HttpResponse, ApiError> {
    let mut image_bytes = Vec::new();
    let mut top_k = 5;
    
    // Process multipart form data
    while let Some(item) = payload.next().await {
        let mut field = item.map_err(|e| ApiError::BadRequest(e.to_string()))?;
        
        let content_disposition = field.content_disposition();
        let field_name = content_disposition.get_name().unwrap_or("");
        
        match field_name {
            "image" => {
                // Read image data
                while let Some(chunk) = field.next().await {
                    let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                    image_bytes.extend_from_slice(&data);
                }
            }
            "top_k" => {
                // Read top_k parameter
                let mut data = Vec::new();
                while let Some(chunk) = field.next().await {
                    let chunk_data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                    data.extend_from_slice(&chunk_data);
                }
                if let Ok(s) = String::from_utf8(data) {
                    top_k = s.parse().unwrap_or(5);
                }
            }
            _ => {}
        }
    }
    
    if image_bytes.is_empty() {
        return Err(ApiError::BadRequest("No image data provided".to_string()));
    }
    
    match state.classifier.classify_bytes(&image_bytes, top_k) {
        Ok(results) => {
            Ok(HttpResponse::Ok().json(ClassifyResponse {
                success: true,
                results: Some(results),
                error: None,
            }))
        }
        Err(e) => {
            Ok(HttpResponse::Ok().json(ClassifyResponse {
                success: false,
                results: None,
                error: Some(e.to_string()),
            }))
        }
    }
}

/// GET /api/classify/classes - Get available classes
pub async fn get_classes(
    state: web::Data<ImageClassificationState>,
) -> Result<HttpResponse, ApiError> {
    let num_classes = state.classifier.num_classes();
    
    // Get first 10 and last 10 classes as examples
    let mut classes = Vec::new();
    for i in 0..num_classes.min(10) {
        if let Some(label) = state.classifier.get_label(i) {
            classes.push(serde_json::json!({
                "id": i,
                "label": label
            }));
        }
    }
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "total_classes": num_classes,
        "sample_classes": classes,
        "message": format!("Classifier supports {} classes", num_classes)
    })))
}

/// GET /api/classify/info - Get classifier information
pub async fn get_classifier_info(
    state: web::Data<ImageClassificationState>,
) -> Result<HttpResponse, ApiError> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "model": "Image Classifier",
        "num_classes": state.classifier.num_classes(),
        "input_size": "224x224",
        "framework": "PyTorch",
        "status": "ready"
    })))
}

// Configure routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/classify")
            .route("/image", web::post().to(classify_image_file))
            .route("/upload", web::post().to(classify_image_upload))
            .route("/classes", web::get().to(get_classes))
            .route("/info", web::get().to(get_classifier_info))
    );
}
