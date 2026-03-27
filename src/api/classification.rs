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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::image_classifier::{ClassificationResult, TopKResults};


    #[test]
    fn test_classify_request_serde_full() {
        let json = r#"{"image_path":"/images/cat.jpg","top_k":3}"#;
        let req: ClassifyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.image_path.as_deref(), Some("/images/cat.jpg"));
        assert_eq!(req.top_k, Some(3));
    }

    #[test]
    fn test_classify_request_serde_minimal() {
        let json = r#"{}"#;
        let req: ClassifyRequest = serde_json::from_str(json).unwrap();
        assert!(req.image_path.is_none());
        assert!(req.top_k.is_none());
    }

    #[test]
    fn test_classify_response_success_serialization() {
        let top_k = TopKResults {
            predictions: vec![
                ClassificationResult {
                    label: "cat".to_string(),
                    confidence: 0.95,
                    class_id: 281,
                },
                ClassificationResult {
                    label: "dog".to_string(),
                    confidence: 0.04,
                    class_id: 207,
                },
            ],
            inference_time_ms: 12.3,
        };
        let resp = ClassifyResponse {
            success: true,
            results: Some(top_k),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], true);
        assert_eq!(back["results"]["predictions"][0]["label"], "cat");
        assert!((back["results"]["inference_time_ms"].as_f64().unwrap() - 12.3).abs() < 1e-5);
    }

    #[test]
    fn test_classify_response_failure_serialization() {
        let resp = ClassifyResponse {
            success: false,
            results: None,
            error: Some("Classification failed".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], false);
        assert_eq!(back["error"], "Classification failed");
        assert!(back["results"].is_null());
    }

    #[test]
    fn test_classification_result_serde_roundtrip() {
        let result = ClassificationResult {
            label: "tabby_cat".to_string(),
            confidence: 0.87,
            class_id: 281,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: ClassificationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.label, "tabby_cat");
        assert_eq!(back.class_id, 281);
        assert!((back.confidence - 0.87).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_results_serde_roundtrip() {
        let results = TopKResults {
            predictions: vec![],
            inference_time_ms: 5.0,
        };
        let json = serde_json::to_string(&results).unwrap();
        let back: TopKResults = serde_json::from_str(&json).unwrap();
        assert!(back.predictions.is_empty());
        assert!((back.inference_time_ms - 5.0).abs() < 1e-5);
    }

    // ── Additional coverage tests ─────────────────────────────────────────────

    #[test]
    fn test_classify_request_only_image_path() {
        let json = r#"{"image_path":"/tmp/test.jpg"}"#;
        let req: ClassifyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.image_path.as_deref(), Some("/tmp/test.jpg"));
        assert!(req.top_k.is_none());
    }

    #[test]
    fn test_classify_request_only_top_k() {
        let json = r#"{"top_k":10}"#;
        let req: ClassifyRequest = serde_json::from_str(json).unwrap();
        assert!(req.image_path.is_none());
        assert_eq!(req.top_k, Some(10));
    }

    #[test]
    fn test_classify_response_success_no_results() {
        // success=true but results=None (edge case)
        let resp = ClassifyResponse {
            success: true,
            results: None,
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], true);
        assert!(back["results"].is_null());
        assert!(back["error"].is_null());
    }

    #[test]
    fn test_classify_response_with_error_string() {
        let resp = ClassifyResponse {
            success: false,
            results: None,
            error: Some("PyTorch not enabled".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["error"], "PyTorch not enabled");
    }

    #[test]
    fn test_classification_result_confidence_boundary() {
        let result_zero = ClassificationResult {
            label: "unknown".to_string(),
            confidence: 0.0,
            class_id: 0,
        };
        let result_one = ClassificationResult {
            label: "cat".to_string(),
            confidence: 1.0,
            class_id: 281,
        };
        assert!((result_zero.confidence - 0.0).abs() < 1e-6);
        assert!((result_one.confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_results_multiple_predictions() {
        let results = TopKResults {
            predictions: vec![
                ClassificationResult { label: "cat".to_string(), confidence: 0.8, class_id: 281 },
                ClassificationResult { label: "dog".to_string(), confidence: 0.15, class_id: 207 },
                ClassificationResult { label: "bird".to_string(), confidence: 0.05, class_id: 10 },
            ],
            inference_time_ms: 7.5,
        };
        assert_eq!(results.predictions.len(), 3);
        assert_eq!(results.predictions[0].label, "cat");
        assert_eq!(results.predictions[2].class_id, 10);
    }

    // ── Handler tests via direct function calls ───────────────────────────────
    //
    // We can only test handlers that return early before touching the model.
    // ImageClassifier::new() always errors without the `torch` feature, so
    // handler tests that need a live classifier are guarded by #[cfg(feature="torch")].

    // classify_image_file: missing image_path returns BadRequest
    // (We need a state even for this path, so skip if torch is unavailable)
    #[cfg(feature = "torch")]
    #[actix_web::test]
    async fn test_classify_image_file_missing_path() {
        // This test only runs when torch is available so we can construct the state.
        // The early-return logic is independent of the actual model.
        use std::path::Path;
        let classifier = Arc::new(
            ImageClassifier::new(
                Path::new("/nonexistent_model_path"),
                vec!["cat".to_string()],
                None,
                None,
            )
            .expect("classifier"),
        );
        let state = web::Data::new(ImageClassificationState { classifier });
        let req = web::Json(ClassifyRequest {
            image_path: None,
            top_k: Some(5),
        });
        let result = classify_image_file(req, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::BadRequest(_)));
    }

    // classify_image_file: path not found returns NotFound
    #[cfg(feature = "torch")]
    #[actix_web::test]
    async fn test_classify_image_file_path_not_found() {
        use std::path::Path;
        let classifier = Arc::new(
            ImageClassifier::new(
                Path::new("/nonexistent_model_path"),
                vec!["cat".to_string()],
                None,
                None,
            )
            .expect("classifier"),
        );
        let state = web::Data::new(ImageClassificationState { classifier });
        let req = web::Json(ClassifyRequest {
            image_path: Some("/tmp/this_image_does_not_exist_xyz.jpg".to_string()),
            top_k: Some(5),
        });
        let result = classify_image_file(req, state).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::error::ApiError::NotFound(_)));
    }
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
