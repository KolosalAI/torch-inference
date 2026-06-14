use crate::core::image_security::{ImageSecurityResult, ImageSecurityValidator, SecurityLevel};
use crate::error::ApiError;
use actix_multipart::Multipart;
use actix_web::{HttpResponse, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ImageSecurityRequest {
    pub security_level: String,
}

#[derive(Debug, Serialize)]
pub struct ImageProcessResponse {
    pub success: bool,
    pub image_base64: Option<String>,
    pub security_result: ImageSecurityResult,
    pub processing_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct ImageValidationResponse {
    pub security_result: ImageSecurityResult,
}

#[derive(Debug, Serialize)]
pub struct ImageSecurityStats {
    pub total_processed: u64,
    pub threats_detected: u64,
    pub threats_blocked: u64,
    pub average_confidence: f32,
    pub threat_types: Vec<ThreatTypeStat>,
}

#[derive(Debug, Serialize)]
pub struct ThreatTypeStat {
    pub threat_type: String,
    pub count: u64,
}

#[derive(Debug, Serialize)]
pub struct ImageHealthResponse {
    pub status: String,
    pub image_backend: String,
    pub security_enabled: bool,
    pub supported_formats: Vec<String>,
}

impl SecurityLevel {
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "low" => Ok(SecurityLevel::Low),
            "medium" => Ok(SecurityLevel::Medium),
            "high" => Ok(SecurityLevel::High),
            "maximum" => Ok(SecurityLevel::Maximum),
            _ => Err(format!("Invalid security level: {}", s)),
        }
    }
}

pub async fn process_image_secure(mut payload: Multipart) -> Result<HttpResponse, ApiError> {
    let start_time = std::time::Instant::now();
    let mut image_data = Vec::new();
    let mut security_level = SecurityLevel::Medium;

    // Parse multipart data
    while let Some(item) = payload.next().await {
        let mut field = item.map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let field_name = field
            .content_disposition()
            .and_then(|cd| cd.get_name())
            .unwrap_or("")
            .to_string();
        let field_name = field_name.as_str();

        if field_name == "security_level" {
            let mut level_str = String::new();
            while let Some(chunk) = field.next().await {
                let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                level_str.push_str(&String::from_utf8_lossy(&data));
            }
            security_level =
                SecurityLevel::from_string(&level_str).map_err(ApiError::BadRequest)?;
        } else if field_name == "image" {
            while let Some(chunk) = field.next().await {
                let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                image_data.extend_from_slice(&data);
            }
        }
    }

    if image_data.is_empty() {
        return Err(ApiError::BadRequest("No image data provided".to_string()));
    }

    let validator = ImageSecurityValidator::new();

    // Validate security
    let security_result = validator
        .validate(&image_data, security_level.clone())
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    // If safe, process and sanitize
    let image_base64 = if security_result.is_safe {
        let img = image::load_from_memory(&image_data)
            .map_err(|e| ApiError::BadRequest(e.to_string()))?;

        let sanitized = validator
            .sanitize(&img, security_level)
            .map_err(|e| ApiError::InternalError(e.to_string()))?;

        // Convert to PNG bytes
        let mut png_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut png_data);
        sanitized
            .write_to(&mut cursor, image::ImageFormat::Png)
            .map_err(|e| ApiError::InternalError(e.to_string()))?;

        Some(base64_encode(&png_data))
    } else {
        None
    };

    let processing_time_ms = start_time.elapsed().as_millis() as u64;

    Ok(HttpResponse::Ok().json(ImageProcessResponse {
        success: security_result.is_safe,
        image_base64,
        security_result,
        processing_time_ms,
    }))
}

pub async fn validate_image_security(mut payload: Multipart) -> Result<HttpResponse, ApiError> {
    let mut image_data = Vec::new();
    let mut security_level = SecurityLevel::Medium;

    while let Some(item) = payload.next().await {
        let mut field = item.map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let field_name = field
            .content_disposition()
            .and_then(|cd| cd.get_name())
            .unwrap_or("")
            .to_string();
        let field_name = field_name.as_str();

        if field_name == "security_level" {
            let mut level_str = String::new();
            while let Some(chunk) = field.next().await {
                let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                level_str.push_str(&String::from_utf8_lossy(&data));
            }
            security_level =
                SecurityLevel::from_string(&level_str).map_err(ApiError::BadRequest)?;
        } else if field_name == "image" {
            while let Some(chunk) = field.next().await {
                let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                image_data.extend_from_slice(&data);
            }
        }
    }

    if image_data.is_empty() {
        return Err(ApiError::BadRequest("No image data provided".to_string()));
    }

    let validator = ImageSecurityValidator::new();
    let security_result = validator
        .validate(&image_data, security_level)
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(ImageValidationResponse { security_result }))
}

pub async fn get_image_security_stats() -> Result<HttpResponse, ApiError> {
    // Placeholder - would track actual statistics
    Ok(HttpResponse::Ok().json(ImageSecurityStats {
        total_processed: 0,
        threats_detected: 0,
        threats_blocked: 0,
        average_confidence: 0.0,
        threat_types: vec![],
    }))
}

pub async fn image_health() -> Result<HttpResponse, ApiError> {
    Ok(HttpResponse::Ok().json(ImageHealthResponse {
        status: "ok".to_string(),
        image_backend: "image + imageproc".to_string(),
        security_enabled: true,
        supported_formats: vec![
            "jpg".to_string(),
            "jpeg".to_string(),
            "png".to_string(),
            "bmp".to_string(),
            "gif".to_string(),
            "webp".to_string(),
        ],
    }))
}

fn base64_encode(data: &[u8]) -> String {
    use base64::{engine::general_purpose, Engine as _};
    general_purpose::STANDARD.encode(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::http::StatusCode;
    use actix_web::{test, web, App};

    // ── actix_web handler tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_image_health_returns_200() {
        let app =
            test::init_service(App::new().route("/image/health", web::get().to(image_health)))
                .await;
        let req = test::TestRequest::get().uri("/image/health").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_image_health_response_body() {
        let app =
            test::init_service(App::new().route("/image/health", web::get().to(image_health)))
                .await;
        let req = test::TestRequest::get().uri("/image/health").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["status"], "ok");
        assert_eq!(body["security_enabled"], true);
        assert!(body["supported_formats"].is_array());
        let formats = body["supported_formats"].as_array().unwrap();
        assert!(formats.len() >= 4);
    }

    #[tokio::test]
    async fn test_image_health_supported_formats_content() {
        let app =
            test::init_service(App::new().route("/image/health", web::get().to(image_health)))
                .await;
        let req = test::TestRequest::get().uri("/image/health").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let formats: Vec<&str> = body["supported_formats"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(formats.contains(&"jpg"));
        assert!(formats.contains(&"png"));
    }

    #[tokio::test]
    async fn test_get_image_security_stats_returns_200() {
        let app = test::init_service(
            App::new().route("/image/stats", web::get().to(get_image_security_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/image/stats").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_image_security_stats_response_body() {
        let app = test::init_service(
            App::new().route("/image/stats", web::get().to(get_image_security_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/image/stats").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["total_processed"], 0);
        assert_eq!(body["threats_detected"], 0);
        assert_eq!(body["threats_blocked"], 0);
        assert!(body["threat_types"].is_array());
    }

    #[tokio::test]
    async fn test_get_image_security_stats_initial_values() {
        let app = test::init_service(
            App::new().route("/image/stats", web::get().to(get_image_security_stats)),
        )
        .await;
        let req = test::TestRequest::get().uri("/image/stats").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        // Stats start at zero placeholders
        assert_eq!(body["threats_blocked"].as_u64().unwrap(), 0);
        let threat_types = body["threat_types"].as_array().unwrap();
        assert!(threat_types.is_empty());
    }

    // ── SecurityLevel::from_string ────────────────────────────────────────────

    #[tokio::test]
    async fn test_security_level_from_string_valid() {
        assert!(matches!(
            SecurityLevel::from_string("low").unwrap(),
            SecurityLevel::Low
        ));
        assert!(matches!(
            SecurityLevel::from_string("medium").unwrap(),
            SecurityLevel::Medium
        ));
        assert!(matches!(
            SecurityLevel::from_string("high").unwrap(),
            SecurityLevel::High
        ));
        assert!(matches!(
            SecurityLevel::from_string("maximum").unwrap(),
            SecurityLevel::Maximum
        ));
    }

    #[tokio::test]
    async fn test_security_level_from_string_case_insensitive() {
        assert!(matches!(
            SecurityLevel::from_string("LOW").unwrap(),
            SecurityLevel::Low
        ));
        assert!(matches!(
            SecurityLevel::from_string("High").unwrap(),
            SecurityLevel::High
        ));
        assert!(matches!(
            SecurityLevel::from_string("MAXIMUM").unwrap(),
            SecurityLevel::Maximum
        ));
    }

    #[tokio::test]
    async fn test_security_level_from_string_invalid() {
        let result = SecurityLevel::from_string("unknown");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid security level"));
    }

    #[tokio::test]
    async fn test_image_security_request_serde() {
        let json = r#"{"security_level":"high"}"#;
        let req: ImageSecurityRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.security_level, "high");
    }

    #[tokio::test]
    async fn test_image_process_response_serialization() {
        let security_result = ImageSecurityResult {
            is_safe: true,
            security_level: SecurityLevel::Medium,
            threats_detected: vec![],
            confidence: 0.99,
            sanitized: false,
        };
        let resp = ImageProcessResponse {
            success: true,
            image_base64: Some("iVBORw0KGgo=".to_string()),
            security_result,
            processing_time_ms: 42,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["success"], true);
        assert_eq!(back["processing_time_ms"], 42);
        assert!(back["image_base64"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_image_validation_response_serialization() {
        let security_result = ImageSecurityResult {
            is_safe: false,
            security_level: SecurityLevel::High,
            threats_detected: vec![],
            confidence: 0.75,
            sanitized: false,
        };
        let resp = ImageValidationResponse { security_result };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["security_result"]["is_safe"], false);
    }

    #[tokio::test]
    async fn test_threat_type_stat_serialization() {
        let stat = ThreatTypeStat {
            threat_type: "AdversarialPattern".to_string(),
            count: 5,
        };
        let json = serde_json::to_string(&stat).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["count"], 5);
        assert_eq!(back["threat_type"], "AdversarialPattern");
    }

    #[tokio::test]
    async fn test_image_security_stats_serialization() {
        let stats = ImageSecurityStats {
            total_processed: 100,
            threats_detected: 10,
            threats_blocked: 8,
            average_confidence: 0.9,
            threat_types: vec![ThreatTypeStat {
                threat_type: "InvalidFormat".to_string(),
                count: 3,
            }],
        };
        let json = serde_json::to_string(&stats).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["total_processed"], 100);
        assert_eq!(back["threats_blocked"], 8);
    }

    #[tokio::test]
    async fn test_image_health_response_serialization() {
        let resp = ImageHealthResponse {
            status: "ok".to_string(),
            image_backend: "image + imageproc".to_string(),
            security_enabled: true,
            supported_formats: vec!["jpg".to_string(), "png".to_string()],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["status"], "ok");
        assert_eq!(back["security_enabled"], true);
        assert_eq!(back["supported_formats"][1], "png");
    }

    #[tokio::test]
    async fn test_base64_encode_helper() {
        let data = b"test data";
        let encoded = base64_encode(data);
        use base64::{engine::general_purpose, Engine as _};
        assert_eq!(encoded, general_purpose::STANDARD.encode(data));
    }

    // ── Helpers for building multipart payloads ───────────────────────────────

    /// Build a raw multipart/form-data body with a boundary.
    /// Returns (body_bytes, boundary_string).
    fn build_multipart_body(parts: &[(&str, &str, &[u8])]) -> (Vec<u8>, String) {
        let boundary = "testboundary123";
        let mut body = Vec::new();
        for (name, content_type, data) in parts {
            body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
            body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{}\"\r\nContent-Type: {}\r\n\r\n",
                    name, content_type
                )
                .as_bytes(),
            );
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());
        (body, boundary.to_string())
    }

    /// Create a minimal valid 1x1 PNG image in bytes.
    fn tiny_png() -> Vec<u8> {
        use image::{DynamicImage, ImageBuffer, Rgb};
        let buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(2, 2, Rgb([100u8, 150u8, 200u8]));
        let img = DynamicImage::ImageRgb8(buf);
        let mut out = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut out),
            image::ImageFormat::Png,
        )
        .unwrap();
        out
    }

    // ── process_image_secure tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_process_image_secure_no_image_returns_bad_request() {
        let app = test::init_service(
            App::new().route("/image/process", web::post().to(process_image_secure)),
        )
        .await;
        // Send an empty multipart – no image field.
        let boundary = "emptyboundary";
        let body = format!("--{}--\r\n", boundary);
        let req = test::TestRequest::post()
            .uri("/image/process")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_process_image_secure_with_valid_image_returns_200() {
        let app = test::init_service(
            App::new().route("/image/process", web::post().to(process_image_secure)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"low"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/process")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_process_image_secure_with_medium_security_level() {
        let app = test::init_service(
            App::new().route("/image/process", web::post().to(process_image_secure)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"medium"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/process")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_process_image_secure_with_high_security_level() {
        let app = test::init_service(
            App::new().route("/image/process", web::post().to(process_image_secure)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"high"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/process")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_process_image_secure_with_maximum_security_level() {
        let app = test::init_service(
            App::new().route("/image/process", web::post().to(process_image_secure)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"maximum"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/process")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_process_image_secure_invalid_security_level_returns_bad_request() {
        let app = test::init_service(
            App::new().route("/image/process", web::post().to(process_image_secure)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"invalid_level"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/process")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_process_image_secure_response_body_has_success_field() {
        let app = test::init_service(
            App::new().route("/image/process", web::post().to(process_image_secure)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"low"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/process")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["success"].is_boolean());
        assert!(body["security_result"].is_object());
        assert!(body["processing_time_ms"].is_number());
    }

    #[tokio::test]
    async fn test_process_image_secure_image_only_no_security_level() {
        // No security_level field → defaults to Medium.
        let app = test::init_service(
            App::new().route("/image/process", web::post().to(process_image_secure)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[("image", "image/png", &image_bytes)]);

        let req = test::TestRequest::post()
            .uri("/image/process")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── validate_image_security tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_validate_image_security_no_image_returns_bad_request() {
        let app = test::init_service(
            App::new().route("/image/validate", web::post().to(validate_image_security)),
        )
        .await;

        let boundary = "emptyboundary2";
        let body = format!("--{}--\r\n", boundary);
        let req = test::TestRequest::post()
            .uri("/image/validate")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_validate_image_security_with_valid_image_returns_200() {
        let app = test::init_service(
            App::new().route("/image/validate", web::post().to(validate_image_security)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"low"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/validate")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_validate_image_security_response_has_security_result() {
        let app = test::init_service(
            App::new().route("/image/validate", web::post().to(validate_image_security)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"medium"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/validate")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert!(body["security_result"].is_object());
        assert!(body["security_result"]["is_safe"].is_boolean());
    }

    #[tokio::test]
    async fn test_validate_image_security_invalid_level_returns_bad_request() {
        let app = test::init_service(
            App::new().route("/image/validate", web::post().to(validate_image_security)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"bogus"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/validate")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_validate_image_security_image_only_default_level() {
        // No security_level field → defaults to Medium.
        let app = test::init_service(
            App::new().route("/image/validate", web::post().to(validate_image_security)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[("image", "image/png", &image_bytes)]);

        let req = test::TestRequest::post()
            .uri("/image/validate")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_validate_image_security_high_level() {
        let app = test::init_service(
            App::new().route("/image/validate", web::post().to(validate_image_security)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"high"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/validate")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_validate_image_security_maximum_level() {
        let app = test::init_service(
            App::new().route("/image/validate", web::post().to(validate_image_security)),
        )
        .await;

        let image_bytes = tiny_png();
        let (body, boundary) = build_multipart_body(&[
            ("security_level", "text/plain", b"maximum"),
            ("image", "image/png", &image_bytes),
        ]);

        let req = test::TestRequest::post()
            .uri("/image/validate")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Sends a non-"image" field so image_data stays empty → covers line 91
    /// (the `if image_data.is_empty()` return in process_image_secure).
    #[tokio::test]
    async fn test_process_image_secure_non_image_field_returns_bad_request() {
        let app = test::init_service(
            App::new().route("/image/process", web::post().to(process_image_secure)),
        )
        .await;

        let (body, boundary) = build_multipart_body(&[("other_field", "text/plain", b"data")]);
        let req = test::TestRequest::post()
            .uri("/image/process")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Sends a non-"image" field so image_data stays empty → covers line 157
    /// (the `if image_data.is_empty()` return in validate_image_security).
    #[tokio::test]
    async fn test_validate_image_security_non_image_field_returns_bad_request() {
        let app = test::init_service(
            App::new().route("/image/validate", web::post().to(validate_image_security)),
        )
        .await;

        let (body, boundary) = build_multipart_body(&[("other_field", "text/plain", b"data")]);
        let req = test::TestRequest::post()
            .uri("/image/validate")
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={}", boundary),
            ))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
