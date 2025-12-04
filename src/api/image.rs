use actix_web::{web, HttpResponse, Result};
use actix_multipart::Multipart;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use crate::core::image_security::{ImageSecurityValidator, SecurityLevel, ImageSecurityResult};
use crate::error::ApiError;

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

pub async fn process_image_secure(
    mut payload: Multipart,
) -> Result<HttpResponse, ApiError> {
    let start_time = std::time::Instant::now();
    let mut image_data = Vec::new();
    let mut security_level = SecurityLevel::Medium;

    // Parse multipart data
    while let Some(item) = payload.next().await {
        let mut field = item.map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let content_disposition = field.content_disposition();
        let field_name = content_disposition.get_name().unwrap_or("");

        if field_name == "security_level" {
            let mut level_str = String::new();
            while let Some(chunk) = field.next().await {
                let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                level_str.push_str(&String::from_utf8_lossy(&data));
            }
            security_level = SecurityLevel::from_string(&level_str)
                .map_err(|e| ApiError::BadRequest(e))?;
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
    let security_result = validator.validate(&image_data, security_level.clone())
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    // If safe, process and sanitize
    let image_base64 = if security_result.is_safe {
        let img = image::load_from_memory(&image_data)
            .map_err(|e| ApiError::BadRequest(e.to_string()))?;
        
        let sanitized = validator.sanitize(&img, security_level)
            .map_err(|e| ApiError::InternalError(e.to_string()))?;

        // Convert to PNG bytes
        let mut png_data = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut png_data);
        sanitized.write_to(&mut cursor, image::ImageOutputFormat::Png)
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

pub async fn validate_image_security(
    mut payload: Multipart,
) -> Result<HttpResponse, ApiError> {
    let mut image_data = Vec::new();
    let mut security_level = SecurityLevel::Medium;

    while let Some(item) = payload.next().await {
        let mut field = item.map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let content_disposition = field.content_disposition();
        let field_name = content_disposition.get_name().unwrap_or("");

        if field_name == "security_level" {
            let mut level_str = String::new();
            while let Some(chunk) = field.next().await {
                let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                level_str.push_str(&String::from_utf8_lossy(&data));
            }
            security_level = SecurityLevel::from_string(&level_str)
                .map_err(|e| ApiError::BadRequest(e))?;
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
    let security_result = validator.validate(&image_data, security_level)
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(HttpResponse::Ok().json(ImageValidationResponse {
        security_result,
    }))
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
    use base64::{Engine as _, engine::general_purpose};
    general_purpose::STANDARD.encode(data)
}
