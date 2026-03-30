#![allow(dead_code)]
use thiserror::Error;
use actix_web::{error::ResponseError, http::StatusCode, HttpResponse};

#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Model load failed: {0}")]
    ModelLoadError(String),
    
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Internal error: {0}")]
    InternalError(String),
    
    #[error("GPU error: {0}")]
    GpuError(String),
    
    #[error("Timeout")]
    Timeout,
}

pub type Result<T> = std::result::Result<T, InferenceError>;

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Bad request: {0}")]
    BadRequest(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Internal server error: {0}")]
    InternalError(String),
    
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    
    #[error("Forbidden: {0}")]
    Forbidden(String),
}

impl ResponseError for ApiError {
    fn error_response(&self) -> HttpResponse {
        let status_code = self.status_code();
        HttpResponse::build(status_code).json(serde_json::json!({
            "error": self.to_string(),
            "status": status_code.as_u16()
        }))
    }

    fn status_code(&self) -> StatusCode {
        match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            ApiError::Forbidden(_) => StatusCode::FORBIDDEN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_error_model_not_found() {
        let error = InferenceError::ModelNotFound("test_model".to_string());
        assert_eq!(error.to_string(), "Model not found: test_model");
    }

    #[test]
    fn test_inference_error_model_load_error() {
        let error = InferenceError::ModelLoadError("load failed".to_string());
        assert_eq!(error.to_string(), "Model load failed: load failed");
    }

    #[test]
    fn test_inference_error_invalid_input() {
        let error = InferenceError::InvalidInput("bad input".to_string());
        assert_eq!(error.to_string(), "Invalid input: bad input");
    }

    #[test]
    fn test_inference_error_timeout() {
        let error = InferenceError::Timeout;
        assert_eq!(error.to_string(), "Timeout");
    }

    #[test]
    fn test_api_error_bad_request() {
        let error = ApiError::BadRequest("invalid data".to_string());
        assert_eq!(error.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_api_error_not_found() {
        let error = ApiError::NotFound("resource not found".to_string());
        assert_eq!(error.status_code(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_api_error_unauthorized() {
        let error = ApiError::Unauthorized("not authorized".to_string());
        assert_eq!(error.status_code(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_api_error_forbidden() {
        let error = ApiError::Forbidden("access denied".to_string());
        assert_eq!(error.status_code(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn test_api_error_internal() {
        let error = ApiError::InternalError("server error".to_string());
        assert_eq!(error.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_inference_error_from_io_error() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let error: InferenceError = io_error.into();
        assert!(matches!(error, InferenceError::IoError(_)));
    }

    #[test]
    fn test_inference_error_from_serde_error() {
        let json_str = "{invalid json}";
        let serde_error = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let error: InferenceError = serde_error.into();
        assert!(matches!(error, InferenceError::SerializationError(_)));
    }

    // ===== Uncovered InferenceError variants =====

    #[test]
    fn test_inference_error_inference_failed() {
        let error = InferenceError::InferenceFailed("inference crashed".to_string());
        assert_eq!(error.to_string(), "Inference failed: inference crashed");
    }

    #[test]
    fn test_inference_error_config_error() {
        let error = InferenceError::ConfigError("missing key".to_string());
        assert_eq!(error.to_string(), "Configuration error: missing key");
    }

    #[test]
    fn test_inference_error_authentication_failed() {
        let error = InferenceError::AuthenticationFailed("invalid token".to_string());
        assert_eq!(error.to_string(), "Authentication failed: invalid token");
    }

    #[test]
    fn test_inference_error_internal_error() {
        let error = InferenceError::InternalError("panic occurred".to_string());
        assert_eq!(error.to_string(), "Internal error: panic occurred");
    }

    #[test]
    fn test_inference_error_gpu_error() {
        let error = InferenceError::GpuError("CUDA out of memory".to_string());
        assert_eq!(error.to_string(), "GPU error: CUDA out of memory");
    }

    #[test]
    fn test_inference_error_io_error_display() {
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "permission denied");
        let error: InferenceError = io_error.into();
        let msg = error.to_string();
        assert!(msg.starts_with("IO error:"));
    }

    #[test]
    fn test_inference_error_serialization_error_display() {
        let json_str = "not valid json at all ~~~";
        let serde_error = serde_json::from_str::<serde_json::Value>(json_str).unwrap_err();
        let error: InferenceError = serde_error.into();
        let msg = error.to_string();
        assert!(msg.starts_with("Serialization error:"));
    }

    #[test]
    fn test_inference_error_debug() {
        let error = InferenceError::Timeout;
        let debug = format!("{:?}", error);
        assert!(debug.contains("Timeout"));
    }

    // ===== Uncovered ApiError variants =====

    #[test]
    fn test_api_error_display_bad_request() {
        let error = ApiError::BadRequest("missing field".to_string());
        assert_eq!(error.to_string(), "Bad request: missing field");
    }

    #[test]
    fn test_api_error_display_not_found() {
        let error = ApiError::NotFound("model not found".to_string());
        assert_eq!(error.to_string(), "Not found: model not found");
    }

    #[test]
    fn test_api_error_display_internal() {
        let error = ApiError::InternalError("crash".to_string());
        assert_eq!(error.to_string(), "Internal server error: crash");
    }

    #[test]
    fn test_api_error_display_unauthorized() {
        let error = ApiError::Unauthorized("expired token".to_string());
        assert_eq!(error.to_string(), "Unauthorized: expired token");
    }

    #[test]
    fn test_api_error_display_forbidden() {
        let error = ApiError::Forbidden("no access".to_string());
        assert_eq!(error.to_string(), "Forbidden: no access");
    }

    #[test]
    fn test_api_error_debug() {
        let error = ApiError::NotFound("x".to_string());
        let debug = format!("{:?}", error);
        assert!(debug.contains("NotFound"));
    }

    #[test]
    fn test_api_error_error_response_bad_request() {
        let error = ApiError::BadRequest("bad".to_string());
        let response = error.error_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_api_error_error_response_not_found() {
        let error = ApiError::NotFound("missing".to_string());
        let response = error.error_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_api_error_error_response_internal() {
        let error = ApiError::InternalError("oops".to_string());
        let response = error.error_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_api_error_error_response_unauthorized() {
        let error = ApiError::Unauthorized("no token".to_string());
        let response = error.error_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_api_error_error_response_forbidden() {
        let error = ApiError::Forbidden("denied".to_string());
        let response = error.error_response();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }
}
