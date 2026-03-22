use serde_json::Value;
use validator::{Validate, ValidationError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ValidationErrorType {
    #[error("Input too large: {0} bytes (max: {1})")]
    InputTooLarge(usize, usize),
    
    #[error("Too many inputs: {0} (max: {1})")]
    TooManyInputs(usize, usize),
    
    #[error("Invalid model name: {0}")]
    InvalidModelName(String),
    
    #[error("Invalid value: {0}")]
    InvalidValue(String),
    
    #[error("Malicious pattern detected: {0}")]
    MaliciousPattern(String),
    
    #[error("Validation error: {0}")]
    ValidationFailed(String),
}

pub struct RequestValidator {
    max_input_size_bytes: usize,
    max_inputs_count: usize,
    max_model_name_length: usize,
    allowed_model_pattern: regex::Regex,
}

impl Default for RequestValidator {
    fn default() -> Self {
        Self {
            max_input_size_bytes: 10_000_000, // 10MB
            max_inputs_count: 100,
            max_model_name_length: 256,
            allowed_model_pattern: regex::Regex::new(r"^[a-zA-Z0-9_\-\.]+$").unwrap(),
        }
    }
}

impl RequestValidator {
    pub fn new(
        max_input_size_bytes: usize,
        max_inputs_count: usize,
        max_model_name_length: usize,
    ) -> Self {
        Self {
            max_input_size_bytes,
            max_inputs_count,
            max_model_name_length,
            allowed_model_pattern: regex::Regex::new(r"^[a-zA-Z0-9_\-\.]+$").unwrap(),
        }
    }

    /// Validate model name
    pub fn validate_model_name(&self, model_name: &str) -> Result<(), ValidationErrorType> {
        if model_name.is_empty() {
            return Err(ValidationErrorType::InvalidModelName(
                "Model name cannot be empty".to_string(),
            ));
        }

        if model_name.len() > self.max_model_name_length {
            return Err(ValidationErrorType::InvalidModelName(format!(
                "Model name too long: {} (max: {})",
                model_name.len(),
                self.max_model_name_length
            )));
        }

        if !self.allowed_model_pattern.is_match(model_name) {
            return Err(ValidationErrorType::InvalidModelName(
                "Model name contains invalid characters".to_string(),
            ));
        }

        // Check for path traversal attempts
        if model_name.contains("..") || model_name.contains('/') || model_name.contains('\\') {
            return Err(ValidationErrorType::MaliciousPattern(
                "Path traversal attempt detected".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate inputs array
    pub fn validate_inputs(&self, inputs: &[Value]) -> Result<(), ValidationErrorType> {
        if inputs.is_empty() {
            return Err(ValidationErrorType::InvalidValue(
                "Inputs array cannot be empty".to_string(),
            ));
        }

        if inputs.len() > self.max_inputs_count {
            return Err(ValidationErrorType::TooManyInputs(
                inputs.len(),
                self.max_inputs_count,
            ));
        }

        for (idx, input) in inputs.iter().enumerate() {
            self.validate_input_size(input, idx)?;
            self.sanitize_malicious_patterns(input, idx)?;
        }

        Ok(())
    }

    /// Validate single input size
    fn validate_input_size(&self, value: &Value, idx: usize) -> Result<(), ValidationErrorType> {
        let size = serde_json::to_string(value)
            .map(|s| s.len())
            .unwrap_or(0);

        if size > self.max_input_size_bytes {
            return Err(ValidationErrorType::InputTooLarge(
                size,
                self.max_input_size_bytes,
            ));
        }

        Ok(())
    }

    /// Check for malicious patterns
    fn sanitize_malicious_patterns(
        &self,
        value: &Value,
        idx: usize,
    ) -> Result<(), ValidationErrorType> {
        let value_str = serde_json::to_string(value).unwrap_or_default();

        // Check for script injection
        if value_str.to_lowercase().contains("<script")
            || value_str.to_lowercase().contains("javascript:")
        {
            return Err(ValidationErrorType::MaliciousPattern(
                "Script injection attempt detected".to_string(),
            ));
        }

        // Check for SQL injection patterns
        if value_str.contains("' OR '1'='1")
            || value_str.contains("'; DROP TABLE")
            || value_str.contains("-- ")
        {
            return Err(ValidationErrorType::MaliciousPattern(
                "SQL injection attempt detected".to_string(),
            ));
        }

        // Check for command injection
        if value_str.contains("$(") || value_str.contains("`") || value_str.contains("|") {
            return Err(ValidationErrorType::MaliciousPattern(
                "Command injection attempt detected".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate priority value
    pub fn validate_priority(&self, priority: Option<i32>) -> Result<(), ValidationErrorType> {
        if let Some(p) = priority {
            if p < -10 || p > 10 {
                return Err(ValidationErrorType::InvalidValue(format!(
                    "Priority must be between -10 and 10, got: {}",
                    p
                )));
            }
        }
        Ok(())
    }

    /// Validate timeout value
    pub fn validate_timeout(&self, timeout_ms: Option<u64>) -> Result<(), ValidationErrorType> {
        if let Some(t) = timeout_ms {
            if t < 100 {
                return Err(ValidationErrorType::InvalidValue(
                    "Timeout must be at least 100ms".to_string(),
                ));
            }
            if t > 300_000 {
                return Err(ValidationErrorType::InvalidValue(
                    "Timeout cannot exceed 300 seconds".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Comprehensive validation
    pub fn validate_request(
        &self,
        model_name: &str,
        inputs: &[Value],
        priority: Option<i32>,
        timeout_ms: Option<u64>,
    ) -> Result<(), ValidationErrorType> {
        self.validate_model_name(model_name)?;
        self.validate_inputs(inputs)?;
        self.validate_priority(priority)?;
        self.validate_timeout(timeout_ms)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_valid_model_name() {
        let validator = RequestValidator::default();
        assert!(validator.validate_model_name("yolov8n").is_ok());
        assert!(validator.validate_model_name("resnet-50").is_ok());
        assert!(validator.validate_model_name("model_v1.0").is_ok());
    }

    #[test]
    fn test_invalid_model_name() {
        let validator = RequestValidator::default();
        
        assert!(validator.validate_model_name("").is_err());
        assert!(validator.validate_model_name("model/../etc/passwd").is_err());
        assert!(validator.validate_model_name("model/path").is_err());
        assert!(validator.validate_model_name("model name").is_err());
    }

    #[test]
    fn test_input_size_validation() {
        let validator = RequestValidator::new(100, 10, 256);
        
        let small_input = vec![json!({"data": "small"})];
        assert!(validator.validate_inputs(&small_input).is_ok());
        
        let large_data = "x".repeat(200);
        let large_input = vec![json!({"data": large_data})];
        assert!(validator.validate_inputs(&large_input).is_err());
    }

    #[test]
    fn test_too_many_inputs() {
        let validator = RequestValidator::new(10000, 5, 256);
        
        let inputs: Vec<Value> = (0..6).map(|i| json!(i)).collect();
        assert!(validator.validate_inputs(&inputs).is_err());
    }

    #[test]
    fn test_malicious_pattern_detection() {
        let validator = RequestValidator::default();
        
        // Script injection
        let script_input = vec![json!("<script>alert('xss')</script>")];
        assert!(validator.validate_inputs(&script_input).is_err());
        
        // SQL injection
        let sql_input = vec![json!("' OR '1'='1")];
        assert!(validator.validate_inputs(&sql_input).is_err());
        
        // Command injection
        let cmd_input = vec![json!("$(whoami)")];
        assert!(validator.validate_inputs(&cmd_input).is_err());
    }

    #[test]
    fn test_priority_validation() {
        let validator = RequestValidator::default();
        
        assert!(validator.validate_priority(Some(0)).is_ok());
        assert!(validator.validate_priority(Some(10)).is_ok());
        assert!(validator.validate_priority(Some(-10)).is_ok());
        assert!(validator.validate_priority(Some(11)).is_err());
        assert!(validator.validate_priority(Some(-11)).is_err());
    }

    #[test]
    fn test_timeout_validation() {
        let validator = RequestValidator::default();
        
        assert!(validator.validate_timeout(Some(1000)).is_ok());
        assert!(validator.validate_timeout(Some(100)).is_ok());
        assert!(validator.validate_timeout(Some(50)).is_err());
        assert!(validator.validate_timeout(Some(400_000)).is_err());
    }

    #[test]
    fn test_comprehensive_validation() {
        let validator = RequestValidator::default();
        
        let result = validator.validate_request(
            "yolov8n",
            &[json!({"image": "base64data"})],
            Some(5),
            Some(5000),
        );
        
        assert!(result.is_ok());
    }
}
