pub mod validation;
pub mod sanitizer;

use serde::{Deserialize, Serialize};

pub use validation::{RequestValidator, ValidationErrorType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMitigation {
    pub enable_sanitization: bool,
    pub enable_adversarial_detection: bool,
    pub enable_threat_detection: bool,
}

impl SecurityMitigation {
    pub fn new() -> Self {
        Self {
            enable_sanitization: true,
            enable_adversarial_detection: true,
            enable_threat_detection: true,
        }
    }
    
    pub fn validate_input(&self, _input: &serde_json::Value) -> Result<(), String> {
        // Implement input validation
        Ok(())
    }
    
    pub fn sanitize_output(&self, output: serde_json::Value) -> serde_json::Value {
        // Implement output sanitization
        output
    }
}

impl Default for SecurityMitigation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_security_mitigation_new_defaults() {
        let sm = SecurityMitigation::new();
        assert!(sm.enable_sanitization);
        assert!(sm.enable_adversarial_detection);
        assert!(sm.enable_threat_detection);
    }

    #[test]
    fn test_security_mitigation_default_equals_new() {
        let a = SecurityMitigation::new();
        let b = SecurityMitigation::default();
        assert_eq!(a.enable_sanitization, b.enable_sanitization);
        assert_eq!(a.enable_adversarial_detection, b.enable_adversarial_detection);
        assert_eq!(a.enable_threat_detection, b.enable_threat_detection);
    }

    #[test]
    fn test_validate_input_returns_ok() {
        let sm = SecurityMitigation::new();
        let input = json!({"key": "value"});
        assert!(sm.validate_input(&input).is_ok());
    }

    #[test]
    fn test_sanitize_output_passthrough() {
        let sm = SecurityMitigation::new();
        let output = json!({"result": 42});
        let sanitized = sm.sanitize_output(output.clone());
        assert_eq!(sanitized, output);
    }

    #[test]
    fn test_security_mitigation_serialize_deserialize() {
        let sm = SecurityMitigation::new();
        let json_str = serde_json::to_string(&sm).unwrap();
        let restored: SecurityMitigation = serde_json::from_str(&json_str).unwrap();
        assert_eq!(sm.enable_sanitization, restored.enable_sanitization);
        assert_eq!(sm.enable_adversarial_detection, restored.enable_adversarial_detection);
        assert_eq!(sm.enable_threat_detection, restored.enable_threat_detection);
    }

    #[test]
    fn test_security_mitigation_clone_and_debug() {
        let sm = SecurityMitigation::new();
        let cloned = sm.clone();
        // Debug formatting should not panic
        let _ = format!("{:?}", cloned);
    }
}
