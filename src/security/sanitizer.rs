use serde_json::{Value, Map};
use log::{info, warn};
use std::sync::Arc;
use regex::Regex;
use lazy_static::lazy_static;

use crate::config::SanitizerConfig;

lazy_static! {
    // Regex to match control characters (except common whitespace)
    static ref CONTROL_CHARS: Regex = Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]").unwrap();
}

pub struct Sanitizer {
    config: SanitizerConfig,
}

impl Sanitizer {
    pub fn new(config: SanitizerConfig) -> Self {
        Self { config }
    }

    /// Sanitize input data before inference
    pub fn sanitize_input(&self, input: &Value) -> Result<Value, String> {
        match input {
            Value::Object(map) => {
                let mut new_map = Map::new();
                for (k, v) in map {
                    // Sanitize keys (prevent injection in keys)
                    let safe_key = self.sanitize_text(k)?;
                    
                    // Sanitize values based on key or type
                    let safe_value = if k == "image" || k == "image_data" {
                        self.sanitize_image_input(v)?
                    } else if k == "text" || k == "prompt" {
                        self.sanitize_text_input(v)?
                    } else {
                        self.sanitize_input(v)?
                    };
                    
                    new_map.insert(safe_key, safe_value);
                }
                Ok(Value::Object(new_map))
            },
            Value::Array(arr) => {
                let mut new_arr = Vec::new();
                for v in arr {
                    new_arr.push(self.sanitize_input(v)?);
                }
                Ok(Value::Array(new_arr))
            },
            Value::String(s) => {
                if self.config.sanitize_text {
                    Ok(Value::String(self.sanitize_text(s)?))
                } else {
                    Ok(Value::String(s.clone()))
                }
            },
            Value::Number(n) => {
                // Check for NaN/Inf if possible (serde_json usually handles this, but good to be safe)
                if let Some(f) = n.as_f64() {
                    if f.is_nan() || f.is_infinite() {
                        return Err("Input contains NaN or Infinite values".to_string());
                    }
                }
                Ok(Value::Number(n.clone()))
            },
            _ => Ok(input.clone()),
        }
    }

    /// Sanitize text string
    fn sanitize_text(&self, text: &str) -> Result<String, String> {
        if text.len() > self.config.max_text_length {
            return Err(format!("Text length {} exceeds maximum {}", text.len(), self.config.max_text_length));
        }

        // Remove control characters
        let cleaned = CONTROL_CHARS.replace_all(text, "").to_string();
        
        // Normalize Unicode (NFKC) - simplified here as just checking for homoglyphs is complex
        // In a full implementation, we would use the `unicode-normalization` crate
        
        Ok(cleaned)
    }

    fn sanitize_text_input(&self, value: &Value) -> Result<Value, String> {
        match value {
            Value::String(s) => Ok(Value::String(self.sanitize_text(s)?)),
            Value::Array(arr) => {
                let mut new_arr = Vec::new();
                for v in arr {
                    new_arr.push(self.sanitize_text_input(v)?);
                }
                Ok(Value::Array(new_arr))
            },
            _ => Ok(value.clone()),
        }
    }

    fn sanitize_image_input(&self, value: &Value) -> Result<Value, String> {
        // If it's a base64 string, we could validate it's valid base64
        if let Value::String(s) = value {
            if s.len() > 10_000_000 { // 10MB limit for base64 string
                return Err("Image data too large".to_string());
            }
            // Basic base64 char check
            if s.chars().any(|c| !c.is_ascii_alphanumeric() && c != '+' && c != '/' && c != '=') {
                return Err("Invalid characters in image data".to_string());
            }
        }
        // If it's an object (e.g. {"width": 100, "data": "..."}), check dimensions
        if let Value::Object(map) = value {
            if self.config.sanitize_image_dimensions {
                if let (Some(w), Some(h)) = (map.get("width").and_then(|v| v.as_u64()), map.get("height").and_then(|v| v.as_u64())) {
                    if w > self.config.max_image_width as u64 || h > self.config.max_image_height as u64 {
                        return Err(format!("Image dimensions {}x{} exceed limit {}x{}", w, h, self.config.max_image_width, self.config.max_image_height));
                    }
                }
            }
        }
        Ok(value.clone())
    }

    /// Sanitize output data after inference
    pub fn sanitize_output(&self, output: &Value) -> Value {
        match output {
            Value::Object(map) => {
                let mut new_map = Map::new();
                for (k, v) in map {
                    // Filter out potentially sensitive keys if needed
                    if k.starts_with("_") || k == "internal_state" {
                        continue;
                    }
                    
                    let v = if self.config.remove_null_values && v.is_null() {
                        continue;
                    } else {
                        self.sanitize_output(v)
                    };
                    
                    new_map.insert(k.clone(), v);
                }
                Value::Object(new_map)
            },
            Value::Array(arr) => {
                let mut new_arr = Vec::new();
                for v in arr {
                    new_arr.push(self.sanitize_output(v));
                }
                Value::Array(new_arr)
            },
            Value::Number(n) => {
                if self.config.round_probabilities {
                    if let Some(f) = n.as_f64() {
                        // Round float to N decimal places
                        let factor = 10f64.powi(self.config.probability_decimals as i32);
                        let rounded = (f * factor).round() / factor;
                        if let Some(n_rounded) = serde_json::Number::from_f64(rounded) {
                            return Value::Number(n_rounded);
                        }
                    }
                }
                Value::Number(n.clone())
            },
            _ => output.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_sanitize_text() {
        let config = SanitizerConfig::default();
        let sanitizer = Sanitizer::new(config);
        
        let input = json!({"text": "Hello\x00World"});
        let sanitized = sanitizer.sanitize_input(&input).unwrap();
        
        assert_eq!(sanitized["text"], "HelloWorld");
    }

    #[test]
    fn test_sanitize_output_rounding() {
        let config = SanitizerConfig {
            probability_decimals: 2,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        
        let output = json!({"score": 0.123456});
        let sanitized = sanitizer.sanitize_output(&output);
        
        assert_eq!(sanitized["score"], 0.12);
    }

    #[test]
    fn test_sanitize_output_nulls() {
        let config = SanitizerConfig {
            remove_null_values: true,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        
        let output = json!({"valid": 1, "invalid": null});
        let sanitized = sanitizer.sanitize_output(&output);
        
        assert!(sanitized.get("valid").is_some());
        assert!(sanitized.get("invalid").is_none());
    }
}
