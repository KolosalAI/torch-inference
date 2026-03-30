use serde_json::{Value, Map};
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

    fn default_sanitizer() -> Sanitizer {
        Sanitizer::new(SanitizerConfig::default())
    }

    // ── existing tests ────────────────────────────────────────────────────────

    #[test]
    fn test_sanitize_text() {
        let sanitizer = default_sanitizer();
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

    // ── sanitize_input – additional paths ────────────────────────────────────

    #[test]
    fn test_sanitize_input_array() {
        let sanitizer = default_sanitizer();
        let input = json!(["hello\x01world", "clean"]);
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result[0], "helloworld");
        assert_eq!(result[1], "clean");
    }

    #[test]
    fn test_sanitize_input_number_valid() {
        let sanitizer = default_sanitizer();
        let input = json!(42.5);
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result, json!(42.5));
    }

    #[test]
    fn test_sanitize_input_bool_and_null() {
        let sanitizer = default_sanitizer();
        assert_eq!(sanitizer.sanitize_input(&json!(true)).unwrap(), json!(true));
        assert_eq!(sanitizer.sanitize_input(&json!(null)).unwrap(), json!(null));
    }

    #[test]
    fn test_sanitize_input_string_sanitize_text_disabled() {
        let config = SanitizerConfig {
            sanitize_text: false,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        // Control char should pass through unchanged when sanitize_text is false
        let input = json!("hello\x01world");
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result, json!("hello\x01world"));
    }

    #[test]
    fn test_sanitize_input_string_sanitize_text_enabled() {
        let sanitizer = default_sanitizer();
        let input = json!("hello\x01world");
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result, json!("helloworld"));
    }

    #[test]
    fn test_sanitize_input_object_with_prompt_key() {
        let sanitizer = default_sanitizer();
        let input = json!({"prompt": "clean prompt"});
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result["prompt"], "clean prompt");
    }

    #[test]
    fn test_sanitize_input_object_with_image_key() {
        let sanitizer = default_sanitizer();
        // Valid base64-ish data
        let input = json!({"image": "aGVsbG8="});
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result["image"], "aGVsbG8=");
    }

    // ── sanitize_text – error path ────────────────────────────────────────────

    #[test]
    fn test_sanitize_text_too_long() {
        let config = SanitizerConfig {
            max_text_length: 5,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        let input = json!({"text": "toolongstring"});
        let result = sanitizer.sanitize_input(&input);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("exceeds maximum"));
    }

    // ── sanitize_text_input – array and non-string paths ─────────────────────

    #[test]
    fn test_sanitize_text_input_array_of_strings() {
        let sanitizer = default_sanitizer();
        let input = json!({"text": ["hello\x01", "world\x02"]});
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result["text"][0], "hello");
        assert_eq!(result["text"][1], "world");
    }

    #[test]
    fn test_sanitize_text_input_non_string_passthrough() {
        let sanitizer = default_sanitizer();
        // A number as the "text" value should pass through unchanged
        let input = json!({"text": 42});
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result["text"], 42);
    }

    // ── sanitize_image_input ──────────────────────────────────────────────────

    #[test]
    fn test_sanitize_image_input_invalid_chars() {
        let sanitizer = default_sanitizer();
        let input = json!({"image": "not valid base64 !"});
        let result = sanitizer.sanitize_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid characters"));
    }

    #[test]
    fn test_sanitize_image_input_dimensions_exceeded() {
        let config = SanitizerConfig {
            sanitize_image_dimensions: true,
            max_image_width: 100,
            max_image_height: 100,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        let input = json!({"image": {"width": 200, "height": 50}});
        let result = sanitizer.sanitize_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceed limit"));
    }

    #[test]
    fn test_sanitize_image_input_dimensions_within_limit() {
        let config = SanitizerConfig {
            sanitize_image_dimensions: true,
            max_image_width: 1024,
            max_image_height: 1024,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        let input = json!({"image": {"width": 100, "height": 100}});
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result["image"]["width"], 100);
    }

    #[test]
    fn test_sanitize_image_input_dimensions_disabled() {
        let config = SanitizerConfig {
            sanitize_image_dimensions: false,
            max_image_width: 10,
            max_image_height: 10,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        // Oversized dimensions but checking is disabled – should succeed
        let input = json!({"image": {"width": 9999, "height": 9999}});
        let result = sanitizer.sanitize_input(&input).unwrap();
        assert_eq!(result["image"]["width"], 9999);
    }

    // ── sanitize_output – additional paths ───────────────────────────────────

    #[test]
    fn test_sanitize_output_filters_internal_keys() {
        let sanitizer = default_sanitizer();
        let output = json!({
            "result": "ok",
            "_secret": "hidden",
            "internal_state": "also_hidden"
        });
        let sanitized = sanitizer.sanitize_output(&output);
        assert!(sanitized.get("result").is_some());
        assert!(sanitized.get("_secret").is_none());
        assert!(sanitized.get("internal_state").is_none());
    }

    #[test]
    fn test_sanitize_output_array() {
        // Use round_probabilities: false so integer values are not cast to f64
        let config = SanitizerConfig {
            round_probabilities: false,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        let output = json!([1, 2, 3]);
        let sanitized = sanitizer.sanitize_output(&output);
        assert_eq!(sanitized, json!([1, 2, 3]));
    }

    #[test]
    fn test_sanitize_output_bool_and_null_passthrough() {
        let sanitizer = default_sanitizer();
        assert_eq!(sanitizer.sanitize_output(&json!(true)), json!(true));
        assert_eq!(sanitizer.sanitize_output(&json!(null)), json!(null));
    }

    #[test]
    fn test_sanitize_output_string_passthrough() {
        let sanitizer = default_sanitizer();
        let output = json!("some string");
        assert_eq!(sanitizer.sanitize_output(&output), json!("some string"));
    }

    #[test]
    fn test_sanitize_output_rounding_disabled() {
        let config = SanitizerConfig {
            round_probabilities: false,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        let output = json!({"score": 0.123456});
        let sanitized = sanitizer.sanitize_output(&output);
        // Value should not be rounded
        assert!((sanitized["score"].as_f64().unwrap() - 0.123456).abs() < 1e-9);
    }

    #[test]
    fn test_sanitize_output_null_not_removed_when_disabled() {
        let config = SanitizerConfig {
            remove_null_values: false,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        let output = json!({"present": null});
        let sanitized = sanitizer.sanitize_output(&output);
        // null value should remain when remove_null_values is false
        assert!(sanitized.get("present").is_some());
    }

    #[test]
    fn test_sanitize_output_nested_object_in_array() {
        let config = SanitizerConfig {
            remove_null_values: true,
            ..Default::default()
        };
        let sanitizer = Sanitizer::new(config);
        let output = json!([{"keep": 1, "drop": null}]);
        let sanitized = sanitizer.sanitize_output(&output);
        assert!(sanitized[0].get("keep").is_some());
        assert!(sanitized[0].get("drop").is_none());
    }

    #[test]
    fn test_sanitize_image_input_too_large_string() {
        // Exercises line 105: s.len() > 10_000_000 → Err("Image data too large")
        let sanitizer = default_sanitizer();
        // Create a base64-clean string (all 'A') longer than 10 MB
        let large_data = "A".repeat(10_000_001);
        let input = json!({"image": large_data});
        let result = sanitizer.sanitize_input(&input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Image data too large");
    }
}
