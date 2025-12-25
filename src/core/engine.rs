use std::sync::Arc;
use std::time::Instant;
use serde_json::json;
use log::info;

use crate::config::Config;
use crate::error::Result;
use crate::models::manager::ModelManager;
use crate::telemetry::metrics::MetricsCollector;
use crate::security::sanitizer::Sanitizer;

#[allow(dead_code)]
pub struct InferenceEngine {
    pub model_manager: Arc<ModelManager>,
    metrics: MetricsCollector,
    config: Config,
    sanitizer: Sanitizer,
}

impl InferenceEngine {
    pub fn new(model_manager: Arc<ModelManager>, config: &Config) -> Self {
        Self {
            model_manager,
            metrics: MetricsCollector::new(),
            config: config.clone(),
            sanitizer: Sanitizer::new(config.sanitizer.clone()),
        }
    }
    
    pub async fn warmup(&self, config: &Config) -> Result<()> {
        info!("Starting model warmup with {} iterations", config.performance.warmup_iterations);
        
        for model_name in &config.models.auto_load {
            info!("Warming up model: {}", model_name);
            if let Ok(_model) = self.model_manager.get_model(model_name) {
                // Perform dummy inference
                let dummy_input = json!({"test": true});
                let _ = self.infer(model_name, &dummy_input).await;
            }
        }
        
        info!("Warmup completed");
        Ok(())
    }
    
    pub async fn infer(&self, model_name: &str, inputs: &serde_json::Value) -> Result<serde_json::Value> {
        let start = Instant::now();
        
        // Sanitize input
        let sanitized_inputs = self.sanitizer.sanitize_input(inputs)
            .map_err(|e| crate::error::InferenceError::InvalidInput(e))?;
        
        // Try registered model first
        let result = if let Ok(_) = self.model_manager.get_model_metadata(model_name) {
            self.model_manager.infer_registered(model_name, &sanitized_inputs).await?
        } else {
            // Fallback to legacy model
            let model = self.model_manager.get_model(model_name)?;
            model.forward(&sanitized_inputs).await?
        };
        
        // Sanitize output
        let sanitized_result = self.sanitizer.sanitize_output(&result);
        
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.record_inference(model_name, elapsed);
        
        Ok(sanitized_result)
    }
    
    pub async fn tts_synthesize(&self, model_name: &str, text: &str) -> Result<String> {
        info!("TTS synthesis requested for model: {}", model_name);
        
        // Sanitize text input
        let sanitized_text = self.sanitizer.sanitize_input(&json!(text))
            .map_err(|e| crate::error::InferenceError::InvalidInput(e))?
            .as_str()
            .ok_or_else(|| crate::error::InferenceError::InvalidInput("Sanitized text is not a string".to_string()))?
            .to_string();
        
        let _model = self.model_manager.get_model(model_name)?;
        
        // Placeholder: In real implementation, would call actual TTS model
        let audio_data = format!("base64_audio_for_{}_words", sanitized_text.split_whitespace().count());
        
        self.metrics.record_request();
        
        Ok(audio_data)
    }
    
    pub fn health_check(&self) -> serde_json::Value {
        let metrics = self.metrics.get_request_metrics();
        
        json!({
            "healthy": true,
            "checks": {
                "models": true,
                "engine": true
            },
            "timestamp": chrono::Utc::now().timestamp_millis(),
            "stats": {
                "total_requests": metrics.total_requests,
                "total_errors": metrics.total_errors,
                "avg_latency_ms": metrics.avg_latency_ms
            }
        })
    }
    
    pub fn get_stats(&self) -> serde_json::Value {
        let metrics = self.metrics.get_request_metrics();
        
        json!({
            "total_requests": metrics.total_requests,
            "total_errors": metrics.total_errors,
            "average_latency_ms": metrics.avg_latency_ms,
            "max_latency_ms": metrics.max_latency_ms,
            "min_latency_ms": metrics.min_latency_ms
        })
    }
}
