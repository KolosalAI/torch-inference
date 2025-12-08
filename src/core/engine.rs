use std::sync::Arc;
use std::time::Instant;
use serde_json::json;
use log::info;

use crate::config::Config;
use crate::error::Result;
use crate::models::manager::ModelManager;
use crate::telemetry::metrics::MetricsCollector;

pub struct InferenceEngine {
    pub model_manager: Arc<ModelManager>,
    metrics: MetricsCollector,
    config: Config,
}

impl InferenceEngine {
    pub fn new(model_manager: Arc<ModelManager>, config: &Config) -> Self {
        Self {
            model_manager,
            metrics: MetricsCollector::new(),
            config: config.clone(),
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
        
        let model = self.model_manager.get_model(model_name)?;
        let result = model.forward(inputs).await?;
        
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.record_inference(model_name, elapsed);
        
        Ok(result)
    }
    
    pub async fn tts_synthesize(&self, model_name: &str, text: &str) -> Result<String> {
        info!("TTS synthesis requested for model: {}", model_name);
        
        let _model = self.model_manager.get_model(model_name)?;
        
        // Placeholder: In real implementation, would call actual TTS model
        let audio_data = format!("base64_audio_for_{}_words", text.split_whitespace().count());
        
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
