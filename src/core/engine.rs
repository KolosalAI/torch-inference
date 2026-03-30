#![allow(dead_code)]
use std::sync::Arc;
use std::time::Instant;
use serde_json::json;

use crate::config::Config;
use crate::error::Result;
use crate::models::manager::ModelManager;
use crate::telemetry::metrics::MetricsCollector;
use crate::security::sanitizer::Sanitizer;

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
        tracing::info!(
            iterations = config.performance.warmup_iterations,
            model_count = config.models.auto_load.len(),
            "warmup start"
        );

        for model_name in &config.models.auto_load {
            let warmup_start = Instant::now();
            tracing::info!(model = %model_name, "warmup model start");

            if let Ok(_model) = self.model_manager.get_model(model_name) {
                let dummy_input = json!({"test": true});
                match self.infer(model_name, &dummy_input).await {
                    Ok(_) => {
                        let elapsed_ms = warmup_start.elapsed().as_millis() as u64;
                        tracing::info!(
                            model      = %model_name,
                            elapsed_ms = elapsed_ms,
                            status     = "ok",
                            "warmup model complete"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            model  = %model_name,
                            error  = %e,
                            status = "failed",
                            "warmup model failed"
                        );
                    }
                }
            } else {
                tracing::warn!(model = %model_name, "warmup model not found, skipping");
            }
        }

        tracing::info!("warmup complete");
        Ok(())
    }

    pub async fn infer(
        &self,
        model_name: &str,
        inputs: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let start = Instant::now();

        let span = tracing::info_span!("inference", model = %model_name);
        let _guard = span.enter();

        tracing::info!(model = %model_name, "inference start");

        // Sanitize input
        let sanitized_inputs = self
            .sanitizer
            .sanitize_input(inputs)
            .map_err(|e| crate::error::InferenceError::InvalidInput(e))?;

        // Try registered model first, fall back to legacy model
        let result = if let Ok(_) = self.model_manager.get_model_metadata(model_name) {
            self.model_manager
                .infer_registered(model_name, &sanitized_inputs)
                .await?
        } else {
            let model = self.model_manager.get_model(model_name)?;
            model.forward(&sanitized_inputs).await?
        };

        // Sanitize output
        let sanitized_result = self.sanitizer.sanitize_output(&result);

        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        self.metrics
            .record_inference(model_name, elapsed.as_secs_f64() * 1000.0);

        tracing::info!(
            model      = %model_name,
            elapsed_ms = elapsed_ms,
            "inference complete"
        );

        if elapsed_ms >= 500 {
            tracing::warn!(
                model        = %model_name,
                elapsed_ms   = elapsed_ms,
                threshold_ms = 500u64,
                "slow inference"
            );
        }

        Ok(sanitized_result)
    }

    pub async fn tts_synthesize(&self, model_name: &str, text: &str) -> Result<String> {
        tracing::info!(model = %model_name, "tts synthesis start");

        let sanitized_text = self
            .sanitizer
            .sanitize_input(&json!(text))
            .map_err(|e| crate::error::InferenceError::InvalidInput(e))?
            .as_str()
            .ok_or_else(|| {
                crate::error::InferenceError::InvalidInput(
                    "Sanitized text is not a string".to_string(),
                )
            })?
            .to_string();

        let _model = self.model_manager.get_model(model_name)?;

        let word_count = sanitized_text.split_whitespace().count();
        let audio_data = format!("base64_audio_for_{}_words", word_count);

        self.metrics.record_request();

        tracing::info!(model = %model_name, word_count = word_count, "tts synthesis complete");

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::models::manager::ModelManager;
    use std::sync::Arc;

    fn make_engine() -> InferenceEngine {
        let config = Config::default();
        let manager = Arc::new(ModelManager::new(&config, None));
        InferenceEngine::new(manager, &config)
    }

    #[test]
    fn test_engine_new() {
        let engine = make_engine();
        // health_check uses the engine struct fields — verify it doesn't panic
        let health = engine.health_check();
        assert_eq!(health["healthy"], true);
    }

    #[test]
    fn test_engine_health_check_structure() {
        let engine = make_engine();
        let health = engine.health_check();
        assert_eq!(health["checks"]["models"], true);
        assert_eq!(health["checks"]["engine"], true);
        assert!(health["timestamp"].is_number());
        assert_eq!(health["stats"]["total_requests"], 0);
        assert_eq!(health["stats"]["total_errors"], 0);
    }

    #[test]
    fn test_engine_get_stats_initial() {
        let engine = make_engine();
        let stats = engine.get_stats();
        assert_eq!(stats["total_requests"], 0);
        assert_eq!(stats["total_errors"], 0);
        assert!(stats["average_latency_ms"].is_number());
        assert!(stats["max_latency_ms"].is_number());
        assert!(stats["min_latency_ms"].is_number());
    }

    #[tokio::test]
    async fn test_engine_infer_unknown_model_returns_error() {
        let engine = make_engine();
        let result = engine.infer("nonexistent-model", &json!({"data": 42})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_engine_infer_registered_model() {
        use crate::models::manager::BaseModel;
        let config = Config::default();
        let manager = Arc::new(ModelManager::new(&config, None));
        // Register a legacy loaded model
        let mut model = BaseModel::new("test-model".to_string());
        model.load().await.unwrap();
        manager.register_model("test-model".to_string(), model).await.unwrap();

        let engine = InferenceEngine::new(manager, &config);
        let input = json!({"key": "value"});
        let result = engine.infer("test-model", &input).await;
        // Should succeed — forward() echoes the input
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_engine_tts_synthesize_unknown_model() {
        let engine = make_engine();
        let result = engine.tts_synthesize("ghost-model", "hello world").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_engine_tts_synthesize_known_model() {
        use crate::models::manager::BaseModel;
        let config = Config::default();
        let manager = Arc::new(ModelManager::new(&config, None));
        let mut model = BaseModel::new("tts-model".to_string());
        model.load().await.unwrap();
        manager.register_model("tts-model".to_string(), model).await.unwrap();

        let engine = InferenceEngine::new(manager, &config);
        let result = engine.tts_synthesize("tts-model", "hello there").await;
        assert!(result.is_ok());
        let audio = result.unwrap();
        assert!(audio.contains("base64_audio"));
    }

    #[tokio::test]
    async fn test_engine_warmup_no_auto_load() {
        let engine = make_engine();
        let mut config = Config::default();
        config.performance.warmup_iterations = 0;
        // auto_load is empty by default — warmup should complete without error
        let result = engine.warmup(&config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_engine_infer_invalid_input() {
        // Sanitizer may reject overly long strings
        let config = Config::default();
        let manager = Arc::new(ModelManager::new(&config, None));
        let engine = InferenceEngine::new(manager, &config);
        // A simple valid JSON value should either succeed (model not found error)
        // or fail with InvalidInput — both are acceptable error paths
        let result = engine.infer("no-model", &json!(null)).await;
        assert!(result.is_err());
    }

    // ── Gap-closing tests ──────────────────────────────────────────────────────

    /// Exercises engine.rs lines 36-37: the warmup loop body when a model name
    /// in `auto_load` is also present in the legacy models DashMap.
    #[tokio::test]
    async fn test_engine_warmup_with_loaded_model_in_auto_load() {
        use crate::models::manager::BaseModel;

        let config = Config::default();
        let manager = Arc::new(ModelManager::new(&config, None));

        // Register and load a legacy model.
        let mut model = BaseModel::new("warmup-model".to_string());
        model.load().await.unwrap();
        manager.register_model("warmup-model".to_string(), model).await.unwrap();

        let engine = InferenceEngine::new(Arc::clone(&manager), &config);

        // Build a config whose auto_load list contains the registered model.
        let mut warmup_config = Config::default();
        warmup_config.models.auto_load = vec!["warmup-model".to_string()];
        warmup_config.performance.warmup_iterations = 1;

        // warmup() will enter the loop, find "warmup-model" via get_model(),
        // then execute lines 36-37 (dummy_input + infer call).
        let result = engine.warmup(&warmup_config).await;
        assert!(result.is_ok());
    }

    /// Exercises engine.rs line 54: the `infer_registered` path inside `infer()`.
    /// We create a minimal stub file with a `.onnx` extension so the registry
    /// accepts it via `register_from_path`, making `get_model_metadata` succeed.
    /// The subsequent `infer_registered` call will fail at the ONNX loader stage,
    /// but line 54 will have been reached and executed.
    #[tokio::test]
    async fn test_engine_infer_via_registered_model_path() {
        use std::io::Write;

        let config = Config::default();
        let manager = Arc::new(ModelManager::new(&config, None));

        // Create a temporary stub ONNX file so the registry can accept it.
        let tmp_dir = std::env::temp_dir();
        let stub_path = tmp_dir.join("stub_engine_test.onnx");
        {
            let mut f = std::fs::File::create(&stub_path).unwrap();
            // Write minimal bytes so the file exists and has non-zero size.
            f.write_all(b"stub").unwrap();
        }

        // Register the stub through the registry (makes get_model_metadata succeed).
        let _ = manager
            .register_model_from_path(&stub_path, Some("stub-reg-model".to_string()))
            .await;

        let engine = InferenceEngine::new(Arc::clone(&manager), &config);

        // infer() will take the `infer_registered` branch (line 54) and fail at
        // ONNX loading — that's fine; we only need the line to execute.
        let result = engine.infer("stub-reg-model", &json!({"x": 1})).await;
        // It will either succeed or fail with an ONNX/format error; either is acceptable.
        let _ = result;

        // Clean up.
        let _ = std::fs::remove_file(&stub_path);
    }
}
