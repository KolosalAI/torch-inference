# Logging Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all `println!`/`log::` calls with `tracing::`, add startup phase spans with live timing, enrich the request logger with full request/response state, and add structured inference lifecycle events.

**Architecture:** Four isolated file-level tasks applied in order: engine → manager → request_logger → main. Each task is self-contained and the test suite passes after each commit. No new dependencies are added; the existing `tracing` setup in `structured_logging.rs` already captures span close events via `FmtSpan::CLOSE`.

**Tech Stack:** Rust, `tracing` 0.1, `actix-web` 4.8, `tokio` async runtime.

---

## File Map

| File | Change |
|---|---|
| `src/core/engine.rs` | Remove `use log::info`, add `tracing::info_span!` around inference, structured start/complete/slow events in `infer()` and `warmup()` |
| `src/models/manager.rs` | Remove `use log::{info, warn}`, add `use std::time::Instant`, replace all `log::` calls, add load timing to `load_pytorch_model`, `load_onnx_model`, `BaseModel::load` |
| `src/middleware/request_logger.rs` | Add `status_class()` helper, add `query`/`user_agent`/`content_len` to `request_received`, add `status_class`/`response_bytes` to `request_completed`, add `slow_request` warn, add tests |
| `src/main.rs` | Remove `use log::info`, remove `middleware as actix_middleware` import, remove `.wrap(actix_middleware::Logger::new(...))`, replace all `println!` and `log::` calls with `tracing::`, add phase spans, rewrite `log_system_info()` |

---

## Task 1: Structured inference logging in `src/core/engine.rs`

**Files:**
- Modify: `src/core/engine.rs`

- [ ] **Step 1: Verify existing tests pass before making any changes**

```bash
cargo test 2>&1 | tail -5
```

Expected: test result: ok (or failures unrelated to engine.rs)

- [ ] **Step 2: Replace the full `engine.rs` content**

Replace the entire file with the following. Key changes: remove `use log::info`, add inference span with start/complete/slow events, add per-model warmup events.

```rust
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
```

- [ ] **Step 3: Run the test suite to confirm no regressions**

```bash
cargo test 2>&1 | tail -10
```

Expected: test result: ok (same pass count as before)

- [ ] **Step 4: Commit**

```bash
git add src/core/engine.rs
git commit -m "feat(logging): add structured inference spans and live events in engine.rs"
```

---

## Task 2: Structured model lifecycle logging in `src/models/manager.rs`

**Files:**
- Modify: `src/models/manager.rs:1-10` (imports), `src/models/manager.rs:31-45` (BaseModel::load), `src/models/manager.rs:92-197` (load_pytorch_model, load_onnx_model), `src/models/manager.rs:370-433` (initialize_default_models)`

- [ ] **Step 1: Remove `use log::{info, warn}` and add `use std::time::Instant`**

Find line 4:
```rust
use log::{info, warn};
```
Replace with:
```rust
use std::time::Instant;
```

- [ ] **Step 2: Update `BaseModel::load()` (currently line 32-36)**

Replace:
```rust
    pub async fn load(&mut self) -> Result<()> {
        info!("Loading model: {}", self.name);
        self.is_loaded = true;
        Ok(())
    }
```
With:
```rust
    pub async fn load(&mut self) -> Result<()> {
        tracing::info!(model = %self.name, "model load start");
        self.is_loaded = true;
        tracing::info!(model = %self.name, "model load complete");
        Ok(())
    }
```

- [ ] **Step 3: Update `register_model_from_path()` (currently lines 92-98)**

Replace:
```rust
    pub async fn register_model_from_path(&self, path: &Path, name: Option<String>) -> Result<String> {
        info!("Registering model from path: {:?}", path);
        let model_id = self.registry.register_from_path(path, name).await
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        info!("Model registered with ID: {}", model_id);
        Ok(model_id)
    }
```
With:
```rust
    pub async fn register_model_from_path(&self, path: &Path, name: Option<String>) -> Result<String> {
        tracing::info!(path = ?path, "model registration start");
        let model_id = self.registry.register_from_path(path, name).await
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
        tracing::info!(model_id = %model_id, "model registered");
        Ok(model_id)
    }
```

- [ ] **Step 4: Update `scan_and_register()` (currently lines 101-105)**

Replace:
```rust
    pub async fn scan_and_register(&self, dir_path: &Path) -> Result<Vec<String>> {
        info!("Scanning directory for models: {:?}", dir_path);
        self.registry.scan_directory(dir_path).await
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))
    }
```
With:
```rust
    pub async fn scan_and_register(&self, dir_path: &Path) -> Result<Vec<String>> {
        tracing::info!(dir = ?dir_path, "scanning directory for models");
        self.registry.scan_directory(dir_path).await
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))
    }
```

- [ ] **Step 5: Update `load_pytorch_model()` (the `#[cfg(feature = "torch")]` version, currently lines 108-155)**

Replace from `pub async fn load_pytorch_model` through its closing `}`:
```rust
    #[cfg(feature = "torch")]
    pub async fn load_pytorch_model(&self, model_id: &str) -> Result<()> {
        let start = Instant::now();
        tracing::info!(model = %model_id, format = "pytorch", "model load start");

        let metadata = self.registry.get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        if metadata.format != ModelFormat::PyTorch {
            return Err(InferenceError::ModelLoadError(
                format!("Model {} is not a PyTorch model", model_id)
            ));
        }

        let devices = if let Some(ids) = &self.config.device.device_ids {
            ids.clone()
        } else {
            vec![self.config.device.device_id]
        };

        let mut loaded_models = Vec::new();

        for device_id in &devices {
            let device_str = if self.config.device.device_type == "cuda"
                || self.config.device.device_type == "auto"
            {
                format!("cuda:{}", device_id)
            } else {
                self.config.device.device_type.clone()
            };

            tracing::info!(model = %model_id, device = %device_str, "loading pytorch model on device");
            let loaded_model = self.pytorch_loader.load_model(&metadata.path, Some(device_str))
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            loaded_models.push(loaded_model);
        }

        self.loaded_pytorch_models.insert(model_id.to_string(), loaded_models);
        self.registry.mark_loaded(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            model       = %model_id,
            format      = "pytorch",
            elapsed_ms  = elapsed_ms,
            device_count = devices.len(),
            "model load complete"
        );
        Ok(())
    }
```

- [ ] **Step 6: Update `load_onnx_model()` (currently lines 157-196)**

Replace from `pub async fn load_onnx_model` through its closing `}`:
```rust
    pub async fn load_onnx_model(&self, model_id: &str) -> Result<()> {
        let start = Instant::now();
        tracing::info!(model = %model_id, format = "onnx", "model load start");

        let metadata = self.registry.get_model(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        if metadata.format != ModelFormat::ONNX {
            return Err(InferenceError::ModelLoadError(
                format!("Model {} is not an ONNX model", model_id)
            ));
        }

        let devices = if let Some(ids) = &self.config.device.device_ids {
            ids.clone()
        } else {
            vec![self.config.device.device_id]
        };

        let mut loaded_models = Vec::new();

        for device_id in &devices {
            tracing::info!(model = %model_id, device_id = device_id, "loading onnx model on device");
            let loaded_model = self.onnx_loader.load_model(&metadata.path, Some(*device_id))
                .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;
            loaded_models.push(loaded_model);
        }

        self.loaded_onnx_models.insert(model_id.to_string(), loaded_models);
        self.registry.mark_loaded(model_id)
            .map_err(|e| InferenceError::ModelLoadError(e.to_string()))?;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            model        = %model_id,
            format       = "onnx",
            elapsed_ms   = elapsed_ms,
            device_count = devices.len(),
            "model load complete"
        );
        Ok(())
    }
```

- [ ] **Step 7: Update inference log calls (lines ~210, 238, 244, 272)**

Replace:
```rust
        info!("Running inference on model: {}", model_id);
```
With:
```rust
        tracing::info!(model = %model_id, format = "pytorch", "inference start");
```

Replace (line ~238):
```rust
        info!("Inference completed for model: {}", model_id);
```
With:
```rust
        tracing::info!(model = %model_id, format = "pytorch", "inference complete");
```

Replace (line ~244):
```rust
        info!("Running inference on ONNX model: {}", model_id);
```
With:
```rust
        tracing::info!(model = %model_id, format = "onnx", "inference start");
```

Replace (line ~272):
```rust
        info!("Inference completed for model: {}", model_id);
```
With:
```rust
        tracing::info!(model = %model_id, format = "onnx", "inference complete");
```

- [ ] **Step 8: Update `register_model()` legacy method (line ~339)**

Replace:
```rust
        info!("Registering legacy model: {}", name);
```
With:
```rust
        tracing::info!(model = %name, "registering legacy model");
```

- [ ] **Step 9: Update `initialize_default_models()` (lines ~371-432)**

Replace the entire method body's log calls:
```rust
    pub async fn initialize_default_models(&self) -> Result<()> {
        tracing::info!("initializing default models");

        let example_model = BaseModel::new("example".to_string());
        self.register_model("example".to_string(), example_model).await?;

        let model_dirs = vec![
            PathBuf::from(&self.config.models.cache_dir),
            PathBuf::from("./models"),
            PathBuf::from("./models/audio"),
        ];

        for dir in model_dirs {
            if dir.exists() && dir.is_dir() {
                match self.scan_and_register(&dir).await {
                    Ok(models) => {
                        tracing::info!(dir = ?dir, model_count = models.len(), "models registered from directory");
                    }
                    Err(e) => {
                        tracing::warn!(dir = ?dir, error = %e, "failed to scan model directory");
                    }
                }
            }
        }

        for model_name in &self.config.models.auto_load {
            if let Ok(model) = self.get_model(model_name) {
                let mut m = model;
                if let Err(e) = m.load().await {
                    tracing::warn!(model = %model_name, error = %e, "failed to load legacy model");
                }
            } else {
                #[cfg(feature = "torch")]
                {
                    if let Ok(metadata) = self.registry.get_model(model_name) {
                        if metadata.format == ModelFormat::PyTorch {
                            if let Err(e) = self.load_pytorch_model(model_name).await {
                                tracing::warn!(model = %model_name, error = %e, format = "pytorch", "auto-load failed");
                            }
                        }
                    }
                }

                {
                    if let Ok(metadata) = self.registry.get_model(model_name) {
                        if metadata.format == ModelFormat::ONNX {
                            if let Err(e) = self.load_onnx_model(model_name).await {
                                tracing::warn!(model = %model_name, error = %e, format = "onnx", "auto-load failed");
                            }
                        }
                    }
                }
            }
        }

        tracing::info!("model initialization complete");
        Ok(())
    }
```

> **Note:** The spec mentions `ModelManager::unload_model()` — this method does not exist in the current codebase. If an unload path is added in future, add `tracing::info!(model = %name, reason = %reason, "model unload")` at that point. No action needed here.

- [ ] **Step 10: Run the test suite**

```bash
cargo test 2>&1 | tail -10
```

Expected: same pass count as before

- [ ] **Step 11: Commit**

```bash
git add src/models/manager.rs
git commit -m "feat(logging): replace log:: with tracing:: and add load timing in manager.rs"
```

---

## Task 3: Enrich request logger with live request state

**Files:**
- Modify: `src/middleware/request_logger.rs`

- [ ] **Step 1: Write failing test for `status_class` helper**

Add this test at the bottom of the `#[cfg(test)]` block (before the closing `}`):

```rust
    #[test]
    fn test_status_class_2xx() {
        assert_eq!(status_class(200), "2xx");
        assert_eq!(status_class(201), "2xx");
        assert_eq!(status_class(299), "2xx");
    }

    #[test]
    fn test_status_class_3xx() {
        assert_eq!(status_class(301), "3xx");
        assert_eq!(status_class(304), "3xx");
    }

    #[test]
    fn test_status_class_4xx() {
        assert_eq!(status_class(400), "4xx");
        assert_eq!(status_class(404), "4xx");
        assert_eq!(status_class(422), "4xx");
    }

    #[test]
    fn test_status_class_5xx() {
        assert_eq!(status_class(500), "5xx");
        assert_eq!(status_class(503), "5xx");
    }

    #[test]
    fn test_status_class_other() {
        assert_eq!(status_class(100), "other");
        assert_eq!(status_class(102), "other");
    }
```

- [ ] **Step 2: Run to confirm tests fail (status_class not yet defined)**

```bash
cargo test test_status_class 2>&1 | tail -10
```

Expected: `error[E0425]: cannot find function 'status_class'`

- [ ] **Step 3: Add `status_class` helper above the `impl` block**

Add this function after the `pub struct RequestLoggerMiddleware<S>` block (before its `impl`):

```rust
fn status_class(status: u16) -> &'static str {
    match status {
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _         => "other",
    }
}
```

- [ ] **Step 4: Run status_class tests to confirm they pass**

```bash
cargo test test_status_class 2>&1 | tail -10
```

Expected: 5 tests passed

- [ ] **Step 5: Write new integration tests for the enriched fields**

Add these tests at the bottom of the `#[cfg(test)]` block:

```rust
    #[actix_web::test]
    async fn test_request_logger_with_query_string() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route("/search", web::get().to(|| async { HttpResponse::Ok().finish() })),
        ).await;

        let req = awtest::TestRequest::get()
            .uri("/search?q=hello&top_k=5")
            .to_request();
        let resp = awtest::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_request_logger_with_user_agent() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route("/", web::get().to(|| async { HttpResponse::Ok().finish() })),
        ).await;

        let req = awtest::TestRequest::get()
            .uri("/")
            .insert_header(("User-Agent", "test-client/1.0"))
            .to_request();
        let resp = awtest::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_request_logger_without_user_agent() {
        // Exercises the "unknown" fallback path
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route("/", web::get().to(|| async { HttpResponse::Ok().finish() })),
        ).await;

        let req = awtest::TestRequest::get().uri("/").to_request();
        let resp = awtest::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_request_logger_with_content_length() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route("/data", web::post().to(|| async { HttpResponse::Ok().finish() })),
        ).await;

        let req = awtest::TestRequest::post()
            .uri("/data")
            .insert_header(("Content-Length", "128"))
            .to_request();
        let resp = awtest::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_request_logger_response_with_content_length_header() {
        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route(
                    "/sized",
                    web::get().to(|| async {
                        HttpResponse::Ok()
                            .insert_header(("Content-Length", "11"))
                            .body("hello world")
                    }),
                ),
        ).await;

        let req = awtest::TestRequest::get().uri("/sized").to_request();
        let resp = awtest::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }
```

- [ ] **Step 6: Run new tests — they should pass already (fields not checked, only that the middleware doesn't panic)**

```bash
cargo test test_request_logger_with_query test_request_logger_with_user test_request_logger_without_user test_request_logger_with_content test_request_logger_response_with_content 2>&1 | tail -10
```

Expected: FAIL — the new fields are not extracted yet, so compiling the test triggers the need for the fields

- [ ] **Step 7: Replace the `call()` method with the enriched version**

Replace the entire `fn call(&self, req: ServiceRequest) -> Self::Future` block (lines 46-104) with:

```rust
    fn call(&self, req: ServiceRequest) -> Self::Future {
        let method = req.method().to_string();
        let path = req.path().to_string();
        let remote_addr = req
            .connection_info()
            .peer_addr()
            .unwrap_or("unknown")
            .to_string();
        let query = req.query_string().to_string();
        let user_agent = req
            .headers()
            .get("User-Agent")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_string();
        let content_len: u64 = req
            .headers()
            .get("Content-Length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let correlation_id = req
            .headers()
            .get("X-Correlation-ID")
            .and_then(|v| v.to_str().ok())
            .map(|s| CorrelationId::from_header(s))
            .unwrap_or_else(CorrelationId::new);

        let metrics = RequestMetrics::new(correlation_id.clone());

        tracing::info!(
            correlation_id = %correlation_id.as_str(),
            method         = %method,
            path           = %path,
            remote_addr    = %remote_addr,
            query          = %query,
            user_agent     = %user_agent,
            content_len    = content_len,
            event          = "request_received",
        );

        let fut = self.service.call(req);

        Box::pin(async move {
            match fut.await {
                Ok(res) => {
                    let status = res.status().as_u16();
                    let response_bytes: u64 = res
                        .headers()
                        .get("content-length")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let duration_ms = metrics.duration_ms();

                    tracing::info!(
                        correlation_id = %metrics.correlation_id.as_str(),
                        method         = %method,
                        path           = %path,
                        status         = status,
                        status_class   = %status_class(status),
                        duration_ms    = duration_ms,
                        response_bytes = response_bytes,
                        event          = "request_completed",
                    );

                    if duration_ms >= 500 {
                        tracing::warn!(
                            correlation_id = %metrics.correlation_id.as_str(),
                            method         = %method,
                            path           = %path,
                            duration_ms    = duration_ms,
                            threshold_ms   = 500u64,
                            event          = "slow_request",
                        );
                    }

                    Ok(res)
                }
                Err(err) => {
                    tracing::error!(
                        correlation_id = %metrics.correlation_id.as_str(),
                        method         = %method,
                        path           = %path,
                        error          = %err,
                        duration_ms    = %metrics.duration_ms(),
                        status_class   = "5xx",
                        event          = "request_error",
                    );

                    Err(err)
                }
            }
        })
    }
```

- [ ] **Step 8: Run all request_logger tests**

```bash
cargo test middleware::request_logger 2>&1 | tail -10
```

Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add src/middleware/request_logger.rs
git commit -m "feat(logging): enrich request logger with query, user-agent, response size, slow-request warn"
```

---

## Task 4: Unify logging in `src/main.rs` — remove println!, add phase spans

**Files:**
- Modify: `src/main.rs`

This task has many changes. Apply them in order. Run `cargo build` after each section to catch compile errors early.

- [ ] **Step 1: Fix imports — remove `use log::info` and remove `middleware as actix_middleware`**

Replace line:
```rust
use actix_web::{web, App, HttpServer, middleware as actix_middleware};
use log::info;
```
With:
```rust
use actix_web::{web, App, HttpServer};
```

- [ ] **Step 2: Verify it compiles (will fail on undefined `info!` and `actix_middleware` — expected)**

```bash
cargo build 2>&1 | grep "^error" | head -15
```

Note every line number for `info!`, `log::warn!`, `actix_middleware` errors — these are what we'll fix next.

- [ ] **Step 3: Replace the startup banner (lines ~155-157)**

Replace:
```rust
    println!("\n{}", "═".repeat(80));
    println!("  [START] PyTorch Inference Framework v1.0.0");
    println!("{}\n", "═".repeat(80));
```
With:
```rust
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "torch-inference starting");
```

- [ ] **Step 4: Replace config load block (lines ~159-160) with a phase span**

Replace:
```rust
    let mut config = Config::load().expect("Failed to load configuration");
    info!("[OK] Configuration loaded successfully");
```
With:
```rust
    let mut config = {
        let _span = tracing::info_span!("config_load").entered();
        let cfg = Config::load().expect("Failed to load configuration");
        tracing::info!(
            device_type = %cfg.device.device_type,
            host        = %cfg.server.host,
            port        = cfg.server.port,
            "config loaded"
        );
        cfg
    };
```

- [ ] **Step 5: Wrap device detection block in a `device_detect` span (lines ~163-219)**

Replace the entire `if config.device.device_type == "auto"` block with:
```rust
    {
        let _span = tracing::info_span!("device_detect").entered();
        if config.device.device_type == "auto" {
            tracing::info!("auto-detecting compute device");
            let temp_gpu_manager = GpuManager::new();
            match temp_gpu_manager.get_info() {
                Ok(info) => {
                    match info.backend {
                        crate::core::gpu::GpuBackend::Cuda => {
                            config.device.device_type = "cuda".to_string();
                            tracing::info!(backend = "cuda", "cuda detected");
                            if config.device.device_ids.is_none() && info.count > 0 {
                                let ids: Vec<usize> = (0..info.count).collect();
                                config.device.device_ids = Some(ids.clone());
                                tracing::info!(
                                    device_count = info.count,
                                    device_ids   = ?ids,
                                    "cuda devices configured"
                                );
                            }
                        },
                        crate::core::gpu::GpuBackend::Metal => {
                            config.device.device_type = "mps".to_string();
                            tracing::info!(backend = "mps", "metal detected");
                            if config.device.metal_optimize_for_apple_silicon {
                                let optimal_threads = (num_cpus::get() * 3) / 4;
                                config.device.num_threads = optimal_threads;
                                tracing::info!(
                                    threads        = optimal_threads,
                                    fp16           = config.device.use_fp16,
                                    shader_caching = config.device.metal_cache_shaders,
                                    "apple silicon configured"
                                );
                            }
                            if config.device.device_ids.is_none() {
                                config.device.device_ids = Some(vec![0]);
                            }
                        },
                        crate::core::gpu::GpuBackend::Cpu => {
                            config.device.device_type = "cpu".to_string();
                            tracing::info!(backend = "cpu", "no gpu detected, using cpu");
                        }
                    }
                },
                Err(e) => {
                    tracing::warn!(error = %e, backend = "cpu", "gpu detection failed, defaulting to cpu");
                    config.device.device_type = "cpu".to_string();
                }
            }
            tracing::info!(backend = %config.device.device_type, "device detection complete");
        } else {
            tracing::info!(backend = %config.device.device_type, "device configured from config");
        }
    }
```

- [ ] **Step 6: Wrap `log_system_info()` call and ORT detection in spans (lines ~221-230)**

Replace:
```rust
    log_system_info();

    // Log ORT auto-detection result
    if let Ok(detected) = std::env::var("_ORT_AUTODETECTED") {
        info!("[AUTO] ORT library auto-detected: {}", detected);
    } else if let Ok(explicit) = std::env::var("ORT_DYLIB_PATH") {
        info!("[ORT] Using ORT_DYLIB_PATH from environment: {}", explicit);
    } else {
        log::warn!("[WARN] ORT library not found. Set ORT_DYLIB_PATH if ONNX inference is needed.");
    }
```
With:
```rust
    {
        let _span = tracing::info_span!("system_info").entered();
        log_system_info();
    }

    if let Ok(detected) = std::env::var("_ORT_AUTODETECTED") {
        tracing::info!(ort_path = %detected, source = "auto-detected", "ort library found");
    } else if let Ok(explicit) = std::env::var("ORT_DYLIB_PATH") {
        tracing::info!(ort_path = %explicit, source = "environment", "ort library configured");
    } else {
        tracing::warn!("ort library not found; set ORT_DYLIB_PATH if onnx inference is needed");
    }
```

- [ ] **Step 7: Update `#[cfg(feature = "torch")]` init block (lines ~232-252)**

Replace:
```rust
    #[cfg(feature = "torch")]
    {
        info!("[INIT] Initializing PyTorch environment...");

        tch::maybe_init_cuda();

        match crate::core::torch_autodetect::initialize_torch().await {
            Ok(torch_config) => {
                info!("[OK] PyTorch initialized successfully");
                info!("   ├─ Backend: {:?}", torch_config.backend);
                info!("   ├─ Path: {:?}", torch_config.libtorch_path);
                info!("   └─ Version: {}", torch_config.version);
            }
            Err(e) => {
                log::warn!("[WARN]  PyTorch initialization failed: {}", e);
                log::warn!("   └─ ML inference features will be limited");
            }
        }
    }
```
With:
```rust
    #[cfg(feature = "torch")]
    {
        let _span = tracing::info_span!("pytorch_init").entered();
        tch::maybe_init_cuda();
        match crate::core::torch_autodetect::initialize_torch().await {
            Ok(torch_config) => {
                tracing::info!(
                    backend = ?torch_config.backend,
                    path    = ?torch_config.libtorch_path,
                    version = %torch_config.version,
                    "pytorch initialized"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "pytorch initialization failed; ml inference limited");
            }
        }
    }
```

- [ ] **Step 8: Update tensor pool init (lines ~254-260)**

Replace:
```rust
    let tensor_pool = if config.performance.enable_tensor_pooling {
        info!("[OPT] Tensor pooling enabled (max: {} tensors)", config.performance.max_pooled_tensors);
        Some(Arc::new(crate::tensor_pool::TensorPool::new(config.performance.max_pooled_tensors)))
    } else {
        None
    };
```
With:
```rust
    let tensor_pool = if config.performance.enable_tensor_pooling {
        tracing::info!(max_tensors = config.performance.max_pooled_tensors, "tensor pooling enabled");
        Some(Arc::new(crate::tensor_pool::TensorPool::new(config.performance.max_pooled_tensors)))
    } else {
        None
    };
```

- [ ] **Step 9: Add `components_init` span (lines ~262-294)**

Enter the span before the component creation block and drop it explicitly after. This requires **no restructuring** of existing variable bindings — just wrapping with a span entry/exit.

Add this line immediately before `// Initialize components`:
```rust
    let _comp_span = tracing::info_span!("components_init").entered();
    tracing::info!("initializing core components");
```

Then in the `_compression` block, replace the existing `info!` call:
```rust
        info!("[OPT] Result compression enabled (level: {})", config.performance.compression_level);
```
With:
```rust
        tracing::info!(level = config.performance.compression_level, "result compression enabled");
```

Then replace the closing `info!("[OK] Core components initialized")` line:
```rust
    info!("[OK] Core components initialized");
```
With:
```rust
    tracing::info!(
        workers_min   = config.performance.min_workers,
        workers_max   = config.performance.max_workers,
        cache_size_mb = config.performance.cache_size_mb,
        "core components initialized"
    );
    drop(_comp_span);
```

- [ ] **Step 10: Add `gpu_init` span (lines ~296-331)**

Replace the GPU manager block:
```rust
    // Initialize GPU manager
    info!("[GPU] Initializing GPU manager...");
    let gpu_manager = Arc::new(GpuManager::new());

    // Try to detect GPUs
    match gpu_manager.get_info() {
        Ok(info) => { ... }
        Err(e) => { ... }
    }
```
With:
```rust
    let gpu_manager = Arc::new(GpuManager::new());
    {
        let _span = tracing::info_span!("gpu_init").entered();
        match gpu_manager.get_info() {
            Ok(info) if info.available => {
                let device_names: Vec<String> = info.devices.iter()
                    .map(|d| format!("{}:{}", d.name, d.id))
                    .collect();
                tracing::info!(
                    backend      = ?info.backend,
                    gpu_count    = info.count,
                    device_names = %device_names.join(", "),
                    "gpu init complete"
                );
            }
            Ok(_) => {
                tracing::info!(backend = "cpu", "gpu init complete, no gpu available");
            }
            Err(e) => {
                tracing::warn!(error = %e, "gpu detection failed, falling back to cpu");
            }
        }
    }
```

- [ ] **Step 11: Add `download_init` span (lines ~333-342)**

Replace:
```rust
    // Initialize model download manager
    info!("[DOWNLOAD] Initializing model download manager...");
    let cache_dir = std::env::var("MODEL_CACHE_DIR")
        .unwrap_or_else(|_| "./models".to_string());
    let download_manager = Arc::new(
        ModelDownloadManager::new(&cache_dir)
            .expect("Failed to create model download manager")
    );
    download_manager.initialize().await.expect("Failed to initialize download manager");
    info!("[OK] Model download manager ready at: {}", cache_dir);
```
With:
```rust
    let download_manager = {
        let _span = tracing::info_span!("download_init").entered();
        let cache_dir = std::env::var("MODEL_CACHE_DIR")
            .unwrap_or_else(|_| "./models".to_string());
        let dm = Arc::new(
            ModelDownloadManager::new(&cache_dir)
                .expect("Failed to create model download manager")
        );
        dm.initialize().await.expect("Failed to initialize download manager");
        tracing::info!(cache_dir = %cache_dir, "download manager ready");
        dm
    };
```

- [ ] **Step 12: Add `audio_init` span (lines ~344-350)**

Replace:
```rust
    // Initialize audio model manager (legacy)
    info!("[AUDIO] Initializing audio model manager...");
    let audio_model_dir = std::env::var("AUDIO_MODEL_DIR")
        .unwrap_or_else(|_| "./models/audio".to_string());
    let audio_model_manager = Arc::new(crate::core::audio_models::AudioModelManager::new(&audio_model_dir));
    audio_model_manager.initialize_default_models().await.ok();
    info!("[OK] Audio model manager ready at: {}", audio_model_dir);
```
With:
```rust
    let audio_model_manager = {
        let _span = tracing::info_span!("audio_init").entered();
        let audio_model_dir = std::env::var("AUDIO_MODEL_DIR")
            .unwrap_or_else(|_| "./models/audio".to_string());
        let am = Arc::new(crate::core::audio_models::AudioModelManager::new(&audio_model_dir));
        am.initialize_default_models().await.ok();
        tracing::info!(model_dir = %audio_model_dir, "audio manager ready");
        am
    };
```

- [ ] **Step 13: Add `tts_init` span (lines ~352-361)**

Replace:
```rust
    // Initialize modern TTS manager
    info!("[TTS]  Initializing TTS engines...");
    let tts_config = crate::core::tts_manager::TTSManagerConfig::default();
    let tts_manager = Arc::new(crate::core::tts_manager::TTSManager::new(tts_config));
    tts_manager.initialize_defaults().await.expect("Failed to initialize TTS manager");
    let tts_stats = tts_manager.get_stats();
    info!("[OK] TTS Manager ready - {} engine(s) loaded", tts_stats.total_engines);
    for engine_id in &tts_stats.engine_ids {
        info!("   └─ {}", engine_id);
    }
```
With:
```rust
    let tts_manager = {
        let _span = tracing::info_span!("tts_init").entered();
        let tts_config = crate::core::tts_manager::TTSManagerConfig::default();
        let tm = Arc::new(crate::core::tts_manager::TTSManager::new(tts_config));
        tm.initialize_defaults().await.expect("Failed to initialize TTS manager");
        let tts_stats = tm.get_stats();
        tracing::info!(
            engine_count = tts_stats.total_engines,
            engine_ids   = %tts_stats.engine_ids.join(", "),
            "tts manager ready"
        );
        tm
    };
```

- [ ] **Step 14: Update model preload block (lines ~365-372)**

Replace:
```rust
    if config.performance.preload_models_on_startup {
        info!("[WARMUP] Pre-loading models on startup...");
        for model_name in &config.models.auto_load {
            if let Ok(_) = model_manager.get_model(model_name) {
                info!("   └─ Pre-loaded: {}", model_name);
            }
        }
    }
```
With:
```rust
    if config.performance.preload_models_on_startup {
        let _span = tracing::info_span!("model_preload").entered();
        tracing::info!(model_count = config.models.auto_load.len(), "preloading models on startup");
        for model_name in &config.models.auto_load {
            if let Ok(_) = model_manager.get_model(model_name) {
                tracing::info!(model = %model_name, status = "loaded", "model preloaded");
            } else {
                tracing::warn!(model = %model_name, status = "not_found", "model preload skipped");
            }
        }
    }
```

- [ ] **Step 15: Update warmup spawn block (lines ~374-387)**

Replace:
```rust
    info!("[WARMUP] Warming up inference engine (background)...");
    {
        let warmup_engine = inference_engine.clone();
        let config_cloned = config.clone();
        tokio::spawn(async move {
            match warmup_engine.warmup(&config_cloned).await {
                Ok(()) => info!("[OK] Inference engine warmup complete"),
                Err(e) => log::warn!("[WARN] Inference engine warmup failed: {}", e),
            }
        });
    }
```
With:
```rust
    tracing::info!("starting background warmup");
    {
        let warmup_engine = inference_engine.clone();
        let config_cloned = config.clone();
        tokio::spawn(async move {
            let _span = tracing::info_span!("warmup").entered();
            match warmup_engine.warmup(&config_cloned).await {
                Ok(()) => tracing::info!(status = "ok", "warmup complete"),
                Err(e) => tracing::warn!(error = %e, status = "failed", "warmup failed"),
            }
        });
    }
```

- [ ] **Step 16: Update server bind logging (lines ~389-391)**

Replace:
```rust
    let addr = format!("{}:{}", config.server.host, config.server.port);
    let display_addr = format!("localhost:{}", config.server.port);
    info!("[SERVER] Starting HTTP server on {}...", addr);
```
With:
```rust
    let addr = format!("{}:{}", config.server.host, config.server.port);
    let display_addr = format!("localhost:{}", config.server.port);
    tracing::info!(addr = %addr, workers = config.server.workers, "binding http server");
```

- [ ] **Step 17: Replace the "server ready" println! banner (lines ~431-451)**

Replace the entire block from `println!("\n{}", "═".repeat(80));` through `println!("{}\n", "═".repeat(80));` with:
```rust
    tracing::info!(
        server_url    = %format!("http://{}", display_addr),
        health_url    = %format!("http://{}/health", display_addr),
        liveness_url  = %format!("http://{}/health/live", display_addr),
        readiness_url = %format!("http://{}/health/ready", display_addr),
        workers       = config.server.workers,
        "server ready"
    );
    tracing::info!(
        tensor_pooling = config.performance.enable_tensor_pooling,
        compression    = config.performance.enable_result_compression,
        adaptive_batch = config.performance.adaptive_batch_timeout,
        lru_caching    = config.performance.enable_caching,
        cache_size_mb  = config.performance.cache_size_mb,
        "server features"
    );
    #[cfg(feature = "metrics")]
    tracing::info!(
        metrics_url = %format!("http://{}/metrics", display_addr),
        "prometheus metrics available"
    );
```

- [ ] **Step 18: Update the remaining `info!` call after server start (line ~453)**

Replace:
```rust
    info!("[OK] Server started successfully - Workers: {}", config.server.workers);
```
With:
```rust
    tracing::info!(workers = config.server.workers, "server started successfully");
```

- [ ] **Step 19: Remove the `actix_middleware::Logger` wrap from the App builder (lines ~475-477)**

Remove these three lines entirely:
```rust
            .wrap(actix_middleware::Logger::new(
                r#"%a "%r" %s %b "%{Referer}i" %T %{X-Correlation-ID}o"#
            ))
```

- [ ] **Step 20: Update shutdown signal handlers (lines ~502-536)**

Replace:
```rust
        info!("[SHUTDOWN] Shutdown signal received, draining requests...");
```
With:
```rust
        tracing::info!("shutdown signal received, draining requests");
```

Replace (in `shutdown_signal()`):
```rust
        _ = ctrl_c => {
            info!("[SHUTDOWN] Received Ctrl+C signal");
        },
        _ = terminate => {
            info!("[SHUTDOWN] Received SIGTERM signal");
        },
```
With:
```rust
        _ = ctrl_c => {
            tracing::info!(signal = "ctrl_c", "shutdown signal received");
        },
        _ = terminate => {
            tracing::info!(signal = "sigterm", "shutdown signal received");
        },
```

- [ ] **Step 21: Rewrite `log_system_info()` to use tracing and remove all println! calls**

Replace the entire `fn log_system_info()` function with:
```rust
fn log_system_info() {
    let cpu_count = num_cpus::get();
    tracing::info!(cpu_cores = cpu_count, "cpu info");

    if let Ok(sys_info) = sys_info::mem_info() {
        let total_gb = sys_info.total as f64 / 1024.0 / 1024.0;
        let avail_gb = sys_info.avail as f64 / 1024.0 / 1024.0;
        tracing::info!(ram_total_gb = total_gb, ram_avail_gb = avail_gb, "memory info");
    }

    #[cfg(target_os = "macos")]
    {
        if GpuManager::is_metal_available() {
            let gpu_manager = GpuManager::new();
            if let Some(metal_info) = gpu_manager.get_metal_info_string() {
                tracing::info!(metal_gpu = %metal_info, "metal available");
            } else {
                tracing::info!(metal_gpu = "apple silicon", "metal available");
            }
        } else {
            tracing::info!(metal = false, "metal not available");
        }
    }

    if cfg!(feature = "cuda") {
        if GpuManager::is_cuda_runtime_available() {
            if let Some(cuda_info) = GpuManager::get_cuda_info() {
                tracing::info!(cuda = true, cuda_info = %cuda_info, "cuda runtime available");
            } else {
                tracing::info!(cuda = true, "cuda runtime available");
            }
        } else {
            tracing::warn!(cuda_feature = true, cuda_runtime = false, "cuda feature enabled but runtime not detected");
        }
    } else {
        #[cfg(not(target_os = "macos"))]
        tracing::info!(cuda = false, "cuda disabled");
    }

    if cfg!(feature = "onnx") {
        tracing::info!(onnx = true, "onnx enabled");
    } else {
        tracing::info!(onnx = false, "onnx disabled");
    }

    #[cfg(feature = "audio")]
    tracing::info!(audio = true, "audio processing enabled");
    #[cfg(not(feature = "audio"))]
    tracing::info!(audio = false, "audio processing disabled");

    #[cfg(feature = "image-security")]
    tracing::info!(image_security = true, "image security enabled");
    #[cfg(not(feature = "image-security"))]
    tracing::info!(image_security = false, "image security disabled");

    tracing::info!(
        os   = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        "os info"
    );
}
```

- [ ] **Step 22: Update `#[cfg(feature = "metrics")]` Prometheus init block (lines ~146-153)**

Replace:
```rust
    #[cfg(feature = "metrics")]
    {
        if let Err(e) = prometheus::init_metrics() {
            log::warn!("[WARN] Failed to initialize Prometheus metrics: {}", e);
        } else {
            info!("[OK] Prometheus metrics initialized");
        }
    }
```
With:
```rust
    #[cfg(feature = "metrics")]
    {
        if let Err(e) = prometheus::init_metrics() {
            tracing::warn!(error = %e, "prometheus metrics initialization failed");
        } else {
            tracing::info!("prometheus metrics initialized");
        }
    }
```

- [ ] **Step 23: Build to confirm zero errors**

```bash
cargo build 2>&1 | grep "^error" | head -20
```

Expected: no output (zero errors)

- [ ] **Step 24: Run the full test suite**

```bash
cargo test 2>&1 | tail -15
```

Expected: all tests pass, no new failures

- [ ] **Step 25: Verify no println! or log:: calls remain in main.rs**

```bash
grep -n "println!\|log::" src/main.rs
```

Expected: no output

- [ ] **Step 26: Commit**

```bash
git add src/main.rs
git commit -m "feat(logging): unify main.rs to tracing::, add startup phase spans, remove println!"
```

---

## Acceptance Criteria Checklist

Run these after all 4 tasks are complete:

```bash
# No println! in main.rs
grep -n "println!" src/main.rs && echo "FAIL: println found" || echo "PASS: no println"

# No log:: in main.rs, engine.rs, manager.rs
grep -rn "use log::" src/main.rs src/core/engine.rs src/models/manager.rs \
  && echo "FAIL: log:: import found" || echo "PASS: no log:: imports"

# actix Logger removed
grep -n "actix_middleware::Logger" src/main.rs \
  && echo "FAIL: actix Logger still present" || echo "PASS: actix Logger removed"

# status_class helper present
grep -n "fn status_class" src/middleware/request_logger.rs \
  && echo "PASS: status_class found" || echo "FAIL: status_class missing"

# slow_request event present
grep -n "slow_request" src/middleware/request_logger.rs \
  && echo "PASS: slow_request found" || echo "FAIL: slow_request missing"

# inference span present
grep -n "info_span.*inference" src/core/engine.rs \
  && echo "PASS: inference span found" || echo "FAIL: inference span missing"

# Full test suite
cargo test 2>&1 | tail -3
```
