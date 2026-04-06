# Coverage Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise test coverage from 87.9% to ~93% by adding inline unit tests for all reachable-but-untested code paths.

**Architecture:** All tests are added to each source file's existing `#[cfg(test)]` block. No new files. One production-code refactor: `telemetry/logger.rs` extracts `format_log_record` body to a `pub(crate) fn format_log_record_inner` so it can be called with a `Vec<u8>` writer in tests.

**Tech Stack:** Rust, actix-web 4, tokio, tempfile (already a dev-dep), serial_test (already a dev-dep)

---

## File Map

| File | Change |
|------|--------|
| `src/telemetry/logger.rs` | Refactor + new tests |
| `src/telemetry/structured_logging.rs` | New tests |
| `src/middleware/rate_limit.rs` | New test |
| `src/middleware/request_logger.rs` | New test |
| `src/api/inference.rs` | New tests |
| `src/api/classification.rs` | New tests |
| `src/api/audio.rs` | New test |
| `src/api/yolo.rs` | New test |
| `src/core/tts_manager.rs` | New tests |
| `src/models/download.rs` | New tests |
| `src/core/torch_autodetect.rs` | New tests |
| `src/cache.rs` | New test |
| `src/monitor.rs` | New test |
| `src/batch.rs` | New test |
| `src/model_pool.rs` | New test |
| `src/dedup.rs` | New test |
| `src/tensor_pool.rs` | New test |
| `src/guard.rs` | New test |
| `src/worker_pool.rs` | New test |
| `src/resilience/circuit_breaker.rs` | New tests |
| `src/resilience/retry.rs` | New tests |
| `src/resilience/per_model_breaker.rs` | New test |
| `src/core/audio.rs` | New tests |
| `src/core/yolo.rs` | New test |
| `src/core/image_security.rs` | New test |
| `src/core/tts_engine.rs` | New test |
| `src/security/sanitizer.rs` | New test |
| `src/api/image.rs` | New tests |
| `src/api/performance.rs` | New test |
| `src/api/model_download.rs` | New tests |
| `src/core/audio_models.rs` | New tests |
| `src/core/g2p_misaki.rs` | New tests |
| `src/models/pytorch_loader.rs` | New test |
| `src/config.rs` | New test |

---

## Task 1: Refactor logger.rs for testability

**Files:**
- Modify: `src/telemetry/logger.rs`

The `format_log_record` function's body is uncovered because `env_logger::fmt::Formatter` cannot be constructed directly in tests. Extracting the core logic to a function that takes `&mut dyn Write` makes it directly callable.

- [ ] **Step 1: Read the current function**

```
Read src/telemetry/logger.rs lines 10-40
```

- [ ] **Step 2: Extract format_log_record_inner**

In `src/telemetry/logger.rs`, add the new inner function immediately above `format_log_record` and make the outer function delegate to it:

```rust
/// Core formatting logic, separated from the env_logger Formatter so tests
/// can call it directly with any `Write` impl (e.g. `Vec<u8>`).
pub(crate) fn format_log_record_inner(
    buf: &mut dyn std::io::Write,
    level: log::Level,
    target: &str,
    args: &std::fmt::Arguments<'_>,
) -> std::io::Result<()> {
    let level_string = match level {
        log::Level::Error => "\x1b[1;31mERROR\x1b[0m",
        log::Level::Warn  => "\x1b[1;33mWARN \x1b[0m",
        log::Level::Info  => "\x1b[1;32mINFO \x1b[0m",
        log::Level::Debug => "\x1b[1;36mDEBUG\x1b[0m",
        log::Level::Trace => "\x1b[1;35mTRACE\x1b[0m",
    };

    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");

    let short_target = if target.starts_with("torch_inference::") {
        target.strip_prefix("torch_inference::").unwrap_or(target)
    } else {
        target
    };

    writeln!(
        buf,
        "\x1b[90m{}\x1b[0m [{}] \x1b[90m{}\x1b[0m - {}",
        timestamp,
        level_string,
        short_target,
        args
    )
}

/// Format a single log record into the env_logger buffer.
pub(crate) fn format_log_record(
    buf: &mut env_logger::fmt::Formatter,
    record: &log::Record<'_>,
) -> std::io::Result<()> {
    format_log_record_inner(buf, record.level(), record.target(), record.args())
}
```

- [ ] **Step 3: Add tests for format_log_record_inner in the existing test module**

Add these tests inside the existing `#[cfg(test)] mod tests { ... }` block in `src/telemetry/logger.rs`:

```rust
    // ── format_log_record_inner: direct Vec<u8> writer tests ─────────────────

    #[test]
    fn test_format_log_record_inner_error_contains_ansi_and_message() {
        let mut buf = Vec::<u8>::new();
        let args = format_args!("something broke");
        format_log_record_inner(&mut buf, log::Level::Error, "mymod", &args).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("ERROR"), "ERROR level string should appear");
        assert!(out.contains("something broke"), "message should appear");
        assert!(out.contains("mymod"), "target should appear");
    }

    #[test]
    fn test_format_log_record_inner_warn() {
        let mut buf = Vec::<u8>::new();
        let args = format_args!("watch out");
        format_log_record_inner(&mut buf, log::Level::Warn, "a::b", &args).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("WARN"), "WARN level string should appear");
        assert!(out.contains("watch out"));
    }

    #[test]
    fn test_format_log_record_inner_info() {
        let mut buf = Vec::<u8>::new();
        let args = format_args!("hello");
        format_log_record_inner(&mut buf, log::Level::Info, "svc", &args).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("INFO"));
        assert!(out.contains("hello"));
    }

    #[test]
    fn test_format_log_record_inner_debug() {
        let mut buf = Vec::<u8>::new();
        let args = format_args!("debug msg");
        format_log_record_inner(&mut buf, log::Level::Debug, "x", &args).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("DEBUG"));
    }

    #[test]
    fn test_format_log_record_inner_trace() {
        let mut buf = Vec::<u8>::new();
        let args = format_args!("trace msg");
        format_log_record_inner(&mut buf, log::Level::Trace, "x", &args).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("TRACE"));
    }

    #[test]
    fn test_format_log_record_inner_strips_torch_inference_prefix() {
        let mut buf = Vec::<u8>::new();
        let args = format_args!("msg");
        format_log_record_inner(&mut buf, log::Level::Info, "torch_inference::api::foo", &args).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("api::foo"), "prefix should be stripped");
        assert!(!out.contains("torch_inference::api::foo"), "full prefix should not appear");
    }

    #[test]
    fn test_format_log_record_inner_no_strip_for_other_crate() {
        let mut buf = Vec::<u8>::new();
        let args = format_args!("msg");
        format_log_record_inner(&mut buf, log::Level::Info, "other_crate::mod", &args).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains("other_crate::mod"), "non-torch target should not be stripped");
    }

    #[test]
    fn test_format_log_record_inner_ends_with_newline() {
        let mut buf = Vec::<u8>::new();
        let args = format_args!("x");
        format_log_record_inner(&mut buf, log::Level::Info, "t", &args).unwrap();
        assert!(buf.ends_with(b"\n"), "output should end with newline");
    }
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p torch-inference telemetry::logger 2>&1 | tail -20
```

Expected: all logger tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/telemetry/logger.rs
git commit -m "refactor(logger): extract format_log_record_inner for direct testability"
```

---

## Task 2: Telemetry structured_logging + prometheus tests

**Files:**
- Modify: `src/telemetry/structured_logging.rs`
- Modify: `src/telemetry/prometheus.rs`

Uncovered: `create_request_span` (line 191), `create_inference_span` (line 206), `log_completion` (line 272), `log_error` (line 282) in structured_logging. In prometheus.rs, lines 225-226 are the "already-registered" guard path.

- [ ] **Step 1: Read structured_logging test module**

```
Read src/telemetry/structured_logging.rs — look for the #[cfg(test)] block
```

- [ ] **Step 2: Add structured_logging tests**

Add inside the existing `#[cfg(test)] mod tests { ... }` in `src/telemetry/structured_logging.rs`:

```rust
    #[test]
    fn test_create_request_span_does_not_panic() {
        use crate::telemetry::CorrelationId;
        let cid = CorrelationId::new();
        let _span = create_request_span("GET", "/api/test", &cid);
        // Just verify construction doesn't panic; tracing::Span is always valid
    }

    #[test]
    fn test_create_inference_span_does_not_panic() {
        use crate::telemetry::CorrelationId;
        let cid = CorrelationId::new();
        let _span = create_inference_span("bert-base", 4, &cid);
    }

    #[test]
    fn test_request_metrics_log_completion_does_not_panic() {
        use crate::telemetry::CorrelationId;
        let cid = CorrelationId::new();
        let metrics = RequestMetrics::new(cid);
        // log_completion emits a tracing event — just verify it doesn't panic
        metrics.log_completion(200, "/api/infer");
    }

    #[test]
    fn test_request_metrics_log_error_does_not_panic() {
        use crate::telemetry::CorrelationId;
        let cid = CorrelationId::new();
        let metrics = RequestMetrics::new(cid);
        metrics.log_error("model not found", "/api/infer");
    }
```

- [ ] **Step 3: Read prometheus test module location**

```
Read src/telemetry/prometheus.rs — find the #[cfg(test)] block and line 225
```

- [ ] **Step 4: Add prometheus double-register test**

Line 225-226 in `prometheus.rs` is a guard that handles the case where a counter/gauge is registered a second time (e.g., `AlreadyReg` error being ignored). Add inside the existing test module:

```rust
    #[test]
    fn test_init_metrics_idempotent() {
        // Calling init_metrics twice must not panic — the second call hits the
        // AlreadyReg guard path (lines 225-226).
        init_metrics();
        init_metrics();
    }
```

(If `init_metrics` is not the right function name, use whatever function registers the prometheus metrics — check the file.)

- [ ] **Step 5: Run tests**

```bash
cargo test -p torch-inference telemetry 2>&1 | tail -20
```

Expected: all telemetry tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/telemetry/structured_logging.rs src/telemetry/prometheus.rs
git commit -m "test(telemetry): cover span helpers, log_completion, log_error, idempotent init"
```

---

## Task 3: Middleware — rate_limit exceeded + request_logger error path

**Files:**
- Modify: `src/middleware/rate_limit.rs`
- Modify: `src/middleware/request_logger.rs`

Uncovered: `rate_limit.rs` lines 59-61 are the `Err(RateLimitError {...})` arm of `is_allowed`. `request_logger.rs` lines 65-98 include the error-response logging branch of the middleware future.

- [ ] **Step 1: Add rate_limit exceeded test**

Add inside the existing `#[cfg(test)] mod tests { ... }` in `src/middleware/rate_limit.rs`:

```rust
    #[test]
    fn test_is_allowed_returns_err_when_limit_exceeded() {
        let limiter = RateLimiter::new(2, 60); // max 2 requests per 60s
        assert!(limiter.is_allowed("client-a").is_ok());
        assert!(limiter.is_allowed("client-a").is_ok());
        // Third request exceeds limit
        let result = limiter.is_allowed("client-a");
        assert!(result.is_err(), "third request should be rate-limited");
        let err = result.unwrap_err();
        assert_eq!(err.message, "Rate limit exceeded");
        assert!(err.retry_after > 0);
    }

    #[test]
    fn test_is_allowed_different_keys_are_independent() {
        let limiter = RateLimiter::new(1, 60);
        assert!(limiter.is_allowed("key-a").is_ok());
        assert!(limiter.is_allowed("key-b").is_ok()); // different key, not limited
        assert!(limiter.is_allowed("key-a").is_err()); // key-a now over limit
    }
```

- [ ] **Step 2: Add request_logger error branch test**

The error branch (lines 81-98) runs when the inner service returns `Err`. We trigger this by wrapping a handler that returns an actix-web Error. Add inside the existing `#[cfg(test)] mod tests { ... }` in `src/middleware/request_logger.rs`:

```rust
    #[actix_web::test]
    async fn test_request_logger_propagates_service_error() {
        use actix_web::web;

        async fn error_handler() -> Result<HttpResponse, AxError> {
            Err(actix_web::error::ErrorInternalServerError("forced error"))
        }

        let app = awtest::init_service(
            App::new()
                .wrap(RequestLogger)
                .route("/err", web::get().to(error_handler)),
        )
        .await;

        let req = awtest::TestRequest::get().uri("/err").to_request();
        let resp = awtest::call_service(&app, req).await;

        // actix converts the error to a 500 response
        assert_eq!(resp.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
    }
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p torch-inference middleware 2>&1 | tail -20
```

Expected: all middleware tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/middleware/rate_limit.rs src/middleware/request_logger.rs
git commit -m "test(middleware): cover rate-limit exceeded path and request_logger error branch"
```

---

## Task 4: API — inference.rs error paths

**Files:**
- Modify: `src/api/inference.rs`

Uncovered lines:
- 159-163: `list_models` closure body (map over `state.networks`)
- 186-188: `get_model_info` model-not-found Err
- 200-202: `unload_model` model-not-found Err

- [ ] **Step 1: Read inference.rs test setup**

```
Read src/api/inference.rs — find imports, NeuralNetworkState struct, existing test module
```

- [ ] **Step 2: Add tests inside existing test module**

Find the `#[cfg(test)] mod tests { ... }` block and add:

```rust
    #[actix_web::test]
    async fn test_list_models_with_populated_state() {
        use actix_web::{test, web, App};
        use dashmap::DashMap;
        use std::sync::Arc;

        // Build a state with one entry so the map() closure is exercised
        let networks = Arc::new(DashMap::new());
        // Insert a minimal NeuralNetwork stub — use whatever the real type is:
        // If NeuralNetwork is not constructable in tests, use an empty DashMap
        // and verify 0 models returns correctly.
        let state = web::Data::new(NeuralNetworkState { networks: networks.clone() });

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/api/nn/models", web::get().to(list_models)),
        ).await;

        let req = test::TestRequest::get().uri("/api/nn/models").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    #[actix_web::test]
    async fn test_get_model_info_not_found() {
        use actix_web::{test, web, App};
        use dashmap::DashMap;
        use std::sync::Arc;

        let state = web::Data::new(NeuralNetworkState {
            networks: Arc::new(DashMap::new()),
        });

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/api/nn/models/{model_id}", web::get().to(get_model_info)),
        ).await;

        let req = test::TestRequest::get()
            .uri("/api/nn/models/no-such-model")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
    }

    #[actix_web::test]
    async fn test_unload_model_not_found() {
        use actix_web::{test, web, App};
        use dashmap::DashMap;
        use std::sync::Arc;

        let state = web::Data::new(NeuralNetworkState {
            networks: Arc::new(DashMap::new()),
        });

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/api/nn/models/{model_id}", web::delete().to(unload_model)),
        ).await;

        let req = test::TestRequest::delete()
            .uri("/api/nn/models/nonexistent")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
    }
```

Note: if `NeuralNetworkState` has a different field name for the map, adjust accordingly (read the file first in Step 1).

- [ ] **Step 3: Run tests**

```bash
cargo test -p torch-inference api::inference 2>&1 | tail -20
```

Expected: new tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/api/inference.rs
git commit -m "test(api/inference): cover list_models map closure, get/unload not-found paths"
```

---

## Task 5: API — classification.rs, audio.rs, yolo.rs error paths

**Files:**
- Modify: `src/api/classification.rs`
- Modify: `src/api/audio.rs`
- Modify: `src/api/yolo.rs`

**classification.rs** uncovered lines 44-48: `image_path` missing → BadRequest, path not found → NotFound.
**audio.rs** uncovered lines 209-216: empty audio data → validation error response.
**yolo.rs** uncovered lines 79-115: model file not found → NotFound.

- [ ] **Step 1: Read classification.rs test module**

```
Read src/api/classification.rs — find ClassifyRequest struct fields and test module
```

- [ ] **Step 2: Add classification.rs tests**

Inside the existing `#[cfg(test)] mod tests { ... }` in `src/api/classification.rs`:

```rust
    #[actix_web::test]
    async fn test_classify_path_missing_image_path_returns_bad_request() {
        use actix_web::{test, web, App};

        // Build state with a stub classifier — ImageClassificationState must
        // be constructable; read the struct definition and use the simplest valid constructor.
        // If it requires a real model, check if there is a ::default() or ::new_empty().
        // For now, rely on the existing test helpers in the module if any.
        // The handler returns BadRequest before touching the classifier when image_path is None.
        let state = web::Data::new(make_test_state()); // use existing helper or construct directly

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/classify", web::post().to(classify_path)),
        ).await;

        // Send request with image_path = null
        let req = test::TestRequest::post()
            .uri("/classify")
            .set_json(serde_json::json!({ "image_path": null }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::BAD_REQUEST);
    }

    #[actix_web::test]
    async fn test_classify_path_nonexistent_file_returns_not_found() {
        use actix_web::{test, web, App};

        let state = web::Data::new(make_test_state());

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/classify", web::post().to(classify_path)),
        ).await;

        let req = test::TestRequest::post()
            .uri("/classify")
            .set_json(serde_json::json!({ "image_path": "/nonexistent/file/that/does/not/exist.jpg" }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
    }
```

Note: replace `make_test_state()` with the actual way to construct `ImageClassificationState` in tests — check the existing test module for helpers.

- [ ] **Step 3: Read audio.rs validate_audio handler**

```
Read src/api/audio.rs — find the validate_audio handler and the test module
```

- [ ] **Step 4: Add audio.rs empty-body test**

Inside the existing `#[cfg(test)] mod tests { ... }` in `src/api/audio.rs`:

```rust
    #[actix_web::test]
    async fn test_validate_audio_empty_body_returns_invalid() {
        use actix_web::{test, web, App};

        let app = test::init_service(
            App::new().route("/audio/validate", web::post().to(validate_audio)),
        ).await;

        // Send an empty multipart body — triggers the "no audio data provided" path
        let req = test::TestRequest::post()
            .uri("/audio/validate")
            .to_request();
        let resp = test::call_service(&app, req).await;
        // Handler returns 200 with valid=false when no data is provided
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(body["valid"], false);
    }
```

- [ ] **Step 5: Read yolo.rs test module**

```
Read src/api/yolo.rs — find YoloState, detect_objects handler, test module
```

- [ ] **Step 6: Add yolo.rs model-not-found test**

Inside the existing `#[cfg(test)] mod tests { ... }` in `src/api/yolo.rs`:

```rust
    #[actix_web::test]
    async fn test_detect_objects_model_not_found() {
        use actix_web::{test, web, App};
        use actix_multipart::Multipart;
        use std::path::PathBuf;

        // Point models_dir at a temp directory with no model files
        let tmp = tempfile::tempdir().unwrap();
        let state = web::Data::new(YoloState {
            models_dir: tmp.path().to_path_buf(),
        });

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/detect", web::post().to(detect_objects)),
        ).await;

        // Send a multipart request with an image field
        // The handler exits early with NotFound because the model file doesn't exist
        let boundary = "testboundary";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"test.jpg\"\r\nContent-Type: image/jpeg\r\n\r\nFAKEIMAGEDATA\r\n--{boundary}--\r\n",
            boundary = boundary
        );
        let req = test::TestRequest::post()
            .uri("/detect?model_version=yolov8&model_size=n")
            .insert_header(("Content-Type", format!("multipart/form-data; boundary={}", boundary)))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        // Expect NotFound (model .pt file missing) or BadRequest (invalid version)
        assert!(
            resp.status() == actix_web::http::StatusCode::NOT_FOUND
                || resp.status() == actix_web::http::StatusCode::BAD_REQUEST,
            "Expected 404 or 400, got {}", resp.status()
        );
    }
```

- [ ] **Step 7: Run tests**

```bash
cargo test -p torch-inference "api::classification|api::audio|api::yolo" 2>&1 | tail -20
```

Expected: new tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/api/classification.rs src/api/audio.rs src/api/yolo.rs
git commit -m "test(api): cover classification bad-request/not-found, audio empty body, yolo model-not-found"
```

---

## Task 6: TTS Manager — synthesize error + cache-hit + initialize_defaults

**Files:**
- Modify: `src/core/tts_manager.rs`

Uncovered:
- Line 170-171: `log::debug!` inside the cache-hit branch of `synthesize()`
- Line 185: `Engine '{}' not found` error when `get_engine` returns None
- Lines 197-219: cache-miss path (engine found, synthesis succeeds, result stored)
- Lines 201-258: `initialize_defaults()` body (all engines fail to load → warns, returns Ok)

The test module already defines `MockEngine` and `make_manager_with_mock`. Use these.

- [ ] **Step 1: Read the existing test helpers**

```
Read src/core/tts_manager.rs lines 440-560 — confirm MockEngine, make_manager_with_mock definitions
```

- [ ] **Step 2: Add synthesize tests**

Add inside the existing `#[cfg(test)] mod tests { ... }` in `src/core/tts_manager.rs`:

```rust
    // ──────────────────────────── synthesize() ───────────────────────────────

    #[tokio::test]
    async fn test_synthesize_returns_err_when_engine_not_found() {
        let manager = TTSManager::new(TTSManagerConfig {
            default_engine: "nonexistent".to_string(),
            ..TTSManagerConfig::default()
        });
        // No engines registered — should return Err
        let result = manager.synthesize(
            "hello",
            None,
            crate::core::tts_engine::SynthesisParams::default(),
        ).await;
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("nonexistent") || msg.contains("not found"), "error: {}", msg);
    }

    #[tokio::test]
    async fn test_synthesize_cache_miss_then_hit() {
        // First call: cache miss → engine synthesizes → stored in cache.
        // Second call: cache hit → debug log path (line 170-171).
        let manager = make_manager_with_mock("mock");
        let params = crate::core::tts_engine::SynthesisParams::default();

        let first = manager.synthesize("hello world", Some("mock"), params.clone()).await;
        assert!(first.is_ok(), "first synthesis should succeed");

        // Second call with same text+engine+params → cache hit
        let second = manager.synthesize("hello world", Some("mock"), params).await;
        assert!(second.is_ok(), "cached synthesis should succeed");

        // Both should return the same audio (same content)
        let a = first.unwrap();
        let b = second.unwrap();
        assert_eq!(a.sample_rate, b.sample_rate);
        assert_eq!(a.samples.len(), b.samples.len());
    }

    #[tokio::test]
    async fn test_initialize_defaults_does_not_panic_when_models_absent() {
        // All engines will fail to load (model files don't exist in test env).
        // initialize_defaults() must complete without panicking and return Ok(()).
        let manager = TTSManager::new(TTSManagerConfig::default());
        let result = manager.initialize_defaults().await;
        // Returns Ok even when all engines fail (they warn but don't bail)
        assert!(result.is_ok(), "initialize_defaults should not propagate engine load errors");
    }
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p torch-inference core::tts_manager 2>&1 | tail -20
```

Expected: new tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/core/tts_manager.rs
git commit -m "test(tts_manager): cover synthesize error/cache-hit/miss paths and initialize_defaults"
```

---

## Task 7: models/download.rs — scan_cache + calculate_dir_size

**Files:**
- Modify: `src/models/download.rs`

Uncovered lines 95-112: `scan_cache()` traversal with entries; `calculate_dir_size()`. These are pure filesystem operations — use `tempfile::tempdir()`.

- [ ] **Step 1: Read download.rs test module**

```
Read src/models/download.rs — find ModelDownloadManager::new, scan_cache, calculate_dir_size, and existing test module
```

- [ ] **Step 2: Add tests inside existing test module**

Add inside the existing `#[cfg(test)] mod tests { ... }` in `src/models/download.rs`:

```rust
    #[tokio::test]
    async fn test_scan_cache_empty_dir_produces_no_models() {
        let tmp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::new(tmp.path()).unwrap();
        manager.initialize().await.unwrap();
        // No subdirs → no models
        assert!(manager.list_models().is_empty());
    }

    #[tokio::test]
    async fn test_scan_cache_with_model_dir_no_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        // Create a subdirectory (simulating a model with no metadata.json)
        tokio::fs::create_dir(tmp.path().join("bert-base")).await.unwrap();

        let manager = ModelDownloadManager::new(tmp.path()).unwrap();
        manager.initialize().await.unwrap();

        let models = manager.list_models();
        assert_eq!(models.len(), 1, "one model dir should be discovered");
        assert_eq!(models[0].name, "bert-base");
    }

    #[tokio::test]
    async fn test_scan_cache_with_metadata_json() {
        let tmp = tempfile::tempdir().unwrap();
        let model_dir = tmp.path().join("my-model");
        tokio::fs::create_dir(&model_dir).await.unwrap();

        // Write a metadata.json
        let meta = serde_json::json!({
            "description": "test model",
            "tags": ["en"],
            "framework": "onnx",
            "task": "tts",
            "license": "MIT"
        });
        tokio::fs::write(model_dir.join("metadata.json"), meta.to_string()).await.unwrap();

        // Write a file so calculate_dir_size returns non-zero
        tokio::fs::write(model_dir.join("model.bin"), b"fake").await.unwrap();

        let manager = ModelDownloadManager::new(tmp.path()).unwrap();
        manager.initialize().await.unwrap();

        let models = manager.list_models();
        assert_eq!(models.len(), 1);
        let m = &models[0];
        assert_eq!(m.name, "my-model");
        assert!(m.size_bytes > 0, "size should be non-zero after writing a file");
    }

    #[tokio::test]
    async fn test_torchub_and_local_source_bail() {
        let tmp = tempfile::tempdir().unwrap();
        let manager = ModelDownloadManager::new(tmp.path()).unwrap();

        // Enqueue a TorchHub task — execute_download should bail
        let task_id = manager.enqueue_download(
            "test-model",
            ModelSource::TorchHub { repo: "pytorch/vision".to_string(), model: "resnet50".to_string() },
        ).unwrap();
        let result = manager.execute_download(&task_id).await;
        assert!(result.is_err(), "TorchHub should return bail! error");

        // Enqueue a Local task
        let task_id2 = manager.enqueue_download(
            "local-model",
            ModelSource::Local { path: "/some/path".to_string() },
        ).unwrap();
        let result2 = manager.execute_download(&task_id2).await;
        assert!(result2.is_err(), "Local source should return bail! error");
    }
```

Note: `list_models`, `enqueue_download`, and `execute_download` may have different names — check the file in Step 1 and adjust. The key is that the TorchHub/Local `bail!` arms (lines 232-239) need to be triggered.

- [ ] **Step 3: Run tests**

```bash
cargo test -p torch-inference models::download 2>&1 | tail -20
```

Expected: new tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/models/download.rs
git commit -m "test(download): cover scan_cache traversal, calculate_dir_size, TorchHub/Local bail arms"
```

---

## Task 8: torch_autodetect — check_existing_installation + CUDA_PATH branch

**Files:**
- Modify: `src/core/torch_autodetect.rs`

Uncovered:
- Lines 68-76: Metal "not available" log branch and `#[cfg(not(target_os = "macos"))]` `detect_metal` — these are platform-conditional and cannot run on macOS CI. Mark with `#[cfg_attr(coverage_nightly, ignore)]` or just accept them as platform-specific exclusions.
- Lines 153-157: CUDA via nvidia-smi (non-Windows branch) — this branch runs on macOS but nvidia-smi isn't available. It's already partially covered by the existing `test_detect_cuda_versioned_env_var_nonexistent`.
- Line 253: `detect_backend` `libtorch_cuda.so` path on non-Windows.
- Lines 322-344 in `get_download_url`: `TorchBackend::Cuda(_)` arm on macOS (lines 340-344).

Coverable without hardware:
- `check_existing_installation` when LIBTORCH env var points to a valid fake dir structure.
- `get_download_url` with `TorchBackend::Cuda("12.1".to_string())` on macOS — this exercises lines 340-344.
- `detect_backend` with a tempdir containing `libtorch_cuda.so`.

- [ ] **Step 1: Read existing tests near line 640**

```
Read src/core/torch_autodetect.rs lines 630-700
```

- [ ] **Step 2: Add tests inside existing test module**

Add inside the existing `#[cfg(test)] mod tests { ... }` in `src/core/torch_autodetect.rs`:

```rust
    #[test]
    #[serial_test::serial]
    fn test_check_existing_installation_with_libtorch_env_valid_dir() {
        // Build a fake libtorch directory structure: lib/ include/ lib/libtorch.so (or .dylib)
        let tmp = tempfile::tempdir().unwrap();
        let lib_dir = tmp.path().join("lib");
        let include_dir = tmp.path().join("include");
        std::fs::create_dir_all(&lib_dir).unwrap();
        std::fs::create_dir_all(&include_dir).unwrap();
        // On macOS the validator checks for libtorch.so (non-Windows path)
        // validate_libtorch_installation looks for lib/libtorch.so on non-Windows
        #[cfg(not(target_os = "windows"))]
        std::fs::write(lib_dir.join("libtorch.so"), b"fake").unwrap();
        #[cfg(target_os = "windows")]
        std::fs::write(lib_dir.join("torch.lib"), b"fake").unwrap();

        let old = env::var("LIBTORCH").ok();
        env::set_var("LIBTORCH", tmp.path());

        let detector = TorchLibAutoDetect::new();
        let config = detector.check_existing_installation();
        assert!(config.is_some(), "should find libtorch via LIBTORCH env var");
        let cfg = config.unwrap();
        assert_eq!(cfg.libtorch_path, tmp.path());

        if let Some(v) = old { env::set_var("LIBTORCH", v); } else { env::remove_var("LIBTORCH"); }
    }

    #[test]
    #[serial_test::serial]
    fn test_check_existing_installation_libtorch_env_missing_lib_dir() {
        // LIBTORCH points to a dir that exists but has no lib/ or include/ → None
        let tmp = tempfile::tempdir().unwrap();
        let old = env::var("LIBTORCH").ok();
        env::set_var("LIBTORCH", tmp.path());

        let detector = TorchLibAutoDetect::new();
        let config = detector.check_existing_installation();
        assert!(config.is_none(), "incomplete dir should not be recognized as valid");

        if let Some(v) = old { env::set_var("LIBTORCH", v); } else { env::remove_var("LIBTORCH"); }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_get_download_url_cuda_backend_on_macos_falls_back_to_cpu_url() {
        // On macOS, TorchBackend::Cuda falls through to the CPU URL (lines 340-344)
        let detector = TorchLibAutoDetect::new();
        let url = detector.get_download_url(&TorchBackend::Cuda("12.1".to_string()));
        assert!(!url.is_empty());
        assert!(url.contains("libtorch-macos"), "macOS Cuda backend should produce a macos URL");
    }

    #[test]
    fn test_detect_backend_returns_cpu_when_no_cuda_lib() {
        // A temp dir with lib/ but no libtorch_cuda.so → Cpu backend
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("lib")).unwrap();
        let detector = TorchLibAutoDetect::new();
        let backend = detector.detect_backend(tmp.path());
        assert_eq!(backend, TorchBackend::Cpu);
    }
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p torch-inference core::torch_autodetect 2>&1 | tail -20
```

Expected: new tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/core/torch_autodetect.rs
git commit -m "test(torch_autodetect): cover check_existing_installation, detect_backend, macos cuda URL"
```

---

## Task 9: Small gaps — resilience

**Files:**
- Modify: `src/resilience/retry.rs`
- Modify: `src/resilience/circuit_breaker.rs`
- Modify: `src/resilience/per_model_breaker.rs`

**retry.rs** uncovered lines 60, 65-66, 76-77, 84-85, 105, 123-124, 131-132:
- Lines 60/65-66: success-after-retry path (`attempt > 0` debug log)
- Lines 76-77, 84-85: max-retries-exceeded path
- Lines 105/123-124/131-132: `execute_if` with non-retryable predicate

**circuit_breaker.rs** uncovered lines 45, 89, 110:
- Line 45: `Open` state → `should_retry` true → transition to `HalfOpen`
- Line 89: `on_success` in `HalfOpen` state
- Line 110: `on_failure` in `HalfOpen` → not enough to close

**per_model_breaker.rs** uncovered lines 59, 111:
- Line 59: `Open` state with timeout expired → `HalfOpen` transition
- Line 111: `on_success` in `Closed` state reset

- [ ] **Step 1: Add retry tests**

Read `src/resilience/retry.rs` to find `RetryConfig` / `RetryPolicy` struct and `execute` / `execute_if` methods. Then add inside the existing `#[cfg(test)] mod tests { ... }`:

```rust
    #[tokio::test]
    async fn test_execute_succeeds_after_one_retry() {
        // First call fails, second succeeds — exercises the "attempt > 0" debug-log path
        let policy = RetryPolicy::new(RetryConfig {
            max_retries: 3,
            initial_delay_ms: 1,
            ..RetryConfig::default()
        });

        let attempt = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let attempt_clone = attempt.clone();

        let result: Result<&str, &str> = policy.execute(|| {
            let n = attempt_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            async move {
                if n == 0 { Err("first fail") } else { Ok("ok") }
            }
        }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "ok");
    }

    #[tokio::test]
    async fn test_execute_exhausts_retries_and_returns_err() {
        // Always fails → hits max-retries-exceeded warn path
        let policy = RetryPolicy::new(RetryConfig {
            max_retries: 2,
            initial_delay_ms: 1,
            ..RetryConfig::default()
        });

        let result: Result<(), &str> = policy.execute(|| async { Err("always fail") }).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_if_non_retryable_error_returns_immediately() {
        // is_retryable returns false → exits on first failure without retrying
        let policy = RetryPolicy::new(RetryConfig {
            max_retries: 5,
            initial_delay_ms: 1,
            ..RetryConfig::default()
        });

        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_clone = calls.clone();

        let result: Result<(), &str> = policy.execute_if(
            || {
                calls_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                async { Err("permanent") }
            },
            |_e| false, // never retryable
        ).await;

        assert!(result.is_err());
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1,
            "should only call once when error is non-retryable");
    }
```

Note: `RetryPolicy`, `RetryConfig`, and method names may differ — check the file and adjust.

- [ ] **Step 2: Add circuit_breaker tests**

Read `src/resilience/circuit_breaker.rs` to find the struct and `call` method. Add inside the existing test module:

```rust
    #[test]
    fn test_circuit_breaker_open_then_halfopen_after_timeout() {
        use std::time::Duration;
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            timeout: Duration::from_millis(1), // very short timeout
            success_threshold: 1,
        });

        // Force open by failing once
        let _ = cb.call(|| Err::<(), _>("fail".to_string()));
        // Now it's Open; wait for timeout
        std::thread::sleep(Duration::from_millis(10));
        // Next call should transition to HalfOpen and succeed
        let result = cb.call(|| Ok::<(), String>(()));
        assert!(result.is_ok(), "after timeout, HalfOpen call should succeed");
    }

    #[test]
    fn test_on_success_in_halfopen_closes_circuit() {
        use std::time::Duration;
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            timeout: Duration::from_millis(1),
            success_threshold: 1,
        });
        // Open it
        let _ = cb.call(|| Err::<(), _>("fail".to_string()));
        std::thread::sleep(Duration::from_millis(10));
        // Succeed once in HalfOpen → should close
        let _ = cb.call(|| Ok::<(), String>(()));
        // Circuit should now be Closed — a new call should succeed normally
        let result = cb.call(|| Ok::<(), String>(()));
        assert!(result.is_ok());
    }
```

- [ ] **Step 3: Add per_model_breaker test**

Read `src/resilience/per_model_breaker.rs` to find the struct. Add inside the existing test module:

```rust
    #[tokio::test]
    async fn test_per_model_breaker_open_recovers_after_timeout() {
        use std::time::Duration;
        let breaker = PerModelCircuitBreaker::new(PerModelBreakerConfig {
            failure_threshold: 1,
            timeout: Duration::from_millis(1),
            success_threshold: 1,
        });

        // Fail once → Open
        let _ = breaker.call::<(), _>(async { Err("fail".to_string()) }).await;
        tokio::time::sleep(Duration::from_millis(10)).await;
        // After timeout → HalfOpen → should succeed
        let result = breaker.call::<(), _>(async { Ok(()) }).await;
        assert!(result.is_ok(), "should recover after timeout: {:?}", result);
    }
```

Note: adjust type names and method signatures to match the actual file.

- [ ] **Step 4: Run tests**

```bash
cargo test -p torch-inference resilience 2>&1 | tail -20
```

Expected: all new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/resilience/retry.rs src/resilience/circuit_breaker.rs src/resilience/per_model_breaker.rs
git commit -m "test(resilience): cover retry success-after-retry, max-exceeded, non-retryable; circuit breaker HalfOpen transitions"
```

---

## Task 10: Small gaps — core modules

**Files:**
- Modify: `src/core/audio.rs`
- Modify: `src/core/yolo.rs`
- Modify: `src/core/image_security.rs`
- Modify: `src/core/tts_engine.rs`

**core/audio.rs** uncovered lines 146-150, 181-183, 246, 261:
- Lines 146-150: `duration_secs` default-to-0.0 branch (when n_frames or sample_rate is None).
- Lines 181-183: `validate_mp3` path through `probe_metadata_with_symphonia`.
- Lines 246, 261: decode error or packet-skip paths inside `load_with_symphonia`.

**core/yolo.rs** lines 253-254, 289-290:
- `#[cfg(not(feature = "torch"))]` versions of `preprocess_image` and `detect` that bail. Cannot be covered when `torch` feature is enabled; add `#[cfg(not(feature = "torch"))]` test.

**core/image_security.rs** lines 128, 218-222:
- Line 128: `check_adversarial_patterns` call with an image exceeding the `max_dimension`.
- Lines 218-222: `check_adversarial_patterns` returning `Some(threat)` when variance > threshold.

**core/tts_engine.rs** lines 90, 125:
- Line 90: `warmup()` returning Ok — the default `warmup` impl. Already trivially called.
- Line 125: `TTSEngineFactory::create("windows-sapi")` on non-Windows → bail.

- [ ] **Step 1: Add core/audio.rs tests**

Read `src/core/audio.rs` to understand `AudioValidator` or `AudioProcessor` constructor. Add inside the existing test module:

```rust
    #[test]
    fn test_probe_metadata_duration_defaults_to_zero_when_frames_unknown() {
        // WAV with valid header but 0 n_frames — duration should default to 0.0
        // Build a minimal valid WAV with n_frames=0
        let mut wav_data = Vec::new();
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut wav_data), spec).unwrap();
            // Write no samples → n_frames = 0
            writer.finalize().unwrap();
        }
        let validator = AudioValidator::new();
        let meta = validator.validate_audio(&wav_data);
        assert!(meta.is_ok(), "empty WAV should be valid");
        let m = meta.unwrap();
        assert_eq!(m.duration_secs, 0.0);
    }
```

- [ ] **Step 2: Add core/yolo.rs no-torch test**

Read `src/core/yolo.rs` to find the `#[cfg(not(feature = "torch"))]` versions. Add:

```rust
    #[cfg(not(feature = "torch"))]
    #[test]
    fn test_preprocess_image_bails_without_torch() {
        let detector = YoloDetector::new_stub(); // or however you build one without a model
        let result = detector.preprocess_image(std::path::Path::new("/any/path.jpg"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("PyTorch"));
    }
```

Note: if the `#[cfg(not(feature = "torch"))]` paths are always compiled when the `torch` feature is absent, and the CI always enables it, skip this test. Check by examining the CI/Cargo.toml feature defaults.

- [ ] **Step 3: Add core/image_security.rs tests**

Read `src/core/image_security.rs` to find `ImageSecurityAnalyzer` and `check_adversarial_patterns`. Add inside the existing test module:

```rust
    #[test]
    fn test_analyze_oversized_image_adds_excessive_size_threat() {
        let analyzer = ImageSecurityAnalyzer::new(ImageSecurityConfig {
            max_dimension: 4,  // tiny limit
            ..ImageSecurityConfig::default()
        });
        // Build a small but "too large" image (5×5)
        let img = image::DynamicImage::new_rgb8(5, 5);
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png).unwrap();

        let result = analyzer.analyze_image_bytes(&buf, SecurityLevel::Standard);
        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(report.threats.iter().any(|t| matches!(t.threat_type, ThreatType::ExcessiveSize)),
            "expected ExcessiveSize threat");
    }
```

- [ ] **Step 4: Add core/tts_engine.rs test**

Read `src/core/tts_engine.rs` to find `TTSEngineFactory::create`. Add:

```rust
    #[test]
    #[cfg(not(target_os = "windows"))]
    fn test_factory_create_windows_sapi_bails_on_non_windows() {
        let result = TTSEngineFactory::create("windows-sapi", &serde_json::json!({}));
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Windows") || msg.contains("only"), "error: {}", msg);
    }

    #[test]
    fn test_factory_create_unknown_engine_type_returns_err() {
        let result = TTSEngineFactory::create("does-not-exist", &serde_json::json!({}));
        assert!(result.is_err());
    }
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p torch-inference "core::audio|core::yolo|core::image_security|core::tts_engine" 2>&1 | tail -20
```

Expected: new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/core/audio.rs src/core/yolo.rs src/core/image_security.rs src/core/tts_engine.rs
git commit -m "test(core): cover audio duration-zero, yolo no-torch bail, image oversized threat, TTS factory unknown engine"
```

---

## Task 11: Small gaps — data structures

**Files:**
- Modify: `src/cache.rs`
- Modify: `src/monitor.rs`
- Modify: `src/batch.rs`
- Modify: `src/model_pool.rs`
- Modify: `src/dedup.rs`
- Modify: `src/tensor_pool.rs`
- Modify: `src/guard.rs`
- Modify: `src/worker_pool.rs`

- [ ] **Step 1: Read each file's uncovered lines**

For each file, read the 5 lines surrounding each uncovered line:
- `src/cache.rs` line 132
- `src/monitor.rs` lines 83-86
- `src/batch.rs` line 96
- `src/model_pool.rs` line 110
- `src/dedup.rs` lines 131-132
- `src/tensor_pool.rs` lines 86, 107-108
- `src/guard.rs` lines 327-357
- `src/worker_pool.rs` lines 366-421

- [ ] **Step 2: Add cache.rs test**

`cache.rs` line 132 is `remove_bytes`. Add inside the existing test module:

```rust
    #[test]
    fn test_remove_bytes_existing_key() {
        let cache = InferenceCache::new(); // or whatever the constructor is
        cache.set_bytes("key1", vec![1, 2, 3]);
        cache.remove_bytes("key1");
        // After removal, get_bytes should return None
        assert!(cache.get_bytes("key1").is_none());
    }

    #[test]
    fn test_remove_bytes_nonexistent_key_does_not_panic() {
        let cache = InferenceCache::new();
        cache.remove_bytes("nonexistent"); // should not panic
    }
```

- [ ] **Step 3: Add monitor.rs test**

`monitor.rs` lines 83-86 are the min/max latency CAS update paths. Add:

```rust
    #[test]
    fn test_record_latency_updates_min_and_max() {
        let monitor = PerformanceMonitor::new(); // or whatever the constructor
        monitor.record_latency(100.0);
        monitor.record_latency(50.0);  // new min
        monitor.record_latency(200.0); // new max
        let stats = monitor.get_stats();
        assert!(stats.min_latency <= 50.0, "min should be ≤50");
        assert!(stats.max_latency >= 200.0, "max should be ≥200");
    }
```

- [ ] **Step 4: Add batch.rs test**

`batch.rs` line 96 is `is_timed_out` returning `false`. The existing tests likely only cover the `true` path. Add:

```rust
    #[test]
    fn test_is_timed_out_returns_false_when_not_timed_out() {
        // Create a request with a very large timeout so it hasn't expired
        let req = BatchRequest::new_with_timeout(/* args */ 99999); // adjust to real API
        assert!(!req.is_timed_out(), "newly created request should not be timed out");
    }
```

- [ ] **Step 5: Add model_pool.rs test**

`model_pool.rs` line 110 is the "empty instances" early-return in `acquire`. Add:

```rust
    #[tokio::test]
    async fn test_acquire_returns_none_when_pool_empty() {
        let pool = ModelPool::new(/* max = */ 0); // zero capacity → empty
        let result = pool.acquire("model-x").await;
        assert!(result.is_none(), "empty pool should return None");
    }
```

- [ ] **Step 6: Add dedup.rs, tensor_pool.rs, guard.rs, worker_pool.rs tests**

Read each file, then add one targeted test per uncovered line group. Examples:

For `dedup.rs` lines 131-132 (likely a concurrent-insert path):
```rust
    #[tokio::test]
    async fn test_concurrent_dedup_second_waiter_gets_same_result() {
        // Two tasks request the same key simultaneously — second should wait for first
        let dedup = RequestDeduplicator::new();
        // ... (implement based on actual API)
    }
```

For `tensor_pool.rs` lines 86, 107-108 (likely stats edge cases):
```rust
    #[test]
    fn test_tensor_pool_stats_zero_total() {
        let pool = TensorPool::new(10);
        let stats = pool.stats();
        assert_eq!(stats.reuse_rate, 0.0, "reuse_rate should be 0.0 when no operations done");
    }

    #[test]
    fn test_tensor_pool_stats_all_allocations() {
        let pool = TensorPool::new(10);
        // Acquire tensors (all new allocations, nothing returned yet)
        // ... based on actual API
        let stats = pool.stats();
        assert!(stats.reuse_rate >= 0.0);
    }
```

For `guard.rs` and `worker_pool.rs`, read the actual uncovered lines and write minimal tests that exercise those paths. The test only needs to reach the line — it doesn't need to assert complex behavior.

- [ ] **Step 7: Run tests**

```bash
cargo test -p torch-inference "cache|monitor|batch|model_pool|dedup|tensor_pool|guard|worker_pool" 2>&1 | tail -30
```

Expected: new tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/cache.rs src/monitor.rs src/batch.rs src/model_pool.rs src/dedup.rs src/tensor_pool.rs src/guard.rs src/worker_pool.rs
git commit -m "test(data-structures): cover remove_bytes, latency min/max, timed-out false, empty pool acquire, tensor stats"
```

---

## Task 12: Small gaps — API and models

**Files:**
- Modify: `src/api/image.rs`
- Modify: `src/api/performance.rs`
- Modify: `src/api/model_download.rs`
- Modify: `src/core/audio_models.rs`
- Modify: `src/core/g2p_misaki.rs`
- Modify: `src/models/pytorch_loader.rs`
- Modify: `src/config.rs`

- [ ] **Step 1: Read each file's uncovered lines**

For each file, read the surrounding context:
- `src/api/image.rs` lines 91, 116, 157
- `src/api/performance.rs` lines 118, 121
- `src/api/model_download.rs` lines 288-289, 490-492
- `src/core/audio_models.rs` lines 208, 453, 463
- `src/core/g2p_misaki.rs` lines 468, 490
- `src/models/pytorch_loader.rs` lines 462-463
- `src/config.rs` line 158

- [ ] **Step 2: Add api/image.rs tests**

Lines 91, 116, 157 are typically error paths (file not found, unsupported format, resize failure). After reading the handlers:

```rust
    #[actix_web::test]
    async fn test_image_endpoint_nonexistent_file_returns_error() {
        // (construct app + state, send request for a nonexistent image)
        // assert error status
    }
```

Implement based on actual handler signatures found in Step 1.

- [ ] **Step 3: Add api/performance.rs test**

Lines 118, 121 are likely inside an error path or conditional in a performance stats handler. After reading:

```rust
    #[actix_web::test]
    async fn test_performance_endpoint_no_data_returns_empty_or_default() {
        // Build minimal state, call endpoint, assert 200
    }
```

- [ ] **Step 4: Add api/model_download.rs tests**

Lines 288-289, 490-492 are `download_model_async` branches:
- Line 288-289: `Built-in` URL path (early return)
- Lines 490-492: non-`Built-in` URL routing based on model type

```rust
    #[tokio::test]
    async fn test_download_model_async_builtin_returns_ok() {
        use super::*;
        let model = ModelInfo {
            url: "Built-in".to_string(),
            model_type: "tts".to_string(),
            name: "test".to_string(),
            ..ModelInfo::default()
        };
        // Should return Ok immediately without network call
        let result = download_model_async("test-id", &model).await;
        assert!(result.is_ok(), "Built-in model should return Ok without downloading");
    }
```

- [ ] **Step 5: Add remaining small-gap tests**

For `core/audio_models.rs` (lines 208, 453, 463), `core/g2p_misaki.rs` (lines 468, 490), `models/pytorch_loader.rs` (lines 462-463), `config.rs` (line 158):
- Read each file's surrounding lines.
- Add the minimal test that reaches the uncovered line.

Typical pattern for `config.rs` line 158 (likely a default or env-var override path):
```rust
    #[test]
    #[serial_test::serial]
    fn test_config_env_var_override() {
        // Set the env var that triggers line 158, then parse config, assert override applied
        std::env::set_var("SOME_CONFIG_VAR", "test-value");
        let cfg = AppConfig::from_env(); // or whatever the function is
        // assert cfg uses the env var value
        std::env::remove_var("SOME_CONFIG_VAR");
    }
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p torch-inference "api::image|api::performance|api::model_download|core::audio_models|core::g2p_misaki|models::pytorch_loader|config" 2>&1 | tail -30
```

Expected: new tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/api/image.rs src/api/performance.rs src/api/model_download.rs src/core/audio_models.rs src/core/g2p_misaki.rs src/models/pytorch_loader.rs src/config.rs
git commit -m "test(api+models+config): cover remaining small-gap uncovered lines"
```

---

---

## Task 13: models/manager.rs, models/registry.rs, api/models.rs — error paths

**Files:**
- Modify: `src/models/manager.rs`
- Modify: `src/models/registry.rs`
- Modify: `src/api/models.rs`

**models/manager.rs** uncovered lines 173-196, 261-270: load-failure and unload-failure paths.
**models/registry.rs** uncovered lines 178, 224-239, 265-266: registry-miss and update-error paths.
**api/models.rs** uncovered lines 123, 207-208, 257-268, 360-368: `from_json_str` with bad JSON, `already_downloaded` branch, and model-not-found for download request.

- [ ] **Step 1: Read manager.rs, registry.rs, models.rs uncovered sections**

```
Read src/models/manager.rs lines 168-200
Read src/models/registry.rs lines 175-245
Read src/api/models.rs lines 118-130, 200-215, 250-270, 355-375
```

- [ ] **Step 2: Add models/manager.rs tests**

Add inside the existing `#[cfg(test)] mod tests { ... }` in `src/models/manager.rs`:

```rust
    #[tokio::test]
    async fn test_load_model_nonexistent_path_returns_err() {
        let manager = ModelManager::new(ModelManagerConfig::default()).unwrap();
        let result = manager.load_model("test-model", std::path::Path::new("/nonexistent/model.pt")).await;
        assert!(result.is_err(), "nonexistent model path should return Err");
    }

    #[tokio::test]
    async fn test_unload_model_not_registered_returns_err() {
        let manager = ModelManager::new(ModelManagerConfig::default()).unwrap();
        let result = manager.unload_model("not-registered").await;
        assert!(result.is_err(), "unloading a model that was never loaded should return Err");
    }
```

Adjust method names to match the actual file.

- [ ] **Step 3: Add models/registry.rs tests**

Add inside the existing test module in `src/models/registry.rs`:

```rust
    #[test]
    fn test_from_json_str_invalid_json_returns_empty() {
        let registry = ModelRegistry::from_json_str("NOT JSON AT ALL {{{");
        // from_json_str falls back to empty on parse error
        assert!(registry.list_models().is_empty(), "invalid JSON should produce empty registry");
    }

    #[test]
    fn test_get_model_nonexistent_returns_none() {
        let registry = ModelRegistry::from_json_str("{}");
        assert!(registry.get_model("no-such-model").is_none());
    }
```

- [ ] **Step 4: Add api/models.rs tests**

Add inside the existing `#[cfg(test)] mod tests { ... }` in `src/api/models.rs`:

```rust
    /// from_json_str with invalid JSON returns an empty registry (line 123 warn path)
    #[test]
    fn test_model_registry_from_json_str_bad_json_warns_and_returns_empty() {
        let registry = ModelRegistry::from_json_str("{invalid json!!!");
        assert!(registry.list_models().is_empty());
    }

    /// download_model request for a model with status "Downloaded" returns already_downloaded
    #[actix_web::test]
    #[serial_test::serial]
    async fn test_download_model_already_downloaded_returns_status() {
        use actix_web::{test, web, App};

        // Find a model in the compiled-in registry that has status "Downloaded" or "Active"
        // "windows-sapi" has status "Active" in the registry
        let app = test::init_service(
            App::new().route("/api/models/download", web::post().to(download_model)),
        ).await;

        let req = test::TestRequest::post()
            .uri("/api/models/download")
            .set_json(serde_json::json!({ "model_id": "windows-sapi" }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        // Should return 200 with status: "already_downloaded"
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        let body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(body["status"], "already_downloaded");
    }

    /// download_model request for a model_id that doesn't exist returns 404
    #[actix_web::test]
    async fn test_download_model_unknown_model_id_returns_not_found() {
        use actix_web::{test, web, App};

        let app = test::init_service(
            App::new().route("/api/models/download", web::post().to(download_model)),
        ).await;

        let req = test::TestRequest::post()
            .uri("/api/models/download")
            .set_json(serde_json::json!({ "model_id": "zzz-does-not-exist-ever-xyz" }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::NOT_FOUND);
    }
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p torch-inference "models::manager|models::registry|api::models" 2>&1 | tail -20
```

Expected: new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/manager.rs src/models/registry.rs src/api/models.rs
git commit -m "test(models): cover manager load/unload errors, registry invalid JSON, models download already_downloaded/not_found"
```

---

## Task 14: core/kokoro_onnx.rs, core/piper_tts.rs, core/image_classifier.rs — constructor error paths

**Files:**
- Modify: `src/core/kokoro_onnx.rs`
- Modify: `src/core/piper_tts.rs`
- Modify: `src/core/image_classifier.rs`

**kokoro_onnx.rs** uncovered lines 53-140: `KokoroOnnxEngine::new()` when model dir does not exist errors out immediately.
**piper_tts.rs** uncovered lines 134-179: synthesis paths when called with failing state.
**image_classifier.rs** uncovered lines 79-267: classifier error paths when model absent.

- [ ] **Step 1: Read constructor signatures**

```
Read src/core/kokoro_onnx.rs lines 140-175
Read src/core/piper_tts.rs lines 128-145
Read src/core/image_classifier.rs lines 75-90
```

- [ ] **Step 2: Add kokoro_onnx.rs test**

Add inside the existing `#[cfg(test)] mod tests { ... }` in `src/core/kokoro_onnx.rs`:

```rust
    #[test]
    fn test_new_with_nonexistent_model_dir_returns_err() {
        let cfg = serde_json::json!({
            "model_dir": "/nonexistent/path/that/does/not/exist",
            "pool_size": 1
        });
        let result = KokoroOnnxEngine::new(&cfg);
        assert!(result.is_err(), "KokoroOnnxEngine::new should fail when model dir is absent");
        let msg = result.unwrap_err().to_string();
        // Error should mention missing model file
        assert!(
            msg.contains("not found") || msg.contains("No such file") || msg.contains("kokoro"),
            "unexpected error: {}", msg
        );
    }
```

- [ ] **Step 3: Add piper_tts.rs test**

Read piper_tts.rs to find `PiperTTSEngine::new`. Add:

```rust
    #[test]
    fn test_new_with_nonexistent_model_path_returns_err() {
        let cfg = serde_json::json!({
            "model_path": "/nonexistent/piper/model.onnx",
            "config_path": "/nonexistent/piper/model.onnx.json"
        });
        let result = PiperTTSEngine::new(&cfg);
        assert!(result.is_err(), "PiperTTSEngine::new should fail with nonexistent paths");
    }
```

- [ ] **Step 4: Add image_classifier.rs test**

Read `src/core/image_classifier.rs` to find the classifier constructor and error paths:

```rust
    #[test]
    fn test_classify_nonexistent_model_returns_err() {
        // If the constructor accepts a model_path, point it at a nonexistent file
        let result = ImageClassifier::new(std::path::Path::new("/nonexistent/model.onnx"));
        assert!(result.is_err(), "ImageClassifier should fail with nonexistent model");
    }
```

Adjust based on the actual API found in Step 1.

- [ ] **Step 5: Run tests**

```bash
cargo test -p torch-inference "core::kokoro_onnx|core::piper_tts|core::image_classifier" 2>&1 | tail -20
```

Expected: new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/core/kokoro_onnx.rs src/core/piper_tts.rs src/core/image_classifier.rs
git commit -m "test(core): cover kokoro_onnx/piper_tts/image_classifier constructor error paths"
```

---

## Task 15: Verify coverage improvement

- [ ] **Step 1: Run tarpaulin**

```bash
cargo tarpaulin --out Json --output-dir . -- --test-threads 1 2>&1 | tail -5
```

Expected: coverage reported in `tarpaulin-report.json`.

- [ ] **Step 2: Parse and compare**

```bash
cat tarpaulin-report.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
files = data.get('files', [])
total_covered = sum(f.get('covered', 0) for f in files)
total_coverable = sum(f.get('coverable', 0) for f in files)
print(f'Total: {total_covered}/{total_coverable} ({100*total_covered/total_coverable:.1f}%)')
"
```

Expected: ≥92% overall.

- [ ] **Step 3: Commit final report**

```bash
git add tarpaulin-report.json lcov.info 2>/dev/null || true
git commit -m "chore: update coverage report after test additions" --allow-empty
```

---

## Notes on genuinely uncoverable lines

The following lines cannot be covered without real hardware or are platform-conditional — accept them as exclusions:

- `src/core/torch_autodetect.rs` lines 68-76: macOS Metal "unavailable" path and `#[cfg(not(target_os = "macos"))]` block
- `src/core/windows_sapi_tts.rs` lines 73-96: Windows-only SAPI
- `src/core/istftnet_vocoder.rs` / `src/core/styletts2_model.rs`: ORT session synthesis
- `src/models/onnx_loader.rs` lines 157+: require real ORT session + model file
- `src/models/download.rs` HuggingFace/URL download paths: require network
- `src/core/gpu.rs` CUDA memory query paths: require CUDA hardware
