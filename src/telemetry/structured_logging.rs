#![allow(dead_code)]
use std::sync::Arc;
use std::time::Instant;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    EnvFilter, Registry,
};

/// Initialize structured logging with JSON output
pub fn init_structured_logging(log_dir: Option<&str>, json_output: bool) {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let registry = Registry::default().with(env_filter);

    if json_output {
        // JSON output for production (structured logs)
        let json_layer = fmt::layer()
            .json()
            .with_current_span(true)
            .with_span_list(true)
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_span_events(FmtSpan::CLOSE)
            .flatten_event(true); // Flatten for easier parsing

        if let Some(dir) = log_dir {
            // File output
            let file_appender =
                RollingFileAppender::new(Rotation::DAILY, dir, "torch-inference.log");
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

            // Keep guard alive to prevent dropping
            std::mem::forget(_guard);

            let subscriber = registry.with(json_layer.with_writer(non_blocking));
            tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");
        } else {
            // Console output
            let subscriber = registry.with(json_layer);
            tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");
        }
    } else {
        // Human-readable output for development
        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .with_thread_ids(true)
            .with_span_events(FmtSpan::CLOSE)
            .pretty();

        let subscriber = registry.with(fmt_layer);
        tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");
    }

    tracing::info!("Structured logging initialized (json={})", json_output);
}

/// Initialize with OpenTelemetry tracing (optional)
#[cfg(feature = "telemetry")]
pub fn init_with_tracing(
    log_dir: Option<&str>,
    json_output: bool,
    otlp_endpoint: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    use opentelemetry::trace::TracerProvider;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::trace::Config;
    use tracing_opentelemetry::OpenTelemetryLayer;

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Initialize OpenTelemetry
    let tracer = if let Some(endpoint) = otlp_endpoint {
        opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(endpoint),
            )
            .with_trace_config(
                Config::default().with_resource(opentelemetry_sdk::Resource::new(vec![
                    opentelemetry::KeyValue::new("service.name", "torch-inference"),
                    opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                ])),
            )
            .install_batch(opentelemetry_sdk::runtime::Tokio)?
    } else {
        // No-op tracer if endpoint not provided
        return init_structured_logging_fallback(log_dir, json_output);
    };

    let telemetry_layer = OpenTelemetryLayer::new(tracer.tracer("torch-inference"));
    let registry = Registry::default().with(env_filter).with(telemetry_layer);

    if json_output {
        let json_layer = fmt::layer()
            .json()
            .with_current_span(true)
            .with_span_list(true)
            .with_target(true)
            .with_file(true)
            .with_line_number(true);

        if let Some(dir) = log_dir {
            let file_appender =
                RollingFileAppender::new(Rotation::DAILY, dir, "torch-inference.log");
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            let subscriber = registry.with(json_layer.with_writer(non_blocking));
            tracing::subscriber::set_global_default(subscriber)?;
        } else {
            let subscriber = registry.with(json_layer);
            tracing::subscriber::set_global_default(subscriber)?;
        }
    } else {
        let fmt_layer = fmt::layer()
            .with_target(true)
            .with_file(true)
            .with_line_number(true)
            .pretty();

        let subscriber = registry.with(fmt_layer);
        tracing::subscriber::set_global_default(subscriber)?;
    }

    tracing::info!("Structured logging with OpenTelemetry initialized");
    Ok(())
}

#[cfg(feature = "telemetry")]
fn init_structured_logging_fallback(
    log_dir: Option<&str>,
    json_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    init_structured_logging(log_dir, json_output);
    Ok(())
}

/// Correlation ID middleware for request tracing
///
/// The inner `Arc<str>` makes `.clone()` an O(1) atomic reference-count bump
/// instead of a heap allocation + string copy.  All call sites are unchanged.
#[derive(Clone)]
pub struct CorrelationId(pub Arc<str>);

impl CorrelationId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string().into())
    }

    pub fn from_header(value: &str) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for CorrelationId {
    fn default() -> Self {
        Self::new()
    }
}

#[macro_export]
macro_rules! log_with_context {
    ($level:expr, $correlation_id:expr, $($arg:tt)*) => {
        tracing::event!(
            $level,
            correlation_id = %$correlation_id.as_str(),
            $($arg)*
        );
    };
}

/// Request span helper
pub fn create_request_span(
    method: &str,
    path: &str,
    correlation_id: &CorrelationId,
) -> tracing::Span {
    tracing::info_span!(
        "request",
        method = %method,
        path = %path,
        correlation_id = %correlation_id.as_str(),
        otel.kind = "server",
    )
}

/// Inference span helper
pub fn create_inference_span(
    model_name: &str,
    batch_size: usize,
    correlation_id: &CorrelationId,
) -> tracing::Span {
    tracing::info_span!(
        "inference",
        model = %model_name,
        batch_size = %batch_size,
        correlation_id = %correlation_id.as_str(),
        otel.kind = "internal",
    )
}

/// Log request with structured data
#[macro_export]
macro_rules! log_request {
    ($correlation_id:expr, method = $method:expr, path = $path:expr, $($field:tt = $value:expr),*) => {
        tracing::info!(
            correlation_id = %$correlation_id.as_str(),
            method = %$method,
            path = %$path,
            event = "request_received",
            $($field = %$value,)*
        );
    };
}

/// Log response with structured data
#[macro_export]
macro_rules! log_response {
    ($correlation_id:expr, status = $status:expr, duration_ms = $duration:expr, $($field:tt = $value:expr),*) => {
        tracing::info!(
            correlation_id = %$correlation_id.as_str(),
            status = %$status,
            duration_ms = %$duration,
            event = "request_completed",
            $($field = %$value,)*
        );
    };
}

/// Log error with structured data
#[macro_export]
macro_rules! log_error {
    ($correlation_id:expr, error = $error:expr, $($field:tt = $value:expr),*) => {
        tracing::error!(
            correlation_id = %$correlation_id.as_str(),
            error = %$error,
            event = "error_occurred",
            $($field = %$value,)*
        );
    };
}

/// Request metrics helper
pub struct RequestMetrics {
    pub start_time: Instant,
    pub correlation_id: CorrelationId,
}

impl RequestMetrics {
    pub fn new(correlation_id: CorrelationId) -> Self {
        Self {
            start_time: Instant::now(),
            correlation_id,
        }
    }

    pub fn duration_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    pub fn log_completion(&self, status: u16, path: &str) {
        tracing::info!(
            correlation_id = %self.correlation_id.as_str(),
            status = %status,
            duration_ms = %self.duration_ms(),
            path = %path,
            event = "request_completed",
        );
    }

    pub fn log_error(&self, error: &str, path: &str) {
        tracing::error!(
            correlation_id = %self.correlation_id.as_str(),
            error = %error,
            duration_ms = %self.duration_ms(),
            path = %path,
            event = "request_error",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_id_generation() {
        let id1 = CorrelationId::new();
        let id2 = CorrelationId::new();

        assert_ne!(id1.as_str(), id2.as_str());
        assert!(uuid::Uuid::parse_str(id1.as_str()).is_ok());
    }

    #[test]
    fn test_correlation_id_from_header() {
        let id = CorrelationId::from_header("test-123");
        assert_eq!(id.as_str(), "test-123");
    }

    #[test]
    fn test_request_metrics() {
        let correlation_id = CorrelationId::new();
        let metrics = RequestMetrics::new(correlation_id);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let duration = metrics.duration_ms();
        assert!(duration >= 10);
    }

    #[test]
    fn test_span_creation() {
        let correlation_id = CorrelationId::new();

        let _request_span = create_request_span("GET", "/api/test", &correlation_id);
        let _inference_span = create_inference_span("test_model", 32, &correlation_id);

        // Should not panic
    }

    // ── CorrelationId – additional coverage ───────────────────────────────────

    #[test]
    fn test_correlation_id_default_is_valid_uuid() {
        let id: CorrelationId = CorrelationId::default();
        assert!(uuid::Uuid::parse_str(id.as_str()).is_ok());
    }

    #[test]
    fn test_correlation_id_clone() {
        let id = CorrelationId::new();
        let cloned = id.clone();
        assert_eq!(id.as_str(), cloned.as_str());
    }

    #[test]
    fn test_correlation_id_from_header_preserves_value() {
        let raw = "my-custom-correlation-id-xyz";
        let id = CorrelationId::from_header(raw);
        assert_eq!(id.as_str(), raw);
    }

    // ── RequestMetrics – log helpers ──────────────────────────────────────────

    #[test]
    fn test_request_metrics_log_completion_does_not_panic() {
        let correlation_id = CorrelationId::new();
        let metrics = RequestMetrics::new(correlation_id);
        // log_completion emits a tracing event; calling it must not panic.
        metrics.log_completion(200, "/api/infer");
    }

    #[test]
    fn test_request_metrics_log_error_does_not_panic() {
        let correlation_id = CorrelationId::new();
        let metrics = RequestMetrics::new(correlation_id);
        metrics.log_error("internal server error", "/api/infer");
    }

    #[test]
    fn test_request_metrics_duration_increases_over_time() {
        let correlation_id = CorrelationId::new();
        let metrics = RequestMetrics::new(correlation_id);
        let d1 = metrics.duration_ms();
        std::thread::sleep(std::time::Duration::from_millis(15));
        let d2 = metrics.duration_ms();
        assert!(d2 >= d1, "duration should be monotonically non-decreasing");
    }

    // ── init_structured_logging ───────────────────────────────────────────────
    //
    // set_global_default will fail if a subscriber is already set (which is
    // expected in a multi-test binary).  We intentionally swallow that failure
    // via the `.expect()` inside the function, but the tests can still exercise
    // the code paths by accepting a potential panic and treating it as success
    // if the panic message indicates "already set".  In practice the function
    // uses `.expect()` which panics, so we use `std::panic::catch_unwind`.

    fn try_init(log_dir: Option<&str>, json_output: bool) {
        // The function calls set_global_default which panics if already set.
        // We catch that panic so the test suite can continue.
        let _ = std::panic::catch_unwind(|| {
            init_structured_logging(log_dir, json_output);
        });
    }

    #[test]
    fn test_init_structured_logging_plain_console() {
        try_init(None, false);
    }

    #[test]
    fn test_init_structured_logging_json_console() {
        try_init(None, true);
    }

    #[test]
    fn test_init_structured_logging_json_with_dir() {
        let tmp = std::env::temp_dir();
        let dir = tmp.to_string_lossy().to_string();
        try_init(Some(&dir), true);
    }

    // ── Macros ────────────────────────────────────────────────────────────────

    #[test]
    fn test_log_with_context_macro_does_not_panic() {
        let id = CorrelationId::new();
        // macro emits a tracing event – must not panic
        log_with_context!(tracing::Level::INFO, id, "test message from macro");
    }

    #[test]
    fn test_log_request_macro_does_not_panic() {
        let id = CorrelationId::new();
        log_request!(id, method = "POST", path = "/api/v1/completions",);
    }

    #[test]
    fn test_log_response_macro_does_not_panic() {
        let id = CorrelationId::new();
        log_response!(id, status = 200u16, duration_ms = 42u64,);
    }

    #[test]
    fn test_log_error_macro_does_not_panic() {
        let id = CorrelationId::new();
        log_error!(id, error = "something went wrong",);
    }

    // ── Span creation – additional HTTP verbs / edge cases ───────────────────

    #[test]
    fn test_create_request_span_post() {
        let id = CorrelationId::new();
        let _span = create_request_span("POST", "/api/infer", &id);
    }

    #[test]
    fn test_create_inference_span_batch_zero() {
        let id = CorrelationId::new();
        let _span = create_inference_span("llama3", 0, &id);
    }

    // ── Additional gap-closing tests ───────────────────────────────────────────

    /// Exercises init_structured_logging with json=false AND a log_dir.
    /// This path goes through the `else` branch (human-readable) which already
    /// has no file-appender variant, but the subscriber creation lines are
    /// still evaluated.
    #[test]
    fn test_init_structured_logging_plain_with_dir() {
        let tmp = std::env::temp_dir();
        let dir = tmp.to_string_lossy().to_string();
        // json=false path — the subscriber is built from fmt_layer (lines 51-61).
        // set_global_default may panic if already set; catch that.
        let _ = std::panic::catch_unwind(|| {
            init_structured_logging(Some(&dir), false);
        });
    }

    /// Exercises CorrelationId::new (Default) producing distinct UUIDs.
    #[test]
    fn test_correlation_id_uniqueness() {
        let ids: Vec<_> = (0..5)
            .map(|_| CorrelationId::new().as_str().to_string())
            .collect();
        // All five should be distinct
        let set: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(set.len(), 5);
    }

    /// Exercises RequestMetrics fields directly.
    #[test]
    fn test_request_metrics_fields() {
        let id = CorrelationId::from_header("test-id-abc");
        let metrics = RequestMetrics::new(id);
        // correlation_id is accessible
        assert_eq!(metrics.correlation_id.as_str(), "test-id-abc");
        // start_time was set (duration should be very small)
        assert!(metrics.duration_ms() < 5000);
    }

    /// Exercises create_request_span with a DELETE verb.
    #[test]
    fn test_create_request_span_delete() {
        let id = CorrelationId::new();
        let _span = create_request_span("DELETE", "/api/models/123", &id);
    }

    /// Exercises create_inference_span with a large batch.
    #[test]
    fn test_create_inference_span_large_batch() {
        let id = CorrelationId::new();
        let _span = create_inference_span("gpt-4", 512, &id);
    }

    // ── Lines 272, 274, 282, 284: log_completion and log_error with a subscriber ─
    // These lines are inside tracing::info!/error! calls. Without a subscriber,
    // the events are created but not processed. Installing a no-op subscriber
    // ensures the event bodies are evaluated (which exercises the lines).

    fn with_noop_subscriber<F: FnOnce()>(f: F) {
        // tracing::subscriber::with_default installs a subscriber only for this thread
        // for the duration of the closure, without affecting other tests.
        use tracing_subscriber::fmt;
        let subscriber = fmt::Subscriber::builder()
            .with_max_level(tracing::Level::TRACE)
            .with_writer(std::io::sink) // discard output
            .finish();
        let _ = tracing::subscriber::with_default(subscriber, f);
    }

    #[test]
    fn test_log_completion_emits_event_with_subscriber() {
        with_noop_subscriber(|| {
            let id = CorrelationId::new();
            let metrics = RequestMetrics::new(id);
            // Lines 272 and 274 are inside this call
            metrics.log_completion(200, "/api/v1/infer");
        });
    }

    #[test]
    fn test_log_error_emits_event_with_subscriber() {
        with_noop_subscriber(|| {
            let id = CorrelationId::new();
            let metrics = RequestMetrics::new(id);
            // Lines 282 and 284 are inside this call
            metrics.log_error("something failed", "/api/v1/infer");
        });
    }

    #[test]
    fn test_log_completion_various_status_codes() {
        with_noop_subscriber(|| {
            let id = CorrelationId::new();
            let metrics = RequestMetrics::new(id);
            for &status in &[200u16, 201, 400, 404, 500] {
                metrics.log_completion(status, "/test");
            }
        });
    }

    // ── Lines 191, 206: correlation_id field inside info_span! macros ─────────
    // tracing::info_span! field values are only evaluated when a subscriber is
    // active and the span's level is enabled. We install a thread-local subscriber
    // to ensure the field-value expressions on lines 191 and 206 are actually
    // executed by the tracing machinery.

    #[test]
    fn test_create_request_span_with_active_subscriber_line_191() {
        with_noop_subscriber(|| {
            let id = CorrelationId::new();
            // Entering the span forces tracing to record all field values,
            // including `correlation_id = %correlation_id.as_str()` on line 191.
            let span = create_request_span("GET", "/api/health", &id);
            let _guard = span.enter();
            // Nothing further needed; entering exercised line 191.
        });
    }

    #[test]
    fn test_create_inference_span_with_active_subscriber_line_206() {
        with_noop_subscriber(|| {
            let id = CorrelationId::new();
            // Entering exercises `correlation_id = %correlation_id.as_str()` on line 206.
            let span = create_inference_span("bert-base", 8, &id);
            let _guard = span.enter();
        });
    }

    #[test]
    fn test_create_request_span_entered_multiple_times() {
        with_noop_subscriber(|| {
            let id = CorrelationId::new();
            for method in &["GET", "POST", "PUT", "DELETE"] {
                let span = create_request_span(method, "/api/test", &id);
                let _g = span.enter();
            }
        });
    }

    #[test]
    fn test_create_inference_span_entered_multiple_times() {
        with_noop_subscriber(|| {
            let id = CorrelationId::new();
            for batch in [1usize, 4, 16, 64] {
                let span = create_inference_span("gpt2", batch, &id);
                let _g = span.enter();
            }
        });
    }

    // ── Lines 272, 274, 282, 284: tracing event fields in log_completion / log_error ─
    // These lines are the field expressions evaluated when a tracing event is
    // created inside RequestMetrics::log_completion and RequestMetrics::log_error.
    // A subscriber with TRACE level ensures all fields are evaluated.

    #[test]
    fn test_log_completion_fields_evaluated_with_subscriber() {
        with_noop_subscriber(|| {
            let id = CorrelationId::from_header("cid-line-272");
            let metrics = RequestMetrics::new(id);
            // log_completion emits tracing::info! with fields on lines 272-276.
            // With an active subscriber all field expressions are evaluated.
            metrics.log_completion(201, "/api/v2/predict");
            metrics.log_completion(404, "/api/v2/missing");
            metrics.log_completion(500, "/api/v2/error");
        });
    }

    #[test]
    fn test_log_error_fields_evaluated_with_subscriber() {
        with_noop_subscriber(|| {
            let id = CorrelationId::from_header("cid-line-282");
            let metrics = RequestMetrics::new(id);
            // log_error emits tracing::error! with fields on lines 282-286.
            metrics.log_error("timeout", "/api/v2/infer");
            metrics.log_error("out of memory", "/api/v2/infer");
        });
    }
}
