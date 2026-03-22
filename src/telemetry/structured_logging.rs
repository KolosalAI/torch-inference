use tracing::{Level, Subscriber};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    EnvFilter, Layer, Registry,
};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use std::io;
use std::time::Instant;

/// Initialize structured logging with JSON output
pub fn init_structured_logging(log_dir: Option<&str>, json_output: bool) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

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
            .flatten_event(true);  // Flatten for easier parsing

        if let Some(dir) = log_dir {
            // File output
            let file_appender = RollingFileAppender::new(Rotation::DAILY, dir, "torch-inference.log");
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            
            // Keep guard alive to prevent dropping
            std::mem::forget(_guard);
            
            let subscriber = registry.with(json_layer.with_writer(non_blocking));
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set subscriber");
        } else {
            // Console output
            let subscriber = registry.with(json_layer);
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set subscriber");
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
        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set subscriber");
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

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Initialize OpenTelemetry
    let tracer = if let Some(endpoint) = otlp_endpoint {
        opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(endpoint)
            )
            .with_trace_config(
                Config::default()
                    .with_resource(opentelemetry_sdk::Resource::new(vec![
                        opentelemetry::KeyValue::new("service.name", "torch-inference"),
                        opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                    ]))
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
            let file_appender = RollingFileAppender::new(Rotation::DAILY, dir, "torch-inference.log");
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
fn init_structured_logging_fallback(log_dir: Option<&str>, json_output: bool) -> Result<(), Box<dyn std::error::Error>> {
    init_structured_logging(log_dir, json_output);
    Ok(())
}

/// Correlation ID middleware for request tracing
#[derive(Clone)]
pub struct CorrelationId(pub String);

impl CorrelationId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub fn from_header(value: &str) -> Self {
        Self(value.to_string())
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
}
