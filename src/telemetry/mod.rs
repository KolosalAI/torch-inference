pub mod logger;
pub mod metrics;
pub mod prometheus;
pub mod structured_logging;

#[allow(unused_imports)]
pub use structured_logging::{create_inference_span, create_request_span};
pub use structured_logging::{init_structured_logging, CorrelationId, RequestMetrics};

#[cfg(feature = "telemetry")]
pub use structured_logging::init_with_tracing;

#[cfg(feature = "metrics")]
pub use prometheus::{
    init_metrics, record_batch_size, record_model_load_time, record_queue_time, record_request,
    render_metrics, update_active_requests, update_cache_metrics, update_model_instances,
    update_queue_depth,
};
