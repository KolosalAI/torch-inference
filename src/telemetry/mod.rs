pub mod logger;
pub mod metrics;
pub mod structured_logging;
pub mod prometheus;

pub use structured_logging::{
    init_structured_logging, CorrelationId,
    RequestMetrics,
};
#[allow(unused_imports)]
pub use structured_logging::{create_request_span, create_inference_span};

#[cfg(feature = "telemetry")]
pub use structured_logging::init_with_tracing;

#[cfg(feature = "metrics")]
pub use prometheus::{
    init_metrics, render_metrics,
    record_request, update_active_requests, update_cache_metrics,
    record_batch_size, record_queue_time, record_model_load_time,
    update_model_instances, update_queue_depth,
};

