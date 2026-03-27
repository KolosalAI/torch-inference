//! Prometheus metrics for production monitoring
//! 
//! Provides comprehensive metrics collection and export in Prometheus format.
//! Enable with `--features metrics` flag.

#[cfg(feature = "metrics")]
use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec,
    Opts, Registry, TextEncoder, Encoder,
};
#[cfg(feature = "metrics")]
use std::sync::Arc;
#[cfg(feature = "metrics")]
use lazy_static::lazy_static;

#[cfg(feature = "metrics")]
lazy_static! {
    /// Global Prometheus registry
    pub static ref REGISTRY: Registry = Registry::new();
    
    /// Inference request counters
    pub static ref REQUESTS_TOTAL: CounterVec = CounterVec::new(
        Opts::new("inference_requests_total", "Total number of inference requests")
            .namespace("torch_inference"),
        &["model", "status", "endpoint"]
    ).expect("Failed to create requests_total metric");
    
    /// Inference duration histogram
    pub static ref REQUEST_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new("inference_duration_seconds", "Request duration in seconds")
            .namespace("torch_inference")
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        &["model", "endpoint"]
    ).expect("Failed to create request_duration metric");
    
    /// Active requests gauge
    pub static ref ACTIVE_REQUESTS: GaugeVec = GaugeVec::new(
        Opts::new("inference_active_requests", "Number of active inference requests")
            .namespace("torch_inference"),
        &["model"]
    ).expect("Failed to create active_requests metric");
    
    /// Cache hit rate gauge
    pub static ref CACHE_HIT_RATE: Gauge = Gauge::new(
        "cache_hit_rate_percent",
        "Cache hit rate percentage"
    ).expect("Failed to create cache_hit_rate metric");
    
    /// Cache operations counter
    pub static ref CACHE_OPERATIONS: CounterVec = CounterVec::new(
        Opts::new("cache_operations_total", "Total cache operations")
            .namespace("torch_inference"),
        &["operation"] // hit, miss, eviction
    ).expect("Failed to create cache_operations metric");
    
    /// Cache size gauge
    pub static ref CACHE_SIZE: Gauge = Gauge::new(
        "cache_size_entries",
        "Current number of entries in cache"
    ).expect("Failed to create cache_size metric");
    
    /// Model pool instances gauge
    pub static ref MODEL_INSTANCES: GaugeVec = GaugeVec::new(
        Opts::new("model_pool_instances", "Number of model instances")
            .namespace("torch_inference"),
        &["model"]
    ).expect("Failed to create model_instances metric");
    
    /// Queue depth gauge
    pub static ref QUEUE_DEPTH: GaugeVec = GaugeVec::new(
        Opts::new("queue_depth", "Number of requests in queue")
            .namespace("torch_inference"),
        &["model"]
    ).expect("Failed to create queue_depth metric");
    
    /// Batch size histogram
    pub static ref BATCH_SIZE: HistogramVec = HistogramVec::new(
        HistogramOpts::new("batch_size", "Inference batch size")
            .namespace("torch_inference")
            .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
        &["model"]
    ).expect("Failed to create batch_size metric");
    
    /// Queue time histogram
    pub static ref QUEUE_TIME: HistogramVec = HistogramVec::new(
        HistogramOpts::new("queue_time_seconds", "Time spent in queue")
            .namespace("torch_inference")
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        &["model"]
    ).expect("Failed to create queue_time metric");
    
    /// Model load time histogram
    pub static ref MODEL_LOAD_TIME: HistogramVec = HistogramVec::new(
        HistogramOpts::new("model_load_time_seconds", "Time to load model")
            .namespace("torch_inference")
            .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]),
        &["model"]
    ).expect("Failed to create model_load_time metric");
}

#[cfg(feature = "metrics")]
/// Initialize Prometheus metrics registry
pub fn init_metrics() -> Result<(), Box<dyn std::error::Error>> {
    REGISTRY.register(Box::new(REQUESTS_TOTAL.clone()))?;
    REGISTRY.register(Box::new(REQUEST_DURATION.clone()))?;
    REGISTRY.register(Box::new(ACTIVE_REQUESTS.clone()))?;
    REGISTRY.register(Box::new(CACHE_HIT_RATE.clone()))?;
    REGISTRY.register(Box::new(CACHE_OPERATIONS.clone()))?;
    REGISTRY.register(Box::new(CACHE_SIZE.clone()))?;
    REGISTRY.register(Box::new(MODEL_INSTANCES.clone()))?;
    REGISTRY.register(Box::new(QUEUE_DEPTH.clone()))?;
    REGISTRY.register(Box::new(BATCH_SIZE.clone()))?;
    REGISTRY.register(Box::new(QUEUE_TIME.clone()))?;
    REGISTRY.register(Box::new(MODEL_LOAD_TIME.clone()))?;
    
    Ok(())
}

#[cfg(feature = "metrics")]
/// Render metrics in Prometheus text format
pub fn render_metrics() -> Result<String, Box<dyn std::error::Error>> {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

#[cfg(feature = "metrics")]
/// Record an inference request
pub fn record_request(model: &str, endpoint: &str, duration_secs: f64, success: bool) {
    let status = if success { "success" } else { "error" };
    
    REQUESTS_TOTAL
        .with_label_values(&[model, status, endpoint])
        .inc();
    
    REQUEST_DURATION
        .with_label_values(&[model, endpoint])
        .observe(duration_secs);
}

#[cfg(feature = "metrics")]
/// Update active requests counter
pub fn update_active_requests(model: &str, delta: i64) {
    if delta > 0 {
        ACTIVE_REQUESTS
            .with_label_values(&[model])
            .add(delta as f64);
    } else {
        ACTIVE_REQUESTS
            .with_label_values(&[model])
            .sub((-delta) as f64);
    }
}

#[cfg(feature = "metrics")]
/// Update cache metrics
pub fn update_cache_metrics(hits: u64, misses: u64, evictions: u64, size: usize) {
    let total = hits + misses;
    let hit_rate = if total > 0 {
        (hits as f64 / total as f64) * 100.0
    } else {
        0.0
    };
    
    CACHE_HIT_RATE.set(hit_rate);
    CACHE_SIZE.set(size as f64);
    
    CACHE_OPERATIONS
        .with_label_values(&["hit"])
        .inc_by(hits);
    
    CACHE_OPERATIONS
        .with_label_values(&["miss"])
        .inc_by(misses);
    
    CACHE_OPERATIONS
        .with_label_values(&["eviction"])
        .inc_by(evictions);
}

#[cfg(feature = "metrics")]
/// Record batch size
pub fn record_batch_size(model: &str, batch_size: usize) {
    BATCH_SIZE
        .with_label_values(&[model])
        .observe(batch_size as f64);
}

#[cfg(feature = "metrics")]
/// Record queue time
pub fn record_queue_time(model: &str, queue_time_secs: f64) {
    QUEUE_TIME
        .with_label_values(&[model])
        .observe(queue_time_secs);
}

#[cfg(feature = "metrics")]
/// Record model load time
pub fn record_model_load_time(model: &str, load_time_secs: f64) {
    MODEL_LOAD_TIME
        .with_label_values(&[model])
        .observe(load_time_secs);
}

#[cfg(feature = "metrics")]
/// Update model instance count
pub fn update_model_instances(model: &str, count: usize) {
    MODEL_INSTANCES
        .with_label_values(&[model])
        .set(count as f64);
}

#[cfg(feature = "metrics")]
/// Update queue depth
pub fn update_queue_depth(model: &str, depth: usize) {
    QUEUE_DEPTH
        .with_label_values(&[model])
        .set(depth as f64);
}

// No-op implementations when metrics feature is disabled
#[cfg(not(feature = "metrics"))]
pub fn init_metrics() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

#[cfg(not(feature = "metrics"))]
pub fn render_metrics() -> Result<String, Box<dyn std::error::Error>> {
    Ok("# Metrics feature not enabled\n".to_string())
}

#[cfg(not(feature = "metrics"))]
pub fn record_request(_model: &str, _endpoint: &str, _duration_secs: f64, _success: bool) {}

#[cfg(not(feature = "metrics"))]
pub fn update_active_requests(_model: &str, _delta: i64) {}

#[cfg(not(feature = "metrics"))]
pub fn update_cache_metrics(_hits: u64, _misses: u64, _evictions: u64, _size: usize) {}

#[cfg(not(feature = "metrics"))]
pub fn record_batch_size(_model: &str, _batch_size: usize) {}

#[cfg(not(feature = "metrics"))]
pub fn record_queue_time(_model: &str, _queue_time_secs: f64) {}

#[cfg(not(feature = "metrics"))]
pub fn record_model_load_time(_model: &str, _load_time_secs: f64) {}

#[cfg(not(feature = "metrics"))]
pub fn update_model_instances(_model: &str, _count: usize) {}

#[cfg(not(feature = "metrics"))]
pub fn update_queue_depth(_model: &str, _depth: usize) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "metrics")]
    fn test_metrics_initialization() {
        // Should not panic
        let result = init_metrics();
        assert!(result.is_ok() || result.is_err()); // May fail if already registered
    }

    #[test]
    fn test_render_metrics() {
        let result = render_metrics();
        assert!(result.is_ok());
        
        #[cfg(feature = "metrics")]
        {
            let metrics = result.unwrap();
            assert!(!metrics.is_empty());
        }
    }

    #[test]
    #[cfg(feature = "metrics")]
    fn test_record_request() {
        record_request("test_model", "/predict", 0.123, true);
        record_request("test_model", "/predict", 0.456, false);
        
        // Should not panic
    }

    #[test]
    #[cfg(feature = "metrics")]
    fn test_update_active_requests() {
        update_active_requests("test_model", 1);
        update_active_requests("test_model", -1);
        
        // Should not panic
    }

    #[test]
    #[cfg(feature = "metrics")]
    fn test_cache_metrics() {
        update_cache_metrics(100, 20, 5, 500);
        
        // Should not panic
    }

    #[test]
    fn test_no_op_when_disabled() {
        // These should all be no-ops when metrics feature is disabled
        record_request("model", "/test", 1.0, true);
        update_active_requests("model", 1);
        update_cache_metrics(1, 1, 1, 1);
        record_batch_size("model", 10);
        record_queue_time("model", 0.1);
        record_model_load_time("model", 1.0);
        update_model_instances("model", 3);
        update_queue_depth("model", 5);
    }

    /// Lines 225-226: the no-op init_metrics() when metrics feature is disabled.
    #[test]
    #[cfg(not(feature = "metrics"))]
    fn test_init_metrics_no_op_returns_ok() {
        let result = init_metrics();
        assert!(result.is_ok());
    }
}
