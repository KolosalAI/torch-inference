#![allow(dead_code)]
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub total_requests: u64,
    pub total_errors: u64,
    pub avg_latency_ms: f64,
    pub max_latency_ms: f64,
    pub min_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_name: String,
    pub inference_count: u64,
    pub avg_inference_time_ms: f64,
    pub last_used: DateTime<Utc>,
}

pub struct MetricsCollector {
    request_count: AtomicU64,
    error_count: AtomicU64,
    model_metrics: DashMap<String, ModelMetrics>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            request_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            model_metrics: DashMap::new(),
        }
    }

    pub fn record_request(&self) {
        self.request_count.fetch_add(1, Ordering::SeqCst);
    }

    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::SeqCst);
    }

    pub fn record_inference(&self, model_name: &str, latency_ms: f64) {
        self.record_request();

        if let Some(mut entry) = self.model_metrics.get_mut(model_name) {
            entry.value_mut().inference_count += 1;
            entry.value_mut().avg_inference_time_ms = (entry.value_mut().avg_inference_time_ms
                * (entry.value_mut().inference_count as f64 - 1.0)
                + latency_ms)
                / entry.value_mut().inference_count as f64;
            entry.value_mut().last_used = Utc::now();
        } else {
            self.model_metrics.insert(
                model_name.to_string(),
                ModelMetrics {
                    model_name: model_name.to_string(),
                    inference_count: 1,
                    avg_inference_time_ms: latency_ms,
                    last_used: Utc::now(),
                },
            );
        }
    }

    pub fn get_request_metrics(&self) -> RequestMetrics {
        RequestMetrics {
            total_requests: self.request_count.load(Ordering::SeqCst),
            total_errors: self.error_count.load(Ordering::SeqCst),
            avg_latency_ms: 0.0,
            max_latency_ms: 0.0,
            min_latency_ms: 0.0,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MetricsCollector::new / Default ──────────────────────────────────────

    #[test]
    fn test_new_starts_at_zero() {
        let c = MetricsCollector::new();
        let m = c.get_request_metrics();
        assert_eq!(m.total_requests, 0);
        assert_eq!(m.total_errors, 0);
    }

    #[test]
    fn test_default_is_same_as_new() {
        let c: MetricsCollector = MetricsCollector::default();
        let m = c.get_request_metrics();
        assert_eq!(m.total_requests, 0);
        assert_eq!(m.total_errors, 0);
    }

    // ── record_request ────────────────────────────────────────────────────────

    #[test]
    fn test_record_request_increments_count() {
        let c = MetricsCollector::new();
        c.record_request();
        c.record_request();
        let m = c.get_request_metrics();
        assert_eq!(m.total_requests, 2);
    }

    // ── record_error ──────────────────────────────────────────────────────────

    #[test]
    fn test_record_error_increments_error_count() {
        let c = MetricsCollector::new();
        c.record_error();
        let m = c.get_request_metrics();
        assert_eq!(m.total_errors, 1);
        assert_eq!(
            m.total_requests, 0,
            "record_error must not touch request_count"
        );
    }

    #[test]
    fn test_record_error_multiple_times() {
        let c = MetricsCollector::new();
        for _ in 0..5 {
            c.record_error();
        }
        let m = c.get_request_metrics();
        assert_eq!(m.total_errors, 5);
    }

    // ── record_inference (new entry path) ────────────────────────────────────

    #[test]
    fn test_record_inference_new_model_creates_entry() {
        let c = MetricsCollector::new();
        c.record_inference("gpt2", 42.0);

        // record_inference calls record_request internally
        let m = c.get_request_metrics();
        assert_eq!(m.total_requests, 1);

        let entry = c.model_metrics.get("gpt2").expect("entry should exist");
        assert_eq!(entry.inference_count, 1);
        assert!((entry.avg_inference_time_ms - 42.0).abs() < 1e-9);
        assert_eq!(entry.model_name, "gpt2");
    }

    // ── record_inference (existing entry path) ────────────────────────────────

    #[test]
    fn test_record_inference_existing_model_updates_entry() {
        let c = MetricsCollector::new();
        c.record_inference("bert", 10.0);
        c.record_inference("bert", 20.0);

        let m = c.get_request_metrics();
        assert_eq!(m.total_requests, 2);

        let entry = c.model_metrics.get("bert").expect("entry should exist");
        assert_eq!(entry.inference_count, 2);
        // avg should be (10 + 20) / 2 = 15
        assert!((entry.avg_inference_time_ms - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_record_inference_three_calls_average() {
        let c = MetricsCollector::new();
        c.record_inference("model_x", 30.0);
        c.record_inference("model_x", 60.0);
        c.record_inference("model_x", 90.0);

        let entry = c.model_metrics.get("model_x").expect("entry should exist");
        assert_eq!(entry.inference_count, 3);
        // avg should be 60
        assert!((entry.avg_inference_time_ms - 60.0).abs() < 1e-5);
    }

    #[test]
    fn test_record_inference_multiple_models_independent() {
        let c = MetricsCollector::new();
        c.record_inference("alpha", 100.0);
        c.record_inference("beta", 200.0);

        let alpha = c.model_metrics.get("alpha").expect("alpha entry");
        let beta = c.model_metrics.get("beta").expect("beta entry");

        assert_eq!(alpha.inference_count, 1);
        assert_eq!(beta.inference_count, 1);
        assert!((alpha.avg_inference_time_ms - 100.0).abs() < 1e-9);
        assert!((beta.avg_inference_time_ms - 200.0).abs() < 1e-9);
    }

    // ── get_request_metrics ───────────────────────────────────────────────────

    #[test]
    fn test_get_request_metrics_latency_fields() {
        // The current implementation always returns 0.0 for latency fields –
        // verify this contract so a future change is caught.
        let c = MetricsCollector::new();
        c.record_inference("m", 50.0);
        let m = c.get_request_metrics();
        assert_eq!(m.avg_latency_ms, 0.0);
        assert_eq!(m.max_latency_ms, 0.0);
        assert_eq!(m.min_latency_ms, 0.0);
    }

    // ── RequestMetrics / ModelMetrics – derive traits ─────────────────────────

    #[test]
    fn test_request_metrics_clone_and_debug() {
        let rm = RequestMetrics {
            total_requests: 7,
            total_errors: 2,
            avg_latency_ms: 1.5,
            max_latency_ms: 3.0,
            min_latency_ms: 0.5,
        };
        let cloned = rm.clone();
        assert_eq!(cloned.total_requests, 7);
        // Debug must not panic
        let _ = format!("{:?}", cloned);
    }

    #[test]
    fn test_model_metrics_clone_and_debug() {
        let mm = ModelMetrics {
            model_name: "test".to_string(),
            inference_count: 3,
            avg_inference_time_ms: 25.0,
            last_used: Utc::now(),
        };
        let cloned = mm.clone();
        assert_eq!(cloned.model_name, "test");
        let _ = format!("{:?}", cloned);
    }

    // ── Serialisation round-trip ──────────────────────────────────────────────

    #[test]
    fn test_request_metrics_serde_roundtrip() {
        let rm = RequestMetrics {
            total_requests: 10,
            total_errors: 1,
            avg_latency_ms: 2.5,
            max_latency_ms: 5.0,
            min_latency_ms: 1.0,
        };
        let json = serde_json::to_string(&rm).expect("serialize");
        let back: RequestMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.total_requests, 10);
        assert_eq!(back.total_errors, 1);
    }

    #[test]
    fn test_model_metrics_serde_roundtrip() {
        let mm = ModelMetrics {
            model_name: "roundtrip_model".to_string(),
            inference_count: 5,
            avg_inference_time_ms: 12.3,
            last_used: Utc::now(),
        };
        let json = serde_json::to_string(&mm).expect("serialize");
        let back: ModelMetrics = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model_name, "roundtrip_model");
        assert_eq!(back.inference_count, 5);
    }
}
