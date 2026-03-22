use dashmap::DashMap;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
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
            entry.value_mut().avg_inference_time_ms = 
                (entry.value_mut().avg_inference_time_ms * (entry.value_mut().inference_count as f64 - 1.0) + latency_ms) 
                / entry.value_mut().inference_count as f64;
            entry.value_mut().last_used = Utc::now();
        } else {
            self.model_metrics.insert(model_name.to_string(), ModelMetrics {
                model_name: model_name.to_string(),
                inference_count: 1,
                avg_inference_time_ms: latency_ms,
                last_used: Utc::now(),
            });
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

