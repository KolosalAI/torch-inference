use serde::{Serialize, Deserialize};
use std::sync::atomic::{AtomicU64, Ordering};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use log::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub timestamp: DateTime<Utc>,
    pub uptime_seconds: u64,
    pub memory_mb: u64,
    pub cpu_percent: f64,
    pub active_requests: u64,
    pub error_count: u64,
    pub response_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub total_requests: u64,
    pub total_errors: u64,
    pub total_processed: u64,
    pub avg_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub throughput_rps: f64,
    pub uptime_seconds: u64,
    pub started_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointStats {
    pub path: String,
    pub calls: u64,
    pub errors: u64,
    pub avg_latency_ms: f64,
    pub last_called: DateTime<Utc>,
}

pub struct Monitor {
    total_requests: AtomicU64,
    total_errors: AtomicU64,
    total_processed: AtomicU64,
    active_requests: AtomicU64,
    total_latency: AtomicU64,
    min_latency: AtomicU64,
    max_latency: AtomicU64,
    started_at: DateTime<Utc>,
    endpoint_stats: DashMap<String, EndpointStats>,
}

impl Monitor {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_processed: AtomicU64::new(0),
            active_requests: AtomicU64::new(0),
            total_latency: AtomicU64::new(0),
            min_latency: AtomicU64::new(u64::MAX),
            max_latency: AtomicU64::new(0),
            started_at: Utc::now(),
            endpoint_stats: DashMap::new(),
        }
    }

    pub fn record_request_start(&self) {
        self.total_requests.fetch_add(1, Ordering::SeqCst);
        self.active_requests.fetch_add(1, Ordering::SeqCst);
    }

    pub fn record_request_end(&self, latency_ms: u64, endpoint: &str, success: bool) {
        self.active_requests.fetch_sub(1, Ordering::SeqCst);
        self.total_processed.fetch_add(1, Ordering::SeqCst);
        self.total_latency.fetch_add(latency_ms, Ordering::SeqCst);

        // Update min/max latency
        let mut min = self.min_latency.load(Ordering::SeqCst);
        while latency_ms < min && self.min_latency.compare_exchange(
            min,
            latency_ms,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ).is_err() {
            min = self.min_latency.load(Ordering::SeqCst);
        }

        let mut max = self.max_latency.load(Ordering::SeqCst);
        while latency_ms > max && self.max_latency.compare_exchange(
            max,
            latency_ms,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ).is_err() {
            max = self.max_latency.load(Ordering::SeqCst);
        }

        // Update endpoint stats
        self.endpoint_stats
            .entry(endpoint.to_string())
            .and_modify(|stats| {
                stats.calls += 1;
                stats.avg_latency_ms = (stats.avg_latency_ms * (stats.calls as f64 - 1.0) + latency_ms as f64) 
                    / stats.calls as f64;
                stats.last_called = Utc::now();
                if !success {
                    stats.errors += 1;
                }
            })
            .or_insert_with(|| EndpointStats {
                path: endpoint.to_string(),
                calls: 1,
                errors: if success { 0 } else { 1 },
                avg_latency_ms: latency_ms as f64,
                last_called: Utc::now(),
            });

        if !success {
            self.total_errors.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub fn get_health_status(&self) -> HealthStatus {
        let uptime = (Utc::now() - self.started_at).num_seconds() as u64;
        
        HealthStatus {
            healthy: self.total_errors.load(Ordering::SeqCst) < 100,
            timestamp: Utc::now(),
            uptime_seconds: uptime,
            memory_mb: self.estimate_memory(),
            cpu_percent: self.estimate_cpu_usage(),
            active_requests: self.active_requests.load(Ordering::SeqCst),
            error_count: self.total_errors.load(Ordering::SeqCst),
            response_time_ms: self.get_avg_latency(),
        }
    }

    pub fn get_metrics(&self) -> SystemMetrics {
        let total_processed_val = self.total_processed.load(Ordering::SeqCst);
        let uptime = (Utc::now() - self.started_at).num_seconds() as u64;
        let throughput = if uptime > 0 {
            total_processed_val as f64 / uptime as f64
        } else {
            0.0
        };

        SystemMetrics {
            total_requests: self.total_requests.load(Ordering::SeqCst),
            total_errors: self.total_errors.load(Ordering::SeqCst),
            total_processed: total_processed_val,
            avg_latency_ms: self.get_avg_latency(),
            min_latency_ms: self.min_latency.load(Ordering::SeqCst) as f64,
            max_latency_ms: self.max_latency.load(Ordering::SeqCst) as f64,
            throughput_rps: throughput,
            uptime_seconds: uptime,
            started_at: self.started_at,
        }
    }

    pub fn get_endpoint_stats(&self) -> Vec<EndpointStats> {
        self.endpoint_stats
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    fn get_avg_latency(&self) -> f64 {
        let total = self.total_processed.load(Ordering::SeqCst);
        if total == 0 {
            0.0
        } else {
            self.total_latency.load(Ordering::SeqCst) as f64 / total as f64
        }
    }

    fn estimate_memory(&self) -> u64 {
        25
    }

    fn estimate_cpu_usage(&self) -> f64 {
        5.0
    }

    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::SeqCst);
        self.total_errors.store(0, Ordering::SeqCst);
        self.total_processed.store(0, Ordering::SeqCst);
        self.active_requests.store(0, Ordering::SeqCst);
        self.total_latency.store(0, Ordering::SeqCst);
        self.min_latency.store(u64::MAX, Ordering::SeqCst);
        self.max_latency.store(0, Ordering::SeqCst);
        self.endpoint_stats.clear();
        info!("Monitor metrics reset");
    }
}

impl Default for Monitor {
    fn default() -> Self {
        Self::new()
    }
}


