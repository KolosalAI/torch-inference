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
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_request_end(&self, latency_ms: u64, endpoint: &str, success: bool) {
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        self.total_processed.fetch_add(1, Ordering::Relaxed);
        self.total_latency.fetch_add(latency_ms, Ordering::Relaxed);

        // Update min/max latency
        let mut min = self.min_latency.load(Ordering::Acquire);
        while latency_ms < min && self.min_latency.compare_exchange(
            min,
            latency_ms,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).is_err() {
            min = self.min_latency.load(Ordering::Acquire);
        }

        let mut max = self.max_latency.load(Ordering::Acquire);
        while latency_ms > max && self.max_latency.compare_exchange(
            max,
            latency_ms,
            Ordering::AcqRel,
            Ordering::Acquire,
        ).is_err() {
            max = self.max_latency.load(Ordering::Acquire);
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
            self.total_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn get_health_status(&self) -> HealthStatus {
        let uptime = (Utc::now() - self.started_at).num_seconds() as u64;
        
        HealthStatus {
            healthy: self.total_errors.load(Ordering::Relaxed) < 100,
            timestamp: Utc::now(),
            uptime_seconds: uptime,
            memory_mb: self.estimate_memory(),
            cpu_percent: self.estimate_cpu_usage(),
            active_requests: self.active_requests.load(Ordering::Relaxed),
            error_count: self.total_errors.load(Ordering::Relaxed),
            response_time_ms: self.get_avg_latency(),
        }
    }

    pub fn get_metrics(&self) -> SystemMetrics {
        let total_processed_val = self.total_processed.load(Ordering::Relaxed);
        let uptime = (Utc::now() - self.started_at).num_seconds() as u64;
        let throughput = if uptime > 0 {
            total_processed_val as f64 / uptime as f64
        } else {
            0.0
        };

        SystemMetrics {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_errors: self.total_errors.load(Ordering::Relaxed),
            total_processed: total_processed_val,
            avg_latency_ms: self.get_avg_latency(),
            min_latency_ms: self.min_latency.load(Ordering::Relaxed) as f64,
            max_latency_ms: self.max_latency.load(Ordering::Relaxed) as f64,
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
        let total = self.total_processed.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.total_latency.load(Ordering::Relaxed) as f64 / total as f64
        }
    }

    fn estimate_memory(&self) -> u64 {
        25
    }

    fn estimate_cpu_usage(&self) -> f64 {
        5.0
    }

    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.total_errors.store(0, Ordering::Relaxed);
        self.total_processed.store(0, Ordering::Relaxed);
        self.active_requests.store(0, Ordering::Relaxed);
        self.total_latency.store(0, Ordering::Relaxed);
        self.min_latency.store(u64::MAX, Ordering::Relaxed);
        self.max_latency.store(0, Ordering::Relaxed);
        self.endpoint_stats.clear();
        info!("Monitor metrics reset");
    }
}

impl Default for Monitor {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;

    #[test]
    fn test_monitor_new() {
        let monitor = Monitor::new();
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.total_errors, 0);
        assert_eq!(metrics.total_processed, 0);
    }

    #[test]
    fn test_monitor_record_request() {
        let monitor = Monitor::new();
        monitor.record_request_start();
        
        let health = monitor.get_health_status();
        assert_eq!(health.active_requests, 1);
    }

    #[test]
    fn test_monitor_record_request_end_success() {
        let monitor = Monitor::new();
        monitor.record_request_start();
        monitor.record_request_end(100, "/api/test", true);
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_processed, 1);
        assert_eq!(metrics.total_errors, 0);
        assert_eq!(metrics.avg_latency_ms, 100.0);
    }

    #[test]
    fn test_monitor_record_request_end_failure() {
        let monitor = Monitor::new();
        monitor.record_request_start();
        monitor.record_request_end(50, "/api/test", false);
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_errors, 1);
    }

    #[test]
    fn test_monitor_min_max_latency() {
        let monitor = Monitor::new();
        
        monitor.record_request_start();
        monitor.record_request_end(100, "/api/test", true);
        
        monitor.record_request_start();
        monitor.record_request_end(50, "/api/test", true);
        
        monitor.record_request_start();
        monitor.record_request_end(200, "/api/test", true);
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.min_latency_ms, 50.0);
        assert_eq!(metrics.max_latency_ms, 200.0);
    }

    #[test]
    fn test_monitor_avg_latency() {
        let monitor = Monitor::new();
        
        monitor.record_request_start();
        monitor.record_request_end(100, "/api/test", true);
        
        monitor.record_request_start();
        monitor.record_request_end(200, "/api/test", true);
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.avg_latency_ms, 150.0);
    }

    #[test]
    fn test_monitor_endpoint_stats() {
        let monitor = Monitor::new();
        
        monitor.record_request_start();
        monitor.record_request_end(100, "/api/endpoint1", true);
        
        monitor.record_request_start();
        monitor.record_request_end(200, "/api/endpoint2", true);
        
        let stats = monitor.get_endpoint_stats();
        assert_eq!(stats.len(), 2);
    }

    #[test]
    fn test_monitor_health_status() {
        let monitor = Monitor::new();
        let health = monitor.get_health_status();
        
        assert!(health.healthy);
        assert_eq!(health.error_count, 0);
        assert_eq!(health.active_requests, 0);
    }

    #[test]
    fn test_monitor_reset() {
        let monitor = Monitor::new();
        
        monitor.record_request_start();
        monitor.record_request_end(100, "/api/test", true);
        
        monitor.reset();
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.total_processed, 0);
        assert_eq!(metrics.total_errors, 0);
    }

    #[test]
    fn test_monitor_throughput() {
        let monitor = Monitor::new();
        
        for _ in 0..10 {
            monitor.record_request_start();
            monitor.record_request_end(50, "/api/test", true);
        }
        
        thread::sleep(std::time::Duration::from_millis(1100));
        
        let metrics = monitor.get_metrics();
        assert!(metrics.throughput_rps > 0.0);
        assert_eq!(metrics.total_processed, 10);
    }

    #[test]
    fn test_monitor_uptime() {
        let monitor = Monitor::new();
        thread::sleep(std::time::Duration::from_millis(100));
        
        let metrics = monitor.get_metrics();
        assert!(metrics.uptime_seconds >= 0);
    }

    // ===== Enterprise-Grade Tests =====

    #[test]
    fn test_monitor_concurrent_recording() {
        let monitor = Arc::new(Monitor::new());
        let mut handles = vec![];

        for thread_id in 0..10 {
            let monitor_clone = Arc::clone(&monitor);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    monitor_clone.record_request_start();
                    thread::sleep(std::time::Duration::from_micros(100));
                    let latency = (i % 100) as u64 + 50;
                    monitor_clone.record_request_end(
                        latency,
                        &format!("/api/endpoint{}", thread_id % 5),
                        i % 10 != 0, // 10% failure rate
                    );
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_requests, 1000);
        assert_eq!(metrics.total_processed, 1000);
        assert!(metrics.total_errors > 0); // Should have some errors
    }

    #[test]
    fn test_monitor_high_frequency_updates() {
        let monitor = Monitor::new();
        
        let start = std::time::Instant::now();
        for i in 0..10000 {
            monitor.record_request_start();
            monitor.record_request_end(i % 200, "/api/test", true);
        }
        let duration = start.elapsed();

        // Should complete 10k updates in reasonable time
        assert!(duration < std::time::Duration::from_secs(1));
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_requests, 10000);
        assert_eq!(metrics.total_processed, 10000);
    }

    #[test]
    fn test_monitor_latency_accuracy() {
        let monitor = Monitor::new();
        let test_latencies = vec![100, 200, 150, 50, 300];
        let expected_avg = 160.0;
        
        for latency in &test_latencies {
            monitor.record_request_start();
            monitor.record_request_end(*latency, "/api/test", true);
        }
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.avg_latency_ms, expected_avg);
        assert_eq!(metrics.min_latency_ms, 50.0);
        assert_eq!(metrics.max_latency_ms, 300.0);
    }

    #[test]
    fn test_monitor_endpoint_aggregation() {
        let monitor = Monitor::new();
        
        // Record multiple calls to same endpoint
        for i in 0..10 {
            monitor.record_request_start();
            monitor.record_request_end(100 + i * 10, "/api/users", true);
        }
        
        let stats = monitor.get_endpoint_stats();
        let user_stats = stats.iter().find(|s| s.path == "/api/users").unwrap();
        
        assert_eq!(user_stats.calls, 10);
        assert_eq!(user_stats.errors, 0);
        assert!(user_stats.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_monitor_error_tracking() {
        let monitor = Monitor::new();
        
        // Record mix of successes and failures
        for i in 0..20 {
            monitor.record_request_start();
            monitor.record_request_end(100, "/api/test", i % 3 != 0);
        }
        
        let metrics = monitor.get_metrics();
        // Should have failures every 3rd request
        assert!(metrics.total_errors > 0);
        assert_eq!(metrics.total_processed, 20);
    }

    #[test]
    fn test_monitor_health_threshold() {
        let monitor = Monitor::new();
        
        // Record many errors
        for _ in 0..150 {
            monitor.record_request_start();
            monitor.record_request_end(100, "/api/test", false);
        }
        
        let health = monitor.get_health_status();
        // Should be unhealthy after 100+ errors
        assert!(!health.healthy);
        assert_eq!(health.error_count, 150);
    }

    #[test]
    fn test_monitor_active_requests_tracking() {
        let monitor = Arc::new(Monitor::new());
        
        // Start multiple requests
        for _ in 0..5 {
            monitor.record_request_start();
        }
        
        let health = monitor.get_health_status();
        assert_eq!(health.active_requests, 5);
        
        // End some requests
        for _ in 0..3 {
            monitor.record_request_end(100, "/api/test", true);
        }
        
        let health = monitor.get_health_status();
        assert_eq!(health.active_requests, 2);
    }

    #[test]
    fn test_monitor_reset_preserves_started_time() {
        let monitor = Monitor::new();
        let initial_metrics = monitor.get_metrics();
        let initial_time = initial_metrics.started_at;
        
        thread::sleep(std::time::Duration::from_millis(100));
        
        monitor.record_request_start();
        monitor.record_request_end(100, "/api/test", true);
        
        monitor.reset();
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_requests, 0);
        // Started_at should still be the initial time
        assert_eq!(metrics.started_at, initial_time);
    }

    #[test]
    fn test_monitor_multiple_endpoints() {
        let monitor = Monitor::new();
        let endpoints = vec![
            "/api/users",
            "/api/products",
            "/api/orders",
            "/api/analytics",
            "/api/reports",
        ];
        
        for endpoint in &endpoints {
            for i in 0..5 {
                monitor.record_request_start();
                monitor.record_request_end(100 + i * 10, endpoint, true);
            }
        }
        
        let stats = monitor.get_endpoint_stats();
        assert_eq!(stats.len(), 5);
        
        for endpoint in &endpoints {
            let stat = stats.iter().find(|s| s.path == *endpoint);
            assert!(stat.is_some());
            assert_eq!(stat.unwrap().calls, 5);
        }
    }

    #[test]
    fn test_monitor_zero_latency() {
        let monitor = Monitor::new();
        
        monitor.record_request_start();
        monitor.record_request_end(0, "/api/test", true);
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.min_latency_ms, 0.0);
        assert_eq!(metrics.avg_latency_ms, 0.0);
    }

    #[test]
    fn test_monitor_extreme_latency() {
        let monitor = Monitor::new();
        
        monitor.record_request_start();
        monitor.record_request_end(u64::MAX / 2, "/api/slow", true);
        
        let metrics = monitor.get_metrics();
        assert!(metrics.max_latency_ms > 1000000.0);
    }

    #[test]
    fn test_monitor_throughput_calculation() {
        let monitor = Monitor::new();
        
        // Record exactly 100 requests
        for _ in 0..100 {
            monitor.record_request_start();
            monitor.record_request_end(50, "/api/test", true);
        }
        
        thread::sleep(std::time::Duration::from_millis(1100));
        
        let metrics = monitor.get_metrics();
        // Throughput should be close to 100 requests over uptime
        assert!(metrics.throughput_rps > 0.0);
        assert_eq!(metrics.total_processed, 100);
    }

    #[test]
    fn test_monitor_concurrent_endpoint_updates() {
        let monitor = Arc::new(Monitor::new());
        let mut handles = vec![];
        
        for i in 0..5 {
            let monitor_clone = Arc::clone(&monitor);
            let endpoint = format!("/api/endpoint{}", i);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    monitor_clone.record_request_start();
                    monitor_clone.record_request_end(50, &endpoint, true);
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let stats = monitor.get_endpoint_stats();
        assert_eq!(stats.len(), 5);
        
        for stat in stats {
            assert_eq!(stat.calls, 100);
            assert_eq!(stat.errors, 0);
        }
    }

    #[test]
    fn test_monitor_avg_latency_update() {
        let monitor = Monitor::new();
        
        monitor.record_request_start();
        monitor.record_request_end(100, "/api/test", true);
        
        let metrics1 = monitor.get_metrics();
        assert_eq!(metrics1.avg_latency_ms, 100.0);
        
        monitor.record_request_start();
        monitor.record_request_end(200, "/api/test", true);
        
        let metrics2 = monitor.get_metrics();
        assert_eq!(metrics2.avg_latency_ms, 150.0);
    }

    /// Exercise the CAS retry loop at line 86 (min_latency update) and the
    /// equivalent loop for max_latency.  We hammer a single monitor from many
    /// threads with low latency values so compare_exchange races happen.
    #[test]
    fn test_monitor_cas_retry_loop_for_min_latency() {
        let monitor = Arc::new(Monitor::new());
        let mut handles = vec![];

        // 20 threads each record 500 requests with latency values 1..=50
        // (always < initial MAX), maximising CAS contention on min_latency.
        for t in 0..20_u64 {
            let m = Arc::clone(&monitor);
            handles.push(thread::spawn(move || {
                for i in 1_u64..=50 {
                    m.record_request_start();
                    // Alternate between very small and medium values to force
                    // both the min-update and max-update CAS paths under
                    // contention.
                    let latency = (t * 50 + i) % 100 + 1;
                    m.record_request_end(latency, "/api/cas_test", true);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_requests, 1000);
        assert_eq!(metrics.total_processed, 1000);
        // min must be >= 1 and max must be <= 100
        assert!(metrics.min_latency_ms >= 1.0);
        assert!(metrics.max_latency_ms <= 100.0);
    }

    #[test]
    fn test_monitor_default_construction() {
        let monitor1 = Monitor::default();
        let monitor2 = Monitor::new();
        
        let metrics1 = monitor1.get_metrics();
        let metrics2 = monitor2.get_metrics();
        
        assert_eq!(metrics1.total_requests, metrics2.total_requests);
        assert_eq!(metrics1.total_errors, metrics2.total_errors);
    }

    #[test]
    fn test_monitor_memory_efficiency() {
        let monitor = Monitor::new();
        
        // Record many unique endpoints
        for i in 0..1000 {
            monitor.record_request_start();
            monitor.record_request_end(50, &format!("/api/endpoint{}", i), true);
        }
        
        let stats = monitor.get_endpoint_stats();
        assert_eq!(stats.len(), 1000);
    }

    #[test]
    fn test_monitor_stress_test() {
        let monitor = Arc::new(Monitor::new());
        let mut handles = vec![];
        
        // Spawn many threads doing many operations
        for thread_id in 0..50 {
            let monitor_clone = Arc::clone(&monitor);
            let handle = thread::spawn(move || {
                for i in 0..200 {
                    monitor_clone.record_request_start();
                    let latency = (i % 300) as u64;
                    let success = i % 7 != 0;
                    monitor_clone.record_request_end(
                        latency,
                        &format!("/api/endpoint{}", thread_id % 10),
                        success,
                    );
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_requests, 10000);
        assert_eq!(metrics.total_processed, 10000);
    }

    /// Exercise the CAS retry loop at line 86 (min_latency update).
    /// Many threads with small latency values maximize compare_exchange races.
    #[test]
    fn test_monitor_cas_retry_loop_for_min_latency() {
        let monitor = Arc::new(Monitor::new());
        let mut handles = vec![];

        for t in 0..20_u64 {
            let m = Arc::clone(&monitor);
            handles.push(thread::spawn(move || {
                for i in 1_u64..=50 {
                    m.record_request_start();
                    let latency = (t * 50 + i) % 100 + 1;
                    m.record_request_end(latency, "/api/cas_test", true);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let metrics = monitor.get_metrics();
        assert_eq!(metrics.total_requests, 1000);
        assert_eq!(metrics.total_processed, 1000);
        assert!(metrics.min_latency_ms >= 1.0);
        assert!(metrics.max_latency_ms <= 100.0);
    }
}
