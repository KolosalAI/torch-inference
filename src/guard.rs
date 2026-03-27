use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};
use tokio::sync::RwLock;
use log::{warn, error, info, debug};
use std::collections::HashMap;

/// Guard violation severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Guard violation record
#[derive(Debug, Clone)]
pub struct GuardViolation {
    pub guard_name: String,
    pub severity: ViolationSeverity,
    pub message: String,
    pub timestamp: Instant,
    pub value: f64,
    pub threshold: f64,
}

/// System guard configuration
#[derive(Debug, Clone)]
pub struct GuardConfig {
    // Memory guards
    pub max_memory_mb: usize,
    pub memory_warning_threshold_percent: f64,
    
    // Request guards
    pub max_requests_per_second: usize,
    pub max_queue_depth: usize,
    pub request_timeout_secs: u64,
    
    // Worker guards
    pub max_worker_idle_time_secs: u64,
    pub min_worker_success_rate: f64,
    
    // Cache guards
    pub min_cache_hit_rate: f64,
    pub max_cache_eviction_rate: f64,
    
    // GPU guards
    pub max_gpu_memory_percent: f64,
    pub max_gpu_utilization_percent: f64,
    
    // Error guards
    pub max_error_rate: f64,
    pub max_consecutive_errors: usize,
    
    // Circuit breaker
    pub circuit_breaker_threshold: usize,
    pub circuit_breaker_timeout_secs: u64,
    
    // Auto-mitigation
    pub enable_auto_mitigation: bool,
    pub enable_auto_scaling: bool,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 8192,
            memory_warning_threshold_percent: 80.0,
            max_requests_per_second: 1000,
            max_queue_depth: 500,
            request_timeout_secs: 30,
            max_worker_idle_time_secs: 600,
            min_worker_success_rate: 95.0,
            min_cache_hit_rate: 60.0,
            max_cache_eviction_rate: 10.0,
            max_gpu_memory_percent: 90.0,
            max_gpu_utilization_percent: 98.0,
            max_error_rate: 5.0,
            max_consecutive_errors: 10,
            circuit_breaker_threshold: 50,
            circuit_breaker_timeout_secs: 60,
            enable_auto_mitigation: true,
            enable_auto_scaling: true,
        }
    }
}

/// System guard manager
pub struct SystemGuard {
    config: GuardConfig,
    
    // Violation tracking
    violations: Arc<RwLock<Vec<GuardViolation>>>,
    total_violations: AtomicU64,
    critical_violations: AtomicU64,
    
    // Request tracking
    request_count: AtomicU64,
    request_window_start: Arc<RwLock<Instant>>,
    consecutive_errors: AtomicU64,
    
    // Circuit breaker
    circuit_open: AtomicBool,
    circuit_open_time: Arc<RwLock<Option<Instant>>>,
    circuit_failures: AtomicU64,
    
    // Mitigation actions
    mitigation_enabled: AtomicBool,
    auto_scale_enabled: AtomicBool,
    
    // Guard states
    guard_states: Arc<RwLock<HashMap<String, bool>>>,
}

impl SystemGuard {
    pub fn new(config: GuardConfig) -> Self {
        let enable_auto_mitigation = config.enable_auto_mitigation;
        let enable_auto_scaling = config.enable_auto_scaling;
        
        Self {
            config,
            violations: Arc::new(RwLock::new(Vec::new())),
            total_violations: AtomicU64::new(0),
            critical_violations: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
            request_window_start: Arc::new(RwLock::new(Instant::now())),
            consecutive_errors: AtomicU64::new(0),
            circuit_open: AtomicBool::new(false),
            circuit_open_time: Arc::new(RwLock::new(None)),
            circuit_failures: AtomicU64::new(0),
            mitigation_enabled: AtomicBool::new(enable_auto_mitigation),
            auto_scale_enabled: AtomicBool::new(enable_auto_scaling),
            guard_states: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Check memory usage
    pub async fn check_memory(&self, current_memory_mb: usize) -> Result<(), GuardViolation> {
        let usage_percent = (current_memory_mb as f64 / self.config.max_memory_mb as f64) * 100.0;
        
        if current_memory_mb > self.config.max_memory_mb {
            let violation = GuardViolation {
                guard_name: "memory_limit".to_string(),
                severity: ViolationSeverity::Critical,
                message: format!("Memory usage {} MB exceeds limit {} MB", 
                    current_memory_mb, self.config.max_memory_mb),
                timestamp: Instant::now(),
                value: current_memory_mb as f64,
                threshold: self.config.max_memory_mb as f64,
            };
            
            self.record_violation(violation.clone()).await;
            return Err(violation);
        }
        
        if usage_percent > self.config.memory_warning_threshold_percent {
            let violation = GuardViolation {
                guard_name: "memory_warning".to_string(),
                severity: ViolationSeverity::Medium,
                message: format!("Memory usage {:.1}% exceeds warning threshold {:.1}%", 
                    usage_percent, self.config.memory_warning_threshold_percent),
                timestamp: Instant::now(),
                value: usage_percent,
                threshold: self.config.memory_warning_threshold_percent,
            };
            
            warn!("{}", violation.message);
            self.record_violation(violation.clone()).await;
        }
        
        Ok(())
    }
    
    /// Check request rate
    pub async fn check_request_rate(&self) -> Result<(), GuardViolation> {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        
        let mut window_start = self.request_window_start.write().await;
        let elapsed = window_start.elapsed().as_secs();
        
        if elapsed >= 1 {
            let count = self.request_count.swap(0, Ordering::Relaxed);
            *window_start = Instant::now();
            
            if count > self.config.max_requests_per_second as u64 {
                let violation = GuardViolation {
                    guard_name: "request_rate".to_string(),
                    severity: ViolationSeverity::High,
                    message: format!("Request rate {} req/s exceeds limit {} req/s", 
                        count, self.config.max_requests_per_second),
                    timestamp: Instant::now(),
                    value: count as f64,
                    threshold: self.config.max_requests_per_second as f64,
                };
                
                self.record_violation(violation.clone()).await;
                return Err(violation);
            }
        }
        
        Ok(())
    }
    
    /// Check queue depth
    pub async fn check_queue_depth(&self, queue_depth: usize) -> Result<(), GuardViolation> {
        if queue_depth > self.config.max_queue_depth {
            let violation = GuardViolation {
                guard_name: "queue_depth".to_string(),
                severity: ViolationSeverity::High,
                message: format!("Queue depth {} exceeds limit {}", 
                    queue_depth, self.config.max_queue_depth),
                timestamp: Instant::now(),
                value: queue_depth as f64,
                threshold: self.config.max_queue_depth as f64,
            };
            
            self.record_violation(violation.clone()).await;
            return Err(violation);
        }
        
        Ok(())
    }
    
    /// Check worker success rate
    pub async fn check_worker_success_rate(&self, success_rate: f64) -> Result<(), GuardViolation> {
        if success_rate < self.config.min_worker_success_rate {
            let violation = GuardViolation {
                guard_name: "worker_success_rate".to_string(),
                severity: ViolationSeverity::High,
                message: format!("Worker success rate {:.2}% below minimum {:.2}%", 
                    success_rate, self.config.min_worker_success_rate),
                timestamp: Instant::now(),
                value: success_rate,
                threshold: self.config.min_worker_success_rate,
            };
            
            self.record_violation(violation.clone()).await;
            return Err(violation);
        }
        
        Ok(())
    }
    
    /// Check cache hit rate
    pub async fn check_cache_hit_rate(&self, hit_rate: f64) -> Result<(), GuardViolation> {
        if hit_rate < self.config.min_cache_hit_rate {
            let violation = GuardViolation {
                guard_name: "cache_hit_rate".to_string(),
                severity: ViolationSeverity::Medium,
                message: format!("Cache hit rate {:.2}% below minimum {:.2}%", 
                    hit_rate, self.config.min_cache_hit_rate),
                timestamp: Instant::now(),
                value: hit_rate,
                threshold: self.config.min_cache_hit_rate,
            };
            
            warn!("{}", violation.message);
            self.record_violation(violation.clone()).await;
        }
        
        Ok(())
    }
    
    /// Check GPU utilization
    pub async fn check_gpu_utilization(&self, gpu_memory_percent: f64, gpu_util_percent: f64) -> Result<(), GuardViolation> {
        if gpu_memory_percent > self.config.max_gpu_memory_percent {
            let violation = GuardViolation {
                guard_name: "gpu_memory".to_string(),
                severity: ViolationSeverity::Critical,
                message: format!("GPU memory {:.1}% exceeds limit {:.1}%", 
                    gpu_memory_percent, self.config.max_gpu_memory_percent),
                timestamp: Instant::now(),
                value: gpu_memory_percent,
                threshold: self.config.max_gpu_memory_percent,
            };
            
            self.record_violation(violation.clone()).await;
            return Err(violation);
        }
        
        if gpu_util_percent > self.config.max_gpu_utilization_percent {
            debug!("GPU utilization high: {:.1}%", gpu_util_percent);
        }
        
        Ok(())
    }
    
    /// Record error
    pub async fn record_error(&self) -> Result<(), GuardViolation> {
        let consecutive = self.consecutive_errors.fetch_add(1, Ordering::Relaxed) + 1;
        
        if consecutive > self.config.max_consecutive_errors as u64 {
            let violation = GuardViolation {
                guard_name: "consecutive_errors".to_string(),
                severity: ViolationSeverity::Critical,
                message: format!("Consecutive errors {} exceeds limit {}", 
                    consecutive, self.config.max_consecutive_errors),
                timestamp: Instant::now(),
                value: consecutive as f64,
                threshold: self.config.max_consecutive_errors as f64,
            };
            
            self.record_violation(violation.clone()).await;
            self.trigger_circuit_breaker().await;
            
            return Err(violation);
        }
        
        Ok(())
    }
    
    /// Record success (resets error counter)
    pub fn record_success(&self) {
        self.consecutive_errors.store(0, Ordering::Relaxed);
    }
    
    /// Check if circuit breaker is open
    pub async fn is_circuit_open(&self) -> bool {
        if !self.circuit_open.load(Ordering::Relaxed) {
            return false;
        }
        
        // Check if timeout has elapsed
        if let Some(open_time) = *self.circuit_open_time.read().await {
            if open_time.elapsed().as_secs() > self.config.circuit_breaker_timeout_secs {
                self.reset_circuit_breaker().await;
                return false;
            }
        }
        
        true
    }
    
    /// Trigger circuit breaker
    async fn trigger_circuit_breaker(&self) {
        let failures = self.circuit_failures.fetch_add(1, Ordering::Relaxed) + 1;
        
        if failures >= self.config.circuit_breaker_threshold as u64 {
            self.circuit_open.store(true, Ordering::Relaxed);
            *self.circuit_open_time.write().await = Some(Instant::now());
            
            error!("Circuit breaker OPENED after {} failures", failures);
            
            if self.mitigation_enabled.load(Ordering::Relaxed) {
                self.apply_mitigation().await;
            }
        }
    }
    
    /// Reset circuit breaker
    async fn reset_circuit_breaker(&self) {
        self.circuit_open.store(false, Ordering::Relaxed);
        *self.circuit_open_time.write().await = None;
        self.circuit_failures.store(0, Ordering::Relaxed);
        
        info!("Circuit breaker RESET");
    }
    
    /// Apply automatic mitigation
    async fn apply_mitigation(&self) {
        warn!("Applying automatic mitigation strategies");
        
        // Clear violation history older than 5 minutes
        let mut violations = self.violations.write().await;
        violations.retain(|v| v.timestamp.elapsed().as_secs() < 300);
        
        // Additional mitigation strategies can be added here
        // e.g., reduce worker count, clear caches, etc.
    }
    
    /// Record a violation
    async fn record_violation(&self, violation: GuardViolation) {
        if violation.severity == ViolationSeverity::Critical {
            self.critical_violations.fetch_add(1, Ordering::Relaxed);
            error!("CRITICAL VIOLATION: {}", violation.message);
        }
        
        self.total_violations.fetch_add(1, Ordering::Relaxed);
        
        let mut violations = self.violations.write().await;
        violations.push(violation);
        
        // Keep only last 1000 violations
        if violations.len() > 1000 {
            let excess = violations.len() - 1000;
            violations.drain(0..excess);
        }
    }
    
    /// Get recent violations
    pub async fn get_recent_violations(&self, limit: usize) -> Vec<GuardViolation> {
        let violations = self.violations.read().await;
        violations.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    /// Get violation statistics
    pub async fn get_violation_stats(&self) -> ViolationStats {
        let violations = self.violations.read().await;
        
        let by_severity: HashMap<ViolationSeverity, usize> = violations.iter()
            .fold(HashMap::new(), |mut acc, v| {
                *acc.entry(v.severity).or_insert(0) += 1;
                acc
            });
        
        let by_guard: HashMap<String, usize> = violations.iter()
            .fold(HashMap::new(), |mut acc, v| {
                *acc.entry(v.guard_name.clone()).or_insert(0) += 1;
                acc
            });
        
        ViolationStats {
            total_violations: self.total_violations.load(Ordering::Relaxed),
            critical_violations: self.critical_violations.load(Ordering::Relaxed),
            recent_violations: violations.len(),
            by_severity,
            by_guard,
            circuit_breaker_open: self.circuit_open.load(Ordering::Relaxed),
            circuit_failures: self.circuit_failures.load(Ordering::Relaxed),
        }
    }
    
    /// Enable/disable specific guard
    pub async fn set_guard_enabled(&self, guard_name: &str, enabled: bool) {
        let mut states = self.guard_states.write().await;
        states.insert(guard_name.to_string(), enabled);
        
        info!("Guard '{}' {}", guard_name, if enabled { "enabled" } else { "disabled" });
    }
    
    /// Check if guard is enabled
    pub async fn is_guard_enabled(&self, guard_name: &str) -> bool {
        let states = self.guard_states.read().await;
        *states.get(guard_name).unwrap_or(&true)
    }
    
    /// Get guard statistics
    pub async fn get_stats(&self) -> GuardStats {
        let violation_stats = self.get_violation_stats().await;
        let guard_states = self.guard_states.read().await.clone();
        
        GuardStats {
            violations: violation_stats,
            mitigation_enabled: self.mitigation_enabled.load(Ordering::Relaxed),
            auto_scale_enabled: self.auto_scale_enabled.load(Ordering::Relaxed),
            guard_states,
            config: self.config.clone(),
        }
    }
    
    /// Reset all statistics
    pub async fn reset_stats(&self) {
        self.total_violations.store(0, Ordering::Relaxed);
        self.critical_violations.store(0, Ordering::Relaxed);
        self.consecutive_errors.store(0, Ordering::Relaxed);
        self.circuit_failures.store(0, Ordering::Relaxed);
        
        let mut violations = self.violations.write().await;
        violations.clear();
        
        info!("Guard statistics reset");
    }
}

#[derive(Debug, Clone)]
pub struct ViolationStats {
    pub total_violations: u64,
    pub critical_violations: u64,
    pub recent_violations: usize,
    pub by_severity: HashMap<ViolationSeverity, usize>,
    pub by_guard: HashMap<String, usize>,
    pub circuit_breaker_open: bool,
    pub circuit_failures: u64,
}

#[derive(Debug, Clone)]
pub struct GuardStats {
    pub violations: ViolationStats,
    pub mitigation_enabled: bool,
    pub auto_scale_enabled: bool,
    pub guard_states: HashMap<String, bool>,
    pub config: GuardConfig,
}

impl Default for SystemGuard {
    fn default() -> Self {
        Self::new(GuardConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_guard() {
        let config = GuardConfig {
            max_memory_mb: 1000,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);
        
        assert!(guard.check_memory(500).await.is_ok());
        assert!(guard.check_memory(1001).await.is_err());
    }

    #[tokio::test]
    async fn test_queue_depth_guard() {
        let config = GuardConfig {
            max_queue_depth: 100,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);
        
        assert!(guard.check_queue_depth(50).await.is_ok());
        assert!(guard.check_queue_depth(101).await.is_err());
    }

    #[tokio::test]
    async fn test_consecutive_errors() {
        let config = GuardConfig {
            max_consecutive_errors: 3,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);
        
        assert!(guard.record_error().await.is_ok());
        assert!(guard.record_error().await.is_ok());
        assert!(guard.record_error().await.is_ok());
        assert!(guard.record_error().await.is_err());
    }

    #[tokio::test]
    async fn test_error_reset() {
        let guard = SystemGuard::default();
        
        guard.record_error().await.ok();
        guard.record_error().await.ok();
        guard.record_success();
        
        assert_eq!(guard.consecutive_errors.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = GuardConfig {
            circuit_breaker_threshold: 2,
            max_consecutive_errors: 1,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);
        
        guard.record_error().await.ok();
        guard.record_error().await.ok();
        guard.record_error().await.ok(); // Need 3 calls: 2nd exceeds consecutive, 3rd reaches circuit threshold
        
        assert!(guard.is_circuit_open().await);
    }

    #[tokio::test]
    async fn test_violation_tracking() {
        let guard = SystemGuard::default();
        
        guard.check_memory(10000).await.ok();
        guard.check_queue_depth(1000).await.ok();
        
        let stats = guard.get_violation_stats().await;
        assert!(stats.total_violations >= 2);
    }

    #[tokio::test]
    async fn test_guard_enable_disable() {
        let guard = SystemGuard::default();
        
        guard.set_guard_enabled("memory_limit", false).await;
        assert!(!guard.is_guard_enabled("memory_limit").await);
        
        guard.set_guard_enabled("memory_limit", true).await;
        assert!(guard.is_guard_enabled("memory_limit").await);
    }

    #[tokio::test]
    async fn test_violation_history() {
        let guard = SystemGuard::default();
        
        guard.check_memory(10000).await.ok();
        guard.check_queue_depth(1000).await.ok();
        
        let violations = guard.get_recent_violations(10).await;
        assert_eq!(violations.len(), 2);
    }

    #[tokio::test]
    async fn test_stats_reset() {
        let guard = SystemGuard::default();
        
        guard.check_memory(10000).await.ok();
        guard.reset_stats().await;
        
        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.total_violations, 0);
    }

    #[tokio::test]
    async fn test_cache_hit_rate_guard() {
        let config = GuardConfig {
            min_cache_hit_rate: 70.0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        assert!(guard.check_cache_hit_rate(80.0).await.is_ok());
        guard.check_cache_hit_rate(50.0).await.ok();

        let stats = guard.get_violation_stats().await;
        assert!(stats.total_violations > 0);
    }

    // ---- NEW TESTS ----

    #[tokio::test]
    async fn test_cache_hit_rate_above_threshold_no_violation() {
        let config = GuardConfig {
            min_cache_hit_rate: 60.0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        assert!(guard.check_cache_hit_rate(75.0).await.is_ok());
        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.total_violations, 0);
    }

    #[tokio::test]
    async fn test_cache_hit_rate_exactly_at_threshold_no_violation() {
        let config = GuardConfig {
            min_cache_hit_rate: 60.0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // exactly at threshold — NOT below, so no violation
        assert!(guard.check_cache_hit_rate(60.0).await.is_ok());
        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.total_violations, 0);
    }

    #[tokio::test]
    async fn test_worker_success_rate_violation() {
        let config = GuardConfig {
            min_worker_success_rate: 95.0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        let result = guard.check_worker_success_rate(80.0).await;
        assert!(result.is_err());
        let violation = result.unwrap_err();
        assert_eq!(violation.guard_name, "worker_success_rate");
        assert_eq!(violation.severity, ViolationSeverity::High);
        assert!((violation.value - 80.0).abs() < 1e-6);
        assert!((violation.threshold - 95.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_worker_success_rate_ok() {
        let config = GuardConfig {
            min_worker_success_rate: 95.0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        assert!(guard.check_worker_success_rate(100.0).await.is_ok());
        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.total_violations, 0);
    }

    #[tokio::test]
    async fn test_gpu_memory_violation() {
        let config = GuardConfig {
            max_gpu_memory_percent: 90.0,
            max_gpu_utilization_percent: 98.0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        let result = guard.check_gpu_utilization(95.0, 50.0).await;
        assert!(result.is_err());
        let violation = result.unwrap_err();
        assert_eq!(violation.guard_name, "gpu_memory");
        assert_eq!(violation.severity, ViolationSeverity::Critical);
    }

    #[tokio::test]
    async fn test_gpu_utilization_high_but_memory_ok() {
        let config = GuardConfig {
            max_gpu_memory_percent: 90.0,
            max_gpu_utilization_percent: 98.0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // Memory is fine, utilization is above limit — only a debug log, not an error
        let result = guard.check_gpu_utilization(50.0, 99.0).await;
        assert!(result.is_ok());
        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.total_violations, 0);
    }

    #[tokio::test]
    async fn test_gpu_utilization_both_ok() {
        let guard = SystemGuard::default();
        assert!(guard.check_gpu_utilization(50.0, 50.0).await.is_ok());
    }

    #[tokio::test]
    async fn test_memory_warning_threshold() {
        let config = GuardConfig {
            max_memory_mb: 1000,
            memory_warning_threshold_percent: 80.0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // 850 MB = 85% > 80% warning threshold but < 1000 MB limit
        let result = guard.check_memory(850).await;
        assert!(result.is_ok()); // returns Ok even for warning
        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.total_violations, 1);
        assert_eq!(stats.recent_violations, 1);
        // The warning violation is Medium severity
        assert_eq!(stats.by_severity.get(&ViolationSeverity::Medium).copied().unwrap_or(0), 1);
    }

    #[tokio::test]
    async fn test_memory_below_warning_no_violation() {
        let config = GuardConfig {
            max_memory_mb: 1000,
            memory_warning_threshold_percent: 80.0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        assert!(guard.check_memory(500).await.is_ok());
        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.total_violations, 0);
    }

    #[tokio::test]
    async fn test_violation_stats_by_severity() {
        let config = GuardConfig {
            max_memory_mb: 100,
            max_queue_depth: 10,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // Critical violation (memory exceeds limit)
        guard.check_memory(200).await.ok();
        // High violation (queue depth exceeded)
        guard.check_queue_depth(20).await.ok();

        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.total_violations, 2);
        assert_eq!(stats.critical_violations, 1);
        assert_eq!(stats.by_severity.get(&ViolationSeverity::Critical).copied().unwrap_or(0), 1);
        assert_eq!(stats.by_severity.get(&ViolationSeverity::High).copied().unwrap_or(0), 1);
    }

    #[tokio::test]
    async fn test_violation_stats_by_guard_name() {
        let config = GuardConfig {
            max_queue_depth: 10,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        guard.check_queue_depth(20).await.ok();
        guard.check_queue_depth(30).await.ok();

        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.by_guard.get("queue_depth").copied().unwrap_or(0), 2);
    }

    #[tokio::test]
    async fn test_get_stats_full() {
        let config = GuardConfig {
            max_queue_depth: 10,
            enable_auto_mitigation: true,
            enable_auto_scaling: false,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        guard.check_queue_depth(50).await.ok();
        guard.set_guard_enabled("custom_guard", false).await;

        let stats = guard.get_stats().await;
        assert!(stats.violations.total_violations >= 1);
        assert!(stats.mitigation_enabled);
        assert!(!stats.auto_scale_enabled);
        assert_eq!(stats.guard_states.get("custom_guard"), Some(&false));
    }

    #[tokio::test]
    async fn test_is_guard_enabled_unknown_returns_true() {
        let guard = SystemGuard::default();
        // An unknown guard defaults to enabled (true)
        assert!(guard.is_guard_enabled("nonexistent_guard").await);
    }

    #[tokio::test]
    async fn test_guard_enable_disable_multiple() {
        let guard = SystemGuard::default();

        guard.set_guard_enabled("memory_limit", false).await;
        guard.set_guard_enabled("queue_depth", false).await;
        guard.set_guard_enabled("worker_rate", true).await;

        assert!(!guard.is_guard_enabled("memory_limit").await);
        assert!(!guard.is_guard_enabled("queue_depth").await);
        assert!(guard.is_guard_enabled("worker_rate").await);
    }

    #[tokio::test]
    async fn test_violation_history_limit() {
        let config = GuardConfig {
            max_queue_depth: 0, // every call violates
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // Insert 1005 violations to exceed the 1000-entry cap
        for _ in 0..1005 {
            guard.check_queue_depth(1).await.ok();
        }

        let violations = guard.violations.read().await;
        assert_eq!(violations.len(), 1000);
        drop(violations);

        let recent = guard.get_recent_violations(10).await;
        assert_eq!(recent.len(), 10);
    }

    #[tokio::test]
    async fn test_get_recent_violations_limit_respected() {
        let config = GuardConfig {
            max_queue_depth: 0,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        for _ in 0..5 {
            guard.check_queue_depth(1).await.ok();
        }

        let recent = guard.get_recent_violations(3).await;
        assert_eq!(recent.len(), 3);
    }

    #[tokio::test]
    async fn test_reset_stats_clears_violations_and_counts() {
        let config = GuardConfig {
            max_memory_mb: 100,
            max_queue_depth: 10,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        guard.check_memory(200).await.ok();  // critical
        guard.check_queue_depth(20).await.ok(); // high
        guard.record_error().await.ok();

        guard.reset_stats().await;

        let stats = guard.get_violation_stats().await;
        assert_eq!(stats.total_violations, 0);
        assert_eq!(stats.critical_violations, 0);
        assert_eq!(stats.recent_violations, 0);
        assert_eq!(guard.consecutive_errors.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_not_open_initially() {
        let guard = SystemGuard::default();
        assert!(!guard.is_circuit_open().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_stats_reflected() {
        let config = GuardConfig {
            circuit_breaker_threshold: 2,
            max_consecutive_errors: 1,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // Each call to record_error that exceeds consecutive limit calls trigger_circuit_breaker
        guard.record_error().await.ok(); // consecutive=1, ok
        guard.record_error().await.ok(); // consecutive=2 > 1, triggers circuit_breaker (failures=1)
        guard.record_error().await.ok(); // consecutive=3 > 1, triggers circuit_breaker (failures=2 >= threshold)

        let stats = guard.get_violation_stats().await;
        assert!(stats.circuit_breaker_open);
        assert!(stats.circuit_failures >= 2);
    }

    #[tokio::test]
    async fn test_violation_severity_low_and_medium() {
        // Verify Low and Medium severity variants exist and are distinct
        assert_ne!(ViolationSeverity::Low, ViolationSeverity::Medium);
        assert_ne!(ViolationSeverity::Medium, ViolationSeverity::High);
        assert_ne!(ViolationSeverity::High, ViolationSeverity::Critical);
    }

    #[tokio::test]
    async fn test_guard_config_default_values() {
        let config = GuardConfig::default();
        assert_eq!(config.max_memory_mb, 8192);
        assert_eq!(config.memory_warning_threshold_percent, 80.0);
        assert_eq!(config.max_requests_per_second, 1000);
        assert_eq!(config.max_queue_depth, 500);
        assert_eq!(config.request_timeout_secs, 30);
        assert_eq!(config.max_worker_idle_time_secs, 600);
        assert_eq!(config.min_worker_success_rate, 95.0);
        assert_eq!(config.min_cache_hit_rate, 60.0);
        assert_eq!(config.max_cache_eviction_rate, 10.0);
        assert_eq!(config.max_gpu_memory_percent, 90.0);
        assert_eq!(config.max_gpu_utilization_percent, 98.0);
        assert_eq!(config.max_error_rate, 5.0);
        assert_eq!(config.max_consecutive_errors, 10);
        assert_eq!(config.circuit_breaker_threshold, 50);
        assert_eq!(config.circuit_breaker_timeout_secs, 60);
        assert!(config.enable_auto_mitigation);
        assert!(config.enable_auto_scaling);
    }

    #[tokio::test]
    async fn test_system_guard_default() {
        let guard = SystemGuard::default();
        assert!(!guard.circuit_open.load(Ordering::Relaxed));
        assert_eq!(guard.total_violations.load(Ordering::Relaxed), 0);
        assert_eq!(guard.critical_violations.load(Ordering::Relaxed), 0);
        assert_eq!(guard.consecutive_errors.load(Ordering::Relaxed), 0);
        assert_eq!(guard.circuit_failures.load(Ordering::Relaxed), 0);
    }

    // ── reset_circuit_breaker direct coverage (lines 352-357) ────────────────
    //
    // reset_circuit_breaker is a private async fn on SystemGuard.  Tests in the
    // same file can call private methods directly.  We open the circuit manually
    // and then call reset_circuit_breaker() to exercise lines 352-357.
    //
    // Lines 327-328 (inside is_circuit_open) are skipped here because calling
    // is_circuit_open() when the timeout has elapsed would cause a deadlock:
    // is_circuit_open holds circuit_open_time.read() while reset_circuit_breaker
    // tries to acquire circuit_open_time.write().

    #[tokio::test]
    async fn test_reset_circuit_breaker_clears_all_state() {
        let config = GuardConfig {
            circuit_breaker_threshold: 100,
            max_consecutive_errors: 100,
            circuit_breaker_timeout_secs: 60,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // Manually open the circuit with a known state
        guard.circuit_open.store(true, Ordering::Relaxed);
        guard.circuit_failures.store(7, Ordering::Relaxed);
        {
            let mut t = guard.circuit_open_time.write().await;
            *t = Some(Instant::now());
        }

        // Pre-conditions
        assert!(guard.circuit_open.load(Ordering::Relaxed));
        assert_eq!(guard.circuit_failures.load(Ordering::Relaxed), 7);
        {
            let t = guard.circuit_open_time.read().await;
            assert!(t.is_some());
        }

        // Call reset_circuit_breaker directly — covers lines 352-357
        guard.reset_circuit_breaker().await;

        // Post-conditions: circuit is closed, failures zeroed, open_time cleared
        assert!(!guard.circuit_open.load(Ordering::Relaxed), "circuit_open should be false after reset");
        assert_eq!(guard.circuit_failures.load(Ordering::Relaxed), 0, "circuit_failures should be zero after reset");
        {
            let t = guard.circuit_open_time.read().await;
            assert!(t.is_none(), "circuit_open_time should be None after reset");
        }
    }

    #[tokio::test]
    async fn test_reset_circuit_breaker_idempotent_when_already_closed() {
        // Calling reset_circuit_breaker when circuit is already closed should be safe.
        let guard = SystemGuard::default();
        assert!(!guard.circuit_open.load(Ordering::Relaxed));

        // Reset on a closed circuit — should not panic and state remains correct
        guard.reset_circuit_breaker().await;

        assert!(!guard.circuit_open.load(Ordering::Relaxed));
        assert_eq!(guard.circuit_failures.load(Ordering::Relaxed), 0);
        let t = guard.circuit_open_time.read().await;
        assert!(t.is_none());
    }

    // NOTE: lines 327-328 (inside is_circuit_open when the timeout has elapsed)
    // are not covered because is_circuit_open holds a tokio RwLock read guard
    // on circuit_open_time across the if-let body, and reset_circuit_breaker
    // (called at line 327) tries to acquire a write lock on the same RwLock —
    // causing a deadlock.  This is a design limitation in the production code.

    // ── check_request_rate coverage (lines 176-202) ───────────────────────────

    /// Calling check_request_rate() once covers the basic path (lines 177-180, 202).
    /// The elapsed time is < 1 s so the inner if-block is NOT entered.
    #[tokio::test]
    async fn test_check_request_rate_ok_within_window() {
        let config = GuardConfig {
            max_requests_per_second: 100,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // First call: increments counter, checks elapsed (< 1s), returns Ok
        let result = guard.check_request_rate().await;
        assert!(result.is_ok());
        // Counter was incremented but NOT swapped yet (elapsed < 1s)
        assert_eq!(guard.request_count.load(Ordering::Relaxed), 1);
    }

    /// If we set the window_start to > 1 second ago AND the accumulated count
    /// is within the limit, the window resets and Ok is returned (lines 182-184, 186 false, 202).
    #[tokio::test]
    async fn test_check_request_rate_window_expired_count_within_limit() {
        let config = GuardConfig {
            max_requests_per_second: 100,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // Pre-load the counter with a count that does NOT exceed the limit
        guard.request_count.store(50, Ordering::Relaxed);

        // Wind the window start back by 2 seconds so elapsed >= 1
        {
            let mut ws = guard.request_window_start.write().await;
            *ws = Instant::now() - Duration::from_secs(2);
        }

        let result = guard.check_request_rate().await;
        assert!(result.is_ok(), "50 req/s <= 100 req/s limit should be Ok");

        // After the window reset the counter is swapped to 0 (line 183), then
        // the single fetch_add at line 177 already happened before the swap.
        // The swap returns the OLD value and sets count to 0.  No further
        // fetch_add occurs after the swap, so count == 0 post-call.
        assert_eq!(guard.request_count.load(Ordering::Relaxed), 0);
    }

    /// If we set the window_start to > 1 second ago AND the accumulated count
    /// exceeds the limit, a violation is returned (lines 182-198).
    #[tokio::test]
    async fn test_check_request_rate_exceeds_limit_returns_violation() {
        let config = GuardConfig {
            max_requests_per_second: 10,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        // Pre-load the counter so that after the window-expiry swap it reads 500
        // (swap returns the OLD value; the new fetch_add at line 177 adds 1 first,
        // then the swap at line 183 reads that + whatever was pre-stored).
        // We store 499 so fetch_add makes it 500, then swap reads 500 > 10.
        guard.request_count.store(499, Ordering::Relaxed);

        // Wind the window start back so elapsed >= 1
        {
            let mut ws = guard.request_window_start.write().await;
            *ws = Instant::now() - Duration::from_secs(2);
        }

        let result = guard.check_request_rate().await;
        assert!(result.is_err(), "500 req/s > 10 req/s limit should be an error");
        let violation = result.unwrap_err();
        assert_eq!(violation.guard_name, "request_rate");
        assert_eq!(violation.severity, ViolationSeverity::High);
        assert!(violation.value > 10.0);
        assert!((violation.threshold - 10.0).abs() < 1e-6);
    }

    /// Multiple calls within a fresh window all return Ok; only the first call
    /// that arrives after the 1-second window expires checks the rate.
    #[tokio::test]
    async fn test_check_request_rate_multiple_calls_within_window_all_ok() {
        let config = GuardConfig {
            max_requests_per_second: 1000,
            ..Default::default()
        };
        let guard = SystemGuard::new(config);

        for _ in 0..50 {
            assert!(guard.check_request_rate().await.is_ok());
        }
        // All 50 increments land in the same window, none exceed 1000 req/s
        assert_eq!(guard.request_count.load(Ordering::Relaxed), 50);
    }
}
