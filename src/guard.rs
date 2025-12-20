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
}
