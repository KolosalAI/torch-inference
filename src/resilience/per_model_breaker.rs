#![allow(dead_code)]
use std::sync::Arc;
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use log::{warn, info};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,      // Normal operation
    Open,        // Failing, reject requests
    HalfOpen,    // Testing recovery
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub success_threshold: usize,
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        }
    }
}

pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            last_failure_time: Arc::new(RwLock::new(None)),
            config,
        }
    }
    
    pub async fn call<F, T, E>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: std::future::Future<Output = Result<T, E>>,
    {
        // Check circuit state
        let mut state = self.state.write().await;
        
        match *state {
            CircuitState::Open => {
                // Check if timeout expired
                let should_try_recovery = {
                    let last_failure = self.last_failure_time.read().await;
                    last_failure
                        .map(|t| t.elapsed() > self.config.timeout)
                        .unwrap_or(false)
                };
                
                if should_try_recovery {
                    info!("Circuit breaker entering half-open state");
                    *state = CircuitState::HalfOpen;
                    drop(state);
                } else {
                    return Err(CircuitBreakerError::Open);
                }
            }
            _ => {
                drop(state);
            }
        }
        
        // Execute the function
        match f.await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                Err(CircuitBreakerError::Inner(e))
            }
        }
    }
    
    async fn on_success(&self) {
        let mut state = self.state.write().await;
        
        match *state {
            CircuitState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.config.success_threshold {
                    info!("Circuit breaker recovered, closing");
                    *state = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::Relaxed);
                    self.success_count.store(0, Ordering::Relaxed);
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            _ => {}
        }
    }
    
    async fn on_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure_time.write().await = Some(Instant::now());
        
        if failures >= self.config.failure_threshold {
            let mut state = self.state.write().await;
            if *state != CircuitState::Open {
                warn!("Circuit breaker opening after {} failures", failures);
                *state = CircuitState::Open;
                self.success_count.store(0, Ordering::Relaxed);
            }
        }
    }
    
    pub async fn get_state(&self) -> CircuitState {
        *self.state.read().await
    }
}

#[derive(Debug)]
pub enum CircuitBreakerError<E> {
    Open,
    Inner(E),
}

impl<E: std::fmt::Display> std::fmt::Display for CircuitBreakerError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitBreakerError::Open => write!(f, "Circuit breaker is open"),
            CircuitBreakerError::Inner(e) => write!(f, "{}", e),
        }
    }
}

impl<E: std::error::Error> std::error::Error for CircuitBreakerError<E> {}

/// Registry for managing per-model circuit breakers
pub struct CircuitBreakerRegistry {
    breakers: DashMap<String, Arc<CircuitBreaker>>,
    default_config: CircuitBreakerConfig,
}

impl CircuitBreakerRegistry {
    pub fn new(default_config: CircuitBreakerConfig) -> Self {
        Self {
            breakers: DashMap::new(),
            default_config,
        }
    }
    
    pub fn get_or_create(&self, model_name: &str) -> Arc<CircuitBreaker> {
        self.breakers
            .entry(model_name.to_string())
            .or_insert_with(|| {
                Arc::new(CircuitBreaker::new(self.default_config.clone()))
            })
            .clone()
    }
    
    pub async fn get_all_states(&self) -> Vec<(String, CircuitState)> {
        let mut states = Vec::new();
        for entry in self.breakers.iter() {
            let state = entry.value().get_state().await;
            states.push((entry.key().clone(), state));
        }
        states
    }
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ── CircuitBreakerConfig ──────────────────────────────────────────────────

    #[test]
    fn test_config_default_values() {
        let cfg = CircuitBreakerConfig::default();
        assert_eq!(cfg.failure_threshold, 5);
        assert_eq!(cfg.success_threshold, 2);
        assert_eq!(cfg.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_config_clone() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 1,
            timeout: Duration::from_secs(10),
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.failure_threshold, 3);
        assert_eq!(cloned.success_threshold, 1);
        assert_eq!(cloned.timeout, Duration::from_secs(10));
    }

    // ── CircuitBreakerError ───────────────────────────────────────────────────

    #[test]
    fn test_error_open_display() {
        let err: CircuitBreakerError<String> = CircuitBreakerError::Open;
        assert_eq!(format!("{}", err), "Circuit breaker is open");
    }

    #[test]
    fn test_error_inner_display() {
        let err: CircuitBreakerError<String> = CircuitBreakerError::Inner("inner error".to_string());
        assert_eq!(format!("{}", err), "inner error");
    }

    #[test]
    fn test_error_debug() {
        let err: CircuitBreakerError<String> = CircuitBreakerError::Open;
        let s = format!("{:?}", err);
        assert!(s.contains("Open"));
    }

    // ── CircuitState ─────────────────────────────────────────────────────────

    #[test]
    fn test_circuit_state_equality() {
        assert_eq!(CircuitState::Closed, CircuitState::Closed);
        assert_eq!(CircuitState::Open, CircuitState::Open);
        assert_eq!(CircuitState::HalfOpen, CircuitState::HalfOpen);
        assert_ne!(CircuitState::Closed, CircuitState::Open);
    }

    #[test]
    fn test_circuit_state_copy() {
        let s = CircuitState::Closed;
        let s2 = s; // Copy
        assert_eq!(s, s2);
    }

    // ── CircuitBreaker ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_new_breaker_is_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_call_success_stays_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let result = cb.call(async { Ok::<_, String>("ok") }).await;
        assert!(result.is_ok());
        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_call_failure_increments_count() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        };
        let cb = CircuitBreaker::new(cfg);

        // One failure – still closed
        let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_breaker_opens_after_threshold() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        };
        let cb = CircuitBreaker::new(cfg);

        for _ in 0..3 {
            let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        }

        assert_eq!(cb.get_state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_open_breaker_rejects_calls() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        };
        let cb = CircuitBreaker::new(cfg);

        for _ in 0..2 {
            let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        }

        assert_eq!(cb.get_state().await, CircuitState::Open);

        // Should immediately return CircuitBreakerError::Open without executing f
        let result = cb.call(async { Ok::<(), String>(()) }).await;
        match result {
            Err(CircuitBreakerError::Open) => {}
            other => panic!("Expected Open error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_open_transitions_to_half_open_after_timeout() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(50),
        };
        let cb = CircuitBreaker::new(cfg);

        for _ in 0..2 {
            let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        }
        assert_eq!(cb.get_state().await, CircuitState::Open);

        tokio::time::sleep(Duration::from_millis(100)).await;

        // First call after timeout should attempt execution (HalfOpen)
        let _ = cb.call(async { Ok::<(), String>(()) }).await;
        // After a success it may close or stay half-open; either way it is no longer Open
        let state = cb.get_state().await;
        assert_ne!(state, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_half_open_closes_after_enough_successes() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(50),
        };
        let cb = CircuitBreaker::new(cfg);

        for _ in 0..2 {
            let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        }

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Two successes should close the breaker
        let _ = cb.call(async { Ok::<(), String>(()) }).await;
        let _ = cb.call(async { Ok::<(), String>(()) }).await;
        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_success_in_closed_resets_failure_count() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 1,
            timeout: Duration::from_secs(60),
        };
        let cb = CircuitBreaker::new(cfg);

        // Two failures (below threshold)
        let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        assert_eq!(cb.get_state().await, CircuitState::Closed);

        // A success should reset the failure counter
        let _ = cb.call(async { Ok::<(), String>(()) }).await;

        // Now we need 3 new failures to open
        let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        assert_eq!(cb.get_state().await, CircuitState::Closed);

        let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        assert_eq!(cb.get_state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_inner_error_is_propagated() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let result: Result<(), CircuitBreakerError<String>> =
            cb.call(async { Err("my inner error".to_string()) }).await;
        match result {
            Err(CircuitBreakerError::Inner(msg)) => assert_eq!(msg, "my inner error"),
            other => panic!("Expected Inner error, got {:?}", other),
        }
    }

    // ── CircuitBreakerRegistry ────────────────────────────────────────────────

    #[test]
    fn test_registry_default() {
        let _registry = CircuitBreakerRegistry::default();
    }

    #[test]
    fn test_registry_get_or_create_returns_same_breaker() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());
        let b1 = registry.get_or_create("model-a");
        let b2 = registry.get_or_create("model-a");
        // Same Arc – pointer equality
        assert!(Arc::ptr_eq(&b1, &b2));
    }

    #[test]
    fn test_registry_different_keys_are_independent() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());
        let b1 = registry.get_or_create("model-a");
        let b2 = registry.get_or_create("model-b");
        assert!(!Arc::ptr_eq(&b1, &b2));
    }

    #[tokio::test]
    async fn test_registry_get_all_states_empty() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());
        let states = registry.get_all_states().await;
        assert!(states.is_empty());
    }

    #[tokio::test]
    async fn test_registry_get_all_states_with_entries() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());
        registry.get_or_create("alpha");
        registry.get_or_create("beta");

        let states = registry.get_all_states().await;
        assert_eq!(states.len(), 2);

        let names: Vec<&str> = states.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));

        for (_, state) in &states {
            assert_eq!(*state, CircuitState::Closed);
        }
    }

    #[tokio::test]
    async fn test_registry_state_reflects_breaker_state() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        };
        let registry = CircuitBreakerRegistry::new(cfg);
        let breaker = registry.get_or_create("my-model");

        // Open the breaker
        let _ = breaker.call(async { Err::<(), String>("fail".to_string()) }).await;
        let _ = breaker.call(async { Err::<(), String>("fail".to_string()) }).await;

        let states = registry.get_all_states().await;
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].0, "my-model");
        assert_eq!(states[0].1, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_registry_custom_config_is_applied() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 1,
            timeout: Duration::from_secs(60),
        };
        let registry = CircuitBreakerRegistry::new(cfg);
        let breaker = registry.get_or_create("fast-fail");

        // Single failure should open the breaker
        let _ = breaker.call(async { Err::<(), String>("fail".to_string()) }).await;
        assert_eq!(breaker.get_state().await, CircuitState::Open);
    }

    // ── Dedicated gap-closing tests (lines 59, 111) ───────────────────────────

    /// Line 59 (CircuitState::Open arm in call()): Open state with timeout
    /// expired → should_try_recovery = true → transition to HalfOpen and call
    /// succeeds → circuit closes.
    #[tokio::test]
    async fn test_open_timeout_expired_halfopen_then_closes() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout: Duration::from_millis(1), // very short timeout
        };
        let cb = CircuitBreaker::new(cfg);

        // Open the circuit
        let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        let _ = cb.call(async { Err::<(), String>("fail".to_string()) }).await;
        assert_eq!(cb.get_state().await, CircuitState::Open);

        // Wait for timeout to elapse
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Line 59: Open arm entered, should_try_recovery = true
        // → state becomes HalfOpen, then call succeeds
        // → on_success HalfOpen arm with threshold=1 → Closed
        let result = cb.call(async { Ok::<(), String>(()) }).await;
        assert!(result.is_ok());
        assert_eq!(cb.get_state().await, CircuitState::Closed);
    }

    /// Line 111 (on_success `_ => {}` catch-all): on_success is called when the
    /// circuit is in Closed state.  The Closed arm (lines 107-110) resets the
    /// failure count; the `_ => {}` arm is the dead-code catch-all.
    /// We verify on_success in Closed resets failure_count so that subsequent
    /// failures still need the full threshold to open the circuit.
    #[tokio::test]
    async fn test_on_success_in_closed_state_resets_failure_count() {
        let cfg = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 1,
            timeout: Duration::from_secs(60),
        };
        let cb = CircuitBreaker::new(cfg);

        // Two failures (below threshold) — still Closed
        let _ = cb.call(async { Err::<(), String>("f1".to_string()) }).await;
        let _ = cb.call(async { Err::<(), String>("f2".to_string()) }).await;
        assert_eq!(cb.get_state().await, CircuitState::Closed);

        // A success while Closed → on_success Closed arm (line 107-110) runs,
        // resetting failure_count to 0.
        let r = cb.call(async { Ok::<(), String>(()) }).await;
        assert!(r.is_ok());
        assert_eq!(cb.get_state().await, CircuitState::Closed);

        // We now need 3 fresh failures to open (count was reset)
        let _ = cb.call(async { Err::<(), String>("g1".to_string()) }).await;
        let _ = cb.call(async { Err::<(), String>("g2".to_string()) }).await;
        assert_eq!(cb.get_state().await, CircuitState::Closed); // still closed

        let _ = cb.call(async { Err::<(), String>("g3".to_string()) }).await;
        assert_eq!(cb.get_state().await, CircuitState::Open);
    }
}
