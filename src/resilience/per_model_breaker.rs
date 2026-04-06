#![allow(dead_code)]
use dashmap::DashMap;
use log::{info, warn};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

static PER_MODEL_EPOCH: OnceLock<Instant> = OnceLock::new();
#[inline]
fn per_model_epoch() -> Instant {
    *PER_MODEL_EPOCH.get_or_init(Instant::now)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing, reject requests
    HalfOpen, // Testing recovery
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

/// Per-model async circuit breaker.
///
/// Internal state is protected by `parking_lot::Mutex` instead of
/// `Arc<tokio::sync::RwLock<_>>`.  Since no lock is ever held across an
/// `.await` point, a sync mutex is correct and eliminates the async future
/// overhead on every call to `on_success` / `on_failure` / `get_state`.
pub struct CircuitBreaker {
    state: Mutex<CircuitState>,
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    /// Nanos since PER_MODEL_EPOCH, or 0 = no recorded failure.
    /// AtomicU64 replaces Mutex<Option<Instant>> — no mutex on failure hot path.
    last_failure_nanos: AtomicU64,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        per_model_epoch(); // ensure epoch is set before any call
        Self {
            state: Mutex::new(CircuitState::Closed),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            last_failure_nanos: AtomicU64::new(0),
            config,
        }
    }

    pub async fn call<F, T, E>(&self, f: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: std::future::Future<Output = Result<T, E>>,
    {
        // Phase 1: check/transition state under a brief sync lock, then release.
        {
            let mut state = self.state.lock();
            if *state == CircuitState::Open {
                let nanos = self.last_failure_nanos.load(Ordering::Relaxed);
                let should_try_recovery = nanos > 0 && {
                    let last = per_model_epoch() + Duration::from_nanos(nanos);
                    last.elapsed() > self.config.timeout
                };

                if should_try_recovery {
                    info!("Circuit breaker entering half-open state");
                    *state = CircuitState::HalfOpen;
                } else {
                    return Err(CircuitBreakerError::Open);
                }
            }
        } // ← both locks released here, before the async call

        // Phase 2: execute `f` without holding any lock.
        match f.await {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                self.on_failure();
                Err(CircuitBreakerError::Inner(e))
            }
        }
    }

    fn on_success(&self) {
        let mut state = self.state.lock();
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

    fn on_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        let nanos = Instant::now()
            .checked_duration_since(per_model_epoch())
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;
        self.last_failure_nanos.store(nanos.max(1), Ordering::Relaxed);

        if failures >= self.config.failure_threshold {
            let mut state = self.state.lock();
            if *state != CircuitState::Open {
                warn!("Circuit breaker opening after {} failures", failures);
                *state = CircuitState::Open;
                self.success_count.store(0, Ordering::Relaxed);
            }
        }
    }

    pub fn get_state(&self) -> CircuitState {
        *self.state.lock()
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
            .or_insert_with(|| Arc::new(CircuitBreaker::new(self.default_config.clone())))
            .clone()
    }

    pub fn get_all_states(&self) -> Vec<(String, CircuitState)> {
        self.breakers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().get_state()))
            .collect()
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
        let err: CircuitBreakerError<String> =
            CircuitBreakerError::Inner("inner error".to_string());
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
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_call_success_stays_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let result = cb.call(async { Ok::<_, String>("ok") }).await;
        assert!(result.is_ok());
        assert_eq!(cb.get_state(), CircuitState::Closed);
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
        let _ = cb
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        assert_eq!(cb.get_state(), CircuitState::Closed);
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
            let _ = cb
                .call(async { Err::<(), String>("fail".to_string()) })
                .await;
        }

        assert_eq!(cb.get_state(), CircuitState::Open);
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
            let _ = cb
                .call(async { Err::<(), String>("fail".to_string()) })
                .await;
        }

        assert_eq!(cb.get_state(), CircuitState::Open);

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
            let _ = cb
                .call(async { Err::<(), String>("fail".to_string()) })
                .await;
        }
        assert_eq!(cb.get_state(), CircuitState::Open);

        tokio::time::sleep(Duration::from_millis(100)).await;

        // First call after timeout should attempt execution (HalfOpen)
        let _ = cb.call(async { Ok::<(), String>(()) }).await;
        // After a success it may close or stay half-open; either way it is no longer Open
        let state = cb.get_state();
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
            let _ = cb
                .call(async { Err::<(), String>("fail".to_string()) })
                .await;
        }

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Two successes should close the breaker
        let _ = cb.call(async { Ok::<(), String>(()) }).await;
        let _ = cb.call(async { Ok::<(), String>(()) }).await;
        assert_eq!(cb.get_state(), CircuitState::Closed);
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
        let _ = cb
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        let _ = cb
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        assert_eq!(cb.get_state(), CircuitState::Closed);

        // A success should reset the failure counter
        let _ = cb.call(async { Ok::<(), String>(()) }).await;

        // Now we need 3 new failures to open
        let _ = cb
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        let _ = cb
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        assert_eq!(cb.get_state(), CircuitState::Closed);

        let _ = cb
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        assert_eq!(cb.get_state(), CircuitState::Open);
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
        let states = registry.get_all_states();
        assert!(states.is_empty());
    }

    #[tokio::test]
    async fn test_registry_get_all_states_with_entries() {
        let registry = CircuitBreakerRegistry::new(CircuitBreakerConfig::default());
        registry.get_or_create("alpha");
        registry.get_or_create("beta");

        let states = registry.get_all_states();
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
        let _ = breaker
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        let _ = breaker
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;

        let states = registry.get_all_states();
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
        let _ = breaker
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        assert_eq!(breaker.get_state(), CircuitState::Open);
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
        let _ = cb
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        let _ = cb
            .call(async { Err::<(), String>("fail".to_string()) })
            .await;
        assert_eq!(cb.get_state(), CircuitState::Open);

        // Wait for timeout to elapse
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Line 59: Open arm entered, should_try_recovery = true
        // → state becomes HalfOpen, then call succeeds
        // → on_success HalfOpen arm with threshold=1 → Closed
        let result = cb.call(async { Ok::<(), String>(()) }).await;
        assert!(result.is_ok());
        assert_eq!(cb.get_state(), CircuitState::Closed);
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
        assert_eq!(cb.get_state(), CircuitState::Closed);

        // A success while Closed → on_success Closed arm (line 107-110) runs,
        // resetting failure_count to 0.
        let r = cb.call(async { Ok::<(), String>(()) }).await;
        assert!(r.is_ok());
        assert_eq!(cb.get_state(), CircuitState::Closed);

        // We now need 3 fresh failures to open (count was reset)
        let _ = cb.call(async { Err::<(), String>("g1".to_string()) }).await;
        let _ = cb.call(async { Err::<(), String>("g2".to_string()) }).await;
        assert_eq!(cb.get_state(), CircuitState::Closed); // still closed

        let _ = cb.call(async { Err::<(), String>("g3".to_string()) }).await;
        assert_eq!(cb.get_state(), CircuitState::Open);
    }
}
