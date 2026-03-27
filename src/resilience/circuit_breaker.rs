use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use log::{warn, info};

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    state: std::sync::Mutex<CircuitState>,
    failure_count: std::sync::atomic::AtomicU32,
    success_count: std::sync::atomic::AtomicU32,
    last_failure_time: std::sync::Mutex<Option<Instant>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: std::sync::Mutex::new(CircuitState::Closed),
            failure_count: std::sync::atomic::AtomicU32::new(0),
            success_count: std::sync::atomic::AtomicU32::new(0),
            last_failure_time: std::sync::Mutex::new(None),
            config,
        }
    }

    pub fn call<F, T>(&self, f: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
    {
        let mut state = self.state.lock().unwrap();

        match *state {
            CircuitState::Open => {
                let should_retry = self.last_failure_time
                    .lock()
                    .unwrap()
                    .map(|t| t.elapsed() >= self.config.timeout)
                    .unwrap_or(false);

                if should_retry {
                    info!("Circuit breaker transitioning to HalfOpen");
                    *state = CircuitState::HalfOpen;
                } else {
                    return Err("Circuit breaker is open".to_string());
                }
            }
            _ => {}
        }

        match f() {
            Ok(result) => {
                self.on_success(&mut state);
                Ok(result)
            }
            Err(e) => {
                self.on_failure(&mut state);
                Err(e)
            }
        }
    }

    fn on_success(&self, state: &mut CircuitState) {
        self.failure_count.store(0, Ordering::SeqCst);

        match *state {
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                if success_count >= self.config.success_threshold {
                    info!("Circuit breaker transitioning to Closed");
                    *state = CircuitState::Closed;
                    self.success_count.store(0, Ordering::SeqCst);
                }
            }
            CircuitState::Closed => {
                self.success_count.store(0, Ordering::SeqCst);
            }
            _ => {}
        }
    }

    fn on_failure(&self, state: &mut CircuitState) {
        self.success_count.store(0, Ordering::SeqCst);
        *self.last_failure_time.lock().unwrap() = Some(Instant::now());

        let failure_count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;

        match *state {
            CircuitState::Closed => {
                if failure_count >= self.config.failure_threshold {
                    warn!("Circuit breaker transitioning to Open");
                    *state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                warn!("Circuit breaker transitioning to Open (HalfOpen failure)");
                *state = CircuitState::Open;
            }
            _ => {}
        }
    }

    pub fn get_state(&self) -> CircuitState {
        *self.state.lock().unwrap()
    }

    pub fn reset(&self) {
        *self.state.lock().unwrap() = CircuitState::Closed;
        self.failure_count.store(0, Ordering::SeqCst);
        self.success_count.store(0, Ordering::SeqCst);
        *self.last_failure_time.lock().unwrap() = None;
        info!("Circuit breaker reset");
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_initial_state() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_successful_call() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let result = cb.call(|| Ok("success"));
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_failed_call() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        let result: Result<String, String> = cb.call(|| Err("error".to_string()));
        
        assert!(result.is_err());
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_open_after_threshold() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        });
        
        for _ in 0..3 {
            let _ = cb.call(|| Err::<(), String>("error".to_string()));
        }
        
        assert_eq!(cb.get_state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_reject_when_open() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        });
        
        // Trigger open state
        for _ in 0..2 {
            let _ = cb.call(|| Err::<(), String>("error".to_string()));
        }
        
        // Next call should be rejected
        let result = cb.call(|| Ok("should not run"));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Circuit breaker is open");
    }

    #[test]
    fn test_circuit_breaker_half_open_transition() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
        });
        
        // Trigger open state
        for _ in 0..2 {
            let _ = cb.call(|| Err::<(), String>("error".to_string()));
        }
        
        assert_eq!(cb.get_state(), CircuitState::Open);
        
        // Wait for timeout
        std::thread::sleep(Duration::from_millis(150));
        
        // Should transition to HalfOpen
        let _ = cb.call(|| Ok("test"));
        assert_ne!(cb.get_state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_half_open_to_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
        });
        
        // Trigger open
        for _ in 0..2 {
            let _ = cb.call(|| Err::<(), String>("error".to_string()));
        }
        
        // Wait and recover
        std::thread::sleep(Duration::from_millis(150));
        
        // Successful calls to close
        let _ = cb.call(|| Ok("success"));
        let _ = cb.call(|| Ok("success"));
        
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        });
        
        // Trigger open state
        for _ in 0..2 {
            let _ = cb.call(|| Err::<(), String>("error".to_string()));
        }
        
        assert_eq!(cb.get_state(), CircuitState::Open);
        
        cb.reset();
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_config_default() {
        let config = CircuitBreakerConfig::default();
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 2);
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    // ── Additional gap-closing tests ──────────────────────────────────────────

    /// Exercises on_success in HalfOpen when success_count has NOT yet reached
    /// success_threshold (lines 79-84: the inner `if` is false on first call).
    #[test]
    fn test_circuit_breaker_half_open_partial_recovery_stays_half_open() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 3, // need 3 successes to close
            timeout: Duration::from_millis(50),
        });

        // Open the circuit
        for _ in 0..2 {
            let _ = cb.call(|| Err::<(), String>("err".to_string()));
        }
        assert_eq!(cb.get_state(), CircuitState::Open);

        // Wait for timeout so it transitions to HalfOpen on first call
        std::thread::sleep(Duration::from_millis(100));

        // First success: transitions to HalfOpen, success_count becomes 1 (< 3)
        let _ = cb.call(|| Ok("ok"));
        // State should be HalfOpen (not yet Closed)
        assert_eq!(cb.get_state(), CircuitState::HalfOpen);

        // Second success: success_count becomes 2 (< 3) — still HalfOpen
        let _ = cb.call(|| Ok("ok"));
        assert_eq!(cb.get_state(), CircuitState::HalfOpen);

        // Third success: success_count becomes 3 (>= 3) — now Closed
        let _ = cb.call(|| Ok("ok"));
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    /// Exercises on_failure when state is HalfOpen (lines 106-109).
    #[test]
    fn test_circuit_breaker_half_open_failure_reopens() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(50),
        });

        // Open the circuit
        for _ in 0..2 {
            let _ = cb.call(|| Err::<(), String>("err".to_string()));
        }
        assert_eq!(cb.get_state(), CircuitState::Open);

        // Wait for timeout to allow HalfOpen transition
        std::thread::sleep(Duration::from_millis(100));

        // A failure while HalfOpen should re-open the circuit
        let _ = cb.call(|| Err::<(), String>("fail in half-open".to_string()));
        assert_eq!(cb.get_state(), CircuitState::Open);
    }

    /// Exercises on_success when the circuit is already Closed (line 86-88).
    /// Also verifies that multiple consecutive successful calls on a Closed
    /// circuit do not accidentally open it.
    #[test]
    fn test_circuit_breaker_success_in_closed_resets_counters() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
        });

        // A couple of failures, then a success — should stay Closed
        let _ = cb.call(|| Err::<(), String>("e".to_string()));
        let _ = cb.call(|| Err::<(), String>("e".to_string()));
        assert_eq!(cb.get_state(), CircuitState::Closed);

        // Success resets failure_count via on_success → Closed branch
        let result = cb.call(|| Ok("recovered"));
        assert!(result.is_ok());
        assert_eq!(cb.get_state(), CircuitState::Closed);

        // After reset, we need failure_threshold failures again to open
        for _ in 0..2 {
            let _ = cb.call(|| Err::<(), String>("after recovery".to_string()));
        }
        // Only 2 failures after the reset success; threshold is 3 → still Closed
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }

    /// Exercises the Open → still-open (timeout not elapsed) path more directly.
    #[test]
    fn test_circuit_breaker_open_rejects_before_timeout() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 1,
            timeout: Duration::from_secs(60), // very long
        });

        let _ = cb.call(|| Err::<(), String>("trigger open".to_string()));
        assert_eq!(cb.get_state(), CircuitState::Open);

        // Multiple calls while open — all rejected immediately
        for _ in 0..3 {
            let result = cb.call(|| Ok("should be blocked"));
            assert_eq!(result.unwrap_err(), "Circuit breaker is open");
        }
        assert_eq!(cb.get_state(), CircuitState::Open);
    }

    // ── Logger-enabled tests to cover log macro argument lines ────────────────
    //
    // The `log` crate's warn!/info! macros only evaluate their arguments when a
    // logger with a matching level is installed.  Line 45 is inside the Open arm
    // of call(); lines 89 and 110 are `_ => {}` catch-all arms (unreachable in
    // normal operation).  We install a TRACE logger so all reachable macro bodies
    // are evaluated.

    fn init_logger() {
        let _ = env_logger::Builder::new()
            .filter_level(log::LevelFilter::Trace)
            .is_test(true)
            .try_init();
    }

    /// Covers line 45 (let should_retry = ...) in the Open arm with the
    /// timeout-not-yet-elapsed path (should_retry = false → return Err).
    #[test]
    fn test_open_arm_timeout_not_elapsed_with_logger() {
        init_logger();
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 1,
            timeout: Duration::from_secs(60),
        });
        // Open the circuit
        let _ = cb.call(|| Err::<(), String>("e".to_string()));
        assert_eq!(cb.get_state(), CircuitState::Open);
        // Call while open: evaluates line 45 then returns the error (line 56)
        let result = cb.call(|| Ok("x"));
        assert!(result.is_err());
        assert_eq!(cb.get_state(), CircuitState::Open);
    }

    /// Covers line 45 (let should_retry = ...) with the timeout-elapsed path
    /// (should_retry = true → transition to HalfOpen).
    #[test]
    fn test_open_arm_timeout_elapsed_with_logger() {
        init_logger();
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 1,
            timeout: Duration::from_millis(10),
        });
        let _ = cb.call(|| Err::<(), String>("e".to_string()));
        assert_eq!(cb.get_state(), CircuitState::Open);
        // Wait for the timeout to elapse
        std::thread::sleep(Duration::from_millis(30));
        // Now line 45 evaluates to should_retry = true → transitions to HalfOpen
        let result = cb.call(|| Ok("recovered"));
        assert!(result.is_ok());
        assert_ne!(cb.get_state(), CircuitState::Open);
    }

    /// Exercises the warn! on transition to Open (line 102) and the warn! on
    /// HalfOpen failure (line 107) with an active logger so macro args are
    /// evaluated.  Also exercises info! in reset (line 123).
    #[test]
    fn test_transition_logs_with_logger() {
        init_logger();
        let cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout: Duration::from_millis(10),
        });
        // Trigger Open (evaluates warn! at line 102)
        let _ = cb.call(|| Err::<(), String>("a".to_string()));
        let _ = cb.call(|| Err::<(), String>("b".to_string()));
        assert_eq!(cb.get_state(), CircuitState::Open);

        std::thread::sleep(Duration::from_millis(30));
        // Failure in HalfOpen → re-open (evaluates warn! at line 107)
        let _ = cb.call(|| Err::<(), String>("c".to_string()));
        assert_eq!(cb.get_state(), CircuitState::Open);

        // Reset (evaluates info! at line 123)
        cb.reset();
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }
}
