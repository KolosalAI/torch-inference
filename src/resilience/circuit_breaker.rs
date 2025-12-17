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
}
