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
