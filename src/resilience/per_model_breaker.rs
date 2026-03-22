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
