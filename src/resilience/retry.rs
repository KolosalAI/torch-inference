use std::time::Duration;
use std::future::Future;
use std::pin::Pin;
use tokio::time::sleep;
use rand::Rng;
use tracing::{warn, debug};

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub multiplier: f64,
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryPolicy {
    pub fn new(max_retries: usize) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    pub fn with_delays(
        max_retries: usize,
        base_delay: Duration,
        max_delay: Duration,
    ) -> Self {
        Self {
            max_retries,
            base_delay,
            max_delay,
            ..Default::default()
        }
    }

    /// Execute a function with retry logic
    pub async fn execute<F, Fut, T, E>(&self, mut f: F) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Debug + Clone,
    {
        let mut attempt = 0;
        let mut last_error: Option<E> = None;

        loop {
            match f().await {
                Ok(result) => {
                    if attempt > 0 {
                        debug!(
                            "Operation succeeded after {} retries",
                            attempt
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    attempt += 1;
                    
                    if attempt > self.max_retries {
                        warn!(
                            "Max retries ({}) exceeded. Last error: {:?}",
                            self.max_retries, e
                        );
                        return Err(e);
                    }

                    let delay = self.calculate_delay(attempt);
                    warn!(
                        "Retry attempt {}/{} after {:?}. Error: {:?}",
                        attempt, self.max_retries, delay, e
                    );

                    last_error = Some(e);
                    sleep(delay).await;
                }
            }
        }
    }

    /// Execute with a predicate to determine if error is retryable
    pub async fn execute_if<F, Fut, T, E, P>(&self, mut f: F, is_retryable: P) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Debug + Clone,
        P: Fn(&E) -> bool,
    {
        let mut attempt = 0;

        loop {
            match f().await {
                Ok(result) => {
                    if attempt > 0 {
                        debug!("Operation succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    if !is_retryable(&e) {
                        debug!("Error is not retryable: {:?}", e);
                        return Err(e);
                    }

                    attempt += 1;
                    
                    if attempt > self.max_retries {
                        warn!(
                            "Max retries ({}) exceeded. Last error: {:?}",
                            self.max_retries, e
                        );
                        return Err(e);
                    }

                    let delay = self.calculate_delay(attempt);
                    warn!(
                        "Retry attempt {}/{} after {:?}. Error: {:?}",
                        attempt, self.max_retries, delay, e
                    );

                    sleep(delay).await;
                }
            }
        }
    }

    fn calculate_delay(&self, attempt: usize) -> Duration {
        let base_ms = self.base_delay.as_millis() as f64;
        let delay_ms = base_ms * self.multiplier.powi(attempt as i32 - 1);
        let capped_ms = delay_ms.min(self.max_delay.as_millis() as f64);

        let final_ms = if self.jitter {
            let mut rng = rand::thread_rng();
            let jitter = rng.gen_range(-0.15..=0.15); // ±15% jitter
            (capped_ms * (1.0 + jitter)).max(0.0)
        } else {
            capped_ms
        };

        Duration::from_millis(final_ms as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_retry_success_on_first_attempt() {
        let policy = RetryPolicy::new(3);
        let result = policy.execute(|| async { Ok::<_, String>(42) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let policy = RetryPolicy::new(3);
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_clone = attempts.clone();

        let result = policy
            .execute(|| {
                let attempts = attempts_clone.clone();
                async move {
                    let count = attempts.fetch_add(1, Ordering::SeqCst);
                    if count < 2 {
                        Err("fail")
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result, Ok(42));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_max_retries_exceeded() {
        let policy = RetryPolicy::new(2);
        let result = policy.execute(|| async { Err::<i32, _>("fail") }).await;
        assert_eq!(result, Err("fail"));
    }

    #[tokio::test]
    async fn test_retry_with_predicate() {
        let policy = RetryPolicy::new(3);
        
        // Should not retry on non-retryable error
        let result = policy
            .execute_if(
                || async { Err::<i32, _>("non-retryable") },
                |e| *e == "retryable",
            )
            .await;
        
        assert_eq!(result, Err("non-retryable"));
    }

    #[test]
    fn test_delay_calculation() {
        let policy = RetryPolicy {
            max_retries: 5,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
            jitter: false,
        };

        let delay1 = policy.calculate_delay(1);
        let delay2 = policy.calculate_delay(2);
        let delay3 = policy.calculate_delay(3);

        assert_eq!(delay1, Duration::from_millis(100));
        assert_eq!(delay2, Duration::from_millis(200));
        assert_eq!(delay3, Duration::from_millis(400));
    }

    #[test]
    fn test_delay_capping() {
        let policy = RetryPolicy {
            max_retries: 10,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            multiplier: 2.0,
            jitter: false,
        };

        let delay = policy.calculate_delay(10);
        assert!(delay <= Duration::from_secs(5));
    }
}
