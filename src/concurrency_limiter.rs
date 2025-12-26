//! Concurrency limiter to prevent thread pool exhaustion
//! 
//! Provides bounded concurrency control using semaphores to:
//! - Prevent blocking thread pool exhaustion
//! - Maintain stable throughput under load
//! - Enable graceful degradation

use std::sync::Arc;
use tokio::sync::Semaphore;

/// Limits concurrent CPU-bound operations
#[derive(Clone)]
pub struct ConcurrencyLimiter {
    semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

impl ConcurrencyLimiter {
    /// Create a new concurrency limiter
    /// 
    /// # Arguments
    /// * `max_concurrent` - Maximum number of concurrent operations
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            max_concurrent,
        }
    }
    
    /// Execute a CPU-bound operation with concurrency limiting
    #[inline]
    pub async fn execute<F, T>(&self, f: F) -> T
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let _permit = self.semaphore.acquire().await.expect("Semaphore closed");
        tokio::task::spawn_blocking(f)
            .await
            .expect("Blocking task failed")
    }
    
    /// Try to execute without waiting if capacity available
    /// 
    /// # Returns
    /// Some(result) if permit was available, None otherwise
    pub async fn try_execute<F, T>(&self, f: F) -> Option<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        // Try to acquire permit without waiting
        let permit = self.semaphore.try_acquire().ok()?;
        
        let result = tokio::task::spawn_blocking(f)
            .await
            .expect("Blocking task failed");
        
        drop(permit);
        
        Some(result)
    }
    
    /// Get number of available permits
    pub fn available(&self) -> usize {
        self.semaphore.available_permits()
    }
    
    /// Get maximum concurrent limit
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
    
    /// Get number of currently active operations
    pub fn active(&self) -> usize {
        self.max_concurrent - self.semaphore.available_permits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_limiter_basic() {
        let limiter = ConcurrencyLimiter::new(2);
        
        let result = limiter.execute(|| 42).await;
        assert_eq!(result, 42);
    }
    
    #[tokio::test]
    async fn test_limiter_concurrent() {
        let limiter = Arc::new(ConcurrencyLimiter::new(2));
        
        let mut tasks = Vec::new();
        
        for i in 0..4 {
            let limiter = Arc::clone(&limiter);
            tasks.push(tokio::spawn(async move {
                limiter.execute(move || {
                    std::thread::sleep(Duration::from_millis(100));
                    i
                }).await
            }));
        }
        
        let results: Vec<_> = futures::future::join_all(tasks)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        
        assert_eq!(results, vec![0, 1, 2, 3]);
    }
    
    #[tokio::test]
    async fn test_limiter_metrics() {
        let limiter = ConcurrencyLimiter::new(10);
        
        assert_eq!(limiter.max_concurrent(), 10);
        assert_eq!(limiter.available(), 10);
        assert_eq!(limiter.active(), 0);
    }
}
