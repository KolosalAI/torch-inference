use std::sync::Arc;
use tokio::sync::Mutex;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use dashmap::DashMap;

/// Token bucket rate limiter
pub struct TokenBucket {
    capacity: f64,
    tokens: Arc<Mutex<f64>>,
    refill_rate: f64,  // tokens per second
    last_refill: Arc<Mutex<Instant>>,
}

impl TokenBucket {
    pub fn new(capacity: usize, refill_rate: f64) -> Self {
        Self {
            capacity: capacity as f64,
            tokens: Arc::new(Mutex::new(capacity as f64)),
            refill_rate,
            last_refill: Arc::new(Mutex::new(Instant::now())),
        }
    }

    pub async fn try_acquire(&self, tokens_needed: f64) -> bool {
        self.refill().await;

        let mut tokens = self.tokens.lock().await;
        if *tokens >= tokens_needed {
            *tokens -= tokens_needed;
            true
        } else {
            false
        }
    }

    pub async fn acquire(&self, tokens_needed: f64) -> Result<(), Duration> {
        if self.try_acquire(tokens_needed).await {
            Ok(())
        } else {
            let tokens = self.tokens.lock().await;
            let deficit = tokens_needed - *tokens;
            let wait_time = Duration::from_secs_f64(deficit / self.refill_rate);
            Err(wait_time)
        }
    }

    async fn refill(&self) {
        let mut last_refill = self.last_refill.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(*last_refill).as_secs_f64();

        let tokens_to_add = elapsed * self.refill_rate;
        if tokens_to_add > 0.0 {
            let mut tokens = self.tokens.lock().await;
            *tokens = (*tokens + tokens_to_add).min(self.capacity);
            *last_refill = now;
        }
    }

    pub async fn available_tokens(&self) -> f64 {
        self.refill().await;
        *self.tokens.lock().await
    }

    pub async fn time_until_token(&self) -> Duration {
        self.refill().await;
        let tokens = self.tokens.lock().await;
        
        if *tokens >= 1.0 {
            Duration::ZERO
        } else {
            let needed = 1.0 - *tokens;
            Duration::from_secs_f64(needed / self.refill_rate)
        }
    }
}

/// Per-key rate limiter using token buckets
pub struct KeyedRateLimiter {
    limiters: DashMap<String, Arc<TokenBucket>>,
    default_capacity: usize,
    default_rate: f64,
}

impl KeyedRateLimiter {
    pub fn new(default_capacity: usize, default_rate: f64) -> Self {
        Self {
            limiters: DashMap::new(),
            default_capacity,
            default_rate,
        }
    }

    pub async fn try_acquire(&self, key: &str, tokens: f64) -> bool {
        let limiter = self.get_or_create_limiter(key);
        limiter.try_acquire(tokens).await
    }

    pub async fn acquire(&self, key: &str, tokens: f64) -> Result<(), Duration> {
        let limiter = self.get_or_create_limiter(key);
        limiter.acquire(tokens).await
    }

    fn get_or_create_limiter(&self, key: &str) -> Arc<TokenBucket> {
        self.limiters
            .entry(key.to_string())
            .or_insert_with(|| {
                Arc::new(TokenBucket::new(self.default_capacity, self.default_rate))
            })
            .clone()
    }

    pub async fn get_stats(&self, key: &str) -> Option<f64> {
        self.limiters
            .get(key)
            .map(|limiter| {
                let limiter = limiter.clone();
                drop(limiter);
            });
        
        if let Some(limiter) = self.limiters.get(key) {
            Some(limiter.available_tokens().await)
        } else {
            None
        }
    }

    pub fn remove(&self, key: &str) {
        self.limiters.remove(key);
    }

    pub fn clear(&self) {
        self.limiters.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 1.0);
        
        // Should have full capacity initially
        assert!(bucket.try_acquire(5.0).await);
        assert!(bucket.try_acquire(5.0).await);
        assert!(!bucket.try_acquire(1.0).await); // No tokens left
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(10, 10.0); // 10 tokens/sec
        
        // Exhaust tokens
        assert!(bucket.try_acquire(10.0).await);
        assert!(!bucket.try_acquire(1.0).await);
        
        // Wait for refill
        sleep(Duration::from_millis(200)).await;
        
        // Should have ~2 tokens now
        assert!(bucket.try_acquire(2.0).await);
    }

    #[tokio::test]
    async fn test_token_bucket_capacity_cap() {
        let bucket = TokenBucket::new(5, 10.0);
        
        // Wait longer than needed to fill
        sleep(Duration::from_secs(1)).await;
        
        // Should still be capped at capacity
        assert!(bucket.try_acquire(5.0).await);
        assert!(!bucket.try_acquire(1.0).await);
    }

    #[tokio::test]
    async fn test_keyed_rate_limiter() {
        let limiter = KeyedRateLimiter::new(10, 1.0);
        
        // Different keys should have independent limits
        assert!(limiter.try_acquire("key1", 5.0).await);
        assert!(limiter.try_acquire("key2", 5.0).await);
        
        // Same key should share limit
        assert!(limiter.try_acquire("key1", 5.0).await);
        assert!(!limiter.try_acquire("key1", 1.0).await);
    }

    #[tokio::test]
    async fn test_time_until_token() {
        let bucket = TokenBucket::new(1, 1.0); // 1 token/sec
        
        // Exhaust tokens
        bucket.try_acquire(1.0).await;
        
        let wait_time = bucket.time_until_token().await;
        assert!(wait_time > Duration::ZERO);
        assert!(wait_time <= Duration::from_secs(1));
    }
}
