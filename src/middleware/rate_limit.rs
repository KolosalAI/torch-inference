use actix_web::{http::StatusCode, HttpResponse, ResponseError};
use std::fmt;

#[derive(Debug)]
pub struct RateLimitError {
    pub message: String,
    pub retry_after: u64,
}

impl fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl ResponseError for RateLimitError {
    fn status_code(&self) -> StatusCode {
        StatusCode::TOO_MANY_REQUESTS
    }

    fn error_response(&self) -> HttpResponse {
        HttpResponse::TooManyRequests()
            .insert_header(("Retry-After", self.retry_after.to_string()))
            .json(serde_json::json!({
                "error": self.message,
                "retry_after": self.retry_after
            }))
    }
}

use dashmap::DashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, Ordering};

/// High-performance rate limiter with lazy cleanup
pub struct RateLimiter {
    request_counts: DashMap<String, (u64, u64)>, // (count, window_start)
    max_requests: u64,
    window_seconds: u64,
    last_cleanup: AtomicU64,
}

impl RateLimiter {
    pub fn new(max_requests: u64, window_seconds: u64) -> Self {
        Self {
            request_counts: DashMap::new(),
            max_requests,
            window_seconds,
            last_cleanup: AtomicU64::new(0),
        }
    }

    #[inline]
    fn current_time() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Check if request is allowed - optimized for hot path
    #[inline]
    pub fn is_allowed(&self, key: &str) -> Result<(), RateLimitError> {
        let now = Self::current_time();
        
        // Lazy cleanup: only run every 60 seconds
        let last_cleanup = self.last_cleanup.load(Ordering::Relaxed);
        if now - last_cleanup > 60 {
            if self.last_cleanup.compare_exchange(
                last_cleanup, now, Ordering::Relaxed, Ordering::Relaxed
            ).is_ok() {
                self.cleanup_old_entries_internal(now);
            }
        }

        let mut entry = self.request_counts.entry(key.to_string()).or_insert((0, now));
        let (count, timestamp) = entry.value_mut();

        if now - *timestamp > self.window_seconds {
            // Window expired, reset
            *count = 1;
            *timestamp = now;
            return Ok(());
        }
        
        if *count < self.max_requests {
            *count += 1;
            return Ok(());
        }
        
        let retry_after = self.window_seconds - (now - *timestamp);
        Err(RateLimitError {
            message: "Rate limit exceeded".to_string(),
            retry_after,
        })
    }

    fn cleanup_old_entries_internal(&self, now: u64) {
        self.request_counts.retain(|_, (_, timestamp)| {
            now - *timestamp < self.window_seconds * 2
        });
    }

    pub fn cleanup_old_entries(&self) {
        let now = Self::current_time();
        self.cleanup_old_entries_internal(now);
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(1000, 60) // Increased default from 100 to 1000 requests per minute
    }
}
