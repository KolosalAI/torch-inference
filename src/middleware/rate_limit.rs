use actix_web::{middleware, http::StatusCode, HttpResponse, ResponseError};
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

pub struct RateLimiter {
    request_counts: DashMap<String, (u64, u64)>,
    max_requests: u64,
    window_seconds: u64,
}

impl RateLimiter {
    pub fn new(max_requests: u64, window_seconds: u64) -> Self {
        Self {
            request_counts: DashMap::new(),
            max_requests,
            window_seconds,
        }
    }

    pub fn is_allowed(&self, key: &str) -> Result<(), RateLimitError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut entry = self.request_counts.entry(key.to_string()).or_insert((0, now));
        let (count, timestamp) = entry.value_mut();

        if now - *timestamp > self.window_seconds {
            *count = 1;
            *timestamp = now;
            Ok(())
        } else if *count < self.max_requests {
            *count += 1;
            Ok(())
        } else {
            let retry_after = self.window_seconds - (now - *timestamp);
            Err(RateLimitError {
                message: "Rate limit exceeded".to_string(),
                retry_after,
            })
        }
    }

    pub fn cleanup_old_entries(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.request_counts.retain(|_, (_, timestamp)| {
            now - *timestamp < self.window_seconds * 2
        });
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(100, 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── RateLimiter construction ─────────────────────────────────────────────

    #[test]
    fn rate_limiter_new_and_default_construct_without_panic() {
        let _r1 = RateLimiter::new(10, 60);
        let _r2 = RateLimiter::default();
    }

    // ── is_allowed ───────────────────────────────────────────────────────────

    #[test]
    fn first_request_is_allowed() {
        let limiter = RateLimiter::new(5, 60);
        assert!(limiter.is_allowed("client_a").is_ok(), "first request should be Ok");
    }

    #[test]
    fn multiple_requests_within_limit_are_all_ok() {
        let limiter = RateLimiter::new(5, 60);
        for i in 0..5 {
            assert!(
                limiter.is_allowed("client_b").is_ok(),
                "request {} should be within limit",
                i + 1
            );
        }
    }

    #[test]
    fn request_after_limit_exceeded_returns_err_with_retry_after() {
        // max_requests = 3, window = 60 s
        let limiter = RateLimiter::new(3, 60);
        let key = "client_c";
        // Exhaust the allowance
        for _ in 0..3 {
            let _ = limiter.is_allowed(key);
        }
        // Next request must be rejected
        let result = limiter.is_allowed(key);
        assert!(result.is_err(), "should be Err after limit exceeded");
        let err = result.unwrap_err();
        // retry_after must be within the configured window
        assert!(err.retry_after <= 60, "retry_after should not exceed the window");
    }

    #[test]
    fn different_keys_are_independent() {
        let limiter = RateLimiter::new(2, 60);
        // Exhaust key_1
        let _ = limiter.is_allowed("key_1");
        let _ = limiter.is_allowed("key_1");
        let _ = limiter.is_allowed("key_1"); // This should be rejected

        // key_2 has not been touched — first request must still succeed
        assert!(limiter.is_allowed("key_2").is_ok(), "independent key should still be allowed");
    }

    // ── cleanup_old_entries ──────────────────────────────────────────────────

    #[test]
    fn cleanup_old_entries_does_not_panic() {
        let limiter = RateLimiter::new(10, 1);
        let _ = limiter.is_allowed("cleanup_client");
        limiter.cleanup_old_entries(); // must not panic
    }

    // ── RateLimitError ───────────────────────────────────────────────────────

    #[test]
    fn rate_limit_error_display() {
        let err = RateLimitError {
            message: "Rate limit exceeded".to_string(),
            retry_after: 42,
        };
        assert_eq!(format!("{}", err), "Rate limit exceeded");
    }

    #[test]
    fn rate_limit_error_status_code_is_429() {
        let err = RateLimitError {
            message: "Rate limit exceeded".to_string(),
            retry_after: 10,
        };
        assert_eq!(err.status_code(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn rate_limit_error_response_has_429_status() {
        let err = RateLimitError {
            message: "Rate limit exceeded".to_string(),
            retry_after: 5,
        };
        let response = err.error_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn window_reset_after_expiry_lines_59_61() {
        // window_seconds=0: the window expires after 0 seconds.
        // First call: entry inserted, count=1 (count < max=100 branch).
        // After sleeping 1 second: now - timestamp = 1 > window_seconds (0) → reset branch
        // (lines 59-61): *count = 1, *timestamp = now → Ok(()).
        let limiter = RateLimiter::new(100, 0);
        let key = "window_reset_key";

        // First request creates the entry via the count < max branch
        assert!(limiter.is_allowed(key).is_ok());

        // Sleep to ensure `now - timestamp > 0` (i.e., at least 1 second passes)
        std::thread::sleep(std::time::Duration::from_secs(1));

        // Second request: now - timestamp >= 1 > window_seconds (0) → window reset (lines 59-61)
        let result = limiter.is_allowed(key);
        assert!(result.is_ok(), "should be Ok after window expired (reset branch)");
    }
}
