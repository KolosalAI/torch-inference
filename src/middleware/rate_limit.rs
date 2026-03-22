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
