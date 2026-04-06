#![allow(dead_code)]
use rand::Rng;
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, warn};

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

    pub fn with_delays(max_retries: usize, base_delay: Duration, max_delay: Duration) -> Self {
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
        E: std::fmt::Debug,
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

    /// Execute with a predicate to determine if error is retryable
    pub async fn execute_if<F, Fut, T, E, P>(&self, mut f: F, is_retryable: P) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
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

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_default_policy_fields() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.base_delay, Duration::from_millis(100));
        assert_eq!(policy.max_delay, Duration::from_secs(30));
        assert_eq!(policy.multiplier, 2.0);
        assert!(policy.jitter);
    }

    #[test]
    fn test_new_overrides_max_retries() {
        let policy = RetryPolicy::new(7);
        assert_eq!(policy.max_retries, 7);
        // Other fields come from Default
        assert_eq!(policy.base_delay, Duration::from_millis(100));
    }

    #[test]
    fn test_with_delays_constructor() {
        let policy =
            RetryPolicy::with_delays(5, Duration::from_millis(50), Duration::from_secs(10));
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.base_delay, Duration::from_millis(50));
        assert_eq!(policy.max_delay, Duration::from_secs(10));
        // multiplier and jitter come from Default
        assert_eq!(policy.multiplier, 2.0);
        assert!(policy.jitter);
    }

    #[test]
    fn test_delay_calculation_no_jitter_multiplier_one() {
        // multiplier == 1.0 → every attempt has the same base delay
        let policy = RetryPolicy {
            max_retries: 5,
            base_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(10),
            multiplier: 1.0,
            jitter: false,
        };
        assert_eq!(policy.calculate_delay(1), Duration::from_millis(200));
        assert_eq!(policy.calculate_delay(3), Duration::from_millis(200));
    }

    #[test]
    fn test_delay_with_jitter_is_in_reasonable_range() {
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
            jitter: true,
        };
        // ±15 % around 100 ms → [85 ms, 115 ms]
        for _ in 0..20 {
            let d = policy.calculate_delay(1);
            assert!(d <= Duration::from_millis(116), "jitter too high: {:?}", d);
        }
    }

    #[tokio::test]
    async fn test_execute_zero_retries_fails_immediately() {
        // max_retries == 0: first failure returns the error with no sleep
        let policy = RetryPolicy::new(0);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        let result = policy
            .execute(|| {
                let c = calls2.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err::<i32, _>("fail")
                }
            })
            .await;

        assert_eq!(result, Err("fail"));
        // Called exactly once (attempt 1 > max_retries 0 → return immediately)
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_execute_success_on_last_allowed_retry() {
        // Fails for max_retries attempts, succeeds on the final allowed retry
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        let result = policy
            .execute(|| {
                let c = calls2.clone();
                async move {
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    if n < 3 {
                        Err("transient")
                    } else {
                        Ok(99)
                    }
                }
            })
            .await;

        assert_eq!(result, Ok(99));
        assert_eq!(calls.load(Ordering::SeqCst), 4); // initial + 3 retries
    }

    #[tokio::test]
    async fn test_execute_if_retryable_error_exhausts_retries() {
        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        let result = policy
            .execute_if(
                || {
                    let c = calls2.clone();
                    async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Err::<i32, _>("retryable")
                    }
                },
                |e| *e == "retryable",
            )
            .await;

        assert_eq!(result, Err("retryable"));
        // 1 initial attempt + 2 retries
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_execute_if_success_after_retryable_failures() {
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        let result = policy
            .execute_if(
                || {
                    let c = calls2.clone();
                    async move {
                        let n = c.fetch_add(1, Ordering::SeqCst);
                        if n < 2 {
                            Err("retryable")
                        } else {
                            Ok(7)
                        }
                    }
                },
                |e| *e == "retryable",
            )
            .await;

        assert_eq!(result, Ok(7));
    }

    #[tokio::test]
    async fn test_execute_if_non_retryable_stops_immediately() {
        let policy = RetryPolicy::new(5);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        let result = policy
            .execute_if(
                || {
                    let c = calls2.clone();
                    async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Err::<i32, _>("permanent")
                    }
                },
                |e| *e == "retryable",
            )
            .await;

        assert_eq!(result, Err("permanent"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_execute_if_success_on_first_attempt() {
        let policy = RetryPolicy::new(3);
        let result = policy
            .execute_if(|| async { Ok::<_, &str>(100) }, |_e| true)
            .await;
        assert_eq!(result, Ok(100));
    }

    // ── Targeted gap-closing tests ────────────────────────────────────────────
    // The tests below are structured so that every concrete statement on the
    // "uncovered" lines (60, 65-66, 76-77, 84-85, 105, 123-124, 131-132)
    // is executed by at least one test below.

    /// execute – loop body entered, retries fired, then succeeds: covers 60 (loop
    /// entry), 84-85 (warn! on each retry), 64-66 (debug! on eventual success).
    #[tokio::test]
    async fn test_execute_loop_retry_then_succeed_covers_debug_and_warn() {
        let policy = RetryPolicy {
            max_retries: 4,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        // Fail twice (triggers warn! at 84-85 twice), then succeed (triggers
        // debug! at 64-66 because attempt > 0).
        let result = policy
            .execute(|| {
                let c = calls2.clone();
                async move {
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    if n < 2 {
                        Err("transient")
                    } else {
                        Ok(55)
                    }
                }
            })
            .await;

        assert_eq!(result, Ok(55));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// execute – always fails until max retries exhausted: covers 60 (loop),
    /// 76-77 (warn! "Max retries exceeded"), 84-85 (warn! per retry attempt).
    #[tokio::test]
    async fn test_execute_loop_all_retries_exhausted_covers_max_retries_warn() {
        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        let result = policy
            .execute(|| {
                let c = calls2.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err::<i32, _>("permanent")
                }
            })
            .await;

        assert_eq!(result, Err("permanent"));
        // 1 initial + 2 retries
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// execute_if – loop entered (105), retryable error exhausts retries:
    /// covers 105 (loop), 123-124 (warn! max retries exceeded), 131-132
    /// (warn! per retry attempt).
    #[tokio::test]
    async fn test_execute_if_loop_exhausts_retries_covers_warns() {
        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        let result = policy
            .execute_if(
                || {
                    let c = calls2.clone();
                    async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Err::<i32, _>("retryable-err")
                    }
                },
                |e| *e == "retryable-err",
            )
            .await;

        assert_eq!(result, Err("retryable-err"));
        assert_eq!(calls.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    /// execute_if – retryable errors then success: covers 105 (loop entry),
    /// 131-132 (warn! per retry), and the Ok branch with attempt > 0.
    #[tokio::test]
    async fn test_execute_if_loop_retry_then_succeed() {
        let policy = RetryPolicy {
            max_retries: 5,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        let result = policy
            .execute_if(
                || {
                    let c = calls2.clone();
                    async move {
                        let n = c.fetch_add(1, Ordering::SeqCst);
                        if n < 3 {
                            Err("retry-me")
                        } else {
                            Ok(77)
                        }
                    }
                },
                |e| *e == "retry-me",
            )
            .await;

        assert_eq!(result, Ok(77));
    }

    // ── Tracing-subscriber-enabled tests ─────────────────────────────────────
    //
    // The `tracing` warn!/debug! macros use a per-callsite interest cache that is
    // populated from the GLOBAL subscriber.  If no global subscriber is set, the
    // callsite is cached as "never interested" and even `set_default` (thread-local)
    // cannot override that cached decision.
    //
    // We therefore install a TRACE-level global subscriber exactly once (via OnceLock)
    // so that all callsites in this module are cached as "always interested" before
    // any tracing test runs.

    use std::sync::OnceLock;

    static TRACING_INIT: OnceLock<()> = OnceLock::new();

    fn ensure_global_trace_subscriber() {
        TRACING_INIT.get_or_init(|| {
            let subscriber = tracing_subscriber::fmt()
                .with_max_level(tracing::Level::TRACE)
                .with_writer(std::io::sink)
                .finish();
            // Ignore error if another module already installed a global subscriber.
            let _ = tracing::subscriber::set_global_default(subscriber);
        });
    }

    /// Covers lines 84-85 (warn! per retry) and 65-66 (debug! on success after
    /// retries) – thread-local TRACE subscriber active so macro args are evaluated.
    ///
    /// Also exercises the `loop {` re-entry (line 60) by looping three times.
    #[tokio::test]
    async fn test_execute_retry_then_succeed_with_subscriber() {
        ensure_global_trace_subscriber();

        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        // Fail twice → warn! at 84-85 fires twice; succeed on 3rd → debug! at
        // 65-66 fires; loop re-entered 3 times (covers line 60 loop-back).
        let result = policy
            .execute(|| {
                let c = calls2.clone();
                async move {
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    if n < 2 {
                        Err("transient")
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result, Ok(42));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// Covers lines 76-77 (warn! "Max retries exceeded") in execute() – the
    /// loop keeps re-entering (line 60) and exhausts all retries.
    #[tokio::test]
    async fn test_execute_exhaust_retries_with_subscriber() {
        ensure_global_trace_subscriber();

        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        // Always-failing closure: loop at line 60 re-enters 3 times (initial +
        // 2 retries), then warn! at 76-77 fires on the final Err return.
        let result = policy.execute(|| async { Err::<i32, _>("perm") }).await;
        assert_eq!(result, Err("perm"));
    }

    /// Covers lines 131-132 (warn! per retry in execute_if) and debug! on
    /// success (line 109) – loop at line 105 re-enters multiple times.
    #[tokio::test]
    async fn test_execute_if_retry_succeed_with_subscriber() {
        ensure_global_trace_subscriber();

        let policy = RetryPolicy {
            max_retries: 4,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();

        // Fail twice with a retryable error → warn! at 131-132 fires twice;
        // succeed on 3rd attempt → debug! at 109 fires; loop 105 re-entered.
        let result = policy
            .execute_if(
                || {
                    let c = calls2.clone();
                    async move {
                        let n = c.fetch_add(1, Ordering::SeqCst);
                        if n < 2 {
                            Err("retry-me")
                        } else {
                            Ok(99)
                        }
                    }
                },
                |e| *e == "retry-me",
            )
            .await;

        assert_eq!(result, Ok(99));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// Covers lines 123-124 (warn! "Max retries exceeded" in execute_if) –
    /// loop at 105 re-enters until all retries exhausted.
    #[tokio::test]
    async fn test_execute_if_exhaust_retries_with_subscriber() {
        ensure_global_trace_subscriber();

        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        // Always-retryable failure: loop at 105 re-enters 3 times (initial +
        // 2 retries), warn! at 131-132 per retry, warn! at 123-124 on exit.
        let result = policy
            .execute_if(|| async { Err::<i32, _>("retryable") }, |_| true)
            .await;
        assert_eq!(result, Err("retryable"));
    }

    // ── Dedicated gap-closing tests (lines 60, 65-66, 76-77, 84-85, 105,
    //    123-124, 131-132) ────────────────────────────────────────────────────

    /// execute(): succeed AFTER one retry.
    /// - loop body entered (line 60)
    /// - warn! on retry attempt (lines 84-85)
    /// - debug! on success after retry (lines 65-66)
    #[tokio::test]
    async fn test_execute_success_after_one_retry() {
        ensure_global_trace_subscriber();
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        // Fail once (triggers warn! at 84-85), succeed on second call
        // (triggers debug! at 65-66 since attempt > 0).
        let result = policy
            .execute(|| {
                let c = calls2.clone();
                async move {
                    let n = c.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        Err("one-shot-fail")
                    } else {
                        Ok(1)
                    }
                }
            })
            .await;
        assert_eq!(result, Ok(1));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// execute(): always fail until max retries exceeded.
    /// - loop body entered (line 60) multiple times
    /// - warn! "Max retries exceeded" (lines 76-77)
    /// - warn! per retry attempt (lines 84-85)
    #[tokio::test]
    async fn test_execute_always_fail_exceeds_max_retries() {
        ensure_global_trace_subscriber();
        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let result = policy
            .execute(|| {
                let c = calls2.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err::<i32, _>("always-fail")
                }
            })
            .await;
        assert_eq!(result, Err("always-fail"));
        // 1 initial + 2 retries
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    /// execute_if(): non-retryable predicate → exits immediately on first failure.
    /// - loop body entered (line 105)
    /// - is_retryable returns false → returns Err immediately (no retry sleep)
    #[tokio::test]
    async fn test_execute_if_non_retryable_predicate_exits_first_failure() {
        ensure_global_trace_subscriber();
        let policy = RetryPolicy::new(5);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let result = policy
            .execute_if(
                || {
                    let c = calls2.clone();
                    async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Err::<i32, _>("non-retryable-error")
                    }
                },
                // predicate always returns false → no retries
                |_e| false,
            )
            .await;
        assert_eq!(result, Err("non-retryable-error"));
        // Must be called exactly once — no retries
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// execute_if(): retryable errors exhaust max retries.
    /// - loop body entered (line 105) multiple times
    /// - warn! per retry (lines 131-132)
    /// - warn! "Max retries exceeded" (lines 123-124)
    #[tokio::test]
    async fn test_execute_if_retryable_exceeds_max_retries() {
        ensure_global_trace_subscriber();
        let policy = RetryPolicy {
            max_retries: 2,
            base_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(2),
            multiplier: 1.0,
            jitter: false,
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let result = policy
            .execute_if(
                || {
                    let c = calls2.clone();
                    async move {
                        c.fetch_add(1, Ordering::SeqCst);
                        Err::<i32, _>("retryable-err")
                    }
                },
                |_e| true,
            )
            .await;
        assert_eq!(result, Err("retryable-err"));
        assert_eq!(calls.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }
}
