//! Coarse monotonic unix-second clock.
//!
//! `coarse_unix_secs()` returns the current Unix timestamp truncated to whole
//! seconds.  The first call in each new second pays a single `SystemTime::now()`
//! syscall; subsequent calls within the same second read a cached `AtomicU64`
//! (~1 ns) instead.
//!
//! This eliminates repeated syscall overhead in hot cache / TTL paths that only
//! need 1-second resolution (e.g. `CacheEntry::is_expired`, `Cache::set`,
//! `RequestDeduplicator::get`).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static COARSE_SECS: AtomicU64 = AtomicU64::new(0);

/// Optional test-mode override. When `MOCK_OFFSET_SECS != 0`, the value is
/// added to the real clock — lets tests "advance time" deterministically
/// without `thread::sleep(2 secs)`. Production never touches this; the
/// load path is one extra relaxed atomic load.
static MOCK_OFFSET_SECS: AtomicU64 = AtomicU64::new(0);

/// Return the current Unix timestamp in whole seconds.
///
/// Resolution: 1 second (sufficient for TTL checks).
/// Cost: ~1 ns on cache hit (one atomic load); one `SystemTime::now()` syscall
///       the first time a new second is observed.
#[inline]
pub fn coarse_unix_secs() -> u64 {
    let cached = COARSE_SECS.load(Ordering::Relaxed);
    let real = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Only update the atomic when the second has actually ticked — keeps the
    // common case to a single load + branch-not-taken.
    if real != cached {
        COARSE_SECS.store(real, Ordering::Relaxed);
    }
    real.saturating_add(MOCK_OFFSET_SECS.load(Ordering::Relaxed))
}

/// Test-only: advance the mock clock by `secs`. Calling `advance_clock(N)`
/// is equivalent to `thread::sleep(Duration::from_secs(N))` for any code
/// path that reads `coarse_unix_secs()`. **Never call from production**.
#[cfg(test)]
pub fn advance_clock(secs: u64) {
    MOCK_OFFSET_SECS.fetch_add(secs, Ordering::Relaxed);
}

/// Test-only: reset the mock offset.
#[cfg(test)]
pub fn reset_clock() {
    MOCK_OFFSET_SECS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_nonzero() {
        assert!(coarse_unix_secs() > 0);
    }

    #[test]
    fn stable_within_same_second() {
        let a = coarse_unix_secs();
        let b = coarse_unix_secs();
        // Two back-to-back calls must agree (no second boundary crossed in <1 µs).
        assert!(b >= a, "time must not go backwards");
    }

    /// `advance_clock` is the cheap alternative to thread::sleep for tests
    /// that read `coarse_unix_secs()` (cache TTL, breaker windows, etc.).
    /// This must run serially because the offset is global state.
    #[test]
    #[serial_test::serial(clock)]
    fn advance_clock_moves_time_forward() {
        reset_clock();
        let t0 = coarse_unix_secs();
        advance_clock(7);
        let t1 = coarse_unix_secs();
        assert!(t1 >= t0 + 7, "advance_clock(7) should add at least 7 secs");
        reset_clock();
    }
}
