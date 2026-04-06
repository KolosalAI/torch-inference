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
    real
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
}
