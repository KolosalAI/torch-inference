#![allow(dead_code)] // TODO: remove once model files (Tasks 3-5) consume this module
use anyhow::Result;
use lru::LruCache;
use parking_lot::Mutex;
use std::any::Any;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ── FNV-1a 64-bit ────────────────────────────────────────────────────────────
// Stable across Rust versions and process restarts (unlike DefaultHasher).
// Same algorithm used by TTSManager.

const FNV_OFFSET: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

fn fnv1a(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Compute a stable u64 cache key from three independent byte slices.
///
/// Streams FNV-1a over each slice in turn, inserting NUL-byte separators
/// between fields so "ab"+"c" ≠ "a"+"bc".  Zero heap allocation — the
/// previous implementation built a `Vec<u8>` before hashing.
pub fn cache_key(model_id: &str, input: &[u8], params: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in model_id.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= 0u64; // NUL separator
    h = h.wrapping_mul(FNV_PRIME);
    for &b in input {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h ^= 0u64; // NUL separator
    h = h.wrapping_mul(FNV_PRIME);
    for &b in params {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

// ── CacheStats ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

// ── ModelCache ────────────────────────────────────────────────────────────────

/// Model-agnostic LRU result cache.
///
/// Results are stored as `Arc<dyn Any + Send + Sync>` — the concrete value lives
/// on the heap behind a type-erased pointer.  On a hit the `Arc` pointer is
/// cloned (~5 ns) and then downcast back to `T` via `downcast_ref` + `T::clone`.
/// No serialization or deserialization ever touches the hot path.
/// Works for any `T: Any + Send + Sync + Clone + 'static`.
pub struct ModelCache {
    cache: Mutex<LruCache<u64, Arc<dyn Any + Send + Sync>>>,
    capacity: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ModelCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).expect("capacity >= 1");
        Self {
            cache: Mutex::new(LruCache::new(cap)),
            capacity: cap.get(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Return a cached result for `key`, or run `f`, cache its result, and return it.
    pub fn get_or_run<T, F>(&self, key: u64, f: F) -> Result<T>
    where
        T: Any + Send + Sync + Clone + 'static,
        F: FnOnce() -> Result<T>,
    {
        // Check cache (hold lock only while reading).
        let cached = {
            let mut guard = self.cache.lock();
            guard.get(&key).cloned() // clones the Arc pointer, not the data
        };

        if let Some(arc) = cached {
            // Downcast: if the stored type matches T, return a clone.
            // Mismatch is treated as a cache miss (impossible in practice).
            if let Some(val) = arc.downcast_ref::<T>() {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(val.clone());
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = f()?;

        // Store the concrete value wrapped in Arc<dyn Any + Send + Sync>.
        {
            let mut guard = self.cache.lock();
            guard.put(key, Arc::new(result.clone()) as Arc<dyn Any + Send + Sync>);
        }

        Ok(result)
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        };
        CacheStats {
            hits,
            misses,
            hit_rate,
        }
    }

    pub fn clear(&self) {
        let mut guard = self.cache.lock();
        guard.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_stable() {
        let k1 = cache_key("model.pt", b"input_bytes", b"params");
        let k2 = cache_key("model.pt", b"input_bytes", b"params");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_model_id() {
        let k1 = cache_key("model_a.pt", b"input", b"params");
        let k2 = cache_key("model_b.pt", b"input", b"params");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_input() {
        let k1 = cache_key("model.pt", b"input_a", b"params");
        let k2 = cache_key("model.pt", b"input_b", b"params");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_params() {
        let k1 = cache_key("model.pt", b"input", b"params_a");
        let k2 = cache_key("model.pt", b"input", b"params_b");
        assert_ne!(k1, k2);
    }

    /// Key collision resistance: "ab" + "c" must differ from "a" + "bc".
    #[test]
    fn test_cache_key_no_cross_field_collision() {
        let k1 = cache_key("ab", b"c", b"");
        let k2 = cache_key("a", b"bc", b"");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_get_or_run_miss_then_hit() {
        let cache: ModelCache = ModelCache::new(4);
        let key = cache_key("m", b"input", b"p");

        let mut call_count = 0u32;
        let r1: u32 = cache
            .get_or_run(key, || {
                call_count += 1;
                Ok(42u32)
            })
            .unwrap();
        assert_eq!(r1, 42);
        assert_eq!(call_count, 1);

        let r2: u32 = cache
            .get_or_run(key, || {
                call_count += 1;
                Ok(99u32)
            })
            .unwrap();
        assert_eq!(r2, 42); // cached value, not 99
        assert_eq!(call_count, 1); // f was NOT called again
    }

    #[test]
    fn test_stats_hit_and_miss_counting() {
        let cache = ModelCache::new(4);
        let key = cache_key("m", b"x", b"y");

        let _: u32 = cache.get_or_run(key, || Ok(1u32)).unwrap(); // miss
        let _: u32 = cache.get_or_run(key, || Ok(1u32)).unwrap(); // hit
        let _: u32 = cache.get_or_run(key, || Ok(1u32)).unwrap(); // hit

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 2);
        assert!((stats.hit_rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_empty_cache_hit_rate_is_zero() {
        let cache = ModelCache::new(4);
        let stats = cache.stats();
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[test]
    fn test_clear_resets_stats_and_cache() {
        let cache = ModelCache::new(4);
        let key = cache_key("m", b"a", b"b");
        let _: u32 = cache.get_or_run(key, || Ok(7u32)).unwrap();

        cache.clear();
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        // After clear, same key triggers a miss (f is called again).
        let mut calls = 0u32;
        let _: u32 = cache
            .get_or_run(key, || {
                calls += 1;
                Ok(7u32)
            })
            .unwrap();
        assert_eq!(calls, 1);
    }

    #[test]
    fn test_capacity_respected_evicts_oldest() {
        let cache = ModelCache::new(2);
        let k1 = cache_key("m", b"1", b"");
        let k2 = cache_key("m", b"2", b"");
        let k3 = cache_key("m", b"3", b"");

        let _: u32 = cache.get_or_run(k1, || Ok(1u32)).unwrap();
        let _: u32 = cache.get_or_run(k2, || Ok(2u32)).unwrap();
        let _: u32 = cache.get_or_run(k3, || Ok(3u32)).unwrap(); // evicts k1

        // k1 should be a miss now
        let mut calls = 0u32;
        let _: u32 = cache
            .get_or_run(k1, || {
                calls += 1;
                Ok(1u32)
            })
            .unwrap();
        assert_eq!(calls, 1, "k1 should have been evicted");
    }

    #[test]
    fn test_capacity_accessor() {
        let cache = ModelCache::new(256);
        assert_eq!(cache.capacity(), 256);
    }

    #[test]
    fn test_new_zero_capacity_clamps_to_one() {
        let cache = ModelCache::new(0);
        assert_eq!(cache.capacity(), 1);
    }

    #[test]
    fn test_get_or_run_no_serde_on_hit() {
        // With Arc<dyn Any>, hitting the cache never calls serde.
        // We verify this by caching a value that is NOT Serialize/Deserialize —
        // if it compiles and works, there's no serde on the hot path.
        #[derive(Clone, Debug, PartialEq)]
        struct NoSerde { x: u32 }

        let cache = ModelCache::new(4);
        let key = cache_key("m", b"input", b"params");

        let r1: NoSerde = cache.get_or_run(key, || Ok(NoSerde { x: 42 })).unwrap();
        assert_eq!(r1.x, 42);

        let r2: NoSerde = cache.get_or_run(key, || Ok(NoSerde { x: 99 })).unwrap();
        assert_eq!(r2.x, 42, "cached value must be returned on hit");
    }

    #[test]
    fn test_get_or_run_error_propagates() {
        let cache = ModelCache::new(4);
        let key = cache_key("m", b"err", b"");
        let result: anyhow::Result<u32> =
            cache.get_or_run(key, || anyhow::bail!("inference failed"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("inference failed"));
        // Error must NOT be cached — next call runs f again.
        let mut calls = 0u32;
        let ok: u32 = cache
            .get_or_run(key, || {
                calls += 1;
                Ok(55u32)
            })
            .unwrap();
        assert_eq!(ok, 55);
        assert_eq!(calls, 1);
    }
}
