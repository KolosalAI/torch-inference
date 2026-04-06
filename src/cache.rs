#![allow(dead_code)]
use bytes::Bytes;
use dashmap::DashMap;
use log::debug;
use rand::seq::SliceRandom;
use serde_json::Value;
use crate::clock::coarse_unix_secs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
pub struct CacheEntry {
    pub data: Value,
    pub timestamp: u64,
    pub ttl: u64,
    pub last_access: u64,
    pub access_count: u64,
    pub insertion_order: u64,
}

/// Zero-copy cache entry backed by `Arc<Bytes>`.
///
/// Reading from this entry costs ~5 ns (Arc pointer increment) vs ~300 ns
/// for a deep clone of `serde_json::Value`.
#[derive(Clone)]
pub struct BytesCacheEntry {
    /// Shared reference to the raw bytes (e.g. a serialised JSON response).
    pub data: Arc<Bytes>,
    pub timestamp: u64,
    pub ttl: u64,
}

impl BytesCacheEntry {
    pub fn is_expired(&self) -> bool {
        let now = coarse_unix_secs();
        self.is_expired_at(now)
    }

    #[inline]
    pub fn is_expired_at(&self, now: u64) -> bool {
        now - self.timestamp > self.ttl
    }
}

impl CacheEntry {
    pub fn is_expired(&self) -> bool {
        let now = coarse_unix_secs();
        self.is_expired_at(now)
    }

    #[inline]
    pub fn is_expired_at(&self, now: u64) -> bool {
        now - self.timestamp > self.ttl
    }
}

pub struct Cache {
    data: DashMap<String, CacheEntry>,
    /// Zero-copy byte-slice cache.  Entries here share the same TTL / max-size
    /// budget as the JSON cache but avoid deep cloning on every read.
    bytes_data: DashMap<String, BytesCacheEntry>,
    max_size: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    insertion_counter: AtomicU64,
    eviction_samples: AtomicU64, // Track number of entries sampled during eviction
}

impl Cache {
    pub fn new(max_size: usize) -> Self {
        Self {
            data: DashMap::new(),
            bytes_data: DashMap::new(),
            max_size,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            insertion_counter: AtomicU64::new(0),
            eviction_samples: AtomicU64::new(0),
        }
    }

    // ── Zero-copy Arc<Bytes> path ─────────────────────────────────────────

    /// Store pre-serialised bytes (e.g. a JSON response body) without copying.
    ///
    /// On a cache hit, [`get_bytes`] returns a clone of the `Arc<Bytes>` —
    /// just an atomic pointer increment (~5 ns), no heap allocation.
    pub fn set_bytes(&self, key: String, value: Arc<Bytes>, ttl: u64) -> Result<(), String> {
        // Evict if over capacity (reuses same max_size budget as JSON cache).
        if self.bytes_data.len() >= self.max_size && !self.bytes_data.contains_key(&key) {
            // Collect the eviction key in a separate statement so that the
            // DashMap Iter (and its shard read-lock) is fully dropped before
            // we call remove() — otherwise remove() tries to acquire a write
            // lock on the same shard the Iter is already holding, causing a
            // deadlock when the evicted entry hashes to the same shard.
            let evict_key: Option<String> = self.bytes_data.iter().next().map(|e| e.key().clone());
            if let Some(evict_key) = evict_key {
                self.bytes_data.remove(&evict_key);
            }
        }
        let now = coarse_unix_secs();
        self.bytes_data.insert(
            key,
            BytesCacheEntry {
                data: value,
                timestamp: now,
                ttl,
            },
        );
        Ok(())
    }

    /// Retrieve bytes by key.  Returns `None` on miss or expiry.
    ///
    /// Clone cost is O(1) — increments the `Arc` reference count only.
    pub fn get_bytes(&self, key: &str) -> Option<Arc<Bytes>> {
        let now = coarse_unix_secs();
        if let Some(entry) = self.bytes_data.get(key) {
            if entry.is_expired_at(now) {
                drop(entry);
                self.bytes_data.remove(key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(Arc::clone(&entry.data));
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Remove a bytes entry by key.
    pub fn remove_bytes(&self, key: &str) {
        self.bytes_data.remove(key);
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        let now = coarse_unix_secs();
        if let Some(mut entry) = self.data.get_mut(key) {
            if entry.is_expired_at(now) {
                drop(entry);
                self.data.remove(key);
                debug!("Cache entry expired: {}", key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            // Update LRU metadata — reuse `now` computed above
            entry.last_access = now;
            entry.access_count += 1;

            self.hits.fetch_add(1, Ordering::Relaxed);
            debug!("Cache hit: {}", key);
            return Some(entry.data.clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        debug!("Cache miss: {}", key);
        None
    }

    pub fn set(&self, key: String, value: Value, ttl: u64) -> Result<(), String> {
        // LRU eviction if cache is full
        if self.data.len() >= self.max_size && !self.data.contains_key(&key) {
            self.evict_lru();
        }

        let now = coarse_unix_secs();

        let insertion_order = self.insertion_counter.fetch_add(1, Ordering::Relaxed);

        self.data.insert(
            key.clone(),
            CacheEntry {
                data: value,
                timestamp: now,
                ttl,
                last_access: now,
                access_count: 0,
                insertion_order,
            },
        );

        debug!("Cache set: {} (TTL: {}s)", key, ttl);
        Ok(())
    }

    /// Compute adaptive eviction sample size based on current hit rate.
    ///
    /// Scales from 20 (healthy cache, high hit rate) to 100 (struggling cache,
    /// low hit rate). Reduces eviction overhead when cache is working well;
    /// improves victim selection quality when it is not.
    pub fn evict_sample_size(&self) -> usize {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 {
            1.0_f64
        } else {
            hits as f64 / total as f64
        };
        ((20.0 + (1.0 - hit_rate) * 80.0) as usize).clamp(20, 100)
    }

    /// SLRU-style eviction using random sampling (O(k) amortised with k ≪ n)
    ///
    /// Entries are partitioned into two logical segments:
    /// - **Probationary** (`access_count == 0`): newly inserted, never fetched yet.
    /// - **Protected** (`access_count > 0`): promoted by at least one `get` call.
    ///
    /// Eviction preference: probationary entries are evicted first (oldest
    /// `last_access` wins). Protected entries are only evicted when no
    /// probationary candidates exist in the sample.
    fn evict_lru(&self) {
        struct EvictCandidate {
            key: String,
            last_access: u64,
            access_count: u64,
            insertion_order: u64,
        }

        let sample_size = self.evict_sample_size();

        // Compute `now` once — avoids one syscall per entry in the filter below
        // (sample_size * 2 = 40–200 entries × ~30 ns/syscall = up to 6 µs saved).
        let now = coarse_unix_secs();

        // Collect at most sample_size * 2 non-expired entries — the .take() stops
        // the DashMap iterator early instead of scanning the entire map (O(k) vs O(n)).
        let all_candidates: Vec<EvictCandidate> = self
            .data
            .iter()
            .filter(|e| !e.value().is_expired_at(now))
            .take(sample_size * 2)
            .map(|e| EvictCandidate {
                key: e.key().clone(),
                last_access: e.value().last_access,
                access_count: e.value().access_count,
                insertion_order: e.value().insertion_order,
            })
            .collect();

        let mut rng = rand::thread_rng();
        let candidates: Vec<&EvictCandidate> = all_candidates
            .choose_multiple(&mut rng, (sample_size * 2).min(all_candidates.len()))
            .collect();

        self.eviction_samples
            .fetch_add(candidates.len() as u64, Ordering::Relaxed);

        if candidates.is_empty() {
            // Fallback: evict by insertion_order (evict oldest)
            let oldest = self
                .data
                .iter()
                .min_by_key(|e| e.value().insertion_order)
                .map(|e| e.key().clone());
            if let Some(key) = oldest {
                self.data.remove(&key);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
            return;
        }

        // SLRU: probationary (access_count == 0, never fetched after insertion) → evict first.
        // Protected (access_count > 0, fetched at least once) → evict only if no probationary exists.
        // Tiebreaker: insertion_order (oldest inserted wins).
        let victim_key = {
            let probationary: Vec<&&EvictCandidate> =
                candidates.iter().filter(|c| c.access_count == 0).collect();

            if !probationary.is_empty() {
                probationary
                    .iter()
                    .min_by_key(|c| (c.last_access, c.insertion_order))
                    .map(|c| c.key.clone())
            } else {
                candidates
                    .iter()
                    .min_by_key(|c| (c.last_access, c.insertion_order))
                    .map(|c| c.key.clone())
            }
        };

        if let Some(key) = victim_key {
            self.data.remove(&key);
            self.evictions.fetch_add(1, Ordering::Relaxed);
            debug!(
                "Evicted SLRU entry: {} (samples: {})",
                key,
                candidates.len()
            );
        }
    }

    pub fn get_stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            (hits as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let evictions = self.evictions.load(Ordering::Relaxed);
        let avg_samples_per_eviction = if evictions > 0 {
            self.eviction_samples.load(Ordering::Relaxed) as f64 / evictions as f64
        } else {
            0.0
        };

        CacheStats {
            hits,
            misses,
            evictions,
            size: self.data.len(),
            hit_rate,
            avg_samples_per_eviction,
        }
    }

    pub fn remove(&self, key: &str) {
        self.data.remove(key);
        debug!("Cache removed: {}", key);
    }

    pub fn clear(&self) {
        self.data.clear();
        debug!("Cache cleared");
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn cleanup_expired(&self) {
        let now = coarse_unix_secs();
        let expired: Vec<String> = self
            .data
            .iter()
            .filter(|entry| entry.value().is_expired_at(now))
            .map(|entry| entry.key().clone())
            .collect();

        for key in expired {
            self.data.remove(&key);
            debug!("Cleaned up expired entry: {}", key);
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: usize,
    pub hit_rate: f64,
    pub avg_samples_per_eviction: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // ===== Basic Functionality Tests =====

    #[test]
    fn test_cache_set_and_get() {
        let cache = Cache::new(100);
        let key = "test_key".to_string();
        let value = serde_json::json!({"data": "test_value"});

        assert!(cache.set(key.clone(), value.clone(), 60).is_ok());
        assert_eq!(cache.get(&key), Some(value));

        let stats = cache.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_miss() {
        let cache = Cache::new(100);
        assert_eq!(cache.get("nonexistent"), None);
    }

    #[test]
    fn test_cache_expiration() {
        let cache = Cache::new(100);
        let key = "expire_test".to_string();
        let value = serde_json::json!({"data": "will_expire"});

        // Set with 1 second TTL
        assert!(cache.set(key.clone(), value.clone(), 1).is_ok());

        // Wait for expiration
        thread::sleep(std::time::Duration::from_secs(2));

        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn test_cache_full() {
        let cache = Cache::new(2);
        assert!(cache
            .set("key1".to_string(), serde_json::json!("val1"), 60)
            .is_ok());
        assert!(cache
            .set("key2".to_string(), serde_json::json!("val2"), 60)
            .is_ok());
        // LRU eviction allows third entry
        assert!(cache
            .set("key3".to_string(), serde_json::json!("val3"), 60)
            .is_ok());

        // key1 should be evicted (LRU)
        assert!(cache.get("key1").is_none());
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());
    }

    #[test]
    fn test_cache_remove() {
        let cache = Cache::new(100);
        let key = "remove_test".to_string();

        cache
            .set(key.clone(), serde_json::json!("value"), 60)
            .unwrap();
        assert!(cache.get(&key).is_some());

        cache.remove(&key);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = Cache::new(100);
        cache
            .set("key1".to_string(), serde_json::json!("val1"), 60)
            .unwrap();
        cache
            .set("key2".to_string(), serde_json::json!("val2"), 60)
            .unwrap();

        assert_eq!(cache.size(), 2);
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_cleanup_expired() {
        let cache = Cache::new(100);

        // Set one entry with 1 second TTL
        cache
            .set("key1".to_string(), serde_json::json!("val1"), 1)
            .unwrap();
        // Set another with 60 second TTL
        cache
            .set("key2".to_string(), serde_json::json!("val2"), 60)
            .unwrap();

        // Wait for key1 to expire
        thread::sleep(std::time::Duration::from_secs(2));
        cache.cleanup_expired();

        assert_eq!(cache.size(), 1);
        assert!(cache.get("key2").is_some());
    }

    #[test]
    fn test_cache_entry_is_expired() {
        let now = coarse_unix_secs();

        let entry = CacheEntry {
            data: serde_json::json!("test"),
            timestamp: now - 100,
            ttl: 50,
            last_access: now,
            access_count: 0,
            insertion_order: 0,
        };

        assert!(entry.is_expired());
    }

    #[test]
    fn test_cache_entry_not_expired() {
        let now = coarse_unix_secs();

        let entry = CacheEntry {
            data: serde_json::json!("test"),
            timestamp: now,
            ttl: 60,
            last_access: now,
            access_count: 0,
            insertion_order: 0,
        };

        assert!(!entry.is_expired());
    }

    // ===== Enterprise-Grade Tests =====

    #[test]
    fn test_cache_concurrent_access() {
        let cache = Arc::new(Cache::new(1000));
        let mut handles = vec![];

        // Spawn 10 threads, each doing 100 operations
        for thread_id in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let key = format!("thread_{}_key_{}", thread_id, i);
                    let value = serde_json::json!({"thread": thread_id, "iteration": i});
                    cache_clone.set(key.clone(), value.clone(), 60).unwrap();

                    // Verify we can read what we wrote
                    let retrieved = cache_clone.get(&key);
                    assert_eq!(retrieved, Some(value));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify cache size (should have 1000 entries)
        assert_eq!(cache.size(), 1000);
    }

    #[test]
    fn test_cache_concurrent_read_write() {
        let cache = Arc::new(Cache::new(100));

        // Pre-populate cache
        for i in 0..50 {
            cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .unwrap();
        }

        let mut handles = vec![];

        // Readers
        for _ in 0..5 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    for i in 0..50 {
                        let _ = cache_clone.get(&format!("key_{}", i));
                    }
                }
            });
            handles.push(handle);
        }

        // Writers
        for thread_id in 0..3 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..20 {
                    let key = format!("writer_{}_key_{}", thread_id, i);
                    let _ = cache_clone.set(key, serde_json::json!(i), 60);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_cache_update_same_key() {
        let cache = Cache::new(100);
        let key = "update_test".to_string();

        cache.set(key.clone(), serde_json::json!("v1"), 60).unwrap();
        assert_eq!(cache.get(&key), Some(serde_json::json!("v1")));

        cache.set(key.clone(), serde_json::json!("v2"), 60).unwrap();
        assert_eq!(cache.get(&key), Some(serde_json::json!("v2")));

        // Size should still be 1
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_cache_large_values() {
        let cache = Cache::new(10);
        let large_value = serde_json::json!({
            "data": vec!["x"; 10000].join(""),
            "nested": {
                "level1": {
                    "level2": {
                        "level3": vec![1; 1000]
                    }
                }
            }
        });

        assert!(cache
            .set("large_key".to_string(), large_value.clone(), 60)
            .is_ok());
        assert_eq!(cache.get("large_key"), Some(large_value));
    }

    #[test]
    fn test_cache_unicode_keys() {
        let cache = Cache::new(100);
        let keys = vec![
            "🔑_emoji_key",
            "中文_chinese",
            "日本語_japanese",
            "한국어_korean",
            "العربية_arabic",
        ];

        for key in keys {
            let value = serde_json::json!({"key": key});
            assert!(cache.set(key.to_string(), value.clone(), 60).is_ok());
            assert_eq!(cache.get(key), Some(value));
        }
    }

    #[test]
    fn test_cache_zero_ttl() {
        let cache = Cache::new(100);
        cache
            .set("zero_ttl".to_string(), serde_json::json!("value"), 0)
            .unwrap();

        // With 0 TTL, should expire after 1 second
        thread::sleep(std::time::Duration::from_secs(1));
        assert_eq!(cache.get("zero_ttl"), None);
    }

    #[test]
    fn test_cache_very_long_ttl() {
        let cache = Cache::new(100);
        let max_ttl = u64::MAX;

        assert!(cache
            .set("long_ttl".to_string(), serde_json::json!("value"), max_ttl)
            .is_ok());
        assert_eq!(cache.get("long_ttl"), Some(serde_json::json!("value")));
    }

    #[test]
    fn test_cache_boundary_conditions() {
        // Test with size 0
        let cache = Cache::new(0);
        assert!(cache
            .set("key".to_string(), serde_json::json!("val"), 60)
            .is_ok());
        // With LRU, it should evict immediately but still work

        // Test with size 1
        let cache = Cache::new(1);
        assert!(cache
            .set("key1".to_string(), serde_json::json!("val1"), 60)
            .is_ok());
        assert!(cache
            .set("key2".to_string(), serde_json::json!("val2"), 60)
            .is_ok());
        // key1 should be evicted
        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_cache_cleanup_performance() {
        let cache = Cache::new(10000);

        // Add 1000 entries with short TTL
        for i in 0..1000 {
            cache
                .set(format!("expire_{}", i), serde_json::json!(i), 1)
                .unwrap();
        }

        // Add 1000 entries with long TTL
        for i in 0..1000 {
            cache
                .set(format!("keep_{}", i), serde_json::json!(i), 3600)
                .unwrap();
        }

        thread::sleep(std::time::Duration::from_secs(2));

        let start = std::time::Instant::now();
        cache.cleanup_expired();
        let duration = start.elapsed();

        // Cleanup should be fast even with many entries
        assert!(duration < std::time::Duration::from_millis(100));
        assert_eq!(cache.size(), 1000); // Only non-expired entries remain
    }

    #[test]
    fn test_cache_memory_efficiency() {
        let cache = Cache::new(1000);

        // Fill cache to capacity
        for i in 0..1000 {
            cache
                .set(format!("key_{}", i), serde_json::json!({"index": i}), 60)
                .unwrap();
        }

        assert_eq!(cache.size(), 1000);

        // Clear and verify memory is released
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_idempotent_operations() {
        let cache = Cache::new(100);

        // Multiple removes should be safe
        cache.remove("nonexistent");
        cache.remove("nonexistent");

        // Multiple clears should be safe
        cache.clear();
        cache.clear();

        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_get_does_not_modify() {
        let cache = Cache::new(100);
        cache
            .set("key".to_string(), serde_json::json!("value"), 60)
            .unwrap();

        let size_before = cache.size();
        for _ in 0..100 {
            let _ = cache.get("key");
        }
        let size_after = cache.size();

        assert_eq!(size_before, size_after);
    }

    #[test]
    fn test_cache_expired_entry_auto_removal() {
        let cache = Cache::new(100);
        cache
            .set("expire_me".to_string(), serde_json::json!("value"), 1)
            .unwrap();

        assert_eq!(cache.size(), 1);
        thread::sleep(std::time::Duration::from_secs(2));

        // Access should trigger removal
        assert_eq!(cache.get("expire_me"), None);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_mixed_operations() {
        let cache = Arc::new(Cache::new(1000));
        let mut handles = vec![];

        // Mixed operations from multiple threads
        for thread_id in 0..5 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..50 {
                    let key = format!("key_{}", i);

                    // Set
                    cache_clone.set(key.clone(), serde_json::json!(i), 60).ok();

                    // Get
                    let _ = cache_clone.get(&key);

                    // Update
                    cache_clone
                        .set(key.clone(), serde_json::json!(i * 2), 60)
                        .ok();

                    // Maybe remove
                    if thread_id % 2 == 0 && i % 10 == 0 {
                        cache_clone.remove(&key);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_cache_default_construction() {
        let cache = Cache::default();

        // Should create with default size (1000)
        for i in 0..1000 {
            assert!(cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .is_ok());
        }

        // 1001st should trigger LRU eviction
        assert!(cache
            .set("overflow".to_string(), serde_json::json!("ok"), 60)
            .is_ok());
        let stats = cache.get_stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_cache_entry_clone() {
        let entry = CacheEntry {
            data: serde_json::json!({"test": "data"}),
            timestamp: 12345,
            ttl: 60,
            last_access: 12345,
            access_count: 0,
            insertion_order: 0,
        };

        let cloned = entry.clone();
        assert_eq!(entry.data, cloned.data);
        assert_eq!(entry.timestamp, cloned.timestamp);
        assert_eq!(entry.ttl, cloned.ttl);
    }

    #[test]
    fn test_cache_stress_test() {
        let cache = Arc::new(Cache::new(5000));
        let mut handles = vec![];

        // High concurrency stress test
        for thread_id in 0..20 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let key = format!("stress_{}_{}", thread_id, i);
                    let value = serde_json::json!({
                        "thread": thread_id,
                        "iteration": i,
                        "timestamp": SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis()
                    });

                    let _ = cache_clone.set(key.clone(), value, 60);
                    let _ = cache_clone.get(&key);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    // ===== Sampling-Based LRU Tests =====

    #[test]
    fn test_sampled_lru_eviction() {
        let cache = Cache::new(10);

        // Fill cache to capacity
        for i in 0..10 {
            cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .unwrap();
            thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamps
        }

        // Access first few entries to update their last_access
        for i in 0..5 {
            cache.get(&format!("key_{}", i));
        }

        // Add one more entry to trigger eviction
        cache
            .set("new_key".to_string(), serde_json::json!("new"), 60)
            .unwrap();

        // One of the unaccessed entries (5-9) should be evicted
        let stats = cache.get_stats();
        assert_eq!(stats.evictions, 1);
        assert!(stats.avg_samples_per_eviction > 0.0);
    }

    #[test]
    fn test_sampled_eviction_performance() {
        let cache = Cache::new(1000);

        // Fill cache
        for i in 0..1000 {
            cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .unwrap();
        }

        // Measure eviction time
        let start = std::time::Instant::now();
        for i in 1000..1100 {
            cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .unwrap();
        }
        let duration = start.elapsed();

        // 100 evictions should be fast (< 200ms total, averaging <2ms per eviction)
        // Note: Relaxed timing to account for CI/debug builds
        assert!(duration < std::time::Duration::from_millis(200));

        let stats = cache.get_stats();
        assert_eq!(stats.evictions, 100);
    }

    #[test]
    fn test_eviction_sample_size() {
        let cache = Cache::new(5);

        // With small cache (< max sample size 100), should sample all entries
        for i in 0..5 {
            cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .unwrap();
        }

        // Trigger eviction
        cache
            .set("overflow".to_string(), serde_json::json!("x"), 60)
            .unwrap();

        let stats = cache.get_stats();
        assert_eq!(stats.evictions, 1);
        // Should sample all 5 entries when cache size < max sample size (100)
        assert!(stats.avg_samples_per_eviction <= 5.0);
    }

    #[test]
    fn test_large_cache_sampling() {
        let cache = Cache::new(1000);

        // Fill large cache
        for i in 0..1000 {
            cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .unwrap();
        }

        // Trigger multiple evictions
        for i in 1000..1010 {
            cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .unwrap();
        }

        let stats = cache.get_stats();
        assert_eq!(stats.evictions, 10);
        // Should sample up to 100 entries per eviction
        assert!(stats.avg_samples_per_eviction <= 100.0);
        assert!(stats.avg_samples_per_eviction > 0.0);
    }

    #[test]
    fn test_eviction_statistics() {
        let cache = Cache::new(100);

        // Add entries
        for i in 0..100 {
            cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .unwrap();
        }

        let stats_before = cache.get_stats();
        assert_eq!(stats_before.evictions, 0);

        // Trigger evictions
        for i in 100..110 {
            cache
                .set(format!("key_{}", i), serde_json::json!(i), 60)
                .unwrap();
        }

        let stats_after = cache.get_stats();
        assert_eq!(stats_after.evictions, 10);
        assert_eq!(stats_after.size, 100); // Cache maintains max_size
    }

    // ── Zero-copy Arc<Bytes> cache ────────────────────────────────────────

    #[test]
    fn test_set_bytes_and_get_bytes_roundtrip() {
        let cache = Cache::new(100);
        let data = Arc::new(Bytes::from(b"hello bytes cache".to_vec()));
        cache
            .set_bytes("k1".to_string(), Arc::clone(&data), 60)
            .unwrap();
        let got = cache.get_bytes("k1").unwrap();
        assert_eq!(&*got, &*data);
    }

    #[test]
    fn test_get_bytes_miss_returns_none() {
        let cache = Cache::new(100);
        assert!(cache.get_bytes("nonexistent").is_none());
    }

    #[test]
    fn test_get_bytes_returns_arc_clone_not_deep_copy() {
        let cache = Cache::new(100);
        let data = Arc::new(Bytes::from(vec![1u8, 2, 3]));
        cache
            .set_bytes("k".to_string(), Arc::clone(&data), 60)
            .unwrap();
        let got1 = cache.get_bytes("k").unwrap();
        let got2 = cache.get_bytes("k").unwrap();
        // Both arcs point to the same allocation.
        assert!(Arc::ptr_eq(&got1, &got2));
    }

    #[test]
    fn test_set_bytes_overwrites_existing() {
        let cache = Cache::new(100);
        cache
            .set_bytes("k".to_string(), Arc::new(Bytes::from("v1")), 60)
            .unwrap();
        cache
            .set_bytes("k".to_string(), Arc::new(Bytes::from("v2")), 60)
            .unwrap();
        let got = cache.get_bytes("k").unwrap();
        assert_eq!(got.as_ref(), b"v2".as_ref());
    }

    #[test]
    fn test_remove_bytes_removes_entry() {
        let cache = Cache::new(100);
        cache
            .set_bytes("k".to_string(), Arc::new(Bytes::from("v")), 60)
            .unwrap();
        cache.remove_bytes("k");
        assert!(cache.get_bytes("k").is_none());
    }

    #[test]
    fn test_bytes_cache_respects_ttl_expiry() {
        let cache = Cache::new(100);
        cache
            .set_bytes("k".to_string(), Arc::new(Bytes::from("expire")), 1)
            .unwrap();
        thread::sleep(std::time::Duration::from_secs(2));
        assert!(cache.get_bytes("k").is_none());
    }

    #[test]
    fn test_bytes_entry_is_expired_true() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = coarse_unix_secs();
        let entry = BytesCacheEntry {
            data: Arc::new(Bytes::from("x")),
            timestamp: now - 200,
            ttl: 100,
        };
        assert!(entry.is_expired());
    }

    #[test]
    fn test_bytes_entry_is_expired_false() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = coarse_unix_secs();
        let entry = BytesCacheEntry {
            data: Arc::new(Bytes::from("x")),
            timestamp: now,
            ttl: 600,
        };
        assert!(!entry.is_expired());
    }

    #[test]
    #[allow(unexpected_cfgs)]
    #[cfg_attr(
        tarpaulin,
        ignore = "DashMap shard lock interacts with tarpaulin ptrace instrumentation"
    )]
    fn test_bytes_cache_evicts_when_full() {
        // max_size = 2; inserting 3 should evict one.
        let cache = Cache::new(2);
        cache
            .set_bytes("a".to_string(), Arc::new(Bytes::from("1")), 60)
            .unwrap();
        cache
            .set_bytes("b".to_string(), Arc::new(Bytes::from("2")), 60)
            .unwrap();
        cache
            .set_bytes("c".to_string(), Arc::new(Bytes::from("3")), 60)
            .unwrap();
        // Only 2 entries remain.
        assert_eq!(cache.bytes_data.len(), 2);
    }

    #[test]
    fn test_bytes_cache_empty_bytes() {
        let cache = Cache::new(100);
        let empty = Arc::new(Bytes::new());
        cache.set_bytes("empty".to_string(), empty, 60).unwrap();
        let got = cache.get_bytes("empty").unwrap();
        assert!(got.is_empty());
    }

    /// Verifies that evict_lru retains the correct number of entries when the
    /// cache is much larger than the maximum sample size (100) — i.e. it only
    /// needs up to 100 keys, not the full cache contents.
    #[test]
    fn test_evict_lru_large_cache_retains_size_minus_one() {
        let size = 100 * 3;
        let cache = Cache::new(size);
        for i in 0..size {
            cache
                .set(format!("k{}", i), serde_json::json!(i), 3600)
                .unwrap();
        }
        assert_eq!(cache.size(), size);
        // Insert one more — triggers evict_lru; cache must not grow beyond size.
        cache
            .set("overflow".to_string(), serde_json::json!(0), 3600)
            .unwrap();
        assert_eq!(
            cache.size(),
            size,
            "cache should stay at max_size after eviction"
        );
    }

    /// Covers the empty-candidates branch in evict_lru().
    /// A zero-capacity cache satisfies `data.len() >= max_size` (0 >= 0) on
    /// every new insertion, so evict_lru() is called with an empty DashMap.
    /// Inside evict_lru(), `all_candidates` is empty → `choose_multiple` yields
    /// nothing → `candidates.is_empty()` is true → the fallback oldest-key path
    /// also finds nothing → function returns without evicting.
    #[test]
    fn test_evict_lru_early_return_on_empty_cache() {
        let cache = Cache::new(0);
        // Inserting any key triggers evict_lru() with an empty data map.
        let _ = cache.set("k".to_string(), serde_json::json!(1), 60);
    }

    /// Verifies that is_expired() on a CacheEntry with timestamp=0 (the Unix
    /// epoch itself) returns `true` and does not panic.  This guards against
    /// the `.unwrap()` on `duration_since(UNIX_EPOCH)` — in a VM with a
    /// clock set before the epoch that call would return Err and panic.
    #[test]
    fn test_cache_entry_is_expired_with_epoch_timestamp_does_not_panic() {
        let entry = CacheEntry {
            data: serde_json::json!({}),
            timestamp: 0,
            ttl: 1,
            last_access: 0,
            access_count: 0,
            insertion_order: 0,
        };
        // Epoch-0 entry with ttl=1 is always expired; must not panic.
        assert!(entry.is_expired());
    }

    #[test]
    fn test_bytes_cache_entry_is_expired_with_epoch_timestamp_does_not_panic() {
        let entry = BytesCacheEntry {
            data: Arc::new(Bytes::new()),
            timestamp: 0,
            ttl: 1,
        };
        assert!(entry.is_expired());
    }

    #[test]
    fn slru_evicts_probationary_before_protected() {
        let cache = Cache::new(2);
        cache
            .set("prot".to_string(), serde_json::json!(1), 3600)
            .unwrap();
        cache.get("prot"); // first fetch → access_count becomes 1 (protected, > 0)
        cache
            .set("prob".to_string(), serde_json::json!(2), 3600)
            .unwrap();
        // "prob" has access_count=0 (probationary, never fetched)

        // Insert a third entry to trigger eviction — prob should be evicted, not prot
        cache
            .set("new".to_string(), serde_json::json!(3), 3600)
            .unwrap();

        assert!(
            cache.get("prot").is_some(),
            "protected entry should survive eviction"
        );
        assert!(cache.get("new").is_some(), "new entry should be present");
        assert!(
            cache.get("prob").is_none(),
            "probationary entry should have been evicted"
        );
    }

    #[test]
    fn test_evict_sample_size_adapts_to_hit_rate() {
        let cache = Cache::new(1000);

        // Simulate high hit rate: sample size should be near 20
        for i in 0..100 {
            cache
                .set(format!("k{}", i), serde_json::json!(i), 3600)
                .unwrap();
        }
        // 100 gets on populated cache → high hit rate
        for i in 0..100 {
            cache.get(&format!("k{}", i));
        }
        let size_high_hr = cache.evict_sample_size();
        assert!(
            size_high_hr <= 30,
            "expected ~20 at high hit rate, got {}",
            size_high_hr
        );

        // Simulate zero hit rate: sample size should be near 100
        let cold = Cache::new(1000);
        for i in 0..100 {
            cold.get(&format!("miss{}", i)); // all misses
        }
        let size_low_hr = cold.evict_sample_size();
        assert!(
            size_low_hr >= 90,
            "expected ~100 at zero hit rate, got {}",
            size_low_hr
        );
    }
}
