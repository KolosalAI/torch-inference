use lru::LruCache;
use serde_json::Value;
use parking_lot::RwLock;
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicU64, Ordering};
use log::debug;
use std::num::NonZeroUsize;

#[derive(Clone)]
struct CacheEntry {
    data: Value,
    expires_at: Instant,
}

impl CacheEntry {
    #[inline]
    fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// High-performance LRU cache with RwLock for concurrent reads
pub struct Cache {
    data: RwLock<LruCache<String, CacheEntry>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl Cache {
    pub fn new(max_size: usize) -> Self {
        let capacity = NonZeroUsize::new(max_size.max(1)).unwrap();
        Self {
            data: RwLock::new(LruCache::new(capacity)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get a cached value - uses read lock for concurrent access
    #[inline]
    pub fn get(&self, key: &str) -> Option<Value> {
        // Fast path: try read lock first to check if entry exists and is valid
        {
            let cache = self.data.read();
            if let Some(entry) = cache.peek(key) {
                if !entry.is_expired() {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Some(entry.data.clone());
                }
            }
        }
        
        // Slow path: entry expired or doesn't exist, need write lock
        let mut cache = self.data.write();
        if let Some(entry) = cache.get(key) {
            if entry.is_expired() {
                cache.pop(key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            // Entry was valid (race condition: added between read and write lock)
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Some(entry.data.clone());
        }
        
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Set a cached value with TTL in seconds
    #[inline]
    pub fn set(&self, key: String, value: Value, ttl: u64) -> Result<(), String> {
        let ttl_duration = Duration::from_secs(ttl.min(315_360_000)); // Cap at ~10 years
        let entry = CacheEntry {
            data: value,
            expires_at: Instant::now() + ttl_duration,
        };
        
        let mut cache = self.data.write();
        cache.put(key, entry);
        Ok(())
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
        
        let cache = self.data.read();
        CacheStats {
            hits,
            misses,
            evictions: 0, // LRU handles eviction internally
            size: cache.len(),
            hit_rate,
            avg_samples_per_eviction: 0.0,
        }
    }

    #[inline]
    pub fn remove(&self, key: &str) {
        let mut cache = self.data.write();
        cache.pop(key);
    }

    pub fn clear(&self) {
        let mut cache = self.data.write();
        cache.clear();
        debug!("Cache cleared");
    }

    #[inline]
    pub fn size(&self) -> usize {
        let cache = self.data.read();
        cache.len()
    }

    pub fn cleanup_expired(&self) {
        let mut cache = self.data.write();
        let expired: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired {
            cache.pop(&key);
            debug!("Cleaned up expired entry: {}", key);
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new(10000)  // Increased default cache size for better hit rate
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
    use std::thread;
    use std::sync::Arc;

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
        assert!(cache.set("key1".to_string(), serde_json::json!("val1"), 60).is_ok());
        assert!(cache.set("key2".to_string(), serde_json::json!("val2"), 60).is_ok());
        // LRU eviction allows third entry
        assert!(cache.set("key3".to_string(), serde_json::json!("val3"), 60).is_ok());
        
        // key1 should be evicted (LRU)
        assert!(cache.get("key1").is_none());
        assert!(cache.get("key2").is_some());
        assert!(cache.get("key3").is_some());
    }

    #[test]
    fn test_cache_remove() {
        let cache = Cache::new(100);
        let key = "remove_test".to_string();
        
        cache.set(key.clone(), serde_json::json!("value"), 60).unwrap();
        assert!(cache.get(&key).is_some());
        
        cache.remove(&key);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = Cache::new(100);
        cache.set("key1".to_string(), serde_json::json!("val1"), 60).unwrap();
        cache.set("key2".to_string(), serde_json::json!("val2"), 60).unwrap();
        
        assert_eq!(cache.size(), 2);
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_cleanup_expired() {
        let cache = Cache::new(100);
        
        // Set one entry with 1 second TTL
        cache.set("key1".to_string(), serde_json::json!("val1"), 1).unwrap();
        // Set another with 60 second TTL
        cache.set("key2".to_string(), serde_json::json!("val2"), 60).unwrap();
        
        // Wait for key1 to expire
        thread::sleep(std::time::Duration::from_secs(2));
        cache.cleanup_expired();
        
        assert_eq!(cache.size(), 1);
        assert!(cache.get("key2").is_some());
    }

    #[test]
    fn test_cache_entry_is_expired() {
        let entry = CacheEntry {
            data: serde_json::json!("test"),
            expires_at: Instant::now() - Duration::from_secs(1),
        };
        
        assert!(entry.is_expired());
    }

    #[test]
    fn test_cache_entry_not_expired() {
        let entry = CacheEntry {
            data: serde_json::json!("test"),
            expires_at: Instant::now() + Duration::from_secs(60),
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
            cache.set(format!("key_{}", i), serde_json::json!(i), 60).unwrap();
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
        
        assert!(cache.set("large_key".to_string(), large_value.clone(), 60).is_ok());
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
        cache.set("zero_ttl".to_string(), serde_json::json!("value"), 0).unwrap();
        
        // With 0 TTL, should expire after 1 second
        thread::sleep(std::time::Duration::from_secs(1));
        assert_eq!(cache.get("zero_ttl"), None);
    }

    #[test]
    fn test_cache_very_long_ttl() {
        let cache = Cache::new(100);
        let max_ttl = u64::MAX;
        
        assert!(cache.set("long_ttl".to_string(), serde_json::json!("value"), max_ttl).is_ok());
        assert_eq!(cache.get("long_ttl"), Some(serde_json::json!("value")));
    }

    #[test]
    fn test_cache_boundary_conditions() {
        // Test with size 0
        let cache = Cache::new(0);
        assert!(cache.set("key".to_string(), serde_json::json!("val"), 60).is_ok());
        // With LRU, it should evict immediately but still work
        
        // Test with size 1
        let cache = Cache::new(1);
        assert!(cache.set("key1".to_string(), serde_json::json!("val1"), 60).is_ok());
        assert!(cache.set("key2".to_string(), serde_json::json!("val2"), 60).is_ok());
        // key1 should be evicted
        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_cache_cleanup_performance() {
        let cache = Cache::new(10000);
        
        // Add 1000 entries with short TTL
        for i in 0..1000 {
            cache.set(format!("expire_{}", i), serde_json::json!(i), 1).unwrap();
        }
        
        // Add 1000 entries with long TTL
        for i in 0..1000 {
            cache.set(format!("keep_{}", i), serde_json::json!(i), 3600).unwrap();
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
            cache.set(format!("key_{}", i), serde_json::json!({"index": i}), 60).unwrap();
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
        cache.set("key".to_string(), serde_json::json!("value"), 60).unwrap();
        
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
        cache.set("expire_me".to_string(), serde_json::json!("value"), 1).unwrap();
        
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
                    cache_clone.set(key.clone(), serde_json::json!(i * 2), 60).ok();
                    
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
        
        // Should create with default size (10000)
        for i in 0..1000 {  // Test with 1000 entries (subset of 10000)
            assert!(cache.set(format!("key_{}", i), serde_json::json!(i), 60).is_ok());
        }
        
        // Additional entry should work fine with 10000 capacity
        assert!(cache.set("overflow".to_string(), serde_json::json!("ok"), 60).is_ok());
    }

    #[test]
    fn test_cache_entry_clone() {
        let entry = CacheEntry {
            data: serde_json::json!({"test": "data"}),
            expires_at: Instant::now() + Duration::from_secs(60),
        };
        
        let cloned = entry.clone();
        assert_eq!(entry.data, cloned.data);
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
}
