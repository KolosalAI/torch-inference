use lru::LruCache;
use parking_lot::Mutex;
use serde_json::Value;
use sha2::{Sha256, Digest};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use log::{debug, trace};

/// Store the result behind an Arc so that clone on cache-hit is O(1)
/// (pointer copy) instead of a deep-copy of the potentially large JSON value.
#[derive(Clone)]
pub struct DeduplicationEntry {
    pub result: Arc<Value>,
    pub timestamp: u64,
    pub ttl: u64,
}

/// Request deduplicator backed by an LRU cache.
///
/// When at capacity, the least-recently-used entry is evicted automatically.
/// The old `DashMap`-based implementation returned `Err` when full, which the
/// call-site silently ignored — meaning new requests were never cached once
/// the map filled. LRU gives correct behaviour: stale results age out and
/// hot phrases stay cached.
pub struct RequestDeduplicator {
    cache: Mutex<LruCache<String, DeduplicationEntry>>,
}

impl RequestDeduplicator {
    pub fn new(max_entries: usize) -> Self {
        let cap = NonZeroUsize::new(max_entries.max(1))
            .expect("max_entries must be at least 1");
        Self {
            cache: Mutex::new(LruCache::new(cap)),
        }
    }

    pub fn generate_key(&self, model: &str, inputs: &Value) -> String {
        let canonical = canonical_json(inputs);
        let hash = Sha256::digest(canonical.as_bytes());
        // Use first 16 bytes (32 hex chars) — ample collision resistance for a
        // 10-second dedup window.
        let hash_prefix = hex::encode(&hash[..16]);
        let epoch_window = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            / 10;
        format!("{}:{}:{}", model, hash_prefix, epoch_window)
    }

    /// Returns a cheap `Arc` clone of the cached value — O(1), no data copied.
    /// Promotes the entry to most-recently-used on a hit.
    pub fn get(&self, key: &str) -> Option<Arc<Value>> {
        let mut cache = self.cache.lock();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // peek first (no promotion) to check TTL without a mutable borrow conflict
        let is_valid = cache
            .peek(key)
            .map_or(false, |e| now.saturating_sub(e.timestamp) < e.ttl);

        if is_valid {
            // promote to MRU and return
            let result = cache.get(key).map(|e| Arc::clone(&e.result));
            debug!("Deduplication cache hit: {}", key);
            result
        } else {
            cache.pop(key);
            trace!("Deduplication cache miss or expired: {}", key);
            None
        }
    }

    /// Insert a result. When at capacity the LRU entry is evicted automatically.
    /// Wrap `result` in an `Arc` once so all future cache hits share the allocation.
    pub fn set(&self, key: String, result: Value, ttl: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        debug!("Deduplication entry set: {} (TTL: {}s)", key, ttl);
        self.cache.lock().put(
            key,
            DeduplicationEntry {
                result: Arc::new(result),
                timestamp: now,
                ttl,
            },
        );
    }

    pub fn invalidate(&self, key: &str) {
        self.cache.lock().pop(key);
        debug!("Deduplication entry invalidated: {}", key);
    }

    pub fn clear(&self) {
        self.cache.lock().clear();
        debug!("Deduplication cache cleared");
    }

    pub fn cleanup_expired(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut cache = self.cache.lock();
        let expired: Vec<String> = cache
            .iter()
            .filter(|(_, e)| now.saturating_sub(e.timestamp) >= e.ttl)
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired.len();
        for key in &expired {
            cache.pop(key.as_str());
        }
        debug!("Deduplication cleanup: removed {} expired entries", count);
    }

    pub fn size(&self) -> usize {
        self.cache.lock().len()
    }
}

impl Default for RequestDeduplicator {
    fn default() -> Self {
        Self::new(5000)
    }
}

/// Serialise a JSON value to a canonical string where object keys are sorted.
fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<(&String, &Value)> = map.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            let inner = pairs
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap_or_default(),
                        canonical_json(v)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{}}}", inner)
        }
        Value::Array(arr) => {
            let inner = arr.iter().map(canonical_json).collect::<Vec<_>>().join(",");
            format!("[{}]", inner)
        }
        _ => v.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_set_and_get() {
        let dedup = RequestDeduplicator::new(100);
        let key = "test_key".to_string();
        let result = serde_json::json!({"output": "test_result"});

        dedup.set(key.clone(), result.clone(), 60);
        assert_eq!(dedup.get(&key), Some(Arc::new(result)));
    }

    #[test]
    fn test_dedup_miss() {
        let dedup = RequestDeduplicator::new(100);
        assert_eq!(dedup.get("nonexistent"), None);
    }

    #[test]
    fn test_dedup_expiration() {
        let dedup = RequestDeduplicator::new(100);
        let key = "expire_test".to_string();
        dedup.set(key.clone(), serde_json::json!({"output": "will_expire"}), 0);
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert_eq!(dedup.get(&key), None);
    }

    #[test]
    fn test_dedup_lru_eviction_on_full() {
        let dedup = RequestDeduplicator::new(2);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 60);
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60);

        // Access key1 to promote it to MRU; key2 becomes LRU
        let _ = dedup.get("key1");

        // Adding key3 should evict key2 (LRU), not fail
        dedup.set("key3".to_string(), serde_json::json!("val3"), 60);

        assert_eq!(dedup.size(), 2);
        assert!(dedup.get("key1").is_some(), "key1 (MRU) should survive");
        assert!(dedup.get("key2").is_none(), "key2 (LRU) should be evicted");
        assert!(dedup.get("key3").is_some(), "key3 (newly added) should be present");
    }

    #[test]
    fn test_dedup_invalidate() {
        let dedup = RequestDeduplicator::new(100);
        let key = "invalidate_test".to_string();
        dedup.set(key.clone(), serde_json::json!("value"), 60);
        assert!(dedup.get(&key).is_some());
        dedup.invalidate(&key);
        assert!(dedup.get(&key).is_none());
    }

    #[test]
    fn test_dedup_clear() {
        let dedup = RequestDeduplicator::new(100);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 60);
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60);
        assert_eq!(dedup.size(), 2);
        dedup.clear();
        assert_eq!(dedup.size(), 0);
    }

    #[test]
    fn test_dedup_cleanup_expired() {
        let dedup = RequestDeduplicator::new(100);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 0);
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60);
        std::thread::sleep(std::time::Duration::from_secs(1));
        dedup.cleanup_expired();
        assert_eq!(dedup.size(), 1);
        assert!(dedup.get("key2").is_some());
    }

    #[test]
    fn test_dedup_size() {
        let dedup = RequestDeduplicator::new(100);
        assert_eq!(dedup.size(), 0);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 60);
        assert_eq!(dedup.size(), 1);
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60);
        assert_eq!(dedup.size(), 2);
    }

    #[test]
    fn test_generate_key_is_order_independent() {
        let dedup = RequestDeduplicator::new(100);

        let mut map_a = serde_json::Map::new();
        map_a.insert("b".to_string(), serde_json::json!(2));
        map_a.insert("a".to_string(), serde_json::json!(1));

        let mut map_b = serde_json::Map::new();
        map_b.insert("a".to_string(), serde_json::json!(1));
        map_b.insert("b".to_string(), serde_json::json!(2));

        let key_a = dedup.generate_key("model", &Value::Object(map_a));
        let key_b = dedup.generate_key("model", &Value::Object(map_b));
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn test_generate_key_different_values_differ() {
        let dedup = RequestDeduplicator::new(100);
        let key_a = dedup.generate_key("model", &serde_json::json!({"x": 1}));
        let key_b = dedup.generate_key("model", &serde_json::json!({"x": 2}));
        assert_ne!(key_a, key_b);
    }
}
