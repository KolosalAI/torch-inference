#![allow(dead_code)]
use log::{debug, trace};
use lru::LruCache;
use parking_lot::Mutex;
use serde_json::Value;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ── FNV-1a 64-bit ─────────────────────────────────────────────────────────
// Replaces Sha256 for dedup key generation — ~20× faster for short strings,
// no cryptographic properties needed for a 10-second dedup window.
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
        let cap = NonZeroUsize::new(max_entries.max(1)).expect("max_entries must be at least 1");
        Self {
            cache: Mutex::new(LruCache::new(cap)),
        }
    }

    pub fn generate_key(&self, model: &str, inputs: &Value) -> String {
        // Build the canonical form into a pre-allocated buffer, then hash with
        // FNV-1a (non-cryptographic, ~20× faster than SHA-256, sufficient for
        // a 10-second dedup window).
        let mut buf = String::with_capacity(128);
        write_canonical_json(inputs, &mut buf);
        let hash = fnv1a(buf.as_bytes());
        let epoch_window = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            / 10;
        format!("{}:{:016x}:{}", model, hash, epoch_window)
    }

    /// Returns a cheap `Arc` clone of the cached value — O(1), no data copied.
    /// Promotes the entry to most-recently-used on a hit.
    pub fn get(&self, key: &str) -> Option<Arc<Value>> {
        let mut cache = self.cache.lock();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
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
            .unwrap_or_default()
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
            .unwrap_or_default()
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
///
/// Calls `write_canonical_json` into a pre-allocated `String` to avoid the
/// intermediate `Vec<String>` allocations the previous recursive-format
/// approach created at every level of nesting.
fn canonical_json(v: &Value) -> String {
    let mut buf = String::new();
    write_canonical_json(v, &mut buf);
    buf
}

/// Write the canonical JSON representation of `v` into `buf`.
///
/// Object keys are sorted for determinism.  All string-building is done via
/// `push`/`push_str` on a single pre-allocated buffer — no intermediate
/// `Vec<String>` or `format!` per level.
fn write_canonical_json(v: &Value, buf: &mut String) {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<(&String, &Value)> = map.iter().collect();
            pairs.sort_unstable_by_key(|(k, _)| k.as_str());
            buf.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                buf.push_str(&serde_json::to_string(k).unwrap_or_default());
                buf.push(':');
                write_canonical_json(val, buf);
            }
            buf.push('}');
        }
        Value::Array(arr) => {
            buf.push('[');
            for (i, val) in arr.iter().enumerate() {
                if i > 0 {
                    buf.push(',');
                }
                write_canonical_json(val, buf);
            }
            buf.push(']');
        }
        _ => {
            buf.push_str(&v.to_string());
        }
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
        assert!(
            dedup.get("key3").is_some(),
            "key3 (newly added) should be present"
        );
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

    #[test]
    fn test_request_deduplicator_default() {
        // Exercises RequestDeduplicator::default() (lines 131-132)
        let dedup = RequestDeduplicator::default();
        dedup.set("k".to_string(), serde_json::json!("v"), 60);
        assert!(dedup.get("k").is_some());
        assert_eq!(dedup.size(), 1);
    }
}
