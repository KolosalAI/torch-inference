use dashmap::DashMap;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, Ordering};
use log::{debug, trace};
use sha2::{Sha256, Digest};

#[derive(Clone)]
pub struct DeduplicationEntry {
    pub result: Value,
    pub timestamp: u64,
    pub ttl: u64,
}

/// High-performance request deduplicator with hash-based keys
pub struct RequestDeduplicator {
    cache: DashMap<String, DeduplicationEntry>,
    max_entries: usize,
    last_cleanup: AtomicU64,
}

impl RequestDeduplicator {
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: DashMap::new(),
            max_entries,
            last_cleanup: AtomicU64::new(0),
        }
    }

    #[inline]
    fn current_time() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Generate a fast hash-based key instead of serializing the entire input
    #[inline]
    pub fn generate_key(&self, model: &str, inputs: &Value) -> String {
        let now = Self::current_time();
        let window = now / 10; // 10-second deduplication window
        
        // Use compact JSON serialization for hashing
        let mut hasher = Sha256::new();
        hasher.update(model.as_bytes());
        hasher.update(b":");
        
        // Hash the JSON value directly - faster than to_string()
        Self::hash_value(&mut hasher, inputs);
        
        hasher.update(b":");
        hasher.update(window.to_le_bytes());
        
        let result = hasher.finalize();
        // Use first 16 bytes (128 bits) for a shorter but still unique key
        hex::encode(&result[..16])
    }
    
    /// Recursively hash a JSON value without allocating strings
    fn hash_value(hasher: &mut Sha256, value: &Value) {
        match value {
            Value::Null => hasher.update(b"n"),
            Value::Bool(b) => hasher.update(if *b { b"t" } else { b"f" }),
            Value::Number(n) => {
                hasher.update(b"#");
                // Use raw bytes for numbers
                if let Some(i) = n.as_i64() {
                    hasher.update(i.to_le_bytes());
                } else if let Some(u) = n.as_u64() {
                    hasher.update(u.to_le_bytes());
                } else if let Some(f) = n.as_f64() {
                    hasher.update(f.to_le_bytes());
                }
            }
            Value::String(s) => {
                hasher.update(b"s");
                hasher.update(s.as_bytes());
            }
            Value::Array(arr) => {
                hasher.update(b"[");
                for item in arr {
                    Self::hash_value(hasher, item);
                }
                hasher.update(b"]");
            }
            Value::Object(obj) => {
                hasher.update(b"{");
                // Sort keys for consistent hashing
                let mut keys: Vec<_> = obj.keys().collect();
                keys.sort();
                for key in keys {
                    hasher.update(key.as_bytes());
                    hasher.update(b":");
                    Self::hash_value(hasher, &obj[key]);
                }
                hasher.update(b"}");
            }
        }
    }

    #[inline]
    pub fn get(&self, key: &str) -> Option<Value> {
        let now = Self::current_time();
        
        // Lazy cleanup every 30 seconds
        let last_cleanup = self.last_cleanup.load(Ordering::Relaxed);
        if now - last_cleanup > 30 {
            if self.last_cleanup.compare_exchange(
                last_cleanup, now, Ordering::Relaxed, Ordering::Relaxed
            ).is_ok() {
                self.cleanup_expired_internal(now);
            }
        }

        if let Some(entry) = self.cache.get(key) {
            if now - entry.timestamp < entry.ttl {
                trace!("Deduplication cache hit: {}", key);
                return Some(entry.result.clone());
            } else {
                drop(entry);
                self.cache.remove(key);
                trace!("Deduplication entry expired: {}", key);
            }
        }
        None
    }

    #[inline]
    pub fn set(&self, key: String, result: Value, ttl: u64) -> Result<(), String> {
        // Allow some oversubscription before rejecting
        if self.cache.len() >= self.max_entries + 100 {
            // Try cleanup first
            self.cleanup_expired_internal(Self::current_time());
            if self.cache.len() >= self.max_entries {
                return Err("Deduplication cache full".to_string());
            }
        }

        let now = Self::current_time();
        self.cache.insert(key, DeduplicationEntry {
            result,
            timestamp: now,
            ttl,
        });

        Ok(())
    }

    #[inline]
    pub fn invalidate(&self, key: &str) {
        self.cache.remove(key);
    }

    pub fn clear(&self) {
        self.cache.clear();
        debug!("Deduplication cache cleared");
    }

    fn cleanup_expired_internal(&self, now: u64) {
        let before = self.cache.len();
        self.cache.retain(|_, entry| now - entry.timestamp < entry.ttl);
        let removed = before.saturating_sub(self.cache.len());
        if removed > 0 {
            debug!("Deduplication cleanup: removed {} entries", removed);
        }
    }

    pub fn cleanup_expired(&self) {
        self.cleanup_expired_internal(Self::current_time());
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

impl Default for RequestDeduplicator {
    fn default() -> Self {
        Self::new(10000) // Increased default from 5000 to 10000
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
        
        assert!(dedup.set(key.clone(), result.clone(), 60).is_ok());
        assert_eq!(dedup.get(&key), Some(result));
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
        let result = serde_json::json!({"output": "will_expire"});
        
        assert!(dedup.set(key.clone(), result.clone(), 0).is_ok());
        std::thread::sleep(std::time::Duration::from_secs(1));
        
        assert_eq!(dedup.get(&key), None);
    }

    #[test]
    fn test_dedup_full() {
        let dedup = RequestDeduplicator::new(2);
        assert!(dedup.set("key1".to_string(), serde_json::json!("val1"), 60).is_ok());
        assert!(dedup.set("key2".to_string(), serde_json::json!("val2"), 60).is_ok());
        // Allow some oversubscription (up to 100 extra entries before reject)
        assert!(dedup.set("key3".to_string(), serde_json::json!("val3"), 60).is_ok());
        // Fill up the oversubscription buffer
        for i in 4..=103 {
            let _ = dedup.set(format!("key{}", i), serde_json::json!("val"), 60);
        }
        // Now it should be full
        assert!(dedup.set("overflow".to_string(), serde_json::json!("val"), 60).is_err());
    }

    #[test]
    fn test_dedup_invalidate() {
        let dedup = RequestDeduplicator::new(100);
        let key = "invalidate_test".to_string();
        
        dedup.set(key.clone(), serde_json::json!("value"), 60).unwrap();
        assert!(dedup.get(&key).is_some());
        
        dedup.invalidate(&key);
        assert!(dedup.get(&key).is_none());
    }

    #[test]
    fn test_dedup_clear() {
        let dedup = RequestDeduplicator::new(100);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 60).unwrap();
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60).unwrap();
        
        assert_eq!(dedup.size(), 2);
        dedup.clear();
        assert_eq!(dedup.size(), 0);
    }

    #[test]
    fn test_dedup_cleanup_expired() {
        let dedup = RequestDeduplicator::new(100);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 0).unwrap();
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60).unwrap();
        
        std::thread::sleep(std::time::Duration::from_secs(1));
        dedup.cleanup_expired();
        
        assert_eq!(dedup.size(), 1);
        assert!(dedup.get("key2").is_some());
    }

    #[test]
    fn test_dedup_generate_key() {
        let dedup = RequestDeduplicator::new(100);
        let model = "test_model";
        let inputs = serde_json::json!({"data": "test"});
        
        let key1 = dedup.generate_key(model, &inputs);
        let key2 = dedup.generate_key(model, &inputs);
        
        // Keys should be equal within 10 second window
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_dedup_size() {
        let dedup = RequestDeduplicator::new(100);
        assert_eq!(dedup.size(), 0);
        
        dedup.set("key1".to_string(), serde_json::json!("val1"), 60).unwrap();
        assert_eq!(dedup.size(), 1);
        
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60).unwrap();
        assert_eq!(dedup.size(), 2);
    }
}
