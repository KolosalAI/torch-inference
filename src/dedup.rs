use dashmap::DashMap;
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use log::{debug, trace};

#[derive(Clone)]
pub struct DeduplicationEntry {
    pub result: Value,
    pub timestamp: u64,
    pub ttl: u64,
}

pub struct RequestDeduplicator {
    cache: DashMap<String, DeduplicationEntry>,
    max_entries: usize,
}

impl RequestDeduplicator {
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: DashMap::new(),
            max_entries,
        }
    }

    pub fn generate_key(&self, model: &str, inputs: &Value) -> String {
        format!("{}:{}:{}", model, inputs.to_string(), SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() / 10)
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        if let Some(entry) = self.cache.get(key) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            if now - entry.timestamp < entry.ttl {
                debug!("Deduplication cache hit: {}", key);
                return Some(entry.result.clone());
            } else {
                drop(entry);
                self.cache.remove(key);
                debug!("Deduplication entry expired: {}", key);
            }
        }
        trace!("Deduplication cache miss: {}", key);
        None
    }

    pub fn set(&self, key: String, result: Value, ttl: u64) -> Result<(), String> {
        if self.cache.len() >= self.max_entries {
            return Err("Deduplication cache full".to_string());
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.cache.insert(key.clone(), DeduplicationEntry {
            result,
            timestamp: now,
            ttl,
        });

        debug!("Deduplication entry set: {} (TTL: {}s)", key, ttl);
        Ok(())
    }

    pub fn invalidate(&self, key: &str) {
        self.cache.remove(key);
        debug!("Deduplication entry invalidated: {}", key);
    }

    pub fn clear(&self) {
        self.cache.clear();
        debug!("Deduplication cache cleared");
    }

    pub fn cleanup_expired(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let expired: Vec<String> = self.cache
            .iter()
            .filter(|entry| now - entry.value().timestamp >= entry.value().ttl)
            .map(|entry| entry.key().clone())
            .collect();

        let count = expired.len();
        for key in expired.iter() {
            self.cache.remove(key);
        }

        debug!("Deduplication cleanup completed, removed {} entries", count);
    }

    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

impl Default for RequestDeduplicator {
    fn default() -> Self {
        Self::new(5000)
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
        assert!(dedup.set("key3".to_string(), serde_json::json!("val3"), 60).is_err());
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
