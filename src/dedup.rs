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
