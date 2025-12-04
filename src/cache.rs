use dashmap::DashMap;
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use log::debug;

#[derive(Clone)]
pub struct CacheEntry {
    pub data: Value,
    pub timestamp: u64,
    pub ttl: u64,
}

impl CacheEntry {
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now - self.timestamp > self.ttl
    }
}

pub struct Cache {
    data: DashMap<String, CacheEntry>,
    max_size: usize,
}

impl Cache {
    pub fn new(max_size: usize) -> Self {
        Self {
            data: DashMap::new(),
            max_size,
        }
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        if let Some(entry) = self.data.get(key) {
            if entry.is_expired() {
                drop(entry);
                self.data.remove(key);
                debug!("Cache entry expired: {}", key);
                return None;
            }
            debug!("Cache hit: {}", key);
            return Some(entry.data.clone());
        }
        debug!("Cache miss: {}", key);
        None
    }

    pub fn set(&self, key: String, value: Value, ttl: u64) -> Result<(), String> {
        if self.data.len() >= self.max_size {
            return Err("Cache is full".to_string());
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.data.insert(key.clone(), CacheEntry {
            data: value,
            timestamp: now,
            ttl,
        });

        debug!("Cache set: {} (TTL: {}s)", key, ttl);
        Ok(())
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
        let expired: Vec<String> = self.data
            .iter()
            .filter(|entry| entry.value().is_expired())
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
