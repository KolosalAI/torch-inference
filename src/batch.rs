use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use log::{info, debug};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct BatchRequest {
    pub id: String,
    pub model_name: String,
    pub inputs: Vec<Value>,
    pub priority: i32,
    pub timestamp: Instant,
}

pub struct BatchProcessor {
    max_batch_size: usize,
    batch_timeout_ms: u64,
    current_batch: Arc<RwLock<Vec<BatchRequest>>>,
}

impl BatchProcessor {
    pub fn new(max_batch_size: usize, batch_timeout_ms: u64) -> Self {
        Self {
            max_batch_size,
            batch_timeout_ms,
            current_batch: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn add_request(&self, request: BatchRequest) -> Result<bool, String> {
        let mut batch = self.current_batch.write().await;
        
        if batch.len() >= self.max_batch_size {
            return Err("Batch is full".to_string());
        }

        batch.push(request);
        debug!("Request added to batch. Current size: {}", batch.len());
        
        Ok(batch.len() >= self.max_batch_size)
    }

    pub async fn should_process_batch(&self) -> bool {
        let batch = self.current_batch.read().await;
        
        if batch.is_empty() {
            return false;
        }

        if batch.len() >= self.max_batch_size {
            return true;
        }

        if let Some(oldest) = batch.first() {
            let age = oldest.timestamp.elapsed();
            return age >= Duration::from_millis(self.batch_timeout_ms);
        }

        false
    }

    pub async fn get_batch(&self) -> Vec<BatchRequest> {
        let mut batch = self.current_batch.write().await;
        
        let mut requests = Vec::new();
        std::mem::swap(&mut requests, &mut batch);
        
        info!("Processing batch with {} requests", requests.len());
        requests
    }

    pub async fn get_batch_size(&self) -> usize {
        self.current_batch.read().await.len()
    }

    pub async fn clear_batch(&self) {
        let mut batch = self.current_batch.write().await;
        batch.clear();
        debug!("Batch cleared");
    }

    pub async fn process_with_timeout<F, T>(&self, processor: F) -> Result<T, String>
    where
        F: std::future::Future<Output = Result<T, String>>,
    {
        tokio::time::timeout(
            Duration::from_secs(30),
            processor,
        )
        .await
        .map_err(|_| "Batch processing timeout".to_string())?
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new(32, 100)
    }
}
