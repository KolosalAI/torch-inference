use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore, oneshot};
use log::{info, debug, warn};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::collections::VecDeque;

/// Request with response channel for inflight batching
#[derive(Debug)]
pub struct InflightRequest {
    pub id: String,
    pub model_name: String,
    pub inputs: Vec<Value>,
    pub priority: i32,
    pub timestamp: Instant,
    pub response_tx: oneshot::Sender<Result<Value, String>>,
}

/// Inflight batch processor that allows continuous batching
/// New requests can be added while inference is running on previous batch
pub struct InflightBatchProcessor {
    max_batch_size: usize,
    max_inflight_batches: usize,
    batch_timeout_ms: u64,
    
    // Queue of pending requests
    pending_queue: Arc<RwLock<VecDeque<InflightRequest>>>,
    
    // Semaphore to limit concurrent batches
    inflight_semaphore: Arc<Semaphore>,
    
    // Statistics
    processed_batches: AtomicU64,
    processed_requests: AtomicU64,
    total_latency_ms: AtomicU64,
    total_wait_time_ms: AtomicU64,
    current_queue_depth: AtomicUsize,
    peak_queue_depth: AtomicUsize,
    inflight_batches: AtomicUsize,
}

impl InflightBatchProcessor {
    pub fn new(max_batch_size: usize, max_inflight_batches: usize, batch_timeout_ms: u64) -> Self {
        Self {
            max_batch_size,
            max_inflight_batches,
            batch_timeout_ms,
            pending_queue: Arc::new(RwLock::new(VecDeque::new())),
            inflight_semaphore: Arc::new(Semaphore::new(max_inflight_batches)),
            processed_batches: AtomicU64::new(0),
            processed_requests: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            total_wait_time_ms: AtomicU64::new(0),
            current_queue_depth: AtomicUsize::new(0),
            peak_queue_depth: AtomicUsize::new(0),
            inflight_batches: AtomicUsize::new(0),
        }
    }
    
    /// Add a request to the pending queue
    pub async fn add_request(&self, request: InflightRequest) -> Result<(), String> {
        let mut queue = self.pending_queue.write().await;
        
        // Add to queue (sorted by priority)
        let insert_pos = queue.iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(queue.len());
        
        queue.insert(insert_pos, request);
        
        let queue_size = queue.len();
        self.current_queue_depth.store(queue_size, Ordering::Relaxed);
        
        // Update peak
        let current_peak = self.peak_queue_depth.load(Ordering::Relaxed);
        if queue_size > current_peak {
            self.peak_queue_depth.store(queue_size, Ordering::Relaxed);
        }
        
        debug!("Request added to inflight queue. Queue size: {}", queue_size);
        Ok(())
    }
    
    /// Try to form a batch from pending requests
    pub async fn try_form_batch(&self) -> Option<Vec<InflightRequest>> {
        // Try to acquire permit for inflight batch
        let permit = self.inflight_semaphore.try_acquire();
        if permit.is_err() {
            debug!("Max inflight batches reached, waiting...");
            return None;
        }
        
        let mut queue = self.pending_queue.write().await;
        
        if queue.is_empty() {
            return None;
        }
        
        // Check if we should wait for more requests
        let oldest_age = queue.front()
            .map(|r| r.timestamp.elapsed())
            .unwrap_or(Duration::ZERO);
        
        let should_wait = queue.len() < self.max_batch_size 
            && oldest_age < Duration::from_millis(self.batch_timeout_ms);
        
        if should_wait {
            return None;
        }
        
        // Form batch (up to max_batch_size)
        let batch_size = queue.len().min(self.max_batch_size);
        let batch: Vec<_> = queue.drain(..batch_size).collect();
        
        self.current_queue_depth.store(queue.len(), Ordering::Relaxed);
        self.inflight_batches.fetch_add(1, Ordering::Relaxed);
        
        // Forget the permit so it stays acquired until complete_batch is called
        permit.unwrap().forget();
        
        info!(
            "Formed batch with {} requests (queue remaining: {}, inflight: {})",
            batch.len(),
            queue.len(),
            self.inflight_batches.load(Ordering::Relaxed)
        );
        
        Some(batch)
    }
    
    /// Mark batch as complete and release semaphore
    pub fn complete_batch(&self, batch_size: usize, processing_time_ms: u64) {
        self.processed_batches.fetch_add(1, Ordering::Relaxed);
        self.processed_requests.fetch_add(batch_size as u64, Ordering::Relaxed);
        self.total_latency_ms.fetch_add(processing_time_ms, Ordering::Relaxed);
        self.inflight_batches.fetch_sub(1, Ordering::Relaxed);
        
        // Release one permit back to the semaphore
        self.inflight_semaphore.add_permits(1);
        
        debug!(
            "Batch completed: {} requests in {} ms (inflight: {})",
            batch_size,
            processing_time_ms,
            self.inflight_batches.load(Ordering::Relaxed)
        );
    }
    
    /// Get the adaptive batch timeout based on queue depth
    pub fn get_adaptive_timeout(&self) -> Duration {
        let queue_depth = self.current_queue_depth.load(Ordering::Relaxed);
        
        let timeout_ms = match queue_depth {
            0..=2 => self.batch_timeout_ms,
            3..=5 => self.batch_timeout_ms / 2,
            6..=10 => self.batch_timeout_ms / 4,
            11..=20 => self.batch_timeout_ms / 8,
            _ => self.batch_timeout_ms / 16,
        };
        
        Duration::from_millis(timeout_ms)
    }
    
    /// Get current queue depth
    pub fn queue_depth(&self) -> usize {
        self.current_queue_depth.load(Ordering::Relaxed)
    }
    
    /// Get number of inflight batches
    pub fn inflight_count(&self) -> usize {
        self.inflight_batches.load(Ordering::Relaxed)
    }
    
    /// Check if we can accept more batches
    pub fn can_accept_batch(&self) -> bool {
        self.inflight_semaphore.available_permits() > 0
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> InflightBatchStats {
        let batches = self.processed_batches.load(Ordering::Relaxed);
        let requests = self.processed_requests.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        
        let avg_latency = if batches > 0 {
            total_latency / batches
        } else {
            0
        };
        
        let avg_batch_size = if batches > 0 {
            requests as f64 / batches as f64
        } else {
            0.0
        };
        
        InflightBatchStats {
            processed_batches: batches,
            processed_requests: requests,
            avg_latency_ms: avg_latency,
            avg_batch_size,
            current_queue_depth: self.current_queue_depth.load(Ordering::Relaxed),
            peak_queue_depth: self.peak_queue_depth.load(Ordering::Relaxed),
            inflight_batches: self.inflight_batches.load(Ordering::Relaxed),
            max_inflight_batches: self.max_inflight_batches,
        }
    }
    
    /// Clear all pending requests (returns count)
    pub async fn clear_queue(&self) -> usize {
        let mut queue = self.pending_queue.write().await;
        let count = queue.len();
        
        // Send errors to all pending requests
        for request in queue.drain(..) {
            let _ = request.response_tx.send(Err("Queue cleared".to_string()));
        }
        
        self.current_queue_depth.store(0, Ordering::Relaxed);
        warn!("Cleared {} pending requests from queue", count);
        count
    }
}

#[derive(Debug, Clone)]
pub struct InflightBatchStats {
    pub processed_batches: u64,
    pub processed_requests: u64,
    pub avg_latency_ms: u64,
    pub avg_batch_size: f64,
    pub current_queue_depth: usize,
    pub peak_queue_depth: usize,
    pub inflight_batches: usize,
    pub max_inflight_batches: usize,
}

impl Default for InflightBatchProcessor {
    fn default() -> Self {
        Self::new(32, 4, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_inflight_add_request() {
        let processor = InflightBatchProcessor::new(10, 2, 100);
        let (tx, _rx) = oneshot::channel();
        
        let request = InflightRequest {
            id: "test1".to_string(),
            model_name: "model1".to_string(),
            inputs: vec![json!({"data": "test"})],
            priority: 5,
            timestamp: Instant::now(),
            response_tx: tx,
        };
        
        assert!(processor.add_request(request).await.is_ok());
        assert_eq!(processor.queue_depth(), 1);
    }

    #[tokio::test]
    async fn test_inflight_priority_ordering() {
        let processor = InflightBatchProcessor::new(10, 2, 100);
        
        // Add requests with different priorities
        for priority in [1, 5, 3, 10, 2] {
            let (tx, _rx) = oneshot::channel();
            let request = InflightRequest {
                id: format!("req_{}", priority),
                model_name: "model1".to_string(),
                inputs: vec![json!(priority)],
                priority,
                timestamp: Instant::now(),
                response_tx: tx,
            };
            processor.add_request(request).await.unwrap();
        }
        
        assert_eq!(processor.queue_depth(), 5);
        
        // Wait for timeout to form batch
        tokio::time::sleep(Duration::from_millis(150)).await;
        
        let batch = processor.try_form_batch().await.unwrap();
        
        // Should be sorted by priority (highest first)
        assert_eq!(batch[0].priority, 10);
        assert_eq!(batch[1].priority, 5);
        assert_eq!(batch[2].priority, 3);
    }

    #[tokio::test]
    async fn test_inflight_batch_formation() {
        let processor = InflightBatchProcessor::new(3, 2, 100);
        
        // Add 5 requests
        for i in 0..5 {
            let (tx, _rx) = oneshot::channel();
            let request = InflightRequest {
                id: format!("req_{}", i),
                model_name: "model1".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            };
            processor.add_request(request).await.unwrap();
        }
        
        // Should form batch of max_batch_size (3)
        tokio::time::sleep(Duration::from_millis(150)).await;
        let batch = processor.try_form_batch().await.unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(processor.queue_depth(), 2);
    }

    #[tokio::test]
    async fn test_inflight_max_concurrent_batches() {
        let processor = Arc::new(InflightBatchProcessor::new(2, 2, 50));
        
        // Add enough requests for 3 batches
        for i in 0..6 {
            let (tx, _rx) = oneshot::channel();
            let request = InflightRequest {
                id: format!("req_{}", i),
                model_name: "model1".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            };
            processor.add_request(request).await.unwrap();
        }
        
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Form first batch
        let batch1 = processor.try_form_batch().await.unwrap();
        assert_eq!(batch1.len(), 2);
        assert_eq!(processor.inflight_count(), 1);
        
        // Form second batch
        let batch2 = processor.try_form_batch().await.unwrap();
        assert_eq!(batch2.len(), 2);
        assert_eq!(processor.inflight_count(), 2);
        
        // Third should fail (max inflight reached)
        let batch3 = processor.try_form_batch().await;
        assert!(batch3.is_none());
        
        // Complete first batch
        processor.complete_batch(batch1.len(), 50);
        assert_eq!(processor.inflight_count(), 1);
        
        // Now can form third batch
        tokio::time::sleep(Duration::from_millis(100)).await;
        let batch3 = processor.try_form_batch().await.unwrap();
        assert_eq!(batch3.len(), 2);
    }

    #[tokio::test]
    async fn test_inflight_adaptive_timeout() {
        let processor = InflightBatchProcessor::new(10, 2, 100);
        
        // Empty queue - full timeout
        assert_eq!(processor.get_adaptive_timeout(), Duration::from_millis(100));
        
        // Add 4 requests - half timeout
        for i in 0..4 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("req_{}", i),
                model_name: "model1".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            }).await.unwrap();
        }
        assert_eq!(processor.get_adaptive_timeout(), Duration::from_millis(50));
        
        // Add more - quarter timeout
        for i in 4..8 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("req_{}", i),
                model_name: "model1".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            }).await.unwrap();
        }
        assert_eq!(processor.get_adaptive_timeout(), Duration::from_millis(25));
    }

    #[tokio::test]
    async fn test_inflight_stats() {
        let processor = InflightBatchProcessor::new(3, 2, 100);
        
        // Add and process some batches
        for i in 0..6 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("req_{}", i),
                model_name: "model1".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            }).await.unwrap();
        }
        
        tokio::time::sleep(Duration::from_millis(150)).await;
        let batch1 = processor.try_form_batch().await.unwrap();
        processor.complete_batch(batch1.len(), 50);
        
        let stats = processor.get_stats();
        assert_eq!(stats.processed_batches, 1);
        assert_eq!(stats.processed_requests, 3);
        assert_eq!(stats.avg_latency_ms, 50);
        assert_eq!(stats.avg_batch_size, 3.0);
        assert_eq!(stats.current_queue_depth, 3);
    }

    #[tokio::test]
    async fn test_inflight_clear_queue() {
        let processor = InflightBatchProcessor::new(10, 2, 100);
        
        for i in 0..5 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("req_{}", i),
                model_name: "model1".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            }).await.unwrap();
        }
        
        assert_eq!(processor.queue_depth(), 5);
        
        let cleared = processor.clear_queue().await;
        assert_eq!(cleared, 5);
        assert_eq!(processor.queue_depth(), 0);
    }

    #[tokio::test]
    async fn test_inflight_peak_queue_depth() {
        let processor = InflightBatchProcessor::new(10, 2, 100);
        
        // Add 5 requests
        for i in 0..5 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("req_{}", i),
                model_name: "model1".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            }).await.unwrap();
        }
        
        let stats = processor.get_stats();
        assert_eq!(stats.peak_queue_depth, 5);
        
        // Form batch (reduces queue)
        tokio::time::sleep(Duration::from_millis(150)).await;
        processor.try_form_batch().await;
        
        let stats = processor.get_stats();
        assert_eq!(stats.peak_queue_depth, 5); // Peak unchanged
        assert!(stats.current_queue_depth < 5); // Current reduced
    }

    #[tokio::test]
    async fn test_inflight_default() {
        let processor = InflightBatchProcessor::default();
        let stats = processor.get_stats();
        
        assert_eq!(stats.processed_batches, 0);
        assert_eq!(stats.max_inflight_batches, 4);
    }
}
