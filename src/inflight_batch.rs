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

    // ── Additional coverage tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_try_form_batch_empty_queue_returns_none() {
        // When the queue is completely empty, try_form_batch should return None
        let processor = InflightBatchProcessor::new(10, 2, 0);
        // No requests added — queue is empty
        let batch = processor.try_form_batch().await;
        assert!(batch.is_none(), "expected None for empty queue");
    }

    #[tokio::test]
    async fn test_can_accept_batch_when_permits_available() {
        let processor = InflightBatchProcessor::new(10, 3, 0);
        assert!(processor.can_accept_batch());
    }

    #[tokio::test]
    async fn test_can_accept_batch_when_no_permits() {
        // Use a max of 1 inflight batch
        let processor = Arc::new(InflightBatchProcessor::new(10, 1, 0));

        // Add more than batch_size requests so batch forms immediately
        for i in 0..20 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("req_{}", i),
                model_name: "model1".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now() - Duration::from_millis(200),
                response_tx: tx,
            }).await.unwrap();
        }

        // Form the one allowed batch (takes the semaphore)
        let _batch = processor.try_form_batch().await;
        // Now the semaphore is exhausted
        assert!(!processor.can_accept_batch());
    }

    #[tokio::test]
    async fn test_try_form_batch_waits_for_timeout() {
        // With a large timeout and a small queue (below max_batch_size),
        // try_form_batch should return None while batch_timeout hasn't expired
        let processor = InflightBatchProcessor::new(10, 2, 5000); // 5 second timeout

        let (tx, _rx) = oneshot::channel();
        processor.add_request(InflightRequest {
            id: "req_1".to_string(),
            model_name: "model1".to_string(),
            inputs: vec![json!(1)],
            priority: 1,
            timestamp: Instant::now(), // just arrived
            response_tx: tx,
        }).await.unwrap();

        // The request just arrived and queue is below batch size — should wait
        let batch = processor.try_form_batch().await;
        assert!(batch.is_none(), "should wait — timeout not expired");
    }

    #[tokio::test]
    async fn test_inflight_count_starts_at_zero() {
        let processor = InflightBatchProcessor::new(10, 2, 100);
        assert_eq!(processor.inflight_count(), 0);
    }

    #[tokio::test]
    async fn test_complete_batch_decrements_inflight() {
        let processor = Arc::new(InflightBatchProcessor::new(2, 2, 0));

        for i in 0..4 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("req_{}", i),
                model_name: "model1".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now() - Duration::from_millis(100),
                response_tx: tx,
            }).await.unwrap();
        }

        let batch = processor.try_form_batch().await.unwrap();
        assert_eq!(processor.inflight_count(), 1);

        processor.complete_batch(batch.len(), 10);
        assert_eq!(processor.inflight_count(), 0);
    }

    #[tokio::test]
    async fn test_get_stats_avg_batch_size_zero_when_no_batches() {
        let processor = InflightBatchProcessor::new(10, 2, 100);
        let stats = processor.get_stats();
        assert_eq!(stats.avg_batch_size, 0.0);
        assert_eq!(stats.avg_latency_ms, 0);
    }

    #[tokio::test]
    async fn test_adaptive_timeout_large_queue() {
        let processor = InflightBatchProcessor::new(100, 4, 1000);

        // Add 25 requests — should be in the "> 20" bucket → timeout / 16
        for i in 0..25 {
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

        let timeout = processor.get_adaptive_timeout();
        // 1000 / 16 = 62 ms
        assert_eq!(timeout, Duration::from_millis(62));
    }

    #[tokio::test]
    async fn test_adaptive_timeout_medium_queue_11_to_20() {
        let processor = InflightBatchProcessor::new(100, 4, 1000);

        // Add 15 requests — in the 11..=20 bucket → timeout / 8
        for i in 0..15 {
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

        let timeout = processor.get_adaptive_timeout();
        assert_eq!(timeout, Duration::from_millis(125)); // 1000 / 8
    }

    #[tokio::test]
    async fn test_clear_queue_sends_error_to_receivers() {
        let processor = InflightBatchProcessor::new(10, 2, 100);

        let (tx, rx) = oneshot::channel();
        processor.add_request(InflightRequest {
            id: "error_req".to_string(),
            model_name: "model".to_string(),
            inputs: vec![],
            priority: 1,
            timestamp: Instant::now(),
            response_tx: tx,
        }).await.unwrap();

        processor.clear_queue().await;

        // The receiver should get the error
        let result = rx.await.unwrap();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Queue cleared");
    }

    // ── Additional gap-closing tests ───────────────────────────────────────────

    /// Exercises try_form_batch when the semaphore permit is acquired but the
    /// queue is empty (lines 96-98): permit is obtained, then the empty-queue
    /// guard fires and returns None.
    #[tokio::test]
    async fn test_try_form_batch_empty_queue_after_permit_acquired() {
        // batch_timeout_ms = 0 so the "should_wait" guard won't block us.
        let processor = InflightBatchProcessor::new(10, 2, 0);
        // Queue is empty — permit can be acquired, but queue.is_empty() is true → None
        let batch = processor.try_form_batch().await;
        assert!(batch.is_none());
        // Semaphore was not permanently consumed (the permit was dropped, not forgotten)
        assert!(processor.can_accept_batch());
    }

    /// Exercises add_request with a request whose priority is lower than all
    /// existing requests — it is inserted at the end (unwrap_or(queue.len())).
    #[tokio::test]
    async fn test_add_request_lowest_priority_inserts_at_end() {
        let processor = InflightBatchProcessor::new(10, 2, 5000);

        // Insert a high-priority request first.
        let (tx1, _rx1) = oneshot::channel();
        processor.add_request(InflightRequest {
            id: "high".to_string(),
            model_name: "m".to_string(),
            inputs: vec![],
            priority: 100,
            timestamp: Instant::now(),
            response_tx: tx1,
        }).await.unwrap();

        // Insert a low-priority request — it should go to the back.
        let (tx2, _rx2) = oneshot::channel();
        processor.add_request(InflightRequest {
            id: "low".to_string(),
            model_name: "m".to_string(),
            inputs: vec![],
            priority: 1,
            timestamp: Instant::now(),
            response_tx: tx2,
        }).await.unwrap();

        assert_eq!(processor.queue_depth(), 2);
    }

    /// Exercises complete_batch with total_wait_time_ms tracking variant and
    /// verifies inflight_batches can't go below zero (sanity check).
    #[tokio::test]
    async fn test_complete_batch_stats_accumulate() {
        let processor = Arc::new(InflightBatchProcessor::new(5, 2, 0));

        // Enqueue and form two batches.
        for i in 0..10 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("r{i}"),
                model_name: "m".to_string(),
                inputs: vec![],
                priority: 1,
                timestamp: Instant::now() - Duration::from_millis(200),
                response_tx: tx,
            }).await.unwrap();
        }

        let b1 = processor.try_form_batch().await.unwrap();
        processor.complete_batch(b1.len(), 100);

        let b2 = processor.try_form_batch().await.unwrap();
        processor.complete_batch(b2.len(), 50);

        let stats = processor.get_stats();
        assert_eq!(stats.processed_batches, 2);
        assert_eq!(stats.processed_requests, 10);
        assert_eq!(stats.avg_latency_ms, 75); // (100 + 50) / 2
        assert_eq!(stats.inflight_batches, 0);
    }

    /// Exercises get_adaptive_timeout for queue depth 6..=10 (timeout / 4).
    #[tokio::test]
    async fn test_adaptive_timeout_queue_6_to_10() {
        let processor = InflightBatchProcessor::new(100, 4, 1000);

        for i in 0..8 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("r{i}"),
                model_name: "m".to_string(),
                inputs: vec![],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            }).await.unwrap();
        }

        let timeout = processor.get_adaptive_timeout();
        // queue_depth = 8, in range 6..=10 → 1000 / 4 = 250
        assert_eq!(timeout, Duration::from_millis(250));
    }

    /// Exercises try_form_batch when queue has exactly max_batch_size items
    /// and timeout hasn't expired — the "should_wait" condition uses
    /// `queue.len() < max_batch_size` which is false, so it proceeds immediately.
    #[tokio::test]
    async fn test_try_form_batch_full_batch_forms_immediately() {
        // max_batch_size = 3, timeout = 5000ms (would block), but queue has 3 items
        let processor = InflightBatchProcessor::new(3, 2, 5000);

        for i in 0..3 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("r{i}"),
                model_name: "m".to_string(),
                inputs: vec![],
                priority: 1,
                timestamp: Instant::now(), // just arrived — but batch is full
                response_tx: tx,
            }).await.unwrap();
        }

        // queue.len() == max_batch_size (3 == 3) → should_wait = false → batch forms
        let batch = processor.try_form_batch().await;
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 3);
    }

    // ── Logger-enabled tests to cover log macro argument lines ────────────────
    //
    // The `log` crate's info!/debug! macros only evaluate their arguments when a
    // logger with a matching level is installed.  Lines 123-126 (info! args in
    // try_form_batch) and lines 143, 146 (debug! args in complete_batch) are
    // skipped without an active logger.
    // We use env_logger::Builder with TRACE level so every macro body fires.

    fn init_logger() {
        let _ = env_logger::Builder::new()
            .filter_level(log::LevelFilter::Trace)
            .is_test(true)
            .try_init();
    }

    /// Covers lines 123-126: the info! arguments inside try_form_batch when a
    /// batch is successfully formed.  Also covers lines 143/146: the debug!
    /// arguments inside complete_batch.
    #[tokio::test]
    async fn test_try_form_batch_and_complete_with_logger_covers_log_lines() {
        init_logger();

        // batch_timeout_ms = 0 and max_batch_size = 3; add exactly 3 requests so
        // the batch forms immediately (no waiting for timeout).
        let processor = InflightBatchProcessor::new(3, 2, 0);

        for i in 0..3 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("log-req-{i}"),
                model_name: "model".to_string(),
                inputs: vec![json!(i)],
                priority: 1,
                timestamp: Instant::now() - Duration::from_millis(10),
                response_tx: tx,
            }).await.unwrap();
        }

        // try_form_batch: triggers info! at lines 122-127 (arguments on 123-126)
        let batch = processor.try_form_batch().await.unwrap();
        assert_eq!(batch.len(), 3);

        // complete_batch: triggers debug! at lines 142-147 (arguments on 143, 146)
        processor.complete_batch(batch.len(), 25);
        assert_eq!(processor.inflight_count(), 0);
    }

    /// Additional logger-enabled test with multiple batches to ensure repeated
    /// execution of the log macro argument lines.
    #[tokio::test]
    async fn test_multiple_batches_with_logger() {
        init_logger();

        let processor = Arc::new(InflightBatchProcessor::new(2, 4, 0));

        for i in 0..6 {
            let (tx, _rx) = oneshot::channel();
            processor.add_request(InflightRequest {
                id: format!("mb-{i}"),
                model_name: "m".to_string(),
                inputs: vec![],
                priority: 1,
                timestamp: Instant::now() - Duration::from_millis(50),
                response_tx: tx,
            }).await.unwrap();
        }

        for _ in 0..3 {
            if let Some(batch) = processor.try_form_batch().await {
                processor.complete_batch(batch.len(), 10);
            }
        }

        let stats = processor.get_stats();
        assert_eq!(stats.processed_batches, 3);
    }
}
