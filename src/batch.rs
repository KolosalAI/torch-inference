#![allow(dead_code)]
use serde_json::Value;
use std::sync::Arc;
use parking_lot::Mutex;
use log::{info, debug};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

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
    min_batch_size: usize,
    batch_timeout_ms: u64,
    adaptive_timeout_enabled: bool,
    current_batch: Arc<Mutex<Vec<BatchRequest>>>,
    processed_batches: AtomicU64,
    total_latency_ms: AtomicU64,
    queue_depth: AtomicUsize,
}

impl BatchProcessor {
    pub fn new(max_batch_size: usize, batch_timeout_ms: u64) -> Self {
        Self {
            max_batch_size,
            min_batch_size: 1,
            batch_timeout_ms,
            adaptive_timeout_enabled: true,
            current_batch: Arc::new(Mutex::new(Vec::new())),
            processed_batches: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            queue_depth: AtomicUsize::new(0),
        }
    }
    
    pub fn with_adaptive_batching(mut self, enabled: bool) -> Self {
        self.adaptive_timeout_enabled = enabled;
        self
    }
    
    pub fn with_min_batch_size(mut self, size: usize) -> Self {
        self.min_batch_size = size;
        self
    }

    pub async fn add_request(&self, request: BatchRequest) -> Result<bool, String> {
        // Use block to release lock immediately
        let len = {
            let mut batch = self.current_batch.lock();
            
            if batch.len() >= self.max_batch_size {
                return Err("Batch is full".to_string());
            }

            batch.push(request);
            batch.len()
        };
        
        self.queue_depth.store(len, Ordering::Relaxed);
        debug!("Request added to batch. Current size: {}", len);
        
        Ok(len >= self.max_batch_size)
    }

    pub async fn should_process_batch(&self) -> bool {
        let (len, oldest_age) = {
            let batch = self.current_batch.lock();
            if batch.is_empty() {
                return false;
            }
            let age = batch.first().map(|r| r.timestamp.elapsed());
            (batch.len(), age)
        };
        
        // Process immediately if max size reached
        if len >= self.max_batch_size {
            return true;
        }
        
        // Check min batch size
        if len < self.min_batch_size {
            return false;
        }

        if let Some(age) = oldest_age {
            let timeout = self.get_adaptive_timeout();
            return age >= Duration::from_millis(timeout);
        }

        false
    }
    
    fn get_adaptive_timeout(&self) -> u64 {
        if !self.adaptive_timeout_enabled {
            return self.batch_timeout_ms;
        }
        
        let queue_depth = self.queue_depth.load(Ordering::Relaxed);
        
        // Adaptive timeout based on queue depth
        // More items waiting = shorter timeout
        match queue_depth {
            0..=2 => self.batch_timeout_ms,
            3..=5 => self.batch_timeout_ms / 2,
            6..=10 => self.batch_timeout_ms / 4,
            _ => self.batch_timeout_ms / 8,
        }
    }

    pub async fn get_batch(&self) -> Vec<BatchRequest> {
        let mut requests = {
            let mut batch = self.current_batch.lock();
            // Pre-allocate with exact capacity to avoid reallocation
            let capacity = batch.len();
            let mut requests = Vec::with_capacity(capacity);
            std::mem::swap(&mut requests, &mut batch);
            requests
        };
        
        // Sort by priority (higher priority first) - uses unstable sort for better performance
        requests.sort_unstable_by(|a, b| b.priority.cmp(&a.priority));
        
        self.queue_depth.store(0, Ordering::Relaxed);
        self.processed_batches.fetch_add(1, Ordering::Relaxed);
        
        info!("Processing batch with {} requests (sorted by priority)", requests.len());
        requests
    }
    
    pub fn record_batch_latency(&self, latency_ms: u64) {
        self.total_latency_ms.fetch_add(latency_ms, Ordering::Relaxed);
    }
    
    pub fn get_stats(&self) -> BatchStats {
        let batches = self.processed_batches.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        let avg_latency = if batches > 0 {
            total_latency / batches
        } else {
            0
        };
        
        BatchStats {
            processed_batches: batches,
            avg_latency_ms: avg_latency,
            current_queue_depth: self.queue_depth.load(Ordering::Relaxed),
        }
    }

    pub async fn get_batch_size(&self) -> usize {
        self.current_batch.lock().len()
    }

    pub async fn clear_batch(&self) {
        let mut batch = self.current_batch.lock();
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

#[derive(Debug, Clone)]
pub struct BatchStats {
    pub processed_batches: u64,
    pub avg_latency_ms: u64,
    pub current_queue_depth: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ===== Basic Functionality Tests =====

    #[tokio::test]
    async fn test_batch_add_request() {
        let processor = BatchProcessor::new(10, 100);
        let request = BatchRequest {
            id: "test1".to_string(),
            model_name: "model1".to_string(),
            inputs: vec![serde_json::json!({"data": "test"})],
            priority: 1,
            timestamp: Instant::now(),
        };
        
        assert!(processor.add_request(request).await.is_ok());
        assert_eq!(processor.get_batch_size().await, 1);
    }

    #[tokio::test]
    async fn test_batch_full() {
        let processor = BatchProcessor::new(2, 100);
        
        for i in 0..2 {
            let request = BatchRequest {
                id: format!("test{}", i),
                model_name: "model1".to_string(),
                inputs: vec![serde_json::json!({"data": i})],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }
        
        let request = BatchRequest {
            id: "overflow".to_string(),
            model_name: "model1".to_string(),
            inputs: vec![serde_json::json!({"data": "extra"})],
            priority: 1,
            timestamp: Instant::now(),
        };
        
        assert!(processor.add_request(request).await.is_err());
    }

    #[tokio::test]
    async fn test_batch_should_process_when_full() {
        let processor = BatchProcessor::new(2, 1000);
        
        let request1 = BatchRequest {
            id: "test1".to_string(),
            model_name: "model1".to_string(),
            inputs: vec![serde_json::json!({"data": 1})],
            priority: 1,
            timestamp: Instant::now(),
        };
        processor.add_request(request1).await.unwrap();
        
        let request2 = BatchRequest {
            id: "test2".to_string(),
            model_name: "model1".to_string(),
            inputs: vec![serde_json::json!({"data": 2})],
            priority: 1,
            timestamp: Instant::now(),
        };
        processor.add_request(request2).await.unwrap();
        
        assert!(processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_batch_should_process_timeout() {
        let processor = BatchProcessor::new(10, 50);
        
        let request = BatchRequest {
            id: "test1".to_string(),
            model_name: "model1".to_string(),
            inputs: vec![serde_json::json!({"data": 1})],
            priority: 1,
            timestamp: Instant::now(),
        };
        processor.add_request(request).await.unwrap();
        
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_get_batch() {
        let processor = BatchProcessor::new(10, 100);
        
        for i in 0..3 {
            let request = BatchRequest {
                id: format!("test{}", i),
                model_name: "model1".to_string(),
                inputs: vec![serde_json::json!({"data": i})],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }
        
        let batch = processor.get_batch().await;
        assert_eq!(batch.len(), 3);
        assert_eq!(processor.get_batch_size().await, 0);
    }

    #[tokio::test]
    async fn test_clear_batch() {
        let processor = BatchProcessor::new(10, 100);
        
        let request = BatchRequest {
            id: "test1".to_string(),
            model_name: "model1".to_string(),
            inputs: vec![serde_json::json!({"data": 1})],
            priority: 1,
            timestamp: Instant::now(),
        };
        processor.add_request(request).await.unwrap();
        
        processor.clear_batch().await;
        assert_eq!(processor.get_batch_size().await, 0);
    }

    #[tokio::test]
    async fn test_process_with_timeout_success() {
        let processor = BatchProcessor::new(10, 100);
        
        let result = processor.process_with_timeout(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok("success".to_string())
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_process_with_timeout_failure() {
        let processor = BatchProcessor::new(10, 100);
        
        let result = processor.process_with_timeout(async {
            tokio::time::sleep(Duration::from_secs(35)).await;
            Ok("too_late".to_string())
        }).await;
        
        assert!(result.is_err());
    }

    // ===== Enterprise-Grade Tests =====

    #[tokio::test]
    async fn test_batch_concurrent_additions() {
        let processor = Arc::new(BatchProcessor::new(1000, 500));
        let mut handles = vec![];

        for i in 0..10 {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                for j in 0..50 {
                    let request = BatchRequest {
                        id: format!("thread_{}_req_{}", i, j),
                        model_name: "concurrent_model".to_string(),
                        inputs: vec![serde_json::json!({"thread": i, "req": j})],
                        priority: i as i32,
                        timestamp: Instant::now(),
                    };
                    processor_clone.add_request(request).await.ok();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(processor.get_batch_size().await, 500);
    }

    #[tokio::test]
    async fn test_batch_priority_handling() {
        let processor = BatchProcessor::new(100, 1000);
        
        // Add requests with different priorities (higher number = higher priority)
        for i in 0..10 {
            let request = BatchRequest {
                id: format!("req_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: i as i32,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }
        
        let batch = processor.get_batch().await;
        assert_eq!(batch.len(), 10);
        
        // Verify requests are sorted by priority (highest first)
        for i in 0..9 {
            assert!(batch[i].priority >= batch[i + 1].priority,
                   "Priority should be sorted in descending order");
        }
        
        // Verify all requests are present
        for i in 0..10 {
            assert!(batch.iter().any(|r| r.id == format!("req_{}", i)));
        }
    }

    #[tokio::test]
    async fn test_batch_timestamp_ordering() {
        let processor = BatchProcessor::new(100, 1000);
        
        let mut timestamps = vec![];
        for i in 0..5 {
            let request = BatchRequest {
                id: format!("req_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            timestamps.push(request.timestamp);
            processor.add_request(request).await.unwrap();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        let batch = processor.get_batch().await;
        
        // First request should be oldest
        assert!(batch[0].timestamp <= batch[batch.len() - 1].timestamp);
    }

    #[tokio::test]
    async fn test_batch_get_empties_queue() {
        let processor = BatchProcessor::new(10, 100);
        
        for i in 0..5 {
            let request = BatchRequest {
                id: format!("req_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }
        
        assert_eq!(processor.get_batch_size().await, 5);
        let _batch = processor.get_batch().await;
        assert_eq!(processor.get_batch_size().await, 0);
    }

    #[tokio::test]
    async fn test_batch_multiple_gets() {
        let processor = BatchProcessor::new(20, 100);
        
        // Add first batch
        for i in 0..5 {
            let request = BatchRequest {
                id: format!("batch1_req_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }
        
        let batch1 = processor.get_batch().await;
        assert_eq!(batch1.len(), 5);
        
        // Add second batch
        for i in 0..3 {
            let request = BatchRequest {
                id: format!("batch2_req_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }
        
        let batch2 = processor.get_batch().await;
        assert_eq!(batch2.len(), 3);
    }

    #[tokio::test]
    async fn test_batch_empty_batch_processing() {
        let processor = BatchProcessor::new(10, 100);
        
        assert!(!processor.should_process_batch().await);
        
        let batch = processor.get_batch().await;
        assert_eq!(batch.len(), 0);
    }

    #[tokio::test]
    async fn test_batch_size_boundary() {
        let processor = BatchProcessor::new(1, 100);
        
        let request = BatchRequest {
            id: "single".to_string(),
            model_name: "model".to_string(),
            inputs: vec![serde_json::json!(1)],
            priority: 1,
            timestamp: Instant::now(),
        };
        
        assert!(processor.add_request(request).await.is_ok());
        assert!(processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_batch_large_inputs() {
        let processor = BatchProcessor::new(10, 100);
        
        let large_input = serde_json::json!({
            "data": vec!["x"; 10000].join(""),
            "array": vec![1; 1000]
        });
        
        let request = BatchRequest {
            id: "large".to_string(),
            model_name: "model".to_string(),
            inputs: vec![large_input.clone()],
            priority: 1,
            timestamp: Instant::now(),
        };
        
        assert!(processor.add_request(request).await.is_ok());
        let batch = processor.get_batch().await;
        assert_eq!(batch[0].inputs[0], large_input);
    }

    #[tokio::test]
    async fn test_batch_timeout_precision() {
        let processor = BatchProcessor::new(10, 100);
        
        let start = Instant::now();
        let request = BatchRequest {
            id: "timeout_test".to_string(),
            model_name: "model".to_string(),
            inputs: vec![serde_json::json!(1)],
            priority: 1,
            timestamp: start,
        };
        processor.add_request(request).await.unwrap();
        
        // Should not be ready immediately
        assert!(!processor.should_process_batch().await);
        
        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(110)).await;
        
        // Should be ready after timeout
        assert!(processor.should_process_batch().await);
        
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_batch_concurrent_get_and_add() {
        let processor = Arc::new(BatchProcessor::new(100, 200));
        
        let processor_reader = Arc::clone(&processor);
        let reader = tokio::spawn(async move {
            for _ in 0..10 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                let _batch = processor_reader.get_batch().await;
            }
        });
        
        let processor_writer = Arc::clone(&processor);
        let writer = tokio::spawn(async move {
            for i in 0..50 {
                let request = BatchRequest {
                    id: format!("req_{}", i),
                    model_name: "model".to_string(),
                    inputs: vec![serde_json::json!(i)],
                    priority: 1,
                    timestamp: Instant::now(),
                };
                processor_writer.add_request(request).await.ok();
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
        
        reader.await.unwrap();
        writer.await.unwrap();
    }

    #[tokio::test]
    async fn test_batch_stress_test() {
        let processor = Arc::new(BatchProcessor::new(10000, 1000));
        let mut handles = vec![];
        
        // Multiple producers
        for i in 0..20 {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                for j in 0..100 {
                    let request = BatchRequest {
                        id: format!("stress_{}_{}", i, j),
                        model_name: "stress_model".to_string(),
                        inputs: vec![serde_json::json!({"producer": i, "item": j})],
                        priority: (i % 5) as i32,
                        timestamp: Instant::now(),
                    };
                    let _ = processor_clone.add_request(request).await;
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_batch_clear_during_processing() {
        let processor = BatchProcessor::new(10, 100);
        
        for i in 0..5 {
            let request = BatchRequest {
                id: format!("req_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }
        
        processor.clear_batch().await;
        assert_eq!(processor.get_batch_size().await, 0);
        assert!(!processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_batch_request_clone() {
        let request = BatchRequest {
            id: "test".to_string(),
            model_name: "model".to_string(),
            inputs: vec![serde_json::json!({"test": "data"})],
            priority: 5,
            timestamp: Instant::now(),
        };
        
        let cloned = request.clone();
        assert_eq!(request.id, cloned.id);
        assert_eq!(request.model_name, cloned.model_name);
        assert_eq!(request.priority, cloned.priority);
    }

    #[tokio::test]
    async fn test_batch_default_construction() {
        let processor = BatchProcessor::default();
        
        // Should have default max_batch_size of 32
        for i in 0..32 {
            let request = BatchRequest {
                id: format!("req_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            assert!(processor.add_request(request).await.is_ok());
        }
        
        assert!(processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_batch_timeout_with_multiple_requests() {
        let processor = BatchProcessor::new(100, 100);

        // Add requests over time
        for i in 0..5 {
            let request = BatchRequest {
                id: format!("req_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        // Wait for oldest to timeout
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(processor.should_process_batch().await);
    }

    // ===== Additional coverage tests =====

    #[tokio::test]
    async fn test_with_adaptive_batching_disabled() {
        let processor = BatchProcessor::new(10, 200)
            .with_adaptive_batching(false);

        let request = BatchRequest {
            id: "adapt_test".to_string(),
            model_name: "model".to_string(),
            inputs: vec![serde_json::json!(1)],
            priority: 1,
            timestamp: Instant::now(),
        };
        processor.add_request(request).await.unwrap();

        // With adaptive disabled and 1 item the timeout is 200ms; should not trigger yet
        assert!(!processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_with_adaptive_batching_enabled_high_queue() {
        // With adaptive enabled and many items in queue (> 10), timeout is /8
        // Use a base timeout large enough that base is slow, but /8 fires quickly.
        let processor = BatchProcessor::new(1000, 800);

        // Fill queue to > 10 so adaptive kicks in to /8 = 100ms
        for i in 0..11 {
            let request = BatchRequest {
                id: format!("q_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }

        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_with_min_batch_size_not_reached() {
        let processor = BatchProcessor::new(100, 50)
            .with_min_batch_size(5);

        // Add fewer than min_batch_size
        for i in 0..3 {
            let request = BatchRequest {
                id: format!("min_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }

        // Even after timeout, min_batch_size not reached — should not process
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_with_min_batch_size_reached() {
        let processor = BatchProcessor::new(100, 50)
            .with_min_batch_size(3);

        for i in 0..3 {
            let request = BatchRequest {
                id: format!("min_ok_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_get_stats_initial() {
        let processor = BatchProcessor::new(10, 100);
        let stats = processor.get_stats();
        assert_eq!(stats.processed_batches, 0);
        assert_eq!(stats.avg_latency_ms, 0);
        assert_eq!(stats.current_queue_depth, 0);
    }

    #[tokio::test]
    async fn test_get_stats_after_batch() {
        let processor = BatchProcessor::new(10, 100);

        let request = BatchRequest {
            id: "stats_req".to_string(),
            model_name: "model".to_string(),
            inputs: vec![serde_json::json!(1)],
            priority: 1,
            timestamp: Instant::now(),
        };
        processor.add_request(request).await.unwrap();
        let _batch = processor.get_batch().await;

        let stats = processor.get_stats();
        assert_eq!(stats.processed_batches, 1);
        assert_eq!(stats.current_queue_depth, 0);
    }

    #[tokio::test]
    async fn test_record_batch_latency() {
        let processor = BatchProcessor::new(10, 100);

        processor.record_batch_latency(50);
        processor.record_batch_latency(100);

        // Simulate a get_batch so processed_batches > 0 for avg
        let request = BatchRequest {
            id: "lat_req".to_string(),
            model_name: "model".to_string(),
            inputs: vec![serde_json::json!(1)],
            priority: 1,
            timestamp: Instant::now(),
        };
        processor.add_request(request).await.unwrap();
        let _batch = processor.get_batch().await;
        processor.record_batch_latency(75);

        let stats = processor.get_stats();
        // total_latency = 50 + 100 + 75 = 225, processed_batches = 1
        assert_eq!(stats.avg_latency_ms, 225);
    }

    #[tokio::test]
    async fn test_batch_stats_clone() {
        let stats = BatchStats {
            processed_batches: 5,
            avg_latency_ms: 42,
            current_queue_depth: 3,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.processed_batches, 5);
        assert_eq!(cloned.avg_latency_ms, 42);
        assert_eq!(cloned.current_queue_depth, 3);
    }

    #[test]
    fn test_batch_stats_debug() {
        let stats = BatchStats {
            processed_batches: 1,
            avg_latency_ms: 10,
            current_queue_depth: 0,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("BatchStats"));
    }

    #[tokio::test]
    async fn test_add_request_returns_true_when_full() {
        // max_batch_size=1, so adding 1 item fills the batch → returns true
        let processor = BatchProcessor::new(1, 100);
        let request = BatchRequest {
            id: "full_check".to_string(),
            model_name: "model".to_string(),
            inputs: vec![serde_json::json!(1)],
            priority: 1,
            timestamp: Instant::now(),
        };
        let result = processor.add_request(request).await.unwrap();
        assert!(result); // batch is full after this add
    }

    #[tokio::test]
    async fn test_add_request_returns_false_when_not_full() {
        let processor = BatchProcessor::new(10, 100);
        let request = BatchRequest {
            id: "not_full".to_string(),
            model_name: "model".to_string(),
            inputs: vec![serde_json::json!(1)],
            priority: 1,
            timestamp: Instant::now(),
        };
        let result = processor.add_request(request).await.unwrap();
        assert!(!result); // still space left
    }

    #[tokio::test]
    async fn test_adaptive_timeout_medium_queue() {
        // Queue depth 3..=5 → timeout/2; use 400ms base → effective 200ms
        let processor = BatchProcessor::new(1000, 400);

        for i in 0..4 {
            let request = BatchRequest {
                id: format!("med_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }

        tokio::time::sleep(Duration::from_millis(220)).await;
        assert!(processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_adaptive_timeout_high_queue_6_to_10() {
        // Queue depth 6..=10 → timeout/4; use 400ms base → effective 100ms
        let processor = BatchProcessor::new(1000, 400);

        for i in 0..8 {
            let request = BatchRequest {
                id: format!("high_{}", i),
                model_name: "model".to_string(),
                inputs: vec![serde_json::json!(i)],
                priority: 1,
                timestamp: Instant::now(),
            };
            processor.add_request(request).await.unwrap();
        }

        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(processor.should_process_batch().await);
    }

    #[tokio::test]
    async fn test_batch_request_debug() {
        let request = BatchRequest {
            id: "debug_req".to_string(),
            model_name: "debug_model".to_string(),
            inputs: vec![serde_json::json!({"key": "value"})],
            priority: 3,
            timestamp: Instant::now(),
        };
        let debug = format!("{:?}", request);
        assert!(debug.contains("BatchRequest"));
        assert!(debug.contains("debug_req"));
    }
}
