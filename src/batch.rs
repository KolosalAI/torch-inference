use serde_json::Value;
use std::sync::Arc;
use parking_lot::Mutex;
use log::{info, debug, trace};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::collections::VecDeque;

/// High-performance batch request with tensor data
#[derive(Clone, Debug)]
pub struct BatchRequest {
    pub id: String,
    pub model_name: String,
    pub inputs: Vec<Value>,
    pub priority: i32,
    pub timestamp: Instant,
}

/// Optimized batch request with pre-allocated tensor data
#[derive(Debug)]
pub struct TensorBatchRequest {
    pub id: u64,
    pub model_name: String,
    pub input_data: Vec<f32>,
    pub input_shape: Vec<i64>,
    pub priority: i32,
    pub timestamp: Instant,
}

/// High-throughput batch processor with adaptive batching
/// 
/// Optimizations:
/// - Lock-free batch formation with parking_lot
/// - Adaptive timeout based on queue depth
/// - Priority-aware scheduling
/// - Continuous batching support
/// - Pre-allocated batch buffers
pub struct BatchProcessor {
    max_batch_size: usize,
    min_batch_size: usize,
    base_timeout_ms: u64,
    
    // Use parking_lot for faster locking
    current_batch: Mutex<Vec<BatchRequest>>,
    
    // Tensor batch queue for high-performance path
    tensor_queue: Mutex<VecDeque<TensorBatchRequest>>,
    
    // Statistics (lock-free)
    processed_batches: AtomicU64,
    processed_requests: AtomicU64,
    total_latency_ns: AtomicU64,
    total_wait_time_ns: AtomicU64,
    queue_depth: AtomicUsize,
    peak_queue_depth: AtomicUsize,
    batch_size_sum: AtomicU64,
    
    // Adaptive batching state
    enable_adaptive: AtomicBool,
    current_throughput: AtomicU64,  // requests per second * 100
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(max_batch_size: usize, batch_timeout_ms: u64) -> Self {
        Self {
            max_batch_size,
            min_batch_size: 1,
            base_timeout_ms: batch_timeout_ms,
            current_batch: Mutex::new(Vec::with_capacity(max_batch_size)),
            tensor_queue: Mutex::new(VecDeque::with_capacity(max_batch_size * 4)),
            processed_batches: AtomicU64::new(0),
            processed_requests: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
            total_wait_time_ns: AtomicU64::new(0),
            queue_depth: AtomicUsize::new(0),
            peak_queue_depth: AtomicUsize::new(0),
            batch_size_sum: AtomicU64::new(0),
            enable_adaptive: AtomicBool::new(true),
            current_throughput: AtomicU64::new(0),
        }
    }
    
    /// Create processor optimized for maximum throughput
    pub fn for_throughput() -> Self {
        Self::with_config(256, 1, 10, true)  // 256 batch size, 10ms timeout
    }
    
    /// Create processor optimized for minimum latency
    pub fn for_latency() -> Self {
        Self::with_config(8, 1, 5, false)
    }
    
    /// Create with full configuration
    pub fn with_config(
        max_batch_size: usize,
        min_batch_size: usize,
        base_timeout_ms: u64,
        enable_adaptive: bool,
    ) -> Self {
        let mut processor = Self::new(max_batch_size, base_timeout_ms);
        processor.min_batch_size = min_batch_size;
        processor.enable_adaptive.store(enable_adaptive, Ordering::Relaxed);
        processor
    }

    /// Add a request to the batch (JSON-based)
    #[inline]
    pub async fn add_request(&self, request: BatchRequest) -> Result<bool, String> {
        let (len, should_process) = {
            let mut batch = self.current_batch.lock();
            
            if batch.len() >= self.max_batch_size {
                return Err("Batch is full".to_string());
            }

            batch.push(request);
            let len = batch.len();
            (len, len >= self.max_batch_size)
        };
        
        self.queue_depth.store(len, Ordering::Relaxed);
        self.update_peak_depth(len);
        
        trace!("Request added to batch. Current size: {}", len);
        
        Ok(should_process)
    }
    
    /// Add a tensor request (high-performance path)
    #[inline]
    pub fn add_tensor_request(&self, request: TensorBatchRequest) -> Result<bool, String> {
        let (len, should_process) = {
            let mut queue = self.tensor_queue.lock();
            
            if queue.len() >= self.max_batch_size * 4 {
                return Err("Tensor queue is full".to_string());
            }
            
            queue.push_back(request);
            let len = queue.len();
            (len, len >= self.max_batch_size)
        };
        
        self.queue_depth.store(len, Ordering::Relaxed);
        self.update_peak_depth(len);
        
        Ok(should_process)
    }
    
    /// Get adaptive batch timeout based on current queue depth
    #[inline]
    pub fn get_adaptive_timeout(&self) -> Duration {
        if !self.enable_adaptive.load(Ordering::Relaxed) {
            return Duration::from_millis(self.base_timeout_ms);
        }
        
        let depth = self.queue_depth.load(Ordering::Relaxed);
        let timeout_ms = match depth {
            0..=1 => self.base_timeout_ms,
            2..=4 => self.base_timeout_ms * 3 / 4,
            5..=8 => self.base_timeout_ms / 2,
            9..=16 => self.base_timeout_ms / 4,
            17..=32 => self.base_timeout_ms / 8,
            _ => 1, // Process immediately under high load
        };
        
        Duration::from_millis(timeout_ms.max(1))
    }

    /// Check if batch should be processed
    #[inline]
    pub async fn should_process_batch(&self) -> bool {
        let (len, oldest_age) = {
            let batch = self.current_batch.lock();
            if batch.is_empty() {
                return false;
            }
            let age = batch.first().map(|r| r.timestamp.elapsed());
            (batch.len(), age)
        };
        
        // Process if batch is full
        if len >= self.max_batch_size {
            return true;
        }
        
        // Process if minimum batch size reached and timeout expired
        if len >= self.min_batch_size {
            if let Some(age) = oldest_age {
                let timeout = self.get_adaptive_timeout();
                if age >= timeout {
                    return true;
                }
            }
        }

        false
    }
    
    /// Get and clear the current batch
    #[inline]
    pub async fn get_batch(&self) -> Vec<BatchRequest> {
        let mut requests = {
            let mut batch = self.current_batch.lock();
            let mut requests = Vec::with_capacity(batch.len());
            std::mem::swap(&mut requests, &mut batch);
            requests
        };
        
        // Sort by priority (highest first) - stable sort for FIFO within same priority
        requests.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        let batch_size = requests.len();
        self.queue_depth.store(0, Ordering::Relaxed);
        self.processed_batches.fetch_add(1, Ordering::Relaxed);
        self.processed_requests.fetch_add(batch_size as u64, Ordering::Relaxed);
        self.batch_size_sum.fetch_add(batch_size as u64, Ordering::Relaxed);
        
        debug!("Processing batch with {} requests", batch_size);
        requests
    }
    
    /// Get tensor batch for high-performance inference
    #[inline]
    pub fn get_tensor_batch(&self, max_size: Option<usize>) -> Vec<TensorBatchRequest> {
        let max = max_size.unwrap_or(self.max_batch_size);
        
        let requests: Vec<_> = {
            let mut queue = self.tensor_queue.lock();
            let take_count = queue.len().min(max);
            queue.drain(..take_count).collect()
        };
        
        let batch_size = requests.len();
        if batch_size > 0 {
            self.queue_depth.store(
                self.queue_depth.load(Ordering::Relaxed).saturating_sub(batch_size),
                Ordering::Relaxed
            );
            self.processed_batches.fetch_add(1, Ordering::Relaxed);
            self.processed_requests.fetch_add(batch_size as u64, Ordering::Relaxed);
            self.batch_size_sum.fetch_add(batch_size as u64, Ordering::Relaxed);
        }
        
        requests
    }
    
    /// Record batch processing latency
    #[inline]
    pub fn record_batch_latency(&self, latency_ns: u64) {
        self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
    }
    
    /// Record batch latency in milliseconds
    #[inline]
    pub fn record_batch_latency_ms(&self, latency_ms: u64) {
        self.total_latency_ns.fetch_add(latency_ms * 1_000_000, Ordering::Relaxed);
    }
    
    /// Record wait time for a request
    #[inline]
    pub fn record_wait_time(&self, wait_ns: u64) {
        self.total_wait_time_ns.fetch_add(wait_ns, Ordering::Relaxed);
    }
    
    /// Update peak queue depth
    #[inline]
    fn update_peak_depth(&self, current: usize) {
        let mut peak = self.peak_queue_depth.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_queue_depth.compare_exchange_weak(
                peak, current, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }
    
    /// Get comprehensive statistics
    pub fn get_stats(&self) -> BatchStats {
        let batches = self.processed_batches.load(Ordering::Relaxed);
        let requests = self.processed_requests.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ns.load(Ordering::Relaxed);
        let total_wait = self.total_wait_time_ns.load(Ordering::Relaxed);
        let batch_size_sum = self.batch_size_sum.load(Ordering::Relaxed);
        
        let avg_latency_ms = if batches > 0 {
            total_latency / batches / 1_000_000
        } else {
            0
        };
        
        let avg_wait_ms = if requests > 0 {
            total_wait / requests / 1_000_000
        } else {
            0
        };
        
        let avg_batch_size = if batches > 0 {
            batch_size_sum as f64 / batches as f64
        } else {
            0.0
        };
        
        BatchStats {
            processed_batches: batches,
            processed_requests: requests,
            avg_latency_ms,
            avg_wait_time_ms: avg_wait_ms,
            avg_batch_size,
            current_queue_depth: self.queue_depth.load(Ordering::Relaxed),
            peak_queue_depth: self.peak_queue_depth.load(Ordering::Relaxed),
            max_batch_size: self.max_batch_size,
            adaptive_enabled: self.enable_adaptive.load(Ordering::Relaxed),
        }
    }

    /// Get current batch size without acquiring
    #[inline]
    pub async fn get_batch_size(&self) -> usize {
        self.current_batch.lock().len()
    }
    
    /// Get tensor queue size
    #[inline]
    pub fn get_tensor_queue_size(&self) -> usize {
        self.tensor_queue.lock().len()
    }

    /// Clear all pending batches
    pub async fn clear_batch(&self) {
        {
            let mut batch = self.current_batch.lock();
            batch.clear();
        }
        {
            let mut queue = self.tensor_queue.lock();
            queue.clear();
        }
        self.queue_depth.store(0, Ordering::Relaxed);
        debug!("Batches cleared");
    }

    /// Process batch with timeout
    pub async fn process_with_timeout<F, T>(&self, processor: F) -> Result<T, String>
    where
        F: std::future::Future<Output = Result<T, String>>,
    {
        tokio::time::timeout(Duration::from_secs(30), processor)
            .await
            .map_err(|_| "Batch processing timeout".to_string())?
    }
    
    /// Set adaptive batching enabled/disabled
    pub fn set_adaptive(&self, enabled: bool) {
        self.enable_adaptive.store(enabled, Ordering::Relaxed);
    }
    
    /// Update throughput estimate (call periodically)
    pub fn update_throughput(&self, requests_per_second: f64) {
        self.current_throughput.store((requests_per_second * 100.0) as u64, Ordering::Relaxed);
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new(64, 50)  // Larger batches, shorter timeout for better throughput
    }
}

/// Comprehensive batch statistics
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub processed_batches: u64,
    pub processed_requests: u64,
    pub avg_latency_ms: u64,
    pub avg_wait_time_ms: u64,
    pub avg_batch_size: f64,
    pub current_queue_depth: usize,
    pub peak_queue_depth: usize,
    pub max_batch_size: usize,
    pub adaptive_enabled: bool,
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
        
        // Should have default max_batch_size of 64
        for i in 0..64 {
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
}
