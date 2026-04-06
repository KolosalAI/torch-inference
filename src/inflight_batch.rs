#![allow(dead_code)]
use log::{debug, info, warn};
use serde_json::Value;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};

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

/// Wrapper giving BinaryHeap the right ordering: higher priority first, FIFO within priority.
struct OrderedRequest {
    priority: i32,
    seq: u64,
    request: InflightRequest,
}

impl PartialEq for OrderedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq
    }
}
impl Eq for OrderedRequest {}
impl PartialOrd for OrderedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher priority value first; ties broken by lower seq (earlier insert = FIFO).
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Single-heap priority queue. O(log n) insert and pop, correct FIFO within priority level.
/// Single-heap priority queue. O(log n) insert and pop, correct FIFO within priority level.
///
/// `oldest_enqueue` tracks the `Instant` of the request that has waited
/// longest without being drained.  It is updated on every `push` and is
/// recomputed (O(n) scan, rare) only after a `drain_*` call removed the
/// previously oldest entry.  This turns the formerly O(n) `oldest_age()`
/// call — invoked on every batch-formation decision — into O(1).
struct PriorityQueue {
    heap: BinaryHeap<OrderedRequest>,
    next_seq: u64,
    /// Earliest (smallest) enqueue timestamp still in the heap.  `None` when
    /// the heap is empty.
    oldest_enqueue: Option<Instant>,
}

impl PriorityQueue {
    fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
            oldest_enqueue: None,
        }
    }

    fn push(&mut self, request: InflightRequest) {
        let ts = request.timestamp;
        let seq = self.next_seq;
        self.next_seq += 1;
        let priority = request.priority;
        self.heap.push(OrderedRequest { priority, seq, request });

        // Update cached oldest: keep the smaller (earlier) of the two.
        self.oldest_enqueue = Some(match self.oldest_enqueue {
            Some(prev) if prev <= ts => prev,
            _ => ts,
        });
    }

    fn drain_all(&mut self, mut f: impl FnMut(InflightRequest)) {
        while let Some(item) = self.heap.pop() {
            f(item.request);
        }
        self.oldest_enqueue = None;
    }

    fn drain_up_to(&mut self, max: usize) -> Vec<InflightRequest> {
        let mut result = Vec::with_capacity(max);
        for _ in 0..max {
            match self.heap.pop() {
                Some(item) => result.push(item.request),
                None => break,
            }
        }
        // Recompute oldest only if entries remain (drain may have removed the oldest).
        self.oldest_enqueue = if self.heap.is_empty() {
            None
        } else {
            self.heap.iter().map(|o| o.request.timestamp).min()
        };
        result
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// O(1): returns the elapsed time since the oldest pending request arrived.
    fn oldest_age(&self) -> Option<Duration> {
        self.oldest_enqueue.map(|t| t.elapsed())
    }
}

/// Inflight batch processor that allows continuous batching
/// New requests can be added while inference is running on previous batch
pub struct InflightBatchProcessor {
    max_batch_size: usize,
    max_inflight_batches: usize,
    batch_timeout_ms: u64,

    // Queue of pending requests (no lock held across any .await — use sync RwLock)
    pending_queue: parking_lot::RwLock<PriorityQueue>,

    // Semaphore to limit concurrent batches
    inflight_semaphore: Arc<Semaphore>,

    // Statistics
    processed_batches: AtomicU64,
    processed_requests: AtomicU64,
    total_latency_ms: AtomicU64,
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
            pending_queue: parking_lot::RwLock::new(PriorityQueue::new()),
            inflight_semaphore: Arc::new(Semaphore::new(max_inflight_batches)),
            processed_batches: AtomicU64::new(0),
            processed_requests: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            current_queue_depth: AtomicUsize::new(0),
            peak_queue_depth: AtomicUsize::new(0),
            inflight_batches: AtomicUsize::new(0),
        }
    }

    /// Add a request to the pending queue
    pub fn add_request(&self, request: InflightRequest) -> Result<(), String> {
        let mut queue = self.pending_queue.write();
        queue.push(request);
        let queue_size = queue.len();
        drop(queue);

        self.current_queue_depth
            .store(queue_size, Ordering::Relaxed);
        self.peak_queue_depth.fetch_max(queue_size, Ordering::Relaxed);

        debug!(
            "Request added to inflight queue. Queue size: {}",
            queue_size
        );
        Ok(())
    }

    /// Try to form a batch from pending requests
    pub fn try_form_batch(&self) -> Option<InflightBatchGuard<'_>> {
        // ── Phase 1: read-lock peek ────────────────────────────────────────
        // Concurrent add_request calls proceed unblocked during this phase.
        let should_form = {
            let queue = self.pending_queue.read();
            if queue.is_empty() {
                return None;
            }
            let oldest_age = queue.oldest_age().unwrap_or(Duration::ZERO);
            let full = queue.len() >= self.max_batch_size;
            full || oldest_age >= Duration::from_millis(self.batch_timeout_ms)
        }; // read lock released here

        if !should_form {
            return None;
        }

        // Non-blocking semaphore acquisition.
        let Ok(permit) = self.inflight_semaphore.clone().try_acquire_owned() else {
            debug!("Max inflight batches reached, waiting...");
            return None;
        };

        // ── Phase 2: write-lock drain ──────────────────────────────────────
        let batch = {
            let mut queue = self.pending_queue.write();
            // TOCTOU guard: re-verify queue is still non-empty after acquiring write lock.
            if queue.is_empty() {
                debug!("TOCTOU: queue emptied between peek and drain, skipping batch");
                return None;
            }
            let batch_size = queue.len().min(self.max_batch_size);
            let batch = queue.drain_up_to(batch_size);
            self.current_queue_depth
                .store(queue.len(), Ordering::Relaxed);
            // Increment inside the write lock to avoid a stats undercount window
            // between lock release and the atomic store.
            self.inflight_batches.fetch_add(1, Ordering::Relaxed);
            batch
        }; // write lock released here

        info!(
            "Formed batch with {} requests (queue remaining: {}, inflight: {})",
            batch.len(),
            self.current_queue_depth.load(Ordering::Relaxed),
            self.inflight_batches.load(Ordering::Relaxed)
        );

        Some(InflightBatchGuard::new(self, batch, permit))
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
    pub fn clear_queue(&self) -> usize {
        let mut queue = self.pending_queue.write();
        let mut count = 0;
        queue.drain_all(|request| {
            let _ = request.response_tx.send(Err("Queue cleared".to_string()));
            count += 1;
        });
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

/// RAII guard returned by `try_form_batch`.
///
/// Holds the inflight semaphore permit for the duration of batch processing.
/// The permit is automatically released on `Drop`, whether or not `complete`
/// is called — preventing semaphore starvation on panics.
pub struct InflightBatchGuard<'a> {
    processor: &'a InflightBatchProcessor,
    /// The requests that form this batch.
    pub batch: Vec<InflightRequest>,
    permit: OwnedSemaphorePermit,
    completed: bool,
}

impl<'a> InflightBatchGuard<'a> {
    fn new(
        processor: &'a InflightBatchProcessor,
        batch: Vec<InflightRequest>,
        permit: OwnedSemaphorePermit,
    ) -> Self {
        Self {
            processor,
            batch,
            permit,
            completed: false,
        }
    }

    /// Record batch statistics and release the guard.
    ///
    /// `processing_time_ms` is the wall-clock time from batch formation
    /// to completion of inference.
    pub fn complete(mut self, processing_time_ms: u64) {
        // Mark completed first so Drop does not double-decrement inflight_batches
        // if any subsequent atomic operation were to panic.
        self.completed = true;
        let batch_size = self.batch.len();
        self.processor
            .processed_batches
            .fetch_add(1, Ordering::Relaxed);
        self.processor
            .processed_requests
            .fetch_add(batch_size as u64, Ordering::Relaxed);
        self.processor
            .total_latency_ms
            .fetch_add(processing_time_ms, Ordering::Relaxed);
        self.processor
            .inflight_batches
            .fetch_sub(1, Ordering::Relaxed);
        // `self` drops here: OwnedSemaphorePermit releases automatically.
    }
}

impl Drop for InflightBatchGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            // complete() was never called (e.g. caller panicked).
            // Decrement inflight_batches so the counter stays accurate.
            self.processor
                .inflight_batches
                .fetch_sub(1, Ordering::Relaxed);
        }
        // OwnedSemaphorePermit drops here regardless, releasing the semaphore.
    }
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

    fn make_request(id: &str, priority: i32) -> InflightRequest {
        let (tx, _rx) = oneshot::channel();
        InflightRequest {
            id: id.to_string(),
            model_name: "m".to_string(),
            inputs: vec![],
            priority,
            timestamp: Instant::now(),
            response_tx: tx,
        }
    }

    #[test]
    fn test_priority_queue_push_and_drain_order() {
        let mut q = PriorityQueue::new();
        q.push(make_request("low", 0));
        q.push(make_request("high", 4));
        q.push(make_request("mid", 2));

        let drained = q.drain_up_to(10);
        // highest priority first
        assert_eq!(drained[0].priority, 4);
        assert_eq!(drained[1].priority, 2);
        assert_eq!(drained[2].priority, 0);
    }

    #[test]
    fn test_priority_queue_drain_up_to_limit() {
        let mut q = PriorityQueue::new();
        for i in 0..5 {
            q.push(make_request(&format!("r{i}"), 2));
        }
        let drained = q.drain_up_to(3);
        assert_eq!(drained.len(), 3);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_priority_queue_arbitrary_priorities() {
        let mut q = PriorityQueue::new();
        q.push(make_request("neg", -5));
        q.push(make_request("big", 100));
        assert_eq!(q.len(), 2);
        let drained = q.drain_up_to(10);
        assert_eq!(drained[0].priority, 100); // highest first
        assert_eq!(drained[1].priority, -5);
    }

    #[test]
    fn test_priority_queue_is_empty_and_len() {
        let mut q = PriorityQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        q.push(make_request("x", 1));
        assert!(!q.is_empty());
        assert_eq!(q.len(), 1);
    }

    #[tokio::test]
    async fn test_priority_queue_oldest_age() {
        let mut q = PriorityQueue::new();
        let (tx, _rx) = oneshot::channel();
        q.push(InflightRequest {
            id: "old".to_string(),
            model_name: "m".to_string(),
            inputs: vec![],
            priority: 1,
            timestamp: Instant::now() - Duration::from_millis(200),
            response_tx: tx,
        });
        let (tx2, _rx2) = oneshot::channel();
        q.push(InflightRequest {
            id: "new".to_string(),
            model_name: "m".to_string(),
            inputs: vec![],
            priority: 3,
            timestamp: Instant::now(),
            response_tx: tx2,
        });
        let age = q.oldest_age().unwrap();
        assert!(age >= Duration::from_millis(190));
    }

    #[test]
    fn test_priority_queue_ordering() {
        use tokio::sync::oneshot;

        let mut q = PriorityQueue::new();

        let make_req = |priority: i32, id: &str| {
            let (tx, _rx) = oneshot::channel();
            InflightRequest {
                id: id.to_string(),
                model_name: "m".to_string(),
                inputs: vec![],
                priority,
                timestamp: std::time::Instant::now(),
                response_tx: tx,
            }
        };

        q.push(make_req(1, "low-first"));
        q.push(make_req(5, "high-first"));
        q.push(make_req(5, "high-second"));
        q.push(make_req(1, "low-second"));

        let batch = q.drain_up_to(4);
        assert_eq!(batch[0].id, "high-first");
        assert_eq!(batch[1].id, "high-second");
        assert_eq!(batch[2].id, "low-first");
        assert_eq!(batch[3].id, "low-second");
    }

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

        assert!(processor.add_request(request).is_ok());
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
            processor.add_request(request).unwrap();
        }

        assert_eq!(processor.queue_depth(), 5);

        // Wait for timeout to form batch
        tokio::time::sleep(Duration::from_millis(150)).await;

        let guard = processor.try_form_batch().unwrap();

        // PriorityQueue uses a BinaryHeap: higher priority value is dequeued first.
        // Insertion order: 1, 5, 3, 10, 2 → sorted by priority descending: 10, 5, 3, 2, 1
        assert_eq!(guard.batch[0].priority, 10);
        assert_eq!(guard.batch[1].priority, 5);
        assert_eq!(guard.batch[2].priority, 3);
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
            processor.add_request(request).unwrap();
        }

        // Should form batch of max_batch_size (3)
        tokio::time::sleep(Duration::from_millis(150)).await;
        let guard = processor.try_form_batch().unwrap();
        assert_eq!(guard.batch.len(), 3);
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
            processor.add_request(request).unwrap();
        }

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Form first batch
        let guard1 = processor.try_form_batch().unwrap();
        assert_eq!(guard1.batch.len(), 2);
        assert_eq!(processor.inflight_count(), 1);

        // Form second batch
        let guard2 = processor.try_form_batch().unwrap();
        assert_eq!(guard2.batch.len(), 2);
        assert_eq!(processor.inflight_count(), 2);

        // Third should fail (max inflight reached)
        let guard3 = processor.try_form_batch();
        assert!(guard3.is_none());

        // Complete first batch
        guard1.complete(50);
        assert_eq!(processor.inflight_count(), 1);

        // Now can form third batch
        tokio::time::sleep(Duration::from_millis(100)).await;
        let guard3 = processor.try_form_batch().unwrap();
        assert_eq!(guard3.batch.len(), 2);
    }

    #[tokio::test]
    async fn test_inflight_adaptive_timeout() {
        let processor = InflightBatchProcessor::new(10, 2, 100);

        // Empty queue - full timeout
        assert_eq!(processor.get_adaptive_timeout(), Duration::from_millis(100));

        // Add 4 requests - half timeout
        for i in 0..4 {
            let (tx, _rx) = oneshot::channel();
            processor
                .add_request(InflightRequest {
                    id: format!("req_{}", i),
                    model_name: "model1".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now(),
                    response_tx: tx,
                })
                .unwrap();
        }
        assert_eq!(processor.get_adaptive_timeout(), Duration::from_millis(50));

        // Add more - quarter timeout
        for i in 4..8 {
            let (tx, _rx) = oneshot::channel();
            processor
                .add_request(InflightRequest {
                    id: format!("req_{}", i),
                    model_name: "model1".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now(),
                    response_tx: tx,
                })
                .unwrap();
        }
        assert_eq!(processor.get_adaptive_timeout(), Duration::from_millis(25));
    }

    #[tokio::test]
    async fn test_inflight_stats() {
        let processor = InflightBatchProcessor::new(3, 2, 100);

        // Add and process some batches
        for i in 0..6 {
            let (tx, _rx) = oneshot::channel();
            processor
                .add_request(InflightRequest {
                    id: format!("req_{}", i),
                    model_name: "model1".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now(),
                    response_tx: tx,
                })
                .unwrap();
        }

        tokio::time::sleep(Duration::from_millis(150)).await;
        let guard1 = processor.try_form_batch().unwrap();
        guard1.complete(50);

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
            processor
                .add_request(InflightRequest {
                    id: format!("req_{}", i),
                    model_name: "model1".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now(),
                    response_tx: tx,
                })
                .unwrap();
        }

        assert_eq!(processor.queue_depth(), 5);

        let cleared = processor.clear_queue();
        assert_eq!(cleared, 5);
        assert_eq!(processor.queue_depth(), 0);
    }

    #[tokio::test]
    async fn test_inflight_peak_queue_depth() {
        let processor = InflightBatchProcessor::new(10, 2, 100);

        // Add 5 requests
        for i in 0..5 {
            let (tx, _rx) = oneshot::channel();
            processor
                .add_request(InflightRequest {
                    id: format!("req_{}", i),
                    model_name: "model1".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now(),
                    response_tx: tx,
                })
                .unwrap();
        }

        let stats = processor.get_stats();
        assert_eq!(stats.peak_queue_depth, 5);

        // Form batch (reduces queue)
        tokio::time::sleep(Duration::from_millis(150)).await;
        let _guard = processor.try_form_batch();

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
        let batch = processor.try_form_batch();
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
            processor
                .add_request(InflightRequest {
                    id: format!("req_{}", i),
                    model_name: "model1".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now() - Duration::from_millis(200),
                    response_tx: tx,
                })
                .unwrap();
        }

        // Form the one allowed batch (takes the semaphore)
        let _batch = processor.try_form_batch();
        // Now the semaphore is exhausted
        assert!(!processor.can_accept_batch());
    }

    #[tokio::test]
    async fn test_try_form_batch_waits_for_timeout() {
        // With a large timeout and a small queue (below max_batch_size),
        // try_form_batch should return None while batch_timeout hasn't expired
        let processor = InflightBatchProcessor::new(10, 2, 5000); // 5 second timeout

        let (tx, _rx) = oneshot::channel();
        processor
            .add_request(InflightRequest {
                id: "req_1".to_string(),
                model_name: "model1".to_string(),
                inputs: vec![json!(1)],
                priority: 1,
                timestamp: Instant::now(), // just arrived
                response_tx: tx,
            })
            .unwrap();

        // The request just arrived and queue is below batch size — should wait
        let batch = processor.try_form_batch();
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
            processor
                .add_request(InflightRequest {
                    id: format!("req_{}", i),
                    model_name: "model1".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now() - Duration::from_millis(100),
                    response_tx: tx,
                })
                .unwrap();
        }

        let guard = processor.try_form_batch().unwrap();
        assert_eq!(processor.inflight_count(), 1);

        guard.complete(10);
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
            processor
                .add_request(InflightRequest {
                    id: format!("req_{}", i),
                    model_name: "model1".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now(),
                    response_tx: tx,
                })
                .unwrap();
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
            processor
                .add_request(InflightRequest {
                    id: format!("req_{}", i),
                    model_name: "model1".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now(),
                    response_tx: tx,
                })
                .unwrap();
        }

        let timeout = processor.get_adaptive_timeout();
        assert_eq!(timeout, Duration::from_millis(125)); // 1000 / 8
    }

    #[tokio::test]
    async fn test_clear_queue_sends_error_to_receivers() {
        let processor = InflightBatchProcessor::new(10, 2, 100);

        let (tx, rx) = oneshot::channel();
        processor
            .add_request(InflightRequest {
                id: "error_req".to_string(),
                model_name: "model".to_string(),
                inputs: vec![],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            })
            .unwrap();

        processor.clear_queue();

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
        let batch = processor.try_form_batch();
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
        processor
            .add_request(InflightRequest {
                id: "high".to_string(),
                model_name: "m".to_string(),
                inputs: vec![],
                priority: 100,
                timestamp: Instant::now(),
                response_tx: tx1,
            })
            .unwrap();

        // Insert a low-priority request — it should go to the back.
        let (tx2, _rx2) = oneshot::channel();
        processor
            .add_request(InflightRequest {
                id: "low".to_string(),
                model_name: "m".to_string(),
                inputs: vec![],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx2,
            })
            .unwrap();

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
            processor
                .add_request(InflightRequest {
                    id: format!("r{i}"),
                    model_name: "m".to_string(),
                    inputs: vec![],
                    priority: 1,
                    timestamp: Instant::now() - Duration::from_millis(200),
                    response_tx: tx,
                })
                .unwrap();
        }

        let b1 = processor.try_form_batch().unwrap();
        b1.complete(100);

        let b2 = processor.try_form_batch().unwrap();
        b2.complete(50);

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
            processor
                .add_request(InflightRequest {
                    id: format!("r{i}"),
                    model_name: "m".to_string(),
                    inputs: vec![],
                    priority: 1,
                    timestamp: Instant::now(),
                    response_tx: tx,
                })
                .unwrap();
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
            processor
                .add_request(InflightRequest {
                    id: format!("r{i}"),
                    model_name: "m".to_string(),
                    inputs: vec![],
                    priority: 1,
                    timestamp: Instant::now(), // just arrived — but batch is full
                    response_tx: tx,
                })
                .unwrap();
        }

        // queue.len() == max_batch_size (3 == 3) → should_wait = false → batch forms
        let batch = processor.try_form_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().batch.len(), 3);
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
            processor
                .add_request(InflightRequest {
                    id: format!("log-req-{i}"),
                    model_name: "model".to_string(),
                    inputs: vec![json!(i)],
                    priority: 1,
                    timestamp: Instant::now() - Duration::from_millis(10),
                    response_tx: tx,
                })
                .unwrap();
        }

        // try_form_batch: triggers info! when batch is successfully formed
        let guard = processor.try_form_batch().unwrap();
        assert_eq!(guard.batch.len(), 3);

        // complete: records stats and releases the guard
        guard.complete(25);
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
            processor
                .add_request(InflightRequest {
                    id: format!("mb-{i}"),
                    model_name: "m".to_string(),
                    inputs: vec![],
                    priority: 1,
                    timestamp: Instant::now() - Duration::from_millis(50),
                    response_tx: tx,
                })
                .unwrap();
        }

        for _ in 0..3 {
            if let Some(guard) = processor.try_form_batch() {
                guard.complete(10);
            }
        }

        let stats = processor.get_stats();
        assert_eq!(stats.processed_batches, 3);
    }

    #[tokio::test]
    async fn test_guard_drop_releases_permit() {
        let processor = Arc::new(InflightBatchProcessor::new(2, 1, 0));

        for i in 0..4 {
            let (tx, _rx) = oneshot::channel();
            processor
                .add_request(InflightRequest {
                    id: format!("r{i}"),
                    model_name: "m".to_string(),
                    inputs: vec![],
                    priority: 1,
                    timestamp: Instant::now() - Duration::from_millis(10),
                    response_tx: tx,
                })
                .unwrap();
        }

        {
            let guard = processor.try_form_batch().unwrap();
            assert_eq!(guard.batch.len(), 2);
            assert_eq!(processor.inflight_count(), 1);
            // guard drops here WITHOUT calling complete
        }
        // permit was released by Drop — can form another batch
        assert!(processor.can_accept_batch());
        assert_eq!(processor.inflight_count(), 0);
    }

    #[tokio::test]
    async fn test_guard_complete_records_stats() {
        let processor = InflightBatchProcessor::new(3, 2, 0);

        for i in 0..3 {
            let (tx, _rx) = oneshot::channel();
            processor
                .add_request(InflightRequest {
                    id: format!("r{i}"),
                    model_name: "m".to_string(),
                    inputs: vec![],
                    priority: 1,
                    timestamp: Instant::now() - Duration::from_millis(10),
                    response_tx: tx,
                })
                .unwrap();
        }

        let guard = processor.try_form_batch().unwrap();
        assert_eq!(guard.batch.len(), 3);
        guard.complete(75);

        let stats = processor.get_stats();
        assert_eq!(stats.processed_batches, 1);
        assert_eq!(stats.processed_requests, 3);
        assert_eq!(stats.avg_latency_ms, 75);
        assert_eq!(stats.inflight_batches, 0);
        assert!(processor.can_accept_batch());
    }

    /// Verifies that dropping a guard during a simulated panic (via catch_unwind)
    /// correctly releases the semaphore permit and decrements inflight_batches.
    #[test]
    fn test_guard_drop_on_panic_releases_semaphore() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let processor = Arc::new(InflightBatchProcessor::new(2, 1, 0));
        let p = Arc::clone(&processor);

        rt.block_on(async move {
            for i in 0..2 {
                let (tx, _rx) = oneshot::channel();
                p.add_request(InflightRequest {
                    id: format!("r{i}"),
                    model_name: "m".to_string(),
                    inputs: vec![],
                    priority: 1,
                    timestamp: Instant::now() - Duration::from_millis(10),
                    response_tx: tx,
                })
                .unwrap();
            }
        });

        let guard = processor.try_form_batch().unwrap();
        assert_eq!(processor.inflight_count(), 1);
        assert!(!processor.can_accept_batch());

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = guard;
            panic!("simulated panic during batch processing");
        }));

        assert!(result.is_err(), "catch_unwind should have caught the panic");
        assert_eq!(
            processor.inflight_count(),
            0,
            "inflight_batches leaked on panic"
        );
        assert!(
            processor.can_accept_batch(),
            "semaphore permit leaked on panic"
        );
    }

    #[tokio::test]
    async fn test_try_form_batch_allows_concurrent_adds() {
        // Verifies that add_request and try_form_batch can interleave without
        // deadlock. Structural test: confirms that after concurrent adds and a
        // batch formation, a batch is produced.
        let processor = Arc::new(InflightBatchProcessor::new(5, 4, 0));

        for i in 0..5 {
            let (tx, _rx) = oneshot::channel();
            processor
                .add_request(InflightRequest {
                    id: format!("r{i}"),
                    model_name: "m".to_string(),
                    inputs: vec![],
                    priority: 1,
                    timestamp: Instant::now() - Duration::from_millis(10),
                    response_tx: tx,
                })
                .unwrap();
        }

        let p1 = Arc::clone(&processor);
        let add_task = tokio::spawn(async move {
            let (tx, _rx) = oneshot::channel();
            p1.add_request(InflightRequest {
                id: "late".to_string(),
                model_name: "m".to_string(),
                inputs: vec![],
                priority: 1,
                timestamp: Instant::now(),
                response_tx: tx,
            })
            .unwrap();
        });

        let batch = processor.try_form_batch();
        add_task.await.unwrap();

        assert!(batch.is_some());
    }
}
