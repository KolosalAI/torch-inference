# Throughput & Latency Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four root-cause bottlenecks in the inference hot path to reduce first-response latency and increase sustained throughput across CPU, CUDA, and MPS platforms.

**Architecture:** Two files are modified: `src/torch_optimization.rs` receives a real tensor-batching fix (stacking inputs into a single forward pass); `src/inflight_batch.rs` receives a bucketed O(1) priority queue, two-phase read/write locking on the batch-formation path, and a RAII semaphore guard replacing the leaky `permit.forget()` pattern.

**Tech Stack:** Rust, tch (libtorch bindings), tokio async runtime, parking_lot, dashmap.

---

## File Map

| File | Change |
|------|--------|
| `src/torch_optimization.rs` | Rewrite `infer_batch` to stack homogeneous tensors and run one forward pass; add shape-validation helper |
| `src/inflight_batch.rs` | Replace `VecDeque` with `PriorityBuckets`; two-phase locking in `try_form_batch`; add `InflightBatchGuard`; remove `complete_batch` method; update all tests |

---

## Task 1: Real tensor batching in `infer_batch`

**Files:**
- Modify: `src/torch_optimization.rs:201-212`

### Background

`infer_batch` currently calls `self.infer(input)?` in a loop — N separate forward passes. Real GPU/CPU batching requires stacking all inputs into a single tensor and doing **one** forward pass, then splitting the output back.

The fix only applies when all inputs share the same shape. If shapes differ, fall back to the serial path to preserve correctness.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` block at the bottom of `src/torch_optimization.rs`:

```rust
#[test]
fn test_same_shape_check_passes() {
    // shapes_are_homogeneous returns true when all tensors share the same size
    let t1 = tch::Tensor::zeros(&[3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    let t2 = tch::Tensor::zeros(&[3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    assert!(shapes_are_homogeneous(&[&t1, &t2]));
}

#[test]
fn test_same_shape_check_fails_on_mismatch() {
    let t1 = tch::Tensor::zeros(&[3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    let t2 = tch::Tensor::zeros(&[3, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    assert!(!shapes_are_homogeneous(&[&t1, &t2]));
}

#[test]
fn test_same_shape_check_single_input() {
    let t1 = tch::Tensor::zeros(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
    assert!(shapes_are_homogeneous(&[&t1]));
}

#[test]
fn test_same_shape_check_empty() {
    assert!(shapes_are_homogeneous(&[]));
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test test_same_shape_check 2>&1 | grep -E "FAILED|error"
```

Expected: compile error — `shapes_are_homogeneous` does not exist yet.

- [ ] **Step 3: Add the shape-validation helper and rewrite `infer_batch`**

Add this free function just above `impl OptimizedTorchModel` in `src/torch_optimization.rs`:

```rust
/// Returns true when all tensors share the same shape (required for stacking).
pub fn shapes_are_homogeneous(inputs: &[&Tensor]) -> bool {
    match inputs.first() {
        None => true,
        Some(first) => {
            let target = first.size();
            inputs.iter().all(|t| t.size() == target)
        }
    }
}
```

Replace the `infer_batch` method (lines 201-212):

```rust
/// Batch inference.
///
/// When all inputs share the same shape, they are stacked into a single
/// batched tensor and processed in **one** forward pass (GPU/CPU kernel
/// fusion, no per-sample overhead).  When shapes differ, falls back to N
/// serial forward passes to preserve correctness.
pub fn infer_batch(&self, inputs: Vec<&Tensor>) -> Result<Vec<Tensor>, String> {
    let _guard = tch::no_grad_guard();

    if inputs.is_empty() {
        return Ok(vec![]);
    }

    if shapes_are_homogeneous(&inputs) {
        // Stack → [N, original_dims...], one forward pass, then unbind back.
        let batched = Tensor::stack(&inputs, 0);
        let batched_output = self.infer(&batched)?;
        Ok(batched_output.unbind(0))
    } else {
        // Heterogeneous shapes: serial fallback.
        debug!("infer_batch: heterogeneous shapes, falling back to serial inference");
        inputs.iter().map(|input| self.infer(input)).collect()
    }
}
```

- [ ] **Step 4: Run the new tests**

```bash
cargo test test_same_shape_check 2>&1 | tail -5
```

Expected output: all four tests pass.

```
test test_same_shape_check_empty ... ok
test test_same_shape_check_fails_on_mismatch ... ok
test test_same_shape_check_passes ... ok
test test_same_shape_check_single_input ... ok
```

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
cargo test 2>&1 | tail -10
```

Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/torch_optimization.rs
git commit -m "perf: stack homogeneous tensors in infer_batch for one forward pass

Previously N separate forward passes were issued in a loop.
Stacking inputs along dim-0 lets the GPU/CPU process the entire
batch in a single kernel call. Heterogeneous shapes fall back to
the serial path to preserve correctness.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: O(1) priority queue via `PriorityBuckets`

**Files:**
- Modify: `src/inflight_batch.rs`

### Background

`add_request` calls `VecDeque::insert(pos, request)` — O(n) because it scans for the insertion point and then shifts all elements after it. Under load this dominates lock-hold time.

Replace the inner `VecDeque<InflightRequest>` with a `PriorityBuckets` struct: five `VecDeque`s indexed by priority 0–4. Enqueue is O(1) append to the matching bucket. Drain iterates buckets highest-first.

- [ ] **Step 1: Write failing tests for `PriorityBuckets`**

Add to the `#[cfg(test)]` block in `src/inflight_batch.rs` (before the existing tests, inside the `mod tests` block):

```rust
// ── PriorityBuckets unit tests ───────────────────────────────────────────

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
fn test_priority_buckets_push_and_drain_order() {
    let mut buckets = PriorityBuckets::new();
    buckets.push(make_request("low", 0));
    buckets.push(make_request("high", 4));
    buckets.push(make_request("mid", 2));

    let drained = buckets.drain_up_to(10);
    // highest priority first
    assert_eq!(drained[0].priority, 4);
    assert_eq!(drained[1].priority, 2);
    assert_eq!(drained[2].priority, 0);
}

#[test]
fn test_priority_buckets_drain_up_to_limit() {
    let mut buckets = PriorityBuckets::new();
    for i in 0..5 {
        buckets.push(make_request(&format!("r{i}"), 2));
    }
    let drained = buckets.drain_up_to(3);
    assert_eq!(drained.len(), 3);
    assert_eq!(buckets.len(), 2);
}

#[test]
fn test_priority_buckets_clamps_priority() {
    let mut buckets = PriorityBuckets::new();
    buckets.push(make_request("neg", -5));   // goes to bucket 0
    buckets.push(make_request("big", 100));  // goes to bucket 4
    assert_eq!(buckets.len(), 2);
    let drained = buckets.drain_up_to(10);
    assert_eq!(drained[0].priority, 100); // highest first
    assert_eq!(drained[1].priority, -5);
}

#[test]
fn test_priority_buckets_is_empty_and_len() {
    let mut buckets = PriorityBuckets::new();
    assert!(buckets.is_empty());
    assert_eq!(buckets.len(), 0);
    buckets.push(make_request("x", 1));
    assert!(!buckets.is_empty());
    assert_eq!(buckets.len(), 1);
}

#[test]
fn test_priority_buckets_oldest_age() {
    let mut buckets = PriorityBuckets::new();
    let (tx, _rx) = oneshot::channel();
    buckets.push(InflightRequest {
        id: "old".to_string(),
        model_name: "m".to_string(),
        inputs: vec![],
        priority: 1,
        timestamp: Instant::now() - Duration::from_millis(200),
        response_tx: tx,
    });
    let (tx2, _rx2) = oneshot::channel();
    buckets.push(InflightRequest {
        id: "new".to_string(),
        model_name: "m".to_string(),
        inputs: vec![],
        priority: 3,
        timestamp: Instant::now(),
        response_tx: tx2,
    });
    let age = buckets.oldest_age().unwrap();
    assert!(age >= Duration::from_millis(190));
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test test_priority_buckets 2>&1 | grep -E "error|FAILED"
```

Expected: compile errors — `PriorityBuckets` does not exist.

- [ ] **Step 3: Implement `PriorityBuckets`**

Add this struct **before** the `InflightRequest` struct at the top of `src/inflight_batch.rs`:

```rust
/// Five-bucket priority queue.  Each bucket is a FIFO for one priority level.
/// Buckets are indexed 0 (lowest) through 4 (highest).
/// Enqueue: O(1).  Drain: O(batch_size).
struct PriorityBuckets {
    buckets: [VecDeque<InflightRequest>; 5],
}

impl PriorityBuckets {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| VecDeque::new()),
        }
    }

    /// Clamp `priority` to 0–4 and append to the matching bucket.
    fn push(&mut self, request: InflightRequest) {
        let idx = request.priority.clamp(0, 4) as usize;
        self.buckets[idx].push_back(request);
    }

    /// Drain up to `max` requests, highest-priority bucket first.
    fn drain_up_to(&mut self, max: usize) -> Vec<InflightRequest> {
        let mut result = Vec::with_capacity(max);
        for bucket in self.buckets.iter_mut().rev() {
            let take = (max - result.len()).min(bucket.len());
            result.extend(bucket.drain(..take));
            if result.len() >= max {
                break;
            }
        }
        result
    }

    fn len(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    fn is_empty(&self) -> bool {
        self.buckets.iter().all(|b| b.is_empty())
    }

    /// Elapsed time of the request that has been waiting the longest.
    fn oldest_age(&self) -> Option<Duration> {
        self.buckets
            .iter()
            .filter_map(|b| b.front())
            .map(|r| r.timestamp.elapsed())
            .max()
    }
}
```

- [ ] **Step 4: Swap `VecDeque<InflightRequest>` for `PriorityBuckets` in `InflightBatchProcessor`**

Change the struct field:

```rust
// Before:
pending_queue: Arc<RwLock<VecDeque<InflightRequest>>>,

// After:
pending_queue: Arc<RwLock<PriorityBuckets>>,
```

Change the constructor:

```rust
// Before:
pending_queue: Arc::new(RwLock::new(VecDeque::new())),

// After:
pending_queue: Arc::new(RwLock::new(PriorityBuckets::new())),
```

- [ ] **Step 5: Update `add_request` to use `PriorityBuckets::push`**

Replace the body of `add_request`:

```rust
pub async fn add_request(&self, request: InflightRequest) -> Result<(), String> {
    let mut queue = self.pending_queue.write().await;
    queue.push(request);
    let queue_size = queue.len();
    drop(queue);

    self.current_queue_depth.store(queue_size, Ordering::Relaxed);

    let current_peak = self.peak_queue_depth.load(Ordering::Relaxed);
    if queue_size > current_peak {
        self.peak_queue_depth.store(queue_size, Ordering::Relaxed);
    }

    debug!("Request added to inflight queue. Queue size: {}", queue_size);
    Ok(())
}
```

- [ ] **Step 6: Update `try_form_batch` to use `PriorityBuckets`**

Replace the section that accesses the queue inside `try_form_batch` (the part after permit acquisition):

```rust
let mut queue = self.pending_queue.write().await;

if queue.is_empty() {
    return None;
}

let oldest_age = queue.oldest_age().unwrap_or(Duration::ZERO);

let should_wait = queue.len() < self.max_batch_size
    && oldest_age < Duration::from_millis(self.batch_timeout_ms);

if should_wait {
    return None;
}

let batch_size = queue.len().min(self.max_batch_size);
let batch = queue.drain_up_to(batch_size);

self.current_queue_depth.store(queue.len(), Ordering::Relaxed);
self.inflight_batches.fetch_add(1, Ordering::Relaxed);
```

Also update `clear_queue` to iterate the buckets:

```rust
pub async fn clear_queue(&self) -> usize {
    let mut queue = self.pending_queue.write().await;
    let mut count = 0;
    for bucket in queue.buckets.iter_mut() {
        for request in bucket.drain(..) {
            let _ = request.response_tx.send(Err("Queue cleared".to_string()));
            count += 1;
        }
    }
    self.current_queue_depth.store(0, Ordering::Relaxed);
    warn!("Cleared {} pending requests from queue", count);
    count
}
```

- [ ] **Step 7: Remove the `use std::collections::VecDeque;` import if no longer needed elsewhere**

Check: `VecDeque` is still used inside `PriorityBuckets`, so the import stays.

- [ ] **Step 8: Run the new and existing tests**

```bash
cargo test test_priority_buckets 2>&1 | tail -8
cargo test --lib 2>&1 | tail -10
```

Expected: all `test_priority_buckets_*` tests pass; all existing tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/inflight_batch.rs
git commit -m "perf: replace O(n) VecDeque::insert with O(1) PriorityBuckets

Five per-priority VecDeques replace a single sorted deque.
Enqueue is now O(1) append; drain iterates buckets highest-first
in O(batch_size). Eliminates element-shifting under the write lock
on every add_request call.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Two-phase locking in `try_form_batch`

**Files:**
- Modify: `src/inflight_batch.rs`

### Background

`try_form_batch` currently holds a write lock for the entire peek-and-drain sequence. Concurrent `add_request` calls are blocked the whole time. Fix: use a read lock to peek, release it, then acquire a write lock only to drain. Re-check emptiness after the write lock is acquired (TOCTOU guard).

- [ ] **Step 1: Write a failing test for concurrent add + form**

Add to the test block:

```rust
#[tokio::test]
async fn test_try_form_batch_allows_concurrent_adds() {
    // This test verifies that add_request can proceed while try_form_batch
    // is in its peek phase (read lock). It is a structural/logic test rather
    // than a timing test — it simply confirms that after concurrent adds and
    // a batch formation, all requests are accounted for.
    let processor = Arc::new(InflightBatchProcessor::new(5, 4, 0));

    for i in 0..5 {
        let (tx, _rx) = oneshot::channel();
        processor.add_request(InflightRequest {
            id: format!("r{i}"),
            model_name: "m".to_string(),
            inputs: vec![],
            priority: 1,
            timestamp: Instant::now() - Duration::from_millis(10),
            response_tx: tx,
        }).await.unwrap();
    }

    // Both add and try_form_batch should complete without deadlock
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
        }).await.unwrap();
    });

    let batch = processor.try_form_batch().await;
    add_task.await.unwrap();

    assert!(batch.is_some());
}
```

- [ ] **Step 2: Run the test to confirm it passes against current code (baseline)**

```bash
cargo test test_try_form_batch_allows_concurrent_adds 2>&1 | tail -5
```

This test passes with current code — it establishes a correctness baseline. The improvement is structural (reduced contention), not observable in a single-threaded test.

- [ ] **Step 3: Rewrite `try_form_batch` with two-phase locking**

Replace the entire `try_form_batch` method:

```rust
pub async fn try_form_batch(&self) -> Option<Vec<InflightRequest>> {
    // ── Phase 1: read-lock peek ────────────────────────────────────────
    // Concurrent add_request calls proceed during this phase.
    let should_form = {
        let queue = self.pending_queue.read().await;
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

    // Try to acquire a semaphore permit (non-blocking).
    let Ok(permit) = self.inflight_semaphore.try_acquire() else {
        debug!("Max inflight batches reached, waiting...");
        return None;
    };

    // ── Phase 2: write-lock drain ──────────────────────────────────────
    let batch = {
        let mut queue = self.pending_queue.write().await;
        // TOCTOU guard: re-verify queue is still non-empty.
        if queue.is_empty() {
            return None;
        }
        let batch_size = queue.len().min(self.max_batch_size);
        let batch = queue.drain_up_to(batch_size);
        self.current_queue_depth.store(queue.len(), Ordering::Relaxed);
        batch
    }; // write lock released here

    self.inflight_batches.fetch_add(1, Ordering::Relaxed);

    // Permanently consume the permit until complete_batch is called.
    permit.forget();

    info!(
        "Formed batch with {} requests (queue remaining: {}, inflight: {})",
        batch.len(),
        self.current_queue_depth.load(Ordering::Relaxed),
        self.inflight_batches.load(Ordering::Relaxed)
    );

    Some(batch)
}
```

- [ ] **Step 4: Run the full test suite**

```bash
cargo test --lib 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/inflight_batch.rs
git commit -m "perf: two-phase locking in try_form_batch reduces producer contention

Phase 1 (peek) uses a read lock — concurrent add_request calls
proceed unblocked. Phase 2 (drain) uses a write lock only when a
batch is confirmed ready. A TOCTOU re-check after the write lock
ensures correctness.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: RAII semaphore guard replaces `permit.forget()`

**Files:**
- Modify: `src/inflight_batch.rs`

### Background

`permit.forget()` permanently leaks a semaphore permit if the caller panics or forgets to call `complete_batch`. Over time `max_inflight_batches` silently shrinks. Fix: return an `InflightBatchGuard` from `try_form_batch` that holds the permit and decrements `inflight_batches` on `Drop`. `complete_batch` on `InflightBatchProcessor` is removed; stats are recorded via `guard.complete(latency_ms)`.

### New types

```
InflightBatchGuard<'a>
  .batch: Vec<InflightRequest>          (pub)
  .complete(self, processing_time_ms)   (consumes guard, records stats)
  Drop impl                             (releases permit; decrements inflight_batches if not completed)
```

- [ ] **Step 1: Write failing tests for `InflightBatchGuard`**

Add to the test block:

```rust
#[tokio::test]
async fn test_guard_drop_releases_permit() {
    let processor = Arc::new(InflightBatchProcessor::new(2, 1, 0));

    for i in 0..4 {
        let (tx, _rx) = oneshot::channel();
        processor.add_request(InflightRequest {
            id: format!("r{i}"),
            model_name: "m".to_string(),
            inputs: vec![],
            priority: 1,
            timestamp: Instant::now() - Duration::from_millis(10),
            response_tx: tx,
        }).await.unwrap();
    }

    assert!(!processor.can_accept_batch()); // all permits taken by first form
    {
        let guard = processor.try_form_batch().await.unwrap();
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
        processor.add_request(InflightRequest {
            id: format!("r{i}"),
            model_name: "m".to_string(),
            inputs: vec![],
            priority: 1,
            timestamp: Instant::now() - Duration::from_millis(10),
            response_tx: tx,
        }).await.unwrap();
    }

    let guard = processor.try_form_batch().await.unwrap();
    assert_eq!(guard.batch.len(), 3);
    guard.complete(75);

    let stats = processor.get_stats();
    assert_eq!(stats.processed_batches, 1);
    assert_eq!(stats.processed_requests, 3);
    assert_eq!(stats.avg_latency_ms, 75);
    assert_eq!(stats.inflight_batches, 0);
    assert!(processor.can_accept_batch());
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test test_guard_ 2>&1 | grep -E "error|FAILED"
```

Expected: compile error — `InflightBatchGuard` does not exist.

- [ ] **Step 3: Add `OwnedSemaphorePermit` to imports**

In the `use` block at the top of `src/inflight_batch.rs`, change:

```rust
// Before:
use tokio::sync::{RwLock, Semaphore, oneshot};

// After:
use tokio::sync::{RwLock, Semaphore, OwnedSemaphorePermit, oneshot};
```

- [ ] **Step 4: Add `InflightBatchGuard`**

Add this struct and its impls **after** the `InflightBatchStats` struct and **before** the `impl Default for InflightBatchProcessor` block:

```rust
/// RAII guard returned by `try_form_batch`.
///
/// Holds the inflight semaphore permit for the duration of batch processing.
/// The permit is automatically released on `Drop`, whether or not `complete`
/// is called — preventing semaphore starvation on panics.
pub struct InflightBatchGuard<'a> {
    processor: &'a InflightBatchProcessor,
    /// The requests that form this batch. Public for callers to process.
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
        Self { processor, batch, permit: permit, completed: false }
    }

    /// Record batch statistics and release the guard.
    ///
    /// `processing_time_ms` is the wall-clock time from batch formation
    /// to completion of inference.
    pub fn complete(mut self, processing_time_ms: u64) {
        let batch_size = self.batch.len();
        self.processor.processed_batches.fetch_add(1, Ordering::Relaxed);
        self.processor.processed_requests.fetch_add(batch_size as u64, Ordering::Relaxed);
        self.processor.total_latency_ms.fetch_add(processing_time_ms, Ordering::Relaxed);
        self.processor.inflight_batches.fetch_sub(1, Ordering::Relaxed);
        self.completed = true;
        // `self` drops here: permit releases automatically via OwnedSemaphorePermit::Drop.
    }
}

impl Drop for InflightBatchGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            // complete() was never called (e.g. caller panicked).
            // Decrement inflight_batches so the counter stays accurate.
            self.processor.inflight_batches.fetch_sub(1, Ordering::Relaxed);
        }
        // OwnedSemaphorePermit is dropped here regardless, releasing the semaphore.
    }
}
```

- [ ] **Step 5: Update `try_form_batch` to return `InflightBatchGuard`**

Change the signature and the permit handling at the end of `try_form_batch`:

```rust
pub async fn try_form_batch(&self) -> Option<InflightBatchGuard<'_>> {
    // ── Phase 1: read-lock peek ────────────────────────────────────────
    let should_form = {
        let queue = self.pending_queue.read().await;
        if queue.is_empty() {
            return None;
        }
        let oldest_age = queue.oldest_age().unwrap_or(Duration::ZERO);
        let full = queue.len() >= self.max_batch_size;
        full || oldest_age >= Duration::from_millis(self.batch_timeout_ms)
    };

    if !should_form {
        return None;
    }

    // Non-blocking permit acquisition.
    let Ok(permit) = self.inflight_semaphore.clone().try_acquire_owned() else {
        debug!("Max inflight batches reached, waiting...");
        return None;
    };

    // ── Phase 2: write-lock drain ──────────────────────────────────────
    let batch = {
        let mut queue = self.pending_queue.write().await;
        if queue.is_empty() {
            return None; // TOCTOU guard
        }
        let batch_size = queue.len().min(self.max_batch_size);
        let batch = queue.drain_up_to(batch_size);
        self.current_queue_depth.store(queue.len(), Ordering::Relaxed);
        batch
    };

    self.inflight_batches.fetch_add(1, Ordering::Relaxed);

    info!(
        "Formed batch with {} requests (queue remaining: {}, inflight: {})",
        batch.len(),
        self.current_queue_depth.load(Ordering::Relaxed),
        self.inflight_batches.load(Ordering::Relaxed)
    );

    Some(InflightBatchGuard::new(self, batch, permit))
}
```

Note: `try_acquire_owned()` requires `Arc<Semaphore>` (which `inflight_semaphore` already is). It returns `OwnedSemaphorePermit` which is `'static` and safe to embed in the guard.

- [ ] **Step 6: Remove `complete_batch` from `InflightBatchProcessor`**

Delete the entire `complete_batch` method from `impl InflightBatchProcessor`:

```rust
// DELETE this entire method:
pub fn complete_batch(&self, batch_size: usize, processing_time_ms: u64) {
    self.processed_batches.fetch_add(1, Ordering::Relaxed);
    self.processed_requests.fetch_add(batch_size as u64, Ordering::Relaxed);
    self.total_latency_ms.fetch_add(processing_time_ms, Ordering::Relaxed);
    self.inflight_batches.fetch_sub(1, Ordering::Relaxed);
    self.inflight_semaphore.add_permits(1);
}
```

- [ ] **Step 7: Update all tests that call `complete_batch` or access raw batch vec**

The pattern for every test that previously did:
```rust
let batch = processor.try_form_batch().await.unwrap();
assert_eq!(batch.len(), N);
processor.complete_batch(batch.len(), latency_ms);
```

Becomes:
```rust
let guard = processor.try_form_batch().await.unwrap();
assert_eq!(guard.batch.len(), N);
guard.complete(latency_ms);
```

And tests that drop the batch without completing (e.g. `test_inflight_peak_queue_depth`) just call `let _guard = processor.try_form_batch().await;` — the guard drops, permit releases, no other change needed.

Apply this to every affected test. The full list of tests to update:

**`test_inflight_batch_formation`** — change:
```rust
let batch = processor.try_form_batch().await.unwrap();
assert_eq!(batch.len(), 3);
assert_eq!(processor.queue_depth(), 2);
```
to:
```rust
let guard = processor.try_form_batch().await.unwrap();
assert_eq!(guard.batch.len(), 3);
assert_eq!(processor.queue_depth(), 2);
```

**`test_inflight_max_concurrent_batches`** — change all `batch1`/`batch2`/`batch3` vars to guards, and update `complete_batch` calls:
```rust
let guard1 = processor.try_form_batch().await.unwrap();
assert_eq!(guard1.batch.len(), 2);
assert_eq!(processor.inflight_count(), 1);

let guard2 = processor.try_form_batch().await.unwrap();
assert_eq!(guard2.batch.len(), 2);
assert_eq!(processor.inflight_count(), 2);

let batch3 = processor.try_form_batch().await;
assert!(batch3.is_none());

guard1.complete(50);  // was: processor.complete_batch(batch1.len(), 50)
assert_eq!(processor.inflight_count(), 1);

tokio::time::sleep(Duration::from_millis(100)).await;
let guard3 = processor.try_form_batch().await.unwrap();
assert_eq!(guard3.batch.len(), 2);
```

**`test_inflight_stats`**:
```rust
let guard1 = processor.try_form_batch().await.unwrap();
guard1.complete(50);

let stats = processor.get_stats();
assert_eq!(stats.processed_batches, 1);
assert_eq!(stats.processed_requests, 3);
assert_eq!(stats.avg_latency_ms, 50);
assert_eq!(stats.avg_batch_size, 3.0);
assert_eq!(stats.current_queue_depth, 3);
```

**`test_inflight_peak_queue_depth`** — the existing `processor.try_form_batch().await;` call (result discarded) is fine; the guard will drop and release the permit. No change needed for the assertion pattern but make the drop explicit for clarity:
```rust
let _guard = processor.try_form_batch().await;
```

**`test_complete_batch_decrements_inflight`**:
```rust
let guard = processor.try_form_batch().await.unwrap();
assert_eq!(processor.inflight_count(), 1);
guard.complete(10);
assert_eq!(processor.inflight_count(), 0);
```

**`test_complete_batch_stats_accumulate`**:
```rust
let b1 = processor.try_form_batch().await.unwrap();
b1.complete(100);

let b2 = processor.try_form_batch().await.unwrap();
b2.complete(50);

let stats = processor.get_stats();
assert_eq!(stats.processed_batches, 2);
assert_eq!(stats.processed_requests, 10);
assert_eq!(stats.avg_latency_ms, 75);
assert_eq!(stats.inflight_batches, 0);
```

**`test_try_form_batch_and_complete_with_logger_covers_log_lines`**:
```rust
let guard = processor.try_form_batch().await.unwrap();
assert_eq!(guard.batch.len(), 3);
guard.complete(25);
assert_eq!(processor.inflight_count(), 0);
```

**`test_multiple_batches_with_logger`**:
```rust
for _ in 0..3 {
    if let Some(guard) = processor.try_form_batch().await {
        guard.complete(10);
    }
}
```

- [ ] **Step 8: Run the new guard tests**

```bash
cargo test test_guard_ 2>&1 | tail -5
```

Expected:
```
test test_guard_complete_records_stats ... ok
test test_guard_drop_releases_permit ... ok
```

- [ ] **Step 9: Run the full test suite**

```bash
cargo test --lib 2>&1 | tail -10
```

Expected: all tests pass, zero failures.

- [ ] **Step 10: Commit**

```bash
git add src/inflight_batch.rs
git commit -m "fix: RAII InflightBatchGuard prevents semaphore permit leaks

permit.forget() permanently leaked a semaphore slot on panic or
missed complete_batch calls. InflightBatchGuard holds an
OwnedSemaphorePermit that releases automatically on Drop, keeping
inflight_batches accurate in all code paths. complete_batch removed.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Fix `infer_batch` serial loop | Task 1 |
| Shape-mismatch fallback path | Task 1, Step 3 |
| Replace VecDeque with bucketed O(1) queue | Task 2 |
| Priority clamping (i32 → 0..4) | Task 2, Step 3 |
| Two-phase locking in `try_form_batch` | Task 3 |
| TOCTOU re-check after write lock | Task 3, Step 3 |
| RAII guard replaces `permit.forget()` | Task 4 |
| Guard `Drop` handles panic case | Task 4, Step 4 |
| `complete_batch` removed | Task 4, Step 6 |
| All test callsites updated | Task 4, Step 7 |

**Placeholder scan:** None found.

**Type consistency:**
- `shapes_are_homogeneous` defined in Task 1 Step 3, used in same step — consistent.
- `PriorityBuckets` defined in Task 2 Step 3, used in Steps 4–6 — consistent.
- `InflightBatchGuard<'a>` defined in Task 4 Step 4, returned in Step 5, tested in Step 8 — consistent.
- `try_form_batch` returns `Option<Vec<...>>` in Tasks 2–3, then changes to `Option<InflightBatchGuard<'_>>` in Task 4 Step 5 — sequential, no conflict.
- `OwnedSemaphorePermit` added to imports in Task 4 Step 3 before it's used in Step 4 — correct order.
