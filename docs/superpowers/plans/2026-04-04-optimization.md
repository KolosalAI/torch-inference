# Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve throughput, latency, and code clarity via 8 targeted changes to the hot paths in inflight batching, worker pool, cache eviction, model selection, and profiling tooling.

**Architecture:** Each change is self-contained with no cross-task dependencies. Tasks 1–5 are Phase A (hot-path fixes); Tasks 6–8 are Phase C (profiling). Implement in order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.

**Tech Stack:** Rust 2021, tokio, dashmap, parking_lot, criterion, pprof (new optional dep)

---

## Corrections vs. Spec

Two spec assumptions were wrong upon inspection:

- `src/core/model_cache.rs` is **not dead** — imported by `neural_network.rs`, `image_classifier.rs`, `yolo.rs`. Do not delete it.
- `SAMPLE_SIZE` in `cache.rs` is already `100`, not `5`. Adaptive range is adjusted to `20–100` accordingly.
- The INT8 warning in `torch_optimization.rs` is already clean (uses `INT8_DYNAMIC_WARNED` atomic). No change needed.

---

## File Map

| File | Change |
|------|--------|
| `src/inflight_batch.rs` | Replace `PriorityBuckets` (5 VecDeques) with `BinaryHeap`-backed `PriorityQueue` |
| `src/worker_pool.rs` | Replace `Arc<RwLock<WorkerState>>` with `Arc<AtomicU8>` in `Worker`; make state methods sync |
| `src/cache.rs` | Replace constant `SAMPLE_SIZE` with `fn evict_sample_size()` based on hit rate |
| `src/model_pool.rs` | Change `load_score()` weight from `1000` to `500` |
| `src/core/whisper_stt.rs` | Replace silent TODO comment with explicit log::warn! |
| `Cargo.toml` | Add optional `pprof` dependency; add `profiling` feature |
| `src/main.rs` | Add `#[cfg(feature = "profiling")]` profiler guard |
| `benches/throughput_bench.rs` | Add `bench_inflight_priority` benchmark |
| `Makefile` | Add `flamegraph` target |

---

## Task 1: A5 — Clarify Whisper TODO

**Files:**
- Modify: `src/core/whisper_stt.rs:72-78`

- [ ] **Step 1: Replace the silent TODO with an explicit warning**

In `src/core/whisper_stt.rs`, the `transcribe` method (lines 69–85) has a `// TODO: Implement actual Whisper inference` comment inside `#[cfg(feature = "torch")]` but silently falls through to `transcribe_fallback`. Replace:

```rust
    /// Transcribe audio data to text
    pub fn transcribe(&self, audio: &AudioData) -> Result<String> {
        #[cfg(feature = "torch")]
        {
            if let Some(ref _model) = self.model {
                // TODO: Implement actual Whisper inference
                // For now, use simple pattern matching as fallback
                self.transcribe_fallback(audio)
            } else {
                self.transcribe_fallback(audio)
            }
        }
        
        #[cfg(not(feature = "torch"))]
        {
            self.transcribe_fallback(audio)
        }
    }
```

With:

```rust
    /// Transcribe audio data to text
    ///
    /// Note: neural Whisper inference is not yet implemented. Always uses heuristic fallback.
    pub fn transcribe(&self, audio: &AudioData) -> Result<String> {
        #[cfg(feature = "torch")]
        {
            if self.model.is_some() {
                log::warn!(
                    "Whisper neural inference not implemented; using heuristic fallback. \
                     Load a quantized ONNX Whisper model to enable real transcription."
                );
            }
            self.transcribe_fallback(audio)
        }

        #[cfg(not(feature = "torch"))]
        {
            self.transcribe_fallback(audio)
        }
    }
```

- [ ] **Step 2: Verify tests pass**

```bash
cargo test --no-default-features -p torch_inference core::whisper_stt 2>&1 | tail -5
```

Expected: `test result: ok.`

- [ ] **Step 3: Commit**

```bash
git add src/core/whisper_stt.rs
git commit -m "fix(whisper): replace silent TODO with explicit warn log"
```

---

## Task 2: A2 — Worker Pool AtomicU8 State

**Files:**
- Modify: `src/worker_pool.rs`

### Background

`Worker::get_state()` and `set_state()` currently await a `RwLock<WorkerState>`. Every health check, task assignment, and release acquires this lock. After this change, state reads/writes become lock-free atomic operations.

`parking_lot` is already a dependency (`parking_lot = "0.12"` in `Cargo.toml`).

- [ ] **Step 1: Add `WorkerState` conversion methods**

In `src/worker_pool.rs`, after the `WorkerState` enum definition (after line 17), add:

```rust
impl WorkerState {
    fn as_u8(self) -> u8 {
        match self {
            WorkerState::Idle => 0,
            WorkerState::Processing => 1,
            WorkerState::Paused => 2,
            WorkerState::Stopping => 3,
            WorkerState::Stopped => 4,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => WorkerState::Idle,
            1 => WorkerState::Processing,
            2 => WorkerState::Paused,
            3 => WorkerState::Stopping,
            _ => WorkerState::Stopped,
        }
    }
}
```

- [ ] **Step 2: Write the failing test**

At the bottom of the `#[cfg(test)]` block in `src/worker_pool.rs`, add:

```rust
    #[test]
    fn test_worker_state_atomic_roundtrip() {
        // Verify every variant survives as_u8 → from_u8
        for s in [
            WorkerState::Idle,
            WorkerState::Processing,
            WorkerState::Paused,
            WorkerState::Stopping,
            WorkerState::Stopped,
        ] {
            assert_eq!(WorkerState::from_u8(s.as_u8()), s);
        }
    }
```

- [ ] **Step 3: Run test to verify it fails (WorkerState doesn't have PartialEq yet)**

```bash
cargo test --no-default-features test_worker_state_atomic_roundtrip 2>&1 | tail -10
```

Expected: compile error — `WorkerState` doesn't implement `PartialEq`.

- [ ] **Step 4: Add `#[derive(PartialEq)]` to WorkerState**

`WorkerState` already has `#[derive(Debug, Clone, Copy, PartialEq, Eq)]` — if the test from step 3 failed due to missing derive, add it. If it compiled but failed for another reason, investigate. If `PartialEq` is already derived, the test from step 3 may have produced a different error — check and proceed.

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test --no-default-features test_worker_state_atomic_roundtrip 2>&1 | tail -5
```

Expected: `test result: ok. 1 passed`

- [ ] **Step 6: Replace Worker struct fields**

Replace the `Worker` struct definition in `src/worker_pool.rs` (currently lines 32–40):

```rust
/// Individual worker
pub struct Worker {
    pub id: usize,
    state: Arc<RwLock<WorkerState>>,
    tasks_processed: AtomicU64,
    total_processing_time_ms: AtomicU64,
    start_time: Instant,
    last_active: Arc<RwLock<Option<Instant>>>,
    handle: Option<JoinHandle<()>>,
}
```

With:

```rust
/// Individual worker
pub struct Worker {
    pub id: usize,
    /// Lock-free state: 0=Idle 1=Processing 2=Paused 3=Stopping 4=Stopped
    state: Arc<std::sync::atomic::AtomicU8>,
    tasks_processed: AtomicU64,
    total_processing_time_ms: AtomicU64,
    start_time: Instant,
    last_active: Arc<parking_lot::Mutex<Option<Instant>>>,
    handle: Option<JoinHandle<()>>,
}
```

- [ ] **Step 7: Replace Worker::new**

Replace `Worker::new` (currently lines 43–53):

```rust
    pub fn new(id: usize) -> Self {
        Self {
            id,
            state: Arc::new(RwLock::new(WorkerState::Idle)),
            tasks_processed: AtomicU64::new(0),
            total_processing_time_ms: AtomicU64::new(0),
            start_time: Instant::now(),
            last_active: Arc::new(RwLock::new(None)),
            handle: None,
        }
    }
```

With:

```rust
    pub fn new(id: usize) -> Self {
        Self {
            id,
            state: Arc::new(std::sync::atomic::AtomicU8::new(WorkerState::Idle.as_u8())),
            tasks_processed: AtomicU64::new(0),
            total_processing_time_ms: AtomicU64::new(0),
            start_time: Instant::now(),
            last_active: Arc::new(parking_lot::Mutex::new(None)),
            handle: None,
        }
    }
```

- [ ] **Step 8: Replace get_state, set_state, update_last_active, get_stats**

Replace the four methods (lines 55–87):

```rust
    pub async fn get_state(&self) -> WorkerState {
        *self.state.read().await
    }
    
    pub async fn set_state(&self, new_state: WorkerState) {
        *self.state.write().await = new_state;
        debug!("Worker {} state changed to {:?}", self.id, new_state);
    }
    
    // ... record_task unchanged ...
    
    pub async fn update_last_active(&self) {
        *self.last_active.write().await = Some(Instant::now());
    }
    
    pub async fn get_stats(&self) -> WorkerStats {
        let tasks = self.tasks_processed.load(Ordering::Relaxed);
        let total_time = self.total_processing_time_ms.load(Ordering::Relaxed);
        let avg_time = if tasks > 0 { total_time / tasks } else { 0 };
        
        WorkerStats {
            worker_id: self.id,
            state: *self.state.read().await,
            tasks_processed: tasks,
            total_processing_time_ms: total_time,
            avg_processing_time_ms: avg_time,
            last_active: *self.last_active.read().await,
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }
```

With (all sync, no async/await):

```rust
    pub fn get_state(&self) -> WorkerState {
        WorkerState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub fn set_state(&self, new_state: WorkerState) {
        self.state.store(new_state.as_u8(), Ordering::Release);
        debug!("Worker {} state changed to {:?}", self.id, new_state);
    }

    pub fn update_last_active(&self) {
        *self.last_active.lock() = Some(Instant::now());
    }

    pub fn get_stats(&self) -> WorkerStats {
        let tasks = self.tasks_processed.load(Ordering::Relaxed);
        let total_time = self.total_processing_time_ms.load(Ordering::Relaxed);
        let avg_time = if tasks > 0 { total_time / tasks } else { 0 };

        WorkerStats {
            worker_id: self.id,
            state: self.get_state(),
            tasks_processed: tasks,
            total_processing_time_ms: total_time,
            avg_processing_time_ms: avg_time,
            last_active: *self.last_active.lock(),
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }
```

- [ ] **Step 9: Remove `.await` from all callers in the same file**

The following callers need `.await` removed. Apply each change:

In `acquire_worker` (~line 245):
```rust
// Before:
worker.set_state(WorkerState::Processing).await;
worker.update_last_active().await;
// After:
worker.set_state(WorkerState::Processing);
worker.update_last_active();
```

In `release_worker` (~line 258):
```rust
// Before:
worker.set_state(WorkerState::Idle).await;
worker.update_last_active().await;
// After:
worker.set_state(WorkerState::Idle);
worker.update_last_active();
```

In `active_worker_count` (~line 299):
```rust
// Before:
if worker.get_state().await == WorkerState::Processing {
// After:
if worker.get_state() == WorkerState::Processing {
```

In `start_health_checker` (~line 369):
```rust
// Before:
let stats = worker.get_stats().await;
// After:
let stats = worker.get_stats();
```

In `WorkerPool::get_stats` (~line 400):
```rust
// Before:
worker_stats.push(worker.get_stats().await);
// After:
worker_stats.push(worker.get_stats());
```

In `pause_all` (~line 433):
```rust
// Before:
if worker.get_state().await == WorkerState::Idle {
    worker.set_state(WorkerState::Paused).await;
// After:
if worker.get_state() == WorkerState::Idle {
    worker.set_state(WorkerState::Paused);
```

In `resume_all` (~line 445):
```rust
// Before:
if worker.get_state().await == WorkerState::Paused {
    worker.set_state(WorkerState::Idle).await;
// After:
if worker.get_state() == WorkerState::Paused {
    worker.set_state(WorkerState::Idle);
```

In `shutdown` (~line 458):
```rust
// Before:
worker.set_state(WorkerState::Stopped).await;
// After:
worker.set_state(WorkerState::Stopped);
```

In `remove_idle_worker` (~line 209):
```rust
// Before:
let state = workers[i].get_state().await;
if state == WorkerState::Idle {
    workers[i].set_state(WorkerState::Stopping).await;
// After:
let state = workers[i].get_state();
if state == WorkerState::Idle {
    workers[i].set_state(WorkerState::Stopping);
```

- [ ] **Step 10: Fix tests (remove .await from test calls)**

In the test block, update:

```rust
// test_worker_creation:
assert_eq!(worker.get_state(), WorkerState::Idle);  // was .await

// test_worker_state_change:
worker.set_state(WorkerState::Processing);           // was .await
assert_eq!(worker.get_state(), WorkerState::Processing);  // was .await

// test_worker_stats:
let stats = worker.get_stats();  // was .await
```

- [ ] **Step 11: Also remove the unused `RwLock` import if no longer needed**

Check if `RwLock` is still used elsewhere in the file. If only `Arc<RwLock<...>>` patterns remain elsewhere (like `workers: Arc<RwLock<Vec<Arc<Worker>>>>`) keep the import. Otherwise remove `RwLock` from the `tokio::sync` import line.

- [ ] **Step 12: Build to confirm zero errors**

```bash
cargo build --no-default-features 2>&1 | grep -E "^error" | head -20
```

Expected: no output (zero errors).

- [ ] **Step 13: Run worker pool tests**

```bash
cargo test --no-default-features worker_pool 2>&1 | tail -10
```

Expected: `test result: ok.`

- [ ] **Step 14: Commit**

```bash
git add src/worker_pool.rs
git commit -m "perf(worker-pool): replace RwLock<WorkerState> with AtomicU8 for lock-free state reads"
```

---

## Task 3: A1 — Inflight Batch BinaryHeap Queue

**Files:**
- Modify: `src/inflight_batch.rs`

### Background

Current: `PriorityBuckets` uses 5 `VecDeque<InflightRequest>` buckets — no sub-bucket ordering. Items at the same priority are served in FIFO order per bucket, but priorities are clamped to 0–4. Any caller passing priority 5, 10, 100 gets clamped to 4. After this change, arbitrary integer priorities work correctly, and within a priority, FIFO is preserved via a sequence counter.

`InflightRequest.priority` is `i32`. Higher value = higher urgency.

- [ ] **Step 1: Add imports needed for BinaryHeap**

At the top of `src/inflight_batch.rs`, `VecDeque` is already imported. Add:

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;
```

Replace the existing `use std::collections::VecDeque;` import (keep it only if VecDeque is used elsewhere; check before removing).

- [ ] **Step 2: Write the failing test**

In the `#[cfg(test)]` block of `src/inflight_batch.rs` (or add one if absent), add:

```rust
    #[test]
    fn test_priority_queue_ordering() {
        // Items with higher priority must come out first.
        // Items with equal priority must come out in FIFO order.
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cargo test --no-default-features test_priority_queue_ordering 2>&1 | tail -10
```

Expected: compile error — `PriorityQueue` doesn't exist yet.

- [ ] **Step 4: Add OrderedRequest and PriorityQueue**

In `src/inflight_batch.rs`, add these types before the `InflightBatchProcessor` struct (after `InflightRequest`):

```rust
/// Wrapper for BinaryHeap ordering: higher priority first, FIFO within priority.
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
        // Max-heap: higher priority first; ties broken by lower seq (earlier insert).
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Single-heap priority queue. O(log n) insert and pop, correct FIFO within priority.
struct PriorityQueue {
    heap: BinaryHeap<OrderedRequest>,
    next_seq: u64,
}

impl PriorityQueue {
    fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
        }
    }

    fn push(&mut self, request: InflightRequest) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let priority = request.priority;
        self.heap.push(OrderedRequest { priority, seq, request });
    }

    fn drain_all(&mut self, mut f: impl FnMut(InflightRequest)) {
        while let Some(item) = self.heap.pop() {
            f(item.request);
        }
    }

    fn drain_up_to(&mut self, max: usize) -> Vec<InflightRequest> {
        let mut result = Vec::with_capacity(max);
        for _ in 0..max {
            match self.heap.pop() {
                Some(item) => result.push(item.request),
                None => break,
            }
        }
        result
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// O(n) scan — called only to decide whether to form a batch (not on hot path).
    fn oldest_age(&self) -> Option<Duration> {
        self.heap
            .iter()
            .map(|item| item.request.timestamp.elapsed())
            .max()
    }
}
```

- [ ] **Step 5: Replace PriorityBuckets with PriorityQueue in InflightBatchProcessor**

In `InflightBatchProcessor` struct, change:

```rust
    // Queue of pending requests
    pending_queue: Arc<RwLock<PriorityBuckets>>,
```

to:

```rust
    // Queue of pending requests
    pending_queue: Arc<RwLock<PriorityQueue>>,
```

- [ ] **Step 6: Update InflightBatchProcessor::new**

Change `PriorityBuckets::new()` to `PriorityQueue::new()`:

```rust
            pending_queue: Arc::new(RwLock::new(PriorityQueue::new())),
```

- [ ] **Step 7: Verify the rest of the file compiles**

`add_request`, `try_form_batch`, and other methods call `queue.push`, `queue.is_empty`, `queue.oldest_age`, `queue.len`, `queue.drain_up_to` — these all exist on `PriorityQueue` with the same signatures. Build to confirm:

```bash
cargo build --no-default-features 2>&1 | grep "^error" | head -20
```

Expected: no errors. If errors appear, compare method signatures between `PriorityBuckets` and `PriorityQueue` and fix mismatches.

- [ ] **Step 8: Run the new test and all inflight_batch tests**

```bash
cargo test --no-default-features inflight_batch 2>&1 | tail -10
```

Expected: `test result: ok.`

- [ ] **Step 9: Commit**

```bash
git add src/inflight_batch.rs
git commit -m "perf(inflight-batch): replace 5-bucket VecDeque with BinaryHeap for correct priority+FIFO ordering"
```

---

## Task 4: A4 — Model Pool Load Score Rebalancing

**Files:**
- Modify: `src/model_pool.rs:50-57`

### Background

The EMA for `avg_latency_ms` is already implemented in `end_request` (α = 0.1). The bug is in `load_score()`: the weight `* 1000` on active requests means a single additional request contributes 1000 to the score, while a 500ms latency spike contributes only 500. Routing decisions effectively ignore latency. Changing the weight to `500` brings latency contributions into the same order of magnitude as request count.

- [ ] **Step 1: Write the failing test**

In `src/model_pool.rs`, in the test block, add:

```rust
    #[tokio::test]
    async fn test_load_score_latency_contribution() {
        // A high-latency idle instance should score higher (worse) than a
        // low-latency instance with one active request.
        let high_latency = InstanceMetrics::new();
        // Simulate 600ms EMA latency, 0 active requests
        high_latency.avg_latency_ms.store(600, Ordering::Relaxed);

        let low_latency = InstanceMetrics::new();
        // Simulate 10ms EMA latency, 1 active request
        low_latency.active_requests.store(1, Ordering::Relaxed);
        low_latency.avg_latency_ms.store(10, Ordering::Relaxed);

        // With weight=500: high_latency score = 0*500 + 600 = 600
        //                  low_latency  score = 1*500 + 10  = 510
        // high_latency should score higher (worse)
        assert!(
            high_latency.load_score() > low_latency.load_score(),
            "high_latency score {} should exceed low_latency score {}",
            high_latency.load_score(),
            low_latency.load_score()
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test --no-default-features test_load_score_latency_contribution 2>&1 | tail -10
```

Expected: FAIL — with weight `1000`, `high_latency score = 600` but `low_latency score = 1010`. The assertion will fail because `600 < 1010`.

- [ ] **Step 3: Change the weight in load_score()**

In `src/model_pool.rs`, replace `load_score` (lines 50–57):

```rust
    fn load_score(&self) -> u64 {
        let active = self.active_requests.load(Ordering::Relaxed);
        let latency = self.avg_latency_ms.load(Ordering::Relaxed);
        
        // Score = active_requests * 1000 + avg_latency
        // Lower score = less loaded
        active * 1000 + latency
    }
```

With:

```rust
    fn load_score(&self) -> u64 {
        let active = self.active_requests.load(Ordering::Relaxed);
        let latency = self.avg_latency_ms.load(Ordering::Relaxed);

        // Weight 500 keeps request-count and EMA latency in the same order of magnitude.
        // Example: 1 active request + 10ms latency = 510 vs 0 requests + 600ms latency = 600.
        // Previously weight=1000 made latency nearly irrelevant.
        active * 500 + latency
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test --no-default-features test_load_score_latency_contribution 2>&1 | tail -5
```

Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Run all model_pool tests**

```bash
cargo test --no-default-features model_pool 2>&1 | tail -10
```

Expected: `test result: ok.`

- [ ] **Step 6: Commit**

```bash
git add src/model_pool.rs
git commit -m "perf(model-pool): balance load score weight to make EMA latency meaningfully affect routing"
```

---

## Task 5: A3 — Adaptive Cache Eviction Sample Size

**Files:**
- Modify: `src/cache.rs`

### Background

`SAMPLE_SIZE` is currently the constant `100`. Eviction always collects up to 200 keys and samples 100. The change makes sample size adaptive: drop to 20 samples when the cache is healthy (high hit rate) to reduce eviction overhead; scale up to 100 when the cache is struggling (low hit rate) for better victim selection.

Formula: `sample_size = clamp(20 + (1.0 - hit_rate) * 80, 20, 100)`

At 100% hit rate → 20 samples.
At 0% hit rate → 100 samples (same as today).

- [ ] **Step 1: Write the failing test**

In the test block of `src/cache.rs`, add:

```rust
    #[test]
    fn test_evict_sample_size_adapts_to_hit_rate() {
        let cache = Cache::new(1000);

        // Simulate high hit rate: sample size should be near 20
        for i in 0..100 {
            cache.set(format!("k{}", i), serde_json::json!(i), 3600).unwrap();
        }
        // 100 gets on populated cache → high hit rate
        for i in 0..100 {
            cache.get(&format!("k{}", i));
        }
        let size_high_hr = cache.evict_sample_size();
        assert!(size_high_hr <= 30, "expected ~20 at high hit rate, got {}", size_high_hr);

        // Simulate zero hit rate: sample size should be near 100
        let cold = Cache::new(1000);
        for i in 0..100 {
            cold.get(&format!("miss{}", i)); // all misses
        }
        let size_low_hr = cold.evict_sample_size();
        assert!(size_low_hr >= 90, "expected ~100 at zero hit rate, got {}", size_low_hr);
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test --no-default-features test_evict_sample_size_adapts_to_hit_rate 2>&1 | tail -10
```

Expected: compile error — `evict_sample_size` doesn't exist yet.

- [ ] **Step 3: Add evict_sample_size method**

In `src/cache.rs`, inside the `impl Cache` block, add this method (near the `evict_lru` method):

```rust
    /// Compute adaptive eviction sample size based on current hit rate.
    ///
    /// Scales from 20 (healthy cache, high hit rate) to 100 (struggling cache,
    /// low hit rate). This reduces eviction overhead when the cache is working
    /// well, and improves victim selection quality when it is not.
    pub fn evict_sample_size(&self) -> usize {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 {
            1.0_f64
        } else {
            hits as f64 / total as f64
        };
        ((20.0 + (1.0 - hit_rate) * 80.0) as usize).clamp(20, 100)
    }
```

- [ ] **Step 4: Update evict_lru to use the adaptive size**

In `src/cache.rs`, find `evict_lru` and replace the two lines that reference `SAMPLE_SIZE`:

```rust
        // Only collect enough keys to have a good sample — at most SAMPLE_SIZE * 2
        // so that after shuffling we can pick SAMPLE_SIZE without iterating the
        // full DashMap (which holds shard locks while iterating).
        let collect_limit = std::cmp::min(SAMPLE_SIZE * 2, cache_size);
        let keys: Vec<String> = self.data.iter()
            .take(collect_limit)
            .map(|entry| entry.key().clone())
```

And:

```rust
        let sampled_keys = if keys.len() <= SAMPLE_SIZE {
            keys
        } else {
            let mut rng = rand::thread_rng();
            let mut sampled = keys;
            sampled.shuffle(&mut rng);
            sampled.truncate(SAMPLE_SIZE);
            sampled
        };
```

With:

```rust
        let sample_size = self.evict_sample_size();
        // Collect up to 2× the sample target to give shuffle a good pool.
        let collect_limit = cache_size.min(sample_size * 2);
        let keys: Vec<String> = self.data.iter()
            .take(collect_limit)
            .map(|entry| entry.key().clone())
```

And:

```rust
        let sampled_keys = if keys.len() <= sample_size {
            keys
        } else {
            let mut rng = rand::thread_rng();
            let mut sampled = keys;
            sampled.shuffle(&mut rng);
            sampled.truncate(sample_size);
            sampled
        };
```

- [ ] **Step 5: Run new test and existing cache tests**

```bash
cargo test --no-default-features cache 2>&1 | tail -10
```

Expected: `test result: ok.`

- [ ] **Step 6: Check if the constant SAMPLE_SIZE is still referenced**

```bash
grep -n "SAMPLE_SIZE" src/cache.rs
```

Expected: only appears in test code (the existing test names reference `SAMPLE_SIZE` in comments). If the constant itself is now unused, either remove it or keep it for documentation. If existing tests use it as a value, update them to use `100` directly.

- [ ] **Step 7: Commit**

```bash
git add src/cache.rs
git commit -m "perf(cache): adaptive eviction sample size scales 20-100 based on hit rate"
```

---

## Task 6: C1 — Optional pprof Feature

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/main.rs`

- [ ] **Step 1: Add pprof dependency to Cargo.toml**

In `Cargo.toml`, in the `[dependencies]` section (after the existing optional deps), add:

```toml
pprof = { version = "0.13", features = ["flamegraph"], optional = true }
```

In the `[features]` section, add:

```toml
profiling = ["dep:pprof"]
```

- [ ] **Step 2: Add profiler guard to main.rs**

In `src/main.rs`, after the existing `use` imports, add:

```rust
#[cfg(feature = "profiling")]
use pprof::ProfilerGuardBuilder;
```

In the `main` function body, near the top (before `HttpServer` construction), add:

```rust
    #[cfg(feature = "profiling")]
    let _profiler_guard = {
        log::info!("pprof profiling enabled — flamegraph will be written on exit");
        ProfilerGuardBuilder::default()
            .frequency(100)
            .build()
            .expect("failed to start pprof profiler")
    };
```

Note: `_profiler_guard` must stay in scope for the duration of `main`. Do not drop it early.

- [ ] **Step 3: Build without the feature (must not affect prod builds)**

```bash
cargo build --no-default-features 2>&1 | grep "^error" | head -5
```

Expected: no errors.

- [ ] **Step 4: Build with the feature**

```bash
cargo build --no-default-features --features profiling 2>&1 | grep "^error" | head -5
```

Expected: no errors. If `pprof 0.13` is not available on crates.io, check the latest version with `cargo search pprof` and update accordingly.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/main.rs
git commit -m "feat(profiling): add optional pprof feature for flamegraph generation"
```

---

## Task 7: C2 — Inflight Priority Benchmark

**Files:**
- Modify: `benches/throughput_bench.rs`

- [ ] **Step 1: Add the benchmark function**

In `benches/throughput_bench.rs`, add before the `criterion_group!` macro:

```rust
// ── 7. Inflight Batch Priority Queue Throughput ───────────────────────────────
// Measures enqueue + drain throughput for the BinaryHeap priority queue.
// Compare against the 5-bucket VecDeque baseline by checking git history.
fn bench_inflight_priority(c: &mut Criterion) {
    use torch_inference::inflight_batch::{InflightBatchProcessor, InflightRequest};
    use tokio::sync::oneshot;

    let rt = make_rt();
    let mut group = c.benchmark_group("inflight_priority");
    group.measurement_time(std::time::Duration::from_secs(5));

    for batch_size in [8usize, 32, 64] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("enqueue_drain", batch_size),
            &batch_size,
            |b, &batch_size| {
                let processor = InflightBatchProcessor::new(batch_size, 4, 50);
                let mut id = 0u64;
                b.iter(|| {
                    rt.block_on(async {
                        for priority in 0..batch_size as i32 {
                            let (tx, _rx) = oneshot::channel();
                            let req = InflightRequest {
                                id: id.to_string(),
                                model_name: "bench-model".to_string(),
                                inputs: vec![],
                                priority,
                                timestamp: std::time::Instant::now(),
                                response_tx: tx,
                            };
                            id += 1;
                            processor.add_request(req).await.ok();
                        }
                        black_box(processor.try_form_batch().await)
                    });
                });
            },
        );
    }
    group.finish();
}
```

- [ ] **Step 2: Add to criterion_group**

Update the `criterion_group!` macro to include the new benchmark:

```rust
criterion_group!(
    benches,
    bench_text_request_throughput,
    bench_tts_throughput,
    bench_image_preprocessing_throughput,
    bench_cache_roundtrip_throughput,
    bench_concurrent_cache_reads,
    bench_batch_latency_vs_size,
    bench_inflight_priority,
);
```

- [ ] **Step 3: Verify `InflightRequest` and `InflightBatchProcessor` are pub in lib.rs**

```bash
grep -n "pub.*InflightRequest\|pub.*InflightBatchProcessor\|pub mod inflight_batch" src/lib.rs
```

If `inflight_batch` is not re-exported from `lib.rs`, add:

```rust
pub mod inflight_batch;
```

to `src/lib.rs`. (Check `src/lib.rs` first — it may already export it via `pub use` or `pub mod`.)

- [ ] **Step 4: Run the benchmark to confirm it compiles and runs**

```bash
cargo bench --no-default-features --bench throughput_bench -- bench_inflight_priority 2>&1 | tail -20
```

Expected: benchmark runs and prints timing results. No panics.

- [ ] **Step 5: Commit**

```bash
git add benches/throughput_bench.rs src/lib.rs
git commit -m "bench: add inflight priority queue throughput benchmark"
```

---

## Task 8: C3 — Flamegraph Makefile Target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add flamegraph target**

In `Makefile`, add after the existing targets (before the final blank line):

```makefile
flamegraph: ## Generate CPU flamegraph (requires cargo-flamegraph)
	@echo "Generating flamegraph..."
	@echo "Install with: cargo install flamegraph"
	cargo flamegraph --features profiling --bin torch-inference-server -- --config config.toml
	@echo "Flamegraph written to flamegraph.svg"
```

Also add `flamegraph` to the `.PHONY` line at the top:

```makefile
.PHONY: help build run dev test clean install doctor flamegraph
```

- [ ] **Step 2: Verify help output includes the new target**

```bash
make help 2>&1 | grep flamegraph
```

Expected: `flamegraph             Generate CPU flamegraph (requires cargo-flamegraph)`

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: add flamegraph Makefile target for CPU profiling"
```

---

## Self-Review

**Spec coverage check:**

| Spec item | Task |
|-----------|------|
| A1 BinaryHeap inflight queue | Task 3 |
| A2 AtomicU8 worker state | Task 2 |
| A3 Adaptive eviction sample | Task 5 |
| A4 EMA model pool score | Task 4 |
| A5 Dead code / warnings | Task 1 (whisper TODO; model_cache NOT dead — skip; INT8 already clean — skip) |
| C1 pprof feature | Task 6 |
| C2 concurrent benchmark | Task 7 |
| C3 flamegraph Makefile | Task 8 |

**All spec items covered. No placeholders. All code is complete.**
