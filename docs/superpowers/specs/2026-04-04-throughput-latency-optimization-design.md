# Throughput & Latency Optimization Design

**Date:** 2026-04-04  
**Scope:** `src/torch_optimization.rs`, `src/inflight_batch.rs`  
**Goals:** Reduce first-response latency (TTFA) and increase sustained requests/sec across CPU, CUDA, and MPS platforms.

---

## Problem Summary

Four bugs/bottlenecks were identified through code review:

1. `infer_batch` runs N separate forward passes instead of one batched forward pass — defeats the purpose of batching entirely.
2. `InflightBatchProcessor::add_request` uses `VecDeque::insert` (O(n)) to maintain priority order — every enqueue shifts elements under lock.
3. `InflightBatchProcessor::try_form_batch` acquires a write lock just to peek at queue state — blocks all writers during the check.
4. `permit.forget()` in `try_form_batch` leaks semaphore permits on panic, gradually starving throughput over time.

---

## Fix 1: Real tensor batching in `infer_batch`

**File:** `src/torch_optimization.rs`

**Current behavior:** Calls `self.infer(input)` in a loop — N separate forward passes, no GPU utilization benefit.

**New behavior:**
- Validate all inputs share the same shape; if not, fall back to the existing serial path (preserving correctness for heterogeneous batches).
- Stack homogeneous inputs along dim 0: `Tensor::stack(&inputs, 0)` → shape `[N, dims...]`.
- Run one forward pass on the batched tensor.
- Split result back with `batched_output.unbind(0)` → N tensors of shape `[dims...]`.

**Signature stays the same.** The fallback path means no breaking change for callers using mixed-shape inputs.

**Expected impact:** For CUDA/MPS, GPU utilization increases significantly; for CPU, avoids N×kernel-launch overhead.

---

## Fix 2: O(1) priority queue via bucketed `VecDeque`

**File:** `src/inflight_batch.rs`

**Current behavior:** `VecDeque::insert(pos, request)` scans the queue to find the insertion point, then shifts all following elements — O(n) per enqueue.

**New behavior:** Replace `Arc<RwLock<VecDeque<InflightRequest>>>` with `Arc<RwLock<PriorityBuckets>>` where:

```rust
struct PriorityBuckets {
    buckets: [VecDeque<InflightRequest>; 5],  // indices 0 (lowest) to 4 (highest)
}
```

- Enqueue: clamp `priority` to `0..=4`, append to the matching bucket — O(1).
- Drain: iterate buckets 4 → 0, drain up to `max_batch_size` from the highest non-empty buckets first — O(batch_size).
- `queue_len()`: sum of all bucket lengths.

The existing `i32` priority field is preserved. Values `<= 0` go into bucket 0; values `>= 4` go into bucket 4; values 1–3 map directly.

**Expected impact:** Eliminates O(n) shifting under lock on every enqueue, especially important under high concurrency.

---

## Fix 3: Two-phase locking in `try_form_batch`

**File:** `src/inflight_batch.rs`

**Current behavior:** Acquires `pending_queue.write()` immediately, blocking all concurrent `add_request` calls for the duration of the peek and drain.

**New behavior:**
1. Acquire **read lock**: check `is_empty()` and compute `oldest_age`. Return `None` fast if no batch should form.
2. Only if the check passes, drop the read lock and acquire **write lock** to drain.
3. After acquiring the write lock, **re-verify** the queue is still non-empty (TOCTOU safety) before draining.

This allows `add_request` to proceed concurrently during the peek phase. The write lock is held only for the brief drain operation.

**Expected impact:** Reduces contention between the dispatcher and producers under high load.

---

## Fix 4: RAII semaphore guard replaces `permit.forget()`

**File:** `src/inflight_batch.rs`

**Current behavior:** `permit.forget()` permanently leaks the semaphore permit if the caller panics or drops the batch without calling `complete_batch`. Over time this reduces `max_inflight_batches` below its configured value.

**New behavior:** `try_form_batch` returns `Option<InflightBatchGuard>` instead of `Option<Vec<InflightRequest>>`:

```rust
pub struct InflightBatchGuard<'a> {
    processor: &'a InflightBatchProcessor,
    pub batch: Vec<InflightRequest>,
    permit: OwnedSemaphorePermit,  // released on Drop automatically
}

impl<'a> InflightBatchGuard<'a> {
    /// Record stats and consume the guard cleanly.
    pub fn complete(self, processing_time_ms: u64) { ... }
}

impl Drop for InflightBatchGuard<'_> {
    fn drop(&mut self) {
        // permit released here; inflight_batches decremented if not already completed
    }
}
```

Callers update from:
```rust
let batch = processor.try_form_batch().await?;
// ... process ...
processor.complete_batch(batch.len(), latency_ms);
```
to:
```rust
let guard = processor.try_form_batch().await?;
// ... process guard.batch ...
guard.complete(latency_ms);
```

**Expected impact:** Correctness fix — prevents semaphore starvation on panics. No throughput regression; slight improvement from eliminating the `inflight_batches.fetch_sub` / `add_permits` split across two call sites.

---

## Error handling

- Fix 1 (tensor stacking): shape mismatch → fall back to serial path, no error propagated.
- Fixes 2–4: purely internal restructuring, same error surface as today.

---

## Testing

Each fix has existing test coverage in `src/inflight_batch.rs` and `src/torch_optimization.rs`. Tests will be updated to:
- Fix 1: Add a test asserting `infer_batch` on homogeneous inputs produces the same output as N serial `infer` calls.
- Fix 2: Verify priority ordering is preserved with the bucketed implementation; existing priority tests pass unchanged.
- Fix 3: Existing concurrency tests (`test_batch_concurrent_additions`, `test_batch_concurrent_get_and_add`) cover the two-phase locking path.
- Fix 4: Add a test that drops an `InflightBatchGuard` without calling `complete` and verifies `can_accept_batch()` returns true (permit was released by `Drop`).

---

## Out of scope

- Approach 2 (dedicated dispatcher loop) — deferred to a follow-up.
- Approach 3 (platform-aware startup tuning) — deferred to a follow-up.
- Changes to `batch.rs`, `model_pool.rs`, `worker_pool.rs`, `cache.rs`, `dedup.rs`.
