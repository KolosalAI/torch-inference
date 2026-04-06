# Optimization Design — torch-inference

**Date:** 2026-04-04  
**Priority:** Throughput > Latency > Memory > Code Quality  
**Scope:** Phase A (targeted hot-path fixes) + Phase C (profiling integration)  
**Out of scope:** API surface, auth, resilience patterns, TTS pipeline, ONNX backend, config schema

---

## Background

The codebase is a production Rust inference server with existing optimization infrastructure (LRU cache, dynamic batching, worker pool, tensor pooling, FP16/quantization). Phase A addresses specific bottlenecks identified by code inspection. Phase C adds profiling tooling to validate those gains and surface anything missed.

---

## Phase A — Targeted Hot-Path Improvements

### A1 — Inflight Batch Priority Queue

**File:** `src/inflight_batch.rs`

**Problem:** The current 5-bucket `Vec<Item>` design has no ordering within a priority bucket. Items at the same priority level are served in arbitrary order (last-in can precede first-in), causing head-of-line blocking and unfair scheduling.

**Change:** Replace the 5 `Vec` buckets with a single `BinaryHeap<Reverse<(u8, u64, Item)>>` where the tuple is `(priority, sequence_number, item)`. A global `AtomicU64` provides monotonically increasing sequence numbers. `Reverse` makes it a min-heap so lower priority values (higher urgency) and earlier sequence numbers are dequeued first.

**Impact:** Correct FIFO within priority levels, O(log n) insert and pop (vs O(1) push / O(batch_size) linear scan). Semaphore and batch drain logic unchanged.

---

### A2 — Worker Pool Lock-Free State

**File:** `src/worker_pool.rs`

**Problem:** Each worker holds a `RwLock<WorkerState>`. Every health check, idle poll, and task assignment acquires a read lock. Under high concurrency this creates lock contention on the hot path.

**Change:** Replace `RwLock<WorkerState>` with an `AtomicU8` using these state values:
- `0` = Idle
- `1` = Busy  
- `2` = Draining
- `3` = Dead

State transitions use compare-and-swap (`compare_exchange`). The `RwLock` is kept only for infrequently-written metadata (task count, latency histogram). Hot-path reads (idle check, task assignment) become lock-free atomic loads.

**Impact:** Eliminates read-lock overhead on the most frequently executed path in the worker pool.

---

### A3 — Adaptive Cache Eviction Sample Size

**File:** `src/cache.rs`

**Problem:** The eviction path always samples exactly 5 random candidates regardless of cache pressure or hit rate. Under high miss rate (cache thrashing), 5 candidates is insufficient to find a good eviction target, leading to hot-data eviction.

**Change:** Compute sample size dynamically each eviction cycle:

```
sample_size = clamp(5 + (1.0 - hit_rate) * 15, 5, 20)
```

- Hit rate sourced from existing hit/miss `AtomicU64` counters (no new state).
- At 100% hit rate: sample size = 5 (cheap, cache is healthy).
- At 0% hit rate: sample size = 20 (more thorough, cache is struggling).

**Impact:** Better eviction decisions under load without adding overhead when the cache is working well.

---

### A4 — EMA-Based Model Instance Selection

**File:** `src/model_pool.rs`

**Problem:** Current load score: `active_requests * 1000 + avg_latency_ms`. A latency spike of 100ms adds 100 to the score; one extra active request adds 1000. Latency contributes less than 0.01% of the score in practice, making the selection effectively request-count-only.

**Change:** Replace with EMA latency:

```
ema_latency = α * new_sample_ms + (1 - α) * ema_latency   // α = 0.1
score = active_requests * request_weight + ema_latency
request_weight = 500 (base), adjusted down if ema_latency variance > threshold
```

- EMA smooths out transient spikes while tracking trends.
- Dynamic `request_weight` allows latency to dominate when an instance is consistently slow.
- α and base weight are constants (not config-exposed) to keep it simple.

**Impact:** Routes new requests away from instances experiencing latency degradation, not just queue depth.

---

### A5 — Dead Code & Warning Cleanup

**Files:** `src/core/model_cache.rs`, `src/core/whisper_stt.rs`, `src/torch_optimization.rs`

**Changes:**

1. **`model_cache.rs`** — Delete file. Remove `mod model_cache;` from `src/core/mod.rs`. Verify nothing imports it before deletion.

2. **`whisper_stt.rs`** — The actual inference body is unimplemented. Add `unimplemented!("Whisper STT inference not yet implemented")` to the inference method body. If nothing outside the module calls into it, gate with `#[allow(dead_code)]` or remove the public API stub.

3. **`torch_optimization.rs` ~line 249** — The INT8 dynamic quantization comment/warning implies it can be applied at runtime, but PyTorch requires it at export time. Remove the misleading comment and gate the quantization path with a compile-time or runtime check that emits a clear error if called inappropriately.

**Impact:** Eliminates dead code warnings, reduces binary size marginally, prevents future confusion around the INT8 path.

---

## Phase C — Profiling Integration

### C1 — Optional pprof Feature

**Files:** `Cargo.toml`, `src/main.rs`

Add a `profiling` Cargo feature:

```toml
[features]
profiling = ["dep:pprof"]

[dependencies]
pprof = { version = "0.13", features = ["flamegraph"], optional = true }
```

In `main.rs`, wrap a `ProfilerGuard` behind `#[cfg(feature = "profiling")]`. The guard runs for the process lifetime and writes a flamegraph on `SIGINT` or explicit `/debug/flamegraph` endpoint (admin-only, gated by auth).

Feature is off by default — production builds are unaffected.

---

### C2 — Concurrent Multi-Model Benchmark

**File:** `benches/throughput_bench.rs`

Add `bench_concurrent_multimodel`: spawn 4 model type workers (image classification, TTS, audio, LLM stub) with 50 concurrent callers each, run for 10 seconds, report:
- Total requests/sec (throughput)
- p50, p95, p99 latency per model type
- Worker pool utilization %

This gives a realistic mixed-workload baseline to measure Phase A gains against.

---

### C3 — Flamegraph Makefile Target

**File:** `Makefile`

Add:

```makefile
flamegraph:
	cargo flamegraph --features profiling --bin torch-inference -- --config config.toml
```

---

## Verification

Each Phase A change is verified by:
1. Existing unit tests must pass (`cargo test`)
2. Relevant Criterion benchmark must show no regression (or improvement)
3. Phase A5 cleanup: `cargo build` with zero warnings after removal

Phase C is verified by:
- `cargo build` succeeds with and without `--features profiling`
- `make flamegraph` produces a valid SVG
- New benchmark runs without panicking

---

## Implementation Order

1. A5 (cleanup) — no risk, unblocks clean diffs
2. A2 (worker pool atomics) — self-contained, high impact
3. A1 (inflight batch heap) — self-contained, high impact
4. A4 (model pool EMA) — self-contained, medium impact
5. A3 (adaptive cache eviction) — self-contained, low risk
6. C1 (pprof feature)
7. C2 (benchmark)
8. C3 (Makefile target)
