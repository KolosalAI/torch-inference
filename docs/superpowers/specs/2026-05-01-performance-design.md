# Performance Improvements — Kolosal Inference Server

**Date:** 2026-05-01  
**Status:** Approved  
**Scope:** All four deferred follow-ups from the CLAUDE.md audit (cf2b566 → dcd7cf2)

---

## Overview

Four targeted optimizations addressing the highest-impact bottlenecks across TTS, STT, YOLO detection, and the model cache. Implementation follows Approach B (top-down, highest visible impact first). All changes are full migrations — no backward-compat shims.

---

## 1. YOLO SIMD Preprocessing

**File:** `src/core/ort_yolo.rs`

**Problem:** Input images are resized with `image::imageops::Lanczos3` and normalized with scalar f32 arithmetic. The classifier path (`image_pipeline.rs`) already uses `fast_image_resize` (CatmullRom) + `wide::f32x8` AVX2/NEON vectorized normalization.

**Change:** Replace the resize call and normalize loop in `ort_yolo.rs` with the same SIMD pipeline the classifier uses. No public type changes. Compile-gated behind the existing `simd-image` Cargo feature identically to `image_pipeline.rs`.

**Expected gain:** ~50 ms → ~15 ms per 640×640 frame.

**Constraints:** Must fall back gracefully when `simd-image` feature is not enabled (keep the existing scalar path as the non-feature branch).

---

## 2. Sharded TTS Synthesis Cache + `Arc<AudioData>`

**Files:** `src/core/audio.rs`, `src/core/tts_manager.rs`, all TTS handler call sites

**Problem (part A):** Every cache hit in `tts_manager.rs` clones the full `AudioData` struct (`Vec<f32>` samples, ~100–500 KB) inside the lock.

**Problem (part B):** A single `Mutex<LruCache<u64, AudioData>>` serializes all concurrent synthesis requests.

**Change (part A — `Arc<AudioData>`):** `AudioData` is wrapped in `Arc` at the synthesis output site. All call sites that previously received `AudioData` by value now receive `Arc<AudioData>`. Handlers needing mutation call `Arc::make_mut` (clones only when refcount > 1). Eliminates the large clone on every cache hit.

**Change (part B — 16-way sharded LRU):** Replace the single `Mutex<LruCache<u64, Arc<AudioData>>>` with a `[Mutex<LruCache<u64, Arc<AudioData>>>; 16]` array. Shard index: `key & 0xF` (low 4 bits of the FNV-1a hash, well-distributed). Total capacity unchanged; divided evenly across shards. Pattern mirrors the existing `dedup.rs` implementation. 16 concurrent requests with distinct keys never contend.

**Expected gain:** Eliminates serialization under concurrent load; reduces per-hit allocation from ~100–500 KB clone to a 5 ns `Arc` clone.

**Constraints:** Both changes ship together — the sharded cache stores `Arc<AudioData>`, so part A must land before part B compiles.

---

## 3. STT Resampler Pool

**Files:** `src/core/audio.rs`, `src/main.rs`

**Problem:** `FftFixedInOut` is constructed fresh on every STT decode call. Construction is ~5–15 ms. The resampler is `&mut self` per-process, so it is not `Send + Sync` and cannot be shared across threads directly — it requires a pool.

**Change:** Add `ResamplerPool` struct inside `audio.rs` (not a new file):

```
ResamplerPool {
    inner: Mutex<HashMap<(u32, u32), Vec<FftFixedInOut<f32>>>>,
    max_per_key: usize,  // capped at 8
}
```

- On each STT request: pop an existing resampler for `(input_rate, output_rate)` if available; construct one if the pool is empty for that key.
- After processing: return the resampler to the pool (if pool size < `max_per_key`).
- `AudioProcessor::new` gains an `Arc<ResamplerPool>` parameter. `main.rs` constructs one `ResamplerPool`, wraps in `Arc`, and passes it in. All existing call sites in `main.rs` are updated.

**Expected gain:** 5–15 ms per STT request.

**Constraints:** Pool is bounded at 8 instances per rate pair. Rate pairs beyond 8 concurrent callers fall back to constructing a fresh resampler (no error, just no pooling benefit for that call).

---

## 4. ModelCache `Arc<dyn Any>`

**Files:** `src/core/model_cache.rs`, all call sites writing to / reading from the cache

**Problem:** Cache values are stored as `Arc<Vec<u8>>` (serialized JSON). Every cache hit calls `serde_json::from_slice` to deserialize back to the concrete type — even though the value was just computed.

**Change:** Store `Arc<dyn Any + Send + Sync>` directly. Cache type becomes `LruCache<u64, Arc<dyn Any + Send + Sync>>`. Write path wraps the concrete value in `Arc`; read path calls `Arc::downcast::<T>()`. If downcast fails (impossible in practice given the key encodes the call site), treat as a miss and recompute.

- Remove the serde dependency from `model_cache.rs` entirely.
- `#[derive(Serialize, Deserialize)]` stays on result types — still needed for HTTP responses.
- Eliminates `serde_json::from_slice` from the cache hot path.

**Expected gain:** Measurable at >80% cache hit rate; eliminates the single most expensive per-hit operation.

**Constraints:** Full migration — all call sites updated. No backward-compat JSON fallback path.

---

## 5. Bonus: RateLimiter Map TTL Eviction

**File:** `src/middleware/rate_limit.rs` (or equivalent)

**Problem:** The per-IP `DashMap` in the rate limiter has no TTL eviction. Entries accumulate unboundedly over months in high-traffic deployments.

**Change:** Add a background task (spawned at startup in `main.rs`) that sweeps the map every 60 seconds and removes entries whose last-seen timestamp is older than the rate-limit window. O(N) sweep, runs on the Tokio blocking pool.

**Expected gain:** Prevents long-running memory leak. No latency impact.

---

## Benchmarking & Verification

### Existing bench suite (baseline → after)

Run before any changes, then after each step:

```bash
cargo bench --bench tts_bench
cargo bench --bench audio_bench
cargo bench --bench detection_bench
cargo bench --bench cache_bench
```

### New micro-benchmarks to add

| Bench file | Measures |
|---|---|
| `tts_bench.rs` (extend) | Sharded cache: 16-way concurrent access vs. single-lock baseline |
| `audio_bench.rs` (extend) | Resampler pool: pooled vs. fresh-construct, 48 kHz→16 kHz |
| `detection_bench.rs` (extend) | SIMD preprocess vs. Lanczos3+scalar, 640×640 input |
| `cache_bench.rs` (extend) | `Arc<dyn Any>` hit cost vs. serde round-trip hit cost |

Each bench uses Criterion's `BenchmarkGroup` with two named functions (`_baseline`, `_optimized`) for direct comparison with confidence intervals.

### Pass criteria

- All existing `cargo test` green before and after every step.
- Each new bench shows improvement in the expected direction.
- No regressions on unchanged paths.
- `cargo build --release` clean compile as the final gate.

---

## Implementation Order

1. Run existing benches → record baseline numbers
2. YOLO SIMD preprocessing (self-contained, no type changes)
3. `Arc<AudioData>` migration + 16-way sharded TTS cache (together)
4. STT resampler pool
5. ModelCache `Arc<dyn Any>`
6. RateLimiter TTL sweep
7. Add new bench cases, run full suite, record final numbers
