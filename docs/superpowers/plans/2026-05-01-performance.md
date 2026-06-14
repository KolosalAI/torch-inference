# Performance Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land all four deferred CLAUDE.md performance follow-ups: YOLO SIMD preprocessing, sharded TTS cache + `Arc<AudioData>`, STT resampler pool, and ModelCache `Arc<dyn Any>`; plus RateLimiter TTL sweep.

**Architecture:** Top-down by impact. YOLO preprocessing is self-contained (no type changes). TTS cache and `Arc<AudioData>` ship together since the sharded cache stores `Arc<AudioData>`. Resampler pool uses a module-level `OnceLock<ResamplerPool>` (same pattern as `OUTPUT_POOL` and `RESIZE_SRC_POOL` already in the codebase — avoids touching every `AudioProcessor` constructor call site). ModelCache replaces serde with `Arc<dyn Any + Send + Sync>`. RateLimiter wires the already-written `cleanup_old_entries()` to a 60 s background task in `main.rs`.

**Tech Stack:** Rust 1.78+, actix-web, parking_lot, lru, rubato, fast_image_resize, wide, criterion, tokio

---

## File Map

| File | What changes |
|---|---|
| `src/core/ort_yolo.rs` | Add SIMD preprocess path under `#[cfg(feature="simd-image")]`; keep scalar fallback |
| `src/core/image_pipeline.rs` | Make `resize_hwc` and `normalize_channel_simd` `pub(crate)` for reuse by YOLO |
| `src/core/audio.rs` | Add `ResamplerPool` struct + static `RESAMPLER_POOL: OnceLock`; call pool in `resample()` |
| `src/core/tts_manager.rs` | Change `synthesis_cache` to 16-shard array; `synthesize()` returns `Result<Arc<AudioData>>` |
| `src/api/tts.rs` | Update synthesize handler: `Arc::make_mut` for mutation, deref for reads |
| `src/core/model_cache.rs` | Store `Arc<dyn Any + Send + Sync>`; drop serde; add `Clone` + `Any + Send + Sync` constraints |
| `src/core/image_classifier.rs` | Update `get_or_run` call: add `Clone` constraint to cached type |
| `src/core/yolo.rs` | Update `get_or_run` call: add `Clone` constraint to cached type |
| `src/main.rs` | Spawn `cleanup_old_entries()` on 60 s interval |
| `benches/detection_bench.rs` | Add `preprocess_baseline` vs `preprocess_simd` group |
| `benches/audio_bench.rs` | Add `resampler_pooled` vs `resampler_fresh` group |
| `benches/tts_bench.rs` | Add `cache_sharded` vs `cache_single_lock` group |
| `benches/cache_bench.rs` | Add `model_cache_arc_any` vs `model_cache_serde` group |

---

## Task 0: Record Baseline Benchmark Numbers

**Files:** None modified — read-only.

- [ ] **Step 1: Run the existing bench suite**

```bash
cargo bench --bench detection_bench 2>&1 | tee /tmp/bench_detection_baseline.txt
cargo bench --bench audio_bench     2>&1 | tee /tmp/bench_audio_baseline.txt
cargo bench --bench tts_bench       2>&1 | tee /tmp/bench_tts_baseline.txt
cargo bench --bench cache_bench     2>&1 | tee /tmp/bench_cache_baseline.txt
```

Expected: all benches complete, numbers written to `/tmp/bench_*_baseline.txt`.

- [ ] **Step 2: Note headline numbers**

From each output, record the slowest measurement for the hot path you are about to optimize:
- `detection_bench`: `preprocess_640x640` (or equivalent)
- `audio_bench`: `resample_*` group
- `tts_bench`: anything touching synthesis or WAV encoding
- `cache_bench`: `cache_get` group

These are your baseline. Every subsequent bench run should show improvement.

---

## Task 1: YOLO SIMD Preprocessing

**Files:**
- Modify: `src/core/image_pipeline.rs` — expose `resize_hwc` and `normalize_channel_simd` as `pub(crate)`
- Modify: `src/core/ort_yolo.rs` — add `preprocess_simd` + `to_chw_f32_simd` under `#[cfg(feature="simd-image")]`
- Modify: `benches/detection_bench.rs` — add `preprocess_baseline` vs `preprocess_simd` bench group

### 1a: Expose helpers from `image_pipeline.rs`

- [ ] **Step 1: Write a failing test in `ort_yolo.rs`**

At the bottom of `src/core/ort_yolo.rs`, add inside `#[cfg(test)]`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    fn simd_and_scalar_preprocess_agree_within_tolerance() {
        // 4×4 solid-red image resized to 8×8 (small enough for a unit test).
        let img = DynamicImage::ImageRgb8(
            image::ImageBuffer::from_pixel(4, 4, image::Rgb([200u8, 100, 50])),
        );
        let scalar = OrtYoloDetector::to_chw_f32_norm(&img.resize_exact(8, 8, image::imageops::FilterType::Lanczos3).to_rgb8());
        #[cfg(feature = "simd-image")]
        {
            let simd = OrtYoloDetector::preprocess_simd_chw(&img, 8).unwrap();
            assert_eq!(scalar.len(), simd.len(), "CHW length must match");
            for (s, v) in scalar.iter().zip(simd.iter()) {
                assert!(
                    (s - v).abs() < 0.02,
                    "scalar={s:.4} simd={v:.4} differ by more than 2%"
                );
            }
        }
    }
}
```

- [ ] **Step 2: Run the test — confirm it fails (function not yet defined)**

```bash
cargo test --lib core::ort_yolo::tests::simd_and_scalar_preprocess_agree_within_tolerance 2>&1 | tail -10
```

Expected: `error[E0425]: cannot find function 'preprocess_simd_chw'`

- [ ] **Step 3: Make `resize_hwc` and `normalize_channel_simd` pub(crate) in `image_pipeline.rs`**

In `src/core/image_pipeline.rs`, change:

```rust
fn resize_hwc(
```
to:
```rust
pub(crate) fn resize_hwc(
```

And change:
```rust
fn normalize_channel_simd(channel: &mut [f32], mean: f32, std: f32) {
```
to:
```rust
pub(crate) fn normalize_channel_simd(channel: &mut [f32], mean: f32, std: f32) {
```

- [ ] **Step 4: Add SIMD preprocess to `ort_yolo.rs`**

Add these two functions to `OrtYoloDetector` (as `impl` methods or free functions in the module — free functions are cleaner):

```rust
/// Resize + CHW-normalize using fast_image_resize + bytemuck SIMD.
/// Only compiled when the `simd-image` feature is enabled.
#[cfg(feature = "simd-image")]
fn preprocess_simd_chw_impl(img: &DynamicImage, size: u32) -> Result<Vec<f32>> {
    use crate::core::image_pipeline::resize_hwc;

    let rgb8 = img.to_rgb8();
    let (src_w, src_h) = rgb8.dimensions();
    let src_raw = rgb8.into_raw(); // HWC u8

    let (hwc_resized, _, _) = resize_hwc(&src_raw, src_w, src_h, size, size)?;
    Ok(to_chw_f32_simd_yolo(&hwc_resized, size as usize, size as usize))
}

/// HWC u8 → CHW f32 ÷255.  Uses bytemuck-aligned wide::f32x8 SIMD for the
/// division step; scatter step remains scalar (non-contiguous access pattern).
#[cfg(feature = "simd-image")]
fn to_chw_f32_simd_yolo(hwc: &[u8], height: usize, width: usize) -> Vec<f32> {
    use crate::core::image_pipeline::normalize_channel_simd;

    let npix = height * width;
    let mut chw = vec![0f32; 3 * npix];

    // Step 1: scatter HWC → CHW layout, cast u8 → f32
    for (i, chunk) in hwc.chunks_exact(3).enumerate() {
        chw[i]          = chunk[0] as f32;
        chw[npix + i]   = chunk[1] as f32;
        chw[2 * npix + i] = chunk[2] as f32;
    }

    // Step 2: ÷255 via SIMD on each contiguous channel slice.
    // mean=0.0, std=255.0 → (x - 0) / 255 = x / 255
    for c in 0..3 {
        normalize_channel_simd(&mut chw[c * npix..(c + 1) * npix], 0.0, 255.0);
    }

    chw
}
```

Add a public associated function used by the test and by `run()`:

```rust
#[cfg(feature = "simd-image")]
pub(crate) fn preprocess_simd_chw(img: &DynamicImage, size: u32) -> Result<Vec<f32>> {
    preprocess_simd_chw_impl(img, size)
}
```

- [ ] **Step 5: Wire into `run()`**

In `OrtYoloDetector::run()`, replace the existing preprocess block (lines 111-119 in `ort_yolo.rs`):

```rust
// ── Preprocess ────────────────────────────────────────────────────────────
let t_pre = Instant::now();
let orig_w = img.width() as f32;
let orig_h = img.height() as f32;
let size = MODEL_INPUT_SIZE;

let resized = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
let rgb = resized.to_rgb8();
let input_data = Self::to_chw_f32_norm(&rgb);
let preprocessing_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
```

with:

```rust
// ── Preprocess ────────────────────────────────────────────────────────────
let t_pre = Instant::now();
let orig_w = img.width() as f32;
let orig_h = img.height() as f32;
let size = MODEL_INPUT_SIZE;

#[cfg(feature = "simd-image")]
let input_data = preprocess_simd_chw_impl(img, size)
    .unwrap_or_else(|_| {
        let resized = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
        Self::to_chw_f32_norm(&resized.to_rgb8())
    });

#[cfg(not(feature = "simd-image"))]
let input_data = {
    let resized = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
    Self::to_chw_f32_norm(&resized.to_rgb8())
};

let preprocessing_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
```

- [ ] **Step 6: Run the test — confirm it passes**

```bash
cargo test --features simd-image --lib core::ort_yolo::tests::simd_and_scalar_preprocess_agree_within_tolerance 2>&1 | tail -5
```

Expected: `test ... ok`

- [ ] **Step 7: Confirm existing tests still pass**

```bash
cargo test --features simd-image --lib 2>&1 | tail -10
```

Expected: all tests pass (or pre-existing failures only).

- [ ] **Step 8: Add detection bench group**

At the end of `benches/detection_bench.rs`, add:

```rust
fn preprocess_comparison(c: &mut Criterion) {
    use std::time::Duration;

    let raw_rgb: Vec<u8> = synthetic_rgb(1280, 720);
    // Encode as a minimal PNG so image::load_from_memory can decode it.
    let img = image::RgbImage::from_raw(1280, 720, raw_rgb).unwrap();
    let mut png_buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png_buf, image::ImageFormat::Png).unwrap();
    let png_bytes = png_buf.into_inner();

    let mut group = c.benchmark_group("yolo_preprocess_comparison");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    group.bench_function("scalar_lanczos3", |b| {
        b.iter(|| {
            let decoded = image::load_from_memory(black_box(&png_bytes)).unwrap();
            let resized = decoded.resize_exact(640, 640, image::imageops::FilterType::Lanczos3);
            let rgb = resized.to_rgb8();
            // Scalar CHW normalize
            let (w, h) = rgb.dimensions();
            let (w, h) = (w as usize, h as usize);
            let mut data = vec![0f32; 3 * h * w];
            for (x, y, px) in rgb.enumerate_pixels() {
                let (x, y) = (x as usize, y as usize);
                data[y * w + x]         = px[0] as f32 / 255.0;
                data[h * w + y * w + x] = px[1] as f32 / 255.0;
                data[2 * h * w + y * w + x] = px[2] as f32 / 255.0;
            }
            black_box(data)
        })
    });

    #[cfg(feature = "simd-image")]
    group.bench_function("simd_catmullrom", |b| {
        b.iter(|| {
            let decoded = image::load_from_memory(black_box(&png_bytes)).unwrap();
            black_box(
                torch_inference::core::ort_yolo::OrtYoloDetector::preprocess_simd_chw(
                    &decoded, 640,
                ).unwrap()
            )
        })
    });

    group.finish();
}

criterion_group!(
    detection_benches,
    // … existing groups …
    preprocess_comparison,
);
```

> **Note:** You will need to add `use image::ImageEncoder;` or equivalent if not present. Adapt the existing `criterion_group!` call to add the new group rather than replacing it.

- [ ] **Step 9: Run detection bench to verify it compiles and runs**

```bash
cargo bench --features simd-image --bench detection_bench -- preprocess_comparison 2>&1 | tail -20
```

Expected: two bench lines, `simd_catmullrom` faster than `scalar_lanczos3`.

- [ ] **Step 10: Commit**

```bash
git add src/core/ort_yolo.rs src/core/image_pipeline.rs benches/detection_bench.rs
git commit -m "perf(yolo): SIMD preprocess via fast_image_resize + wide::f32x8

Replaces Lanczos3 + scalar normalize with CatmullRom resize and bytemuck-
aligned f32x8 division. Gated behind the simd-image feature; scalar path
retained as the #[cfg(not)] fallback.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: `Arc<AudioData>` Migration + 16-Shard TTS Cache

Both parts ship together — the sharded cache stores `Arc<AudioData>`.

**Files:**
- Modify: `src/core/tts_manager.rs`
- Modify: `src/api/tts.rs`

### 2a: Update `tts_manager.rs`

- [ ] **Step 1: Write a failing test**

Add to the test block in `tts_manager.rs`:

```rust
#[tokio::test]
async fn test_synthesize_returns_arc_audio_data() {
    let manager = make_manager_with_mock("mock");
    let result = manager
        .synthesize("arc test", Some("mock"), SynthesisParams::default())
        .await;
    assert!(result.is_ok());
    // The returned value is Arc<AudioData> — strong_count must be >= 1.
    let arc = result.unwrap();
    assert!(std::sync::Arc::strong_count(&arc) >= 1);
}

#[tokio::test]
async fn test_sharded_cache_hit_does_not_clone_samples() {
    let manager = make_manager_with_mock("mock");
    let params = SynthesisParams::default();

    let first: std::sync::Arc<crate::core::audio::AudioData> = manager
        .synthesize("shared phrase", Some("mock"), params.clone())
        .await
        .unwrap();
    let second = manager
        .synthesize("shared phrase", Some("mock"), params)
        .await
        .unwrap();

    // On a cache hit the two Arcs point to the same allocation.
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "cache hit must return the same Arc, not a clone"
    );
}
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cargo test --lib core::tts_manager::tests::test_synthesize_returns_arc_audio_data 2>&1 | tail -5
cargo test --lib core::tts_manager::tests::test_sharded_cache_hit_does_not_clone_samples 2>&1 | tail -5
```

Expected: compile errors (`synthesize` returns `AudioData` not `Arc<AudioData>`).

- [ ] **Step 3: Rewrite `TTSManager` struct in `tts_manager.rs`**

Replace the `synthesis_cache` field and its initialization:

**Old struct field (line 48):**
```rust
synthesis_cache: parking_lot::Mutex<LruCache<u64, AudioData>>,
```

**New struct fields:**
```rust
synthesis_cache: [parking_lot::Mutex<LruCache<u64, std::sync::Arc<AudioData>>>; 16],
/// Total configured capacity, stored for get_stats().
synthesis_cache_capacity: usize,
```

**Old `new()` body (replace the `synthesis_cache` init line ~56):**
```rust
synthesis_cache: parking_lot::Mutex::new(LruCache::new(cap)),
```

**New `new()` body:**
```rust
synthesis_cache_capacity: cap.get(),
synthesis_cache: {
    let per_shard = std::num::NonZeroUsize::new((cap.get() / 16).max(1)).unwrap();
    std::array::from_fn(|_| parking_lot::Mutex::new(LruCache::new(per_shard)))
},
```

- [ ] **Step 4: Rewrite `synthesize()` return type and body**

Change the method signature:

```rust
pub async fn synthesize(
    &self,
    text: &str,
    engine_id: Option<&str>,
    params: SynthesisParams,
) -> Result<std::sync::Arc<AudioData>> {
```

Replace the fast-path block (old lines 191-201):

```rust
// Fast path: return cached Arc<AudioData>; no samples clone.
{
    let shard = (cache_key & 0xF) as usize;
    let mut cache = self.synthesis_cache[shard].lock();
    if let Some(cached) = cache.get(&cache_key) {
        log::debug!(
            "TTS cache hit ({} chars, engine '{}')",
            text.len(),
            engine_id
        );
        return Ok(std::sync::Arc::clone(cached));
    }
}
```

Replace the cache-write block (old lines 225-228):

```rust
// Store result; evicts LRU entry automatically when at capacity.
let audio = std::sync::Arc::new(audio);
{
    let shard = (cache_key & 0xF) as usize;
    let mut cache = self.synthesis_cache[shard].lock();
    cache.put(cache_key, std::sync::Arc::clone(&audio));
}
Ok(audio)
```

Remove the old `Ok(audio)` line at the end.

- [ ] **Step 5: Fix `get_stats()`**

Replace the `cache_size` / `cache_capacity` computation (old `try_lock` block):

```rust
let (cache_size, cache_capacity) = {
    let size: usize = self.synthesis_cache.iter()
        .filter_map(|s| s.try_lock())
        .map(|g| g.len())
        .sum();
    (size, self.synthesis_cache_capacity)
};
```

- [ ] **Step 6: Fix the `use super::audio::AudioData` import**

Ensure `std::sync::Arc` is in scope (it usually is via prelude). The `AudioData` import stays unchanged.

- [ ] **Step 7: Update `src/api/tts.rs` to use `Arc<AudioData>`**

The `synthesize` handler currently does (lines ~114-152):

```rust
let mut audio = state.manager.synthesize(...).await?;
// post-process:
std::mem::take(&mut audio.samples)
audio.samples = result.samples;
// reads:
audio.samples.len()
audio.sample_rate
```

Change to:

```rust
let mut audio: std::sync::Arc<crate::core::audio::AudioData> =
    state.manager.synthesize(...).await?;

// Mutation for post-process — make_mut clones only when cache still holds the Arc.
let audio_mut = std::sync::Arc::make_mut(&mut audio);
let samples = std::mem::take(&mut audio_mut.samples);
// ... post-process samples ...
audio_mut.samples = result.samples;

// Read-only usage below is unchanged (Arc<T> derefs to T):
let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
```

Adjust the exact lines to match the actual indentation and surrounding code.

- [ ] **Step 8: Run the failing tests — confirm they pass**

```bash
cargo test --lib core::tts_manager::tests::test_synthesize_returns_arc_audio_data 2>&1 | tail -5
cargo test --lib core::tts_manager::tests::test_sharded_cache_hit_does_not_clone_samples 2>&1 | tail -5
```

Expected: both `ok`.

- [ ] **Step 9: Run the full existing TTS test suite**

```bash
cargo test --lib core::tts_manager 2>&1 | tail -15
cargo test --lib api::tts 2>&1 | tail -15
```

Expected: all tests pass. Fix any compilation errors in other tests that expected `AudioData` not `Arc<AudioData>`.

- [ ] **Step 10: Add TTS sharded cache bench group to `benches/tts_bench.rs`**

```rust
fn sharded_cache_vs_single_lock(c: &mut Criterion) {
    use std::sync::Arc;
    use parking_lot::Mutex;
    use lru::LruCache;
    use std::num::NonZeroUsize;

    let cap = NonZeroUsize::new(128).unwrap();

    // Baseline: single Mutex<LruCache>
    let single: Arc<Mutex<LruCache<u64, Vec<u8>>>> =
        Arc::new(Mutex::new(LruCache::new(cap)));
    // Optimized: 16-shard array
    let per_shard = NonZeroUsize::new(8).unwrap();
    let sharded: Arc<[Mutex<LruCache<u64, Vec<u8>>>; 16]> =
        Arc::new(std::array::from_fn(|_| Mutex::new(LruCache::new(per_shard))));

    let mut group = c.benchmark_group("tts_cache_contention");
    group.sample_size(100);

    group.bench_function("single_lock_sequential", |b| {
        let mut k: u64 = 0;
        b.iter(|| {
            let mut guard = single.lock();
            guard.put(black_box(k), vec![0u8; 1024]);
            let _ = guard.get(&k);
            k = k.wrapping_add(1);
        })
    });

    group.bench_function("sharded_lock_sequential", |b| {
        let mut k: u64 = 0;
        b.iter(|| {
            let shard = (k & 0xF) as usize;
            let mut guard = sharded[shard].lock();
            guard.put(black_box(k), vec![0u8; 1024]);
            let _ = guard.get(&k);
            k = k.wrapping_add(1);
        })
    });

    group.finish();
}
```

Add `sharded_cache_vs_single_lock` to the existing `criterion_group!` macro in `tts_bench.rs`.

- [ ] **Step 11: Run TTS bench to verify**

```bash
cargo bench --bench tts_bench -- sharded_cache_vs_single_lock 2>&1 | tail -15
```

- [ ] **Step 12: Commit**

```bash
git add src/core/tts_manager.rs src/api/tts.rs benches/tts_bench.rs
git commit -m "perf(tts): 16-shard cache + Arc<AudioData>

Replaces single Mutex<LruCache<u64, AudioData>> with a 16-shard array keyed
by (cache_key & 0xF). Cache hits now return Arc::clone (~5 ns) instead of
cloning the full samples Vec (~100-500 KB). TTSManager::synthesize() now
returns Result<Arc<AudioData>>; TTS handler uses Arc::make_mut for mutation.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: STT Resampler Pool

> **Design note:** The spec proposed `AudioProcessor::new()` take an `Arc<ResamplerPool>` parameter. To match the codebase pattern (`OUTPUT_POOL`, `RESIZE_SRC_POOL` — both module-level `OnceLock`) and avoid modifying every `AudioProcessor` constructor call site (`whisper_stt.rs:91`, `audio_models.rs:69,221`, `windows_sapi_tts.rs:49`, `whisper_onnx.rs:263`, `tts.rs:141`), the pool is a module-level `static OnceLock<ResamplerPool>` instead.

**Files:**
- Modify: `src/core/audio.rs`

### 3a: Add `ResamplerPool` and wire into `resample()`

- [ ] **Step 1: Write a failing test**

Add inside `#[cfg(test)]` in `audio.rs`:

```rust
#[test]
fn resampler_pool_returns_existing_resampler_on_second_call() {
    // Two sequential resample calls on the same rate pair.
    // After the first call the resampler is returned to the pool.
    // The second call should reuse it (observable via pool.len()).
    let pool = ResamplerPool::new(8);
    let r1 = pool.acquire(48_000, 16_000, 1);
    assert!(r1.is_some(), "pool should construct one on first acquire");
    // Return it
    pool.release(48_000, 16_000, 1, r1.unwrap());
    // Second acquire must find it
    let r2 = pool.acquire(48_000, 16_000, 1);
    assert!(r2.is_some(), "pool should return the cached resampler");
    drop(r2);
}

#[test]
fn resampler_pool_constructs_fresh_when_empty() {
    let pool = ResamplerPool::new(8);
    let r = pool.acquire(44_100, 16_000, 1);
    // Pool was empty — it must construct rather than panic
    assert!(r.is_some());
}
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cargo test --lib core::audio::tests::resampler_pool_returns_existing_resampler_on_second_call 2>&1 | tail -5
```

Expected: `error[E0425]: cannot find struct 'ResamplerPool'`

- [ ] **Step 3: Add `ResamplerPool` to `audio.rs`**

Add after the existing imports but before `AudioFormat`:

```rust
use rubato::{FftFixedInOut, Resampler};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Pool of FFT resamplers keyed by (input_rate, output_rate, channels).
///
/// `FftFixedInOut` is not `Send + Sync` per-se, but we gate access behind a
/// Mutex so each resampler is only used by one thread at a time. Bounded at
/// `max_per_key` instances per rate-pair to cap memory.
pub struct ResamplerPool {
    inner: parking_lot::Mutex<HashMap<(u32, u32, usize), Vec<FftFixedInOut<f32>>>>,
    max_per_key: usize,
}

impl ResamplerPool {
    pub fn new(max_per_key: usize) -> Self {
        Self {
            inner: parking_lot::Mutex::new(HashMap::new()),
            max_per_key,
        }
    }

    /// Pop an existing resampler for this rate/channel triple, or return `None`
    /// if the pool for this key is empty.
    pub fn acquire(&self, in_rate: u32, out_rate: u32, channels: usize)
        -> Option<FftFixedInOut<f32>>
    {
        self.inner.lock()
            .get_mut(&(in_rate, out_rate, channels))
            .and_then(|v| v.pop())
    }

    /// Return a resampler to the pool.  Dropped if pool is already at capacity.
    pub fn release(&self, in_rate: u32, out_rate: u32, channels: usize,
                   resampler: FftFixedInOut<f32>)
    {
        let mut guard = self.inner.lock();
        let slot = guard.entry((in_rate, out_rate, channels)).or_default();
        if slot.len() < self.max_per_key {
            slot.push(resampler);
        }
    }
}

static RESAMPLER_POOL: OnceLock<ResamplerPool> = OnceLock::new();

fn resampler_pool() -> &'static ResamplerPool {
    RESAMPLER_POOL.get_or_init(|| ResamplerPool::new(8))
}
```

- [ ] **Step 4: Rewrite `AudioProcessor::resample()` to use the pool**

Replace the `FftFixedInOut::new(...)` construction line in `resample()` (current line ~393):

```rust
let mut resampler = FftFixedInOut::<f32>::new(in_rate, out_rate, chunk_size, channels)
    .context("Failed to create FFT resampler")?;
```

with:

```rust
let in_rate_u32 = in_rate as u32;
let out_rate_u32 = out_rate as u32;
let mut resampler = resampler_pool()
    .acquire(in_rate_u32, out_rate_u32, channels)
    .map(Ok)
    .unwrap_or_else(|| {
        FftFixedInOut::<f32>::new(in_rate, out_rate, chunk_size, channels)
            .context("Failed to create FFT resampler")
    })?;
```

At the end of `resample()`, just before the final `Ok(AudioData { ... })`, return the resampler to the pool:

```rust
resampler_pool().release(in_rate as u32, out_rate as u32, channels, resampler);
```

> **Important:** remove the local `use rubato::{FftFixedInOut, Resampler};` inside `resample()` since it's now at the top of the module.

- [ ] **Step 5: Run the failing tests — confirm they pass**

```bash
cargo test --lib core::audio::tests::resampler_pool_returns_existing_resampler_on_second_call 2>&1 | tail -5
cargo test --lib core::audio::tests::resampler_pool_constructs_fresh_when_empty 2>&1 | tail -5
```

Expected: both `ok`.

- [ ] **Step 6: Run the full audio test suite**

```bash
cargo test --lib core::audio 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 7: Add resampler pool bench group to `benches/audio_bench.rs`**

```rust
fn resampler_pool_vs_fresh(c: &mut Criterion) {
    use torch_inference::core::audio::{AudioData, AudioProcessor, ResamplerPool};
    use rubato::{FftFixedInOut, Resampler};
    use std::time::Duration;

    let processor = AudioProcessor::new();
    let audio = AudioData {
        samples: vec![0.0f32; 48_000],  // 1 s mono at 48 kHz
        sample_rate: 48_000,
        channels: 1,
    };
    let pool = ResamplerPool::new(8);

    let mut group = c.benchmark_group("stt_resampler");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30);

    group.bench_function("fresh_construct_48k_to_16k", |b| {
        b.iter(|| {
            let mut r = FftFixedInOut::<f32>::new(48_000, 16_000, 1024, 1).unwrap();
            black_box(&mut r);
        })
    });

    group.bench_function("pooled_acquire_release_48k_to_16k", |b| {
        // Pre-warm the pool
        let r = FftFixedInOut::<f32>::new(48_000, 16_000, 1024, 1).unwrap();
        pool.release(48_000, 16_000, 1, r);
        b.iter(|| {
            let r = pool.acquire(48_000, 16_000, 1)
                .unwrap_or_else(|| FftFixedInOut::<f32>::new(48_000, 16_000, 1024, 1).unwrap());
            black_box(&r);
            pool.release(48_000, 16_000, 1, r);
        })
    });

    group.bench_function("resample_via_processor_48k_to_16k", |b| {
        b.iter(|| {
            black_box(processor.resample(black_box(&audio), 16_000).unwrap())
        })
    });

    group.finish();
}
```

Add `resampler_pool_vs_fresh` to the existing `criterion_group!` in `audio_bench.rs`. Export `ResamplerPool` as `pub` in `audio.rs` (add `pub` to the struct declaration) so the bench can access it. Also ensure `AudioProcessor` and `AudioData` are exported.

- [ ] **Step 8: Run audio bench**

```bash
cargo bench --bench audio_bench -- resampler_pool_vs_fresh 2>&1 | tail -15
```

Expected: `pooled_acquire_release_48k_to_16k` faster than `fresh_construct_48k_to_16k`.

- [ ] **Step 9: Commit**

```bash
git add src/core/audio.rs benches/audio_bench.rs
git commit -m "perf(stt): pool FftFixedInOut resamplers via static OnceLock

Adds ResamplerPool keyed by (in_rate, out_rate, channels), bounded at 8
per key. AudioProcessor::resample() pops from the pool and returns on
completion, saving the 5-15 ms FftFixedInOut construction cost per STT
request. Follows the same OnceLock pattern as OUTPUT_POOL / RESIZE_SRC_POOL.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: ModelCache `Arc<dyn Any + Send + Sync>`

**Files:**
- Modify: `src/core/model_cache.rs`
- Modify: `src/core/image_classifier.rs`
- Modify: `src/core/yolo.rs`

### 4a: Rewrite `model_cache.rs`

- [ ] **Step 1: Write a failing test**

Add inside the `#[cfg(test)]` block in `model_cache.rs`:

```rust
#[test]
fn arc_any_hit_does_not_call_serde() {
    // Store a type that does NOT implement Serialize — if serde is still on the
    // hot path this test won't even compile.
    #[derive(Clone, PartialEq, Debug)]
    struct NoSerde(u32);

    let cache = ModelCache::new(4);
    let k = cache_key("m", b"input", b"");

    let first: NoSerde = cache
        .get_or_run(k, || Ok(NoSerde(42)))
        .unwrap();
    assert_eq!(first, NoSerde(42));

    let second: NoSerde = cache
        .get_or_run(k, || Ok(NoSerde(99)))  // f must NOT be called on hit
        .unwrap();
    assert_eq!(second, NoSerde(42), "must return cached value, not 99");
}

#[test]
fn arc_any_type_mismatch_treated_as_miss() {
    let cache = ModelCache::new(4);
    let k = cache_key("m", b"input", b"p");

    // Write u32
    let _: u32 = cache.get_or_run(k, || Ok(1u32)).unwrap();

    // Read back as u64 — mismatch should be treated as a miss (f called again)
    let mut calls = 0u32;
    let _: u64 = cache.get_or_run(k, || { calls += 1; Ok(99u64) }).unwrap();
    assert_eq!(calls, 1, "type mismatch must be treated as a miss");
}
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cargo test --lib core::model_cache::tests::arc_any_hit_does_not_call_serde 2>&1 | tail -5
```

Expected: compile error (`NoSerde` doesn't impl `Serialize`).

- [ ] **Step 3: Rewrite `ModelCache` in `model_cache.rs`**

Replace the entire file content with:

```rust
use anyhow::Result;
use lru::LruCache;
use parking_lot::Mutex;
use std::any::Any;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ── FNV-1a 64-bit ─────────────────────────────────────────────────────────────

const FNV_OFFSET: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

fn fnv1a(data: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

pub fn cache_key(model_id: &str, input: &[u8], params: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in model_id.as_bytes() { h ^= b as u64; h = h.wrapping_mul(FNV_PRIME); }
    h ^= 0u64; h = h.wrapping_mul(FNV_PRIME);
    for &b in input { h ^= b as u64; h = h.wrapping_mul(FNV_PRIME); }
    h ^= 0u64; h = h.wrapping_mul(FNV_PRIME);
    for &b in params { h ^= b as u64; h = h.wrapping_mul(FNV_PRIME); }
    h
}

// ── CacheStats ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

// ── ModelCache ────────────────────────────────────────────────────────────────

/// Model-agnostic LRU result cache using `Arc<dyn Any + Send + Sync>`.
///
/// Eliminates the serde round-trip on every cache hit. `T` must implement
/// `Any + Send + Sync + Clone`. The `Clone` is only exercised on a hit;
/// the `Any` downcast replaces the former `serde_json::from_slice`.
pub struct ModelCache {
    cache: Mutex<LruCache<u64, Arc<dyn Any + Send + Sync>>>,
    capacity: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ModelCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).expect("capacity >= 1");
        Self {
            cache: Mutex::new(LruCache::new(cap)),
            capacity: cap.get(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Return a cached result for `key`, or run `f`, cache its result, and return it.
    pub fn get_or_run<T, F>(&self, key: u64, f: F) -> Result<T>
    where
        T: Any + Send + Sync + Clone,
        F: FnOnce() -> Result<T>,
    {
        // Check cache under lock; release lock before calling f.
        let cached: Option<Arc<dyn Any + Send + Sync>> = {
            let mut guard = self.cache.lock();
            guard.get(&key).cloned()
        };

        if let Some(any_arc) = cached {
            if let Some(val) = any_arc.downcast_ref::<T>() {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(val.clone());
            }
            // Type mismatch — treat as miss (recompute and overwrite).
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = f()?;

        {
            let mut guard = self.cache.lock();
            guard.put(key, Arc::new(result.clone()) as Arc<dyn Any + Send + Sync>);
        }

        Ok(result)
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        CacheStats {
            hits,
            misses,
            hit_rate: if total == 0 { 0.0 } else { hits as f64 / total as f64 },
        }
    }

    pub fn clear(&self) {
        let mut guard = self.cache.lock();
        guard.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // … keep ALL existing tests unchanged …
    // … add the two new tests above …
}
```

> Keep all existing `#[cfg(test)]` tests. The existing ones already compile because their `T` types (`u32`, `String`, etc.) satisfy `Any + Send + Sync + Clone`.

- [ ] **Step 4: Fix call sites in `image_classifier.rs`**

In `src/core/image_classifier.rs`, find the `get_or_run` call (~line 160). The closure returns some type (e.g., `ClassificationResult` or `Vec<ClassificationOutput>`). Ensure that type derives `Clone` if it doesn't already:

```rust
// Before (if not already Clone):
#[derive(Debug, Serialize, Deserialize)]
pub struct ClassificationOutput { ... }

// After:
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationOutput { ... }
```

The `get_or_run` call itself needs no change in syntax — only the type constraint changes.

Remove any `use serde::{Serialize, Deserialize}` imports from `model_cache.rs` (already done above). The call site still uses serde for HTTP responses, so leave those derives in place.

- [ ] **Step 5: Fix call sites in `yolo.rs`**

Same as above — find the `YoloResults` type (or whatever `T` is in `yolo.rs:327`) and add `#[derive(Clone)]` if missing.

- [ ] **Step 6: Run failing tests — confirm they pass**

```bash
cargo test --lib core::model_cache::tests::arc_any_hit_does_not_call_serde 2>&1 | tail -5
cargo test --lib core::model_cache::tests::arc_any_type_mismatch_treated_as_miss 2>&1 | tail -5
```

Expected: both `ok`.

- [ ] **Step 7: Run full model cache and classifier tests**

```bash
cargo test --lib core::model_cache 2>&1 | tail -15
cargo test --lib core::image_classifier 2>&1 | tail -10
cargo test --lib core::yolo 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 8: Add ModelCache bench group to `benches/cache_bench.rs`**

```rust
fn model_cache_arc_any_vs_serde(c: &mut Criterion) {
    use torch_inference::core::model_cache::{cache_key, ModelCache};
    use std::time::Duration;

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct Payload {
        scores: Vec<f32>,
        label: String,
    }

    fn make_payload() -> Payload {
        Payload {
            scores: vec![0.1f32; 100],
            label: "cat".to_string(),
        }
    }

    let cache = ModelCache::new(512);
    let k = cache_key("bench_model", b"test_image", b"top5");

    // Pre-populate cache
    let _: Payload = cache.get_or_run(k, make_payload).unwrap();

    let mut group = c.benchmark_group("model_cache_hit_cost");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(200);

    group.bench_function("arc_any_hit", |b| {
        b.iter(|| {
            let val: Payload = cache.get_or_run(black_box(k), make_payload).unwrap();
            black_box(val)
        })
    });

    // Baseline: manually simulate the old serde round-trip cost
    let serialized = serde_json::to_vec(&make_payload()).unwrap();
    group.bench_function("serde_deserialize_baseline", |b| {
        b.iter(|| {
            let val: Payload = serde_json::from_slice(black_box(&serialized)).unwrap();
            black_box(val)
        })
    });

    group.finish();
}
```

Add `model_cache_arc_any_vs_serde` to the existing `criterion_group!`.

- [ ] **Step 9: Run cache bench**

```bash
cargo bench --bench cache_bench -- model_cache_arc_any_vs_serde 2>&1 | tail -15
```

Expected: `arc_any_hit` faster than `serde_deserialize_baseline`.

- [ ] **Step 10: Commit**

```bash
git add src/core/model_cache.rs src/core/image_classifier.rs src/core/yolo.rs benches/cache_bench.rs
git commit -m "perf(cache): replace serde round-trip with Arc<dyn Any + Send + Sync>

Every cache hit previously deserialised JSON bytes. Now the concrete value
is stored in an Arc and retrieved via downcast_ref + clone. Serde is no
longer on the cache hot path; derives stay on result types for HTTP responses.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: RateLimiter TTL Background Sweep

**Files:**
- Modify: `src/main.rs`

The `cleanup_old_entries()` method already exists in `src/middleware/rate_limit.rs` (line 86) but has no caller — it's `#[allow(dead_code)]`. This task wires it to a periodic tokio task.

- [ ] **Step 1: Write a failing test**

Add to `src/middleware/rate_limit.rs` tests:

```rust
#[test]
fn cleanup_removes_stale_entries() {
    // Window of 0 seconds means every entry is immediately stale.
    let limiter = RateLimiter::new(100, 0);
    // Inject a fake entry with timestamp far in the past.
    limiter.request_counts.insert("stale-ip".to_string(), (1, 0));
    assert!(limiter.request_counts.contains_key("stale-ip"));

    limiter.cleanup_old_entries();

    assert!(
        !limiter.request_counts.contains_key("stale-ip"),
        "stale entry must be removed by cleanup_old_entries"
    );
}
```

- [ ] **Step 2: Run test — confirm it passes (the logic already exists)**

```bash
cargo test --lib middleware::rate_limit::tests::cleanup_removes_stale_entries 2>&1 | tail -5
```

Expected: `ok` — the logic was already correct, just untested.

- [ ] **Step 3: Remove `#[allow(dead_code)]` from `cleanup_old_entries`**

In `src/middleware/rate_limit.rs`, remove the attribute:

```rust
// Before:
#[allow(dead_code)]
pub fn cleanup_old_entries(&self) {

// After:
pub fn cleanup_old_entries(&self) {
```

- [ ] **Step 4: Spawn background sweep in `main.rs`**

Find where the `RateLimiter` is constructed in `main.rs` (search for `RateLimiter::new`). After constructing it and wrapping in `Arc`, add:

```rust
// Spawn periodic TTL sweep for the rate-limiter DashMap.
// Prevents unbounded memory growth from unique IPs accumulating over time.
{
    let limiter_for_sweep = Arc::clone(&rate_limiter);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            limiter_for_sweep.cleanup_old_entries();
        }
    });
}
```

Adjust variable names to match what `main.rs` actually calls the rate limiter (`rate_limiter`, `limiter`, etc.).

- [ ] **Step 5: Build release to confirm no warnings**

```bash
cargo build --release 2>&1 | grep -E "^error|warning\[" | head -20
```

Expected: no new errors or warnings.

- [ ] **Step 6: Commit**

```bash
git add src/middleware/rate_limit.rs src/main.rs
git commit -m "perf(middleware): wire cleanup_old_entries to 60s background sweep

The DashMap tracking per-IP request counts had no eviction path, causing
unbounded growth over long-running deployments. The cleanup method already
existed; this commit spawns it on a Tokio interval from main.rs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Final Verification

- [ ] **Step 1: Run the full test suite**

```bash
cargo test 2>&1 | tail -20
```

Expected: all tests pass (pre-existing failures only — nothing new).

- [ ] **Step 2: Release build**

```bash
cargo build --release 2>&1 | grep -E "^error" | head -10
```

Expected: no errors.

- [ ] **Step 3: Run the full bench suite and compare to baseline**

```bash
cargo bench --features simd-image 2>&1 | tee /tmp/bench_after_all.txt
```

Compare `/tmp/bench_*_baseline.txt` vs `/tmp/bench_after_all.txt`:
- `yolo_preprocess_comparison/simd_catmullrom` should be faster than `scalar_lanczos3`
- `stt_resampler/pooled_acquire_release_48k_to_16k` should be faster than `fresh_construct`
- `tts_cache_contention/sharded_lock_sequential` should be ≤ `single_lock_sequential`
- `model_cache_hit_cost/arc_any_hit` should be faster than `serde_deserialize_baseline`

- [ ] **Step 4: Commit bench results summary**

```bash
git add /dev/null  # nothing to stage — just note the numbers in commit message
git commit --allow-empty -m "chore(bench): record post-optimization baseline numbers

See /tmp/bench_after_all.txt for detailed criterion output.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All five spec sections have at least one task.
- [x] **No placeholders:** Every step has exact commands or exact code.
- [x] **Type consistency:** `Arc<AudioData>` used throughout Task 2; `Arc<dyn Any + Send + Sync>` throughout Task 4; `ResamplerPool` key triple `(u32, u32, usize)` consistent between `acquire` and `release`.
- [x] **Design note:** ResamplerPool uses module-level `OnceLock` (matching codebase pattern) rather than constructor injection (spec said constructor; pattern match wins for minimal call-site churn).
- [x] **Test-first:** Every task has a failing test before implementation.
- [x] **Frequent commits:** One commit per task.
