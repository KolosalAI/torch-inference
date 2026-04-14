# Comprehensive Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply four targeted optimization passes — build flag alignment, ONNX session tuning, Clippy clean, and hot-path pool wiring — without changing any public API or behaviour.

**Architecture:** Each task is independent and can be executed in any order. Tasks 1–3 are mechanical; Task 4 wires existing pool infrastructure into `OrtClassifier`'s inference hot path using a module-level `OnceLock<TensorPool>`.

**Tech Stack:** Rust 2021, actix-web 4, ort 2.0.0-rc.10, num_cpus, parking_lot, dashmap

---

## File Map

| File | Task | Change |
|------|------|--------|
| `Cargo.toml` | 1 | Add `production` feature alias |
| `Makefile` | 1 | Align `build` and `run` to `--features production` |
| `src/core/kokoro_onnx.rs` | 2 | `with_intra_threads(2)` → physical CPUs, add `with_inter_threads(1)` + `with_memory_pattern(true)` |
| `build.rs` | 3 | Fix 4 clippy warnings |
| `src/core/neural_network.rs` | 3 | Remove unneeded `()` expressions |
| `src/models/pytorch_loader.rs` | 3 | Remove unneeded `()` expressions |
| `src/api/image.rs` | 3 | Replace redundant closures |
| `src/api/classify.rs` | 3 | Replace manual clamp patterns |
| `src/api/tts.rs` | 3 | Replace manual clamp patterns |
| `src/api/ws_infer.rs` | 3 | Replace manual clamp patterns + `.to_vec()` |
| `src/api/yolo.rs` | 3 | Replace manual clamp patterns + `.to_vec()` |
| `src/api/models.rs` | 3 | Fix `map_or`/`is_some_and`, `length comparison`, `redundant_pattern_matching` |
| `src/api/model_download.rs` | 3 | Fix `vec_init_then_push` |
| `src/api/performance.rs` | 3 | Fix `blocks_in_conditions` |
| `src/compression.rs` | 3 | Fix `io_other_error` |
| `src/core/audio.rs` | 3 | Fix `.to_vec()` |
| `src/core/engine.rs` | 3 | Fix redundant closures, `needless_borrows` |
| `src/core/kokoro_onnx.rs` | 3 | Fix 1 redundant closure |
| `src/core/neural_network.rs` | 3 | Fix `needless_range_loop` |
| `src/core/ort_classify.rs` | 3 | Fix redundant closure, `redundant_pattern_matching` |
| `src/core/ort_yolo.rs` | 3 | Fix `.to_vec()`, identity_op/erasing_op |
| `src/core/torch_autodetect.rs` | 3 | Fix `needless_borrows`, `unneeded unit return`, `new_without_default`, `manual_strip`, `double_ended_iterator_last` |
| `src/middleware/correlation_id.rs` | 3 | Fix `redundant_closure`, `unwrap_or_default`, `or_default`, `needless_lifetimes`, `needless_return`, `unneeded unit return` |
| `src/model_pool.rs` | 3 | Fix `manual_div_ceil` |
| `src/models/manager.rs` | 3 | Fix `map_or`/`is_some_and` |
| `src/security/validation.rs` | 3 | Fix `manual_range_contains` |
| `src/api/models.rs` | 3 | Fix `useless_vec` → array literal |
| `src/core/ort_classify.rs` | 4 | Add `OnceLock<TensorPool>`, wire `acquire`/`release` for output buffer |

---

## Task 1: Align Build Feature Flags

**Files:**
- Modify: `Cargo.toml`
- Modify: `Makefile`

- [ ] **Step 1: Add `production` feature to `Cargo.toml`**

Open `Cargo.toml`. In the `[features]` section, add one line after `default`:

```toml
[features]
default = ["jemalloc"]
production = ["jemalloc", "metrics", "telemetry", "simd-image"]
# ... rest of features unchanged
```

- [ ] **Step 2: Update `make build` in `Makefile`**

Find the `build:` target (currently `$(CARGO) build --release --no-default-features`). Replace with:

```makefile
build: ## Build release binary (recommended)
	@echo "Building release binary..."
	$(CARGO) build --release --no-default-features --features production
	@echo ""
	@echo "✅ Build complete: ./target/release/torch-inference-server"
```

- [ ] **Step 3: Update `make run` in `Makefile`**

Find the `run:` target (currently `$(CARGO) run --release --features "metrics,telemetry,simd-image"`). Replace with:

```makefile
run: ## Run server in release mode
	@echo "Starting server (release mode)..."
	$(CARGO) run --release --no-default-features --features production
```

- [ ] **Step 4: Verify the feature compiles**

```bash
cargo check --no-default-features --features production
```

Expected: `Finished` with no errors.

- [ ] **Step 5: Verify default test target still works**

```bash
cargo check --no-default-features
```

Expected: `Finished` with no errors.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Makefile
git commit -m "build: add production feature alias, align make build/run flags"
```

---

## Task 2: ONNX Session Tuning in `kokoro_onnx.rs`

**Files:**
- Modify: `src/core/kokoro_onnx.rs:185-199`

The session builder loop (lines 184–211) currently hardcodes `with_intra_threads(2)` and is missing `with_inter_threads(1)` and `with_memory_pattern(true)`. The CoreML EP is already registered on macOS — we just need the thread and memory settings.

- [ ] **Step 1: Add `num_cpus` import at top of file**

The `num_cpus` crate is already in `Cargo.toml`. Add the use at the top of the session-building block. Find this in `kokoro_onnx.rs`:

```rust
        // Build `pool_size` independent sessions with hardware-accelerated EP.
        let mut sessions = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(2)?;
```

Replace with:

```rust
        // Build `pool_size` independent sessions with hardware-accelerated EP.
        let physical_cpus = num_cpus::get_physical().max(1);
        let mut sessions = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let builder = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(physical_cpus)?
                .with_inter_threads(1)?
                .with_memory_pattern(true)?;
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo check --no-default-features
```

Expected: `Finished` with no errors. If `with_memory_pattern` or `with_inter_threads` does not exist on the `ort 2.0.0-rc.10` builder, check the ort docs:

```bash
cargo doc --package ort --no-deps --open 2>/dev/null || true
grep -r "fn with_inter\|fn with_memory\|fn with_intra" ~/.cargo/registry/src/*/ort-*/src/ 2>/dev/null | head -10
```

If a method is absent, omit that specific call and note it in the commit message.

- [ ] **Step 3: Run tests**

```bash
cargo test --no-default-features 2>&1 | tail -5
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/core/kokoro_onnx.rs
git commit -m "perf(onnx): physical CPU intra-threads, inter=1, memory_pattern=true"
```

---

## Task 3: Clippy Clean (50 warnings → 0)

**Files:** See file map above. Work file-by-file. After every 3–4 files, re-run clippy to check progress.

- [ ] **Step 1: Fix `build.rs` (4 warnings)**

Open `build.rs`. Apply these four changes:

1. Line 156 — `cmp_owned`: `cache_dir == PathBuf::from(".")` → `cache_dir == "."`

2. Line 483 — `needless_borrows_for_generic_args`: `.args(&["-c", "import torch; print(torch.__path__[0])"])` → `.args(["-c", "import torch; print(torch.__path__[0])"])`

3. Line 499 — `ptr_arg`: `fn validate_libtorch(path: &PathBuf) -> bool` → `fn validate_libtorch(path: &Path) -> bool`

4. Line 524 — `ptr_arg`: `fn setup_libtorch_paths(libtorch_path: &PathBuf, system_info: &SystemInfo)` → `fn setup_libtorch_paths(libtorch_path: &Path, system_info: &SystemInfo)`

- [ ] **Step 2: Fix `src/core/neural_network.rs` and `src/models/pytorch_loader.rs` (2 warnings)**

In `src/core/neural_network.rs` line 235: remove the standalone `()` expression.

```rust
// Before
fn some_fn() -> () {
    // ...
    ()
}

// After — remove the trailing () and the -> ()
fn some_fn() {
    // ...
}
```

Same pattern in `src/models/pytorch_loader.rs` line 32.

- [ ] **Step 3: Fix `src/api/image.rs` (2 warnings — redundant closures)**

Line 80: `.map_err(|e| ApiError::BadRequest(e))?` → `.map_err(ApiError::BadRequest)?`

Line 147: same pattern → `.map_err(ApiError::BadRequest)?`

- [ ] **Step 4: Fix clamp patterns in `src/api/tts.rs` (4 warnings)**

Lines 108–109 and 204–205. Pattern to find:

```rust
if req.speed < 0.25 { 0.25 } else if req.speed > 4.0 { 4.0 } else { req.speed }
```

Replace with:

```rust
req.speed.clamp(0.25, 4.0)
```

Same for `req.pitch.clamp(0.5, 2.0)` (2 occurrences each).

- [ ] **Step 5: Fix clamp patterns in `src/api/ws_infer.rs` (3 clamp + 3 `.to_vec()`)**

Lines 358–360: replace three manual clamp patterns:
- `req.top_k` → `req.top_k.clamp(1, 1000)`
- `req.model_width` → `req.model_width.clamp(1, 4096)`
- `req.model_height` → `req.model_height.clamp(1, 4096)`

Line 268 area — find `.iter().copied().collect::<Vec<_>>()` patterns and replace with `.to_vec()`.

- [ ] **Step 6: Fix `src/api/yolo.rs` (3 clamp + 3 `.to_vec()`)**

Lines 281–282 area: replace manual clamp for `top_k`, `width`, `height` with `.clamp()`.

Lines with `.iter().copied().collect()` → `.to_vec()`.

- [ ] **Step 7: Fix `src/api/classify.rs` (3 clamp)**

Lines 215–217: replace `top_k`, `model_width`, `model_height` manual clamp patterns with `.clamp(1, 1000)`, `.clamp(1, 4096)`, `.clamp(1, 4096)` respectively.

- [ ] **Step 8: Fix `src/api/models.rs` (map_or, length comparison, redundant pattern, useless_vec)**

1. `map_or` → `is_some_and`: find `.map_or(false, |x| ...)` → `.is_some_and(|x| ...)`
2. `.len() == 0` → `.is_empty()`
3. Redundant pattern matching: `if let Ok(_) = ...` → `if ....is_ok()`
4. `vec!["v5", "v8", "v10", "v11", "v12"]` → `["v5", "v8", "v10", "v11", "v12"]` (and same for the `["n", "s", "m", "l", "x"]` literal)

- [ ] **Step 9: Fix misc single-occurrence warnings in remaining files**

Run `cargo clippy --no-default-features 2>&1 | grep "^\s*-->"` to get current locations. Fix each remaining warning:

- `src/api/model_download.rs:267` — `vec_init_then_push`: replace `let mut v = Vec::new(); v.push(a); v.push(b);` with `let v = vec![a, b];`
- `src/api/performance.rs:223` — `blocks_in_conditions`: extract the complex block into a `let` binding before the `if`
- `src/compression.rs:28` — `io_other_error`: `std::io::Error::new(std::io::ErrorKind::Other, e)` → `std::io::Error::other(e)`
- `src/core/audio.rs:374-375` — `.iter().copied().collect()` → `.to_vec()`
- `src/core/engine.rs:86,89,134` — redundant closures and `needless_borrows`: apply clippy suggestions inline
- `src/core/kokoro_onnx.rs:449` — redundant closure: apply suggestion
- `src/core/neural_network.rs:234` — `needless_range_loop`: `for f in 0..out_channels.len()` → `for channel in &out_channels` (adjust body accordingly)
- `src/core/ort_classify.rs:174` — redundant closure: apply suggestion
- `src/core/ort_yolo.rs:116,142-143,214-215` — `.to_vec()` + identity_op (`NUM_ANCHORS * 1` → `NUM_ANCHORS`, `h * 1` → `h`)
- `src/core/torch_autodetect.rs:194-195` — `needless_borrows`; `:373` — `should_implement_trait` (rename `from_str` to `parse_str` to avoid confusion); also fix `manual_strip`, `double_ended_iterator_last` (.last() → .next_back()), `new_without_default` (add `impl Default for TorchLibAutoDetect { fn default() -> Self { Self::new() } }`)
- `src/middleware/correlation_id.rs` — fix `redundant_closure` (×2), `unwrap_or_default` (×2), `or_default` (×1), `needless_lifetimes`, `needless_return`, `unneeded unit return`
- `src/model_pool.rs:152` — `manual_div_ceil`: `(max_entries + NUM_SHARDS - 1) / NUM_SHARDS` → `max_entries.div_ceil(NUM_SHARDS)`
- `src/models/manager.rs:378` — `map_or` → `is_some_and`
- `src/security/validation.rs:167` — `manual_range_contains`: `p < -10 || p > 10` → `!(-10..=10).contains(&p)`

- [ ] **Step 10: Verify zero warnings**

```bash
cargo clippy --no-default-features 2>&1 | grep "^warning:" | grep -v "torch_inference@"
```

Expected: no output (zero warnings).

- [ ] **Step 11: Run tests to confirm no regressions**

```bash
cargo test --no-default-features 2>&1 | tail -5
```

Expected: all tests pass.

- [ ] **Step 12: Commit**

```bash
git add -u
git commit -m "chore: fix all 50 clippy warnings (clamp, redundant closures, to_vec, misc)"
```

---

## Task 4: Wire TensorPool into ORT Classification Hot Path

**Files:**
- Modify: `src/core/ort_classify.rs`

**Context:** `TensorPool` reuses `Vec<f32>` buffers across requests. It exists in `src/tensor_pool.rs` and is created in `main.rs`, but `.acquire()`/`.release()` are never called anywhere. The hot path in `OrtClassifier::classify_batch()` allocates a fresh `Vec<f32>` for the ORT output on every image (`raw` at line 174). We wire the pool here using a module-level `OnceLock<TensorPool>` — no API changes needed.

Note: The input tensor `img_nchw` (line 152) cannot be pooled — `Tensor::from_array()` takes ownership and ORT does not return the buffer.

**Spec deviation:** The spec named `onnx_loader.rs` and `image_pipeline.rs` as the pool wiring targets. During planning, we found: (a) `OnnxLoader::infer()` is a placeholder stub — no real inference runs through it; (b) the `Cursor::new(Vec::new())` in `image_pipeline.rs` only appears in test helpers, not the production hot path. The actual hot-path `Vec<f32>` allocation is in `OrtClassifier::classify_batch()` in `ort_classify.rs`. Plan targets this file instead.

- [ ] **Step 1: Read the full `classify_batch` method**

Open `src/core/ort_classify.rs` and read lines 139–200. Confirm:
- Line 174: `let raw: Vec<f32> = raw_view.iter().copied().collect();`
- `raw` is used for `softmax` / `top_k` and then dropped at end of loop iteration
- The output size = number of model classes (e.g., 1000 for ImageNet). This is the buffer we pool.

- [ ] **Step 2: Add `TensorPool` import and module-level static**

At the top of `src/core/ort_classify.rs`, add:

```rust
use std::sync::OnceLock;
use crate::tensor_pool::{TensorPool, TensorShape};

/// Module-level output buffer pool.  Initialized once on first classify call.
/// Pools the per-image output Vec<f32> to avoid per-request heap allocation.
static OUTPUT_POOL: OnceLock<TensorPool> = OnceLock::new();

fn output_pool() -> &'static TensorPool {
    OUTPUT_POOL.get_or_init(|| TensorPool::new(64))
}
```

- [ ] **Step 3: Write a failing test**

At the bottom of `src/core/ort_classify.rs`, inside the existing `#[cfg(test)]` block (or add one), add:

```rust
#[cfg(test)]
mod pool_tests {
    use super::*;
    use crate::tensor_pool::TensorShape;

    #[test]
    fn output_pool_reuses_buffer() {
        let pool = output_pool();
        let shape = TensorShape::new(vec![1000]);

        // Simulate acquire → fill → release cycle twice
        let buf = pool.acquire(shape.clone());
        assert_eq!(buf.len(), 1000);
        pool.release(shape.clone(), buf);

        let stats_before = pool.get_stats();
        let _buf2 = pool.acquire(shape.clone());
        let stats_after = pool.get_stats();

        // Second acquire must come from pool (reuse, not fresh allocation)
        assert!(
            stats_after.reuses > stats_before.reuses,
            "expected pool reuse, got stats: {:?}",
            stats_after
        );
    }
}
```

- [ ] **Step 4: Run the test — expect it to pass (pool logic is already implemented)**

```bash
cargo test --no-default-features -p torch_inference output_pool_reuses_buffer -- --nocapture
```

Expected: PASS. (The test validates `TensorPool` behaviour, not yet the wiring.)

- [ ] **Step 5: Wire pool into `classify_batch`**

In `classify_batch`, find:

```rust
            let (_shape, raw_view) = outputs[0].try_extract_tensor::<f32>()?;
            let raw: Vec<f32> = raw_view.iter().copied().collect();
```

Replace with:

```rust
            let (_shape, raw_view) = outputs[0].try_extract_tensor::<f32>()?;
            let output_len = raw_view.len();
            let output_shape = TensorShape::new(vec![output_len]);
            let mut raw = output_pool().acquire(output_shape.clone());
            raw.clear();
            raw.extend(raw_view.iter().copied());
```

And after `results.push(preds);` at the end of the loop body, add:

```rust
            output_pool().release(output_shape, raw);
```

Both `softmax` and `top_k` take `&[f32]`, so `raw_buf` can be released back to the pool before computing `probs`. Full replacement:

```rust
            let (_shape, raw_view) = outputs[0].try_extract_tensor::<f32>()?;
            let output_len = raw_view.len();
            let output_shape = TensorShape::new(vec![output_len]);
            let mut raw_buf = output_pool().acquire(output_shape.clone());
            raw_buf.clear();
            raw_buf.extend(raw_view.iter().copied());

            // softmax and top_k both take &[f32] — no clone needed
            let probs: Vec<f32> = if self.output_is_prob {
                raw_buf.iter().copied().collect()
            } else {
                Self::softmax(&raw_buf)
            };

            // Return the raw output buffer to the pool before using probs
            output_pool().release(output_shape, raw_buf);

            let top = Self::top_k(&probs, top_k);
```

- [ ] **Step 6: Verify it compiles**

```bash
cargo check --no-default-features
```

Expected: no errors.

- [ ] **Step 7: Run all tests**

```bash
cargo test --no-default-features 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 8: Confirm pool reuse rate is > 0 at runtime (optional smoke check)**

If you have a running server with classification endpoint:

```bash
curl -s http://localhost:8000/metrics | grep tensor_pool
```

Or add a temporary `log::info!` after the loop:

```rust
if i == 0 {
    let stats = output_pool().get_stats();
    log::debug!("classify output_pool: reuse_rate={:.1}%", stats.reuse_rate);
}
```

- [ ] **Step 9: Commit**

```bash
git add src/core/ort_classify.rs
git commit -m "perf: wire TensorPool into OrtClassifier output buffer, eliminating per-request Vec<f32> alloc"
```

---

## Verification

After all four tasks:

- [ ] `cargo check --no-default-features --features production` → no errors
- [ ] `cargo clippy --no-default-features 2>&1 | grep "^warning:" | grep -v "torch_inference@"` → no output
- [ ] `cargo test --no-default-features 2>&1 | tail -3` → all tests pass
- [ ] `make build` compiles with production features (jemalloc + metrics + telemetry + simd-image)
