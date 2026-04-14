# Comprehensive Optimization Design

**Date:** 2026-04-14  
**Status:** Approved  
**Scope:** Build configuration, ONNX runtime tuning, Clippy clean, memory hot-path wiring

---

## 1. Build Configuration

### Problem
The `Makefile` has inconsistent feature flags across targets:
- `make build` uses `--no-default-features` — drops jemalloc, metrics, telemetry, simd-image
- `make run` uses `--features "metrics,telemetry,simd-image"` — misses explicit jemalloc
- No canonical "production" feature set defined anywhere

### Solution
Add a `production` feature alias in `Cargo.toml`:

```toml
[features]
production = ["jemalloc", "metrics", "telemetry", "simd-image"]
```

Update `Makefile`:
- `make build` → `cargo build --release --features production`
- `make run` → `cargo run --release --features production`
- `make test` stays `--no-default-features` (fast, no optional deps)
- `make clippy` stays `--no-default-features` (clean baseline)

### Files
- `Cargo.toml` — add `production` feature
- `Makefile` — update `build` and `run` targets

---

## 2. ONNX Runtime Tuning

### Problem
`KokoroOnnxEngine` builds ONNX sessions without configuring:
- Thread counts (ORT defaults to 1 intra-op thread)
- CoreML Execution Provider (compiled in via `ort` crate feature but not registered at session build time)
- Memory arena / pattern (ORT allocates per-run by default)

### Solution
In `src/core/kokoro_onnx.rs`, update the session builder (wherever `Session::builder()` is called):

```rust
use ort::execution_providers::CoreMLExecutionProvider;

Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(num_cpus::get_physical())?
    .with_inter_threads(1)?           // pool handles concurrency; cross-session parallelism hurts
    .with_memory_pattern(true)?       // ORT internal arena, reduces per-run allocs
    .with_execution_providers([CoreMLExecutionProvider::default().build()])?
    .commit_from_file(&model_path)?
```

### Constraints
- Changes scoped to `KokoroOnnxEngine::build_session()` only
- `num_cpus::get_physical()` already in `Cargo.toml` as `num_cpus = "1.16"`
- `CoreMLExecutionProvider` falls back silently to CPU if CoreML is unavailable
- **Implementer must verify exact method names against `ort = "2.0.0-rc.10"` API** — RC method names differ from stable (e.g. `with_intra_threads` may be `with_intra_op_num_threads`; `with_memory_pattern` may not exist). Check `ort` crate docs or source before writing the call chain.

### Files
- `src/core/kokoro_onnx.rs`

---

## 3. Clippy Clean

### Problem
`cargo clippy --no-default-features` emits 50 warnings. Zero warnings is the target.

### Warning Inventory

| Count | Category | Mechanical Fix |
|-------|----------|---------------|
| 10 | `clippy::manual_clamp` | Replace `if x < min { min } else if x > max { max }` with `x.clamp(min, max)` |
| 6 | `clippy::redundant_closure` | Replace `\|e\| Foo(e)` with `Foo` |
| 3 | `clippy::iter_copied_collect` | Replace `.iter().copied().collect()` with `.to_vec()` |
| 2 | `clippy::useless_vec` | Replace `vec![a, b]` with `[a, b]` where slice suffices |
| 2 | `clippy::unused_unit` | Remove trailing `()` in unit-returning functions |
| 2 | `clippy::needless_return` | Remove explicit `return` at function end |
| 2 | `clippy::ptr_arg` | `&PathBuf` → `&Path` in `build.rs` |
| 2 | `clippy::needless_borrows_for_generic_args` | Remove `&[...]` wrapper in `.args()` call |
| ~19 | misc single-occurrence | Fix inline per clippy suggestion |

### Approach
- Work file-by-file, apply all suggestions mechanically
- No logic changes — purely surface-level
- Goal: `cargo clippy --no-default-features` → zero warnings (excluding `torch_inference@` build script output lines)

### Files
- `build.rs`
- `src/core/neural_network.rs`
- `src/models/pytorch_loader.rs`
- `src/api/image.rs`
- Additional files as clippy output directs

---

## 4. Memory / Hot-Path Pool Wiring

### Problem
`TensorPool` and `BufferPool` are fully implemented in `src/tensor_pool.rs` with:
- `TensorPool` — reuses `Vec<f32>` across ONNX inference calls
- `BufferPool` — reuses `Vec<u8>` scratch buffers for image preprocessing

Both are instantiated in `main.rs` and stored in `ModelManager` / `OnnxLoader` — but **`.acquire()` and `.release()` are never called**. The pool is dead infrastructure. Every inference request heap-allocates fresh tensors.

### Solution

**`src/models/onnx_loader.rs`** — wire `TensorPool` into the input tensor preparation step:
- Before constructing the input `Vec<f32>`, call `tensor_pool.acquire(shape)`
- After `session.run()` completes, call `tensor_pool.release(shape, tensor)`
- Guard with `if let Some(pool) = &self.tensor_pool` — pool is optional per config

**`src/core/image_pipeline.rs`** — wire `BufferPool` into the two `std::io::Cursor::new(Vec::new())` sites:
- Acquire a buffer from the pool before the cursor is created
- After the cursor is consumed, return the inner buffer to the pool
- `BufferPool` must be threaded into `ImagePipeline` (add as a field or use a process-global `OnceLock<BufferPool>`)

### Constraints
- `enable_tensor_pooling` config flag already gates pool creation in `main.rs` — respect this
- Pool wiring must not change return types or public API
- `BufferPool` approach: prefer a `OnceLock<BufferPool>` module-level static in `image_pipeline.rs` to avoid threading it through every call site

### Files
- `src/models/onnx_loader.rs`
- `src/core/image_pipeline.rs`
- `src/tensor_pool.rs` (if `BufferPool` needs a global accessor added)

---

## Success Criteria

| Area | Before | After |
|------|--------|-------|
| Build flags | Inconsistent across targets | `production` feature used consistently |
| ONNX threads | ORT default (1 intra) | `physical_cpus` intra, 1 inter, CoreML registered |
| Clippy | 50 warnings | 0 warnings |
| TensorPool reuse rate | 0% (never called) | >0% on classification/detection requests |
| BufferPool reuse rate | 0% (never called) | >0% on image pipeline requests |

---

## Out of Scope

- Benchmark-driven profiling (Option C — follow-up after this lands)
- STT microservice (`services/stt/`) optimization
- LLM microservice (`services/llm/`) optimization
- Playground HTML changes
- New features of any kind
