# Inference Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a result LRU cache, FP16 half-precision, and QuantizationConfig to ImageClassifier, YOLO, and NeuralNetwork so they match the optimization level of TTS.

**Architecture:** A new `ModelCache` in `src/core/model_cache.rs` provides a model-agnostic LRU cache keyed by FNV-1a hash of (model_id, input_bytes, params_bytes). `OptimizedTorchModel` in `src/torch_optimization.rs` gains `enable_fp16` and `QuantizationConfig`. Each of the three model files replaces its raw `CModule` with `OptimizedTorchModel` and wraps every inference call with `ModelCache::get_or_run`.

**Tech Stack:** Rust, `tch` (libtorch), `lru = "0.12"`, `serde_json = "1.0"`, `std::sync::Mutex`, `std::sync::atomic::AtomicU64`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/torch_optimization.rs` | Modify | Add `enable_fp16`, `QuantizationConfig`, FP16 tensor conversion in `infer()`, builder methods |
| `src/core/model_cache.rs` | Create | `ModelCache`, `CacheStats`, `cache_key()` — model-agnostic LRU result cache |
| `src/core/mod.rs` | Modify | Register `pub mod model_cache;` |
| `src/core/image_classifier.rs` | Modify | Replace `CModule` with `OptimizedTorchModel`, add `ModelCache`, update `new()` + inference calls |
| `src/core/yolo.rs` | Modify | Same pattern as ImageClassifier |
| `src/core/neural_network.rs` | Modify | Same pattern; remove old `quantization` stub (moved to torch_optimization.rs) |

---

## Task 1: Extend `OptimizedTorchModel` with FP16 and QuantizationConfig

**Files:**
- Modify: `src/torch_optimization.rs`

- [ ] **Step 1: Write failing tests for new builder methods and FP16 config**

Add to the `#[cfg(test)] mod tests` block at the bottom of `src/torch_optimization.rs`:

```rust
#[test]
fn test_config_fp16_builder() {
    let config = TorchOptimizationConfigBuilder::new()
        .fp16(true)
        .build();
    assert!(config.enable_fp16);
}

#[test]
fn test_config_fp16_default_is_false() {
    let config = TorchOptimizationConfig::default();
    assert!(!config.enable_fp16);
}

#[test]
fn test_config_quantization_builder() {
    use QuantizationDtype::*;
    use QuantizationMode::*;
    let qcfg = QuantizationConfig { dtype: F16, mode: Dynamic };
    let config = TorchOptimizationConfigBuilder::new()
        .quantization(Some(qcfg))
        .build();
    assert!(config.quantization.is_some());
    assert!(matches!(config.quantization.unwrap().dtype, F16));
}

#[test]
fn test_quantization_dtype_variants() {
    use QuantizationDtype::*;
    let _ = format!("{:?}", F32);
    let _ = format!("{:?}", F16);
    let _ = format!("{:?}", Int8);
}

#[test]
fn test_quantization_mode_variants() {
    use QuantizationMode::*;
    let _ = format!("{:?}", Dynamic);
    let _ = format!("{:?}", Static);
}

#[test]
fn test_quantization_config_clone() {
    use QuantizationDtype::*;
    use QuantizationMode::*;
    let qcfg = QuantizationConfig { dtype: Int8, mode: Static };
    let cloned = qcfg.clone();
    assert!(matches!(cloned.dtype, Int8));
    assert!(matches!(cloned.mode, Static));
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test -p torch-inference torch_optimization -- --nocapture 2>&1 | grep -E "FAILED|error\[|error:"
```

Expected: compile errors — `enable_fp16`, `QuantizationDtype`, `QuantizationMode`, `QuantizationConfig`, `fp16`, `quantization` not defined.

- [ ] **Step 3: Add QuantizationDtype, QuantizationMode, QuantizationConfig to torch_optimization.rs**

Add after the `use` imports at the top of `src/torch_optimization.rs`, before `TorchOptimizationConfig`:

```rust
/// Precision target for model weights/activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationDtype {
    F32,
    F16,
    Int8,
}

/// When quantization was applied.
/// `Dynamic` = apply FP16 at inference time (INT8 must be done at Python export time).
/// `Static`  = model was already quantized before loading; config is informational.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    Dynamic,
    Static,
}

/// Quantization configuration attached to a model.
/// INT8 + Dynamic: logs a warning — use `torch.quantization.quantize_dynamic` in Python
///   before exporting the TorchScript `.pt` file, then load the quantized file here.
/// F16 + Dynamic: handled automatically via `enable_fp16` at inference time.
/// Any  + Static:  informational — logged on load, no runtime action.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub dtype: QuantizationDtype,
    pub mode: QuantizationMode,
}
```

- [ ] **Step 4: Add `enable_fp16` and `quantization` to TorchOptimizationConfig**

In `src/torch_optimization.rs`, find `pub struct TorchOptimizationConfig` and add two fields at the end of the struct body:

```rust
    /// Convert input tensors to f16 before forward pass (GPU/MPS only, no-op on CPU).
    pub enable_fp16: bool,

    /// Optional quantization config — informational for Static, drives FP16 for Dynamic/F16.
    pub quantization: Option<QuantizationConfig>,
```

Update `impl Default for TorchOptimizationConfig`:

```rust
impl Default for TorchOptimizationConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            num_interop_threads: 1,
            cudnn_benchmark: true,
            enable_autocast: false,
            warmup_iterations: 5,
            warmup_shape: vec![1, 3, 224, 224],
            enable_inference_mode: true,
            device: Device::cuda_if_available(),
            enable_fp16: false,
            quantization: None,
        }
    }
}
```

- [ ] **Step 5: Apply FP16 conversion in `OptimizedTorchModel::infer()`**

In `src/torch_optimization.rs`, replace the existing `infer()` method body:

```rust
/// Optimized inference (automatically chooses best path, applies FP16 if configured).
pub fn infer(&self, input: &Tensor) -> Result<Tensor, String> {
    // Warn once if Int8+Dynamic requested — must be done at export time in Python.
    if let Some(ref qcfg) = self.config.quantization {
        if matches!(qcfg.dtype, crate::torch_optimization::QuantizationDtype::Int8)
            && matches!(qcfg.mode, crate::torch_optimization::QuantizationMode::Dynamic)
        {
            warn!(
                "INT8 dynamic quantization is not supported at Rust inference time. \
                 Quantize the model in Python with torch.quantization.quantize_dynamic \
                 before exporting the .pt file."
            );
        }
    }

    let input = if self.config.enable_fp16 && self.device != Device::Cpu {
        input.to_kind(Kind::Half)
    } else {
        input.shallow_clone()
    };

    if self.config.enable_autocast {
        self.forward_with_autocast(&input)
    } else {
        self.forward(&input)
    }
}
```

- [ ] **Step 6: Add `fp16` and `quantization` builder methods**

In the `impl TorchOptimizationConfigBuilder` block, add after the existing `device()` method:

```rust
pub fn fp16(mut self, enabled: bool) -> Self {
    self.config.enable_fp16 = enabled;
    self
}

pub fn quantization(mut self, config: Option<QuantizationConfig>) -> Self {
    self.config.quantization = config;
    self
}
```

- [ ] **Step 7: Run tests and confirm they pass**

```bash
cargo test -p torch-inference torch_optimization -- --nocapture 2>&1 | grep -E "test.*ok|FAILED|error"
```

Expected: all `torch_optimization::tests::` lines show `ok`.

- [ ] **Step 8: Commit**

```bash
git add src/torch_optimization.rs
git commit -m "feat(torch_optimization): add FP16 half-precision and QuantizationConfig"
```

---

## Task 2: Create `src/core/model_cache.rs`

**Files:**
- Create: `src/core/model_cache.rs`

- [ ] **Step 1: Write the test file inline (tests live at the bottom of model_cache.rs)**

Create `src/core/model_cache.rs` with tests-first structure. Write the full file including tests:

```rust
#![allow(dead_code)]
use anyhow::Result;
use lru::LruCache;
use serde::{de::DeserializeOwned, Serialize};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ── FNV-1a 64-bit ────────────────────────────────────────────────────────────
// Stable across Rust versions and process restarts (unlike DefaultHasher).
// Same algorithm used by TTSManager.

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

/// Compute a stable u64 cache key from three independent byte slices.
/// NUL-byte separators prevent cross-field collisions (e.g. "ab"+"c" ≠ "a"+"bc").
pub fn cache_key(model_id: &str, input: &[u8], params: &[u8]) -> u64 {
    let mut data = Vec::with_capacity(model_id.len() + 1 + input.len() + 1 + params.len());
    data.extend_from_slice(model_id.as_bytes());
    data.push(0);
    data.extend_from_slice(input);
    data.push(0);
    data.extend_from_slice(params);
    fnv1a(&data)
}

// ── CacheStats ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

// ── ModelCache ────────────────────────────────────────────────────────────────

/// Model-agnostic LRU result cache.
///
/// Results are serialized to JSON bytes and stored in an `Arc<Vec<u8>>`.
/// On a hit, only the `Arc` pointer is cloned (~5 ns); deserialization happens
/// on the way out. Works for any `T: Serialize + DeserializeOwned`.
pub struct ModelCache {
    cache: Mutex<LruCache<u64, Arc<Vec<u8>>>>,
    capacity: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ModelCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity.max(1)).expect("capacity >= 1");
        Self {
            cache: Mutex::new(LruCache::new(cap)),
            capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Return a cached result for `key`, or run `f`, cache its result, and return it.
    pub fn get_or_run<T, F>(&self, key: u64, f: F) -> Result<T>
    where
        T: Serialize + DeserializeOwned,
        F: FnOnce() -> Result<T>,
    {
        // Check cache (hold lock only while reading — release before calling f).
        let cached = {
            let mut guard = self.cache.lock().expect("model cache poisoned");
            guard.get(&key).cloned()
        };

        if let Some(bytes) = cached {
            self.hits.fetch_add(1, Ordering::Relaxed);
            let value: T = serde_json::from_slice(&bytes)
                .map_err(|e| anyhow::anyhow!("cache deserialize failed: {}", e))?;
            return Ok(value);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = f()?;

        // Serialize and store.
        let bytes = serde_json::to_vec(&result)
            .map_err(|e| anyhow::anyhow!("cache serialize failed: {}", e))?;
        {
            let mut guard = self.cache.lock().expect("model cache poisoned");
            guard.put(key, Arc::new(bytes));
        }

        Ok(result)
    }

    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 { 0.0 } else { hits as f64 / total as f64 };
        CacheStats { hits, misses, hit_rate }
    }

    pub fn clear(&self) {
        let mut guard = self.cache.lock().expect("model cache poisoned");
        guard.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_stable() {
        let k1 = cache_key("model.pt", b"input_bytes", b"params");
        let k2 = cache_key("model.pt", b"input_bytes", b"params");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_model_id() {
        let k1 = cache_key("model_a.pt", b"input", b"params");
        let k2 = cache_key("model_b.pt", b"input", b"params");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_input() {
        let k1 = cache_key("model.pt", b"input_a", b"params");
        let k2 = cache_key("model.pt", b"input_b", b"params");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_params() {
        let k1 = cache_key("model.pt", b"input", b"params_a");
        let k2 = cache_key("model.pt", b"input", b"params_b");
        assert_ne!(k1, k2);
    }

    /// Key collision resistance: "ab" + "c" must differ from "a" + "bc".
    #[test]
    fn test_cache_key_no_cross_field_collision() {
        let k1 = cache_key("ab", b"c", b"");
        let k2 = cache_key("a", b"bc", b"");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_get_or_run_miss_then_hit() {
        let cache: ModelCache = ModelCache::new(4);
        let key = cache_key("m", b"input", b"p");

        let mut call_count = 0u32;
        let r1: u32 = cache
            .get_or_run(key, || { call_count += 1; Ok(42u32) })
            .unwrap();
        assert_eq!(r1, 42);
        assert_eq!(call_count, 1);

        let r2: u32 = cache
            .get_or_run(key, || { call_count += 1; Ok(99u32) })
            .unwrap();
        assert_eq!(r2, 42); // cached value, not 99
        assert_eq!(call_count, 1); // f was NOT called again
    }

    #[test]
    fn test_stats_hit_and_miss_counting() {
        let cache = ModelCache::new(4);
        let key = cache_key("m", b"x", b"y");

        let _: u32 = cache.get_or_run(key, || Ok(1u32)).unwrap(); // miss
        let _: u32 = cache.get_or_run(key, || Ok(1u32)).unwrap(); // hit
        let _: u32 = cache.get_or_run(key, || Ok(1u32)).unwrap(); // hit

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 2);
        assert!((stats.hit_rate - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_empty_cache_hit_rate_is_zero() {
        let cache = ModelCache::new(4);
        let stats = cache.stats();
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[test]
    fn test_clear_resets_stats_and_cache() {
        let cache = ModelCache::new(4);
        let key = cache_key("m", b"a", b"b");
        let _: u32 = cache.get_or_run(key, || Ok(7u32)).unwrap();

        cache.clear();
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);

        // After clear, same key triggers a miss (f is called again).
        let mut calls = 0u32;
        let _: u32 = cache.get_or_run(key, || { calls += 1; Ok(7u32) }).unwrap();
        assert_eq!(calls, 1);
    }

    #[test]
    fn test_capacity_respected_evicts_oldest() {
        let cache = ModelCache::new(2);
        let k1 = cache_key("m", b"1", b"");
        let k2 = cache_key("m", b"2", b"");
        let k3 = cache_key("m", b"3", b"");

        let _: u32 = cache.get_or_run(k1, || Ok(1u32)).unwrap();
        let _: u32 = cache.get_or_run(k2, || Ok(2u32)).unwrap();
        let _: u32 = cache.get_or_run(k3, || Ok(3u32)).unwrap(); // evicts k1

        // k1 should be a miss now
        let mut calls = 0u32;
        let _: u32 = cache.get_or_run(k1, || { calls += 1; Ok(1u32) }).unwrap();
        assert_eq!(calls, 1, "k1 should have been evicted");
    }

    #[test]
    fn test_capacity_accessor() {
        let cache = ModelCache::new(256);
        assert_eq!(cache.capacity(), 256);
    }

    #[test]
    fn test_get_or_run_error_propagates() {
        let cache = ModelCache::new(4);
        let key = cache_key("m", b"err", b"");
        let result: anyhow::Result<u32> = cache.get_or_run(key, || {
            anyhow::bail!("inference failed")
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("inference failed"));
        // Error must NOT be cached — next call runs f again.
        let mut calls = 0u32;
        let ok: u32 = cache.get_or_run(key, || { calls += 1; Ok(55u32) }).unwrap();
        assert_eq!(ok, 55);
        assert_eq!(calls, 1);
    }
}
```

- [ ] **Step 2: Register the new module in `src/core/mod.rs`**

Add one line to `src/core/mod.rs` in the "Neural network and ML modules" section:

```rust
pub mod model_cache;
```

Place it directly before `pub mod neural_network;`.

- [ ] **Step 3: Run the new tests**

```bash
cargo test -p torch-inference model_cache -- --nocapture 2>&1 | grep -E "test.*ok|FAILED|error"
```

Expected: all `core::model_cache::tests::` lines show `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/core/model_cache.rs src/core/mod.rs
git commit -m "feat(model_cache): add LRU result cache with FNV-1a keying and hit/miss stats"
```

---

## Task 3: Migrate `ImageClassifier` to `OptimizedTorchModel` + `ModelCache`

**Files:**
- Modify: `src/core/image_classifier.rs`

- [ ] **Step 1: Add failing tests for cache behaviour and new constructor signature**

Add to the `#[cfg(test)] mod tests` block in `src/core/image_classifier.rs`:

```rust
#[test]
fn test_cache_stats_initially_zero() {
    let clf = ImageClassifier::new_stub(vec!["cat".to_string()]);
    let stats = clf.cache_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p torch-inference image_classifier::tests::test_cache_stats_initially_zero -- --nocapture 2>&1 | grep -E "FAILED|error"
```

Expected: compile error — `cache_stats` not defined.

- [ ] **Step 3: Update imports and struct definition**

Replace the top of `src/core/image_classifier.rs` (imports + struct) with:

```rust
#![allow(dead_code)]
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "torch")]
use tch::{Device, Kind};

#[cfg(not(feature = "torch"))]
type Device = ();

#[cfg(feature = "torch")]
use crate::models::pytorch_loader::get_best_device;

use crate::core::model_cache::{ModelCache, CacheStats, cache_key};
use crate::torch_optimization::{OptimizedTorchModel, TorchOptimizationConfig,
    TorchOptimizationConfigBuilder};

/// Image classification model wrapper
pub struct ImageClassifier {
    #[cfg(feature = "torch")]
    optimizer: OptimizedTorchModel,

    #[cfg(not(feature = "torch"))]
    _optimizer: (),

    labels: Vec<String>,
    input_size: (i64, i64),
    normalize_mean: Vec<f64>,
    normalize_std: Vec<f64>,
    cache: ModelCache,
    model_id: String,
}
```

- [ ] **Step 4: Update `new()` to accept optional `TorchOptimizationConfig`**

Replace the `#[cfg(feature = "torch")] pub fn new(...)` implementation:

```rust
#[cfg(feature = "torch")]
pub fn new(
    model_path: &Path,
    labels: Vec<String>,
    input_size: Option<(i64, i64)>,
    device: Option<Device>,
    opt_config: Option<TorchOptimizationConfig>,
) -> Result<Self> {
    let device = device.unwrap_or_else(|| get_best_device());
    let model_id = model_path.to_string_lossy().to_string();

    let config = opt_config.unwrap_or_else(|| {
        let is_accel = device != Device::Cpu;
        TorchOptimizationConfigBuilder::new()
            .device(device)
            .cudnn_benchmark(true)
            .warmup_iterations(3)
            .autocast(is_accel)
            .fp16(is_accel)
            .build()
    });

    log::info!("Loading image classification model from {:?}", model_path);
    let model_path_str = model_path.to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
    let mut optimizer = OptimizedTorchModel::new(model_path_str, config)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    optimizer.warmup().map_err(|e| anyhow::anyhow!("{}", e))?;

    let input_size = input_size.unwrap_or((224, 224));
    log::info!("Image classifier loaded successfully with {} classes", labels.len());

    Ok(Self {
        optimizer,
        labels,
        input_size,
        normalize_mean: vec![0.485, 0.456, 0.406],
        normalize_std: vec![0.229, 0.224, 0.225],
        cache: ModelCache::new(256),
        model_id,
    })
}

#[cfg(not(feature = "torch"))]
pub fn new(
    _model_path: &Path,
    _labels: Vec<String>,
    _input_size: Option<(i64, i64)>,
    _device: Option<()>,
    _opt_config: Option<()>,
) -> Result<Self> {
    bail!("PyTorch feature not enabled. Compile with --features torch");
}
```

- [ ] **Step 5: Update `preprocess_image()` to use `optimizer` device**

Replace the `#[cfg(feature = "torch")] pub fn preprocess_image(...)` body (keep the same logic, just change device source):

```rust
#[cfg(feature = "torch")]
pub fn preprocess_image(&self, image_path: &Path) -> Result<tch::Tensor> {
    use tch::vision;
    log::debug!("Preprocessing image: {:?}", image_path);

    let image = vision::image::load(image_path)
        .context("Failed to load image")?;
    let resized = vision::image::resize(&image, self.input_size.0, self.input_size.1)?;
    let mut tensor = resized.to_kind(Kind::Float) / 255.0;

    for c in 0..3i64 {
        let mean = self.normalize_mean[c as usize];
        let std = self.normalize_std[c as usize];
        let mut channel = tensor.select(0, c);
        channel -= mean;
        channel /= std;
    }

    Ok(tensor.unsqueeze(0).to_device(self.optimizer.device()))
}
```

- [ ] **Step 6: Update `classify()` and `classify_bytes()` to use cache and optimizer**

Replace both `#[cfg(feature = "torch")]` inference methods:

```rust
/// Classify image from a file path. Reads file bytes and delegates to classify_bytes
/// so the result cache is always used.
#[cfg(feature = "torch")]
pub fn classify(&self, image_path: &Path, top_k: usize) -> Result<TopKResults> {
    let image_bytes = std::fs::read(image_path)
        .with_context(|| format!("Failed to read image file: {:?}", image_path))?;
    self.classify_bytes(&image_bytes, top_k)
}

#[cfg(not(feature = "torch"))]
pub fn classify(&self, _image_path: &Path, _top_k: usize) -> Result<TopKResults> {
    bail!("PyTorch feature not enabled");
}

/// Classify from raw image bytes. Result is cached by (model_id, bytes, top_k).
#[cfg(feature = "torch")]
pub fn classify_bytes(&self, image_bytes: &[u8], top_k: usize) -> Result<TopKResults> {
    let key = cache_key(&self.model_id, image_bytes, &(top_k as u64).to_le_bytes());
    self.cache.get_or_run(key, || {
        let start = std::time::Instant::now();

        // Write to temp file for image loading (tch requires a path).
        let temp_path = std::env::temp_dir()
            .join(format!("clf_{}.jpg", uuid::Uuid::new_v4()));
        std::fs::write(&temp_path, image_bytes)?;
        let input = self.preprocess_image(&temp_path);
        let _ = std::fs::remove_file(&temp_path);
        let input = input?;

        let output = self.optimizer.infer(&input)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let probabilities = output.softmax(-1, Kind::Float);
        let (values, indices) = probabilities.topk(top_k as i64, -1, true, true);
        let values: Vec<f32> = values.try_into()?;
        let indices: Vec<i64> = indices.try_into()?;

        let predictions: Vec<ClassificationResult> = indices.iter().enumerate()
            .map(|(i, &class_id)| {
                let class_id = class_id as usize;
                let label = self.labels.get(class_id)
                    .cloned()
                    .unwrap_or_else(|| format!("class_{}", class_id));
                ClassificationResult { label, confidence: values[i], class_id }
            })
            .collect();

        let inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "Classification complete: {} ({:.2}%) in {:.2}ms",
            predictions[0].label, predictions[0].confidence * 100.0, inference_time_ms
        );
        Ok(TopKResults { predictions, inference_time_ms })
    })
}

#[cfg(not(feature = "torch"))]
pub fn classify_bytes(&self, _image_bytes: &[u8], _top_k: usize) -> Result<TopKResults> {
    bail!("PyTorch feature not enabled");
}
```

- [ ] **Step 7: Add `cache_stats()` method and update `new_stub()` for tests**

Add after `get_label()`:

```rust
/// Expose cache stats for observability and testing.
pub fn cache_stats(&self) -> CacheStats {
    self.cache.stats()
}
```

Update `new_stub()` to include the new fields:

```rust
#[cfg(test)]
pub fn new_stub(labels: Vec<String>) -> Self {
    Self {
        #[cfg(not(feature = "torch"))]
        _optimizer: (),
        labels,
        input_size: (224, 224),
        normalize_mean: vec![0.485, 0.456, 0.406],
        normalize_std: vec![0.229, 0.224, 0.225],
        cache: ModelCache::new(256),
        model_id: "stub".to_string(),
    }
}
```

- [ ] **Step 8: Run tests**

```bash
cargo test -p torch-inference image_classifier -- --nocapture 2>&1 | grep -E "test.*ok|FAILED|error"
```

Expected: all `core::image_classifier::tests::` lines show `ok`.

- [ ] **Step 9: Commit**

```bash
git add src/core/image_classifier.rs
git commit -m "feat(image_classifier): adopt OptimizedTorchModel and ModelCache"
```

---

## Task 4: Migrate `YoloDetector` to `OptimizedTorchModel` + `ModelCache`

**Files:**
- Modify: `src/core/yolo.rs`

- [ ] **Step 1: Add failing cache test**

Find the `#[cfg(test)] mod tests` block in `src/core/yolo.rs` and add:

```rust
#[test]
fn test_yolo_cache_stats_initially_zero() {
    let detector = YoloDetector::new_stub(
        YoloVersion::V8,
        YoloSize::Nano,
        vec!["person".to_string()],
    );
    let stats = detector.cache_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p torch-inference yolo::tests::test_yolo_cache_stats_initially_zero -- --nocapture 2>&1 | grep -E "FAILED|error"
```

Expected: compile error — `new_stub`, `cache_stats` not defined.

- [ ] **Step 3: Update imports and `YoloDetector` struct**

Replace the imports at the top of `src/core/yolo.rs`:

```rust
#![allow(dead_code)]
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "torch")]
use tch::{Device, Kind, Tensor};

#[cfg(not(feature = "torch"))]
type Device = ();

use crate::core::model_cache::{ModelCache, CacheStats, cache_key};
use crate::torch_optimization::{OptimizedTorchModel, TorchOptimizationConfig,
    TorchOptimizationConfigBuilder};
```

Replace the `YoloDetector` struct:

```rust
/// YOLO Object Detector
pub struct YoloDetector {
    #[cfg(feature = "torch")]
    optimizer: OptimizedTorchModel,

    #[cfg(not(feature = "torch"))]
    _optimizer: (),

    version: YoloVersion,
    size: YoloSize,
    class_names: Vec<String>,
    input_size: (i64, i64),
    conf_threshold: f32,
    iou_threshold: f32,
    cache: ModelCache,
    model_id: String,
}
```

- [ ] **Step 4: Update `YoloDetector::new()`**

Replace the `#[cfg(feature = "torch")] pub fn new(...)` with:

```rust
#[cfg(feature = "torch")]
pub fn new(
    model_path: &Path,
    version: YoloVersion,
    size: YoloSize,
    class_names: Vec<String>,
    device: Option<Device>,
    opt_config: Option<TorchOptimizationConfig>,
) -> Result<Self> {
    let device = device.unwrap_or(Device::Cpu);
    let model_id = model_path.to_string_lossy().to_string();

    let config = opt_config.unwrap_or_else(|| {
        let is_accel = device != Device::Cpu;
        TorchOptimizationConfigBuilder::new()
            .device(device)
            .cudnn_benchmark(true)
            .warmup_iterations(3)
            .autocast(is_accel)
            .fp16(is_accel)
            .warmup_shape(vec![1, 3, 640, 640])
            .build()
    });

    log::info!("Loading {} ({}) model from {:?}", version.as_str(), size.suffix(), model_path);
    let model_path_str = model_path.to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
    let mut optimizer = OptimizedTorchModel::new(model_path_str, config)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    optimizer.warmup().map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(Self {
        optimizer,
        version,
        size,
        class_names,
        input_size: (640, 640),
        conf_threshold: 0.25,
        iou_threshold: 0.45,
        cache: ModelCache::new(128),
        model_id,
    })
}

#[cfg(not(feature = "torch"))]
pub fn new(
    _model_path: &Path,
    _version: YoloVersion,
    _size: YoloSize,
    _class_names: Vec<String>,
    _device: Option<()>,
    _opt_config: Option<()>,
) -> Result<Self> {
    bail!("PyTorch feature not enabled. Compile with --features torch");
}
```

- [ ] **Step 5: Update detection method to use cache**

Find the `#[cfg(feature = "torch")] pub fn detect_bytes(...)` method (or equivalent inference entry point) and wrap with cache. The cache key includes `conf_threshold` and `iou_threshold` bytes:

```rust
#[cfg(feature = "torch")]
pub fn detect_bytes(&self, image_bytes: &[u8]) -> Result<YoloResults> {
    let mut params = Vec::with_capacity(8);
    params.extend_from_slice(&self.conf_threshold.to_le_bytes());
    params.extend_from_slice(&self.iou_threshold.to_le_bytes());
    let key = cache_key(&self.model_id, image_bytes, &params);

    self.cache.get_or_run(key, || {
        // Write temp file for image loading.
        let temp_path = std::env::temp_dir()
            .join(format!("yolo_{}.jpg", uuid::Uuid::new_v4()));
        std::fs::write(&temp_path, image_bytes)?;
        let result = self.detect(&temp_path);
        let _ = std::fs::remove_file(&temp_path);
        result
    })
}

#[cfg(not(feature = "torch"))]
pub fn detect_bytes(&self, _image_bytes: &[u8]) -> Result<YoloResults> {
    bail!("PyTorch feature not enabled");
}
```

Update the inner `detect()` method to use `self.optimizer.infer()` instead of `self.model.forward_ts(...)`. Find the line with `self.model.forward_ts(&[input])?` and replace with:

```rust
let output = self.optimizer.infer(&input)
    .map_err(|e| anyhow::anyhow!("{}", e))?;
```

- [ ] **Step 6: Add `cache_stats()`, `new_stub()`, update device references**

Add `cache_stats()` and `new_stub()`:

```rust
pub fn cache_stats(&self) -> CacheStats {
    self.cache.stats()
}

#[cfg(test)]
pub fn new_stub(
    version: YoloVersion,
    size: YoloSize,
    class_names: Vec<String>,
) -> Self {
    Self {
        #[cfg(not(feature = "torch"))]
        _optimizer: (),
        version,
        size,
        class_names,
        input_size: (640, 640),
        conf_threshold: 0.25,
        iou_threshold: 0.45,
        cache: ModelCache::new(128),
        model_id: "stub".to_string(),
    }
}
```

For any remaining references to `self.device` in the torch-gated code, replace with `self.optimizer.device()`.

- [ ] **Step 7: Run tests**

```bash
cargo test -p torch-inference yolo -- --nocapture 2>&1 | grep -E "test.*ok|FAILED|error"
```

Expected: all `core::yolo::tests::` lines show `ok`.

- [ ] **Step 8: Commit**

```bash
git add src/core/yolo.rs
git commit -m "feat(yolo): adopt OptimizedTorchModel and ModelCache"
```

---

## Task 5: Migrate `NeuralNetwork` to `OptimizedTorchModel` + `ModelCache`; remove old quantization stub

**Files:**
- Modify: `src/core/neural_network.rs`

- [ ] **Step 1: Add failing cache test**

Add to `#[cfg(test)] mod tests` in `src/core/neural_network.rs`:

```rust
#[test]
fn test_neural_network_cache_stats_initially_zero() {
    let nn = NeuralNetwork::new_stub(None);
    let stats = nn.cache_stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p torch-inference neural_network::tests::test_neural_network_cache_stats_initially_zero -- --nocapture 2>&1 | grep -E "FAILED|error"
```

Expected: compile error — `cache_stats` not defined.

- [ ] **Step 3: Update imports and struct**

Replace the top of `src/core/neural_network.rs`:

```rust
#![allow(dead_code)]
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "torch")]
use tch::{Device, Kind, Tensor};

#[cfg(not(feature = "torch"))]
type Device = ();

#[cfg(feature = "torch")]
use crate::models::pytorch_loader::get_best_device;

use crate::core::model_cache::{ModelCache, CacheStats, cache_key};
use crate::torch_optimization::{OptimizedTorchModel, TorchOptimizationConfig,
    TorchOptimizationConfigBuilder};
```

Replace `NeuralNetwork` struct:

```rust
/// Generic neural network for inference
pub struct NeuralNetwork {
    #[cfg(feature = "torch")]
    optimizer: OptimizedTorchModel,

    #[cfg(not(feature = "torch"))]
    _optimizer: (),

    input_shapes: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
    metadata: NetworkMetadata,
    cache: ModelCache,
    model_id: String,
}
```

- [ ] **Step 4: Update `new()` signature**

Replace the `#[cfg(feature = "torch")] pub fn new(...)`:

```rust
#[cfg(feature = "torch")]
pub fn new(
    model_path: &Path,
    device: Option<Device>,
    metadata: Option<NetworkMetadata>,
    cache_capacity: Option<usize>,
    opt_config: Option<TorchOptimizationConfig>,
) -> Result<Self> {
    let device = device.unwrap_or_else(|| get_best_device());
    let model_id = model_path.to_string_lossy().to_string();

    let config = opt_config.unwrap_or_else(|| {
        let is_accel = device != Device::Cpu;
        TorchOptimizationConfigBuilder::new()
            .device(device)
            .cudnn_benchmark(true)
            .warmup_iterations(3)
            .autocast(is_accel)
            .fp16(is_accel)
            .build()
    });

    let metadata = metadata.unwrap_or(NetworkMetadata {
        name: "custom_model".to_string(),
        task: "unknown".to_string(),
        framework: "pytorch".to_string(),
        input_names: vec!["input".to_string()],
        output_names: vec!["output".to_string()],
        description: None,
    });

    log::info!("Loading neural network model from {:?}", model_path);
    let model_path_str = model_path.to_str()
        .ok_or_else(|| anyhow::anyhow!("model path is not valid UTF-8"))?;
    let mut optimizer = OptimizedTorchModel::new(model_path_str, config)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    optimizer.warmup().map_err(|e| anyhow::anyhow!("{}", e))?;

    log::info!("Neural network loaded successfully: {}", metadata.name);
    Ok(Self {
        optimizer,
        input_shapes: Vec::new(),
        output_shapes: Vec::new(),
        metadata,
        cache: ModelCache::new(cache_capacity.unwrap_or(512)),
        model_id,
    })
}

#[cfg(not(feature = "torch"))]
pub fn new(
    _model_path: &Path,
    _device: Option<()>,
    _metadata: Option<NetworkMetadata>,
    _cache_capacity: Option<usize>,
    _opt_config: Option<()>,
) -> Result<Self> {
    bail!("PyTorch feature not enabled. Compile with --features torch");
}
```

- [ ] **Step 5: Update `predict_from_slice()` to use cache**

Replace `#[cfg(feature = "torch")] pub fn predict_from_slice(...)`:

```rust
#[cfg(feature = "torch")]
pub fn predict_from_slice(&self, data: &[f32], shape: &[i64]) -> Result<InferenceResult> {
    // Build params bytes from shape.
    let mut params = Vec::with_capacity(shape.len() * 8);
    for &dim in shape {
        params.extend_from_slice(&dim.to_le_bytes());
    }
    // Cast data to byte slice for hashing only (safe: f32 is Pod).
    let data_bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    };
    let key = cache_key(&self.model_id, data_bytes, &params);

    self.cache.get_or_run(key, || {
        let start = std::time::Instant::now();
        let input = Tensor::from_slice(data)
            .reshape(shape)
            .to_device(self.optimizer.device());
        let output = self.optimizer.infer(&input)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let output_vec: Vec<f32> = output.flatten(0, -1).try_into()?;
        let mut outputs = HashMap::new();
        outputs.insert(
            self.metadata.output_names.first()
                .cloned()
                .unwrap_or_else(|| "output".to_string()),
            output_vec,
        );
        Ok(InferenceResult {
            outputs,
            inference_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            device: format!("{:?}", self.optimizer.device()),
        })
    })
}

#[cfg(not(feature = "torch"))]
pub fn predict_from_slice(&self, _data: &[f32], _shape: &[i64]) -> Result<InferenceResult> {
    bail!("PyTorch feature not enabled");
}
```

- [ ] **Step 6: Update `predict()` and `predict_multi()` to use optimizer**

Replace the `#[cfg(feature = "torch")] pub fn predict(...)` body:

```rust
#[cfg(feature = "torch")]
pub fn predict(&self, input: &Tensor) -> Result<Tensor> {
    log::debug!("Running inference with input shape: {:?}", input.size());
    let start = std::time::Instant::now();
    let input = input.to_device(self.optimizer.device());
    let output = self.optimizer.infer(&input)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    log::debug!("Inference completed in {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    Ok(output)
}
```

Replace the `#[cfg(feature = "torch")] pub fn predict_multi(...)` body:

```rust
#[cfg(feature = "torch")]
pub fn predict_multi(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
    log::debug!("Running inference with {} inputs", inputs.len());
    let start = std::time::Instant::now();
    let inputs: Vec<Tensor> = inputs.iter()
        .map(|t| t.to_device(self.optimizer.device()))
        .collect();
    let outputs = self.optimizer.infer(&inputs[0])
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    log::debug!("Multi-input inference completed in {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0);
    Ok(vec![outputs])
}
```

- [ ] **Step 7: Add `cache_stats()`, update `device()`, update `new_stub()`, remove old quantization module**

Add `cache_stats()` and update `device()`:

```rust
pub fn cache_stats(&self) -> CacheStats {
    self.cache.stats()
}

#[cfg(feature = "torch")]
pub fn device(&self) -> Device {
    self.optimizer.device()
}

#[cfg(not(feature = "torch"))]
pub fn device(&self) -> () {
    ()
}
```

Update `new_stub()`:

```rust
#[cfg(test)]
pub fn new_stub(metadata: Option<NetworkMetadata>) -> Self {
    Self {
        #[cfg(not(feature = "torch"))]
        _optimizer: (),
        input_shapes: Vec::new(),
        output_shapes: Vec::new(),
        metadata: metadata.unwrap_or(NetworkMetadata {
            name: "stub".to_string(),
            task: "test".to_string(),
            framework: "none".to_string(),
            input_names: vec!["input".to_string()],
            output_names: vec!["output".to_string()],
            description: None,
        }),
        cache: ModelCache::new(512),
        model_id: "stub".to_string(),
    }
}
```

**Delete the entire `pub mod quantization` block** at the bottom of `neural_network.rs` — it has been superseded by `QuantizationConfig` in `torch_optimization.rs`. The tests for it (`test_quantization_*`) should also be removed from the test block.

- [ ] **Step 8: Run full test suite**

```bash
cargo test -p torch-inference -- --nocapture 2>&1 | grep -E "test result|FAILED|error\["
```

Expected: `test result: ok.` with 0 failures.

- [ ] **Step 9: Commit**

```bash
git add src/core/neural_network.rs
git commit -m "feat(neural_network): adopt OptimizedTorchModel and ModelCache; remove quantization stub"
```

---

## Self-Review

**Spec coverage check:**
- Goal 1 (LRU result cache): Task 2 creates `ModelCache`; Tasks 3–5 wire it into all three models. ✓
- Goal 2 (FP16 half-precision): Task 1 adds `enable_fp16` + conversion in `infer()`. ✓
- Goal 3 (QuantizationConfig): Task 1 adds `QuantizationDtype`, `QuantizationMode`, `QuantizationConfig` to `torch_optimization.rs`; INT8 warning logged at runtime. ✓
- Goal 4 (OptimizedTorchModel adoption): Tasks 3–5 replace `CModule` with `OptimizedTorchModel` in all three models. ✓
- `neural_network.rs` old quantization stub removed. ✓

**Type consistency check:**
- `cache_key()` signature: `(model_id: &str, input: &[u8], params: &[u8]) -> u64` — used consistently in Tasks 3, 4, 5. ✓
- `ModelCache::get_or_run` returns `Result<T>` — used with `?` unwrap in all three model inference methods. ✓
- `CacheStats` returned by `cache.stats()` and by `cache_stats()` accessors. ✓
- `TorchOptimizationConfigBuilder::fp16(bool)` and `.quantization(Option<QuantizationConfig>)` defined in Task 1, used in Tasks 3–5. ✓
- `OptimizedTorchModel::device()` returns `Device` — used in `preprocess_image`, `predict`, `predict_from_slice`. ✓

**No placeholder scan:** No TBD, TODO, "implement later", or "similar to Task N" patterns. All code blocks are complete. ✓
