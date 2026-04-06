# Inference Optimizer Design

**Date:** 2026-04-04  
**Branch:** feature/throughput-latency-optimization  
**Scope:** ImageClassifier, YOLO, NeuralNetwork — caching, FP16, quantization config

---

## Problem

ImageClassifier, YOLO, and NeuralNetwork have no model-specific caching or
optimization. They load a raw `CModule` and run inference directly, relying
entirely on cross-cutting infrastructure (API-level response cache, tensor pool,
model pool). In contrast, TTS has a synthesis LRU cache, G2P cache, and ONNX
session pool. LLM has PagedAttention, continuous batching, and speculative
decoding. The three vision/generic models need parity.

---

## Goals

1. **Result LRU cache** — repeated identical inputs skip inference entirely.
2. **FP16 half-precision** — at inference time, on CUDA/MPS, for free throughput gain.
3. **QuantizationConfig** — wire in the existing stub; document the INT8 export path.
4. **`OptimizedTorchModel` adoption** — replace raw `CModule` in all three models.

---

## Non-Goals

- Runtime INT8 quantization (libtorch/tch does not support this at inference time;
  INT8 is an export-time Python step).
- Changes to LLM, TTS, Whisper, or any cross-cutting module (cache.rs, dedup.rs,
  batch.rs, model_pool.rs).
- New HTTP endpoints or handler changes.

---

## Architecture

Two pieces, three updated model files.

```
src/
  torch_optimization.rs          ← MODIFIED
  core/
    model_cache.rs               ← NEW
    image_classifier.rs          ← MODIFIED
    yolo.rs                      ← MODIFIED
    neural_network.rs            ← MODIFIED
```

### Data flow (per inference call)

```
request
  → cache_key(model_id, input_bytes, params_bytes) → u64
  → ModelCache::get_or_run(key, || {
        OptimizedTorchModel::infer(input)   // FP16 conversion here if enabled
    })
  → Arc<CachedOutput>   // pointer clone on hit (~5 ns), full inference on miss
```

---

## Component 1: `ModelCache` (`src/core/model_cache.rs`)

### Struct

```rust
pub struct ModelCache {
    cache: Mutex<LruCache<u64, Arc<Vec<u8>>>>,
    capacity: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}
```

### Key API

```rust
impl ModelCache {
    pub fn new(capacity: usize) -> Self;

    pub fn get_or_run<T, F>(&self, key: u64, f: F) -> Result<Arc<T>>
    where
        T: Serialize + DeserializeOwned,
        F: FnOnce() -> Result<T>;

    pub fn stats(&self) -> CacheStats;  // hits, misses, hit_rate
    pub fn clear(&self);
}
```

### Key hashing

```rust
pub fn cache_key(model_id: &str, input: &[u8], params: &[u8]) -> u64
```

Uses FNV-1a 64-bit (same as TTS). Fields separated by `\0` to prevent
cross-field collisions.

### Storage

Results are `serde_json` serialized into `Vec<u8>`, wrapped in `Arc`. On a hit
only the `Arc` is cloned (pointer increment). Deserialized on the way out. This
makes the cache model-agnostic — any `Serialize + DeserializeOwned` type works.

### Default capacities

| Model           | Capacity | Rationale                                        |
|-----------------|----------|--------------------------------------------------|
| ImageClassifier | 256      | Images repeat in classification pipelines        |
| YOLO            | 128      | Larger results, less likely to repeat exactly    |
| NeuralNetwork   | 512      | Arbitrary tensors, user-configurable via constructor |

---

## Component 2: `OptimizedTorchModel` changes (`src/torch_optimization.rs`)

### FP16 half-precision

New field in `TorchOptimizationConfig`:

```rust
pub enable_fp16: bool,  // default: false; auto-true on CUDA/MPS
```

Applied in `infer()` before the forward pass:

```rust
let input = if self.config.enable_fp16 && self.device != Device::Cpu {
    input.to_kind(Kind::Half)
} else {
    input.shallow_clone()
};
```

No model file changes needed — conversion is at the tensor level.

### QuantizationConfig

Moved from `neural_network.rs::quantization` stub to `torch_optimization.rs`
(where it belongs alongside other compute config). Extended:

```rust
pub enum QuantizationDtype { F32, F16, Int8 }
pub enum QuantizationMode  { Dynamic, Static }

pub struct QuantizationConfig {
    pub dtype: QuantizationDtype,
    pub mode: QuantizationMode,
}
```

Runtime behavior:
- `F16 + Dynamic` → handled by `enable_fp16` flag at inference time.
- `Int8 + Dynamic` → log a warning; INT8 must be applied at Python export time.
  Document the `torch.quantization.quantize_dynamic` export step.
- `* + Static` → model was pre-quantized; config is informational, logged on load.

### Builder additions

```rust
TorchOptimizationConfigBuilder::fp16(bool)
TorchOptimizationConfigBuilder::quantization(QuantizationConfig)
```

---

## Component 3: Model file changes

### Pattern (same for all three)

**New struct fields:**
```rust
optimizer: OptimizedTorchModel,   // replaces raw CModule
cache: ModelCache,
model_id: String,                 // stable key seed (model file path as string)
```

**Constructor signature change** (shown for ImageClassifier; same for others):
```rust
pub fn new(
    model_path: &Path,
    labels: Vec<String>,
    input_size: Option<(i64, i64)>,
    device: Option<Device>,
    opt_config: Option<TorchOptimizationConfig>,  // NEW, None = sensible default
) -> Result<Self>
```

Default when `None`:
- `enable_fp16`: true on CUDA/MPS, false on CPU
- `warmup_iterations`: 3
- `cudnn_benchmark`: true
- `enable_autocast`: matches `enable_fp16`

**Inference call pattern** (shown for `classify_bytes`):
```rust
pub fn classify_bytes(&self, image_bytes: &[u8], top_k: usize) -> Result<TopKResults> {
    let key = cache_key(&self.model_id, image_bytes, &top_k.to_le_bytes());
    self.cache.get_or_run(key, || {
        // preprocess + self.optimizer.infer(input)
    })
}
```

### Per-model cache key params

| Model           | input bytes       | params bytes                              |
|-----------------|-------------------|-------------------------------------------|
| ImageClassifier | raw image bytes   | `top_k` as `u64::to_le_bytes()`           |
| YOLO            | raw image bytes   | `conf_threshold` + `iou_threshold` as f32 le bytes |
| NeuralNetwork   | input data bytes  | shape as concatenated i64 le bytes        |

The `model_id` (file path string) is always included as the third field in
`cache_key`, ensuring stale results are not returned after a model hot-swap.

---

## Testing

- `model_cache.rs`: unit tests for hit/miss counting, `get_or_run` correct result
  passthrough, `clear()`, key collision resistance (same input different params).
- `torch_optimization.rs`: new tests for `fp16()` and `quantization()` builder
  methods; existing tests unchanged.
- Model files: extend existing stub tests to verify `ModelCache` is populated
  after a call (requires a mock `get_or_run` or an accessible `stats()` call).

---

## Open Questions / Future Work

- INT8 static path: add a Python export script under `scripts/quantize.py` as a
  follow-up; out of scope for this spec.
- `NeuralNetwork` cache capacity should be constructor-configurable (512 default).
- Consider exposing `ModelCache::stats()` in a `/metrics` endpoint (future).
