# Engineering Improvements Design
**Date:** 2026-03-27
**Scope:** torch-inference (Rust ML inference server)
**Approach:** Option B — three theme-grouped batches, each independently buildable and shippable

---

## Overview

Seven targeted engineering fixes across three batches. No new features, no refactors beyond what is necessary to fix each issue. All changes are backward-compatible at the API level.

---

## Batch 1: Correctness

### 1.1 Streaming model downloads

**File:** `src/api/models.rs` — `download_model_async`

**Problem:** `response.bytes().await?` buffers the entire model file in memory before writing to disk. Models range from 60 MB to 2 GB; this will OOM on large models.

**Fix:** Replace with `response.bytes_stream()` piped into `tokio::io::copy`. Use `tokio_util::io::StreamReader` to adapt the `bytes::Bytes` stream into an `AsyncRead`, then stream directly into the `tokio::fs::File`. Memory per download is bounded to the read buffer (~64 KB).

**Dependencies:** Add `tokio-util = { version = "0.7", features = ["io"] }` to `Cargo.toml`.

**Unchanged:** Public function signature, logging, directory creation, the config-download sub-path (which downloads small JSON files and can keep the buffered path).

---

### 1.2 Stable synthesis cache hash

**File:** `src/core/tts_manager.rs` — `synthesis_cache_key`

**Problem:** `std::collections::hash_map::DefaultHasher` is explicitly documented as not stable across Rust versions or process restarts. The synthesis LRU cache uses it as the key, meaning cache hits are unreliable and can vary between runs.

**Fix:** Replace `DefaultHasher` with `fnv::FnvHasher` (already a transitive dependency via `dashmap`). `FnvHasher` is deterministic and stable. The key type remains `u64`; no other changes.

**Alternative considered:** `ahash` (also transitive). FNV is preferred here because it has no randomised seed and the inputs (text + engine_id + params) are short strings where FNV performs well.

---

### 1.3 Deterministic dedup keys

**File:** `src/dedup.rs` — `generate_key`

**Problem:** `inputs.to_string()` serialises a `serde_json::Value::Object`, which is backed by a `Map<String, Value>` (insertion-ordered in serde_json ≥ 1.0, but callers can construct `Value` in any order). Two logically identical requests with differently-ordered keys produce different dedup keys → cache misses.

**Fix:** Canonicalise the JSON before hashing: recursively sort object keys using a helper `canonical_json(v: &Value) -> String`, then take `sha2::Sha256` of the result (truncated to hex). This makes the key independent of insertion order. The `sha2` crate is already a dependency.

Key format changes from:
```
"{model}:{raw_json}:{epoch/10}"
```
to:
```
"{model}:{sha256_of_canonical_json}:{epoch/10}"
```

The key is still a `String`; no other changes to the struct or callers.

---

### 1.4 Dedup cache eviction

**File:** `src/dedup.rs` — `set`, `RequestDeduplicator`

**Problem:** When the cache reaches `max_entries`, `set()` returns `Err("Deduplication cache full")`. The only caller (`handlers.rs:90`) ignores this with `let _ = ...`. New requests are silently not cached once the cache fills.

**Fix:** Replace the `DashMap<String, DeduplicationEntry>` backing store with a `parking_lot::Mutex<LruCache<String, DeduplicationEntry>>`. `LruCache` evicts the least-recently-used entry automatically when at capacity. `set()` becomes infallible (return type changes from `Result<(), String>` to `()`). Update the caller in `handlers.rs` accordingly.

**Why not stay with DashMap + manual eviction:** O(n) scan for minimum-timestamp is technically correct but inconsistent with how `synthesis_cache` in `tts_manager.rs` already uses `LruCache`. Using the same pattern is cleaner and O(1).

**Existing tests:** All `test_dedup_*` tests remain valid; update `test_dedup_full` to verify LRU eviction instead of error return.

---

## Batch 2: Alignment

### 2.1 Load model_registry.json

**File:** `src/api/models.rs` — `ModelRegistry::build`, `ModelInfo`

**Problem:** `ModelRegistry::build()` is a ~380-line function hardcoding model data that duplicates `model_registry.json` (tracked in git, recently modified). The two sources are already diverged (the JSON has newer Kokoro fields, different rank formats).

**Fix:** Replace `build()` with `load()`:
```rust
fn load() -> Self {
    let path = std::env::var("MODEL_REGISTRY_PATH")
        .unwrap_or_else(|_| "model_registry.json".to_string());
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| include_str!("../../model_registry.json").to_string());
    serde_json::from_str(&data)
        .unwrap_or_else(|e| { log::warn!("Failed to parse model_registry.json: {}", e); Self::empty() })
}
```

The `include_str!` fallback embeds the file at compile time so the binary works even if the JSON is absent at runtime. `Self::empty()` returns a registry with no models (graceful degradation).

**`ModelInfo` schema changes:** The JSON uses heterogeneous types for some fields (`rank` can be `"Production"` or `8`; `score` can be `"N/A (Native)"` or `57.1`; `voices` can be `"Multi"` or `3`). Change these fields to `serde_json::Value` with typed accessor methods, or use custom deserialise implementations. Recommended: use `#[serde(default)]` with `Option<f32>` / `Option<i32>` and a `String` fallback field for the display value.

**Unchanged:** `OnceLock<ModelRegistry>` singleton, all API endpoint handlers, response shapes.

---

### 2.2 Fix audio format validation stubs

**File:** `src/core/audio.rs` — `validate_mp3`, `validate_flac`, `validate_ogg`

**Problem:** All three return hardcoded `AudioMetadata { sample_rate: 44100, channels: 2, duration_secs: 0.0, ... }` regardless of the actual file. The `load_with_symphonia` path that follows correctly detects real values, making the metadata from `validate_audio` misleading and untestable.

**Fix:** Extract a shared `probe_metadata_with_symphonia(&self, data: &[u8], format: AudioFormat) -> Result<AudioMetadata>` that uses `symphonia`'s probe pipeline to read the actual codec parameters and duration. Replace all three stub methods with calls to this helper.

The `validate_wav` path stays on `hound` (already correct).

**Note:** `symphonia`'s `CodecParams` gives `sample_rate` and `channels` immediately from the container headers, without decoding all samples. Duration requires reading `n_frames / sample_rate`; if `n_frames` is `None` (e.g. VBR MP3), `duration_secs` is `0.0` — same as the current stub, but now accurate for CBR and FLAC.

---

## Batch 3: Performance

### 3.1 Concurrent HuggingFace repo downloads

**File:** `src/api/models.rs` — `download_huggingface_repo`

**Problem:** Files are downloaded sequentially in a `for` loop. A repo with 20 files downloads them one-by-one; latency is O(n × RTT) instead of O(RTT) bounded by concurrency.

**Fix:** Replace the `for` loop with:
```rust
futures::stream::iter(files_to_download)
    .map(|file_path| async move { /* download single file */ })
    .buffer_unordered(8)
    .collect::<Vec<_>>()
    .await;
```

Concurrency limit 8 matches the existing `pool_max_idle_per_host(8)` on the global `HTTP_CLIENT`. No client changes needed.

**Error handling:** Per-file failures are logged as warnings (same as current). The overall function succeeds even if individual files fail.

**Dependencies:** `futures` is already a dependency.

---

### 3.2 Avoid hot-path Arc clone in get_engine

**File:** `src/core/tts_manager.rs` — `get_engine` and call sites

**Problem:** `get_engine` always clones the `Arc<dyn TTSEngine>`, even for callers that only need to call one method on it. In the synthesis hot path this is called once per request and is minor, but it's unnecessarily wasteful for `get_capabilities` and `list_voices` which only need a brief read.

**Fix:** Add:
```rust
fn with_engine<F, R>(&self, id: &str, f: F) -> Option<R>
where
    F: FnOnce(&dyn TTSEngine) -> R,
{
    self.engines.get(id).map(|e| f(e.as_ref()))
}
```

Update `get_capabilities` and `list_voices` to use `with_engine` (no Arc clone). Keep `get_engine` for `synthesize` (which needs to hold the engine across an `.await`) and for the API handler `list_voices` endpoint (which returns the engine to a caller).

---

## Implementation Order

1. **Batch 1** — Correctness: 1.1 → 1.2 → 1.3 → 1.4
2. **Batch 2** — Alignment: 2.1 → 2.2
3. **Batch 3** — Performance: 3.1 → 3.2

Each batch must compile and pass `cargo test` before the next begins.

---

## Files Changed

| File | Batches |
|------|---------|
| `src/api/models.rs` | 1.1, 2.1, 3.1 |
| `src/core/tts_manager.rs` | 1.2, 3.2 |
| `src/dedup.rs` | 1.3, 1.4 |
| `src/core/audio.rs` | 2.2 |
| `src/api/handlers.rs` | 1.4 (caller update) |
| `Cargo.toml` | 1.1 (tokio-util) |

---

## Testing

- All existing unit tests in `src/dedup.rs`, `src/batch.rs`, `src/core/tts_manager.rs`, `src/core/audio.rs` must pass after each batch.
- `test_dedup_full` must be updated: verify LRU eviction occurs rather than an error being returned.
- `test_audio_format_from_extension` is unchanged.
- No new integration tests required (all changes are internal implementation).
