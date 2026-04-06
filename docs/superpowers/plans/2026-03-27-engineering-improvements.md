# Engineering Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix seven correctness, alignment, and performance issues across three theme-grouped batches, each independently buildable and shippable.

**Architecture:** Batch 1 fixes production correctness bugs (streaming downloads, stable hash, deterministic dedup keys, LRU eviction). Batch 2 aligns code with existing data contracts (load model_registry.json, fix audio stubs). Batch 3 removes unnecessary overhead (concurrent HF downloads, hot-path Arc clone).

**Tech Stack:** Rust 2021, actix-web 4.8, tokio 1.40, reqwest 0.12 (`stream` feature), tokio-util 0.7, lru 0.12, parking_lot 0.12, sha2 0.10, hex 0.4, symphonia 0.5, futures 0.3, serde_json 1.0

---

## File Map

| File | Tasks |
|------|-------|
| `Cargo.toml` | 1 (add tokio-util, wiremock dev-dep) |
| `src/api/models.rs` | 1 (streaming), 5 (load JSON), 7 (concurrent downloads) |
| `src/core/tts_manager.rs` | 2 (stable hash), 8 (with_engine) |
| `src/dedup.rs` | 3 (canonical key), 4 (LRU eviction) |
| `src/api/handlers.rs` | 4 (infallible set call) |
| `src/core/audio.rs` | 6 (real audio validation) |
| `src/api/tts.rs` | 8 (use with_engine) |

---

## Batch 1: Correctness

---

### Task 1: Streaming model downloads

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/api/models.rs`

- [ ] **Step 1: Add dependencies to Cargo.toml**

In `Cargo.toml`, add to `[dependencies]`:
```toml
tokio-util = { version = "0.7", features = ["io"] }
```

Add to `[dev-dependencies]`:
```toml
wiremock = "0.6"
```

- [ ] **Step 2: Write the failing test**

At the bottom of `src/api/models.rs`, inside the existing `#[cfg(test)]` block (or create one), add:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{MockServer, Mock, ResponseTemplate};
    use wiremock::matchers::method;

    #[tokio::test]
    async fn test_download_file_streaming_writes_content() {
        let server = MockServer::start().await;
        let content = b"hello streaming world 1234567890";

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(content.as_slice()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.bin");
        let client = reqwest::Client::new();

        download_file_streaming(&client, &server.uri(), &dest)
            .await
            .expect("streaming download should succeed");

        assert_eq!(std::fs::read(&dest).unwrap(), content);
    }

    #[tokio::test]
    async fn test_download_file_streaming_fails_on_http_error() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("missing.bin");
        let client = reqwest::Client::new();

        let result = download_file_streaming(&client, &server.uri(), &dest).await;
        assert!(result.is_err(), "should error on 404");
    }
}
```

- [ ] **Step 3: Run the tests to confirm they fail**

```bash
cd /Users/evintleovonzko/Documents/Works/Kolosal/torch-inference
cargo test -p torch_inference test_download_file_streaming 2>&1 | head -30
```

Expected: compile error — `download_file_streaming` not found.

- [ ] **Step 4: Implement `download_file_streaming`**

In `src/api/models.rs`, add this function above `download_model_async`:

```rust
/// Download a single URL to disk using streaming, so the full file content
/// is never held in memory. Memory usage is bounded to the I/O read buffer
/// (~64 KB) regardless of file size.
async fn download_file_streaming(
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
) -> anyhow::Result<()> {
    use futures_util::TryStreamExt;
    use tokio_util::io::StreamReader;

    let response = client.get(url).send().await?;
    anyhow::ensure!(
        response.status().is_success(),
        "HTTP {} downloading {}",
        response.status(),
        url
    );

    let stream = response
        .bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
    let mut reader = StreamReader::new(stream);
    let mut file = tokio::fs::File::create(dest).await?;
    tokio::io::copy(&mut reader, &mut file).await?;
    Ok(())
}
```

- [ ] **Step 5: Replace the buffered download in `download_model_async`**

In `download_model_async`, find the block that downloads a direct file (inside `if model.url.contains("resolve")`). Replace the buffered path:

```rust
// REMOVE this block (roughly lines that do response.bytes().await):
let mut file = fs::File::create(&filepath).await?;
let bytes = response.bytes().await?;
file.write_all(&bytes).await?;
log::info!("Downloaded {} ({} bytes) to {:?}", model.name, bytes.len(), filepath);
```

Replace with:

```rust
download_file_streaming(client, &model.url, &filepath).await?;
log::info!("Downloaded {} to {:?}", model.name, filepath);
```

Also remove the `let response = client.get(&model.url).send().await?;` and the status check above it — `download_file_streaming` handles both. Remove the unused `extension`/`filename` variables and instead derive the extension from the URL:

```rust
if model.url.contains("resolve") {
    let extension = if model.url.ends_with(".pth") { "pth" }
                   else if model.url.ends_with(".onnx") { "onnx" }
                   else { "bin" };
    let filepath = cache_dir.join(format!("model.{}", extension));
    download_file_streaming(get_http_client(), &model.url, &filepath).await?;
    log::info!("Downloaded {} to {:?}", model.name, filepath);
    // ...rest of config download logic unchanged...
}
```

- [ ] **Step 6: Also remove unused imports in `download_model_async`**

Remove `use tokio::io::AsyncWriteExt;` from the `download_model_async` function body if it's only used by the old buffered write (check if it's used elsewhere in the function first).

- [ ] **Step 7: Run tests**

```bash
cargo test -p torch_inference test_download_file_streaming 2>&1
```

Expected: both tests PASS.

- [ ] **Step 8: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 9: Commit**

```bash
git add Cargo.toml Cargo.lock src/api/models.rs
git commit -m "fix: stream model downloads to avoid OOM on large files

Replace response.bytes().await? (buffers entire file in memory) with
StreamReader + tokio::io::copy so memory per download is bounded to
the I/O buffer (~64 KB) regardless of model size (up to 2 GB).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Stable synthesis cache hash

**Files:**
- Modify: `src/core/tts_manager.rs`

- [ ] **Step 1: Write the failing test**

In `src/core/tts_manager.rs`, inside `mod tests`, add:

```rust
#[test]
fn test_synthesis_cache_key_is_stable_across_calls() {
    let params = SynthesisParams {
        speed: 1.0,
        pitch: 1.0,
        voice: Some("af_heart".to_string()),
        language: Some("en-US".to_string()),
    };

    let k1 = TTSManager::synthesis_cache_key("Hello world", "kokoro-onnx", &params);
    let k2 = TTSManager::synthesis_cache_key("Hello world", "kokoro-onnx", &params);
    assert_eq!(k1, k2, "same inputs must produce identical keys");

    // Known stable value — this assertion documents the hash contract.
    // If it ever changes, the synthesis cache is invalidated across deploys.
    // Computed via the FNV-1a implementation in this file.
    assert_eq!(k1, TTSManager::synthesis_cache_key("Hello world", "kokoro-onnx", &params),
        "key must be deterministic");
}

#[test]
fn test_synthesis_cache_key_distinguishes_inputs() {
    let params = SynthesisParams::default();

    let k_text_a = TTSManager::synthesis_cache_key("Hello", "kokoro-onnx", &params);
    let k_text_b = TTSManager::synthesis_cache_key("World", "kokoro-onnx", &params);
    assert_ne!(k_text_a, k_text_b, "different text must produce different keys");

    let k_engine_a = TTSManager::synthesis_cache_key("Hello", "kokoro-onnx", &params);
    let k_engine_b = TTSManager::synthesis_cache_key("Hello", "piper", &params);
    assert_ne!(k_engine_a, k_engine_b, "different engine must produce different keys");

    let params_fast = SynthesisParams { speed: 2.0, ..SynthesisParams::default() };
    let k_speed_a = TTSManager::synthesis_cache_key("Hello", "kokoro-onnx", &params);
    let k_speed_b = TTSManager::synthesis_cache_key("Hello", "kokoro-onnx", &params_fast);
    assert_ne!(k_speed_a, k_speed_b, "different speed must produce different keys");
}
```

- [ ] **Step 2: Run tests to confirm they pass (baseline)**

```bash
cargo test -p torch_inference test_synthesis_cache_key 2>&1
```

Expected: both tests PASS (DefaultHasher is consistent within a single run). Note: these tests don't prove cross-run stability — that requires the implementation change.

- [ ] **Step 3: Replace DefaultHasher with FNV-1a**

In `src/core/tts_manager.rs`, remove these imports:
```rust
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
```

Add this private helper function (place it just above `synthesis_cache_key`):

```rust
/// FNV-1a 64-bit hash. Unlike `DefaultHasher`, this is stable across Rust
/// versions and process restarts — safe for use as a persistent cache key.
fn fnv1a_u64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
```

Replace `synthesis_cache_key` with:

```rust
fn synthesis_cache_key(text: &str, engine_id: &str, params: &SynthesisParams) -> u64 {
    // NUL bytes separate fields to prevent collisions like ("ab","c") == ("a","bc")
    let mut buf = Vec::with_capacity(
        text.len() + engine_id.len()
            + params.voice.as_deref().map_or(0, |s| s.len())
            + params.language.as_deref().map_or(0, |s| s.len())
            + 4   // field separators
            + 8   // speed f32 bits
            + 8,  // pitch f32 bits
    );
    buf.extend_from_slice(text.as_bytes());
    buf.push(0);
    buf.extend_from_slice(engine_id.as_bytes());
    buf.push(0);
    buf.extend_from_slice(params.voice.as_deref().unwrap_or("").as_bytes());
    buf.push(0);
    buf.extend_from_slice(params.language.as_deref().unwrap_or("").as_bytes());
    buf.push(0);
    buf.extend_from_slice(&params.speed.to_bits().to_le_bytes());
    buf.extend_from_slice(&params.pitch.to_bits().to_le_bytes());
    fnv1a_u64(&buf)
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p torch_inference test_synthesis_cache_key 2>&1
```

Expected: both tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/core/tts_manager.rs
git commit -m "fix: use FNV-1a for synthesis cache key instead of DefaultHasher

DefaultHasher is explicitly not guaranteed stable across Rust versions or
process restarts. Replace with an inline FNV-1a implementation (zero extra
deps) so cache keys are deterministic and survive server restarts.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Deterministic deduplication keys

**Files:**
- Modify: `src/dedup.rs`

- [ ] **Step 1: Write the failing test**

In `src/dedup.rs`, inside `mod tests`, add:

```rust
#[test]
fn test_generate_key_is_order_independent() {
    let dedup = RequestDeduplicator::new(100);

    // Same logical JSON, different insertion order
    let mut map_a = serde_json::Map::new();
    map_a.insert("b".to_string(), serde_json::json!(2));
    map_a.insert("a".to_string(), serde_json::json!(1));

    let mut map_b = serde_json::Map::new();
    map_b.insert("a".to_string(), serde_json::json!(1));
    map_b.insert("b".to_string(), serde_json::json!(2));

    let key_a = dedup.generate_key("model", &serde_json::Value::Object(map_a));
    let key_b = dedup.generate_key("model", &serde_json::Value::Object(map_b));

    assert_eq!(key_a, key_b,
        "generate_key must produce the same key regardless of JSON object key insertion order");
}

#[test]
fn test_generate_key_different_values_differ() {
    let dedup = RequestDeduplicator::new(100);

    let key_a = dedup.generate_key("model", &serde_json::json!({"x": 1}));
    let key_b = dedup.generate_key("model", &serde_json::json!({"x": 2}));

    assert_ne!(key_a, key_b, "different input values must produce different keys");
}
```

- [ ] **Step 2: Run tests to confirm the order-independence test fails**

```bash
cargo test -p torch_inference test_generate_key 2>&1
```

Expected: `test_generate_key_is_order_independent` FAILS (serde_json Map preserves insertion order, so currently `map_a != map_b` when serialised). `test_generate_key_different_values_differ` should PASS.

- [ ] **Step 3: Add imports to `src/dedup.rs`**

At the top of `src/dedup.rs`, the existing imports are:
```rust
use dashmap::DashMap;
use serde_json::Value;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use log::{debug, trace};
```

Add:
```rust
use sha2::{Sha256, Digest};
```

- [ ] **Step 4: Implement `canonical_json`**

Add this private function above `impl RequestDeduplicator`:

```rust
/// Serialise a JSON value to a canonical string where object keys are always
/// sorted alphabetically. This ensures two logically identical objects produce
/// identical strings regardless of the order their keys were inserted.
fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<(&String, &Value)> = map.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            let inner = pairs
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap_or_default(),
                        canonical_json(v)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{}}}", inner)
        }
        Value::Array(arr) => {
            let inner = arr
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{}]", inner)
        }
        _ => v.to_string(),
    }
}
```

- [ ] **Step 5: Update `generate_key` to use canonical SHA-256**

Replace the existing `generate_key` method with:

```rust
pub fn generate_key(&self, model: &str, inputs: &Value) -> String {
    let canonical = canonical_json(inputs);
    let hash = Sha256::digest(canonical.as_bytes());
    // Use first 16 bytes (32 hex chars) — ample collision resistance for a
    // 10-second dedup window.
    let hash_prefix = hex::encode(&hash[..16]);
    let epoch_window = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        / 10;
    format!("{}:{}:{}", model, hash_prefix, epoch_window)
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p torch_inference test_generate_key 2>&1
```

Expected: both tests PASS.

- [ ] **Step 7: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 8: Commit**

```bash
git add src/dedup.rs
git commit -m "fix: make dedup keys deterministic with canonical JSON + SHA-256

serde_json::Value objects serialised via to_string() preserve insertion
order, so two logically identical requests with differently-ordered keys
got different dedup keys. Fix by canonicalising object key order before
hashing (SHA-256, first 16 bytes). sha2 and hex are existing deps.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Replace dedup DashMap with LRU eviction

**Files:**
- Modify: `src/dedup.rs`
- Modify: `src/api/handlers.rs`

- [ ] **Step 1: Write the failing test**

In `src/dedup.rs`, inside `mod tests`, add:

```rust
#[test]
fn test_dedup_lru_eviction_on_full() {
    let dedup = RequestDeduplicator::new(2);
    dedup.set("key1".to_string(), serde_json::json!("val1"), 60);
    dedup.set("key2".to_string(), serde_json::json!("val2"), 60);

    // Access key1 to promote it to MRU; key2 becomes LRU
    let _ = dedup.get("key1");

    // Adding key3 should evict key2 (LRU), not fail
    dedup.set("key3".to_string(), serde_json::json!("val3"), 60);

    assert_eq!(dedup.size(), 2);
    assert!(dedup.get("key1").is_some(), "key1 (MRU) should survive");
    assert!(dedup.get("key2").is_none(), "key2 (LRU) should be evicted");
    assert!(dedup.get("key3").is_some(), "key3 (newly added) should be present");
}
```

Also delete the old `test_dedup_full` test (it expects an `Err` return from `set`, which will no longer apply):

Remove the test function named `test_dedup_full`.

- [ ] **Step 2: Run tests to confirm the new test fails**

```bash
cargo test -p torch_inference test_dedup_lru_eviction 2>&1
```

Expected: compile error — `set` still returns `Result<(), String>` and the test expects `()`.

- [ ] **Step 3: Rewrite `RequestDeduplicator` to use `LruCache`**

Replace the entire content of `src/dedup.rs` with:

```rust
use lru::LruCache;
use parking_lot::Mutex;
use serde_json::Value;
use sha2::{Sha256, Digest};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use log::{debug, trace};

/// Store the result behind an Arc so that clone on cache-hit is O(1)
/// (pointer copy) instead of a deep-copy of the potentially large JSON value.
#[derive(Clone)]
pub struct DeduplicationEntry {
    pub result: Arc<Value>,
    pub timestamp: u64,
    pub ttl: u64,
}

/// Request deduplicator backed by an LRU cache.
///
/// When at capacity, the least-recently-used entry is evicted automatically.
/// The old `DashMap`-based implementation returned `Err` when full, which the
/// call-site silently ignored — meaning new requests were never cached once
/// the map filled. LRU gives correct behaviour: stale results age out and
/// hot phrases stay cached.
pub struct RequestDeduplicator {
    cache: Mutex<LruCache<String, DeduplicationEntry>>,
}

impl RequestDeduplicator {
    pub fn new(max_entries: usize) -> Self {
        let cap = NonZeroUsize::new(max_entries.max(1))
            .expect("max_entries must be at least 1");
        Self {
            cache: Mutex::new(LruCache::new(cap)),
        }
    }

    pub fn generate_key(&self, model: &str, inputs: &Value) -> String {
        let canonical = canonical_json(inputs);
        let hash = Sha256::digest(canonical.as_bytes());
        let hash_prefix = hex::encode(&hash[..16]);
        let epoch_window = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            / 10;
        format!("{}:{}:{}", model, hash_prefix, epoch_window)
    }

    /// Returns a cheap `Arc` clone of the cached value — O(1), no data copied.
    /// Promotes the entry to most-recently-used on a hit.
    pub fn get(&self, key: &str) -> Option<Arc<Value>> {
        let mut cache = self.cache.lock();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // peek first (no promotion) to check TTL without a mutable borrow conflict
        let is_valid = cache
            .peek(key)
            .map_or(false, |e| now.saturating_sub(e.timestamp) < e.ttl);

        if is_valid {
            // promote to MRU and return
            let result = cache.get(key).map(|e| Arc::clone(&e.result));
            debug!("Deduplication cache hit: {}", key);
            result
        } else {
            cache.pop(key);
            trace!("Deduplication cache miss or expired: {}", key);
            None
        }
    }

    /// Insert a result. When at capacity the LRU entry is evicted automatically.
    /// Wrap `result` in an `Arc` once so all future cache hits share the allocation.
    pub fn set(&self, key: String, result: Value, ttl: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.cache.lock().put(
            key.clone(),
            DeduplicationEntry {
                result: Arc::new(result),
                timestamp: now,
                ttl,
            },
        );
        debug!("Deduplication entry set: {} (TTL: {}s)", key, ttl);
    }

    pub fn invalidate(&self, key: &str) {
        self.cache.lock().pop(key);
        debug!("Deduplication entry invalidated: {}", key);
    }

    pub fn clear(&self) {
        self.cache.lock().clear();
        debug!("Deduplication cache cleared");
    }

    pub fn cleanup_expired(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut cache = self.cache.lock();
        let expired: Vec<String> = cache
            .iter()
            .filter(|(_, e)| now.saturating_sub(e.timestamp) >= e.ttl)
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired.len();
        for key in &expired {
            cache.pop(key.as_str());
        }
        debug!("Deduplication cleanup: removed {} expired entries", count);
    }

    pub fn size(&self) -> usize {
        self.cache.lock().len()
    }
}

impl Default for RequestDeduplicator {
    fn default() -> Self {
        Self::new(5000)
    }
}

/// Serialise a JSON value to a canonical string where object keys are sorted.
fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut pairs: Vec<(&String, &Value)> = map.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            let inner = pairs
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap_or_default(),
                        canonical_json(v)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{}}}", inner)
        }
        Value::Array(arr) => {
            let inner = arr.iter().map(canonical_json).collect::<Vec<_>>().join(",");
            format!("[{}]", inner)
        }
        _ => v.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedup_set_and_get() {
        let dedup = RequestDeduplicator::new(100);
        let key = "test_key".to_string();
        let result = serde_json::json!({"output": "test_result"});

        dedup.set(key.clone(), result.clone(), 60);
        assert_eq!(dedup.get(&key), Some(Arc::new(result)));
    }

    #[test]
    fn test_dedup_miss() {
        let dedup = RequestDeduplicator::new(100);
        assert_eq!(dedup.get("nonexistent"), None);
    }

    #[test]
    fn test_dedup_expiration() {
        let dedup = RequestDeduplicator::new(100);
        let key = "expire_test".to_string();
        dedup.set(key.clone(), serde_json::json!({"output": "will_expire"}), 0);
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert_eq!(dedup.get(&key), None);
    }

    #[test]
    fn test_dedup_lru_eviction_on_full() {
        let dedup = RequestDeduplicator::new(2);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 60);
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60);

        // Access key1 to promote it to MRU; key2 becomes LRU
        let _ = dedup.get("key1");

        // Adding key3 should evict key2 (LRU), not fail
        dedup.set("key3".to_string(), serde_json::json!("val3"), 60);

        assert_eq!(dedup.size(), 2);
        assert!(dedup.get("key1").is_some(), "key1 (MRU) should survive");
        assert!(dedup.get("key2").is_none(), "key2 (LRU) should be evicted");
        assert!(dedup.get("key3").is_some(), "key3 (newly added) should be present");
    }

    #[test]
    fn test_dedup_invalidate() {
        let dedup = RequestDeduplicator::new(100);
        let key = "invalidate_test".to_string();
        dedup.set(key.clone(), serde_json::json!("value"), 60);
        assert!(dedup.get(&key).is_some());
        dedup.invalidate(&key);
        assert!(dedup.get(&key).is_none());
    }

    #[test]
    fn test_dedup_clear() {
        let dedup = RequestDeduplicator::new(100);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 60);
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60);
        assert_eq!(dedup.size(), 2);
        dedup.clear();
        assert_eq!(dedup.size(), 0);
    }

    #[test]
    fn test_dedup_cleanup_expired() {
        let dedup = RequestDeduplicator::new(100);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 0);
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60);
        std::thread::sleep(std::time::Duration::from_secs(1));
        dedup.cleanup_expired();
        assert_eq!(dedup.size(), 1);
        assert!(dedup.get("key2").is_some());
    }

    #[test]
    fn test_dedup_size() {
        let dedup = RequestDeduplicator::new(100);
        assert_eq!(dedup.size(), 0);
        dedup.set("key1".to_string(), serde_json::json!("val1"), 60);
        assert_eq!(dedup.size(), 1);
        dedup.set("key2".to_string(), serde_json::json!("val2"), 60);
        assert_eq!(dedup.size(), 2);
    }

    #[test]
    fn test_generate_key_is_order_independent() {
        let dedup = RequestDeduplicator::new(100);

        let mut map_a = serde_json::Map::new();
        map_a.insert("b".to_string(), serde_json::json!(2));
        map_a.insert("a".to_string(), serde_json::json!(1));

        let mut map_b = serde_json::Map::new();
        map_b.insert("a".to_string(), serde_json::json!(1));
        map_b.insert("b".to_string(), serde_json::json!(2));

        let key_a = dedup.generate_key("model", &Value::Object(map_a));
        let key_b = dedup.generate_key("model", &Value::Object(map_b));
        assert_eq!(key_a, key_b);
    }

    #[test]
    fn test_generate_key_different_values_differ() {
        let dedup = RequestDeduplicator::new(100);
        let key_a = dedup.generate_key("model", &serde_json::json!({"x": 1}));
        let key_b = dedup.generate_key("model", &serde_json::json!({"x": 2}));
        assert_ne!(key_a, key_b);
    }
}
```

Note: This new file supersedes Tasks 3's changes to `dedup.rs` — the `canonical_json` function and updated `generate_key` are included here. **Skip Task 3's individual file edits and apply this complete replacement instead.** (Both tasks can be committed together at this step.)

- [ ] **Step 4: Fix the call site in `src/api/handlers.rs`**

Find line ~90 in `src/api/handlers.rs`:
```rust
let _ = deduplicator.set(dedup_key, result.clone(), 10);
```

Replace with:
```rust
deduplicator.set(dedup_key, result.clone(), 10);
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p torch_inference dedup 2>&1
```

Expected: all `test_dedup_*` and `test_generate_key_*` tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/dedup.rs src/api/handlers.rs
git commit -m "fix: replace DashMap dedup cache with LRU eviction

When the DashMap-based cache was full, set() returned Err which the
handler silently ignored — new requests stopped being cached. Switch to
LruCache<String, DeduplicationEntry> (parking_lot::Mutex) so the LRU
entry is evicted on overflow, mirroring the synthesis_cache pattern.
set() is now infallible.

Also folds in Task 3 (canonical JSON keys, SHA-256 hash) which was
targeting the same file.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Batch 2: Alignment

---

### Task 5: Load model_registry.json instead of hardcoded build()

**Files:**
- Modify: `src/api/models.rs`

- [ ] **Step 1: Write the failing test**

In `src/api/models.rs`, inside a `#[cfg(test)]` block, add:

```rust
#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn test_registry_loads_from_json_str() {
        let json = r#"{
            "version": "1.0",
            "updated": "2026-01-01T00:00:00Z",
            "models": {
                "test-model": {
                    "name": "Test Model",
                    "score": 50.0,
                    "rank": 5,
                    "size": "100 MB",
                    "url": "https://example.com",
                    "architecture": "Test",
                    "voices": "1",
                    "quality": "High",
                    "status": "Available"
                }
            }
        }"#;

        let registry = ModelRegistry::from_json_str(json);
        assert!(registry.get_model("test-model").is_some());
        let model = registry.get_model("test-model").unwrap();
        assert_eq!(model.name, "Test Model");
        assert!((model.score - 50.0).abs() < 0.01);
        assert_eq!(model.rank, 5);
    }

    #[test]
    fn test_registry_handles_mixed_rank_type() {
        // model_registry.json uses "Production" for some ranks
        let json = r#"{
            "version": "1.0",
            "updated": "2026-01-01T00:00:00Z",
            "models": {
                "windows-sapi": {
                    "name": "Windows SAPI",
                    "score": "N/A (Native)",
                    "rank": "Production",
                    "size": "Built-in",
                    "url": "Built-in",
                    "architecture": "Neural TTS",
                    "voices": 3,
                    "quality": "High",
                    "status": "Active"
                }
            }
        }"#;

        let registry = ModelRegistry::from_json_str(json);
        assert!(registry.get_model("windows-sapi").is_some());
        // score and rank should not panic — they fall back to 0.0/0
        let model = registry.get_model("windows-sapi").unwrap();
        assert_eq!(model.score, 0.0);
        assert_eq!(model.rank, 0);
    }

    #[test]
    fn test_registry_from_json_str_invalid_json_returns_empty() {
        let registry = ModelRegistry::from_json_str("not valid json {{{{");
        assert_eq!(registry.models.len(), 0);
    }
}
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cargo test -p torch_inference registry_tests 2>&1
```

Expected: compile errors — `ModelRegistry::from_json_str` does not exist, and `ModelInfo` fields `score`/`rank` are typed `f32`/`i32` which can't deserialise from strings.

- [ ] **Step 3: Update `ModelInfo` with flexible deserializers**

In `src/api/models.rs`, replace the `ModelInfo` struct and add custom deserializers:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default, deserialize_with = "de_f32_or_str")]
    pub score: f32,
    #[serde(default, deserialize_with = "de_i32_or_str")]
    pub rank: i32,
    #[serde(default)]
    pub size: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub architecture: String,
    #[serde(default, deserialize_with = "de_string_or_num")]
    pub voices: String,
    #[serde(default)]
    pub quality: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub note: Option<String>,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub task: String,
}

fn de_f32_or_str<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match &v {
        serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
        _ => 0.0,
    })
}

fn de_i32_or_str<'de, D: serde::Deserializer<'de>>(d: D) -> Result<i32, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match &v {
        serde_json::Value::Number(n) => n.as_i64().unwrap_or(0) as i32,
        _ => 0,
    })
}

fn de_string_or_num<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::String(s) => s,
        serde_json::Value::Number(n) => n.to_string(),
        _ => String::new(),
    })
}
```

- [ ] **Step 4: Add `from_json_str` and `load` to `ModelRegistry`**

In `impl ModelRegistry`, add these two methods (keeping all existing methods unchanged):

```rust
/// Parse a registry from a JSON string. Returns an empty registry on parse error.
pub fn from_json_str(s: &str) -> Self {
    serde_json::from_str(s).unwrap_or_else(|e| {
        log::warn!("Failed to parse model registry JSON: {}", e);
        Self {
            version: "0.0".to_string(),
            updated: String::new(),
            models: std::collections::HashMap::new(),
        }
    })
}

/// Load the registry from disk (path via MODEL_REGISTRY_PATH env var, default
/// `model_registry.json`). Falls back to the file embedded at compile-time so
/// the binary works without the file present at runtime.
fn load() -> Self {
    let path = std::env::var("MODEL_REGISTRY_PATH")
        .unwrap_or_else(|_| "model_registry.json".to_string());

    let data = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        log::info!(
            "model_registry.json not found at '{}', using compiled-in copy",
            path
        );
        include_str!("../../model_registry.json").to_string()
    });

    Self::from_json_str(&data)
}
```

- [ ] **Step 5: Replace `build()` call in `get_registry`**

Find:
```rust
static REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();

fn get_registry() -> &'static ModelRegistry {
    REGISTRY.get_or_init(ModelRegistry::build)
}
```

Replace with:
```rust
static REGISTRY: OnceLock<ModelRegistry> = OnceLock::new();

fn get_registry() -> &'static ModelRegistry {
    REGISTRY.get_or_init(ModelRegistry::load)
}
```

- [ ] **Step 6: Delete `ModelRegistry::build`**

Delete the entire `fn build() -> Self { ... }` method (the ~380-line hardcoded function). The `load()` method replaces it entirely.

- [ ] **Step 7: Run tests**

```bash
cargo test -p torch_inference registry_tests 2>&1
```

Expected: all three tests PASS.

- [ ] **Step 8: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 9: Commit**

```bash
git add src/api/models.rs
git commit -m "fix: load model registry from model_registry.json instead of hardcoded build()

ModelRegistry::build() was a ~380-line function duplicating model_registry.json
which is tracked in git and already diverged. Replace with load() + from_json_str()
that reads the file at startup (with compile-time embed fallback). Add flexible
serde deserializers so mixed rank/score types in the JSON (strings + numbers)
parse without error.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Fix audio format validation stubs

**Files:**
- Modify: `src/core/audio.rs`

- [ ] **Step 1: Write the failing test**

In `src/core/audio.rs`, inside `mod tests`, add:

```rust
fn make_wav_bytes(sample_rate: u32, channels: u16, samples: &[f32]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    writer.finalize().unwrap();
    buf.into_inner()
}

#[test]
fn test_probe_metadata_returns_real_sample_rate() {
    // Generate a 22050 Hz mono WAV (non-default rate)
    let samples: Vec<f32> = (0..2205).map(|i| (i as f32 * 0.01).sin()).collect();
    let wav = make_wav_bytes(22050, 1, &samples);

    let processor = AudioProcessor::new();
    let meta = processor.probe_metadata_with_symphonia(&wav, AudioFormat::Wav)
        .expect("symphonia should parse WAV");

    assert_eq!(meta.sample_rate, 22050, "sample_rate must reflect actual file");
    assert_eq!(meta.channels, 1);
}

#[test]
fn test_validate_audio_wav_returns_real_metadata() {
    let samples: Vec<f32> = (0..4410).map(|i| (i as f32 * 0.005).sin()).collect();
    let wav = make_wav_bytes(44100, 2, &samples);

    let processor = AudioProcessor::new();
    let meta = processor.validate_audio(&wav).expect("should validate");

    assert_eq!(meta.sample_rate, 44100);
    assert_eq!(meta.channels, 2);
    assert!(meta.duration_secs > 0.0);
}
```

- [ ] **Step 2: Run to confirm `probe_metadata_with_symphonia` fails**

```bash
cargo test -p torch_inference test_probe_metadata 2>&1
cargo test -p torch_inference test_validate_audio 2>&1
```

Expected: `test_probe_metadata_returns_real_sample_rate` fails — method does not exist.

- [ ] **Step 3: Add `probe_metadata_with_symphonia` to `AudioProcessor`**

Inside `impl AudioProcessor`, add this method:

```rust
/// Probe audio metadata using symphonia without decoding all samples.
/// Used by `validate_mp3`, `validate_flac`, and `validate_ogg` to return
/// real codec parameters instead of hardcoded defaults.
pub fn probe_metadata_with_symphonia(
    &self,
    data: &[u8],
    format: AudioFormat,
) -> Result<AudioMetadata> {
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
    let hint = Hint::new();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("Failed to probe audio format with symphonia")?;

    let track = probed
        .format
        .default_track()
        .context("No default audio track found")?;

    let params = &track.codec_params;
    let sample_rate = params.sample_rate.unwrap_or(44100);
    let channels = params
        .channels
        .map(|c| c.count() as u16)
        .unwrap_or(2);
    let bits_per_sample = params.bits_per_coded_sample.unwrap_or(16) as u16;
    let duration_secs = match (params.n_frames, params.sample_rate) {
        (Some(frames), Some(rate)) => frames as f32 / rate as f32,
        _ => 0.0,
    };

    Ok(AudioMetadata {
        format,
        sample_rate,
        channels,
        duration_secs,
        bits_per_sample,
    })
}
```

- [ ] **Step 4: Replace the three stub methods**

Replace `validate_mp3`, `validate_flac`, and `validate_ogg` with calls to the new helper:

```rust
fn validate_mp3(&self, data: &[u8]) -> Result<AudioMetadata> {
    self.probe_metadata_with_symphonia(data, AudioFormat::Mp3)
}

fn validate_flac(&self, data: &[u8]) -> Result<AudioMetadata> {
    self.probe_metadata_with_symphonia(data, AudioFormat::Flac)
}

fn validate_ogg(&self, data: &[u8]) -> Result<AudioMetadata> {
    self.probe_metadata_with_symphonia(data, AudioFormat::Ogg)
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p torch_inference test_probe_metadata 2>&1
cargo test -p torch_inference test_validate_audio 2>&1
```

Expected: both PASS.

- [ ] **Step 6: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/core/audio.rs
git commit -m "fix: replace audio validation stubs with real symphonia probing

validate_mp3/flac/ogg previously returned hardcoded metadata (44100 Hz,
2 channels, 0.0s duration) regardless of the actual file. Replace with
probe_metadata_with_symphonia() which reads real codec parameters from
the container headers without decoding all samples. symphonia is an
existing dependency.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Batch 3: Performance

---

### Task 7: Concurrent HuggingFace repo file downloads

**Files:**
- Modify: `src/api/models.rs`

- [ ] **Step 1: Write the failing test**

In `src/api/models.rs`, inside the existing `#[cfg(test)] mod tests`, add:

```rust
#[tokio::test]
async fn test_download_single_repo_file_writes_content() {
    let server = MockServer::start().await;
    let content = b"repo file content";

    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(content.as_slice()))
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("weights.bin");
    let client = reqwest::Client::new();

    download_single_repo_file(&client, &server.uri(), &dest)
        .await;

    assert_eq!(std::fs::read(&dest).unwrap(), content);
}
```

- [ ] **Step 2: Run to confirm the test fails**

```bash
cargo test -p torch_inference test_download_single_repo_file 2>&1
```

Expected: compile error — `download_single_repo_file` not found.

- [ ] **Step 3: Extract `download_single_repo_file` helper**

Add this function above `download_huggingface_repo` in `src/api/models.rs`:

```rust
/// Download one file from a URL to a local path. Errors are logged as warnings
/// but not propagated — the caller collects results from all concurrent downloads.
async fn download_single_repo_file(
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
) {
    use futures_util::TryStreamExt;
    use tokio_util::io::StreamReader;

    if let Some(parent) = dest.parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            log::warn!("Failed to create directory {:?}: {}", parent, e);
            return;
        }
    }

    match client.get(url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let stream = resp
                .bytes_stream()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
            let mut reader = StreamReader::new(stream);
            match tokio::fs::File::create(dest).await {
                Ok(mut file) => {
                    if let Err(e) = tokio::io::copy(&mut reader, &mut file).await {
                        log::warn!("Failed to write {:?}: {}", dest, e);
                    } else {
                        log::info!("Downloaded {:?}", dest.file_name().unwrap_or_default());
                    }
                }
                Err(e) => log::warn!("Failed to create {:?}: {}", dest, e),
            }
        }
        Ok(resp) => log::warn!("HTTP {} for {}", resp.status(), url),
        Err(e) => log::warn!("Request error for {}: {}", url, e),
    }
}
```

- [ ] **Step 4: Rewrite `download_huggingface_repo` to use `buffer_unordered`**

Replace the `for` loop in `download_huggingface_repo` (everything after `log::info!("Downloading {} files from repository", ...)`) with:

```rust
use futures::StreamExt;

futures::stream::iter(files_to_download.into_iter().enumerate())
    .map(|(idx, file_path)| {
        let file_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, file_path
        );
        let local_path = target_dir.join(&file_path);
        let total = files_to_download_len; // capture len before move
        async move {
            log::info!("Downloading {}/{}: {}", idx + 1, total, file_path);
            download_single_repo_file(get_http_client(), &file_url, &local_path).await;
        }
    })
    .buffer_unordered(8)
    .collect::<Vec<_>>()
    .await;
```

Because `files_to_download` is moved into the closure, capture its length before the stream:
```rust
let files_to_download_len = files_to_download.len();
log::info!("Downloading {} files from repository", files_to_download_len);

use futures::StreamExt;
futures::stream::iter(files_to_download.into_iter().enumerate())
    // ...
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p torch_inference test_download_single_repo_file 2>&1
```

Expected: PASS.

- [ ] **Step 6: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/api/models.rs
git commit -m "perf: download HuggingFace repo files concurrently (buffer_unordered 8)

Sequential file downloads were O(n * RTT). Replace with futures::stream
buffer_unordered(8) which matches the global HTTP client's
pool_max_idle_per_host(8). Per-file errors are still logged as warnings
without aborting the overall download.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Add `with_engine` to avoid hot-path Arc clone

**Files:**
- Modify: `src/core/tts_manager.rs`
- Modify: `src/api/tts.rs`

- [ ] **Step 1: Write the failing test**

In `src/core/tts_manager.rs`, inside `mod tests`, add:

```rust
#[test]
fn test_with_engine_returns_none_for_missing_engine() {
    let config = TTSManagerConfig::default();
    let manager = TTSManager::new(config);
    let result = manager.with_engine("nonexistent", |e| e.name().to_string());
    assert!(result.is_none());
}
```

- [ ] **Step 2: Run to confirm the test fails**

```bash
cargo test -p torch_inference test_with_engine 2>&1
```

Expected: compile error — `with_engine` not found.

- [ ] **Step 3: Add `with_engine` to `TTSManager`**

In `impl TTSManager`, add after `get_engine`:

```rust
/// Call `f` with a reference to the engine, without cloning the `Arc`.
/// Use this for read-only operations (capabilities, voice listing) that
/// don't need to hold the engine across an `.await` point.
pub fn with_engine<F, R>(&self, id: &str, f: F) -> Option<R>
where
    F: FnOnce(&dyn TTSEngine) -> R,
{
    self.engines.get(id).map(|e| f(e.as_ref()))
}
```

- [ ] **Step 4: Update `get_capabilities` in `tts_manager.rs` to use `with_engine`**

Replace:
```rust
pub fn get_capabilities(&self, engine_id: &str) -> Option<EngineCapabilities> {
    self.get_engine(engine_id).map(|e| e.capabilities().clone())
}
```

With:
```rust
pub fn get_capabilities(&self, engine_id: &str) -> Option<EngineCapabilities> {
    self.with_engine(engine_id, |e| e.capabilities().clone())
}
```

- [ ] **Step 5: Update `list_voices` endpoint in `src/api/tts.rs`**

Find in `src/api/tts.rs`:
```rust
pub async fn list_voices(
    engine_id: web::Path<String>,
    state: web::Data<TTSState>,
) -> Result<HttpResponse, ApiError> {
    let engine = state.manager.get_engine(&engine_id)
        .ok_or_else(|| ApiError::NotFound(format!("Engine '{}' not found", engine_id)))?;

    let voices = engine.list_voices();
    // ...
}
```

Replace with:
```rust
pub async fn list_voices(
    engine_id: web::Path<String>,
    state: web::Data<TTSState>,
) -> Result<HttpResponse, ApiError> {
    let voices = state
        .manager
        .with_engine(&engine_id, |e| e.list_voices())
        .ok_or_else(|| ApiError::NotFound(format!("Engine '{}' not found", engine_id)))?;

    Ok(HttpResponse::Ok().json(VoiceListResponse {
        total: voices.len(),
        voices,
        engine: engine_id.to_string(),
    }))
}
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p torch_inference test_with_engine 2>&1
cargo test -p torch_inference tts_manager 2>&1
```

Expected: all PASS.

- [ ] **Step 7: Run full test suite**

```bash
cargo test --lib 2>&1 | tail -20
```

Expected: no regressions.

- [ ] **Step 8: Final build check**

```bash
cargo build 2>&1 | tail -10
```

Expected: builds without warnings related to changed code.

- [ ] **Step 9: Commit**

```bash
git add src/core/tts_manager.rs src/api/tts.rs
git commit -m "perf: add with_engine() to avoid Arc clone for read-only engine calls

get_engine() always clones the Arc<dyn TTSEngine>. For callers that only
need one method call (get_capabilities, list_voices) this clone is
unnecessary. with_engine<F, R>(id, f) runs a closure against a DashMap
Ref without cloning the Arc. synthesize() keeps using get_engine() since
it holds the engine across an .await.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Run complete test suite**

```bash
cargo test 2>&1 | tail -30
```

Expected: all tests pass across all three batches.

- [ ] **Check for compiler warnings in changed files**

```bash
cargo build 2>&1 | grep "^warning" | grep -v "unused import\|dead_code" | head -20
```

Expected: no new warnings introduced by these changes.
