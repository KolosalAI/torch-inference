# Coverage Improvement Design

**Date:** 2026-03-29
**Current coverage:** 87.9% (4,959 / 5,643 lines)
**Target:** ~93% (+~5%)
**Approach:** Inline `#[cfg(test)]` unit tests, no new runtime dependencies

---

## Scope

Write tests for all reachable-but-untested code paths. Exclude:
- ORT session creation / ONNX inference (requires real model files + hardware)
- Real CUDA/GPU execution
- Windows SAPI (platform-specific, macOS CI)
- Network downloads (no wiremock in scope for option A)

---

## Architecture

All tests are added inline to each source file's existing `#[cfg(test)]` block, consistent with the codebase convention. No new test files. One dev-dependency (`tempfile`) may be added if not already present.

One minor production refactor: `telemetry/logger.rs` extracts the body of `format_log_record` into a `pub(crate) fn format_log_record_inner(buf: &mut dyn Write, level, target, args)` so tests can call it directly with a `Vec<u8>` writer, bypassing the untestable `env_logger::fmt::Formatter`.

---

## Batches

### Batch 1 — API error paths (~+1.5%)

**Files:** `api/models.rs`, `api/yolo.rs`, `api/classification.rs`, `api/inference.rs`, `api/audio.rs`, `api/metrics_endpoint.rs`

**Strategy:** Use the existing Axum test pattern (`tower::ServiceExt::oneshot`) with a minimal `AppState`. Trigger uncovered branches by constructing requests that cause: model not found (404), invalid/missing fields (422/400), engine errors (500).

**Uncovered lines targeted:**
- `api/models.rs`: lines 123, 207–208, 257–313, 360–368 — error branches in list/get/delete handlers
- `api/yolo.rs`: lines 79–115 — error path when model unavailable
- `api/classification.rs`: lines 44–48, 95–108 — missing model / engine failure branches
- `api/inference.rs`: lines 159–202 — error returns from inference dispatch
- `api/audio.rs`: lines 209–216, 260 — audio processing error paths
- `api/metrics_endpoint.rs`: lines 14–16 — metrics not-yet-initialized path

---

### Batch 2 — Filesystem paths (~+1.5%)

**Files:** `models/download.rs`, `core/kokoro_onnx.rs`, `core/tts_manager.rs`, `models/onnx_loader.rs`, `models/manager.rs`, `models/registry.rs`, `core/piper_tts.rs`, `core/image_classifier.rs`

**Strategy:** Use `tempfile::tempdir()` to create controlled filesystem state. For constructors that require model files, point at nonexistent paths to exercise error branches. For scan/directory-traversal code, populate temp dirs with fake metadata.

**Uncovered lines targeted:**
- `models/download.rs`: lines 95, 111–112, 195–256 — `scan_cache()` traversal, `calculate_dir_size()`, `download_execute()` dispatch arms (TorchHub/Local bail! branches)
- `core/kokoro_onnx.rs`: lines 53–140 — `KokoroOnnxEngine::new()` when model dir does not exist
- `core/tts_manager.rs`: lines 170–258 — `initialize_defaults()` graceful failure paths, `synthesize()` cache-miss + engine-error paths
- `models/onnx_loader.rs`: lines 153–192 — `load_model()` path-not-found and TensorRT config branches (exercised via config-only paths, not actual ORT session creation)
- `models/manager.rs`: lines 173–196, 261–270 — load/unload error paths
- `models/registry.rs`: lines 178, 224–239, 265–266 — registry miss / update error paths
- `core/piper_tts.rs`: lines 134–179 — synthesis error paths
- `core/image_classifier.rs`: lines 79–267 — classifier error paths

---

### Batch 3 — Env-var / OS detection (~+1.0%)

**Files:** `core/torch_autodetect.rs`, `core/gpu.rs`

**Strategy:** Env var mutation tests. Set `CUDA_PATH` to a temp dir path to exercise the "detected" branch; unset it and ensure the fallback branch runs. Use a per-test mutex (`std::sync::Mutex` in a `OnceLock`) to prevent test parallelism from corrupting env state. On macOS, test the Metal detection path by checking `/System/Library/Frameworks/Metal.framework` existence (already present in CI).

**Uncovered lines targeted:**
- `torch_autodetect.rs`: lines 68–76 (CUDA_PATH branch), 153–157 (versioned CUDA_PATH vars), 253, 322–345 (ORT path detection, nvidia-smi branch)
- `gpu.rs`: lines 77, 114–118, 244–249, 311–414 — GPU capability detection branches

---

### Batch 4 — Telemetry & middleware (~+0.5%)

**Files:** `telemetry/logger.rs`, `telemetry/structured_logging.rs`, `telemetry/prometheus.rs`, `middleware/request_logger.rs`, `middleware/rate_limit.rs`

**Strategy:**
- `logger.rs`: Extract `format_log_record` body to `format_log_record_inner(buf: &mut dyn Write, level, target, args)`. Tests call it with `Vec<u8>` as writer, asserting output contains ANSI codes and the message.
- `structured_logging.rs`: Call the uncovered serialization/formatting functions directly.
- `prometheus.rs`: Lines 225–226 are the "already-registered" guard branch — register twice in a test.
- `middleware/request_logger.rs`: Wrap a dummy `axum` handler with the middleware using `tower::ServiceBuilder` and fire requests that exercise logging branches (lines 65–98).
- `middleware/rate_limit.rs`: Lines 59–61 are the rate-limit-exceeded path — fire enough requests to exhaust the bucket.

---

### Batch 5 — Small gaps, 1–3 lines each (~+0.5%)

**Files:** `cache.rs`, `monitor.rs`, `batch.rs`, `model_pool.rs`, `dedup.rs`, `tensor_pool.rs`, `guard.rs`, `worker_pool.rs`, `resilience/circuit_breaker.rs`, `resilience/retry.rs`, `resilience/per_model_breaker.rs`, `core/audio.rs`, `core/yolo.rs`, `core/image_security.rs`, `core/tts_engine.rs`, `core/kokoro_tts.rs`, `security/sanitizer.rs`, `api/image.rs`, `api/performance.rs`, `api/model_download.rs`, `core/audio_models.rs`, `core/g2p_misaki.rs`, `models/pytorch_loader.rs`, `config.rs`

**Strategy:** Direct targeted unit tests for each uncovered line. Most are: LRU eviction paths, concurrent access edge cases, error-return arms, or rarely-hit config branches. Each test is 5–15 lines.

---

## Production code changes

Only one: `telemetry/logger.rs` — extract `format_log_record` body into `pub(crate) format_log_record_inner`. The outer function becomes a 1-line wrapper. No behavioral change.

---

## Out of scope

- ORT session construction / ONNX inference paths in `kokoro_onnx.rs` synthesis methods
- `models/onnx_loader.rs` actual execution provider attachment (CUDA/TensorRT)
- `core/gpu.rs` CUDA memory queries
- `core/windows_sapi_tts.rs` (Windows-only, excluded on macOS CI)
- `core/istftnet_vocoder.rs`, `core/styletts2_model.rs` (0/4 lines each — ORT-dependent synthesis)
- Network download paths in `models/download.rs` (HuggingFace/URL fetchers)
