# Test Coverage Design — Kolosal Inference Server

**Date:** 2026-04-13
**Goal:** Reach ≥ 95% line coverage (tarpaulin) with all tests passing; add Rust HTTP integration tests.

---

## Context

- 2,382 unit tests; 2,332 pass, 18 fail, 32 ignored at baseline
- No Rust integration test crate exists (`tests/` contains only JS/Playwright/Jest suites)
- Tarpaulin v0.35.2 installed; no `.tarpaulin.toml` yet
- ~560 structurally uncoverable lines (ORT internals, Windows-only code, tracing macros)
- Active endpoints in scope: TTS `/tts/*`, STT `/stt/*` `/audio/*`, classify `/classify/*`, detect `/detect/*` `/yolo/*`, health/metrics `/health`, `/system/info`, `/metrics`, `/performance`
- LLM proxy (`/llm/*`, `src/api/llm_proxy.rs`) is explicitly out of scope — excluded from coverage

---

## Approach

**Option C (approved):** Fix ORT env + add HTTP integration tests + tarpaulin exclusions (whole-file for platform code, inline markers for individual branches).

---

## Section 1: Fix the 18 Failing Tests

### Root cause
Tests that initialise ORT panic because `libonnxruntime.dylib` is not on the dynamic linker search path when test binaries run. The server pre-loads it via `ORT_DYLIB_PATH` at startup but tests don't inherit this.

### Fix
Add `.cargo/config.toml`:

```toml
[env]
ORT_DYLIB_PATH = "/opt/homebrew/lib/libonnxruntime.dylib"
DYLD_LIBRARY_PATH = "/opt/homebrew/lib"
```

### Outlier
`api::models::download_coverage_tests::test_download_model_async_empty_model_type_tts_default` asserts `Path::new("models/tts/empty-type-tts-model").exists()` against a path that never exists. Fix: correct the assertion to test the error/absent branch instead.

---

## Section 2: Tarpaulin Configuration

Add `.tarpaulin.toml` at repo root.

### Whole-file exclusions (structurally unreachable)
| File | Reason |
|------|--------|
| `src/core/windows_sapi_tts.rs` | Windows SAPI only, never compiles on macOS/Linux |
| `src/bin/provider_comparison.rs` | Standalone binary, not part of lib surface |
| `src/api/llm_proxy.rs` | Explicitly out of scope per project guidelines |

### Inline exclusions (`// tarpaulin::skip` on individual items)
- ORT `init_from_dylib` panic paths in `kokoro_onnx.rs`, `ort_classify.rs`, `ort_yolo.rs`
- `tracing::instrument` macro-generated spans (compiler-generated, not real branches)
- `#[cfg(target_os = "windows")]` blocks inside cross-platform files

### Config
```toml
[default]
exclude-files = [
  "src/core/windows_sapi_tts.rs",
  "src/bin/provider_comparison.rs",
  "src/api/llm_proxy.rs",
]
fail-under = 95
out = ["Html", "Lcov"]
output-dir = "coverage/"
features = "default"
timeout = "300s"
```

`fail-under = 95` makes `cargo tarpaulin` exit non-zero if coverage drops below 95% — CI gate.

---

## Section 3: Rust Integration Tests

New `tests/integration/` directory. Each file uses `actix_web::test::init_service` — full app stack, no port binding, fast.

### Layout
```
tests/integration/
  mod.rs          — shared helpers: build_test_app(), default_config()
  health.rs       — GET /health, /health/live, /health/ready → 200 + body shape
  system.rs       — GET /system/info, /metrics, /performance → 200 + body shape
  tts.rs          — POST /tts/stream with mock TTSManager → 200 WAV, 400 bad req
  classify.rs     — POST /classify/batch → 400 no-image, JSON error shape
  detect.rs       — POST /detect → 400 no-image, JSON error shape
  audio.rs        — POST /stt/transcribe → 400 bad input, JSON error shape
  middleware.rs   — correlation ID header present, rate-limit 429, request log
  auth.rs         — API key rejection 401, valid key passes through
  error.rs        — 404 unknown route, 405 wrong method, error body shape
```

### Design decisions
- ORT-dependent endpoints (classify, detect) are tested for **error shapes only** — confirms 4xx responses are `{"error": "..."}` JSON, not panics; no real model needed
- TTS handler gets a mock `TTSManager` injected via `web::Data` — exercises real serialization and streaming logic without loading model files
- All tests inherit ORT env vars from `.cargo/config.toml` (Section 1)

---

## Section 4: Coverage Gap Filling

After Sections 1–3 are complete, run `cargo tarpaulin` to generate an LCOV report. Read uncovered lines and add targeted unit tests only for reachable, non-excluded lines. Expected gap modules:

| Module | Gap area |
|--------|----------|
| `src/resilience/` | Half-open state transitions, bulkhead capacity limits, retry exhaustion, token refill |
| `src/security/` | Oversized image/text rejection, null-stripping, invalid JWT claims |
| `src/telemetry/` | Init paths, counter increments, label formatting |
| `src/middleware/` | Rate-limit reject path, missing header generation |
| `src/postprocess/` | NMS edge cases (zero boxes, all-overlap), empty batch, clamp boundaries |
| `src/core/model_cache.rs`, `src/cache.rs`, `src/dedup.rs` | Eviction, TTL expiry, concurrent dedup resolution |

**Rule:** No speculative tests. Only add tests for lines shown as uncovered in the tarpaulin report.

---

## Success Criteria

- `cargo test` exits 0 (all tests pass, none failing)
- `cargo tarpaulin` exits 0 with ≥ 95% line coverage
- LCOV report generated at `coverage/lcov.info`
- HTML report generated at `coverage/tarpaulin-report.html`
- Integration tests in `tests/integration/` cover all active HTTP endpoints
