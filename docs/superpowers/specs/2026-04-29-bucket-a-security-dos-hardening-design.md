# Bucket A — Security & DoS Hardening — Design

Phase 3 sub-project from the 2026-04-29 product audit. Closes the highest-priority security and DoS gaps surfaced in the audit.

## Goal

Three concrete outcomes:

1. The default `Config` cannot ship to production with `auth.enabled = true` and the placeholder `jwt_secret`.
2. Inference endpoints reject oversized requests with HTTP 413 *before* any decode/allocation, instead of OOMing the worker.
3. Playground XSS sinks for model-supplied strings (labels, class names, error messages) are closed.

## Non-goals

- CSRF protection (Bucket G follow-up).
- Refactoring `makeToolCard` to take DOM nodes (Bucket G).
- Cancellation of in-flight inference (Bucket B / D).
- Anything in Bucket B (`spawn_blocking`) or D (timeouts).

## Sub-tasks

### A1 — JWT default hardening

**File:** `src/config.rs`

Add a new method:

```rust
impl Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.auth.enabled {
            if self.auth.jwt_secret == "your-secret-key-here" {
                anyhow::bail!(
                    "auth.enabled is true but jwt_secret is the placeholder \
                     'your-secret-key-here'; set a real secret in config.toml \
                     or via TORCH_AUTH_JWT_SECRET"
                );
            }
            if self.auth.jwt_secret.len() < 32 {
                anyhow::bail!(
                    "auth.enabled is true but jwt_secret is shorter than 32 chars \
                     ({}); use a high-entropy secret",
                    self.auth.jwt_secret.len()
                );
            }
        }
        Ok(())
    }
}
```

Wire the call into `Config::load()` so an invalid config never reaches `main()`. Update the existing `test_auth_jwt_secret_defaults` test to reflect that the default is intentionally invalid (or split: test the value, then test that `validate()` rejects it).

**Why not generate a random secret per process:** silently masks misconfiguration; refresh tokens stop working across restarts; failure mode is "operator confused", not "attacker locked out".

### A2 — Body size caps

#### A2a — Multipart STT (`POST /stt/transcribe`)

**Files:** `src/api/audio.rs`, `src/config.rs`

- Add `config.audio.max_upload_bytes: usize` (default `100 * 1024 * 1024` = 100 MB).
- In `transcribe_audio` (`audio.rs:174-195`), before each `audio_data.extend_from_slice(&data)`, check `audio_data.len() + data.len() > max_upload_bytes` and return `ApiError::PayloadTooLarge(...)` (add this `ApiError` variant if missing → maps to HTTP 413).

#### A2b — Multipart YOLO (`POST /yolo/detect`)

**Files:** `src/api/yolo.rs`, `src/config.rs`

- Add `config.images.max_upload_bytes: usize` (default `10 * 1024 * 1024` = 10 MB).
- In the multipart loop (`yolo.rs:110-120`), track total bytes written to the temp file; bail with 413 on overflow. Clean up the temp file on bailout.

#### A2c — Classify per-item base64 cap

**File:** `src/api/classify.rs`

- Add `config.images.max_base64_bytes: usize` (default `5 * 1024 * 1024` ≈ 3.7 MB raw image, fits 1080p JPEG).
- Validate per-item before decode at `classify.rs:99-108` (batch) and `:205-222` (stream).
- The JSON-level cap (`config.server.json_body_limit_mb = 50`) still gates total payload.

#### A2d — `ApiError::PayloadTooLarge`

**File:** wherever `ApiError` lives (likely `src/error.rs`).

- Add variant if absent. `impl ResponseError` returns `StatusCode::PAYLOAD_TOO_LARGE` (413). Body: structured JSON with `error: "payload_too_large"`, `limit_bytes`, `received_bytes`.

### A3 — Audio decode caps

**File:** `src/core/audio.rs`, `src/config.rs`

- Add `config.audio.max_duration_secs: u32` (default `1800` = 30 min).
- In `validate_wav` (`audio.rs:100-114`): after `let spec = reader.spec()`, check `reader.duration() as u64 > spec.sample_rate as u64 * max_duration_secs as u64` → bail with `"audio too long: {dur}s > {max}s"`. This runs before any sample buffer allocation.
- In `load_with_symphonia` (`audio.rs:272-292`): track `samples.len()`; break with an error (not silently) if `samples.len() > sample_rate * max_duration_secs`. Surface this as `Err`, not as a silent truncation.

This protects against (a) honestly-large files we can't process, and (b) malicious headers claiming huge duration.

### A4 — Playground XSS escape

**File:** `src/api/playground.html`

Add near the top of the script block:

```javascript
function escapeHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
```

Apply at five sites:

| Line | Field | Source |
|---|---|---|
| 5703 | `p.label` | classify response |
| 5736 | `b.class_name` | YOLO response |
| 5682 | `err.message` | TTS error path |
| 5709 | `err.message` | classify error path |
| 5742 | `err.message` | YOLO error path |

`makeToolCard`'s `bodyHTML` parameter stays as raw HTML — the contract becomes "callers must escape any untrusted string before passing in". A comment above the function makes that explicit.

`title` parameters are all hardcoded today; no change needed. Audit lines 5650 / 5677 also pass `bodyHTML` from static / SVG sources — also no change.

## Out of scope (deferred)

- All other Playground items (G).
- CSRF.
- Refactoring `makeToolCard` to a typed-text API.
- Per-route timeouts (D).
- `spawn_blocking` boundaries (B).

## Verification

Manual / spot:
- `cargo check` clean.
- `Config::validate()`:
  - `Config::default()` should now fail validation.
  - A config with `auth.enabled = false` and the placeholder secret should pass (auth off — secret unused).
  - A config with `auth.enabled = true` and a 32+ char secret should pass.
- Hand-issue a 200 MB multipart upload to `/stt/transcribe`: expect 413, no OOM.
- Hand-issue a 50 MB multipart image to `/yolo/detect`: expect 413.
- Send `images: ["A".repeat(10_000_000)]` to `/classify/batch`: expect 413 / 400 with limit message.
- Send a synthesised WAV with header claiming 5 hours of audio: expect "audio too long" error before parse.
- Browser test: feed a fake response with `class_name: "<img src=x onerror=alert(1)>"` (via DevTools fetch override) and confirm it renders as visible text.

Automated:
- New unit tests for `Config::validate()` covering the three branches.
- New unit test for `escapeHtml` would require a JS test runner — out of scope; verify by inspection.

## Risk

- **Breaks `Config::default()` tests** that currently rely on the placeholder secret being accepted. Update those tests.
- **The existing 50 MB JSON limit** (`config.server.json_body_limit_mb`) is unchanged. Operators with batch-classify workflows pushing >50 MB JSON already had to bump it; A2 doesn't touch this.
- **Hardcoded magic numbers in defaults.** Bucket H will eventually move all magic numbers to named constants; we accept the inconsistency for now.

## Files to edit

1. `src/config.rs` — `validate()` method, new fields (`audio.max_upload_bytes`, `audio.max_duration_secs`, `images.max_upload_bytes`, `images.max_base64_bytes`), wire into `Config::load()`.
2. `src/error.rs` (or wherever `ApiError` lives) — `PayloadTooLarge` variant + 413 mapping.
3. `src/api/audio.rs` — multipart cap + duration cap pass-through.
4. `src/api/yolo.rs` — multipart cap.
5. `src/api/classify.rs` — per-item base64 cap (batch and stream).
6. `src/core/audio.rs` — `validate_wav` duration cap; `load_with_symphonia` sample cap.
7. `src/api/playground.html` — `escapeHtml` helper + 5 call-sites.
8. `src/main.rs` — if `Config::load()` doesn't already auto-call `validate()`, call it after load and `expect`/`bail` on failure.

Approximate diff size: 150–250 lines across 8 files. No new dependencies.
