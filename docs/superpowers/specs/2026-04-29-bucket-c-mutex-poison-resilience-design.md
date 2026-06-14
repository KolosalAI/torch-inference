# Bucket C — Mutex Poison Resilience — Design

Phase 3 sub-project from the 2026-04-29 product audit triage. Removes panic-on-poison failure modes around shared inference state.

## Goal

A panic inside a guarded section must not take down the server.

Today, three call-sites use `.lock().unwrap()` or `.expect(...)` on locking primitives that protect shared inference state. If a panic occurs while the guard is held (`std::sync::Mutex` poisons), every subsequent request crashes the worker until restart. Replace these with primitives or error paths that fail gracefully.

## Non-goals

- Preventing inference panics in the first place — that is **Bucket B (async/blocking correctness)**, which adds `spawn_blocking` boundaries and tightens error mapping.
- Recovering inconsistent inference state. ORT session state after a mid-run panic is not meaningfully recoverable; the only safe action is to fail subsequent requests cleanly.
- Auditing every other `.unwrap()` / `.expect()` in the codebase.

## Scope

Exactly three call-sites:

| File:line | Primitive | Pattern |
|---|---|---|
| `src/core/ort_classify.rs:188` | `std::sync::Mutex<Session>` | `self.session.lock().unwrap()` |
| `src/core/ort_yolo.rs:126` | `std::sync::Mutex<Session>` | `self.session.lock().unwrap()` |
| `src/core/kokoro_onnx.rs:72-76` | `tokio::sync::Semaphore` | `.acquire().await.expect("session pool semaphore closed")` |
| `src/core/kokoro_onnx.rs:81` | `parking_lot::Mutex<Vec<Session>>` (already) | `.pop().expect("session available after permit was acquired")` |

The fourth row is in scope as a small symmetry fix.

## Approach

### A. ORT session mutexes → `parking_lot::Mutex`

Drop `std::sync::Mutex` (poison-by-default) for `parking_lot::Mutex` (no poisoning). The crate is already declared in `Cargo.toml` (line 90) and used by `src/core/kokoro_onnx.rs:12`.

Concrete edits:

- `src/core/ort_classify.rs`
  - Replace the `std::sync::Mutex` import (or fully-qualified usage) with `parking_lot::Mutex`.
  - Line 188 becomes `let mut sess = self.session.lock();` (no `.unwrap()` — `parking_lot`'s `lock()` returns the guard directly).
- `src/core/ort_yolo.rs`
  - Same two changes.

Rationale (vs. an `lock_or_503` helper over `std::sync::Mutex`): a poisoned ORT mutex stays poisoned for the life of the process, so the helper-version still produces an unrecoverable server (every later request 503s forever). `parking_lot` removes the failure mode entirely with the smallest diff and matches the rest of the codebase.

### B. Kokoro semaphore acquire → graceful 503

`tokio::sync::Semaphore::acquire().await` errors only when the semaphore is closed, which on this codebase happens during graceful shutdown.

Concrete edit at `src/core/kokoro_onnx.rs:72-76`:

```rust
let permit = self
    .semaphore
    .acquire()
    .await
    .map_err(|_| anyhow::anyhow!("Kokoro session pool closed"))?;
```

The function signature (`acquire`) currently returns `SessionGuard<'_>`; it must change to `anyhow::Result<SessionGuard<'_>>`. Callers already in an `anyhow::Result` context propagate via `?`.

### C. Kokoro `pop().expect(...)` symmetry fix

`src/core/kokoro_onnx.rs:81`:

```rust
let session = {
    let mut guard = self.sessions.lock();
    guard
        .pop()
        .ok_or_else(|| anyhow::anyhow!("session pool empty after permit acquired"))?
};
```

This case is unreachable as long as the semaphore-vs-vec invariant holds, but we prefer not to panic on bug.

## Out of scope (will be picked up later)

- `tokio::sync::Mutex` sites elsewhere in the codebase (none flagged in the audit for Bucket C).
- Other `.unwrap()`/`.expect()` calls — those land in Bucket H (Validation & error mapping).

## Verification

Manual:
- `grep -rnE "\.lock\(\)\.unwrap\(\)|\.lock\(\)\.expect\(" src/` should return zero hits in `src/core/ort_*` and `src/core/kokoro_onnx.rs`.

Automated:
- `cargo check` clean.
- `cargo test` no regressions in modules touching the changed files: `cargo test -p torch-inference-server core::ort_classify core::ort_yolo core::kokoro_onnx`.
- Existing handler-level integration tests for `/classify/batch`, `/detect`, `/tts/stream` still pass.

No new behaviour is introduced, so no new tests are required. (TDD doesn't apply: the change is a refactor of safe behaviour into safer behaviour with the same observable contract on the happy path.)

## Risk

- `parking_lot::Mutex` does **not** drop the guard's contents on a panic in the same way `std::sync::Mutex` poisons; if a panic happens while holding the lock, the next acquirer sees whatever state was left. For ORT sessions, this is acceptable — the state is opaque to us, and we don't trust it post-panic anyway. Bucket B (`spawn_blocking` + error mapping) catches inference panics before they reach the lock site.
- API change: `KokoroSessionPool::acquire` becomes fallible (`Result<...>`). Internal-only function — risk limited to the one crate.

## Out-of-scope clean-up I will not do here

- Renaming `KokoroSessionPool::acquire` for clarity.
- Adding richer error context (this is a one-shot fix, not a redesign of error types).

## Files to edit

- `src/core/ort_classify.rs`
- `src/core/ort_yolo.rs`
- `src/core/kokoro_onnx.rs`

No tests, no `Cargo.toml`, no other modules.
