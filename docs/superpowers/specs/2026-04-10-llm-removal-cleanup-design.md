# LLM Removal Fallout Cleanup — Design

**Date:** 2026-04-10
**Status:** Approved for planning
**Scope:** Sub-project F of the "polish all features" initiative

## Context

LLM support was removed from the codebase prior to this work: `src/core/llm/`, `src/api/llm.rs`, and `benches/llm_bench.rs` are all deleted. Exploration confirmed that the Rust source no longer references the deleted modules — `cargo build --release` currently passes with only pre-existing unrelated dead-code warnings.

What remains are **dead build-config, dead dependencies, and stale documentation** that still advertise LLM support to both future readers and `cargo`. This sub-project deletes that residue so the repo's configuration matches its reality: TTS, STT, classification, and detection only — no LLM.

This is a cosmetic/hygienic cleanup. There is no architecture, no new code, and no behavior change at runtime.

## Goals

1. `Cargo.toml` declares no LLM features and no LLM-only dependencies.
2. `config.yaml` contains no orphan LLM comments.
3. `CLAUDE.md` and `docs/CONFIGURATION.md` do not document LLM env vars, config fields, or feature flags.
4. `cargo build --release` still passes after the cleanup, with no new warnings.

## Non-Goals

- Touching `docs/superpowers/plans/2026-04-05-server-optimization.md` — it is an archived historical plan and references to `tokenizers`/`llm` there are accurate history.
- Touching `playground.html` line 2330 — the "LLM-based TTS" string describes CosyVoice2's *architecture* (a TTS model), not an LLM feature of this server.
- Removing the pre-existing dead-code warnings in `src/core/tts_manager.rs` (`fnv1a_u64`) and `src/core/ort_yolo.rs` (`detect_file`) — unrelated to LLM removal, belongs to a separate polish pass.
- Any source code changes under `src/`.

## Changes

### 1. `Cargo.toml`

**Remove optional dependencies (unused anywhere in `src/`):**

- Line 120: `half = { version = "2.3", features = ["bytemuck", "num-traits"] }` — confirmed unused by grep; only word matches are unrelated comments about "half pixels" / "half-open state".
- Line 124: `tokenizers = { version = "0.19", optional = true, default-features = false }` — confirmed unused by grep; only referenced from the dead `llm` feature flag.

**Remove feature flags (dead):**

- Line 152: `llm = ["tokenizers"]`
- Line 153: `llm-metal = ["llm"]`
- Line 154: `llm-cuda = ["llm", "cuda"]`

**Update comment:**

- Line 116: `# Performance — SIMD, zero-copy, LLM` → `# Performance — SIMD, zero-copy`

### 2. `config.yaml`

- Line 177: delete the orphan comment `# LLM proxy configuration` that sits between the TTS and STT sections with no associated config block.

### 3. `CLAUDE.md`

- **Build & Run section:** remove the example line
  `KOLOSAL_LLM_BASE_URL=http://localhost:11434 ./target/release/torch-inference-server`
  and its accompanying "Run with custom LLM proxy URL" comment.
- **Configuration section:** remove the bullet `llm.proxy_base_url — ignored for playground (LLM UI removed)`.

### 4. `docs/CONFIGURATION.md`

- Line 381: edit the mermaid diagram to drop the `Q5 -->|HuggingFace tokenizers| F_LLM["--features llm<br/>tokenizers 0.19"]` node and any edge that connects to it. If the surrounding subgraph becomes ill-formed, adjust the neighboring nodes so the diagram still renders.

## Verification

After the changes:

1. `cargo build --release` exits 0 with no *new* warnings (the two pre-existing dead-code warnings are allowed to remain).
2. `cargo check --all-targets` exits 0.
3. `grep -rni 'llm\|tokenizers' Cargo.toml config.yaml CLAUDE.md docs/CONFIGURATION.md` returns no matches other than intentional ones (none expected in these four files).
4. `grep -rn 'KOLOSAL_LLM\|proxy_base_url' .` returns no matches in tracked non-archived files.

## Risks

- **Cargo.lock churn:** removing `half` and `tokenizers` will cause `Cargo.lock` to drop their transitive deps. This is expected and should be committed along with `Cargo.toml`.
- **Mermaid diagram breakage:** if the `Q5` node's incoming edge in `docs/CONFIGURATION.md` is structurally required for the rest of the diagram to render, we may need to rewire the neighboring nodes. Mitigation: view the rendered diagram after edit; worst case, replace the removed branch with a neutral "(no LLM support)" terminal node.

## Out of Scope for Follow-up Sub-Projects

After this sub-project lands, the next in the polish queue is **D: ORT backends finalization** (`src/core/onnx_backend.rs`, `ort_classify.rs`, `ort_yolo.rs`). That will get its own spec.
