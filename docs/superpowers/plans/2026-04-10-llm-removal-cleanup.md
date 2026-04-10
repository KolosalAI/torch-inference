# LLM Removal Fallout Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove residual LLM build-config, dead dependencies, and stale documentation so the repo's configuration matches reality (TTS/STT/classification/detection only — no LLM).

**Architecture:** Config- and docs-only cleanup. No source changes under `src/`. Each edited file gets its own commit. Verification is `cargo build --release` + targeted greps.

**Tech Stack:** Cargo, YAML, Markdown, Mermaid.

**Spec:** `docs/superpowers/specs/2026-04-10-llm-removal-cleanup-design.md`

---

## File Structure

Files modified (no files created, no files deleted):

- `Cargo.toml` — remove 3 feature flags, 2 optional deps, update 1 comment
- `config.yaml` — delete 1 orphan comment line
- `CLAUDE.md` — drop LLM env var example, `llm/` directory mention in architecture tree, `llm.proxy_base_url` bullet
- `docs/CONFIGURATION.md` — remove LLM branch from the features-decision mermaid diagram

---

## Task 1: Cargo.toml — remove LLM features and unused deps

**Files:**
- Modify: `Cargo.toml` (lines 116, 120, 124, 152–154)

- [ ] **Step 1: Baseline grep — record the expected-gone strings**

Run: `grep -n 'llm\|tokenizers\|half = \|SIMD, zero-copy, LLM' Cargo.toml`

Expected current output:
```
116:# Performance — SIMD, zero-copy, LLM
120:half = { version = "2.3", features = ["bytemuck", "num-traits"] }    # f16 / bf16 for LLM
124:tokenizers = { version = "0.19", optional = true, default-features = false } # HuggingFace tokenizer
152:llm = ["tokenizers"]                    # LLM inference subsystem
153:llm-metal = ["llm"]                     # LLM with Metal attention (macOS)
154:llm-cuda = ["llm", "cuda"]              # LLM with CUDA FlashAttention
```

- [ ] **Step 2: Update the section comment on line 116**

Change:
```toml
# Performance — SIMD, zero-copy, LLM
```
to:
```toml
# Performance — SIMD, zero-copy
```

- [ ] **Step 3: Delete the `half` dependency line (line 120)**

Delete this entire line:
```toml
half = { version = "2.3", features = ["bytemuck", "num-traits"] }    # f16 / bf16 for LLM
```

- [ ] **Step 4: Delete the `tokenizers` dependency line (line 124)**

Delete this entire line:
```toml
tokenizers = { version = "0.19", optional = true, default-features = false } # HuggingFace tokenizer
```

- [ ] **Step 5: Delete the three LLM feature flags (lines 152–154)**

Delete these three lines from the `[features]` section:
```toml
llm = ["tokenizers"]                    # LLM inference subsystem
llm-metal = ["llm"]                     # LLM with Metal attention (macOS)
llm-cuda = ["llm", "cuda"]              # LLM with CUDA FlashAttention
```

- [ ] **Step 6: Verify the grep is now empty**

Run: `grep -n 'llm\|tokenizers\|half = \|SIMD, zero-copy, LLM' Cargo.toml`
Expected: no output (exit code 1).

- [ ] **Step 7: Build to confirm Cargo.toml still parses and the project compiles**

Run: `cargo build --release 2>&1 | tail -20`
Expected: `Finished \`release\` profile ... target(s)` line. Pre-existing dead-code warnings in `src/core/tts_manager.rs` (`fnv1a_u64`) and `src/core/ort_yolo.rs` (`detect_file`) are allowed. Any error or any *new* warning is a failure.

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
chore(cargo): remove dead LLM features and unused deps

Drops the llm/llm-metal/llm-cuda feature flags and the tokenizers and
half optional dependencies. Neither dependency was imported anywhere in
src/ and the feature flags were orphaned after the LLM modules were
deleted. Updates the section comment on line 116 to match.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: config.yaml — delete orphan LLM comment

**Files:**
- Modify: `config.yaml` (line 177)

- [ ] **Step 1: Baseline grep**

Run: `grep -n 'LLM' config.yaml`
Expected current output:
```
177:# LLM proxy configuration
```

- [ ] **Step 2: Delete line 177**

Delete this single line (context: it sits between the `tts:` block ending at line 175 and the `# STT Configuration` heading at line 178, with no associated config block of its own):
```yaml
# LLM proxy configuration
```

- [ ] **Step 3: Verify the grep is now empty**

Run: `grep -n 'LLM' config.yaml`
Expected: no output (exit code 1).

- [ ] **Step 4: Verify YAML still parses**

Run: `cargo run --release --bin torch-inference-server -- --help 2>&1 | head -5` **OR** if no `--help` flag exists, run: `python3 -c "import yaml; yaml.safe_load(open('config.yaml'))" && echo OK`
Expected: `OK` (or server help output without a YAML parse error).

- [ ] **Step 5: Commit**

```bash
git add config.yaml
git commit -m "$(cat <<'EOF'
chore(config): drop orphan LLM proxy comment

The "# LLM proxy configuration" comment was sitting between the tts and
stt sections with no associated config block after the LLM proxy was
removed.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CLAUDE.md — remove LLM env var example, architecture tree entry, and config bullet

**Files:**
- Modify: `CLAUDE.md` (lines 35–36, line 73, line 107)

- [ ] **Step 1: Baseline grep**

Run: `grep -n 'KOLOSAL_LLM\|proxy_base_url\|llm/' CLAUDE.md`
Expected current output:
```
36:KOLOSAL_LLM_BASE_URL=http://localhost:11434 ./target/release/torch-inference-server
73:    llm/            — HTTP proxy stub; do not extend
107:- `llm.proxy_base_url` — ignored for playground (LLM UI removed)
```

- [ ] **Step 2: Remove the LLM env var example in the Build & Run section**

Delete these two lines (line 35 is the comment, line 36 is the command):
```
# Run with custom LLM proxy URL (ignored, but configurable via env)
KOLOSAL_LLM_BASE_URL=http://localhost:11434 ./target/release/torch-inference-server
```

Also remove the blank line that would be left between the preceding `./target/release/torch-inference-server` line and the following `# Run all tests` line, so the code block stays tidy (one blank line between commands).

- [ ] **Step 3: Remove the `llm/` entry from the architecture tree**

Delete this line from the `core/` tree listing:
```
    llm/            — HTTP proxy stub; do not extend
```

- [ ] **Step 4: Remove the `llm.proxy_base_url` bullet from the Configuration section**

Delete this bullet from the `config.yaml` key sections list:
```
- `llm.proxy_base_url` — ignored for playground (LLM UI removed)
```

- [ ] **Step 5: Verify the grep is now empty**

Run: `grep -n 'KOLOSAL_LLM\|proxy_base_url\|llm/' CLAUDE.md`
Expected: no output (exit code 1).

Note: the "LLM — Explicitly out of scope" directive section (lines 16–24 in the original file) MUST be preserved. That section tells future engineers not to add LLM features — it is guidance, not a stale reference to deleted code.

- [ ] **Step 6: Verify the out-of-scope directive is still present**

Run: `grep -n 'Explicitly out of scope' CLAUDE.md`
Expected: exactly one match pointing at the "## LLM — Explicitly out of scope" heading.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(claude): remove stale LLM env var, tree entry, and config bullet

Deletes the KOLOSAL_LLM_BASE_URL build example, the llm/ directory line
in the architecture tree, and the llm.proxy_base_url bullet in the
Configuration section. The "LLM — Explicitly out of scope" directive
is preserved since it is active guidance, not a stale reference.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: docs/CONFIGURATION.md — remove LLM branch from mermaid diagram

**Files:**
- Modify: `docs/CONFIGURATION.md` (lines 379–388)

- [ ] **Step 1: Baseline grep**

Run: `grep -n 'LLM\|tokenizers\|F_LLM' docs/CONFIGURATION.md`
Expected current output (among possibly other LLM-word matches elsewhere in the file — only the mermaid section matters here):
```
379:    F_OTEL --> Q5{LLM inference?}
381:    Q5 -->|HuggingFace tokenizers| F_LLM["--features llm<br/>tokenizers 0.19"]
382:    Q5 -->|LLM + Apple Metal| F_LLM_METAL["--features llm-metal"]
383:    Q5 -->|LLM + CUDA FlashAttn| F_LLM_CUDA["--features llm-cuda"]
384:    Q5 -->|No LLM| Q6
386:    F_LLM --> Q6{Performance profiling?}
387:    F_LLM_METAL --> Q6
388:    F_LLM_CUDA --> Q6
```

- [ ] **Step 2: Short-circuit F_OTEL directly to Q6 and remove the Q5/F_LLM* nodes**

Replace lines 379–388 (the `F_OTEL --> Q5{LLM inference?}` block through the `F_LLM_CUDA --> Q6` edge) with:

```
    F_OTEL --> Q6{Performance profiling?}
```

This keeps `Q6` defined once (merged with its old definition on line 386) and deletes all LLM-related nodes and edges entirely. The `F_METRICS --> Q4` and `Q4 -->|No tracing| Q5` edges will now need their `Q5` target rewritten too — search the surrounding mermaid block for any remaining edge whose target is `Q5` and retarget it to `Q6`.

- [ ] **Step 3: Verify no `Q5` or `F_LLM` identifiers remain in the mermaid block**

Run: `grep -n 'Q5\|F_LLM' docs/CONFIGURATION.md`
Expected: no output (exit code 1). If any edge still references `Q5`, retarget it to `Q6` and re-run this grep.

- [ ] **Step 4: Verify the mermaid diagram still renders**

Open the file in a mermaid-capable viewer (VS Code with the Markdown Preview Mermaid plugin, or `npx @mermaid-js/mermaid-cli -i docs/CONFIGURATION.md -o /tmp/diagram.svg`). Expected: the features-decision flowchart renders end-to-end from start through `Done`, without any dangling nodes or parse errors. If parsing fails, compare the edited block against the surrounding nodes and fix any stray `-->` edges.

- [ ] **Step 5: Verify no LLM strings remain in the mermaid section**

Run: `sed -n '340,400p' docs/CONFIGURATION.md | grep -n 'LLM\|tokenizers\|llm'`
Expected: no output (exit code 1).

- [ ] **Step 6: Commit**

```bash
git add docs/CONFIGURATION.md
git commit -m "$(cat <<'EOF'
docs(config): drop LLM branch from features-decision diagram

Removes the Q5 "LLM inference?" node and the F_LLM/F_LLM_METAL/F_LLM_CUDA
nodes from the mermaid flowchart, short-circuiting F_OTEL directly to
Q6 (profiling). The LLM features those nodes advertised no longer exist.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final verification

**Files:** (read-only; verification only)

- [ ] **Step 1: Re-build the release binary**

Run: `cargo build --release 2>&1 | tail -20`
Expected: `Finished \`release\` profile ... target(s)` line. The two pre-existing dead-code warnings in `src/core/tts_manager.rs` and `src/core/ort_yolo.rs` are allowed. Any compile error or any *new* warning is a failure.

- [ ] **Step 2: Run cargo check on all targets (catches test/bench regressions)**

Run: `cargo check --all-targets 2>&1 | tail -20`
Expected: exit 0. Pre-existing warnings are allowed; new errors or new warnings referencing `half`, `tokenizers`, or any `llm*` feature are failures.

- [ ] **Step 3: Targeted grep across the four cleaned files**

Run:
```bash
grep -rni 'llm\|tokenizers' Cargo.toml config.yaml docs/CONFIGURATION.md
```
Expected: no output (exit code 1).

Then run:
```bash
grep -n 'KOLOSAL_LLM\|proxy_base_url\|llm/' CLAUDE.md
```
Expected: no output (exit code 1).

Then run (sanity check — the out-of-scope directive must still exist):
```bash
grep -n 'Explicitly out of scope' CLAUDE.md
```
Expected: exactly one match.

- [ ] **Step 4: Repository-wide sanity grep for active code references**

Run:
```bash
grep -rn 'KOLOSAL_LLM\|proxy_base_url' src/ Cargo.toml config.yaml CLAUDE.md docs/CONFIGURATION.md
```
Expected: no output (exit code 1). Archived plans under `docs/superpowers/plans/` may still mention `tokenizers` — that is acceptable historical content and is explicitly out of scope for this plan.

- [ ] **Step 5: Confirm git log shows 4 clean commits**

Run: `git log --oneline -5`
Expected: top 4 commits are (in reverse order) Task 1 through Task 4 from this plan, plus the earlier spec commit (`4ebc214`) below them.

No commit is needed for Task 5 — it is verification only.

---

## Self-Review

**Spec coverage:**
- Goal 1 (Cargo.toml no LLM features/deps) → Task 1 ✓
- Goal 2 (config.yaml no orphan comments) → Task 2 ✓
- Goal 3 (CLAUDE.md + CONFIGURATION.md no LLM docs) → Tasks 3 and 4 ✓
- Goal 4 (`cargo build --release` still passes) → Task 5 Step 1 ✓
- Spec Risk "Cargo.lock churn" → Task 1 Step 8 commits `Cargo.lock` alongside `Cargo.toml` ✓
- Spec Risk "mermaid diagram breakage" → Task 4 Step 4 renders the diagram to verify ✓

**Spec gap found and added:** The spec listed only the Build & Run env var and the Configuration bullet for `CLAUDE.md`, but exploration also found `llm/` listed in the architecture tree on line 73. Task 3 Step 3 removes it. This is an obvious oversight; the spec's intent is clearly "remove LLM references from CLAUDE.md," and this is one.

**Placeholder scan:** No TBDs, no "add appropriate X", no "similar to Task N", no undefined types.

**Type consistency:** N/A — no code types involved.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-10-llm-removal-cleanup.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
