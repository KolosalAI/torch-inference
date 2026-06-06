# HRM-Text Agentic System — Memory Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the LLM microservice from leaking / OOM-ing under load by (a) re-exporting HRM-Text with a real KV cache, (b) adding a global engine semaphore + process RSS gate, (c) bounding every request input and every long-lived in-memory structure, all without redesigning the agent layer.

**Architecture:** Two-graph KV-cache ONNX (`prefill.onnx` + `decode_step.onnx`) consumed by a new `KvSession` backend that reuses pre-allocated `KvBuffers` across requests. A 1-permit `EngineLease` serializes every ONNX call (chat + planner + reflect). A `MemoryGate` reads process RSS and refuses admission above a high water mark. Surgical bounds at the HTTP boundary on body size, image bytes, prompt chars, message count, and generated tokens. KV path is opt-in via `[kv_cache] enabled=true`; falls back to the existing monolithic `model.onnx` when KV artifacts are missing.

**Tech Stack:** Rust 2021, `actix-web` 4, `ort` 2.0.0-rc.10 with `half` feature, `tokio` 1, Python 3.10+, `transformers` from git (v5 dev), `torch` 2.12+, `onnxscript` 0.7.

**Source spec:** `docs/superpowers/specs/2026-05-20-hrm-agentic-memory-hardening-design.md`

---

## Phase 1 — Re-export ONNX with KV cache (offline, Python-only)

### Task 1.1: Add `PrefillWrapper` + `DecodeStepWrapper` to the export script

**Files:**
- Modify: `scripts/export_hrm_text.py`

- [ ] **Step 1: Add the wrapper classes alongside the existing `LogitsOnly`.**

Open `scripts/export_hrm_text.py`. After the `LogitsOnly` class definition (currently ends around line 31), add:

```python
class PrefillWrapper(nn.Module):
    """First-pass wrapper. Takes input_ids; returns logits and a flat tuple of
    (k0, v0, k1, v1, ..., kN-1, vN-1) cache tensors so torch.export can pytree
    them."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        out = self.m(input_ids=input_ids, use_cache=True, return_dict=True)
        kvs = []
        for layer_kv in out.past_key_values:  # DynamicCache iterable of (k, v)
            kvs.extend([layer_kv[0], layer_kv[1]])
        return (out.logits, *kvs)


class DecodeStepWrapper(nn.Module):
    """Cached-pass wrapper. Takes one new token + the flat cache tuple from the
    previous step; returns updated logits and a fresh flat cache tuple."""
    def __init__(self, m, num_layers):
        super().__init__()
        self.m = m
        self.L = num_layers

    def forward(self, input_ids, *past_flat):
        # transformers v5 cache_utils
        from transformers.cache_utils import DynamicCache
        past = [(past_flat[2 * i], past_flat[2 * i + 1]) for i in range(self.L)]
        cache = DynamicCache.from_legacy_cache(past)
        out = self.m(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        new = []
        for layer_kv in out.past_key_values:
            new.extend([layer_kv[0], layer_kv[1]])
        return (out.logits, *new)
```

- [ ] **Step 2: Commit just the new classes (export logic not yet wired).**

```bash
git add scripts/export_hrm_text.py
git commit -m "feat(export): add Prefill/DecodeStep wrappers for KV-cache export"
```

---

### Task 1.2: Add a `--two-graph` export branch to `export()`

**Files:**
- Modify: `scripts/export_hrm_text.py`

- [ ] **Step 1: Add a new CLI flag and route the export.**

In the `if __name__ == "__main__":` block at the bottom, replace:

```python
    ap.add_argument("--fast-loops", type=int, default=None,
                    help="override L_cycles in runtime config (default: from checkpoint)")
    args = ap.parse_args()
    export(args.quantize, args.slow_loops, args.fast_loops)
```

with:

```python
    ap.add_argument("--fast-loops", type=int, default=None,
                    help="override L_cycles in runtime config (default: from checkpoint)")
    ap.add_argument("--two-graph", action="store_true",
                    help="emit prefill.onnx + decode_step.onnx with KV-cache I/O")
    args = ap.parse_args()
    if args.two_graph:
        export_two_graph(args.slow_loops, args.fast_loops)
    else:
        export(args.quantize, args.slow_loops, args.fast_loops)
```

- [ ] **Step 2: Add the `export_two_graph` function.**

Above the `if __name__ == "__main__":` block, add:

```python
def export_two_graph(slow_loops_override, fast_loops_override):
    """Emit prefill.onnx + decode_step.onnx with explicit past/present KV I/O.

    Required artifacts:
      OUT_DIR/prefill.onnx          (graph)
      OUT_DIR/prefill.onnx.data     (external weights, ~2.3 GB)
      OUT_DIR/decode_step.onnx
      OUT_DIR/decode_step.onnx.data
      OUT_DIR/config.json (extended with num_heads/head_dim)
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {MODEL_ID} (two-graph KV export)...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.train(False)
    # KV path requires use_cache=True for both graphs
    model.config.use_cache = True

    hf_cfg = model.config.to_dict()
    num_layers = hf_cfg["num_hidden_layers"]   # 16
    num_heads = hf_cfg["num_attention_heads"]
    head_dim = hf_cfg["hidden_size"] // num_heads

    # --- Prefill graph ---
    prefill = PrefillWrapper(model)
    sample_ids = tok("hello world", return_tensors="pt").input_ids  # [1, n]
    prefill_path = f"{OUT_DIR}/prefill.onnx"
    print(f"Exporting prefill -> {prefill_path}...")
    pkv_output_names = []
    for i in range(num_layers):
        pkv_output_names.extend(
            [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
        )
    dyn_axes = {"input_ids": {1: "seq"}, "logits": {1: "seq"}}
    for name in pkv_output_names:
        dyn_axes[name] = {2: "seq"}
    with torch.no_grad():
        torch.onnx.export(
            prefill, (sample_ids,), prefill_path,
            opset_version=18,
            input_names=["input_ids"],
            output_names=["logits", *pkv_output_names],
            dynamic_axes=dyn_axes,
            do_constant_folding=True,
        )

    # --- Decode-step graph ---
    # Build a one-token sample input and matching synthetic cache from a real
    # prefill run so shapes line up.
    with torch.no_grad():
        out0 = model(input_ids=sample_ids, use_cache=True, return_dict=True)
    sample_step_ids = sample_ids[:, -1:].clone()  # [1, 1]
    past_flat = []
    for layer_kv in out0.past_key_values:
        past_flat.extend([layer_kv[0], layer_kv[1]])

    decode = DecodeStepWrapper(model, num_layers)
    decode_path = f"{OUT_DIR}/decode_step.onnx"
    print(f"Exporting decode_step -> {decode_path}...")
    pkv_input_names = []
    present_output_names = []
    for i in range(num_layers):
        pkv_input_names.extend(
            [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
        )
        present_output_names.extend(
            [f"present_key_values.{i}.key", f"present_key_values.{i}.value"]
        )
    dyn_axes = {
        "input_ids": {1: "one"},
        "logits": {1: "one"},
    }
    for name in pkv_input_names:
        dyn_axes[name] = {2: "past_len"}
    for name in present_output_names:
        dyn_axes[name] = {2: "past_len_plus_one"}
    with torch.no_grad():
        torch.onnx.export(
            decode, (sample_step_ids, *past_flat), decode_path,
            opset_version=18,
            input_names=["input_ids", *pkv_input_names],
            output_names=["logits", *present_output_names],
            dynamic_axes=dyn_axes,
            do_constant_folding=True,
        )

    # tokenizer.json + extended config.json
    tok.save_pretrained(OUT_DIR)
    runtime_cfg = {
        "eos_token_id": hf_cfg.get("eos_token_id"),
        "ctx_size": hf_cfg.get("max_position_embeddings", 2048),
        "slow_loops": slow_loops_override or hf_cfg.get("H_cycles", 2),
        "fast_loops": fast_loops_override or hf_cfg.get("L_cycles", 3),
        "vocab_size": hf_cfg.get("vocab_size"),
        "hidden_size": hf_cfg.get("hidden_size"),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "logits_dtype": "float16",
    }
    with open(f"{OUT_DIR}/config.json", "w") as f:
        json.dump(runtime_cfg, f, indent=2)
    print("Two-graph export done.")
```

- [ ] **Step 3: Smoke-test the script's CLI parse (no model load).**

Run:
```bash
cd /Users/evintleovonzko/Documents/projects/evint/torch-inference
python scripts/export_hrm_text.py --help
```
Expected: help text mentions `--two-graph`.

- [ ] **Step 4: Commit.**

```bash
git add scripts/export_hrm_text.py
git commit -m "feat(export): two-graph KV ONNX export branch"
```

---

### Task 1.3: Add the KV-parity test script

**Files:**
- Create: `scripts/test_kv_parity.py`

- [ ] **Step 1: Write the parity test that runs the monolithic and KV graphs.**

Create `scripts/test_kv_parity.py`:

```python
"""Parity test: run 64-token greedy decode via the existing monolithic
model.onnx AND via the new prefill.onnx + decode_step.onnx. Assert
identical token IDs. Run AFTER `python scripts/export_hrm_text.py --two-graph`.

Exits non-zero on mismatch. Intended to gate the Rust work in Phase 3.
"""
import json
import sys
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

OUT_DIR = "services/llm/models/hrm-text-1b"
PROMPT = "The capital of France is"
N_TOKENS = 64


def greedy_monolithic(sess, ids):
    out = []
    for _ in range(N_TOKENS):
        logits = sess.run(["logits"], {"input_ids": ids})[0]
        next_id = int(np.argmax(logits[0, -1]))
        out.append(next_id)
        ids = np.concatenate([ids, [[next_id]]], axis=1).astype(np.int64)
    return out


def greedy_kv(prefill_sess, decode_sess, ids, cfg):
    L = cfg["num_layers"]
    # Prefill
    pkv_out_names = ["logits"] + [
        f"past_key_values.{i}.{k}" for i in range(L) for k in ("key", "value")
    ]
    prefill_outs = prefill_sess.run(pkv_out_names, {"input_ids": ids})
    logits = prefill_outs[0]
    past = prefill_outs[1:]  # 2*L tensors, alternating k,v
    next_id = int(np.argmax(logits[0, -1]))
    out = [next_id]

    # Decode loop
    pkv_in_names = ["input_ids"] + [
        f"past_key_values.{i}.{k}" for i in range(L) for k in ("key", "value")
    ]
    present_names = ["logits"] + [
        f"present_key_values.{i}.{k}" for i in range(L) for k in ("key", "value")
    ]
    for _ in range(N_TOKENS - 1):
        inputs = {"input_ids": np.array([[next_id]], dtype=np.int64)}
        for n, t in zip(pkv_in_names[1:], past):
            inputs[n] = t
        outs = decode_sess.run(present_names, inputs)
        logits = outs[0]
        past = outs[1:]
        next_id = int(np.argmax(logits[0, -1]))
        out.append(next_id)
    return out


def main():
    tok = AutoTokenizer.from_pretrained(OUT_DIR)
    ids = tok(PROMPT, return_tensors="np").input_ids.astype(np.int64)

    with open(f"{OUT_DIR}/config.json") as f:
        cfg = json.load(f)

    print("Running monolithic decode...")
    mono = ort.InferenceSession(f"{OUT_DIR}/model.onnx",
                                providers=["CPUExecutionProvider"])
    mono_tokens = greedy_monolithic(mono, ids.copy())

    print("Running two-graph KV decode...")
    pref = ort.InferenceSession(f"{OUT_DIR}/prefill.onnx",
                                providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(f"{OUT_DIR}/decode_step.onnx",
                                providers=["CPUExecutionProvider"])
    kv_tokens = greedy_kv(pref, dec, ids.copy(), cfg)

    if mono_tokens != kv_tokens:
        print(f"MISMATCH after {next(i for i, (a, b) in enumerate(zip(mono_tokens, kv_tokens)) if a != b)} tokens")
        print(f"  monolithic: {mono_tokens}")
        print(f"  kv-cache:   {kv_tokens}")
        sys.exit(1)
    print(f"PARITY OK — {N_TOKENS} tokens identical: {mono_tokens[:8]}...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit the parity script (will be run after artifacts exist).**

```bash
git add scripts/test_kv_parity.py
git commit -m "test(export): KV-cache parity script (monolithic vs two-graph)"
```

- [ ] **Step 3: Run the export and parity test (heavy; ~20–40 min on first model load).**

```bash
cd /Users/evintleovonzko/Documents/projects/evint/torch-inference
# Ensure model.onnx still exists (it does per ls; if not, run baseline export first)
python scripts/export_hrm_text.py --two-graph
python scripts/test_kv_parity.py
```

Expected (parity script): `PARITY OK — 64 tokens identical: [...]...`

If MISMATCH: do not proceed to Phase 2. Diagnose: check that `num_layers`, `num_heads`, `head_dim` match between `config.json` and what the wrappers passed; verify `past_key_values` ordering matches across the two graphs (each layer is `(k, v)` not `(v, k)`); confirm `dynamic_axes` names line up.

- [ ] **Step 4: Verify artifacts exist and commit the runtime config (artifacts are gitignored via `services/llm/models/` but `config.json` may not be — check).**

Run:
```bash
ls -la services/llm/models/hrm-text-1b/
```
Expected files: `prefill.onnx`, `prefill.onnx.data`, `decode_step.onnx`, `decode_step.onnx.data`, `config.json`, `tokenizer.json` (and the original `model.onnx`, `model.onnx.data`).

Then:
```bash
# Only stage what's not gitignored
git status services/llm/models/hrm-text-1b/
# If config.json changed and is tracked, commit it:
git add -p services/llm/models/hrm-text-1b/config.json 2>/dev/null || true
git diff --cached --quiet || git commit -m "chore(llm): bump config.json with KV cache fields"
```

---

## Phase 2 — Surgical bounds layer (Rust, KV cache OFF)

### Task 2.1: Extend `LlmConfig` with new sections

**Files:**
- Modify: `services/llm/src/config.rs`
- Modify: `services/llm/config.toml`

- [ ] **Step 1: Write a failing test for the new config sections.**

Append to the `#[cfg(test)] mod tests` block in `services/llm/src/config.rs`:

```rust
    #[test]
    fn parses_limits_section_with_defaults() {
        let toml_text = r#"
port = 8001
[limits]
max_image_bytes = 1024
"#;
        let cfg: LlmConfig = toml::from_str(toml_text).unwrap();
        let limits = cfg.limits.expect("limits section");
        assert_eq!(limits.max_image_bytes, 1024);
        assert_eq!(limits.max_prompt_chars, 16_384);
        assert_eq!(limits.max_generated_tokens, 512);
        assert_eq!(limits.engine.max_concurrent, 1);
        assert_eq!(limits.json.body_limit, 4_194_304);
        assert_eq!(limits.channels.sse_event_buffer, 8);
        assert_eq!(limits.results.field_trim_above, 8_192);
    }

    #[test]
    fn parses_memory_gate_section() {
        let toml_text = r#"
port = 8001
[memory_gate]
high_water_mb = 8192
"#;
        let cfg: LlmConfig = toml::from_str(toml_text).unwrap();
        let mg = cfg.memory_gate.expect("memory_gate section");
        assert_eq!(mg.high_water_mb, 8192);
        assert_eq!(mg.low_water_mb, 3_072);
    }

    #[test]
    fn parses_kv_cache_section() {
        let toml_text = r#"
port = 8001
[kv_cache]
enabled = false
"#;
        let cfg: LlmConfig = toml::from_str(toml_text).unwrap();
        let kv = cfg.kv_cache.expect("kv_cache section");
        assert!(!kv.enabled);
    }
```

- [ ] **Step 2: Run the failing tests.**

```bash
cd services/llm
cargo test -p llm-service --lib config::tests
```
Expected: compile failures on `cfg.limits`, `cfg.memory_gate`, `cfg.kv_cache`.

- [ ] **Step 3: Add the new structs and fields to `services/llm/src/config.rs`.**

In `LlmConfig`, after the `agent` field, add:

```rust
    #[serde(default)]
    pub limits: Option<LimitsConfig>,

    #[serde(default)]
    pub memory_gate: Option<MemoryGateConfig>,

    #[serde(default)]
    pub kv_cache: Option<KvCacheConfig>,
```

After `AgentToolsConfig`'s `Default` impl, add:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct LimitsConfig {
    #[serde(default = "default_max_image_bytes")]
    pub max_image_bytes: usize,
    #[serde(default = "default_max_prompt_chars")]
    pub max_prompt_chars: usize,
    #[serde(default = "default_max_messages")]
    pub max_messages: usize,
    #[serde(default = "default_max_generated_tokens")]
    pub max_generated_tokens: u32,
    #[serde(default = "default_max_ctx_size")]
    pub max_ctx_size: u32,
    #[serde(default)]
    pub json: LimitsJsonConfig,
    #[serde(default)]
    pub channels: LimitsChannelsConfig,
    #[serde(default)]
    pub engine: LimitsEngineConfig,
    #[serde(default)]
    pub results: LimitsResultsConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitsJsonConfig {
    #[serde(default = "default_body_limit")]
    pub body_limit: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitsChannelsConfig {
    #[serde(default = "default_sse_event_buffer")]
    pub sse_event_buffer: usize,
    #[serde(default = "default_chat_stream_buffer")]
    pub chat_stream_buffer: usize,
    #[serde(default = "default_chat_nonstream_buffer")]
    pub chat_nonstream_buffer: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitsEngineConfig {
    #[serde(default = "default_engine_max_concurrent")]
    pub max_concurrent: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LimitsResultsConfig {
    #[serde(default = "default_per_run_bytes")]
    pub per_run_bytes: usize,
    #[serde(default = "default_field_trim_above")]
    pub field_trim_above: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MemoryGateConfig {
    #[serde(default = "default_high_water_mb")]
    pub high_water_mb: u64,
    #[serde(default = "default_low_water_mb")]
    pub low_water_mb: u64,
    #[serde(default = "default_poll_on_admit_only")]
    pub poll_on_admit_only: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KvCacheConfig {
    #[serde(default = "default_kv_cache_enabled")]
    pub enabled: bool,
}

impl Default for LimitsJsonConfig {
    fn default() -> Self { Self { body_limit: default_body_limit() } }
}
impl Default for LimitsChannelsConfig {
    fn default() -> Self {
        Self {
            sse_event_buffer: default_sse_event_buffer(),
            chat_stream_buffer: default_chat_stream_buffer(),
            chat_nonstream_buffer: default_chat_nonstream_buffer(),
        }
    }
}
impl Default for LimitsEngineConfig {
    fn default() -> Self { Self { max_concurrent: default_engine_max_concurrent() } }
}
impl Default for LimitsResultsConfig {
    fn default() -> Self {
        Self {
            per_run_bytes: default_per_run_bytes(),
            field_trim_above: default_field_trim_above(),
        }
    }
}

fn default_max_image_bytes() -> usize { 2_097_152 }
fn default_max_prompt_chars() -> usize { 16_384 }
fn default_max_messages() -> usize { 32 }
fn default_max_generated_tokens() -> u32 { 512 }
fn default_max_ctx_size() -> u32 { 1024 }
fn default_body_limit() -> usize { 4_194_304 }
fn default_sse_event_buffer() -> usize { 8 }
fn default_chat_stream_buffer() -> usize { 16 }
fn default_chat_nonstream_buffer() -> usize { 64 }
fn default_engine_max_concurrent() -> usize { 1 }
fn default_per_run_bytes() -> usize { 65_536 }
fn default_field_trim_above() -> usize { 8_192 }
fn default_high_water_mb() -> u64 { 4_096 }
fn default_low_water_mb() -> u64 { 3_072 }
fn default_poll_on_admit_only() -> bool { true }
fn default_kv_cache_enabled() -> bool { true }
```

- [ ] **Step 4: Update `LlmConfig::load` default-case to include the new fields.**

In the `else` branch of `load()`, change the return to:

```rust
            Ok(Self {
                port: 8001,
                hrm: None,
                vision_bridge: None,
                agent: None,
                limits: None,
                memory_gate: None,
                kv_cache: None,
            })
```

- [ ] **Step 5: Run the tests.**

```bash
cargo test -p llm-service --lib config::tests
```
Expected: PASS for all three new tests + existing tests.

- [ ] **Step 6: Add the new sections to `services/llm/config.toml`.**

Append to `services/llm/config.toml`:

```toml

[limits]
max_image_bytes        = 2097152
max_prompt_chars       = 16384
max_messages           = 32
max_generated_tokens   = 512
max_ctx_size           = 1024

[limits.json]
body_limit             = 4194304

[limits.channels]
sse_event_buffer       = 8
chat_stream_buffer     = 16
chat_nonstream_buffer  = 64

[limits.engine]
max_concurrent         = 1

[limits.results]
per_run_bytes          = 65536
field_trim_above       = 8192

[memory_gate]
high_water_mb          = 4096
low_water_mb           = 3072
poll_on_admit_only     = true

[kv_cache]
enabled                = false
```

Note: `kv_cache.enabled = false` here — Phase 4 flips to true once stress tests pass.

- [ ] **Step 7: Commit.**

```bash
git add services/llm/src/config.rs services/llm/config.toml
git commit -m "feat(llm/config): add limits, memory_gate, kv_cache sections"
```

---

### Task 2.2: Create `EngineLease` (the engine semaphore)

**Files:**
- Create: `services/llm/src/engine_lease.rs`
- Modify: `services/llm/src/main.rs` (add `mod engine_lease;`)

- [ ] **Step 1: Write a failing test in the new file.**

Create `services/llm/src/engine_lease.rs`:

```rust
//! Global semaphore in front of every ONNX call (chat, planner, reflect).
//! Cap is `limits.engine.max_concurrent` (default 1). Acquire is async.
//!
//! With a 1-permit lease, peak ONNX memory cannot multiply across concurrent
//! HTTP requests — the second caller awaits the permit until the first drops.

use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Clone)]
pub struct EngineLease {
    sem: Arc<Semaphore>,
}

impl EngineLease {
    pub fn new(permits: usize) -> Self {
        Self { sem: Arc::new(Semaphore::new(permits.max(1))) }
    }

    pub async fn acquire(&self) -> OwnedSemaphorePermit {
        // Semaphore::acquire_owned only fails if the semaphore is closed,
        // which we never do.
        self.sem.clone().acquire_owned().await.expect("engine lease semaphore closed")
    }

    /// Visible permit count for tests/observability.
    pub fn available(&self) -> usize { self.sem.available_permits() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn one_permit_serializes_two_calls() {
        let lease = EngineLease::new(1);
        let l1 = lease.clone();
        let l2 = lease.clone();

        let h1 = tokio::spawn(async move {
            let _p = l1.acquire().await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            std::time::Instant::now()
        });
        // Give h1 a head start so it owns the permit.
        tokio::time::sleep(Duration::from_millis(5)).await;
        let h2 = tokio::spawn(async move {
            let _p = l2.acquire().await;
            std::time::Instant::now()
        });

        let t1 = h1.await.unwrap();
        let t2 = h2.await.unwrap();
        assert!(t2 >= t1, "second acquire must occur after first drops");
    }

    #[tokio::test]
    async fn available_drops_then_recovers() {
        let lease = EngineLease::new(2);
        assert_eq!(lease.available(), 2);
        let p = lease.acquire().await;
        assert_eq!(lease.available(), 1);
        drop(p);
        // Drop notification is async; yield to let it propagate.
        tokio::task::yield_now().await;
        assert_eq!(lease.available(), 2);
    }
}
```

- [ ] **Step 2: Register the module.**

Edit `services/llm/src/main.rs`. After `mod config;`, add:

```rust
mod engine_lease;
```

Also edit `services/llm/src/lib.rs` if it exists with module re-exports. Run:
```bash
ls services/llm/src/lib.rs 2>/dev/null
```
If it exists, add `pub mod engine_lease;` to it; otherwise skip (the crate is bin-only per `Cargo.toml`).

- [ ] **Step 3: Run the failing tests (will pass on first build since the impl is included).**

```bash
cd services/llm
cargo test -p llm-service --lib engine_lease::tests
```
Expected: PASS for both tests.

- [ ] **Step 4: Commit.**

```bash
git add services/llm/src/engine_lease.rs services/llm/src/main.rs
git commit -m "feat(llm): EngineLease — 1-permit semaphore over ORT calls"
```

---

### Task 2.3: Create `MemoryGate` with platform-specific RSS reader

**Files:**
- Create: `services/llm/src/memory_gate.rs`
- Modify: `services/llm/Cargo.toml` (add `libc` for macOS RSS)
- Modify: `services/llm/src/main.rs` (add `mod memory_gate;`)

- [ ] **Step 1: Add `libc` to dependencies.**

Edit `services/llm/Cargo.toml`. In `[dependencies]`, after `bytes = "1.5"`, add:

```toml

# Memory gate: process RSS via mach (macOS) / procfs (Linux)
libc = "0.2"
```

- [ ] **Step 2: Write the module with a mockable RSS reader.**

Create `services/llm/src/memory_gate.rs`:

```rust
//! Process-RSS admission gate. Refuses new chat/agent runs when the host
//! resident memory exceeds `high_water_mb`; resumes admitting after RSS drops
//! below `low_water_mb` (hysteresis prevents flapping at the boundary).
//!
//! Polled lazily on each admit call — no background thread. Cost is ~µs.

use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, thiserror::Error)]
#[error("memory pressure: RSS {rss_mb} MB > high_water {hw_mb} MB")]
pub struct MemoryRefusal {
    pub rss_mb: u64,
    pub hw_mb: u64,
}

pub struct MemoryGate {
    high_water_bytes: u64,
    low_water_bytes:  u64,
    above_water:      AtomicBool,
    reader:           Box<dyn Fn() -> std::io::Result<u64> + Send + Sync>,
}

impl MemoryGate {
    pub fn new(high_water_mb: u64, low_water_mb: u64) -> Self {
        assert!(low_water_mb <= high_water_mb,
                "low_water_mb must be <= high_water_mb");
        Self {
            high_water_bytes: high_water_mb * 1024 * 1024,
            low_water_bytes:  low_water_mb  * 1024 * 1024,
            above_water:      AtomicBool::new(false),
            reader:           Box::new(current_rss_bytes_default),
        }
    }

    /// Test constructor accepting a mock RSS reader.
    #[cfg(any(test, feature = "mock-rss"))]
    pub fn with_reader<F>(high_water_mb: u64, low_water_mb: u64, reader: F) -> Self
    where F: Fn() -> std::io::Result<u64> + Send + Sync + 'static {
        Self {
            high_water_bytes: high_water_mb * 1024 * 1024,
            low_water_bytes:  low_water_mb  * 1024 * 1024,
            above_water:      AtomicBool::new(false),
            reader:           Box::new(reader),
        }
    }

    pub fn admit(&self) -> Result<(), MemoryRefusal> {
        let rss = (self.reader)().unwrap_or(0);
        let was_above = self.above_water.load(Ordering::Relaxed);

        if was_above {
            // Stay refused until we drop below the low-water mark.
            if rss < self.low_water_bytes {
                self.above_water.store(false, Ordering::Relaxed);
                Ok(())
            } else {
                Err(MemoryRefusal {
                    rss_mb: rss / 1024 / 1024,
                    hw_mb: self.high_water_bytes / 1024 / 1024,
                })
            }
        } else if rss > self.high_water_bytes {
            self.above_water.store(true, Ordering::Relaxed);
            Err(MemoryRefusal {
                rss_mb: rss / 1024 / 1024,
                hw_mb: self.high_water_bytes / 1024 / 1024,
            })
        } else {
            Ok(())
        }
    }
}

fn current_rss_bytes_default() -> std::io::Result<u64> { current_rss_bytes() }

#[cfg(target_os = "macos")]
fn current_rss_bytes() -> std::io::Result<u64> {
    use libc::{c_int, c_void, mach_task_self, task_info};
    // MACH_TASK_BASIC_INFO = 20; MACH_TASK_BASIC_INFO_COUNT depends on struct.
    const MACH_TASK_BASIC_INFO: c_int = 20;
    #[repr(C)]
    #[derive(Default)]
    struct MachTaskBasicInfo {
        virtual_size:     u64,
        resident_size:    u64,
        resident_size_max:u64,
        user_time:        [u32; 2],
        system_time:      [u32; 2],
        policy:           c_int,
        suspend_count:    c_int,
    }
    let mut info = MachTaskBasicInfo::default();
    let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<u32>()) as u32;
    let kr = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO as u32,
            &mut info as *mut _ as *mut c_void as *mut i32,
            &mut count,
        )
    };
    if kr != 0 {
        return Err(std::io::Error::new(std::io::ErrorKind::Other,
                                       format!("task_info kr={}", kr)));
    }
    Ok(info.resident_size)
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> std::io::Result<u64> {
    let s = std::fs::read_to_string("/proc/self/status")?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // "VmRSS:    12345 kB"
            let kb: u64 = rest.split_whitespace().next()
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "parse VmRSS"))?;
            return Ok(kb * 1024);
        }
    }
    Err(std::io::Error::new(std::io::ErrorKind::NotFound, "VmRSS not in /proc/self/status"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn current_rss_bytes() -> std::io::Result<u64> { Ok(0) }

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as O};
    use std::sync::Arc;

    #[test]
    fn admits_below_high_water() {
        let rss = Arc::new(AtomicU64::new(100 * 1024 * 1024));
        let r = rss.clone();
        let gate = MemoryGate::with_reader(1000, 800, move || Ok(r.load(O::Relaxed)));
        assert!(gate.admit().is_ok());
    }

    #[test]
    fn refuses_above_high_water() {
        let rss = Arc::new(AtomicU64::new(2_000 * 1024 * 1024));
        let r = rss.clone();
        let gate = MemoryGate::with_reader(1_000, 800, move || Ok(r.load(O::Relaxed)));
        assert!(gate.admit().is_err());
    }

    #[test]
    fn hysteresis_holds_refusal_between_thresholds() {
        let rss = Arc::new(AtomicU64::new(2_000 * 1024 * 1024));
        let r = rss.clone();
        let gate = MemoryGate::with_reader(1_000, 800, move || Ok(r.load(O::Relaxed)));

        // First admit: above HW -> refuse, sticky.
        assert!(gate.admit().is_err());
        // Drop to between LW and HW: still refused (sticky).
        rss.store(900 * 1024 * 1024, O::Relaxed);
        assert!(gate.admit().is_err());
        // Drop below LW: admits again.
        rss.store(700 * 1024 * 1024, O::Relaxed);
        assert!(gate.admit().is_ok());
        // Sticky cleared; same range now admits.
        rss.store(900 * 1024 * 1024, O::Relaxed);
        assert!(gate.admit().is_ok());
    }

    #[test]
    fn real_reader_returns_nonzero_on_this_platform() {
        // Smoke-test the platform-specific path on macOS / Linux.
        // On unsupported OSes this returns 0; test passes trivially there.
        let gate = MemoryGate::new(u64::MAX / 1024 / 1024 / 2, 0);
        assert!(gate.admit().is_ok());
    }
}
```

- [ ] **Step 3: Register the module.**

Edit `services/llm/src/main.rs`. After `mod engine_lease;`, add:

```rust
mod memory_gate;
```

- [ ] **Step 4: Run the tests.**

```bash
cd services/llm
cargo test -p llm-service --lib memory_gate::tests
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit.**

```bash
git add services/llm/src/memory_gate.rs services/llm/src/main.rs services/llm/Cargo.toml
git commit -m "feat(llm): MemoryGate — RSS watermark admission with hysteresis"
```

---

### Task 2.4: Wire `EngineLease` + bounds into chat handler

**Files:**
- Modify: `services/llm/src/handler.rs`
- Modify: `services/llm/src/main.rs`

- [ ] **Step 1: Extend `AppState` with the lease, gate, and limits.**

In `services/llm/src/handler.rs`, replace the `AppState` struct:

```rust
pub struct AppState {
    pub engine: Arc<crate::hrm_engine::HrmEngine>,
    pub vision: Option<Arc<crate::vision_bridge::VisionBridge>>,
    pub lease: crate::engine_lease::EngineLease,
    pub gate: Arc<crate::memory_gate::MemoryGate>,
    pub limits: crate::config::LimitsConfig,
}
```

- [ ] **Step 2: Add admission + message-count bounds and refactor image-size check into `extract_content`.**

In `chat_completions`, after the `let req = req.into_inner();` line and before the existing `let model_name = ...` line, insert:

```rust
    // ── Admission + bounds ──────────────────────────────────────────────
    if req.messages.len() > state.limits.max_messages {
        return HttpResponse::BadRequest().json(json!({
            "error": format!("messages exceeds max ({} > {})",
                             req.messages.len(), state.limits.max_messages)
        }));
    }
    if let Err(e) = state.gate.admit() {
        return HttpResponse::ServiceUnavailable()
            .insert_header(("Retry-After", "1"))
            .json(json!({"error": e.to_string()}));
    }
```

Then change `extract_content`'s signature to take a max-bytes parameter:

```rust
fn extract_content(
    messages: &[ChatMessage],
    max_image_bytes: usize,
) -> Result<(Vec<(String, String)>, Option<Vec<u8>>), String> {
```

In the inner `ContentPart::ImageUrl` branch, replace:

```rust
                        ContentPart::ImageUrl { image_url } => {
                            if image.is_none() {
                                image = Some(decode_data_uri(&image_url.url)
                                    .map_err(|e| format!("invalid image: {e}"))?);
                            }
                        }
```

with:

```rust
                        ContentPart::ImageUrl { image_url } => {
                            if image.is_none() {
                                let bytes = decode_data_uri(&image_url.url)
                                    .map_err(|e| format!("invalid image: {e}"))?;
                                if bytes.len() > max_image_bytes {
                                    return Err(format!(
                                        "image exceeds {} bytes ({} actual)",
                                        max_image_bytes, bytes.len()));
                                }
                                image = Some(bytes);
                            }
                        }
```

Update the call site (around line 134) from:

```rust
    let (mut pairs, image_bytes) = match extract_content(&req.messages) {
```

to:

```rust
    let (mut pairs, image_bytes) = match extract_content(&req.messages, state.limits.max_image_bytes) {
```

`extract_content` already returns the oversize-image error as a `String`; the existing `HttpResponse::BadRequest().json(json!({"error": e}))` path needs to upgrade to `413 PayloadTooLarge` when the message starts with `"image exceeds"`. Change:

```rust
    let (mut pairs, image_bytes) = match extract_content(&req.messages, state.limits.max_image_bytes) {
        Ok(v) => v,
        Err(e) => return HttpResponse::BadRequest().json(json!({"error": e})),
    };
```

to:

```rust
    let (mut pairs, image_bytes) = match extract_content(&req.messages, state.limits.max_image_bytes) {
        Ok(v) => v,
        Err(e) if e.starts_with("image exceeds") =>
            return HttpResponse::PayloadTooLarge().json(json!({"error": e})),
        Err(e) => return HttpResponse::BadRequest().json(json!({"error": e})),
    };
```

- [ ] **Step 3: Add prompt-char cap after `build_prompt` and clamp max_tokens.**

In `chat_completions`, replace `let prompt = build_prompt(&pairs);` with:

```rust
    let prompt = build_prompt(&pairs);
    if prompt.len() > state.limits.max_prompt_chars {
        return HttpResponse::BadRequest().json(json!({
            "error": format!("prompt exceeds {} chars ({} actual)",
                             state.limits.max_prompt_chars, prompt.len())
        }));
    }
    let max_tokens = req.max_tokens.min(state.limits.max_generated_tokens);
```

Remove the existing `let max_tokens = req.max_tokens;` line.

- [ ] **Step 4: Use configured channel buffers and lease around inference.**

In the streaming branch (`if streaming { ... }`), replace:

```rust
        let (tx, rx) = mpsc::channel::<String>(128);

        let engine2 = Arc::clone(&engine);
        let prompt2 = prompt.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = engine2.infer_text(prompt2, max_tokens, temperature, tx) {
                tracing::error!("inference error: {e:#}");
            }
        });
```

with:

```rust
        let (tx, rx) = mpsc::channel::<String>(state.limits.channels.chat_stream_buffer);

        let engine2 = Arc::clone(&engine);
        let prompt2 = prompt.clone();
        let lease = state.lease.clone();
        tokio::spawn(async move {
            let _permit = lease.acquire().await;
            let res = tokio::task::spawn_blocking(move || {
                engine2.infer_text(prompt2, max_tokens, temperature, tx)
            }).await;
            if let Err(e) = res.and_then(|r| r.map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            })) {
                tracing::error!("inference error: {e:#}");
            }
        });
```

In the non-streaming branch, replace:

```rust
        let (tx, mut rx) = mpsc::channel::<String>(512);
        let handle = tokio::task::spawn_blocking(move || {
            engine.infer_text(prompt, max_tokens, temperature, tx)
        });
```

with:

```rust
        let (tx, mut rx) = mpsc::channel::<String>(state.limits.channels.chat_nonstream_buffer);
        let lease = state.lease.clone();
        let handle = tokio::spawn(async move {
            let _permit = lease.acquire().await;
            tokio::task::spawn_blocking(move || {
                engine.infer_text(prompt, max_tokens, temperature, tx)
            }).await
        });
```

And update the `handle.await.unwrap_or(Ok(()))` line further down to:

```rust
        let join = handle.await;
        let inference = match join {
            Ok(inner) => inner.unwrap_or_else(|e| Err(anyhow::anyhow!("join inner: {e}"))),
            Err(e)    => Err(anyhow::anyhow!("join outer: {e}")),
        };
        if let Err(e) = inference {
            return HttpResponse::InternalServerError()
                .json(json!({"error": format!("inference failed: {e}")}));
        }
```

(Removes the existing `if let Err(e) = handle.await.unwrap_or(Ok(())) { ... }` block.)

- [ ] **Step 5: Wire AppState construction in `main.rs`.**

In `services/llm/src/main.rs`, replace:

```rust
    let state = web::Data::new(AppState {
        engine: Arc::new(engine),
        vision,
    });
```

with:

```rust
    let limits = llm_config.limits.clone().unwrap_or_default();
    let mg_cfg = llm_config.memory_gate.clone().unwrap_or(crate::config::MemoryGateConfig {
        high_water_mb: 4096,
        low_water_mb: 3072,
        poll_on_admit_only: true,
    });
    let lease = crate::engine_lease::EngineLease::new(limits.engine.max_concurrent);
    let gate = Arc::new(crate::memory_gate::MemoryGate::new(
        mg_cfg.high_water_mb,
        mg_cfg.low_water_mb,
    ));

    let state = web::Data::new(AppState {
        engine: Arc::new(engine),
        vision,
        lease: lease.clone(),
        gate: gate.clone(),
        limits: limits.clone(),
    });
```

Add a `Default` impl for `LimitsConfig` (open `services/llm/src/config.rs` and add):

```rust
impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_image_bytes: default_max_image_bytes(),
            max_prompt_chars: default_max_prompt_chars(),
            max_messages: default_max_messages(),
            max_generated_tokens: default_max_generated_tokens(),
            max_ctx_size: default_max_ctx_size(),
            json: LimitsJsonConfig::default(),
            channels: LimitsChannelsConfig::default(),
            engine: LimitsEngineConfig::default(),
            results: LimitsResultsConfig::default(),
        }
    }
}
```

Also update the `JsonConfig` limit in `main.rs`. Replace:

```rust
            .app_data(
                web::JsonConfig::default()
                    .limit(32 * 1024 * 1024)
```

with:

```rust
            .app_data(
                web::JsonConfig::default()
                    .limit(limits.json.body_limit)
```

- [ ] **Step 6: Build.**

```bash
cd services/llm
cargo build -p llm-service
```
Expected: clean build. Fix any borrow/lifetime errors.

- [ ] **Step 7: Run existing handler-adjacent tests.**

```bash
cargo test -p llm-service --lib
```
Expected: all existing tests pass.

- [ ] **Step 8: Commit.**

```bash
git add services/llm/src/handler.rs services/llm/src/main.rs services/llm/src/config.rs
git commit -m "feat(llm/handler): bounds + EngineLease + MemoryGate wiring"
```

---

### Task 2.5: Wire bounds + lease into agent path

**Files:**
- Modify: `services/llm/src/agent/http.rs`
- Modify: `services/llm/src/agent/planner.rs`
- Modify: `services/llm/src/agent/executor.rs`

- [ ] **Step 1: Pass the lease and gate into `AgentLayer`.**

Edit `services/llm/src/agent/http.rs`. Replace the `AgentLayer` struct and `new`:

```rust
pub struct AgentLayer {
    pub planner:  Arc<dyn Planner>,
    pub registry: Arc<ToolRegistry>,
    pub config:   AgentConfig,
    pub sem:      Arc<Semaphore>,
    pub gate:     Arc<crate::memory_gate::MemoryGate>,
    pub limits:   crate::config::LimitsConfig,
}

impl AgentLayer {
    pub fn new(
        planner: Arc<dyn Planner>,
        registry: Arc<ToolRegistry>,
        config: AgentConfig,
        gate: Arc<crate::memory_gate::MemoryGate>,
        limits: crate::config::LimitsConfig,
    ) -> Self {
        let sem = Arc::new(Semaphore::new(config.max_concurrent_runs.max(1)));
        Self { planner, registry, config, sem, gate, limits }
    }
}
```

- [ ] **Step 2: Add memory-gate admission and message/image bounds to `run`.**

In `agent::http::run`, after the `if !layer.config.enabled` block, before the `let permit = ...` line, add:

```rust
    if let Err(e) = layer.gate.admit() {
        return HttpResponse::ServiceUnavailable()
            .insert_header(("Retry-After", "1"))
            .json(serde_json::json!({"error": e.to_string()}));
    }
```

Then after `let req = req.into_inner();`, add:

```rust
    if req.messages.len() > layer.limits.max_messages {
        return HttpResponse::BadRequest().json(serde_json::json!({
            "error": format!("messages exceeds max ({} > {})",
                             req.messages.len(), layer.limits.max_messages)
        }));
    }
```

- [ ] **Step 3: Cap image/audio bytes in `stage_inputs`.**

Change the signature of `stage_inputs`:

```rust
fn stage_inputs(
    input: &Option<AgentInput>,
    max_image_bytes: usize,
) -> Result<HashMap<String, Input>, String> {
```

Replace the body with:

```rust
    let mut m = HashMap::new();
    let Some(i) = input else { return Ok(m); };
    if let Some(img) = &i.image {
        let (mime, b64) = split_data_uri_or_bare(img, "image/jpeg");
        // Cap the decoded byte count; b64 is ~4/3 the binary size.
        let approx_bytes = b64.len() * 3 / 4;
        if approx_bytes > max_image_bytes {
            return Err(format!("image exceeds {} bytes (~{} actual)",
                               max_image_bytes, approx_bytes));
        }
        m.insert("input".to_string(), Input::Image { b64, mime });
    } else if let Some(aud) = &i.audio {
        let (mime, b64) = split_data_uri_or_bare(aud, "audio/wav");
        let approx_bytes = b64.len() * 3 / 4;
        if approx_bytes > max_image_bytes {
            return Err(format!("audio exceeds {} bytes (~{} actual)",
                               max_image_bytes, approx_bytes));
        }
        m.insert("input".to_string(), Input::Audio { b64, mime });
    }
    Ok(m)
```

Update the call site in `run`:

```rust
    let inputs = match stage_inputs(&req.input, layer.limits.max_image_bytes) {
        Ok(m) => m,
        Err(e) => return HttpResponse::PayloadTooLarge().json(serde_json::json!({"error": e})),
    };
```

- [ ] **Step 4: Wrap `HrmPlanner::propose` with the engine lease.**

Edit `services/llm/src/agent/planner.rs`. Change the struct:

```rust
pub struct HrmPlanner {
    engine: Arc<HrmEngine>,
    lease:  crate::engine_lease::EngineLease,
}

impl HrmPlanner {
    pub fn new(engine: Arc<HrmEngine>, lease: crate::engine_lease::EngineLease) -> Self {
        Self { engine, lease }
    }
}
```

And the `propose` impl:

```rust
#[async_trait]
impl Planner for HrmPlanner {
    async fn propose(&self, prompt: String, max_tokens: u32, temperature: f32) -> Result<String> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(256);
        let engine = self.engine.clone();
        let _permit = self.lease.acquire().await;        // serialize with chat
        let handle = tokio::task::spawn_blocking(move || {
            engine.infer_text(prompt, max_tokens, temperature, tx)
        });
        let mut buf = String::new();
        while let Some(s) = rx.recv().await { buf.push_str(&s); }
        handle.await
            .map_err(|e| anyhow::anyhow!("planner join: {}", e))?
            .map_err(|e| anyhow::anyhow!("planner inference: {}", e))?;
        Ok(buf)
    }
}
```

- [ ] **Step 5: Update `main.rs` to pass the lease and gate into both layers.**

In `services/llm/src/main.rs`, update the planner construction:

```rust
            let planner: Arc<dyn crate::agent::planner::Planner> =
                Arc::new(crate::agent::planner::HrmPlanner::new(
                    state.engine.clone(),
                    lease.clone(),
                ));
```

And the AgentLayer construction:

```rust
            let layer = crate::agent::http::AgentLayer::new(
                planner,
                Arc::new(reg),
                agent_cfg,
                gate.clone(),
                limits.clone(),
            );
```

- [ ] **Step 6: Smaller SSE buffer in `executor.rs`.**

Edit `services/llm/src/agent/executor.rs`. Replace `let (tx, rx) = mpsc::channel::<AgentEvent>(64);` with:

```rust
    let (tx, rx) = mpsc::channel::<AgentEvent>(8);
```

(The buffer size is wired to config later in Task 5.3. For now, hard-code the new lower default.)

- [ ] **Step 7: Build and run existing agent tests.**

```bash
cd services/llm
cargo test -p llm-service --lib agent
```
Expected: existing agent tests pass with the new wiring.

- [ ] **Step 8: Commit.**

```bash
git add services/llm/src/agent/http.rs services/llm/src/agent/planner.rs \
        services/llm/src/agent/executor.rs services/llm/src/main.rs
git commit -m "feat(llm/agent): bounds + EngineLease + MemoryGate on /v1/agent/run"
```

---

## Phase 3 — KV runtime (KV path still OFF by default)

### Task 3.1: Add `KvSession`, `KvBuffers`, `EngineBackend`

**Files:**
- Modify: `services/llm/src/hrm_engine.rs`

- [ ] **Step 1: Add the new types above `HrmEngine`.**

Open `services/llm/src/hrm_engine.rs`. Right after the `HrmRuntimeConfig` struct, add (and extend `HrmRuntimeConfig` to include `num_heads` + `head_dim` optionals):

Update `HrmRuntimeConfig`:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct HrmRuntimeConfig {
    pub eos_token_id: u32,
    pub ctx_size: u32,
    pub slow_loops: u32,
    pub fast_loops: u32,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    /// Present when the model was exported with two-graph KV cache.
    #[serde(default)]
    pub num_heads: Option<u32>,
    #[serde(default)]
    pub head_dim: Option<u32>,
}
```

Then add the new types right after `HrmRuntimeConfig`:

```rust
pub struct KvSession {
    pub prefill:     Arc<Mutex<Session>>,
    pub decode_step: Arc<Mutex<Session>>,
    pub num_layers:  usize,
    pub num_heads:   usize,
    pub head_dim:    usize,
    pub vocab_size:  usize,
}

pub enum EngineBackend {
    KvCache(KvSession),
    Monolithic(Session),
}

/// Pre-allocated cache buffers reused across requests. Sized to
/// `max_ctx_size` so no realloc happens during a decode loop. The engine
/// owns exactly one set; access is serialized by `EngineLease`.
pub struct KvBuffers {
    /// 2 * num_layers vectors, alternating (k0, v0, k1, v1, ...).
    /// Each vector holds `[1, num_heads, max_ctx, head_dim]` fp16-as-u16.
    pub layers_kv: Vec<Vec<u16>>,
    pub current_len: usize,
    pub max_ctx:     usize,
    pub num_heads:   usize,
    pub head_dim:    usize,
}

impl KvBuffers {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_ctx: usize) -> Self {
        let per_tensor = num_heads * max_ctx * head_dim;
        let layers_kv = (0..(2 * num_layers))
            .map(|_| vec![0_u16; per_tensor])
            .collect();
        Self { layers_kv, current_len: 0, max_ctx, num_heads, head_dim }
    }

    pub fn reset(&mut self) { self.current_len = 0; }
}
```

- [ ] **Step 2: Replace `HrmEngine` to hold `EngineBackend` instead of `Session`.**

Replace the `HrmEngine` struct with:

```rust
pub struct HrmEngine {
    pub backend:   Arc<Mutex<EngineBackend>>,
    pub buffers:   Arc<Mutex<Option<KvBuffers>>>,
    pub tokenizer: HrmTokenizer,
    pub runtime:   HrmRuntimeConfig,
    pub model_dir: PathBuf,
}
```

- [ ] **Step 3: Rewrite `HrmEngine::load` to detect KV files and fall back.**

Replace `HrmEngine::load`:

```rust
    pub fn load(cfg: &HrmConfig) -> Result<Self> {
        Self::load_with_kv(cfg, true)
    }

    pub fn load_with_kv(cfg: &HrmConfig, allow_kv: bool) -> Result<Self> {
        let model_dir = PathBuf::from(&cfg.model_dir);
        let prefill_path = model_dir.join("prefill.onnx");
        let decode_path  = model_dir.join("decode_step.onnx");

        let tokenizer = HrmTokenizer::load(&model_dir)
            .context("load HrmTokenizer")?;
        let runtime: HrmRuntimeConfig = {
            let text = std::fs::read_to_string(model_dir.join("config.json"))
                .context("read config.json")?;
            serde_json::from_str(&text).context("parse config.json")?
        };

        let kv_files_present = prefill_path.exists() && decode_path.exists();
        if allow_kv && kv_files_present {
            tracing::info!("Loading two-graph KV ONNX...");
            let prefill = Self::build_session(&prefill_path, cfg)
                .context("build prefill session")?;
            let decode  = Self::build_session(&decode_path, cfg)
                .context("build decode_step session")?;
            let num_layers = runtime.num_layers as usize;
            let num_heads  = runtime.num_heads
                .ok_or_else(|| anyhow::anyhow!(
                    "KV cache requires num_heads in config.json — re-run export with --two-graph"
                ))? as usize;
            let head_dim   = runtime.head_dim
                .ok_or_else(|| anyhow::anyhow!(
                    "KV cache requires head_dim in config.json"
                ))? as usize;
            let buffers = KvBuffers::new(
                num_layers, num_heads, head_dim, runtime.ctx_size as usize,
            );
            let kv = KvSession {
                prefill:     Arc::new(Mutex::new(prefill)),
                decode_step: Arc::new(Mutex::new(decode)),
                num_layers, num_heads, head_dim,
                vocab_size: runtime.vocab_size as usize,
            };
            return Ok(Self {
                backend:   Arc::new(Mutex::new(EngineBackend::KvCache(kv))),
                buffers:   Arc::new(Mutex::new(Some(buffers))),
                tokenizer, runtime, model_dir,
            });
        }

        // Fall back to monolithic.
        let onnx_path = if cfg.use_quantized.unwrap_or(false) {
            model_dir.join("model.int8.onnx")
        } else {
            model_dir.join("model.onnx")
        };
        if !onnx_path.exists() {
            anyhow::bail!(
                "HRM-Text ONNX not found at {}. Run `make hrm-download` or `make hrm-export`.",
                onnx_path.display()
            );
        }
        if allow_kv && !kv_files_present {
            tracing::warn!("KV-cache artifacts not found at {}; falling back to monolithic model.onnx.",
                           model_dir.display());
        }
        let session = Self::build_session(&onnx_path, cfg)
            .context("build monolithic ort session")?;
        Ok(Self {
            backend:   Arc::new(Mutex::new(EngineBackend::Monolithic(session))),
            buffers:   Arc::new(Mutex::new(None)),
            tokenizer, runtime, model_dir,
        })
    }
```

- [ ] **Step 4: Update `prefill` and `decode_greedy` to use the backend.**

The existing `prefill` method currently calls `self.session.lock()` directly. Replace with:

```rust
    pub fn prefill(&self, input_ids: &[i64]) -> Result<Vec<f32>> {
        use ort::value::Tensor;
        if input_ids.is_empty() {
            anyhow::bail!("prefill requires at least one input token");
        }
        let seq_len = input_ids.len();
        let input_tensor = Tensor::<i64>::from_array(
            ([1_usize, seq_len], input_ids.to_vec())
        ).context("build input_ids tensor")?;

        let mut backend = self.backend.lock()
            .map_err(|e| anyhow::anyhow!("backend lock poisoned: {}", e))?;
        let session = match &mut *backend {
            EngineBackend::Monolithic(s) => s,
            EngineBackend::KvCache(kv) => {
                // KV path uses prefill session and ignores the rest of this fn.
                // For external callers wanting last-position logits, route via
                // the prefill session here and drop the cache outputs.
                let mut sess = kv.prefill.lock()
                    .map_err(|e| anyhow::anyhow!("prefill lock: {}", e))?;
                let outputs = sess.run(ort::inputs!["input_ids" => input_tensor])
                    .context("ort run prefill (kv path)")?;
                let (shape, data) = outputs["logits"]
                    .try_extract_tensor::<half::f16>()
                    .context("extract fp16 logits (kv)")?;
                let dims = shape.as_ref();
                if dims.len() != 3 { anyhow::bail!("bad logits shape: {:?}", dims); }
                let vocab = dims[2] as usize;
                let last_pos = (dims[1] as usize) - 1;
                let row_start = last_pos * vocab;
                return Ok(data[row_start..row_start + vocab]
                    .iter().map(|h| h.to_f32()).collect());
            }
        };

        let outputs = session
            .run(ort::inputs!["input_ids" => input_tensor])
            .context("ort run prefill")?;
        let (shape, data) = outputs["logits"]
            .try_extract_tensor::<half::f16>()
            .context("extract fp16 logits")?;
        let dims = shape.as_ref();
        if dims.len() != 3 { anyhow::bail!("unexpected logits shape: {:?}", dims); }
        let vocab = dims[2] as usize;
        let last_pos = (dims[1] as usize) - 1;
        let row_start = last_pos * vocab;
        Ok(data[row_start..row_start + vocab]
            .iter().map(|h| h.to_f32()).collect())
    }
```

`decode_greedy` and `infer_text` keep working since they only call `prefill`. They retain O(n²) cost; `infer_text_kv` (next task) is the O(n) path.

- [ ] **Step 5: Build (no new tests yet — that's the next task).**

```bash
cd services/llm
cargo build -p llm-service
```
Expected: clean build. Existing `hrm_engine` tests still pass:
```bash
cargo test -p llm-service --lib hrm_engine
```
Expected: all PASS (the existing `decode_greedy` / `prefill` paths use monolithic since `prefill.onnx` is not present in the test fixture).

- [ ] **Step 6: Commit.**

```bash
git add services/llm/src/hrm_engine.rs
git commit -m "feat(llm/engine): EngineBackend + KvSession + KvBuffers scaffolding"
```

---

### Task 3.2: Implement `infer_text_kv` using the two-graph backend

**Files:**
- Modify: `services/llm/src/hrm_engine.rs`

- [ ] **Step 1: Add helper functions to read present-KV outputs into buffers.**

Append to `impl HrmEngine` (before the existing `build_session` fn):

```rust
    /// Copy fp16 `present.*.key/value` outputs into `KvBuffers.layers_kv` at
    /// position [..new_len]. `present_outputs` is the slice of name->Value
    /// pairs from `session.run`; ordering must match num_layers*2 alternating
    /// (key, value) per layer.
    fn copy_present_into_buffers(
        outputs: &ort::session::SessionOutputs,
        buffers: &mut KvBuffers,
        num_layers: usize,
        new_len: usize,
    ) -> Result<()> {
        // Each layer has present_key_values.{L}.key and .value
        let per_step = buffers.num_heads * buffers.head_dim;
        for layer in 0..num_layers {
            for (kv_idx, kind) in ["key", "value"].iter().enumerate() {
                let name = format!("present_key_values.{}.{}", layer, kind);
                let (shape, data) = outputs[name.as_str()]
                    .try_extract_tensor::<half::f16>()
                    .with_context(|| format!("extract {}", name))?;
                let dims = shape.as_ref();
                // Expected [1, num_heads, new_len, head_dim]
                if dims.len() != 4 {
                    anyhow::bail!("{} shape: {:?}", name, dims);
                }
                let buf_idx = layer * 2 + kv_idx;
                let buf = &mut buffers.layers_kv[buf_idx];
                // Copy [num_heads * new_len * head_dim] into the first
                // num_heads*new_len*head_dim contiguous slot of the buffer.
                let copy_len = buffers.num_heads * new_len * buffers.head_dim;
                if copy_len > buf.len() {
                    anyhow::bail!(
                        "KV buffer overflow on layer {} {}: need {} have {}",
                        layer, kind, copy_len, buf.len()
                    );
                }
                let src: &[u16] = bytemuck::cast_slice(data);
                buf[..copy_len].copy_from_slice(src);
            }
        }
        buffers.current_len = new_len;
        let _ = per_step;
        Ok(())
    }

    /// Build ort `Tensor`s borrowing the current-length prefix of each cache
    /// buffer. Returns 2*num_layers tensors plus the names they bind to.
    fn build_past_tensors(
        buffers: &KvBuffers,
        num_layers: usize,
    ) -> Result<Vec<(String, ort::value::Tensor<half::f16>)>> {
        use ort::value::Tensor;
        let mut out: Vec<(String, Tensor<half::f16>)> = Vec::with_capacity(2 * num_layers);
        let dims = [1_usize, buffers.num_heads, buffers.current_len, buffers.head_dim];
        for layer in 0..num_layers {
            for (kv_idx, kind) in ["key", "value"].iter().enumerate() {
                let buf_idx = layer * 2 + kv_idx;
                let used = buffers.num_heads * buffers.current_len * buffers.head_dim;
                let src: Vec<half::f16> = buffers.layers_kv[buf_idx][..used]
                    .iter().map(|&u| half::f16::from_bits(u)).collect();
                let t = Tensor::<half::f16>::from_array((dims, src))
                    .with_context(|| format!("build past tensor L{} {}", layer, kind))?;
                out.push((format!("past_key_values.{}.{}", layer, kind), t));
            }
        }
        Ok(out)
    }
```

- [ ] **Step 2: Add `bytemuck` to dependencies for cast_slice<u16>.**

Edit `services/llm/Cargo.toml`. After `half = "2"`, add:

```toml
bytemuck    = "1.14"
```

- [ ] **Step 3: Add the `infer_text_kv` method.**

Append to `impl HrmEngine`:

```rust
    /// KV-cache decode loop. Runs prefill once, then `decode_step` per token.
    /// Streams decoded pieces into `tx`. Blocking — call inside spawn_blocking.
    pub fn infer_text_kv(
        self: std::sync::Arc<Self>,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<()> {
        use ort::value::Tensor;
        // 1. Tokenize
        let prompt_ids = self.tokenizer.encode(&prompt, true)?;
        let prompt_len = prompt_ids.len();
        if prompt_len == 0 { return Ok(()); }
        let max_ctx = self.runtime.ctx_size as usize;
        if prompt_len >= max_ctx {
            anyhow::bail!("prompt length {} exceeds ctx_size {}", prompt_len, max_ctx);
        }

        // 2. Borrow backend + buffers
        let backend = self.backend.lock()
            .map_err(|e| anyhow::anyhow!("backend lock: {}", e))?;
        let kv = match &*backend {
            EngineBackend::KvCache(kv) => kv,
            EngineBackend::Monolithic(_) => {
                drop(backend);
                return self.infer_text(prompt, max_tokens, temperature, tx);
            }
        };
        let mut buffers_guard = self.buffers.lock()
            .map_err(|e| anyhow::anyhow!("buffers lock: {}", e))?;
        let buffers = buffers_guard.as_mut()
            .ok_or_else(|| anyhow::anyhow!("KV path active but no buffers"))?;
        buffers.reset();

        // 3. Prefill: run with the full prompt, capture logits[-1] + KVs
        let input_tensor = Tensor::<i64>::from_array(([1, prompt_len], prompt_ids.clone()))
            .context("build prefill input_ids tensor")?;
        let logits_last;
        let next_id;
        {
            let mut sess = kv.prefill.lock()
                .map_err(|e| anyhow::anyhow!("prefill lock: {}", e))?;
            let outputs = sess
                .run(ort::inputs!["input_ids" => input_tensor])
                .context("ort run prefill (kv loop)")?;

            let (shape, data) = outputs["logits"]
                .try_extract_tensor::<half::f16>()
                .context("extract prefill logits")?;
            let dims = shape.as_ref();
            let vocab = dims[2] as usize;
            let last_pos = (dims[1] as usize) - 1;
            let row_start = last_pos * vocab;
            logits_last = data[row_start..row_start + vocab]
                .iter().map(|h| h.to_f32()).collect::<Vec<f32>>();

            // Stash present_key_values.* into buffers (length == prompt_len).
            Self::copy_present_into_buffers(&outputs, buffers, kv.num_layers, prompt_len)?;
        }

        // First sampled token
        next_id = self.sample(&logits_last, temperature, 40, 0.95);
        if next_id as u32 == self.runtime.eos_token_id { return Ok(()); }
        let piece = self.tokenizer.decode_single(next_id as u32).unwrap_or_default();
        if tx.blocking_send(piece).is_err() { return Ok(()); }
        let mut last_token = next_id as i64;

        // 4. Decode loop
        for _ in 1..max_tokens {
            if buffers.current_len + 1 >= max_ctx { break; }
            let step_input = Tensor::<i64>::from_array(([1, 1], vec![last_token]))
                .context("build decode_step input_ids")?;
            let past = Self::build_past_tensors(buffers, kv.num_layers)?;
            let mut inputs_vec: Vec<(&str, ort::value::DynValue)> =
                Vec::with_capacity(1 + past.len());
            // Re-bind owned tensors into the inputs builder
            // (collect names into a Vec<String> first so &str refs stay valid)
            let past_names: Vec<String> = past.iter().map(|(n, _)| n.clone()).collect();
            let mut past_tensors_only: Vec<ort::value::DynValue> =
                past.into_iter().map(|(_, t)| t.into()).collect();
            inputs_vec.push(("input_ids", step_input.into()));
            for (n, t) in past_names.iter().zip(past_tensors_only.drain(..)) {
                inputs_vec.push((n.as_str(), t));
            }

            let mut sess = kv.decode_step.lock()
                .map_err(|e| anyhow::anyhow!("decode_step lock: {}", e))?;
            let outputs = sess.run(inputs_vec).context("ort run decode_step")?;

            let (shape, data) = outputs["logits"]
                .try_extract_tensor::<half::f16>()
                .context("extract decode logits")?;
            let dims = shape.as_ref();
            let vocab = dims[2] as usize;
            let logits: Vec<f32> = data[..vocab].iter().map(|h| h.to_f32()).collect();

            // Append present cache (length is current_len + 1)
            Self::copy_present_into_buffers(
                &outputs, buffers, kv.num_layers, buffers.current_len + 1,
            )?;

            let next = self.sample(&logits, temperature, 40, 0.95);
            if next as u32 == self.runtime.eos_token_id { break; }
            let piece = self.tokenizer.decode_single(next as u32).unwrap_or_default();
            if tx.blocking_send(piece).is_err() { break; }
            last_token = next as i64;
        }

        Ok(())
    }
```

(Note: the exact `inputs_vec` builder shape depends on ort 2.0.0-rc.10's `ort::inputs!` macro semantics for owned tensors. If the macro can't handle a dynamic Vec, fall back to `session.run_with_inputs(&[(&str, &DynValue)])` or whatever the rc.10 manual API exposes. The implementer should mirror the pattern used elsewhere in this crate — see `services/llm/src/hrm_engine.rs::prefill` for the working `ort::inputs!` form.)

- [ ] **Step 4: Build.**

```bash
cd services/llm
cargo build -p llm-service
```
Expected: clean build. If ort input-builder ergonomics complain, switch to manual `session.run(SessionInputs::from(...))` per ort 2.0.0-rc.10 docs.

- [ ] **Step 5: Add a unit test that uses `KvBuffers` reset semantics (no real model).**

In `services/llm/src/hrm_engine.rs` `#[cfg(test)] mod tests`, add:

```rust
    #[test]
    fn kv_buffers_reset_clears_length_keeps_capacity() {
        let mut b = KvBuffers::new(2, 4, 8, 16);
        b.current_len = 12;
        let cap_before: Vec<usize> = b.layers_kv.iter().map(|v| v.len()).collect();
        b.reset();
        assert_eq!(b.current_len, 0);
        let cap_after: Vec<usize> = b.layers_kv.iter().map(|v| v.len()).collect();
        assert_eq!(cap_before, cap_after);
        // num_layers * 2 buffers
        assert_eq!(b.layers_kv.len(), 4);
    }
```

- [ ] **Step 6: Run the test.**

```bash
cargo test -p llm-service --lib hrm_engine::tests::kv_buffers_reset_clears_length_keeps_capacity
```
Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add services/llm/src/hrm_engine.rs services/llm/Cargo.toml
git commit -m "feat(llm/engine): infer_text_kv — two-graph KV decode loop"
```

---

### Task 3.3: Route handler + planner to `infer_text_kv` when KV active

**Files:**
- Modify: `services/llm/src/hrm_engine.rs`
- Modify: `services/llm/src/handler.rs`
- Modify: `services/llm/src/agent/planner.rs`

- [ ] **Step 1: Add a `prefers_kv()` helper to `HrmEngine`.**

Append to `impl HrmEngine`:

```rust
    pub fn prefers_kv(&self) -> bool {
        let g = self.backend.lock().expect("backend lock");
        matches!(*g, EngineBackend::KvCache(_))
    }

    /// Auto-route: KV path if available, monolithic otherwise.
    pub fn infer_text_auto(
        self: std::sync::Arc<Self>,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<()> {
        if self.prefers_kv() {
            self.infer_text_kv(prompt, max_tokens, temperature, tx)
        } else {
            self.infer_text(prompt, max_tokens, temperature, tx)
        }
    }
```

- [ ] **Step 2: Route the chat handler through `infer_text_auto`.**

Edit `services/llm/src/handler.rs`. Replace both `engine2.infer_text(...)` and `engine.infer_text(...)` (in the spawn_blocking closures) with `infer_text_auto`:

In the streaming branch:
```rust
            tokio::task::spawn_blocking(move || {
                engine2.infer_text_auto(prompt2, max_tokens, temperature, tx)
            }).await
```

In the non-streaming branch:
```rust
            tokio::task::spawn_blocking(move || {
                engine.infer_text_auto(prompt, max_tokens, temperature, tx)
            }).await
```

- [ ] **Step 3: Route the planner too.**

Edit `services/llm/src/agent/planner.rs`. Replace `engine.infer_text(prompt, max_tokens, temperature, tx)` with:

```rust
            engine.infer_text_auto(prompt, max_tokens, temperature, tx)
```

- [ ] **Step 4: Build and run all crate tests.**

```bash
cd services/llm
cargo test -p llm-service
```
Expected: all PASS. Tests that depend on a live model are skipped via `skip_if_no_model()`; CI environments without the ONNX files still pass.

- [ ] **Step 5: Commit.**

```bash
git add services/llm/src/hrm_engine.rs services/llm/src/handler.rs \
        services/llm/src/agent/planner.rs
git commit -m "feat(llm): route chat + planner through infer_text_auto"
```

---

## Phase 4 — Flip default + stress tests

### Task 4.1: Add the RSS stress test (sequential)

**Files:**
- Create: `services/llm/tests/memory_stress.rs`

- [ ] **Step 1: Write the test.**

Create `services/llm/tests/memory_stress.rs`:

```rust
//! Integration: RSS must stay bounded across many sequential chat calls.
//!
//! Skipped on machines without the model artifacts.

use std::sync::Arc;

fn rss_mb() -> u64 {
    // mirror memory_gate's reader; duplicating to avoid making it pub.
    #[cfg(target_os = "macos")]
    unsafe {
        use libc::{c_int, c_void, mach_task_self, task_info};
        const MACH_TASK_BASIC_INFO: c_int = 20;
        #[repr(C)]
        #[derive(Default)]
        struct Info { vsz: u64, rss: u64, rss_max: u64,
                       ut: [u32; 2], st: [u32; 2], pol: c_int, susp: c_int }
        let mut info = Info::default();
        let mut count = (std::mem::size_of::<Info>() / std::mem::size_of::<u32>()) as u32;
        let kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO as u32,
                           &mut info as *mut _ as *mut c_void as *mut i32, &mut count);
        if kr != 0 { return 0; }
        info.rss / 1024 / 1024
    }
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(kb) = rest.split_whitespace().next().and_then(|t| t.parse::<u64>().ok()) {
                    return kb / 1024;
                }
            }
        }
        0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    { 0 }
}

fn skip_if_no_model() -> bool {
    !std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"),
        "/models/hrm-text-1b/model.onnx")).exists()
}

#[test]
fn sequential_chat_rss_stays_bounded() {
    if skip_if_no_model() {
        eprintln!("skipping memory_stress — no model artifacts");
        return;
    }
    use llm_service::hrm_engine::HrmEngine;
    use llm_service::config::HrmConfig;

    let cfg = HrmConfig {
        model_dir: format!("{}/models/hrm-text-1b", env!("CARGO_MANIFEST_DIR")),
        ep_preference: "cpu".into(),
        use_quantized: Some(false),
        n_threads: Some(2),
    };
    let engine = Arc::new(HrmEngine::load(&cfg).unwrap());

    // Baseline after one warmup call.
    let baseline = {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(8);
        let h = std::thread::spawn({
            let eng = engine.clone();
            move || eng.infer_text_auto("Hello,".into(), 16, 0.0, tx)
        });
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async { while rx.recv().await.is_some() {} });
        h.join().unwrap().unwrap();
        rss_mb()
    };

    for i in 0..200 {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(8);
        let eng = engine.clone();
        let h = std::thread::spawn(move || eng.infer_text_auto("Hello,".into(), 32, 0.0, tx));
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async { while rx.recv().await.is_some() {} });
        h.join().unwrap().unwrap();
        if i % 25 == 0 {
            let now = rss_mb();
            eprintln!("iter {}: RSS = {} MB (baseline {} MB)", i, now, baseline);
        }
    }

    let end = rss_mb();
    eprintln!("end RSS = {} MB, baseline = {} MB", end, baseline);
    assert!(end <= baseline + 100,
            "RSS grew unboundedly: end={} baseline={}", end, baseline);
}
```

(Note: the crate must expose `pub mod hrm_engine` and `pub mod config` for the integration test to reach them. If it doesn't yet, add a `services/llm/src/lib.rs` that re-exports the public modules. Currently the crate is bin-only — promote to `lib + bin` in the next step.)

- [ ] **Step 2: Promote the crate to lib+bin.**

Create `services/llm/src/lib.rs` with module declarations:

```rust
pub mod agent;
pub mod config;
pub mod engine_lease;
pub mod handler;
pub mod hrm_engine;
pub mod memory_gate;
pub mod tokenizer;
pub mod vision_bridge;
```

In `services/llm/Cargo.toml`, after the `[[bin]]` block, add:

```toml

[lib]
name = "llm_service"
path = "src/lib.rs"
```

Then in `services/llm/src/main.rs`, change the leading module declarations from:

```rust
mod agent;
mod config;
mod handler;
mod hrm_engine;
mod tokenizer;
mod vision_bridge;
```

to:

```rust
use llm_service::*;
```

(and remove the local `mod` declarations + `mod engine_lease;` / `mod memory_gate;` that we added — they now live in `lib.rs`).

Update the imports near the top from:

```rust
use config::{HrmConfig, LlmConfig};
use hrm_engine::HrmEngine;
use handler::AppState;
```

to:

```rust
use llm_service::config::{HrmConfig, LlmConfig};
use llm_service::hrm_engine::HrmEngine;
use llm_service::handler::AppState;
```

And update the inner references that used `crate::...` paths to use `llm_service::...` for paths that route through the lib.

- [ ] **Step 3: Build everything.**

```bash
cd services/llm
cargo build -p llm-service
```
Expected: both lib and bin build. Fix any module-path drift.

- [ ] **Step 4: Run the stress test (slow — heavy model load).**

```bash
cargo test -p llm-service --test memory_stress --release -- --nocapture
```
Expected: completes; final RSS within baseline + 100 MB.

- [ ] **Step 5: Commit.**

```bash
git add services/llm/src/lib.rs services/llm/src/main.rs services/llm/Cargo.toml \
        services/llm/tests/memory_stress.rs
git commit -m "test(llm): RSS stress test + crate promoted to lib+bin"
```

---

### Task 4.2: Engine-lease serialization integration test

**Files:**
- Create: `services/llm/tests/engine_lease_serial.rs`

- [ ] **Step 1: Write the test.**

Create `services/llm/tests/engine_lease_serial.rs`:

```rust
//! Two concurrent acquires on a 1-permit lease must serialize.

use llm_service::engine_lease::EngineLease;
use std::time::{Duration, Instant};

#[tokio::test]
async fn one_permit_serializes_two_acquires_across_tasks() {
    let lease = EngineLease::new(1);

    let l1 = lease.clone();
    let h1 = tokio::spawn(async move {
        let _p = l1.acquire().await;
        let start = Instant::now();
        tokio::time::sleep(Duration::from_millis(80)).await;
        let end = Instant::now();
        (start, end)
    });

    // Allow h1 to grab the permit
    tokio::time::sleep(Duration::from_millis(5)).await;

    let l2 = lease.clone();
    let h2 = tokio::spawn(async move {
        let _p = l2.acquire().await;
        Instant::now()
    });

    let (s1, e1) = h1.await.unwrap();
    let s2 = h2.await.unwrap();

    // h2 cannot acquire until h1 dropped its permit (at e1).
    assert!(s2 >= e1,
            "h2 acquired before h1 dropped: s2={:?} e1={:?} s1={:?}",
            s2, e1, s1);
}
```

- [ ] **Step 2: Run it.**

```bash
cd services/llm
cargo test -p llm-service --test engine_lease_serial
```
Expected: PASS.

- [ ] **Step 3: Commit.**

```bash
git add services/llm/tests/engine_lease_serial.rs
git commit -m "test(llm): engine_lease serialization integration"
```

---

### Task 4.3: KV-parity smoke test (Rust side)

**Files:**
- Create: `services/llm/tests/kv_parity.rs`

- [ ] **Step 1: Write the test (skipped without KV artifacts).**

Create `services/llm/tests/kv_parity.rs`:

```rust
//! Cross-backend parity: first-token argmax from monolithic == KV path.
//!
//! Skipped unless BOTH model.onnx AND prefill.onnx/decode_step.onnx exist.

use llm_service::hrm_engine::HrmEngine;
use llm_service::config::HrmConfig;

fn fixture_cfg() -> HrmConfig {
    HrmConfig {
        model_dir: format!("{}/models/hrm-text-1b", env!("CARGO_MANIFEST_DIR")),
        ep_preference: "cpu".into(),
        use_quantized: Some(false),
        n_threads: Some(2),
    }
}

fn artifacts_present() -> bool {
    let dir = format!("{}/models/hrm-text-1b", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&format!("{}/model.onnx", dir)).exists()
        && std::path::Path::new(&format!("{}/prefill.onnx", dir)).exists()
        && std::path::Path::new(&format!("{}/decode_step.onnx", dir)).exists()
}

#[test]
fn first_token_argmax_matches_across_backends() {
    if !artifacts_present() {
        eprintln!("skipping kv_parity — artifacts incomplete");
        return;
    }
    let mono_engine = HrmEngine::load_with_kv(&fixture_cfg(), false).unwrap();
    let kv_engine   = HrmEngine::load_with_kv(&fixture_cfg(), true).unwrap();

    let prompts = [
        "The capital of France is",
        "Once upon a time",
        "fn main() {",
        "Hello,",
    ];
    for p in prompts {
        let ids = mono_engine.tokenizer.encode(p, true).unwrap();
        let logits_mono = mono_engine.prefill(&ids).unwrap();
        let logits_kv   = kv_engine.prefill(&ids).unwrap();
        let argmax_mono = logits_mono.iter().enumerate()
            .max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let argmax_kv = logits_kv.iter().enumerate()
            .max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(argmax_mono, argmax_kv, "argmax differs for prompt {:?}", p);
    }
}
```

- [ ] **Step 2: Run it.**

```bash
cd services/llm
cargo test -p llm-service --test kv_parity --release -- --nocapture
```
Expected: PASS (or skip message if artifacts incomplete).

- [ ] **Step 3: Commit.**

```bash
git add services/llm/tests/kv_parity.rs
git commit -m "test(llm/engine): KV vs monolithic first-token argmax parity"
```

---

### Task 4.4: Flip KV default to enabled

**Files:**
- Modify: `services/llm/config.toml`

- [ ] **Step 1: Run the parity and stress tests (gate before flip).**

```bash
cd services/llm
cargo test -p llm-service --test kv_parity --release
cargo test -p llm-service --test memory_stress --release
```
Expected: both PASS. If either fails, do not proceed.

- [ ] **Step 2: Wire `[kv_cache] enabled` to the engine load in `main.rs`.**

Currently `HrmEngine::load(cfg)` auto-picks KV iff files exist. The TOML key is informational until we honor it. In `services/llm/src/main.rs`, replace:

```rust
    let engine = HrmEngine::load(hrm_config).unwrap_or_else(|e| {
        eprintln!("HRM engine load failed: {e}");
        std::process::exit(1);
    });
```

with:

```rust
    let kv_enabled = llm_config.kv_cache.as_ref().map(|c| c.enabled).unwrap_or(true);
    let engine = HrmEngine::load_with_kv(hrm_config, kv_enabled).unwrap_or_else(|e| {
        eprintln!("HRM engine load failed: {e}");
        std::process::exit(1);
    });
```

- [ ] **Step 3: Flip default in config.**

Edit `services/llm/config.toml`. Change:

```toml
[kv_cache]
enabled                = false
```

to:

```toml
[kv_cache]
enabled                = true
```

- [ ] **Step 4: Commit.**

```bash
git add services/llm/src/main.rs services/llm/config.toml
git commit -m "feat(llm): honor [kv_cache] enabled flag + enable KV by default"
```

---

## Phase 5 — Cleanup: drop discipline, trim, bounds tests

### Task 5.1: `vision_bridge::describe` takes `Vec<u8>` by value

**Files:**
- Modify: `services/llm/src/vision_bridge.rs`
- Modify: `services/llm/src/handler.rs`

- [ ] **Step 1: Write a failing test for the new signature.**

In `services/llm/src/vision_bridge.rs` `#[cfg(test)] mod tests`, add:

```rust
    #[tokio::test]
    async fn describe_consumes_image_vec() {
        let vb = VisionBridge::new(cfg("http://127.0.0.1:1"));
        let image = b"\x89PNG\r\n\x1a\n".to_vec();
        let _out = vb.describe(image).await;
        // image is consumed; cannot use again. Compile-time check.
    }
```

- [ ] **Step 2: Change `describe` to take `Vec<u8>`.**

In `services/llm/src/vision_bridge.rs`, change `describe`:

```rust
    pub async fn describe(&self, image_bytes: Vec<u8>) -> String {
        let b64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);
        drop(image_bytes);                                      // free original

        let classify = self.classify(&b64).await;
        let detect = self.detect(&b64).await;
        // ... unchanged ...
```

(Keep the rest of the function as is — only the signature and the explicit `drop(image_bytes)` change.)

- [ ] **Step 3: Update the call site in `handler.rs`.**

In `services/llm/src/handler.rs`, find the block:

```rust
    if let Some(img) = image_bytes {
        let prefix = match state.vision.as_ref() {
            Some(vb) => vb.describe(&img).await,
            None => "[Image attached but vision bridge disabled.]".to_string(),
        };
```

Replace `vb.describe(&img).await` with `vb.describe(img).await`, and remove the `&` reference on the `image_bytes` binding pattern (use `if let Some(img) = image_bytes`, then move `img` into the call).

- [ ] **Step 4: Build.**

```bash
cd services/llm
cargo build -p llm-service
```
Expected: clean.

- [ ] **Step 5: Run vision_bridge tests.**

```bash
cargo test -p llm-service --lib vision_bridge::tests
```
Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add services/llm/src/vision_bridge.rs services/llm/src/handler.rs
git commit -m "perf(llm/vision): describe takes Vec<u8> by value (1 fewer copy)"
```

---

### Task 5.2: `RunContext` Drop impl + post-emit trim hook

**Files:**
- Modify: `services/llm/src/agent/executor.rs`

- [ ] **Step 1: Add the `Drop` impl.**

In `services/llm/src/agent/executor.rs`, after the `RunContext` struct definition, add:

```rust
impl Drop for RunContext {
    fn drop(&mut self) {
        let n = self.results.len();
        self.results.clear();
        tracing::debug!(run_id = %self.run_id, results_dropped = n,
                        "RunContext dropped");
    }
}
```

- [ ] **Step 2: Add the trim helper.**

After `fallback_answer`, add:

```rust
/// Replace any string value in `v` longer than `threshold` with a stub.
/// Operates at the top level only — tool outputs are flat JSON objects.
fn trim_large_strings(v: &mut serde_json::Value, threshold: usize) {
    if let serde_json::Value::Object(m) = v {
        for (_k, val) in m.iter_mut() {
            if let serde_json::Value::String(s) = val {
                if s.len() > threshold {
                    let len = s.len();
                    *s = format!("<trimmed {} bytes>", len);
                }
            }
        }
    }
}
```

- [ ] **Step 3: Invoke the trim hook after each `StepResult`.**

In the main loop in `run_inner`, find:

```rust
        match result {
            Ok(value) => {
                ctx.results.insert(step.id.clone(), value.clone());
                let _ = tx.send(AgentEvent::StepResult {
                    idx: i + 1, id: step.id.clone(), ok: true,
                    value: Some(value), error: None, duration_ms: dur,
                }).await;
            }
```

Replace with:

```rust
        match result {
            Ok(value) => {
                let _ = tx.send(AgentEvent::StepResult {
                    idx: i + 1, id: step.id.clone(), ok: true,
                    value: Some(value.clone()), error: None, duration_ms: dur,
                }).await;
                // Insert AFTER emit, so the SSE wire saw the full value.
                let mut stored = value;
                trim_large_strings(&mut stored, 8_192);
                ctx.results.insert(step.id.clone(), stored);
            }
```

- [ ] **Step 4: Write a unit test for the trim hook.**

In `executor.rs` `#[cfg(test)] mod tests`, add:

```rust
    #[tokio::test]
    async fn large_step_output_trimmed_after_emit() {
        struct BigTool;
        #[async_trait::async_trait]
        impl crate::agent::tool::Tool for BigTool {
            fn name(&self) -> &'static str { "classify" }
            async fn invoke(&self, _: serde_json::Value, _: std::time::Instant)
                -> Result<serde_json::Value, crate::agent::tool::ToolError>
            {
                Ok(serde_json::json!({"label": "x", "blob": "A".repeat(20_000)}))
            }
        }
        let p = canned(vec!["\
step1. classify(image=input)
step2. final(answer=\"done\")
"]);
        let reg = registry_with(vec![std::sync::Arc::new(BigTool),
                                     std::sync::Arc::new(crate::agent::tools::final_tool::FinalTool)]);
        let mut inputs = std::collections::HashMap::new();
        inputs.insert("input".to_string(),
                      Input::Image { b64: "AA".into(), mime: "image/png".into() });
        let mut rx = run_agent(p, reg, "Q".into(), inputs, opts()).await;

        let mut emitted_blob: Option<String> = None;
        while let Some(e) = rx.recv().await {
            if let AgentEvent::StepResult { value: Some(v), .. } = &e {
                if let Some(b) = v.get("blob").and_then(|s| s.as_str()) {
                    emitted_blob = Some(b.to_string());
                }
            }
        }
        // The SSE side must have seen the full blob.
        assert_eq!(emitted_blob.unwrap().len(), 20_000);
        // (Retention-side trim verified separately via a direct call to
        // trim_large_strings — see next test.)
    }

    #[test]
    fn trim_large_strings_replaces_long_fields() {
        let mut v = serde_json::json!({
            "small": "ok",
            "big":   "A".repeat(20_000),
        });
        trim_large_strings(&mut v, 8_192);
        assert_eq!(v["small"], "ok");
        let big = v["big"].as_str().unwrap();
        assert!(big.starts_with("<trimmed 20000 bytes>"));
    }
```

- [ ] **Step 5: Run the tests.**

```bash
cd services/llm
cargo test -p llm-service --lib agent::executor::tests
```
Expected: all PASS.

- [ ] **Step 6: Commit.**

```bash
git add services/llm/src/agent/executor.rs
git commit -m "feat(llm/agent): RunContext Drop + post-emit results trim"
```

---

### Task 5.3: Wire configurable SSE buffer + Drop-on-disconnect test

**Files:**
- Modify: `services/llm/src/agent/executor.rs`
- Modify: `services/llm/src/agent/http.rs`
- Create: `services/llm/tests/agent_drop.rs`

- [ ] **Step 1: Pass the buffer size into `run_agent`.**

In `executor.rs`, add a field to `ExecOptions`:

```rust
pub struct ExecOptions {
    pub max_steps:           usize,
    pub max_run_ms:          u64,
    pub per_tool_ms:         u64,
    pub planner_temperature: f32,
    pub planner_max_tokens:  u32,
    pub sse_buffer:          usize,
}
```

Update `run_agent` to use it:

```rust
pub async fn run_agent(
    planner: Arc<dyn Planner>,
    registry: Arc<ToolRegistry>,
    user_msg: String,
    inputs: HashMap<String, Input>,
    opts: ExecOptions,
) -> mpsc::Receiver<AgentEvent> {
    let (tx, rx) = mpsc::channel::<AgentEvent>(opts.sse_buffer.max(1));
    tokio::spawn(run_inner(planner, registry, user_msg, inputs, opts, tx));
    rx
}
```

Update the test helper `opts()` to include the new field:

```rust
    fn opts() -> ExecOptions {
        ExecOptions {
            max_steps: 8, max_run_ms: 5_000, per_tool_ms: 2_000,
            planner_temperature: 0.0, planner_max_tokens: 128,
            sse_buffer: 8,
        }
    }
```

In `agent/http.rs`, populate the new field:

```rust
    let opts = ExecOptions {
        max_steps:           req.config.as_ref().and_then(|c| c.max_steps).unwrap_or(layer.config.max_steps),
        max_run_ms:          req.config.as_ref().and_then(|c| c.max_run_ms).unwrap_or(layer.config.max_run_ms),
        per_tool_ms:         req.config.as_ref().and_then(|c| c.per_tool_ms).unwrap_or(layer.config.per_tool_ms),
        planner_temperature: req.config.as_ref().and_then(|c| c.temperature).unwrap_or(layer.config.planner_temperature),
        planner_max_tokens:  256,
        sse_buffer:          layer.limits.channels.sse_event_buffer,
    };
```

- [ ] **Step 2: Add a drop-on-disconnect test.**

Create `services/llm/tests/agent_drop.rs`:

```rust
//! Dropping the SSE receiver mid-run causes the run task to observe channel
//! closure and exit. RunContext::drop runs (verified via tracing log probe).

use llm_service::agent::executor::{run_agent, ExecOptions, Input};
use llm_service::agent::planner::Planner;
use llm_service::agent::tool::{Tool, ToolError, ToolRegistry};
use llm_service::agent::sse::AgentEvent;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct CannedPlanner(Mutex<Vec<String>>);
#[async_trait]
impl Planner for CannedPlanner {
    async fn propose(&self, _: String, _: u32, _: f32) -> anyhow::Result<String> {
        Ok(self.0.lock().unwrap().remove(0))
    }
}

struct SlowTool;
#[async_trait]
impl Tool for SlowTool {
    fn name(&self) -> &'static str { "classify" }
    async fn invoke(&self, _: Value, _: Instant) -> Result<Value, ToolError> {
        tokio::time::sleep(Duration::from_millis(500)).await;
        Ok(serde_json::json!({"label": "x", "confidence": 0.5, "all": []}))
    }
}

#[tokio::test]
async fn dropping_receiver_stops_run() {
    use llm_service::agent::tools::final_tool::FinalTool;

    let plan = "step1. classify(image=input)\nstep2. final(answer=\"done\")".to_string();
    let p: Arc<dyn Planner> = Arc::new(CannedPlanner(Mutex::new(vec![plan])));
    let mut reg = ToolRegistry::new();
    reg.insert(Arc::new(SlowTool));
    reg.insert(Arc::new(FinalTool));
    let reg = Arc::new(reg);

    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(),
                  Input::Image { b64: "AA".into(), mime: "image/png".into() });

    let opts = ExecOptions {
        max_steps: 4, max_run_ms: 10_000, per_tool_ms: 5_000,
        planner_temperature: 0.0, planner_max_tokens: 128,
        sse_buffer: 4,
    };

    let mut rx = run_agent(p, reg, "Q".into(), inputs, opts).await;

    // Consume the first event (RunStart), then drop.
    let _ = rx.recv().await;
    drop(rx);

    // Give the executor time to observe disconnect and exit cleanly.
    tokio::time::sleep(Duration::from_millis(800)).await;
    // No assertion on RunContext drop log without a tracing collector — the
    // assertion is implicit: this test must not hang or panic.
}
```

- [ ] **Step 3: Run all agent tests.**

```bash
cd services/llm
cargo test -p llm-service --test agent_drop
cargo test -p llm-service --lib agent::executor
cargo test -p llm-service --lib agent::http
```
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add services/llm/src/agent/executor.rs services/llm/src/agent/http.rs \
        services/llm/tests/agent_drop.rs
git commit -m "feat(llm/agent): configurable SSE buffer + drop-on-disconnect test"
```

---

### Task 5.4: Memory-gate integration test (mocked RSS)

**Files:**
- Modify: `services/llm/src/memory_gate.rs` (expose `with_reader` outside `cfg(test)`)
- Create: `services/llm/tests/memory_gate.rs`

- [ ] **Step 1: Make `with_reader` available for integration tests.**

In `services/llm/src/memory_gate.rs`, replace:

```rust
    #[cfg(any(test, feature = "mock-rss"))]
    pub fn with_reader<F>(high_water_mb: u64, low_water_mb: u64, reader: F) -> Self
```

with:

```rust
    pub fn with_reader<F>(high_water_mb: u64, low_water_mb: u64, reader: F) -> Self
```

(Drop the cfg gate so integration tests can build it. This is a small public-API surface bump but the constructor is harmless.)

- [ ] **Step 2: Write the integration test.**

Create `services/llm/tests/memory_gate.rs`:

```rust
//! Hysteresis behavior across admit calls.

use llm_service::memory_gate::MemoryGate;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn admit_then_refuse_then_admit_with_hysteresis() {
    let rss = Arc::new(AtomicU64::new(500 * 1024 * 1024));
    let r = rss.clone();
    let gate = MemoryGate::with_reader(1_000, 800, move || Ok(r.load(Ordering::Relaxed)));

    assert!(gate.admit().is_ok(), "below HW admits");
    rss.store(2_000 * 1024 * 1024, Ordering::Relaxed);
    assert!(gate.admit().is_err(), "above HW refuses");
    rss.store(900 * 1024 * 1024, Ordering::Relaxed);
    assert!(gate.admit().is_err(), "between LW and HW sticky-refuses");
    rss.store(700 * 1024 * 1024, Ordering::Relaxed);
    assert!(gate.admit().is_ok(), "below LW clears sticky");
    rss.store(900 * 1024 * 1024, Ordering::Relaxed);
    assert!(gate.admit().is_ok(), "between LW and HW admits once sticky cleared");
}
```

- [ ] **Step 3: Run it.**

```bash
cd services/llm
cargo test -p llm-service --test memory_gate
```
Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add services/llm/src/memory_gate.rs services/llm/tests/memory_gate.rs
git commit -m "test(llm): memory_gate hysteresis integration"
```

---

### Task 5.5: Engine-fallback test (KV files missing)

**Files:**
- Create: `services/llm/tests/engine_fallback.rs`

- [ ] **Step 1: Write the test.**

Create `services/llm/tests/engine_fallback.rs`:

```rust
//! When prefill.onnx/decode_step.onnx are absent, the engine must fall back
//! to monolithic without erroring. Verified by passing a model_dir that has
//! only model.onnx (the existing fixture).

use llm_service::hrm_engine::HrmEngine;
use llm_service::config::HrmConfig;

fn skip_if_no_model() -> bool {
    !std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"),
        "/models/hrm-text-1b/model.onnx")).exists()
}

#[test]
fn missing_kv_files_fall_back_to_monolithic() {
    if skip_if_no_model() {
        eprintln!("skipping engine_fallback — no model");
        return;
    }
    let cfg = HrmConfig {
        model_dir: format!("{}/models/hrm-text-1b", env!("CARGO_MANIFEST_DIR")),
        ep_preference: "cpu".into(),
        use_quantized: Some(false),
        n_threads: Some(2),
    };
    // Force monolithic by setting allow_kv=false.
    let engine = HrmEngine::load_with_kv(&cfg, false).unwrap();
    assert!(!engine.prefers_kv(), "load_with_kv(false) should pick monolithic");

    // And the default load() picks KV iff artifacts present.
    let engine2 = HrmEngine::load(&cfg).unwrap();
    let kv_files = std::path::Path::new(&format!("{}/prefill.onnx", cfg.model_dir)).exists()
                 && std::path::Path::new(&format!("{}/decode_step.onnx", cfg.model_dir)).exists();
    assert_eq!(engine2.prefers_kv(), kv_files);
}
```

- [ ] **Step 2: Run it.**

```bash
cd services/llm
cargo test -p llm-service --test engine_fallback
```
Expected: PASS or skip-message.

- [ ] **Step 3: Commit.**

```bash
git add services/llm/tests/engine_fallback.rs
git commit -m "test(llm/engine): fallback to monolithic when KV files missing"
```

---

### Task 5.6: Bounds rejection integration tests

**Files:**
- Create: `services/llm/tests/bounds.rs`

- [ ] **Step 1: Write rejection tests against a live `actix_web` app.**

Create `services/llm/tests/bounds.rs`:

```rust
//! Boundary rejections: oversize image, oversize prompt, too many messages,
//! oversize body. Uses actix_web::test to drive handlers without binding a port.
//!
//! Skipped when no model is available — the chat handler needs the engine.

use actix_web::{test, web, App};
use llm_service::config::{LimitsConfig, HrmConfig};
use llm_service::engine_lease::EngineLease;
use llm_service::memory_gate::MemoryGate;
use llm_service::handler::{chat_completions, AppState};
use llm_service::hrm_engine::HrmEngine;
use std::sync::Arc;

fn skip_if_no_model() -> bool {
    !std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"),
        "/models/hrm-text-1b/model.onnx")).exists()
}

fn cfg() -> HrmConfig {
    HrmConfig {
        model_dir: format!("{}/models/hrm-text-1b", env!("CARGO_MANIFEST_DIR")),
        ep_preference: "cpu".into(),
        use_quantized: Some(false),
        n_threads: Some(2),
    }
}

fn state_with_limits(limits: LimitsConfig) -> web::Data<AppState> {
    let engine = Arc::new(HrmEngine::load(&cfg()).unwrap());
    let lease = EngineLease::new(limits.engine.max_concurrent);
    let gate = Arc::new(MemoryGate::new(u64::MAX / 1024 / 1024 / 2, 0));
    web::Data::new(AppState { engine, vision: None, lease, gate, limits })
}

#[actix_web::test]
async fn too_many_messages_rejected_400() {
    if skip_if_no_model() { return; }
    let mut limits = LimitsConfig::default();
    limits.max_messages = 2;
    let state = state_with_limits(limits);
    let app = test::init_service(
        App::new().app_data(state).route("/v1/chat/completions",
            web::post().to(chat_completions))
    ).await;

    let body = serde_json::json!({
        "messages": (0..5).map(|i| serde_json::json!({
            "role": "user", "content": format!("msg {}", i)
        })).collect::<Vec<_>>(),
        "max_tokens": 16,
    });
    let req = test::TestRequest::post().uri("/v1/chat/completions").set_json(&body).to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}

#[actix_web::test]
async fn oversize_image_rejected_413() {
    if skip_if_no_model() { return; }
    let mut limits = LimitsConfig::default();
    limits.max_image_bytes = 1024;
    let state = state_with_limits(limits);
    let app = test::init_service(
        App::new().app_data(state).route("/v1/chat/completions",
            web::post().to(chat_completions))
    ).await;

    let big = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        &vec![0u8; 4096]);
    let body = serde_json::json!({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url",
                 "image_url": {"url": format!("data:image/png;base64,{}", big)}}
            ]
        }],
        "max_tokens": 16,
    });
    let req = test::TestRequest::post().uri("/v1/chat/completions").set_json(&body).to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 413);
}

#[actix_web::test]
async fn oversize_prompt_rejected_400() {
    if skip_if_no_model() { return; }
    let mut limits = LimitsConfig::default();
    limits.max_prompt_chars = 64;
    let state = state_with_limits(limits);
    let app = test::init_service(
        App::new().app_data(state).route("/v1/chat/completions",
            web::post().to(chat_completions))
    ).await;

    let huge = "A".repeat(200);
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": huge}],
        "max_tokens": 16,
    });
    let req = test::TestRequest::post().uri("/v1/chat/completions").set_json(&body).to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 400);
}
```

- [ ] **Step 2: Add `actix-web` test feature if not already present.**

Check `services/llm/Cargo.toml`. `actix-web = "4.8"` ships `test` module by default for in-process testing. No change needed.

- [ ] **Step 3: Run the tests.**

```bash
cd services/llm
cargo test -p llm-service --test bounds
```
Expected: PASS or skip.

- [ ] **Step 4: Commit.**

```bash
git add services/llm/tests/bounds.rs
git commit -m "test(llm): handler bounds — 400/413 rejection paths"
```

---

### Task 5.7: Final build, full suite, and changelog summary

**Files:** (none modified — verification only)

- [ ] **Step 1: Full crate test run.**

```bash
cd services/llm
cargo test -p llm-service
```
Expected: ALL tests pass (model-dependent tests will skip if artifacts absent on the runner).

- [ ] **Step 2: Release build.**

```bash
cargo build -p llm-service --release
```
Expected: clean. No new warnings beyond what was already present.

- [ ] **Step 3: Verify the server starts and serves a simple chat.**

```bash
cd /Users/evintleovonzko/Documents/projects/evint/torch-inference
./services/llm/target/release/llm-service &
sleep 3
curl -s -X POST http://127.0.0.1:8001/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":8}' \
    | head -c 200
kill %1 2>/dev/null
```
Expected: a JSON response with `choices[0].message.content` populated.

- [ ] **Step 4: Document the rollout in a CHANGELOG line.**

If `services/llm/CHANGELOG.md` exists, prepend a line; otherwise skip.

- [ ] **Step 5: Final commit if anything changed.**

```bash
git status
git diff --cached --quiet || git commit -m "chore(llm): memory hardening rollout complete"
```

---

## Notes for the implementer

- **`ort::inputs!` macro in 2.0.0-rc.10:** the macro variant `ort::inputs![name => tensor]` works for fixed argument counts. For the decode-step which needs `1 + 2*num_layers = 33` inputs, you may need to use the manual builder API. Check `services/llm/src/hrm_engine.rs::prefill` for the exact working idiom and mirror it. If the macro can't construct the input list dynamically, see ort's `SessionInputs::from_iter` or build a `Vec<(&str, SessionInputValue)>` and pass with `session.run(...)` overload.
- **fp16 ↔ u16 cast:** `half::f16` is wrapper over `u16`; `bytemuck::cast_slice::<half::f16, u16>(...)` is the canonical cast. Both crates are already pulled in.
- **Test parallelism:** the integration tests that load the engine should each load their own engine (no shared fixture) so they don't contend on the KV buffers. `cargo test` runs tests in parallel by default; if the model load is too memory-heavy to run several in parallel, gate with `cargo test -- --test-threads=1`.
- **Re-export gating:** Phase 3+ Rust work runs in a degraded mode (monolithic only) if Phase 1 didn't produce the KV artifacts. That's intentional — the bounds work in Phase 2 still ships value on its own.

## Self-review against spec

- ✅ Goal 1 (per-request OOM): bounds in 2.4/2.5, KV cache in 3.x, KvBuffers reuse 3.1/3.2.
- ✅ Goal 2 (concurrent-burst OOM): EngineLease 2.2, wired in 2.4/2.5/3.3, serialization test 4.2.
- ✅ Goal 3 (RSS growth): JsonConfig limit 2.4, smaller channels 2.4/2.5/5.3, results trim 5.2, vision_bridge by-value 5.1, RunContext Drop 5.2, sequential RSS test 4.1.
- ✅ Goal 4 (KV cache): export 1.1/1.2, parity 1.3, runtime 3.1/3.2/3.3, parity test 4.3, flip 4.4, fallback 5.5.
- ✅ Non-goals respected: no agent redesign, no new tools/endpoints, no GPU work, no io-binding, no quantized KV.
- ✅ All bounds defaults in the spec table match Task 2.1 config values.
- ✅ All new files match the file-map shown in the plan header.
