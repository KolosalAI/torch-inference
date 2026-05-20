# HRM-Text Runtime Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `services/llm/` inference engine so `POST /v1/chat/completions` is served by `sapientinc/HRM-Text-1B` via `ort`, while preserving the OpenAI-compatible API surface so the proxy, playground, and future agentic layer require no changes.

**Architecture:** Offline Python ONNX export (build-time only, never invoked at runtime) produces `model.onnx` + `tokenizer.json` + `config.json`. New Rust `HrmEngine` over `ort` + `tokenizers` drives an autoregressive decode loop. A small `vision_bridge` posts incoming chat images to the main server's `/classify/batch` and `/yolo/detect` endpoints, then injects a textual description into the prompt so text-only HRM-Text can still answer image questions.

**Tech Stack:** Rust (actix-web 4.8, tokio, tracing) · `ort` 2.0.0-rc.10 · `tokenizers` 0.20 · `ndarray` 0.16 · `reqwest` 0.12 · `serde`/`serde_json` · Python (offline build only) with `transformers`, `torch`, `onnx`, `onnxruntime`, `optimum`.

**Spec reference:** `docs/superpowers/specs/2026-05-20-hrm-text-runtime-swap-design.md` (commit `1a5fa4f`).

---

## Pre-flight — REQUIRED before any production code

### Task 0: ONNX exportability spike (gating)

Per spec §7. This task is run on a **throwaway branch** (`spike/hrm-onnx-export`) and produces an addendum document — no production code merges into `main` from this task.

**Outcome decides everything downstream.** If the spike fails, halt the plan and reissue Spec v2.

**Files:**
- Create: `docs/superpowers/specs/2026-05-20-hrm-text-export-spike.md` (addendum)
- Create (throwaway, on spike branch only): `spike/export_hrm_text.py`
- Create (throwaway, on spike branch only): `spike/decode_harness/Cargo.toml`, `spike/decode_harness/src/main.rs`

- [ ] **Step 1: Create the throwaway branch**

```bash
git checkout -b spike/hrm-onnx-export
mkdir -p spike/decode_harness/src
```

- [ ] **Step 2: Write the minimum ONNX export script**

Create `spike/export_hrm_text.py`:

```python
"""
HRM-Text -> ONNX exportability spike. Throwaway. Goal: produce a
loadable model.onnx (or fail loudly with the exact unsupported op).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "sapientinc/HRM-Text-1B"
OUT_DIR = "spike/out"

def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    model.train(False)  # inference mode (equivalent to .eval())

    # Smallest meaningful prompt: 4 tokens
    ids = tok("The capital of France", return_tensors="pt").input_ids

    print("Tracing forward pass...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (ids,),
            f"{OUT_DIR}/model.onnx",
            opset_version=17,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}},
            do_constant_folding=True,
        )

    tok.save_pretrained(OUT_DIR)
    print("Spike export complete ->", OUT_DIR)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the export in an isolated env**

```bash
uv venv spike/.venv
uv pip install --python spike/.venv 'transformers>=4.45' 'torch>=2.3' 'onnx>=1.16' 'onnxruntime>=1.18' optimum
spike/.venv/bin/python spike/export_hrm_text.py 2>&1 | tee spike/export.log
```

Expected outcomes:
- **CLEAN PASS:** `spike/out/model.onnx` exists, file size >500 MB. Continue to Step 4.
- **FA3 / SDPA failure:** error mentions unsupported attention op. Record exact op; revise script to swap attention to stock SDPA before export; retry. Up to 2 retries.
- **Dynamic control flow failure:** error mentions `aten::if` / `prim::Loop` or hierarchical recurrence ops. This is the **Candle fallback** signal — skip to Step 7.

- [ ] **Step 4: Write a Rust decode harness**

Create `spike/decode_harness/Cargo.toml`:

```toml
[package]
name = "decode-harness"
version = "0.0.1"
edition = "2021"

[dependencies]
ort = "=2.0.0-rc.10"
tokenizers = "0.20"
ndarray = "0.16"
anyhow = "1"
```

Create `spike/decode_harness/src/main.rs`:

```rust
use anyhow::Result;
use ndarray::Array2;
use ort::session::{Session, builder::GraphOptimizationLevel};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file("../out/model.onnx")?;

    let tok = Tokenizer::from_file("../out/tokenizer.json")
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;
    let enc = tok.encode("The capital of France is", true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;

    let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let arr = Array2::from_shape_vec((1, ids.len()), ids.clone())?;

    let outputs = session.run(ort::inputs![ "input_ids" => arr.view() ]?)?;
    let logits = outputs["logits"].try_extract_tensor::<f32>()?;
    println!("logits shape: {:?}", logits.shape());

    // Greedy: argmax of last position
    let last_row = logits.slice(ndarray::s![0, logits.shape()[1]-1, ..]);
    let next = last_row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0 as u32;

    let next_str = tok.decode(&[next], true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!("next token: {:?}  ({})", next_str, next);

    Ok(())
}
```

- [ ] **Step 5: Build and run the harness**

```bash
cd spike/decode_harness && cargo run --release
```

Expected: prints a non-empty next token; ideally `Paris` or a single space + capital letter. Anything plausibly English passes.

- [ ] **Step 6: Write the spike addendum (CLEAN PASS path)**

Create `docs/superpowers/specs/2026-05-20-hrm-text-export-spike.md`:

```markdown
# HRM-Text Export Spike — Addendum to Spec #1

**Date:** YYYY-MM-DD
**Result:** PASS — proceed with Spec #1 as written.
**Spike branch:** `spike/hrm-onnx-export` (commit <SHA>)

## What worked
- ONNX opset 17 export with `attn_implementation="sdpa"`
- Model size: <SIZE> MB (fp16)
- ort 2.0.0-rc.10 loads the graph; greedy decode of "The capital of France is" -> "<TOKEN>"

## Rewrites applied
- <list any attention/op rewrites needed>

## Loop counts (from upstream simple_inference_engine.py)
- slow_loops: <N>
- fast_loops: <M>

## Caveats for plan execution
- <any quirks the implementer must know>
```

Fill in `<SIZE>`, `<TOKEN>`, loop counts, and any caveats from your actual run.

- [ ] **Step 7: Write the spike addendum (CANDLE FALLBACK path) — only if Step 3 failed terminally**

Create `docs/superpowers/specs/2026-05-20-hrm-text-export-spike.md` with `Result: FAIL — Spec v2 required (Candle port)`. Document the failing op(s). **Halt the plan.** Notify Evintkoo; this requires reversing the "no Candle in prod" stance in CLAUDE.md and a re-spec.

- [ ] **Step 8: Merge the addendum to main (CLEAN PASS only)**

```bash
git checkout main
git checkout spike/hrm-onnx-export -- docs/superpowers/specs/2026-05-20-hrm-text-export-spike.md
git add docs/superpowers/specs/2026-05-20-hrm-text-export-spike.md
git commit -m "docs(spec): HRM-Text export spike addendum (CLEAN PASS)"
```

Do **not** merge the throwaway `spike/` directory. Leave the spike branch around for reference.

---

## Production implementation — gated on Task 0 = PASS

After Task 0 passes, all subsequent tasks land on `main`. Each task ends with `cargo build --release` succeeding inside `services/llm/`.

### Task 1: Add new dependencies (keep llama-cpp-2 for now)

**Files:**
- Modify: `services/llm/Cargo.toml`

- [ ] **Step 1: Add ort, tokenizers, ndarray, reqwest**

Edit `services/llm/Cargo.toml`. Add to `[dependencies]` (do **not** remove `llama-cpp-2` yet — we keep both engines compiling until the swap is complete):

```toml
# ONNX runtime — same version as the workspace root uses
ort         = "=2.0.0-rc.10"
tokenizers  = "0.20"
ndarray     = "0.16"
reqwest     = { version = "0.12", features = ["json", "rustls-tls"], default-features = false }
```

- [ ] **Step 2: Verify the build still passes**

```bash
cd services/llm && cargo build --release 2>&1 | tail -20
```

Expected: no errors. New crates downloaded, compiled. Build time ~2-3 min on a cold cache.

- [ ] **Step 3: Commit**

```bash
git add services/llm/Cargo.toml services/llm/Cargo.lock
git commit -m "deps(llm): add ort/tokenizers/ndarray/reqwest for HRM-Text engine"
```

---

### Task 2: Write the production ONNX export pipeline

**Files:**
- Create: `scripts/export_hrm_text.py`
- Create: `scripts/download_hrm_text_artifacts.sh`
- Modify: `Makefile:192-202` (LLM Microservice section)

- [ ] **Step 1: Write the production export script**

Create `scripts/export_hrm_text.py`. Use the spike addendum's loop counts and any rewrites it identified:

```python
"""
HRM-Text -> ONNX exporter (one-time, offline, build-time only).
Output: services/llm/models/hrm-text-1b/{model.onnx, tokenizer.json, config.json}

Run via:  make hrm-export
"""
import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "sapientinc/HRM-Text-1B"
OUT_DIR = "services/llm/models/hrm-text-1b"

# Defaults come from the spike addendum. Override on CLI if needed.
DEFAULT_SLOW_LOOPS = 2
DEFAULT_FAST_LOOPS = 4

def export(quantize: bool, slow_loops: int, fast_loops: int):
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa"
    )
    model.train(False)  # inference mode

    ids = tok("hello", return_tensors="pt").input_ids
    onnx_path = f"{OUT_DIR}/model.onnx"

    with torch.no_grad():
        torch.onnx.export(
            model, (ids,), onnx_path,
            opset_version=17,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}},
            do_constant_folding=True,
        )
    print(f"  -> {onnx_path}")

    tok.save_pretrained(OUT_DIR)
    # save_pretrained leaves extras we do not need; HrmEngine reads only tokenizer.json.

    hf_cfg = model.config.to_dict()
    runtime_cfg = {
        "eos_token_id": hf_cfg.get("eos_token_id"),
        "ctx_size": hf_cfg.get("max_position_embeddings", 2048),
        "slow_loops": slow_loops,
        "fast_loops": fast_loops,
        "vocab_size": hf_cfg.get("vocab_size"),
        "hidden_size": hf_cfg.get("hidden_size"),
        "num_layers": hf_cfg.get("num_hidden_layers"),
    }
    with open(f"{OUT_DIR}/config.json", "w") as f:
        json.dump(runtime_cfg, f, indent=2)
    print(f"  -> {OUT_DIR}/config.json")

    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q_path = f"{OUT_DIR}/model.int8.onnx"
        quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
        print(f"  -> {q_path} (int8)")

    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", action="store_true", help="also emit an int8 variant")
    ap.add_argument("--slow-loops", type=int, default=DEFAULT_SLOW_LOOPS)
    ap.add_argument("--fast-loops", type=int, default=DEFAULT_FAST_LOOPS)
    args = ap.parse_args()
    export(args.quantize, args.slow_loops, args.fast_loops)
```

- [ ] **Step 2: Write the artifact download script**

Create `scripts/download_hrm_text_artifacts.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Downloads pre-exported HRM-Text-1B artifacts from a GitHub Release on
# KolosalAI/torch-inference. Set RELEASE_TAG to pin a specific release.

RELEASE_TAG="${RELEASE_TAG:-hrm-text-1b-v0}"
ASSET_NAME="hrm-text-1b.tar.gz"
OUT_DIR="services/llm/models/hrm-text-1b"
URL="https://github.com/KolosalAI/torch-inference/releases/download/${RELEASE_TAG}/${ASSET_NAME}"

if [ -f "${OUT_DIR}/model.onnx" ]; then
    echo "Artifacts already present at ${OUT_DIR}/. Delete to re-download."
    exit 0
fi

mkdir -p "${OUT_DIR}"
echo "Downloading ${URL}..."
curl -L --fail --progress-bar -o /tmp/hrm-text-1b.tar.gz "${URL}"
tar -xzf /tmp/hrm-text-1b.tar.gz -C "${OUT_DIR}" --strip-components=1
rm /tmp/hrm-text-1b.tar.gz
echo "Done. Artifacts at ${OUT_DIR}/"
```

```bash
chmod +x scripts/download_hrm_text_artifacts.sh
```

- [ ] **Step 3: Add Makefile targets**

Edit `Makefile`. Replace the existing `# -- LLM Microservice --` block (lines 192-202) with:

```makefile
# -- LLM Microservice --------------------------------------------------------
.PHONY: llm-build llm-run llm-download hrm-export hrm-download

hrm-download: ## Download pre-exported HRM-Text-1B ONNX artifacts
	bash scripts/download_hrm_text_artifacts.sh

hrm-export: ## Re-export HRM-Text-1B -> ONNX from upstream (slow, offline)
	uv venv .hrm-export-venv
	uv pip install --python .hrm-export-venv 'transformers>=4.45' 'torch>=2.3' 'onnx>=1.16' 'onnxruntime>=1.18' optimum
	.hrm-export-venv/bin/python scripts/export_hrm_text.py

llm-download: hrm-download ## Alias kept for compatibility; defers to hrm-download

llm-build: ## Build LLM service
	cd services/llm && cargo build --release

llm-run: ## Run LLM service
	cd services/llm && ./target/release/llm-service
```

- [ ] **Step 4: Verify `make hrm-download` works against your dev machine (skip if no release yet)**

If a release tarball already exists:
```bash
make hrm-download
ls -lh services/llm/models/hrm-text-1b/
```

Expected: `model.onnx` (~2 GB), `tokenizer.json`, `config.json`.

If no release yet: run `make hrm-export` instead (takes 5-10 min depending on hardware).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_hrm_text.py scripts/download_hrm_text_artifacts.sh Makefile
git commit -m "build(llm): HRM-Text ONNX export pipeline + Makefile targets"
```

---

### Task 3: Add HRM-specific config alongside existing config

**Files:**
- Modify: `services/llm/src/config.rs`
- Modify: `services/llm/config.toml`

**Why:** keep `LlamaEngine`'s fields valid for now (it still compiles). Add a new `HrmConfig` section that the new engine reads.

- [ ] **Step 1: Write the failing test**

Add to `services/llm/src/config.rs` inside a `#[cfg(test)] mod tests`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hrm_section_with_defaults() {
        let toml_text = r#"
port = 8001
model_path = "models/llava-v1.6-mistral-7b.IQ1_S.gguf"

[hrm]
model_dir = "models/hrm-text-1b"
"#;
        let cfg: LlmConfig = toml::from_str(toml_text).unwrap();
        let hrm = cfg.hrm.expect("hrm section present");
        assert_eq!(hrm.model_dir, "models/hrm-text-1b");
        assert_eq!(hrm.ep_preference, "auto");
        assert!(hrm.use_quantized.is_none() || hrm.use_quantized == Some(false));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/llm && cargo test --lib config:: 2>&1 | tail -10
```

Expected: FAIL — no field `hrm`, `HrmConfig` undefined.

- [ ] **Step 3: Add the HrmConfig struct and field**

Add to `services/llm/src/config.rs` after the existing `LlmConfig` struct fields:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    // ... existing fields ...

    /// HRM-Text engine configuration. When present, the service runs the
    /// new HrmEngine; otherwise it falls back to the legacy LlamaEngine.
    /// Both engines share `port`; the HRM section provides the rest.
    #[serde(default)]
    pub hrm: Option<HrmConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HrmConfig {
    /// Directory containing model.onnx, tokenizer.json, config.json.
    pub model_dir: String,

    /// Execution provider preference. "auto" picks CoreML on macOS, CUDA on
    /// Linux when n_gpu_layers > 0, else CPU. Other values: "cpu", "coreml",
    /// "cuda".
    #[serde(default = "default_ep_preference")]
    pub ep_preference: String,

    /// Use the int8 quantized variant (model.int8.onnx) if true. Defaults to
    /// false (fp16 model.onnx).
    #[serde(default)]
    pub use_quantized: Option<bool>,

    /// Number of CPU threads for ort sessions. Falls back to LlmConfig.n_threads
    /// if None.
    #[serde(default)]
    pub n_threads: Option<i32>,
}

fn default_ep_preference() -> String { "auto".to_string() }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/llm && cargo test --lib config:: 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Update `services/llm/config.toml`**

Replace its contents with:

```toml
# Shared
port = 8001

# Legacy LlamaEngine (kept until handler swap, then removed)
model_path   = "models/llava-v1.6-mistral-7b.IQ1_S.gguf"
mmproj_path  = "models/llava-v1.6-mistral-7b-mmproj-f16.gguf"
ctx_size     = 4096
n_threads    = 4
n_gpu_layers = 0

# HRM-Text engine -- uncomment after running `make hrm-download` or `make hrm-export`
# [hrm]
# model_dir      = "models/hrm-text-1b"
# ep_preference  = "auto"
# use_quantized  = false
# n_threads      = 4
```

- [ ] **Step 6: Commit**

```bash
git add services/llm/src/config.rs services/llm/config.toml
git commit -m "feat(llm): add HrmConfig alongside legacy LlmConfig"
```

---

### Task 4: Tokenizer wrapper

**Files:**
- Create: `services/llm/src/tokenizer.rs`
- Modify: `services/llm/src/main.rs` (declare module)

- [ ] **Step 1: Write the failing test**

Create `services/llm/src/tokenizer.rs`:

```rust
use anyhow::{Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct HrmTokenizer {
    inner: Tokenizer,
}

impl HrmTokenizer {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("load tokenizer at {}: {}", path.display(), e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        let enc = self.inner.encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().iter().map(|&x| x as i64).collect())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }

    pub fn decode_single(&self, id: u32) -> Result<String> {
        self.decode(&[id])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/hrm-text-1b")
    }

    fn skip_if_no_model() -> Option<std::path::PathBuf> {
        let d = fixture_dir();
        if d.join("tokenizer.json").exists() { Some(d) } else { None }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let Some(dir) = skip_if_no_model() else {
            eprintln!("skipping: run `make hrm-download` to enable tokenizer tests");
            return;
        };
        let tok = HrmTokenizer::load(&dir).unwrap();
        let ids = tok.encode("hello world", true).unwrap();
        assert!(!ids.is_empty());
        let id_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        let text = tok.decode(&id_u32).unwrap();
        assert!(text.to_lowercase().contains("hello"));
    }
}
```

- [ ] **Step 2: Declare the module**

Edit `services/llm/src/main.rs:1-3`. Add `mod tokenizer;` after `mod handler;`:

```rust
mod config;
mod engine;
mod handler;
mod tokenizer;
```

- [ ] **Step 3: Run tests**

```bash
cd services/llm && cargo test --lib tokenizer:: 2>&1 | tail -20
```

Expected: PASS (or "skipping" if no model yet — still a pass).

- [ ] **Step 4: Commit**

```bash
git add services/llm/src/tokenizer.rs services/llm/src/main.rs
git commit -m "feat(llm): HrmTokenizer wrapper over HF tokenizers crate"
```

---

### Task 5: HrmEngine — struct, load, runtime config

**Files:**
- Create: `services/llm/src/hrm_engine.rs`
- Modify: `services/llm/src/main.rs` (declare module)

The detailed decode loop comes in Tasks 6-8. This task lands the struct and loading.

- [ ] **Step 1: Write the failing test**

Create `services/llm/src/hrm_engine.rs`:

```rust
use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::config::HrmConfig;
use crate::tokenizer::HrmTokenizer;

#[derive(Debug, Clone, Deserialize)]
pub struct HrmRuntimeConfig {
    pub eos_token_id: u32,
    pub ctx_size: u32,
    pub slow_loops: u32,
    pub fast_loops: u32,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
}

pub struct HrmEngine {
    pub session: Arc<Session>,
    pub tokenizer: HrmTokenizer,
    pub runtime: HrmRuntimeConfig,
    pub model_dir: PathBuf,
}

unsafe impl Send for HrmEngine {}
unsafe impl Sync for HrmEngine {}

impl HrmEngine {
    pub fn load(cfg: &HrmConfig) -> Result<Self> {
        let model_dir = PathBuf::from(&cfg.model_dir);
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

        tracing::info!(path = %onnx_path.display(), "Loading HRM-Text ONNX...");

        let session = Self::build_session(&onnx_path, cfg)
            .context("build ort session")?;

        let tokenizer = HrmTokenizer::load(&model_dir)
            .context("load HrmTokenizer")?;

        let runtime: HrmRuntimeConfig = {
            let text = std::fs::read_to_string(model_dir.join("config.json"))
                .context("read config.json")?;
            serde_json::from_str(&text).context("parse config.json")?
        };

        tracing::info!(
            ctx_size = runtime.ctx_size,
            slow_loops = runtime.slow_loops,
            fast_loops = runtime.fast_loops,
            "HRM-Text loaded"
        );

        Ok(Self {
            session: Arc::new(session),
            tokenizer,
            runtime,
            model_dir,
        })
    }

    fn build_session(onnx_path: &Path, cfg: &HrmConfig) -> Result<Session> {
        let threads = cfg.n_threads.unwrap_or(4).max(1);
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads as usize)?;
        // EP selection per cfg.ep_preference. "auto" -> platform default.
        // Concrete EP wiring is omitted here; ort 2.0.0-rc.10 picks CPU by default
        // and additional EPs (CoreML/CUDA) require feature flags + .with_execution_providers(...).
        // The implementation below is sufficient for v0; gating EPs is a follow-up.
        Ok(builder.commit_from_file(onnx_path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_cfg() -> HrmConfig {
        HrmConfig {
            model_dir: format!("{}/models/hrm-text-1b", env!("CARGO_MANIFEST_DIR")),
            ep_preference: "cpu".into(),
            use_quantized: Some(false),
            n_threads: Some(2),
        }
    }

    fn skip_if_no_model() -> bool {
        !std::path::Path::new(&format!(
            "{}/models/hrm-text-1b/model.onnx",
            env!("CARGO_MANIFEST_DIR")
        )).exists()
    }

    #[test]
    fn load_errors_when_onnx_missing() {
        let cfg = HrmConfig {
            model_dir: "/nonexistent/path".into(),
            ep_preference: "cpu".into(),
            use_quantized: Some(false),
            n_threads: Some(2),
        };
        let err = HrmEngine::load(&cfg).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn load_succeeds_with_artifacts() {
        if skip_if_no_model() {
            eprintln!("skipping: run `make hrm-download` to enable HrmEngine load tests");
            return;
        }
        let cfg = fixture_cfg();
        let eng = HrmEngine::load(&cfg).unwrap();
        assert!(eng.runtime.ctx_size > 0);
        assert!(eng.runtime.slow_loops >= 1);
    }
}
```

- [ ] **Step 2: Declare the module**

Edit `services/llm/src/main.rs`:

```rust
mod config;
mod engine;
mod handler;
mod hrm_engine;
mod tokenizer;
```

- [ ] **Step 3: Run tests**

```bash
cd services/llm && cargo test --lib hrm_engine:: 2>&1 | tail -20
```

Expected: `load_errors_when_onnx_missing` PASS; `load_succeeds_with_artifacts` PASS or "skipping".

- [ ] **Step 4: Commit**

```bash
git add services/llm/src/hrm_engine.rs services/llm/src/main.rs
git commit -m "feat(llm): HrmEngine skeleton — load ONNX, tokenizer, runtime config"
```

---

### Task 6: HrmEngine — prefill

**Files:**
- Modify: `services/llm/src/hrm_engine.rs`

- [ ] **Step 1: Write the failing test**

Append to `services/llm/src/hrm_engine.rs` (inside the existing test module):

```rust
    #[test]
    fn prefill_returns_logits_for_last_position() {
        if skip_if_no_model() {
            eprintln!("skipping: requires hrm-text-1b artifacts");
            return;
        }
        let eng = HrmEngine::load(&fixture_cfg()).unwrap();
        let ids = eng.tokenizer.encode("The capital of France is", true).unwrap();
        let logits = eng.prefill(&ids).unwrap();
        assert_eq!(logits.len() as u32, eng.runtime.vocab_size);
        // Logits should be non-uniform (some variation between positions)
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max - min > 0.1, "logits look uniform: max-min={}", max-min);
    }
```

- [ ] **Step 2: Implement `prefill`**

Add to the `impl HrmEngine` block:

```rust
    /// Run a prefill pass on `input_ids` and return the next-token logits
    /// (over the full vocab) corresponding to the last position.
    ///
    /// Returned shape: `Vec<f32>` of length `runtime.vocab_size`.
    pub fn prefill(&self, input_ids: &[i64]) -> Result<Vec<f32>> {
        use ndarray::Array2;

        if input_ids.is_empty() {
            anyhow::bail!("prefill requires at least one input token");
        }

        let arr = Array2::from_shape_vec((1, input_ids.len()), input_ids.to_vec())
            .context("shape input_ids")?;

        let outputs = self.session
            .run(ort::inputs![ "input_ids" => arr.view() ]
                .context("build inputs")?)
            .context("ort run prefill")?;

        let logits = outputs["logits"]
            .try_extract_tensor::<f32>()
            .context("extract logits")?;

        let shape = logits.shape();
        // Expected: [batch=1, seq, vocab]
        if shape.len() != 3 {
            anyhow::bail!("unexpected logits shape: {:?}", shape);
        }
        let last_pos = shape[1] - 1;
        let last_row = logits.slice(ndarray::s![0, last_pos, ..]);
        Ok(last_row.iter().copied().collect())
    }
```

- [ ] **Step 3: Run tests**

```bash
cd services/llm && cargo test --lib hrm_engine::tests::prefill 2>&1 | tail -10
```

Expected: PASS or "skipping".

- [ ] **Step 4: Commit**

```bash
git add services/llm/src/hrm_engine.rs
git commit -m "feat(llm): HrmEngine prefill — returns last-position logits"
```

---

### Task 7: HrmEngine — autoregressive decode (greedy)

**Files:**
- Modify: `services/llm/src/hrm_engine.rs`

This task lands a greedy (temperature=0) decode loop. Sampling comes in Task 8.

- [ ] **Step 1: Write the failing test**

Append to the test module:

```rust
    #[test]
    fn decode_greedy_produces_tokens_under_max() {
        if skip_if_no_model() {
            eprintln!("skipping");
            return;
        }
        let eng = HrmEngine::load(&fixture_cfg()).unwrap();
        let ids = eng.tokenizer.encode("Hello,", true).unwrap();
        let generated = eng.decode_greedy(&ids, 8).unwrap();
        assert!(!generated.is_empty(), "no tokens generated");
        assert!(generated.len() <= 8, "exceeded max_tokens");
        // None of the generated tokens should equal eos (decode stops on eos)
        let eos = eng.runtime.eos_token_id;
        assert!(!generated.iter().any(|&t| t == eos as i64));
    }
```

- [ ] **Step 2: Implement `decode_greedy`**

Add to `impl HrmEngine`:

```rust
    /// Greedy autoregressive decode. Returns the list of generated token IDs
    /// (not including the prompt, not including EOS).
    pub fn decode_greedy(&self, prompt_ids: &[i64], max_tokens: u32) -> Result<Vec<i64>> {
        let mut ids: Vec<i64> = prompt_ids.to_vec();
        let mut out: Vec<i64> = Vec::with_capacity(max_tokens as usize);

        for _ in 0..max_tokens {
            let logits = self.prefill(&ids)?;
            // argmax
            let (next_id, _) = logits.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                    if v > acc.1 { (i, v) } else { acc }
                });
            let next_id = next_id as i64;

            if next_id as u32 == self.runtime.eos_token_id {
                break;
            }
            if ids.len() as u32 >= self.runtime.ctx_size {
                tracing::warn!("decode hit ctx_size cap");
                break;
            }
            ids.push(next_id);
            out.push(next_id);
        }
        Ok(out)
    }
```

Note: this re-prefills the full sequence each step (O(N^2)). KV-cache optimization is intentionally out of scope for v0; performance work is a follow-up spec.

- [ ] **Step 3: Run test**

```bash
cd services/llm && cargo test --lib hrm_engine::tests::decode 2>&1 | tail -10
```

Expected: PASS or "skipping". May take ~10-30s if model is loaded.

- [ ] **Step 4: Commit**

```bash
git add services/llm/src/hrm_engine.rs
git commit -m "feat(llm): HrmEngine greedy decode — argmax loop, EOS + ctx_size stop"
```

---

### Task 8: HrmEngine — sampling + streaming inference

**Files:**
- Modify: `services/llm/src/hrm_engine.rs`
- Modify: `services/llm/Cargo.toml` (add `rand`)

Match the signature shape of the old `LlamaEngine::infer_text` so the handler can be swapped with minimal changes.

- [ ] **Step 1: Write the failing test**

Append to the test module:

```rust
    #[tokio::test]
    async fn infer_text_streams_tokens_via_channel() {
        if skip_if_no_model() {
            eprintln!("skipping");
            return;
        }
        let eng = std::sync::Arc::new(HrmEngine::load(&fixture_cfg()).unwrap());
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);

        let eng2 = eng.clone();
        let h = tokio::task::spawn_blocking(move || {
            eng2.infer_text("Hello,".to_string(), 8, 0.0, tx)
        });

        let mut received = Vec::new();
        while let Some(s) = rx.recv().await { received.push(s); }
        h.await.unwrap().unwrap();
        assert!(!received.is_empty(), "no streamed tokens");
    }
```

- [ ] **Step 2: Implement sampling + streaming**

Add to `impl HrmEngine`:

```rust
    /// Sample one token from `logits` using top-k, top-p, temperature.
    /// temperature <= 0 -> greedy argmax.
    fn sample(&self, logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> usize {
        if temperature <= 0.0 {
            return logits.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)|
                    if v > acc.1 { (i, v) } else { acc }).0;
        }
        let t = temperature.clamp(0.01, 2.0);

        // top-k
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i,&v)| (i, v/t)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k.max(1));

        // softmax
        let max = indexed[0].1;
        let mut probs: Vec<f32> = indexed.iter().map(|(_, l)| (l - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum; }

        // top-p (nucleus): keep smallest prefix with cumulative prob >= top_p
        let mut cum = 0.0_f32;
        let mut keep = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p { keep = i + 1; break; }
        }
        probs.truncate(keep);
        let renorm: f32 = probs.iter().sum();
        for p in &mut probs { *p /= renorm; }

        // weighted choice
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut acc = 0.0_f32;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r <= acc { return indexed[i].0; }
        }
        indexed.last().unwrap().0
    }

    /// Drop-in replacement for the old LlamaEngine::infer_text. Streams
    /// decoded token strings into `tx`. Blocking — wrap in spawn_blocking.
    pub fn infer_text(
        self: std::sync::Arc<Self>,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<()> {
        let mut ids = self.tokenizer.encode(&prompt, true)?;
        for _ in 0..max_tokens {
            let logits = self.prefill(&ids)?;
            let next = self.sample(&logits, temperature, 40, 0.95);
            let next_i64 = next as i64;

            if next as u32 == self.runtime.eos_token_id { break; }
            if ids.len() as u32 >= self.runtime.ctx_size { break; }

            let piece = self.tokenizer.decode_single(next as u32).unwrap_or_default();
            if tx.blocking_send(piece).is_err() { break; }
            ids.push(next_i64);
        }
        Ok(())
    }
```

- [ ] **Step 3: Add `rand` dep**

Edit `services/llm/Cargo.toml`, add to `[dependencies]`:

```toml
rand        = "0.8"
```

- [ ] **Step 4: Run tests**

```bash
cd services/llm && cargo test --lib hrm_engine:: 2>&1 | tail -20
```

Expected: all tests pass or skip.

- [ ] **Step 5: Commit**

```bash
git add services/llm/Cargo.toml services/llm/Cargo.lock services/llm/src/hrm_engine.rs
git commit -m "feat(llm): HrmEngine sampling (top-k/top-p/temp) + streaming infer_text"
```

---

### Task 9: Swap handler.rs to use HrmEngine

**Files:**
- Modify: `services/llm/src/handler.rs`

**Why:** the handler currently calls `LlamaEngine::infer_text` and `LlamaEngine::infer_multimodal`. The first becomes `HrmEngine::infer_text`. The second is removed (vision bridge replaces it in Task 11-12). Image inputs in this task return 400 — vision_bridge wires up the recovery path next.

- [ ] **Step 1: Replace AppState's engine field**

Edit `services/llm/src/handler.rs:13-17`:

```rust
// -- State ---------------------------------------------------------------------

pub struct AppState {
    pub engine: Arc<crate::hrm_engine::HrmEngine>,
}
```

- [ ] **Step 2: Replace the import and rewrite chat_completions**

Edit `services/llm/src/handler.rs:11`:

```rust
use crate::hrm_engine::HrmEngine;
```

Then replace the body of `chat_completions` (lines ~123-219). The new version drops the multimodal arm and calls `HrmEngine::infer_text`:

```rust
pub async fn chat_completions(
    state: web::Data<AppState>,
    req: web::Json<ChatRequest>,
) -> HttpResponse {
    let req = req.into_inner();
    let model_name = req.model.clone().unwrap_or_else(|| "hrm-text-1b".to_string());
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let streaming = req.stream;

    let (pairs, image_bytes) = match extract_content(&req.messages) {
        Ok(v) => v,
        Err(e) => return HttpResponse::BadRequest().json(json!({"error": e})),
    };

    // Vision bridge wiring lands in Task 12. Until then, reject images.
    if image_bytes.is_some() {
        return HttpResponse::BadRequest().json(json!({
            "error": "image inputs are temporarily disabled — vision bridge is being wired in"
        }));
    }

    let prompt = build_prompt(&pairs);
    let engine = Arc::clone(&state.engine);

    if streaming {
        let (tx, rx) = mpsc::channel::<String>(128);

        let engine2 = Arc::clone(&engine);
        let prompt2 = prompt.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = engine2.infer_text(prompt2, max_tokens, temperature, tx) {
                tracing::error!("inference error: {e:#}");
            }
        });

        let model_for_stream = model_name.clone();
        let token_stream = ReceiverStream::new(rx)
            .map(move |tok| Ok::<Bytes, std::io::Error>(sse_chunk(&tok, &model_for_stream)));
        let done_stream = futures_util::stream::once(async {
            Ok::<Bytes, std::io::Error>(sse_done())
        });
        HttpResponse::Ok()
            .content_type("text/event-stream; charset=utf-8")
            .insert_header(("Cache-Control", "no-cache"))
            .insert_header(("X-Accel-Buffering", "no"))
            .streaming(token_stream.chain(done_stream))
    } else {
        let (tx, mut rx) = mpsc::channel::<String>(512);
        let handle = tokio::task::spawn_blocking(move || {
            engine.infer_text(prompt, max_tokens, temperature, tx)
        });

        let mut content = String::new();
        while let Some(tok) = rx.recv().await {
            content.push_str(&tok);
        }
        if let Err(e) = handle.await.unwrap_or(Ok(())) {
            return HttpResponse::InternalServerError()
                .json(json!({"error": format!("inference failed: {e}")}));
        }

        HttpResponse::Ok().json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }))
    }
}

/// Build a ChatML-formatted prompt. Same shape as the legacy LlamaEngine::build_prompt
/// minus the multimodal marker.
fn build_prompt(messages: &[(String, String)]) -> String {
    let mut buf = String::new();
    for (role, content) in messages {
        buf.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    buf.push_str("<|im_start|>assistant\n");
    buf
}
```

- [ ] **Step 3: Update `list_models`**

Replace `list_models` (bottom of file):

```rust
/// `GET /v1/models`
pub async fn list_models(state: web::Data<AppState>) -> HttpResponse {
    let _ = state;
    HttpResponse::Ok().json(json!({
        "object": "list",
        "data": [{
            "id": "hrm-text-1b",
            "object": "model",
            "owned_by": "local",
            "multimodal": true  // vision_bridge handles images
        }]
    }))
}
```

- [ ] **Step 4: Build**

```bash
cd services/llm && cargo build --release 2>&1 | tail -20
```

Expected: build succeeds. `engine.rs` (LlamaEngine) still compiles — it's now unused.

- [ ] **Step 5: Commit**

```bash
git add services/llm/src/handler.rs
git commit -m "feat(llm): handler uses HrmEngine; images return 400 pending vision bridge"
```

---

### Task 10: Update main.rs to load HrmEngine

**Files:**
- Modify: `services/llm/src/main.rs`
- Modify: `services/llm/config.toml`

- [ ] **Step 1: Replace engine load in main**

Edit `services/llm/src/main.rs`. Replace the whole file:

```rust
mod config;
mod engine;        // legacy LlamaEngine — removed in Task 13
mod handler;
mod hrm_engine;
mod tokenizer;

use actix_web::{middleware, web, App, HttpServer};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

use config::LlmConfig;
use handler::AppState;
use hrm_engine::HrmEngine;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive("llm_service=info".parse().unwrap()),
        )
        .init();

    let config = LlmConfig::load().unwrap_or_else(|e| {
        eprintln!("Config error: {e}");
        std::process::exit(1);
    });

    let hrm_cfg = config.hrm.clone().unwrap_or_else(|| {
        eprintln!("config.toml is missing the [hrm] section. Uncomment it and run `make hrm-download`.");
        std::process::exit(1);
    });

    let engine = HrmEngine::load(&hrm_cfg).unwrap_or_else(|e| {
        eprintln!("HrmEngine load failed: {e}");
        std::process::exit(1);
    });

    let port = config.port;
    let state = web::Data::new(AppState {
        engine: Arc::new(engine),
    });

    tracing::info!("LLM microservice (HRM-Text) listening on 0.0.0.0:{}", port);

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(
                web::JsonConfig::default()
                    .limit(32 * 1024 * 1024)
                    .error_handler(|err, _req| {
                        let msg = err.to_string();
                        actix_web::error::InternalError::from_response(
                            err,
                            actix_web::HttpResponse::BadRequest()
                                .json(serde_json::json!({"error": {"message": msg}})),
                        )
                        .into()
                    }),
            )
            .wrap(middleware::Logger::default())
            .route(
                "/v1/chat/completions",
                web::post().to(handler::chat_completions),
            )
            .route("/v1/models", web::get().to(handler::list_models))
    })
    .workers(1)
    .bind(format!("0.0.0.0:{port}"))?
    .run()
    .await
}
```

- [ ] **Step 2: Uncomment the `[hrm]` section in config**

Edit `services/llm/config.toml`. Uncomment the `[hrm]` block:

```toml
[hrm]
model_dir      = "models/hrm-text-1b"
ep_preference  = "auto"
use_quantized  = false
n_threads      = 4
```

- [ ] **Step 3: Build + smoke-launch**

```bash
cd services/llm && cargo build --release
# launch only if artifacts present
[ -f models/hrm-text-1b/model.onnx ] && (./target/release/llm-service & sleep 5 && curl -s http://127.0.0.1:8001/v1/models | head; kill %1) || echo "skipping launch — run make hrm-download first"
```

Expected (with artifacts): `/v1/models` returns `{"object":"list","data":[{"id":"hrm-text-1b",...}]}`.

- [ ] **Step 4: Commit**

```bash
git add services/llm/src/main.rs services/llm/config.toml
git commit -m "feat(llm): boot HrmEngine in main; require [hrm] config section"
```

---

### Task 11: Vision bridge module (with mocks)

**Files:**
- Create: `services/llm/src/vision_bridge.rs`
- Modify: `services/llm/src/main.rs` (declare module)
- Modify: `services/llm/src/config.rs` (add `VisionBridgeConfig`)
- Modify: `services/llm/config.toml` (add `[vision_bridge]` block)

- [ ] **Step 1: Add `VisionBridgeConfig` to `config.rs`**

Append in `services/llm/src/config.rs`:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct VisionBridgeConfig {
    #[serde(default = "default_vb_enabled")]
    pub enabled: bool,
    #[serde(default = "default_vb_base")]
    pub main_server_base: String,
    #[serde(default = "default_vb_classify")]
    pub classify_endpoint: String,
    #[serde(default = "default_vb_detect")]
    pub detect_endpoint: String,
    #[serde(default = "default_vb_classify_timeout")]
    pub classify_timeout_ms: u64,
    #[serde(default = "default_vb_detect_timeout")]
    pub detect_timeout_ms: u64,
}

fn default_vb_enabled() -> bool { true }
fn default_vb_base() -> String { "http://127.0.0.1:8000".to_string() }
fn default_vb_classify() -> String { "/classify/batch".to_string() }
fn default_vb_detect() -> String { "/yolo/detect".to_string() }
fn default_vb_classify_timeout() -> u64 { 1500 }
fn default_vb_detect_timeout() -> u64 { 2500 }
```

And add to `LlmConfig`:

```rust
    #[serde(default)]
    pub vision_bridge: Option<VisionBridgeConfig>,
```

- [ ] **Step 2: Write the failing test + module**

Create `services/llm/src/vision_bridge.rs`:

```rust
use anyhow::Result;
use base64::Engine as _;
use serde::Deserialize;
use std::time::Duration;

use crate::config::VisionBridgeConfig;

pub struct VisionBridge {
    cfg: VisionBridgeConfig,
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct ClassifyResp { results: Vec<Vec<ClassifyPred>> }

#[derive(Debug, Deserialize)]
struct ClassifyPred { label: String, confidence: f32 }

#[derive(Debug, Deserialize)]
struct DetectResp { detections: Vec<DetectBox> }

#[derive(Debug, Deserialize)]
struct DetectBox {
    label: String,
    #[serde(default)] confidence: f32,
    #[serde(default)] x1: f32, #[serde(default)] y1: f32,
    #[serde(default)] x2: f32, #[serde(default)] y2: f32,
}

impl VisionBridge {
    pub fn new(cfg: VisionBridgeConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(cfg.classify_timeout_ms + cfg.detect_timeout_ms))
            .build()
            .expect("build reqwest client");
        Self { cfg, http }
    }

    /// Produce a textual description of `image_bytes` by calling the main
    /// server's /classify/batch and /yolo/detect endpoints. Always returns
    /// a description; on errors, returns a stub and logs a warning.
    pub async fn describe(&self, image_bytes: &[u8]) -> String {
        let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

        let classify = self.classify(&b64).await;
        let detect = self.detect(&b64).await;

        match (classify, detect) {
            (Ok(cls), Ok(det)) => Self::compose(&cls, &det),
            (Ok(cls), Err(e)) => {
                tracing::warn!(error=%e, "vision_bridge: detect failed");
                Self::compose(&cls, &[])
            }
            (Err(e), Ok(det)) => {
                tracing::warn!(error=%e, "vision_bridge: classify failed");
                Self::compose("(classifier unavailable)", &det)
            }
            (Err(e1), Err(e2)) => {
                tracing::warn!(classify=%e1, detect=%e2, "vision_bridge: both failed");
                "[Image attached but vision tools unavailable.]".to_string()
            }
        }
    }

    async fn classify(&self, b64: &str) -> Result<String> {
        let url = format!("{}{}", self.cfg.main_server_base, self.cfg.classify_endpoint);
        let body = serde_json::json!({ "images": [b64], "top_k": 1 });
        let resp: ClassifyResp = self.http.post(&url)
            .json(&body)
            .timeout(Duration::from_millis(self.cfg.classify_timeout_ms))
            .send().await?
            .error_for_status()?
            .json().await?;
        let pred = resp.results.first()
            .and_then(|preds| preds.first())
            .ok_or_else(|| anyhow::anyhow!("classify: empty"))?;
        Ok(format!("'{}' ({:.2})", pred.label, pred.confidence))
    }

    async fn detect(&self, b64: &str) -> Result<Vec<DetectBox>> {
        let url = format!("{}{}", self.cfg.main_server_base, self.cfg.detect_endpoint);
        let body = serde_json::json!({ "image": b64 });
        let resp: DetectResp = self.http.post(&url)
            .json(&body)
            .timeout(Duration::from_millis(self.cfg.detect_timeout_ms))
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.detections)
    }

    fn compose(class_summary: &str, dets: &[DetectBox]) -> String {
        let mut s = format!("[Image attached. Classifier (top-1): {class_summary}.");
        if dets.is_empty() {
            s.push_str(" No YOLO detections.]");
        } else {
            s.push_str(" YOLO detections: ");
            for (i, d) in dets.iter().enumerate() {
                if i > 0 { s.push_str("; "); }
                s.push_str(&format!(
                    "{} at ({:.0},{:.0},{:.0},{:.0}) score {:.2}",
                    d.label, d.x1, d.y1, d.x2, d.y2, d.confidence
                ));
            }
            s.push_str(".]");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(base: &str) -> VisionBridgeConfig {
        VisionBridgeConfig {
            enabled: true,
            main_server_base: base.into(),
            classify_endpoint: "/classify/batch".into(),
            detect_endpoint: "/yolo/detect".into(),
            classify_timeout_ms: 500,
            detect_timeout_ms: 500,
        }
    }

    #[test]
    fn compose_with_both_results() {
        let det = DetectBox { label: "person".into(), confidence: 0.9, x1: 1.0, y1: 2.0, x2: 3.0, y2: 4.0 };
        let s = VisionBridge::compose("'cat' (0.81)", &[det]);
        assert!(s.contains("classifier") || s.contains("Classifier"));
        assert!(s.contains("'cat'"));
        assert!(s.contains("person at (1,2,3,4)"));
    }

    #[test]
    fn compose_with_no_detections() {
        let s = VisionBridge::compose("'cat' (0.81)", &[]);
        assert!(s.contains("No YOLO detections"));
    }

    #[tokio::test]
    async fn describe_returns_stub_when_server_down() {
        // No server running on this port.
        let vb = VisionBridge::new(cfg("http://127.0.0.1:1"));
        let out = vb.describe(b"\x89PNG\r\n\x1a\n").await;
        assert!(out.contains("vision tools unavailable"));
    }
}
```

- [ ] **Step 3: Declare module and run tests**

Edit `services/llm/src/main.rs` adding:

```rust
mod vision_bridge;
```

```bash
cd services/llm && cargo test --lib vision_bridge:: 2>&1 | tail -20
```

Expected: all three tests pass.

- [ ] **Step 4: Update `services/llm/config.toml`**

Append:

```toml
[vision_bridge]
enabled              = true
main_server_base     = "http://127.0.0.1:8000"
classify_endpoint    = "/classify/batch"
detect_endpoint      = "/yolo/detect"
classify_timeout_ms  = 1500
detect_timeout_ms    = 2500
```

- [ ] **Step 5: Commit**

```bash
git add services/llm/Cargo.toml services/llm/src/vision_bridge.rs services/llm/src/main.rs services/llm/src/config.rs services/llm/config.toml
git commit -m "feat(llm): vision_bridge module — describe image via classify+detect"
```

---

### Task 12: Wire vision_bridge into handler

**Files:**
- Modify: `services/llm/src/handler.rs`
- Modify: `services/llm/src/main.rs` (load bridge into AppState)

- [ ] **Step 1: Update `AppState`**

Edit `services/llm/src/handler.rs:13-17`:

```rust
pub struct AppState {
    pub engine: Arc<crate::hrm_engine::HrmEngine>,
    pub vision: Option<Arc<crate::vision_bridge::VisionBridge>>,
}
```

- [ ] **Step 2: Update handler to inject image description**

In `chat_completions`, replace the "image_bytes is some" early-400 with the bridge call:

```rust
    let (mut pairs, image_bytes) = match extract_content(&req.messages) {
        Ok(v) => v,
        Err(e) => return HttpResponse::BadRequest().json(json!({"error": e})),
    };

    if let Some(img) = image_bytes {
        let prefix = match state.vision.as_ref() {
            Some(vb) => vb.describe(&img).await,
            None => "[Image attached but vision bridge disabled.]".to_string(),
        };
        // Prepend description to the last user message.
        if let Some((_role, content)) = pairs.iter_mut().rev().find(|(r, _)| r == "user") {
            *content = format!("{prefix}\n{content}");
        } else {
            pairs.push(("user".into(), prefix));
        }
    }

    let prompt = build_prompt(&pairs);
```

(Delete the previous `if image_bytes.is_some() { return BadRequest ... }` block.)

- [ ] **Step 3: Wire bridge in `main.rs`**

Edit `services/llm/src/main.rs`. After `let engine = HrmEngine::load(...)`:

```rust
    let vision = config.vision_bridge.clone().and_then(|vbcfg| {
        if vbcfg.enabled {
            Some(Arc::new(vision_bridge::VisionBridge::new(vbcfg)))
        } else { None }
    });

    let state = web::Data::new(AppState {
        engine: Arc::new(engine),
        vision,
    });
```

- [ ] **Step 4: Build**

```bash
cd services/llm && cargo build --release 2>&1 | tail -10
```

Expected: success.

- [ ] **Step 5: Commit**

```bash
git add services/llm/src/handler.rs services/llm/src/main.rs
git commit -m "feat(llm): wire vision_bridge — images injected as text description"
```

---

### Task 13: Remove llama-cpp-2 and the legacy engine

**Files:**
- Modify: `services/llm/Cargo.toml`
- Delete: `services/llm/src/engine.rs`
- Modify: `services/llm/src/main.rs`
- Modify: `services/llm/src/config.rs`
- Create: `services/llm/tests/no_llama_cpp.rs`

- [ ] **Step 1: Delete the legacy engine module**

```bash
git rm services/llm/src/engine.rs
```

- [ ] **Step 2: Remove the module declaration**

Edit `services/llm/src/main.rs` — remove `mod engine;`.

- [ ] **Step 3: Drop legacy fields from LlmConfig**

Edit `services/llm/src/config.rs`. Remove `model_path`, `mmproj_path`, `ctx_size`, `n_threads`, `n_gpu_layers`, `effective_mmproj`. Keep only `port`, `hrm`, `vision_bridge`. Also remove the `load()` defaults that referenced GGUF paths — replace with a minimal fallback:

```rust
impl LlmConfig {
    pub fn load() -> Result<Self> {
        let config_path = std::path::PathBuf::from("config.toml");
        if config_path.exists() {
            let text = std::fs::read_to_string(&config_path).context("read config.toml")?;
            toml::from_str(&text).context("parse config.toml")
        } else {
            tracing::warn!("config.toml not found, using defaults");
            Ok(Self { port: 8001, hrm: None, vision_bridge: None })
        }
    }
}
```

- [ ] **Step 4: Drop legacy fields from `services/llm/config.toml`**

Replace contents with:

```toml
port = 8001

[hrm]
model_dir      = "models/hrm-text-1b"
ep_preference  = "auto"
use_quantized  = false
n_threads      = 4

[vision_bridge]
enabled              = true
main_server_base     = "http://127.0.0.1:8000"
classify_endpoint    = "/classify/batch"
detect_endpoint      = "/yolo/detect"
classify_timeout_ms  = 1500
detect_timeout_ms    = 2500
```

- [ ] **Step 5: Remove llama-cpp-2 from Cargo.toml**

Edit `services/llm/Cargo.toml`. Delete:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
llama-cpp-2 = { version = "0.1", features = ["mtmd", "metal"] }

[target.'cfg(not(target_os = "macos"))'.dependencies]
llama-cpp-2 = { version = "0.1", features = ["mtmd"] }
```

Also remove `image = "0.24"` from the main `[dependencies]` block.

- [ ] **Step 6: Add removed-feature guard test**

Create `services/llm/tests/no_llama_cpp.rs`:

```rust
//! Regression guard: ensures llama-cpp-2 stays out of the dependency tree.

use std::fs;
use std::path::PathBuf;

#[test]
fn cargo_lock_has_no_llama_cpp() {
    let lock = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let text = fs::read_to_string(&lock).expect("read Cargo.lock");
    assert!(
        !text.contains("\"llama-cpp-2\""),
        "llama-cpp-2 reappeared in Cargo.lock — this dep was removed by the HRM-Text swap."
    );
}
```

- [ ] **Step 7: Build + test**

```bash
cd services/llm && cargo build --release 2>&1 | tail -20
cd services/llm && cargo test --test no_llama_cpp 2>&1 | tail -10
cd services/llm && cargo test --lib 2>&1 | tail -20
```

Expected: all three commands succeed. Build is noticeably faster without llama-cpp-2.

- [ ] **Step 8: Commit**

```bash
git add services/llm/Cargo.toml services/llm/Cargo.lock services/llm/src/main.rs services/llm/src/config.rs services/llm/config.toml services/llm/tests/no_llama_cpp.rs
git commit -m "feat(llm): remove llama-cpp-2 + legacy LlamaEngine; add regression guard"
```

---

### Task 14: Smoke test + update docs

**Files:**
- Modify: `CLAUDE.md` (project root) — refresh LLM section
- Modify: `scripts/download_llm_model.sh` — point at HRM-Text

- [ ] **Step 1: Replace `scripts/download_llm_model.sh`**

Replace contents with a thin redirector:

```bash
#!/usr/bin/env bash
# Compatibility shim. The LLM service now uses HRM-Text. See the new script.
echo "scripts/download_llm_model.sh is deprecated."
echo "Run: bash scripts/download_hrm_text_artifacts.sh   (or: make hrm-download)"
exec bash "$(dirname "$0")/download_hrm_text_artifacts.sh" "$@"
```

- [ ] **Step 2: Update `CLAUDE.md` LLM line**

Edit `CLAUDE.md`. In the Scope table, change the LLM row to:

```markdown
| **LLM / Assistant** | ✅ Active | HRM-Text-1B via ONNX/`ort`; OpenAI-compatible `/v1/chat/completions` with streaming SSE. Image inputs are bridged through `/classify/batch` + `/yolo/detect` (caption-then-text). |
```

- [ ] **Step 3: Manual smoke test**

```bash
# 1. Build the main server + LLM service
cargo build --release
cd services/llm && cargo build --release && cd ../..

# 2. Make sure HRM-Text artifacts exist
make hrm-download  # or make hrm-export

# 3. Run the main server (which spawns the LLM service)
./target/release/torch-inference-server &
SERVER_PID=$!
sleep 8

# 4. Sanity: list models
curl -s http://127.0.0.1:8000/llm/v1/models | head

# 5. Chat (non-streaming)
curl -s http://127.0.0.1:8000/llm/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hrm-text-1b","stream":false,"max_tokens":24,"messages":[{"role":"user","content":"Say hi."}]}'

# 6. Tear down
kill $SERVER_PID
```

Expected: model list returns `hrm-text-1b`; chat returns a non-empty `content` field.

- [ ] **Step 4: Manual playground click-through**

Open `http://127.0.0.1:8000/playground` in a browser. Send "hello" — confirm streaming tokens arrive. Attach an image — confirm the assistant references it (the response should mention objects/labels from classify/detect).

- [ ] **Step 5: Commit docs**

```bash
git add CLAUDE.md scripts/download_llm_model.sh
git commit -m "docs: switch LLM service to HRM-Text + caption-bridge for images"
```

---

## Self-review checklist

Run through this before declaring the plan done:

- **Spec coverage:** Every section of the spec is implemented by a task. §2 architecture → Tasks 1, 5, 9, 10, 13. §3 export → Task 2. §4 runtime engine → Tasks 5-8. §5 vision bridge → Tasks 11-12. §6 API surface → Task 9 (handler + list_models). §7 spike → Task 0. §8 tests → embedded in each task. §9 risks → handled or accepted.
- **Placeholder scan:** No "TBD" / "TODO" / "implement later" in tasks. Open question on default quantization → addressed by the `--quantize` flag (default fp16).
- **Type consistency:** `HrmEngine::infer_text` signature matches the existing handler's `LlamaEngine::infer_text` (prompt, max_tokens, temperature, tx) so the handler swap is mechanical. `AppState::engine` switches from `Arc<LlamaEngine>` to `Arc<HrmEngine>` in Task 9 and is final by Task 12.
- **Build is green at every commit:** Tasks 1, 3, 4, 5, 6, 7, 8 are additive (build still passes). Task 9 swaps the handler (build passes). Task 13 removes the legacy code last (build passes).

---

## Out of scope (followup specs)

- Spec #2 — HRM-mirrored planner-executor agentic layer (next session).
- KV-cache optimization in the decode loop (currently O(N^2) re-prefill per token; acceptable for v0).
- Per-request batching / concurrency improvements.
- CoreML / CUDA execution provider wiring (Task 5 lands a CPU baseline; EP gating is a follow-up perf spec).
