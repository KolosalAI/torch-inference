# HRM-Text Runtime Swap — Design Spec

**Date:** 2026-05-20
**Author:** Evintkoo + Claude (brainstorming session)
**Status:** Awaiting user approval
**Scope:** Spec #1 of 2. Spec #2 (HRM-mirrored planner-executor agentic layer) is deferred to a later cycle.

---

## 1. Goal & non-goals

### Goal

Replace the `services/llm/` inference engine so that `POST /v1/chat/completions` is served by `sapientinc/HRM-Text-1B` running natively in Rust on top of the project's existing `ort` runtime. Preserve the OpenAI-compatible API surface so the reverse proxy, the playground UI, and any future client (the Spec #2 agentic layer) require no changes.

### Non-goals

- The agentic / planner-executor layer. That is Spec #2.
- Training, fine-tuning, or further model R&D on HRM-Text.
- Adding new endpoints beyond the existing `POST /v1/chat/completions` and `GET /v1/models`.
- Maintaining backward compatibility with the previous LLaVA-1.6 multimodal model. The caption-then-text bridge (§5) is the new image-handling story; the previous mtmd-based image path is removed.

### Constraints

- **Pure Rust at runtime.** No Python process invoked at request time. Python is permitted only as an offline build tool for the one-time ONNX export.
- The runtime ML backend must be `ort` (the project's standing convention; `candle` is explicitly disallowed by `CLAUDE.md` unless this spec's exportability spike fails — see §7).
- The OpenAI-compatible request/response JSON shape and the SSE streaming format are frozen — they must not change.

---

## 2. Architecture changes

### Before

```
src/api/llm_proxy.rs ── HTTP ──▶ services/llm/  (llama-cpp-2, mtmd)
                                   ├─ engine.rs      LlamaEngine (text + multimodal)
                                   ├─ handler.rs     chat_completions (OpenAI shape)
                                   └─ config.rs      gguf model_path + mmproj_path

models/llava-v1.6-mistral-7b.IQ1_S.gguf
models/llava-v1.6-mistral-7b-mmproj-f16.gguf
```

### After

```
src/api/llm_proxy.rs ── HTTP ──▶ services/llm/  (ort + tokenizers)
                                   ├─ engine.rs           HrmEngine (KV-cache decode loop)
                                   ├─ handler.rs          chat_completions (same OpenAI shape)
                                   ├─ tokenizer.rs        thin wrapper over HF tokenizers crate
                                   ├─ vision_bridge.rs    caption-then-text shim
                                   └─ config.rs           onnx model_dir, EP selection, loop counts

models/hrm-text-1b/
  ├─ model.onnx              (fp16 by default, ~2 GB)
  ├─ tokenizer.json
  └─ config.json             (eos_id, ctx_size, slow_loops, fast_loops, hidden_dims)
```

**Removed dependencies:**
- `llama-cpp-2` (and its Metal/CUDA feature flags and `mtmd` submodule). This is the bulk of the build-time and binary-size savings.
- `image` (only needed by the old multimodal pixel-decode path; the new bridge forwards bytes without decoding).

**Retained dependencies:**
- `base64` — still needed. The handler decodes incoming `data:image/...;base64,...` URIs from the playground; the vision bridge re-encodes raw bytes when posting to `/classify/batch` and `/yolo/detect` (those endpoints take base64-in-JSON).

**Added dependencies:**
- `ort = "=2.0.0-rc.10"` — already used elsewhere in the workspace; reused here for consistency.
- `tokenizers = "0.20"` — HF's Rust tokenizers crate.
- `ndarray = "0.16"` — KV-cache tensor slicing.
- `reqwest = { version = "0.12", features = ["json"] }` (async) — used by the vision bridge to call back into the main server's `/classify/batch` and `/yolo/detect` endpoints.

**Unchanged:**
- `src/api/llm_proxy.rs` (the reverse proxy at `/llm/{tail:.*}` → `127.0.0.1:8001`).
- `src/api/playground.html` (the chat UI; it talks to `/llm/v1/chat/completions` and renders whatever model string comes back).
- Request/response JSON shapes for `chat_completions` and `list_models`.
- SSE stream framing (`data: {…}\n\n`, terminated by `data: [DONE]\n\n`).

---

## 3. Offline export pipeline (one-time per model version)

A new script `scripts/export_hrm_text.py` runs on a developer or CI machine — **never invoked at request time**.

### Steps

1. Create an isolated `uv` Python env. Install `transformers`, `torch`, `onnx`, `onnxruntime`, `optimum`.
2. Load `sapientinc/HRM-Text-1B` via `transformers.AutoModelForCausalLM.from_pretrained(...)`.
3. Replace any FlashAttention 3 module with stock `torch.nn.functional.scaled_dot_product_attention`. (FA3 is not in the default ONNX opset and there is no ONNX EP for it.)
4. Wrap the model forward pass so it takes:
   - `input_ids`: `(batch, seq_len)` int64
   - `past_kv_cache`: tuple of `(key, value)` per layer, each `(batch, num_heads, past_len, head_dim)` fp16
   - `slow_state`: `(batch, hidden_slow)` fp16 — HRM's slow-loop hidden state
   - `fast_state`: `(batch, hidden_fast)` fp16 — HRM's fast-loop hidden state
   - `step_kind`: int32 scalar — `0` for prefill, `1` for slow-step, `2` for fast-step, `3` for head/logits

   and emits:
   - `logits` (only valid when `step_kind == 3`): `(batch, vocab)` fp16
   - `new_kv_cache`, `new_slow_state`, `new_fast_state`

   HRM's hierarchical recurrence (slow loop + fast loop) becomes **per-step**. The recurrent outer loop is driven from Rust (§4); it is **not** unrolled into the graph. This avoids fixed-depth unrolling and lets us tune loop counts at runtime.

5. `torch.onnx.export(..., opset_version=17, do_constant_folding=True)` → `model.onnx`.
6. Copy `tokenizer.json` from the HF checkpoint.
7. Emit a trimmed `config.json` containing:
   - `eos_token_id`
   - `ctx_size` (the model's positional embedding range)
   - `slow_loops` (default: per upstream `simple_inference_engine.py`; expected `2`)
   - `fast_loops` (default: per upstream; expected `4`)
   - `hidden_slow`, `hidden_fast`, `num_layers`, `num_heads`, `head_dim`
8. Optional: `onnxruntime.quantization.quantize_dynamic` to produce an int8 variant (~½ size). Gated behind a `--quantize` flag on the export script. Default is fp16.
9. Place output under `services/llm/models/hrm-text-1b/`.

A companion shell script `scripts/download_hrm_text_artifacts.sh` downloads a pre-exported tarball from a GitHub Release on `KolosalAI/torch-inference` so individual developers don't have to run the export. The release tag and asset name are recorded in this spec's §10 release notes.

A `Makefile` target `make hrm-export` invokes the Python script in the right env.

### Why this is acceptable under "pure Rust"

The `export_hrm_text.py` script is a **build-time artifact producer**, exactly analogous to the existing `convert_kokoro.py`. Runtime Rust binaries never spawn it, never call into it, never link against it. The shipped artifact is an `.onnx` blob — the same form as every other model in `models/`.

---

## 4. Runtime engine (Rust)

`services/llm/src/engine.rs` is replaced. New struct: `HrmEngine`.

### Load

1. Read `config.toml` → `LlmConfig { model_dir, port, ctx_size, n_threads, n_gpu_layers, ep_preference }`.
2. Parse `model_dir/config.json` → HRM-specific runtime parameters (loop counts, hidden sizes, EOS id).
3. Build an `ort::Session`:
   - On macOS: prefer CoreML EP (`use_coreml_program=true`), fall back to CPU.
   - On Linux: prefer CUDA EP if `n_gpu_layers > 0`, fall back to CPU.
   - Threads: `n_threads`.
4. Load tokenizer: `tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))`.
5. Initialize empty KV-cache buffers sized for `ctx_size` (preallocated, slice on use).

### Prefill (per-request)

1. Tokenize the full prompt → `Vec<i64>`.
2. One ONNX call with `step_kind=0`, `input_ids = prompt_tokens`, empty KV-cache, zeroed `slow_state` / `fast_state`.
3. Capture returned `kv_cache`, `slow_state`, `fast_state`.

### Decode loop (per generated token)

1. **Slow loop** — run `slow_loops` times (default 2):
   - Each call: `step_kind=1`, `input_ids=[]`, current KV/slow/fast → updated `slow_state`.
2. **Fast loop** — run `fast_loops` times (default 4):
   - Each call: `step_kind=2`, `input_ids=[]`, current KV/slow/fast → updated `fast_state` and a new KV slice for this position.
3. **Head** — one call: `step_kind=3` → logits over vocab.
4. **Sample** — top-k=40, top-p=0.95, temperature from the request (clamped `[0.01, 2.0]`).
5. **Append** sampled token id to the KV-cache position; decode the token string via the tokenizer.
6. **Yield** the token string into the existing `mpsc::Sender<String>` channel.
7. **Stop** if token id == `eos_token_id`, or if generated count == `max_tokens`.

If the exportability spike (§7) shows that the three step kinds can be fused into one ONNX graph driven by a counter input, the Rust loop collapses to a single per-token call. This is decided at spike time and reflected in an addendum to this spec before plan execution.

### Concurrency model

Unchanged from today: `actix-web::workers(1)` for the LLM service, requests serialized via a single tokio worker, and the decode loop runs inside `tokio::task::spawn_blocking`. This is fine because inference is the bottleneck; per-request batching is out of scope.

---

## 5. Multimodal bridge (caption-then-text)

`services/llm/src/vision_bridge.rs` is a new module.

### Trigger

When `handler::extract_content` finds one or more `ImageUrl` parts in the request, the handler routes the first image through `vision_bridge::describe(...)` before tokenizing.

### Flow

```
POST /llm/v1/chat/completions  (messages contain image_url)
        │
        ▼
  handler::extract_content
        │
        ▼
  vision_bridge::describe(image_bytes)
        │       (re-encode image_bytes as base64 once)
        │   ┌──────────────────────────────────────────────────────────────────┐
        ├──▶│ POST http://127.0.0.1:8000/classify/batch                        │ → top-1 label, confidence
        │   │   JSON: { "images": ["<base64>"], "top_k": 1 }                   │
        │   └──────────────────────────────────────────────────────────────────┘
        │   ┌──────────────────────────────────────────────────────────────────┐
        └──▶│ POST http://127.0.0.1:8000/yolo/detect                           │ → list of {label, bbox, score}
            │   JSON: { "image": "<base64>" } (+ query params for version/size) │
            └──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
            Compose textual description, e.g.:
            "[Image attached. Classifier (top-1): 'golden retriever' (0.81).
              YOLO detections: 1 person at (12,40,120,260) score 0.92;
              1 dog at (140,80,300,240) score 0.88.]"
                            │
                            ▼
        Prepend to the user message text → feed to HrmEngine.infer_text
```

### Configuration

`config.toml` gains:

```toml
[vision_bridge]
enabled = true
main_server_base = "http://127.0.0.1:8000"
classify_endpoint = "/classify/batch"
detect_endpoint = "/yolo/detect"
classify_timeout_ms = 1500
detect_timeout_ms = 2500
```

### Failure modes

- Main server unreachable: the bridge returns a stub description (`"[Image attached but vision tools unavailable.]"`) and HRM-Text proceeds without crashing. A `tracing::warn!` is emitted.
- Classify or detect returns an error: same — partial description, warn, continue.
- Image decoding fails (e.g. bad base64): handler returns 400 to the client, as today.

### Replacement path

Spec #2's planner-executor will subsume the bridge — the orchestrator decides *when* to call vision tools rather than always running both. The bridge stays as a working fallback for non-orchestrated chat (and for clients that bypass the agentic layer entirely).

---

## 6. API surface (unchanged)

### `POST /v1/chat/completions`

Request and response schemas are exactly as today (see `services/llm/src/handler.rs:21-60`). The only observable change is the `model` field in the response:

- Before: `"minicpm-v"` (a hard-coded fallback in the handler).
- After: `"hrm-text-1b"` (still falls back if the client omits the field).

SSE stream framing is unchanged. Token chunks arrive as:

```
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"hrm-text-1b","choices":[{"index":0,"delta":{"content":"…"},"finish_reason":null}]}\n\n
…
data: [DONE]\n\n
```

### `GET /v1/models`

Returns:

```json
{
  "object": "list",
  "data": [{
    "id": "hrm-text-1b",
    "object": "model",
    "owned_by": "local",
    "multimodal": true
  }]
}
```

`multimodal: true` because the vision bridge accepts image inputs even though the underlying LLM is text-only. From the client's perspective, the service handles images.

---

## 7. Exportability spike (gating step)

Before plan execution, a 1-2 day spike. The spike is a separate, throwaway branch and produces no production code — only a go/no-go signal.

### Spike scope

1. Write the minimum viable `scripts/export_hrm_text.py` per §3, with the per-step wrapper.
2. Run it on `sapientinc/HRM-Text-1B`. Capture any export errors.
3. If export succeeds: write a 50-line Rust harness that loads the `.onnx` via `ort`, runs a greedy decode of 5 tokens on the prompt `"The capital of France is"`, and prints them. The decode should produce coherent English (the model isn't a French-specific oracle, but `"Paris"` or similar is expected).
4. If export fails: capture the failing op and decide whether a rewrite is feasible.

### Decision matrix

| Spike outcome | Action |
|---|---|
| Clean export, decode produces coherent tokens | Proceed with the design above as-is. |
| Clean export but FA3 / hierarchical attention needs partial rewrite in the export script | Add the rewrite to §3. No spec changes beyond an addendum. Proceed. |
| Export fails on dynamic shapes or unsupported control flow even after rewrites | Fall back to **Candle port**: re-implement HRM's forward pass in `candle-nn`, load HF safetensors directly. This **reverses the project's "no Candle in prod" stance** documented in `CLAUDE.md` and requires Evintkoo's explicit written sign-off at that point. The spec is reissued as v2 reflecting the Candle plan. |

### Spike output

A short `docs/superpowers/specs/2026-05-20-hrm-text-export-spike.md` addendum to this spec, capturing:
- What worked, what didn't.
- Final loop counts (slow / fast) read from upstream code.
- Exact ONNX opset and any rewrites applied.
- A pointer to the throwaway branch with the spike code.

Plan execution does not start until the addendum is written and acknowledged.

---

## 8. Testing

### Unit tests (`services/llm/src/*` `#[cfg(test)]`)

- `tokenizer.rs`: round-trip encode/decode for a fixture string; UTF-8 multi-byte tokens preserved.
- `engine.rs`:
  - KV-cache shape invariants after prefill and after one decode step.
  - Sampling is deterministic when `temperature == 0` (top-k=1 greedy).
  - `infer_text` yields EOS-then-stop on a prompt where the model produces EOS within `max_tokens`.
- `vision_bridge.rs`:
  - Composes the expected description string given mock classify + detect responses.
  - Falls back to stub description when classify and detect both error.

### Integration tests

- A single in-process test that boots `HrmEngine` against a tiny fixture model (we ship a 10MB synthetic ONNX with the same I/O shape for tests — separate from the real 2GB model), runs a 32-token decode, and asserts no panics + non-empty output.
- `services/llm/` builds with `--release` without `llama-cpp-2` in the dependency tree.

### End-to-end / manual smoke

- Open the playground in a browser, send `"hello"`, confirm streaming tokens arrive and the stream terminates cleanly with `[DONE]`.
- Attach an image to a chat message; confirm a `[Image…]` description is injected into the prompt and the response is coherent.
- Verify `/llm/v1/models` returns `"hrm-text-1b"`.

### Removed-feature guard

A `cargo deny` rule or a tiny test that fails if `llama-cpp-2` reappears in `Cargo.lock`. Prevents accidental re-introduction during a future refactor.

---

## 9. Risks & open questions

| Risk | Likelihood | Mitigation |
|---|---|---|
| HRM-Text fails to export to ONNX even with rewrites | Medium | Gating spike (§7). Candle port fallback. |
| HRM hierarchical loop counts are not constants in upstream code | Low | Read from `simple_inference_engine.py` during spike; encode in `config.json`. |
| `ort` 2.0.0-rc.10 KV-cache via dynamic-shape inputs is fiddly | Medium | Use `IoBinding` for zero-copy reuse. Worst-case fallback: recreate the session per step (correct but slow; not acceptable for prod, would force a path change). |
| HRM-Text-1B context window not documented in the README | Low | Read from `model.config.json` during export; surface as a clear 400 error if request exceeds. |
| Caption-bridge adds latency on image chat | Low | Adds ~150–300 ms per image. Acceptable — image chat is not throughput-critical. |
| Removing `llama-cpp-2` strands existing GGUF files in `models/` | Low | Update `scripts/download_llm_model.sh` to point at HRM-Text. Do not auto-delete old files. Document in release notes. |
| CoreML / CUDA EP availability varies across dev machines | Medium | EP preference is configurable in `config.toml`; CPU is always available as fallback. Document expected throughput per EP in the README. |
| Decode latency is too slow without batching | Medium | Out of scope for v0; revisit in a follow-up perf spec once correctness is established. The current per-request serialization matches today's behavior. |

### Open questions (must be resolved before plan execution)

1. **Artifact hosting:** GitHub Releases on `KolosalAI/torch-inference`. Release tag and asset filename TBD when the export script lands. Confirm.
2. **Default quantization for v0:** fp16 (recommended for quality) or int8 (smaller, faster, possibly quality regression)? Default in this spec is fp16; the `--quantize` flag on the export script is available if you want to flip later.
3. **Vision-bridge concurrency:** classify and detect can be called serially (~300 ms total) or in parallel (~150 ms total). Parallel adds two simultaneous requests to the main server — fine in dev, possibly a concern at high concurrency. Default in this spec is serial; flag if you want parallel.

---

## 10. Migration & release notes

When this lands:

- `models/llava-v1.6-mistral-7b.IQ1_S.gguf` and `models/llava-v1.6-mistral-7b-mmproj-f16.gguf` are no longer used. The script that downloads them (`scripts/download_llm_model.sh`) is replaced by `scripts/download_hrm_text_artifacts.sh`.
- Existing operators with the old files on disk are unaffected — files stay, are simply unused. A note in the changelog mentions they can be deleted.
- The Makefile gains `make hrm-export` and `make hrm-download`; `make llm-build` and `make llm-run` continue to work (they build / run `services/llm/` with the new engine).
- The playground UI does not change; the model name in the header simply reads `hrm-text-1b` after the swap.

---

## 11. Next step

After approval of this spec:
1. Run the exportability spike (§7).
2. Write the spike addendum.
3. Invoke the `writing-plans` skill to produce an implementation plan that decomposes the work into orderable, reviewable tasks with TDD gates.
4. Execute the plan in a follow-up session.

Spec #2 (HRM-mirrored planner-executor agentic layer) begins after Spec #1 is shipped and verified.
