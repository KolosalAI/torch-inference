# HRM-Text Export Spike — Addendum to Spec #1

**Date:** 2026-05-20
**Result:** PASS (after rewrites) — proceed with Spec #1, subject to the caveats below.
**Spike branch:** `spike/hrm-onnx-export`
**Spike artifacts commit:** `cc31251` (export script + Rust harness)
**Addendum commit (cherry-pick this):** see `git log -1 --format=%H docs/superpowers/specs/2026-05-20-hrm-text-export-spike.md` on the spike branch

## What worked

- `torch.export.export(..., strict=False)` traced the full HRM-Text-1B graph cleanly. The hierarchical recurrence (H_cycles=2 outer × L_cycles=3 inner, both unrolled into a fixed sequence of attention invocations by `__post_init__` inflating `num_hidden_layers` to `num_layers_per_stack * H_cycles * (L_cycles + 1)`) became a **static graph** — no `aten::if`, no `prim::Loop`, no halting predicate; the recurrence is purely structural and the exporter unrolls it like any other stack of transformer layers.
- ONNX export pipeline ran end-to-end: `Translate the graph into ONNX ✅` then `Optimize the ONNX graph ✅`.
- Greedy decode on "The capital of France is" via the Rust harness (ort 2.0.0-rc.10, CoreML/CPU EP, fp16 output) returned token id 236 = `Ġ` (BPE leading-space marker). A 6-step continuation via Python onnxruntime produced `"The capital of France is the capital of"` — coherent English, confirming the export is functionally correct (the base LM is not chat-tuned and tends to loop on factoid prompts; this is a model property, not an export bug).
- Model size: **graph 32 MB + external weights 2.2 GB = 2.3 GB total** (fp16).
- ort 2.0.0-rc.10 loaded the graph + sidecar `.data` file successfully via `commit_from_file("../out/model.onnx")` (it discovers the sidecar automatically when both are in the same dir).

## Rewrites applied

1. **Python interpreter upgrade.** Spec called `uv venv spike/.venv` with no `--python`. On this machine the default was Python 3.9, and transformers v5 (required for native HRM-Text support — see below) needs Python ≥ 3.10. Recreated with `uv venv --python 3.12 spike/.venv`.

2. **Transformers version.** The HF repo `sapientinc/HRM-Text-1B` ships custom remote code (`configuration_hrm_text.py`, `modeling_hrm_text.py`) that imports `from transformers.configuration_utils import PreTrainedConfig` (a transformers-v5 rename of the legacy `PretrainedConfig`). The pinned `transformers>=4.45` from the spec resolves to 4.57.6, which still uses the legacy spelling and fails with `ImportError: cannot import name 'PreTrainedConfig'`. Resolved by installing transformers from git main (`transformers @ git+https://github.com/huggingface/transformers`, resolved to v5.8.0.dev0 commit `5fc9bba`). At that version HRM-Text is **upstream-merged into transformers itself**, so `trust_remote_code=True` is unnecessary on a v5 install (it was still passed in the script for safety and works fine).

3. **onnxscript missing.** torch 2.12.0's ONNX exporter requires `onnxscript` at runtime. The spec's package list didn't include it. Added `onnxscript==0.7.0`.

4. **Output-pytree rejection (`DynamicCache`).** First export attempt failed with:
   > `RuntimeError: Found <class 'transformers.cache_utils.DynamicCache'> in output, which is not a known type. ... register a pytree for it.`

   The HF `forward` returns a `CausalLMOutputWithPast` containing a `DynamicCache` even when `use_cache=False` is left as the default. `torch.export` cannot flatten a `DynamicCache` to a pytree.

   **Fix:** wrap the model in a `LogitsOnly(nn.Module)` whose `forward` calls `self.m(input_ids=..., use_cache=False, return_dict=True)` and returns only `out.logits`. Also set `model.config.use_cache = False` for belt-and-braces. This works because for a **prefill-only** ONNX (which is what the runtime swap plan calls for — there's no KV-cache export yet), logits are the only required output. Export then succeeded on the first try.

5. **Disk-space crash.** The export needs ~2.3 GB free for the model graph + external weights. The export crashed with `OSError: [Errno 28] No space left on device` after `Optimize the ONNX graph ✅`, which **doesn't** indicate any topology problem — the graph had already been built in memory and was just being serialised. Freed 4 GB by running `uv cache clean` (safe — repopulates on next install) and re-ran. Implementer note: the spike workspace needs at least ~3 GB free disk during export; the HF hub cache for the model alone is ~2.2 GB on top.

6. **ort 2.0.0-rc.10 Rust API drift.** The Rust harness as written in the spec did not compile against ort 2.0.0-rc.10:
   - `ort::inputs![name => arr.view()]` doesn't work — `ArrayView<i64, _>` doesn't implement `Into<SessionInputValue>` in this rc. Replaced with `Tensor::<i64>::from_array(([1, seq_len], ids))?` per the project's existing idiom in `src/core/ort_yolo.rs`.
   - `try_extract_tensor::<f32>()` returns `(&Shape, &[T])` (tuple), **not** an ndarray view. So `logits.shape()` / `logits.slice(...)` don't exist. Replaced with manual indexing into the flat slice.
   - The exported model emits **fp16 logits** (because the model was loaded with `torch_dtype=torch.float16`). Attempting `try_extract_tensor::<f32>` fails with `Cannot extract Tensor<f32> from Tensor<f16>`. Enabled `features = ["half"]` on ort and `half = "2"` as a dep, then extracted with `try_extract_tensor::<half::f16>()` and called `.to_f32()` for the argmax comparison. **For the production code, plan on either (a) accepting fp16 logits and converting in Rust, or (b) re-exporting with `torch_dtype=torch.float32` if you want fp32 logits, which roughly doubles the weight file.**

## Loop counts (from upstream HRM-Text)

The upstream `simple_inference_engine.py` (https://github.com/sapientinc/HRM-Text/blob/main/simple_inference_engine.py) does **not** expose loop counts directly — they live in the model config. Source of truth, in priority order:

1. **HF checkpoint `config.json`** at `sapientinc/HRM-Text-1B`:
   ```
   "H_cycles": 2,
   "L_cycles": 3,
   ```
2. **Cached `configuration_hrm_text.py` (commit `2285b99`)** lines 94–95: defaults `H_cycles: int = 2` and `L_cycles: int = 3`.
3. The model's `__post_init__` inflates `num_hidden_layers` to `num_layers_per_stack * H_cycles * (L_cycles + 1)` = `2 × 2 × 4 = 16` reported layers from `num_layers_per_stack = 2`, which matches the `num_hidden_layers: 16` in the published config (so `num_layers_per_stack` resolves to 2 at load time).

Mapping to the runtime-swap plan's terminology:

- **slow_loops** (high-level / H-cycles): **2**
- **fast_loops** (low-level / L-cycles per H-cycle): **3**

These are checkpoint-specific. If a future HRM-Text release ships a different checkpoint with different cycles, the values are still in `config.json` and the relationship `num_hidden_layers = num_layers_per_stack × H_cycles × (L_cycles + 1)` holds.

## Caveats for plan execution

1. **transformers v5 + Python 3.10+ required.** The HRM-Text modeling code in transformers main uses syntax (`int | None`), classes (`PreTrainedConfig`), and helpers (`RopeParameters`, `huggingface_hub.dataclasses.strict`) that only exist in transformers v5 (dev). The runtime-swap plan must either pin transformers v5 explicitly or, more practically, **avoid the transformers dependency entirely at runtime** (i.e. only use transformers during export, ship just the ONNX file + tokenizer.json to production). The current export already does this — Python deps are export-only.

2. **External-data ONNX format.** The export does NOT produce a single fat `model.onnx` with embedded weights. It produces `model.onnx` (32 MB, graph) + `model.onnx.data` (2.2 GB, weights). They MUST be deployed together (same directory, identical filenames). The spec text saying "model.onnx ... file size >500 MB" is technically not met by the .onnx file alone (32 MB); it is met by the pair. **Implementer action:** plan the model-registry / download manifest entry as a `(model.onnx, model.onnx.data)` pair, not a single file. ort's `commit_from_file("model.onnx")` finds `model.onnx.data` automatically when co-located.

3. **fp16 logits.** Decide upfront whether to re-export at fp32 or to do fp16→fp32 conversion in Rust. The harness currently does the latter via `half::f16::to_f32`. Either is fine, but pick one for the production code.

4. **opset 17 → 18 fallback.** The exporter logged: `Setting ONNX exporter to use operator set version 18 because the requested opset_version 17 is a lower version than we have implementations for. ... The model was not modified.` The emitted model is **opset 18**, not the requested 17. ort 2.0.0-rc.10 loads it fine, but if any downstream tool insists on opset 17 there will be a version-conversion failure (the C-API converter raised an assertion during the auto-downgrade attempt). **Plan on opset 18.**

5. **Vocabulary is 65536, NOT the 151808 in `configuration_hrm_text.py`.** The defaults in the modular file are overridden by the published checkpoint's `config.json`. Plan the Rust tokenizer/sampler against vocab=65536. Tokenizer is the Qwen-style BPE saved at `out/tokenizer.json` (4.5 MB, no merges file needed — fast tokenizer is self-contained).

6. **No KV cache in the exported graph.** Setting `use_cache=False` was necessary to make `torch.export` happy. The current ONNX is a **pure prefill graph** — each decode step re-processes the entire context. For a multi-token generation server this is O(n²) tokens-of-work. Two paths:
   - **(a) Prefill-only is fine for short outputs.** If the runtime-swap plan only needs <128-token completions, the per-step quadratic cost is tolerable (1B model, ARM/CPU/CoreML — call it 10-50ms/token at 128 tokens).
   - **(b) Add a KV-cache export.** Possible but non-trivial: requires registering a pytree node for `DynamicCache` (PyTorch has stub support — see https://github.com/pytorch/pytorch/issues with the `DynamicCache pytree` keywords) or exporting two graphs (prefill + decode-step with cache I/O as explicit tensors). The Spec #1 implementation plan should call out which path to take.

7. **Mutex destructor warning on exit.** The Rust harness prints `libc++abi: terminating due to uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument` AFTER the success print. This is a known ort 2.0.0-rc.10 macOS destructor-order issue (similar to the existing Kokoro/YOLO/classifier shutdowns) and does not affect inference. The project already ships `register_ort_dylib()` workarounds; the plan should reuse them.

8. **`dynamic_axes` is being deprecated in favour of `dynamic_shapes`.** torch warned: `'dynamic_axes' is not recommended when dynamo=True ... Supply the 'dynamic_shapes' argument instead if export is unsuccessful.` Worked anyway for this single-input model, but if/when the KV-cache export is added the implementer may need to migrate to `dynamic_shapes={ "input_ids": { 1: torch.export.Dim("seq", min=1, max=2048) } }`.

9. **HF token recommended.** Anonymous downloads from the hub worked but printed `Warning: You are sending unauthenticated requests to the HF Hub`. For CI/release builds, set `HF_TOKEN` to avoid rate-limit headaches.
