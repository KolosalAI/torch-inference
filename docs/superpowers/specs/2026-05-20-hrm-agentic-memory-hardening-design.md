# HRM-Text Agentic System — Memory Hardening Design

**Date:** 2026-05-20
**Scope:** `services/llm/`
**Status:** Approved for implementation
**Related specs:**
- `2026-05-20-hrm-text-runtime-swap-design.md` (initial HRM runtime swap)
- `2026-05-20-hrm-agentic-orchestration-design.md` (initial agent layer)
- `2026-05-20-hrm-text-export-spike.md` (export spike — flagged the no-KV-cache constraint addressed here)

## Context

The LLM microservice already runs `sapientinc/HRM-Text-1B` via ONNX (`ort = 2.0.0-rc.10`) and exposes:
- `POST /v1/chat/completions` (OpenAI-compatible, streaming SSE supported)
- `POST /v1/agent/run` (planner/executor with tools: classify, detect, vision, tts, stt, http_fetch, http_json, reflect, final; SSE event stream; `{{stepN.field}}` ref resolution; plan-repair on parse failure; per-step + per-run deadlines)

Prior operation crashed an M4 (high-RAM machine) and showed all three failure modes: RSS grew over time, per-request peak spiked into multi-GB territory, and concurrent load OOM'd. The user explicitly asked for bounded memory without an agent redesign.

## Root causes (from code audit)

1. **`hrm_engine.rs` has no KV cache.** Every decode step calls `prefill(&ids)` on the full growing sequence. ORT's output `logits` tensor is `[1, seq_len, vocab]` fp16. With ctx_size=2048 and vocab=65536, that is **~256 MB transient per token**, allocated and freed each step. 200 tokens = ~50 GB of allocator churn — enough to wedge any machine via fragmentation and peak RSS alone.

2. **`hrm_engine.rs:23`** the engine is `Arc<Mutex<Session>>`. Concurrent ONNX calls serialize through one Mutex, but each holds the ~256 MB transient at the moment it runs. Chat, agent planner, and reflect-tool all path through the same Mutex with no global cap.

3. **`agent::executor::RunContext.results: HashMap<String, Value>`** only grows during a run. Holds full tool outputs including base64 TTS audio (multiple MB per step). Never trimmed even after the value has been emitted on the SSE wire.

4. **`vision_bridge::describe`** keeps three copies of the image alive at peak: inbound `Vec<u8>`, base64 `String` (1.33×), and reqwest's serialized JSON body. A 2 MB image becomes ~6 MB resident during the describe call.

5. **`JsonConfig::default().limit(32 MB)`** lets a single client send 32 MB of base64, decoded to ~24 MB binary, base64-re-encoded to 32 MB string. Multiply by `max_concurrent_runs=4`.

6. **`chat_completions` non-streaming branch** buffers all generated tokens into a `String` with no hard ceiling (client-supplied `max_tokens` is honored verbatim, capped only by `ctx_size`).

7. **No process-level memory admission.** Nothing checks the host's actual memory pressure before accepting a new run.

## Goals

In priority order:

1. Eliminate per-request OOM. Hard upper bound on transient memory per inference.
2. Eliminate concurrent-burst OOM via a single global semaphore (1 permit) in front of the ONNX engine. Chat, planner, and reflect all funnel through it.
3. Stop slow RSS growth. Bound every long-lived structure (agent `results` map, channels, JSON limits) and explicit `Drop`/clear discipline on shared buffers.
4. Bring per-token cost from O(n²) to O(n) by adding a real KV cache. Requires re-exporting the ONNX as two graphs (`prefill.onnx` + `decode_step.onnx`). This is the only structural change.

## Non-goals

- No agent architecture redesign. Planner/executor/tool registry layout stays as is.
- No new tools, endpoints, or models.
- No multi-tenant features, no GPU offloading work, no move off ORT.
- No ORT io-binding for zero-copy KV reuse (follow-up spec if benchmarks demand it).
- No shared-weights export (single `.data` for both graphs). Follow-up.
- No quantized (int8) KV cache.

## Architecture overview

```
                                       ┌─────────────────────────────┐
                                       │ scripts/export_hrm_text.py  │  ← rewrite: emits
                                       │  → prefill.onnx (+data)     │     two graphs
                                       │  → decode_step.onnx (+data) │
                                       └─────────────────────────────┘
                                                   │ (offline artifact)
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ services/llm/src/                                                            │
│                                                                              │
│  hrm_engine.rs           ← KvSession { prefill, decode_step }                │
│                            + KvBuffers (reused tensors, sized to ctx_size)   │
│                            + new `infer_text_kv()`; old path stays as fb     │
│                                                                              │
│  engine_lease.rs (NEW)   ← global `Arc<Semaphore>` (1 permit), wraps every   │
│                            ONNX call — chat, planner, reflect                │
│                                                                              │
│  memory_gate.rs  (NEW)   ← read process RSS (macOS mach + Linux /proc);      │
│                            admission check; returns 503 above high water     │
│                                                                              │
│  handler.rs              ← input bounds (image bytes, prompt chars, msgs);   │
│                            smaller channels; clamp max_tokens                │
│                                                                              │
│  vision_bridge.rs        ← takes Vec<u8> by value; base64 in place; drops    │
│                            original before HTTP call                         │
│                                                                              │
│  agent/executor.rs       ← results-store trim hook (Value fields > 8 KB      │
│                            replaced by stub after SSE emit); smaller SSE     │
│                            channel; explicit Drop on RunContext              │
│                                                                              │
│  agent/http.rs           ← memory_gate check at admission; lower default     │
│                            max_concurrent_runs                               │
│                                                                              │
│  config.rs / config.toml ← new keys: kv_cache, max_image_bytes,              │
│                            max_prompt_chars, max_messages, max_ctx_size,     │
│                            rss_high_water_mb, rss_low_water_mb,              │
│                            per_run_result_bytes                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

No module moved, no public API broken. The HTTP surface (`/v1/chat/completions`, `/v1/models`, `/v1/agent/run`) is unchanged. KV cache is opt-in via config; if `kv_cache.enabled=true` and the two new ONNX files exist, the new path runs — otherwise the engine falls back to the existing `model.onnx`.

## ONNX re-export (offline artifact change)

Goal: produce two graphs so the runtime can do `prefill once + decode_step N times`.

### Why two graphs, not one
`torch.export` chokes on `DynamicCache` in outputs (the reason the current export had to set `use_cache=False`, per the export-spike spec). The cheapest workaround is to pre-flatten the cache to a tuple of plain tensors inside an `nn.Module` wrapper. Two graphs avoids branching inside the graph between "first pass" and "cached pass" — each graph has a fixed signature, which `torch.export` handles cleanly.

### Prefill graph (`prefill.onnx`)

| I/O | Name | dtype | Shape |
|---|---|---|---|
| input  | `input_ids` | i64  | `[1, seq_len]` (dynamic seq_len, max=ctx_size) |
| output | `logits`    | fp16 | `[1, seq_len, vocab=65536]` |
| output | `past_key_values.{L}.key`   (L=0..15) | fp16 | `[1, num_heads, seq_len, head_dim]` |
| output | `past_key_values.{L}.value` (L=0..15) | fp16 | `[1, num_heads, seq_len, head_dim]` |

### Decode-step graph (`decode_step.onnx`)

| I/O | Name | dtype | Shape |
|---|---|---|---|
| input  | `input_ids` | i64 | `[1, 1]` |
| input  | `past_key_values.{L}.key`   (L=0..15) | fp16 | `[1, num_heads, past_len, head_dim]` (dynamic past_len) |
| input  | `past_key_values.{L}.value` (L=0..15) | fp16 | `[1, num_heads, past_len, head_dim]` |
| output | `logits` | fp16 | `[1, 1, vocab=65536]` |
| output | `present_key_values.{L}.key`   | fp16 | `[1, num_heads, past_len+1, head_dim]` |
| output | `present_key_values.{L}.value` | fp16 | `[1, num_heads, past_len+1, head_dim]` |

### Export-script implementation (sketch)

```python
class PrefillWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids):
        out = self.m(input_ids=input_ids, use_cache=True, return_dict=True)
        kvs = []
        for layer_kv in out.past_key_values:    # DynamicCache → list of (k, v)
            kvs.extend([layer_kv[0], layer_kv[1]])
        return (out.logits, *kvs)

class DecodeStepWrapper(nn.Module):
    def __init__(self, m, num_layers):
        super().__init__(); self.m = m; self.L = num_layers
    def forward(self, input_ids, *past_flat):
        past = [(past_flat[2*i], past_flat[2*i+1]) for i in range(self.L)]
        cache = DynamicCache.from_legacy_cache(past)
        out = self.m(input_ids=input_ids, past_key_values=cache,
                     use_cache=True, return_dict=True)
        new = []
        for layer_kv in out.past_key_values:
            new.extend([layer_kv[0], layer_kv[1]])
        return (out.logits, *new)
```

`dynamic_shapes` declares `past_len` and `seq_len` so the runtime can grow the cache. Force `torch_dtype=torch.float16` to match current behavior.

### Risks (export-side)

- **`DynamicCache.from_legacy_cache` API drift** across transformers v5 dev versions. Pin transformers commit explicitly in the export script — the spike used `5fc9bba`. Document the pin in `scripts/export_hrm_text.py`.
- **32 cache I/O tensors** on the decode graph. ORT handles it but the I/O list is verbose; name tensors deterministically to keep diffs reviewable.
- **External-data duplication.** Each graph emits its own `.onnx.data`. Four shipping files: `prefill.onnx`, `prefill.onnx.data`, `decode_step.onnx`, `decode_step.onnx.data`. Disk ≈ 2.3 GB per graph (weights duplicated), so ~4.6 GB on disk. Runtime RAM unaffected — ORT mmaps. Sharing weights via a single external-data blob is a follow-up.

### Export-side validation gate

A parity test inside the export script: run 64-token greedy decode via both the old monolithic `model.onnx` (existing prefill-loop path) and the new (`prefill.onnx` + N×`decode_step.onnx`); assert identical token IDs. If parity fails, do not ship the artifacts.

## Runtime KV cache (Rust side)

### New types in `hrm_engine.rs`

```rust
pub struct KvSession {
    prefill:     Arc<Mutex<Session>>,
    decode_step: Arc<Mutex<Session>>,
    num_layers:  usize,    // 16 for HRM-Text-1B
    num_heads:   usize,
    head_dim:    usize,
    vocab_size:  usize,
}

pub enum EngineBackend {
    KvCache(KvSession),     // preferred path: two-graph, O(n) per token
    Monolithic(Session),    // fallback: existing prefill-only path
}

pub struct HrmEngine {
    backend:   Arc<Mutex<EngineBackend>>,
    tokenizer: HrmTokenizer,
    runtime:   HrmRuntimeConfig,
    model_dir: PathBuf,
}
```

### Load logic

Check for `prefill.onnx` and `decode_step.onnx` in `model_dir`. If both present and `[kv_cache] enabled=true`, load `KvCache`. Otherwise warn (`tracing::warn!`) and fall back to `Monolithic`. Existing tests against the old single-graph path keep working unchanged.

### Inference loop (`infer_text_kv`, replaces `infer_text` when KV is active)

```rust
// 1. Prefill: full prompt → logits_last + 32 cache tensors (held in KvBuffers)
let logits_last = self.prefill_with_kv(&prompt_ids, &mut buffers)?;

// 2. Sample first token from logits_last
let mut next = sample(&logits_last, temperature, top_k, top_p);
send_token(next);

// 3. Decode loop: [1,1] input + cache → logits + grown cache (in-place)
for _ in 1..max_tokens {
    if next == eos || buffers.current_len >= ctx_size { break; }
    let logits = self.decode_step(next, &mut buffers)?;
    next = sample(&logits, temperature, top_k, top_p);
    send_token(next);
}
// 4. EngineLease drop -> buffers.current_len = 0 (capacity preserved for next user)
```

### `KvBuffers` — the reuse model

```rust
struct KvBuffers {
    keys:        Vec<Vec<u16>>,    // 16 layers; each [1*H*max_ctx*D] as raw fp16 bits
    values:      Vec<Vec<u16>>,    // 16 layers; same shape
    current_len: usize,            // grows during one inference; reset at request start
    capacity:    usize,            // = max_ctx_size; never reallocs
}
```

- Allocated **once at engine load**. Sits resident; trades ~256 MB of permanent RAM for elimination of per-token allocation churn.
- One buffer set total, guarded by the engine semaphore (1 permit → 1 user at a time → 1 buffer suffices). If `engine.max_concurrent` is ever raised, the buffer set must be cloned per permit.
- Each `decode_step`:
  1. ORT's session is called with cache tensors built from the buffer slices for `[..current_len]`.
  2. ORT writes new k/v columns; runtime appends them in-place to position `[current_len..current_len+1]`.
  3. `current_len += 1`. No reallocation.
- At start of each new request: `current_len = 0`. Buffer stays warm — capacity preserved.
- The wrapper that gives `&mut KvBuffers` to the inference loop is bound to the `OwnedSemaphorePermit` from `EngineLease`, so aliasing is impossible.

### ORT specifics

`ort::Tensor::from_array((shape, Vec<T>))` moves the `Vec` in and returns it via outputs. Initial implementation uses move-in/move-out. If benchmarks show allocator cost dominates, escalate to ORT's `IoBinding` for true zero-copy reuse. IoBinding wiring in rc.10 is fiddly and out of scope for the initial PR.

### Resident memory math at these settings

- Engine weights: ~2.3 GB mmapped from `prefill.onnx.data` + `decode_step.onnx.data` (kernel-shared with disk; counted once via mmap, weight duplication doesn't double RSS).
- `KvBuffers` preallocated: ~256 MB at `max_ctx_size=1024`.
- Per-request transient: ~2 MB logits row + ~2 MB sample workspace.
- **Steady-state RSS ceiling: ~2.6 GB** regardless of request volume.

### Drop discipline (runtime-side)

- `EngineLease` permit guard resets `current_len = 0` on drop. Even on panic mid-decode the next caller starts clean.
- Old monolithic backend stays available, but is now also gated by the global engine semaphore.

## Bounds, watermark, and drop discipline (surgical)

### New `[limits]`, `[memory_gate]`, `[kv_cache]` config sections

```toml
[limits]
max_image_bytes        = 2_097_152    # 2 MiB, post-base64-decode
max_prompt_chars       = 16_384       # combined-messages cap pre-tokenize
max_messages           = 32           # cap on messages array length
max_generated_tokens   = 512          # clamps client-provided max_tokens
max_ctx_size           = 1024         # tighter default than current 2048

[limits.json]
body_limit             = 4_194_304    # 4 MiB; was 32 MiB

[limits.channels]
sse_event_buffer       = 8            # was 64
chat_stream_buffer     = 16           # was 128
chat_nonstream_buffer  = 64           # was 512

[limits.engine]
max_concurrent         = 1            # engine_lease semaphore permits

[limits.results]
per_run_bytes          = 65_536       # 64 KiB retained across ctx.results
field_trim_above       = 8_192        # any string field > 8 KiB stubbed post-emit

[memory_gate]
high_water_mb          = 4_096        # refuse new runs above this RSS
low_water_mb           = 3_072        # resume admitting below this
poll_on_admit_only     = true         # no background thread

[kv_cache]
enabled                = true         # falls back to monolithic if files missing
```

All defaults chosen to be safe on a 16 GB laptop without strangling throughput. Operators can raise them.

### Where each bound is enforced

| Bound | Enforced at | Failure mode |
|---|---|---|
| `body_limit` | `JsonConfig` in `main.rs` | 400 with `payload too large` |
| `max_image_bytes` | post base64-decode in `handler::extract_content` and `agent/http::stage_inputs` | 413 with `image exceeds N bytes` |
| `max_messages` | top of `chat_completions` and `agent::http::run` | 400 |
| `max_prompt_chars` | after `build_prompt`, before tokenize | 400 |
| `max_generated_tokens` | clamp inside handler (`min(client_max, config_max)`) | silent clamp |
| `max_ctx_size` | `HrmEngine::load` clamps effective ceiling to `min(runtime.ctx_size, limits.max_ctx_size)`; decode loop breaks when reached | silent break + final SSE event with `completed=false` |
| `engine.max_concurrent` | `engine_lease.acquire().await` in front of every ORT call | awaits; never rejects |
| `memory_gate.high_water` | admission check in `agent::http::run` AND `chat_completions` | 503 + `Retry-After: 1` |
| `results.per_run_bytes` + `field_trim_above` | post-emit hook in `executor.rs` after each `StepResult` is sent | silent trim; full value already on the wire |

### `memory_gate.rs` (new)

```rust
pub struct MemoryGate {
    high_water:  u64,
    low_water:   u64,
    above_water: AtomicBool,    // hysteresis: once above, stay refused until low
}

impl MemoryGate {
    pub fn admit(&self) -> Result<(), MemoryRefusal> { /* read rss, compare */ }
}

#[cfg(target_os = "macos")]
fn current_rss_bytes() -> std::io::Result<u64> {
    // mach_task_self() + task_info(MACH_TASK_BASIC_INFO_64)
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> std::io::Result<u64> {
    // parse /proc/self/status VmRSS
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn current_rss_bytes() -> std::io::Result<u64> { Ok(0) }   // gate is no-op
```

- No background thread; polled lazily on each admit (~µs cost).
- Hysteresis prevents flapping at the boundary.
- Unsupported OS: gate is a no-op (returns 0 → always admits). Documented limitation.

### `engine_lease.rs` (new)

```rust
pub struct EngineLease {
    sem: Arc<Semaphore>,            // permits = limits.engine.max_concurrent
}

impl EngineLease {
    pub async fn acquire(&self) -> OwnedSemaphorePermit { /* await */ }
}
```

Wired into `AppState` and passed to:
- `chat_completions` (around the `infer_text` / `infer_text_kv` call)
- `HrmPlanner::propose` (around the `spawn_blocking` call)
- `ReflectTool` (uses `Planner` internally — covered transitively)

### Drop discipline — concrete changes

1. **`vision_bridge::describe`** signature changes to `&self, image_bytes: Vec<u8>` (was `&[u8]`). Caller `chat_completions` passes ownership. Inside `describe`, base64 encoding consumes `image_bytes`, and the original `Vec<u8>` drops before the HTTP send. Two copies live at peak, not three.

2. **`RunContext`** gets `impl Drop { fn drop(&mut self) { self.results.clear(); /* + tracing::debug for leak diagnosis */ } }`. The field would drop on its own; explicit makes intent clear in review and provides a probe.

3. **`executor.rs` post-StepResult trim hook.** After sending a `StepResult` to SSE, scan the inserted `Value` and replace any string field > `field_trim_above` with `"<trimmed N bytes>"`. Additionally, if total `ctx.results` byte count > `per_run_bytes`, drop entries that are not referenced by any subsequent step's args (we parsed the full plan up front, so the ref graph is known). Emit a `results_trimmed` SSE event so the client knows the retention was reduced — the value already went out on the wire intact.

4. **Arc retention audit.** Read every `Arc<HrmEngine>` / `Arc<...>` clone path in the agent layer; document in the implementation PR. No cycles exist today; the audit is the deliverable, with `Weak<>` added if any cycle risk surfaces.

## Testing

Every test below must exist before the implementation lands.

| Test | Where | What it asserts |
|---|---|---|
| KV parity | export script self-test + Rust integration | 64-token greedy decode via monolithic == KV path, token-for-token. Ship-gating. |
| Per-token logits parity | Rust integ test | First-step argmax token from KV `prefill` matches monolithic `prefill` for a 16-prompt corpus. (Bit-identical logits are not expected due to attention numerical order; argmax stability is the meaningful invariant.) |
| Buffer reuse | unit test in `hrm_engine.rs` | After N successive `infer_text_kv` calls, `KvBuffers.capacity` unchanged; no allocations recorded (`dhat` or custom alloc counter under `cfg(test)`). |
| RSS stress — sequential | `tests/memory_stress.rs` | 200 sequential chat requests at `max_tokens=128`. RSS at end ≤ baseline + 50 MB. Build-failing. |
| RSS stress — concurrent | `tests/memory_stress.rs` | 8 concurrent agent runs admitted; ORT calls serialize through engine_lease. No OOM. RSS ceiling holds. |
| Watermark refusal + hysteresis | `tests/memory_gate.rs` | Mock `current_rss_bytes`; assert 503 + `Retry-After` above high water; admit below low water; hysteresis between the two. |
| Bounds rejections | `tests/bounds.rs` | oversize image → 413; oversize prompt → 400; >max_messages → 400; oversize body → 400 from JsonConfig. |
| Result-store trim | `tests/agent_results.rs` | Agent run with a tool returning 100 KB string; subsequent step reads truncated form via `{{stepN.field}}` if referenced, stub if not. SSE stream gets full value. |
| Drop on disconnect | `tests/agent_drop.rs` | Start agent run, drop the response receiver after first `StepResult`, sleep 100ms; run task exits. `RunContext::drop` fired (log probe). |
| Engine-lease serialization | `tests/engine_lease.rs` | Two concurrent chat completions; instrument `prefill` to record start/end; assert no temporal overlap. |
| Fallback path | `tests/engine_fallback.rs` | `kv_cache.enabled=true` but `decode_step.onnx` missing → warning log + monolithic fallback; existing tests pass. |
| Existing suite | `cargo test -p llm-service` | All current tests still pass. Non-negotiable. |

### Benchmarks (not tests, but expected as part of PR)

- Decode tokens/sec at 64, 256, 512 tokens — KV vs monolithic. Expect KV ~5× faster at 512.
- Peak RSS during 512-token decode — KV vs monolithic. Expect KV ~10× lower transient.
- Cold-start latency (first request after boot). KV path JITs one extra session. Budget +200–500 ms; if worse, investigate before ship.

## Rollout sequence

Each step is independently revertable.

1. **Re-export tooling.** Rewrite `scripts/export_hrm_text.py`; generate `prefill.onnx`/`decode_step.onnx` artifacts; export-script KV parity test in CI. **Commit alone, no Rust changes.** Verify parity test before any Rust work starts.

2. **Surgical bounds layer.** Add `memory_gate.rs`, `engine_lease.rs`, request bounds in handlers, smaller channels, lower `JsonConfig` limit. *No KV cache yet.* This alone prevents the M4 crash for current workloads — validates the bounds machinery in isolation.

3. **KV runtime.** Add `KvSession`, `KvBuffers`, `EngineBackend::KvCache`, `infer_text_kv`. KV path off by default (`kv_cache.enabled=false`). Run KV parity test against live model.

4. **Flip default.** `kv_cache.enabled=true`. RSS stress tests in CI must pass before merge.

5. **Cleanup.** Drop discipline (`Vec<u8>` by value in vision_bridge, `Drop` impls, trim hook), Arc retention audit notes.

## Open questions

None at design time. All bounds defaults, the export topology, the two-graph decision, and the buffer-reuse strategy are settled. Implementation may surface follow-ups; capture as separate specs.

## Follow-ups (explicitly out of scope)

- ORT `IoBinding` for true zero-copy KV reuse (if benchmarks show alloc cost dominates).
- Shared-weights external-data file (single ~2.3 GB blob shared by both graphs).
- Quantized (int8) KV cache.
- GPU / CoreML EP wiring beyond what already exists.
- Multi-permit `engine.max_concurrent` with cloned `KvBuffers` (currently 1; raising requires per-permit buffer set).
