# HRM-Text Agentic Orchestration — Design Spec

**Date:** 2026-05-20
**Status:** Draft — awaiting implementation plan
**Author:** Claude (brainstormed with Evintkoo)
**Related:**
- `docs/superpowers/specs/2026-05-20-hrm-text-runtime-swap-design.md` (Spec #1) — landed the HRM-Text runtime engine. This spec is the deferred Spec #2 from that document's "Out of scope" section: *"HRM-mirrored planner-executor agentic layer (next session)."*
- `docs/superpowers/plans/2026-05-20-hrm-text-runtime-swap.md` — implementation plan for Spec #1.

---

## 1. Goal

Expose an **agentic orchestration layer** on top of the existing HRM-Text LLM microservice. A single new endpoint, `POST /llm/v1/agent/run`, accepts an OpenAI-shaped messages array plus optional image/audio input and returns a streamed plan-and-execute trace via Server-Sent Events.

The architecture mirrors HRM-Text's hierarchical recurrent structure:

- **Slow loop (planner):** one `HrmEngine::infer_text` call that emits a structured plan in a small mini-DSL.
- **Fast loop (executor):** iterates over the plan, dispatching each step against a registry of tools (local inference endpoints, vision bridge, in-process LLM reflection, sandboxed HTTP fetch).

The endpoint is **stateless** — each request is self-contained — and streams a typed event protocol so clients can render plan + execution progress live.

## 2. Architecture

### 2.1 Module layout

All new code lives in `services/llm/src/agent/`:

```
services/llm/src/agent/
  mod.rs          — public API: run_agent(req) -> impl Stream<Event>
  prompt.rs       — planner system prompt (DSL spec embedded as const)
  dsl.rs          — Step grammar + parser (regex over line-oriented input)
  tool.rs         — Tool trait, ToolRegistry, ToolError
  tools/
    classify.rs   — HTTP → main server /classify/batch
    detect.rs     — HTTP → main server /yolo/detect (multipart + query shape)
    tts.rs        — HTTP → main server /tts/stream
    stt.rs        — HTTP → main server /stt/transcribe
    vision.rs     — Wraps existing vision_bridge as a named tool
    reflect.rs    — In-process HrmEngine::infer_text with constrained sub-prompt
    http_fetch.rs — reqwest GET with allowlist + 5s timeout
  executor.rs     — Loop: parse step → resolve {{stepN.field}} refs → dispatch → record
  sse.rs          — Event types + serde::Serialize for SSE framing
```

`mod agent;` is declared in `services/llm/src/main.rs`. The actix handler binds at:

```rust
.route("/v1/agent/run", web::post().to(agent::http::run))
```

### 2.2 Request flow

```
client ──POST──▶ actix handler (agent::http::run)
                  │
                  ▼
            agent::run_agent(req)
                  │
                  ├──▶ acquire run permit  (semaphore, max_concurrent_runs)
                  │       └─ if full → 429 (before SSE starts)
                  │
                  ├──▶ planner: HrmEngine::infer_text(system_prompt + user_msg)
                  │       └─ DSL text (max 8 lines)
                  │
                  ├──▶ dsl::parse → Vec<Step>
                  │       └─ 1 retry on parse fail with REPAIR_PROMPT
                  │
                  └──▶ executor::run(steps, &registry, &mut ctx)
                        │  for each step:
                        │     emit SSE step_start
                        │     resolve refs ({{stepN.field}})
                        │     ToolRegistry::dispatch(tool, args, deadline)
                        │     emit SSE step_result
                        │     if step.tool == "final" → emit SSE final, done
                        │  if loop falls through:
                        │     emit SSE final (synthesized fallback, completed:false)
```

A single `Arc<HrmEngine>` is shared between the planner call and any `reflect` tool invocations. State is a per-request `RunContext` (steps so far, deadline, registry handle, SSE channel) — no global mutable state.

### 2.3 Why "in-LLM-service" (vs. main-server orchestrator)

The agent code is tightly coupled to HRM-Text's quirks: 1B reliability, DSL parse retries, step budget tied to context size, in-process `reflect` calls. Co-locating with `HrmEngine` makes that coupling explicit and lets reflection be cheap (zero HTTP hop). The main server stays focused on multimodal inference; tool calls from the agent hit it the same way `vision_bridge` already does.

## 3. Mini-DSL

### 3.1 Grammar

Line-oriented, regex-parseable:

```
plan        := step+
step        := step_id WS tool_call NEWLINE
step_id     := "step" digit+ "."
tool_call   := tool_name "(" args? ")"
tool_name   := [a-z_]+
args        := arg ("," arg)*
arg         := key "=" value
key         := [a-z_][a-z0-9_]*
value       := string | ref | int | float | bool
string      := '"' [^"]* '"'        # no escapes in v1
ref         := "{{" "step" digit+ "." [a-z_]+ "}}"
int         := -? digit+
float       := -? digit+ "." digit+
bool        := "true" | "false"
```

Anything else on a line (blank, prose like "Now I will…", markdown fences) is **skipped silently**. This forgives planner verbosity from a 1B model. Lines that start with a `stepN.` prefix but fail to parse fully count as a parse error and trigger the repair flow.

### 3.2 Example plan

```
step1. classify(image=input, top_k=3)
step2. detect(image=input)
step3. reflect(prompt="Describe a photo containing a {{step1.label}} with {{step2.count}} objects detected. One sentence.")
step4. tts(text={{step3.output}}, voice="af_heart")
step5. final(answer={{step3.output}})
```

### 3.3 Termination

A plan ends when one of:

- An explicit `final(answer="…")` step is executed (normal path).
- Step count reaches `max_steps` (default 8). Executor synthesizes a fallback `final`.
- Wall-clock reaches `max_run_ms` (default 60 000). Executor emits an `error` event then a fallback `final`.

A plan that doesn't end with `final(...)` still parses; the executor just runs through it and synthesizes the fallback.

## 4. Tool surface (v1)

All eight tools are registered in `ToolRegistry::default()` at service boot. Disable individually via `[agent.tools]` config flags.

| Tool | Args | Returns (fields available as refs) |
|---|---|---|
| `classify` | `image`, `top_k:int=1` | `label:string, confidence:float, all:list` |
| `detect` | `image`, `model_version:string="v8"`, `model_size:string="n"` | `count:int, labels:list, raw:json` |
| `vision` | `image` | `description:string` |
| `reflect` | `prompt:string`, `max_tokens:int=128` | `output:string` |
| `tts` | `text:string`, `voice:string="af_heart"` | `audio_url:string, duration_ms:int` |
| `stt` | `audio` | `transcript:string` |
| `http_fetch` | `url:string`, `max_bytes:int=65536` | `status:int, body:string` |
| `final` | `answer:string` | terminal — emits SSE `final`, ends run |

### 4.1 The `input` keyword

If the original request carried image/audio payload, tools can reference it as `image=input` or `audio=input`. The handler stages these into `RunContext.inputs` before the planner runs. Other arg values are literals or `{{stepN.field}}` refs — there are no other special identifiers.

### 4.2 Type coercion

The arg parser is permissive: `top_k=3` and `top_k="3"` both work; bools accept `true`/`false`/`1`/`0`; refs always resolve to their stored type, then coerce into the tool's expected type. Unknown args are passed through to the tool, which decides whether to ignore (with a `tracing::warn!`) or error.

### 4.3 HTTP fetch tool specifics

`http_fetch` is opt-in per host:

- `allowlist` is empty by default → **all requests denied** with `ToolError::Denied`.
- Hosts matched by glob: `["api.openai.com", "*.kolosal.internal"]`.
- Private CIDRs (`10/8`, `172.16/12`, `192.168/16`, `127/8`, `::1`) are **always blocked** unless the host explicitly matches an `*.internal` entry.
- No redirects (`reqwest::redirect::Policy::none()`).
- User-Agent: `kolosal-agent/0.1`.
- Response body truncated to `max_bytes` (default 64 KiB).
- All denials emitted as `step_result{ok:false, error:"http_fetch denied: …"}`.

### 4.4 Reflect tool specifics

`reflect` calls `HrmEngine::infer_text` in-process with greedy decoding (temperature 0.0) and `max_tokens` capped at `reflect_max_tokens` (default 128). The output is collected fully before the step returns — `reflect` is not itself streaming. This bounds the per-call latency budget at roughly `reflect_max_tokens × per_token_ms`.

## 5. Planner prompt + repair

### 5.1 System prompt (held under ~400 tokens)

```
You are the PLANNER half of an HRM-Text agent. Your job: emit a numbered list
of tool calls in this exact format:

  step1. tool_name(arg=value, arg=value)
  step2. tool_name(...)
  step3. final(answer="…")

Rules:
- One step per line. Lowercase tool names. No prose, no markdown, no code fences.
- The LAST step MUST be final(answer="...") with the user-facing reply.
- Reference earlier results with {{stepN.field}} — never invent fields.
- Max 8 steps. Prefer 1–3.
- If the user asks something you can answer from text alone, emit only:
    step1. final(answer="…")

Available tools (name → return fields):
  classify(image, top_k)         → label, confidence, all
  detect(image, model_version, model_size) → count, labels, raw
  vision(image)                  → description
  reflect(prompt, max_tokens)    → output
  tts(text, voice)               → audio_url, duration_ms
  stt(audio)                     → transcript
  http_fetch(url, max_bytes)     → status, body
  final(answer)                  → terminates the run

User request:
{user_message}
{input_summary}   # auto-appended: "Image attached." or "Audio attached." or ""
```

Why this shape works at 1B: strict format is exemplified (not just described); no JSON; `final` is structural, not learned; text-only requests short-circuit to one step.

### 5.2 Parse repair

```
attempt 1: planner.infer_text(system_prompt, user_msg, max_tokens=256, temp=0.0)
            └─ parse(output)
                ├─ OK(steps) → proceed
                └─ Err(parse_err) → attempt 2
attempt 2: planner.infer_text(REPAIR_PROMPT.format(prev_output, parse_err), ..., temp=0.0)
            └─ parse(output)
                ├─ OK → proceed
                └─ Err → emit SSE `error{kind:"plan_unparseable"}` then synthesized `final`
```

`REPAIR_PROMPT` is short: shows the prior bad output, names the parse error, asks for the same content in the exact format. Both passes greedy (temp 0.0) for determinism.

### 5.3 Mid-run replanning

**Not done in v1.** If a step fails, the executor records the error in `RunContext.results` and continues to subsequent steps. The `final` step can reference `{{stepN.error}}` to degrade gracefully. Rationale: re-invoking the planner mid-run is expensive (~6 s per call) and a 1B model is prone to loops without a careful termination signal. Mid-run replanning is deferred to a follow-up spec.

## 6. Executor

### 6.1 RunContext

```rust
enum Input {
    Image { bytes: Bytes, mime: String },
    Audio { bytes: Bytes, mime: String },
    Text  { value: String },
}

struct RunContext {
    run_id:        String,                 // ULID, echoed in run_start
    started_at:    Instant,
    deadline:      Instant,                // started_at + max_run_ms
    inputs:        HashMap<String, Input>, // "input" → Image / Audio / Text staged from request body
    results:       HashMap<String, Value>, // "step1" → {label:"cat", confidence:0.81, ...}
    sse_tx:        mpsc::Sender<AgentEvent>,
    max_steps:     usize,                  // default 8
    per_tool_ms:   u64,                    // default 5000
}
```

The handler stages exactly one entry into `inputs` under the key `"input"`: an `Input::Image` if `request.input.image` is present, otherwise `Input::Audio` if `request.input.audio` is present, otherwise nothing. A future spec may add `input2`, `input3` for multi-image agents.

### 6.2 Main loop

```rust
for (idx, step) in steps.iter().enumerate() {
    if idx >= ctx.max_steps { break; }
    if Instant::now() >= ctx.deadline {
        emit AgentEvent::Error { kind: "deadline_exceeded" };
        break;
    }
    if ctx.sse_tx.is_closed() { return Ok(()); }   // client disconnected

    emit AgentEvent::StepStart { idx: idx+1, id: step.id.clone(), tool: step.tool.clone(), args: redact(&step.args) };

    let resolved = match resolve_refs(&step.args, &ctx.results, &ctx.inputs) {
        Ok(v)  => v,
        Err(e) => { emit_step_err(ctx, idx+1, &step.id, "ref_unresolved", e); continue; }
    };

    let tool_deadline = min(ctx.deadline, Instant::now() + per_tool_ms);
    let out = registry.dispatch(&step.tool, resolved, tool_deadline).await;

    match out {
        Ok(value)    => { ctx.results.insert(step.id.clone(), value.clone());
                          emit AgentEvent::StepResult { idx: idx+1, id: step.id.clone(),
                                                        ok: true, value, duration_ms }; }
        Err(toolerr) => { ctx.results.insert(step.id.clone(),
                              json!({"error": toolerr.to_string()}));
                          emit AgentEvent::StepResult { idx: idx+1, id: step.id.clone(),
                                                        ok: false, error: toolerr.to_string(),
                                                        duration_ms }; }
    }

    if step.tool == "final" { return Ok(()); }
}

emit AgentEvent::Final {
    answer: fallback_summary(&ctx.results),
    completed: false,
    ...
};
```

### 6.3 Ref resolution rules

- `{{stepN.field}}` → lookup `ctx.results["stepN"]`, get `.field`. Missing key → `RefError::Unresolved`.
- `{{stepN.error}}` → empty string when the step succeeded; the error message when it failed. Lets `final` answers degrade.
- Coerced into the arg's expected type.
- **Nested refs are not supported in v1** (`{{step1.all[0].label}}` etc.). Tools needing indexed access expose a flat top-level field (e.g., `classify.label` is top-1).

### 6.4 Safety limits

All in `[agent]` config with the defaults shown:

| Limit | Default | Rationale |
|---|---|---|
| `max_steps` | 8 | Matches planner prompt cap; hard stop. |
| `max_run_ms` | 60000 | Wall-clock; same as reverse-proxy timeout. |
| `per_tool_ms` | 5000 | Bounds any one HTTP tool call. `reflect` overrides to 15000. |
| `max_concurrent_runs` | 4 | Semaphore on actix handler; 429 above. HrmEngine is single-threaded. |
| `reflect_max_tokens` | 128 | Per-call cap on reflection output. |
| `planner_temperature` | 0.0 | Greedy decode for plan + repair determinism. |

### 6.5 Cancellation

If the SSE client disconnects, the `mpsc::Receiver` is dropped; the executor checks `sse_tx.is_closed()` between steps and **returns early without emitting further events** (the channel is closed — no one would receive them). Already-in-flight tool calls finish naturally — no `reqwest` mid-call cancellation in v1 (risk of resource leaks). The semaphore permit is released via RAII when the future drops, so concurrent-run slots free immediately.

## 7. SSE event protocol

### 7.1 Endpoint

`POST /llm/v1/agent/run`, proxied through main server at `http://127.0.0.1:8000/llm/v1/agent/run`. Returns `Content-Type: text/event-stream; charset=utf-8`. Each event is one SSE frame: `event: <name>\ndata: <json>\n\n`.

### 7.2 Request body

```json
{
  "messages": [
    {"role": "user", "content": "Describe what's in this image and read it aloud."}
  ],
  "input": {
    "image": "data:image/jpeg;base64,…",
    "audio": "data:audio/wav;base64,…"
  },
  "config": {
    "max_steps": 6,
    "max_run_ms": 30000,
    "temperature": 0.0
  }
}
```

`messages` is OpenAI-shaped for client familiarity, but only the last user message is fed to the planner (v1 is stateless — no history threading). `input` and `config` are optional.

### 7.3 Event schema

| Event | When | Payload |
|---|---|---|
| `run_start` | First frame, before planner runs | `{run_id, model:"hrm-text-1b", deadline_ms, max_steps}` |
| `plan` | Once, after parse succeeds | `{steps:[{id, tool, args}, ...], retries:0\|1}` |
| `step_start` | Before each tool dispatch | `{idx, id, tool, args}` (args redacted) |
| `step_result` | After each tool returns | `{idx, id, ok:bool, value\|error, duration_ms}` |
| `error` | Non-recoverable run error | `{kind, message}` |
| `final` | Last frame, always emitted | `{answer, steps_executed, total_ms, completed:bool}` |
| `[DONE]` | Stream terminator (raw `data: [DONE]`) | mirrors OpenAI streaming convention |

`completed:false` signals the run hit `max_steps` or `deadline_ms` before reaching `final(...)`. The `answer` field is then a synthesized fallback from accumulated results.

### 7.4 Argument redaction in `step_start`

Raw image/audio payloads aren't echoed — the SSE shows `"image":"<input>"` or `"image":"<32 KB jpeg>"` instead of base64. Keeps the SSE stream small and avoids leaking large payloads through middleware proxies or browser DevTools.

### 7.5 Error semantics

| Server condition | HTTP status | Frames emitted |
|---|---|---|
| Bad request body (missing `messages`, etc.) | 400 | None — JSON error response. |
| Concurrency limit (semaphore full) | 429 | None — JSON error response. |
| Planner fails twice on parse | 200 | `run_start` → `error{kind:"plan_unparseable"}` → `final{completed:false}` → `[DONE]` |
| HrmEngine inference panics | 500 (if before stream starts) / dropped connection (mid-stream) | Partial stream, client treats as failure. |
| Deadline mid-run | 200 | All prior events → `error{kind:"deadline_exceeded"}` → `final{completed:false}` → `[DONE]` |

Why 200 + `final{completed:false}` instead of switching to 5xx mid-stream: SSE has already started; clients are expected to inspect `completed` and `error` events.

### 7.6 Curl example

```bash
curl -N -X POST http://127.0.0.1:8000/llm/v1/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2? Then speak the answer."}]}'

event: run_start
data: {"run_id":"01HZ…","model":"hrm-text-1b","deadline_ms":60000,"max_steps":8}

event: plan
data: {"steps":[
  {"id":"step1","tool":"reflect","args":{"prompt":"2+2 = ?","max_tokens":8}},
  {"id":"step2","tool":"tts","args":{"text":"{{step1.output}}","voice":"af_heart"}},
  {"id":"step3","tool":"final","args":{"answer":"4 (also spoken)."}}
],"retries":0}

event: step_start
data: {"idx":1,"id":"step1","tool":"reflect","args":{"prompt":"2+2 = ?","max_tokens":8}}

event: step_result
data: {"idx":1,"id":"step1","ok":true,"value":{"output":"4"},"duration_ms":1842}

event: step_start
data: {"idx":2,"id":"step2","tool":"tts","args":{"text":"4","voice":"af_heart"}}

event: step_result
data: {"idx":2,"id":"step2","ok":true,"value":{"audio_url":"/tts/cache/…","duration_ms":620},"duration_ms":740}

event: final
data: {"answer":"4 (also spoken).","steps_executed":3,"total_ms":2581,"completed":true}

data: [DONE]
```

## 8. Configuration

Appended to `services/llm/config.toml`:

```toml
[agent]
enabled                = true
max_steps              = 8
max_run_ms             = 60000
per_tool_ms            = 5000
max_concurrent_runs    = 4
reflect_max_tokens     = 128
planner_temperature    = 0.0

[agent.http_fetch]
enabled                = true
allowlist              = []          # empty = block all by default
max_bytes              = 65536
follow_redirects       = false

[agent.tools]
main_server_base       = "http://127.0.0.1:8000"
classify_endpoint      = "/classify/batch"
detect_endpoint        = "/yolo/detect"
tts_endpoint           = "/tts/stream"
stt_endpoint           = "/stt/transcribe"
```

The `main_server_base` defaults to the same value `[vision_bridge]` uses; both sections will share it via a small `Default` impl rather than duplicating the literal in code.

## 9. Testing strategy

### 9.1 DSL parser unit tests (`dsl.rs`, ~12)

- Happy path: each of the 8 tools, valid args.
- Permissive value coercion: `top_k=3` vs `top_k="3"`, bool variants.
- Refs: `{{step1.label}}`, `{{step1.error}}`, missing ref → structured error.
- Forgiving prose-skipping: blank lines, "Now I will…" prose, markdown fences all ignored.
- Hard fails: malformed parens, unknown tool name.
- Repair path: feed a known-bad output through repair prompt, assert clean reparse.

### 9.2 Tool unit tests (`tools/*.rs`, ~16)

- Per HTTP tool: success, 500, 400, timeout, body-too-large.
- `http_fetch` allowlist: denied host returns `ToolError::Denied` without sending; private-CIDR blocking even when host glob matches.
- `reflect`: real `HrmEngine` (gated like existing tests — skip if `models/hrm-text-1b/model.onnx` missing).
- `vision`: delegates to existing `vision_bridge`; verify wiring with a mockito double.

### 9.3 Executor integration tests (`executor.rs`, ~8)

Introduce a tiny `Planner` trait so executor can be tested without the real ONNX model:

```rust
#[async_trait]
trait Planner { async fn propose(&self, prompt: &str) -> Result<String>; }
```

Real impl wraps `HrmEngine`; tests use a canned-output stub.

- Plan parses cleanly, all steps succeed, `final` emitted.
- Plan parses cleanly, step 2 fails, `final` uses `{{step2.error}}` to apologize.
- Plan unparseable twice → `error{kind:"plan_unparseable"}` + fallback final.
- `max_steps` hit → `final{completed:false}` synthesized.
- Deadline trips between step 3 and step 4 → `error{kind:"deadline_exceeded"}` + fallback.
- SSE client disconnect → executor breaks loop before next dispatch (assert via `sse_tx.is_closed()`).

### 9.4 End-to-end smoke test (`tests/agent_smoke.rs`, gated)

Boots the LLM service in-process with `actix_test`, mocks main-server tools via `mockito`, sends `{"messages":[{"role":"user","content":"What is 2+2?"}]}`, asserts the SSE stream contains `plan`, at least one `step_result`, and a `final` with non-empty answer. Single happy-path test; broader scenarios stay in the executor integration suite.

### 9.5 Regression guard

Extend the philosophy of `tests/no_llama_cpp.rs`: add a test asserting the DSL parser rejects JSON-shaped output. Catches the "someone snuck JSON into the planner prompt" regression.

### 9.6 Manual verification

- Boot main + LLM service; hit `/llm/v1/agent/run` with text-only, image, and "speak the answer" queries. Confirm `plan`, `step_*`, `final` events stream in order.
- Verify `max_concurrent_runs` by sending 5 simultaneous requests; expect 4 succeed, 1 returns 429.
- Verify `http_fetch` allowlist by configuring `allowlist=["example.com"]` and requesting both `example.com` and `localhost`; expect success and `denied`.

## 10. Rollout & risk

- **Additive only.** No existing endpoint changes. `/v1/chat/completions` keeps working as today; `/v1/agent/run` is new.
- **Feature flag.** `[agent].enabled=false` makes `POST /v1/agent/run` return 404. Operators can disable the surface without redeploy.
- **Resource ceiling.** `max_concurrent_runs=4` prevents the agent layer from starving simple chat. HrmEngine is the bottleneck; semaphore sized to leave inference headroom.
- **Vision bridge bug inheritance.** The `vision` tool wraps the existing `vision_bridge`, which has a known shape mismatch against `/yolo/detect`. That bug stays out of scope here — the `vision` tool degrades gracefully via the same fallback. The new `detect` tool uses the correct multipart + query shape and works even when `vision` doesn't.
- **HrmEngine threading.** All `HrmEngine::infer_text` calls (planner + reflect) run via `tokio::task::spawn_blocking`. The semaphore limit ensures we don't oversubscribe blocking threads.

## 11. Out of scope (explicit)

- Session/memory across runs (`session_id`, persistent state).
- Nested ref paths (`{{step1.all[0].label}}`).
- Mid-run replanning.
- OpenAI tool-use compatibility shape.
- Playground UI tab (deferred — would consume the SSE protocol the same way the existing chat tab does).
- Multi-agent coordination (a single planner + executor pair, not a fleet).
- KV-cache optimization in `HrmEngine` (inherits Spec #1's O(N²) prefill; performance work is its own spec).

## 12. Design rationale summary

| Decision | Why |
|---|---|
| New endpoint, not extending `/v1/chat/completions` | Keeps OpenAI-compatible chat clean; structured progress benefits from its own SSE schema. |
| Mini-DSL, not JSON | 1B models break JSON; line-oriented DSL with prose tolerance is reliable. |
| In-LLM-service, not main-server | Reflection latency dominates; co-location avoids HTTP per LLM call. |
| Stateless v1 | Smallest surface, no DB / TTL ops burden. Session support is purely additive later. |
| Slow/fast = planner/executor | Mirrors HRM-Text's H-cycles/L-cycles structurally. Plan once (slow), execute many (fast). |
| No mid-run replanning | Replanning at 1B + ~6s/call risks loops; v1 prefers graceful degradation via `{{stepN.error}}`. |
| HTTP fetch off by default | Empty allowlist = denied. Operator opts in per host; safest default for a new surface. |
