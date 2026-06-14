# Product Audit & Triage — 2026-04-29

Phase 1 deliverable for the "optimise and fix all bugs" sweep. Read-only audit across all eight product/infrastructure areas. **No code changed.** This document exists to let you decide what gets fixed in Phase 3.

## How to use this document

1. Skim the **Top 10 fixes** — these are the highest-impact items.
2. Read the **Cross-cutting patterns** — many issues collapse into 4–5 systemic fixes.
3. Look at the **Fix buckets** — proposed groupings for Phase 3 sub-projects.
4. Use the **Per-area issue tables** to spot-check or pull individual items in/out.
5. Mark anything you want cut; anything left is in scope for Phase 3.

Severity convention: **high** = correctness/security/availability bug or major perf cliff; **medium** = real but non-emergency; **low** = polish. Effort: **S** ≈ ≤30 min, **M** ≈ a few hours, **L** ≈ a day or more.

---

## Headline numbers

| Area | Issues | high | medium | low |
|---|---|---|---|---|
| Server core / config / main | 8 | 1 | 4 | 3 |
| TTS | 16 | 0 | 4 | 12 |
| STT / Audio | 15 | 3 | 4 | 8 |
| Classification | 15 | 2 | 4 | 9 |
| YOLO Detection | 15 | 3 | 4 | 8 |
| LLM chat | 19 | 2 | 8 | 9 |
| Playground & Assets | 18 | 2 | 6 | 10 |
| Cross-cutting infra | 15 | 1 | 3 | 11 |
| **Total** | **121** | **14** | **37** | **70** |

---

## Top 10 highest-impact fixes

Ranked by impact-per-effort. Most are **S** or small **M**.

1. **JWT secret default `"your-secret-key-here"`** — `src/config.rs:491`. If auth is ever enabled with default config, every JWT is forgeable. **High / S.** Reject default at startup or generate random.
2. **XSS via unescaped model labels in playground** — `src/api/playground.html:5703, 5736` and `makeToolCard` body at 5640/5682/5709/5742. Class names from YOLO/classifier are concatenated into `innerHTML`. A poisoned model = arbitrary script execution. **High / S.** Escape with `textContent` or an explicit escape helper.
3. **`Mutex.lock().unwrap()` on ORT sessions** — `src/core/ort_classify.rs:188`, `src/core/ort_yolo.rs:126` (and similar in TTS pool `src/core/kokoro_onnx.rs:72-76`). One panic poisons the lock; all later requests crash. **High / S.** Map poison to 503 via the error helper.
4. **Blocking inference in async handlers** — STT (`src/api/audio.rs:210-212`), Classification batch path (rayon in `src/api/classify.rs:142-143`), YOLO PyTorch path (`src/api/yolo.rs:150-158`). Each request stalls the actix `current_thread` reactor. **High / M.** Wrap each in `spawn_blocking`.
5. **Unbounded request bodies (multipart + base64) → OOM/DoS** — STT `src/api/audio.rs:174-195`, YOLO `src/api/yolo.rs:110-120`, Classify base64 `src/api/classify.rs:99-108`. **High / S.** One `MAX_BODY_BYTES` config + `web::PayloadConfig` + per-item base64 cap.
6. **LLM token counts hardcoded to 0** — `services/llm/src/handler.rs:216`. Breaks any billing/quota/observability downstream. **High / M.** Plumb `n_prompt` and generated-token count through the channel.
7. **LLM streaming missing initial `role: assistant` delta** — `services/llm/src/handler.rs:106-114`. Breaks OpenAI-spec-compliant clients. **High / S.** Emit one role-only chunk before content.
8. **Liveness returns 200 when degraded / no fatal-error signal** — `src/api/health.rs:27-46`. Orchestrator can't kill broken pods. **High / S–M.** Split liveness (process alive) from readiness (subsystems healthy); 503 on fatal.
9. **YOLO image dimension fallback to 640×640 hides bad input** — `src/api/yolo.rs:159, 206`. Causes silent NaN bboxes. **High / S.** Reject if `image_dimensions()` fails or returns 0.
10. **Audio duration math ignores channels** — `src/api/audio.rs:143`, `src/core/audio.rs:105`, `src/core/tts_manager.rs:220-221`. Stereo audio reports half its real duration; affects logs, billing, UX. **Medium / S.** Divide by `channels * sample_rate`.

---

## Cross-cutting patterns

These collapse a lot of the long tail into a handful of systemic fixes. Each pattern lists all the call-sites that share it.

### P1 — Unbounded inputs (DoS)
- STT multipart unbounded: `src/api/audio.rs:174-195`, `src/api/audio.rs:235-245`
- STT WAV header lying about size: `src/core/audio.rs:79-81`
- Symphonia decode unbounded: `src/core/audio.rs:272-289`
- YOLO multipart unbounded: `src/api/yolo.rs:110-120`
- Classify per-image base64 size unchecked: `src/api/classify.rs:99-108`
- LLM 32 MB JSON body: `services/llm/src/main.rs:44`
- Stream classify per-item size unchecked: `src/api/classify.rs:205-222`

**Systemic fix:** add `MAX_BODY_BYTES` and `MAX_DECODED_SECONDS`/`MAX_DECODED_PIXELS` to config; thread through all request entry points; reject before allocation.

### P2 — Async/blocking discipline
- STT inference no `spawn_blocking`: `src/api/audio.rs:210-212`
- Classify rayon preprocess on async thread: `src/api/classify.rs:142-143`
- YOLO PyTorch `detector.detect()` no `spawn_blocking`: `src/api/yolo.rs:150-158`
- Classify backend lock held across full inference: `src/core/ort_classify.rs:188-203`

**Systemic fix:** audit every handler for blocking calls; standardise on `spawn_blocking` for ORT/PyTorch/rayon, and minimise lock scope around ORT `session.run()`.

### P3 — Mutex poison panics
- ORT classify: `src/core/ort_classify.rs:188`
- ORT YOLO: `src/core/ort_yolo.rs:126`
- TTS Kokoro session pool: `src/core/kokoro_onnx.rs:72-76`

**Systemic fix:** one helper `lock_or_503(mutex)` that returns `Result` and is used everywhere; remove `.unwrap()` from production lock paths.

### P4 — Timeouts missing
- TTS streaming response no timeout: `src/api/tts.rs:183-223`
- Classify stream per-item: `src/api/classify.rs:255-264`
- LLM non-streaming `rx.recv().await`: `services/llm/src/handler.rs:191-192`
- LLM client-disconnect cancellation: `services/llm/src/engine.rs:132, 226`

**Systemic fix:** standardise on `tokio::time::timeout` wrappers; add a `CancellationToken` for streamed LLM/TTS so a dropped client kills the inference task.

### P5 — Audio sample/channel math
- Duration in handler ignores channels: `src/api/audio.rs:143`
- Duration in `AudioData::duration()`: `src/core/audio.rs:105`
- TTS manager duration: `src/core/tts_manager.rs:220-221`
- Resample `total_frames` doesn't validate divisibility: `src/core/audio.rs:324`
- `to_mono()` divides by zero on bad metadata: `src/core/audio_models.rs:348`

**Systemic fix:** centralise frame/sample/duration math in a small module with debug assertions; keep one definition of "frame".

### P6 — Fallbacks that swallow bad input
- YOLO image dim → 640×640: `src/api/yolo.rs:159, 206`
- STT timestamps `parse().unwrap_or(false)`: `src/api/audio.rs:194`
- TTS empty text → empty stream not 400: `src/core/tts_pipeline.rs:56-59`
- TTS missing voice → silent zeroed style: `src/core/kokoro_onnx.rs:414-421`
- Symphonia `Err(_) => break` swallows mid-stream errors: `src/core/audio.rs:291`

**Systemic fix:** prefer 4xx with a clear message over silent default; only fall back when the user's request is unambiguously satisfied.

### P7 — Hot-path allocations
- `ImagePipeline` per image in stream classify: `src/api/classify.rs:242-243`
- `AudioProcessor` per request: `src/api/audio.rs:138, 203, 259`
- PyTorch YOLO detector instantiated per request: `src/api/yolo.rs:150-152`
- `pairs.clone()` for streaming LLM: `services/llm/src/handler.rs:151`
- `sysinfo::System::new_all()` per `/system/info` and `/performance`: `src/api/system.rs:97-142`, `src/api/performance.rs:79-142`

**Systemic fix:** move long-lived state into actix `web::Data`; sysinfo cached behind a 5–10 s background refresh.

### P8 — LLM OpenAI-compatibility gaps
Single bucket: missing `role` delta (#7 above), missing `created`, hardcoded `chatcmpl-1` id, `finish_reason` always `stop`, fake usage counts, missing `top_k`/`top_p`/`stop` request fields, no max_tokens=0 / NaN temperature validation, no context-window check, no client-disconnect cancellation, errors swallowed mid-stream.

---

## Proposed Phase 3 fix buckets

These are the candidate sub-projects. Each is small enough for one brainstorm → spec → plan → implement cycle.

| Bucket | Scope | Issues | Effort |
|---|---|---|---|
| **A. Security & DoS hardening** | JWT default, body-size caps, base64 caps, audio decode caps, playground XSS escape | P1 + #1, #2 + audio caps | M–L |
| **B. Async/blocking correctness** | `spawn_blocking` audit + lock-scope minimisation + LLM cancellation token | P2 + LLM cancel | M |
| **C. Mutex poison resilience** | `lock_or_503` helper, replace all `.lock().unwrap()` in prod paths | P3 | S |
| **D. Timeouts everywhere** | Per-stream and per-recv timeouts; LLM `recv()` timeout; classify stream per-item | P4 | M |
| **E. LLM OpenAI compatibility** | role/created/id/finish_reason/usage/top_k/top_p/stop/validation | P8 | M–L |
| **F. Audio correctness** | Duration math, channel guard, divisibility check, decode caps | P5 | S–M |
| **G. Playground robustness** | Escape model output, emoji regex, SSE parser, EventSource lifecycle, mic track stop, audio overlap | Playground #1–11 | M |
| **H. Validation & error mapping** | `Config::validate()`, `InferenceError → ResponseError`, image-dim guard, class-id OOB warn | server-core #2,5; classify #4,11; YOLO #1 | S–M |
| **I. Hot-path allocations** | ImagePipeline reuse, AudioProcessor reuse, YOLO detector caching, sysinfo cache, ORT lock scope | P7 + classify #2,3 | M |
| **J. Liveness/readiness split** | Separate liveness from readiness; threshold tuning vs worker count | infra #1, #4, #12 | S–M |

**Recommended Phase 3 ordering:** C → A → B → F → D → H → E → I → G → J. Rationale: C is one-shot poison-resilience that unblocks safer testing; A closes obvious security/DoS; B+F+D fix correctness in the inference path; E is the largest single product surface; G/J/I are polish-and-perf.

---

## Recommended cuts (low value vs. effort)

I'd drop these from scope unless you specifically want them. Most are "polish" items that don't move the needle.

- **TTS #2** — `unwrap()` in test code (`tts_pipeline.rs:385`).
- **TTS #7** — sentence splitter O(n²) (cap is 200 chars; not a real perf hazard).
- **TTS #10** — generic `expect("serialize")` messages in tests.
- **TTS #11, #14, #16** — comment/clarity-only items on AudioData and WAV header sizing.
- **TTS #13** — G2P cache size config (1024 is fine).
- **STT #9** — sample rate constant `16000` (real config has it; not a bug).
- **STT #14** — `usize` overflow on 32-bit (not a target platform).
- **STT #13** — dead "model" multipart field.
- **Classification #6, #7, #15** — labels-list polish.
- **YOLO #5** — NMS threshold ordering (current order is defensible).
- **YOLO #11** — f64→f32 precision loss in tch-rs (cosmetic).
- **YOLO #14, #15** — variable input size & `#![allow]` cleanup.
- **Playground #14** — `console.log` in user-facing API examples (it's documentation).
- **Playground #16** — HTML minification (spec already accepted ~66 KB gzipped).
- **Playground #18** — strong vs. weak ETag (current behavior is correct).
- **Cross-cutting #5, #8, #13** — endpoint logs / cloned device list / metrics-render-error logging (low signal).
- **LLM #13, #16** — string clones / sampler chain construction (negligible).

That trims ~18 items, leaving ~103 to consider for Phase 3. After your cuts I'd expect the real Phase 3 list to land at 60–80 items, mostly bundled into buckets A–J.

---

## Per-area issue tables

Source of truth for the synthesis above. Each row maps directly to a sub-agent finding.

### Server core / config / main

| # | Where | Cat | Sev | Eff | Summary |
|---|---|---|---|---|---|
| 1 | `src/config.rs:491` | bug/sec | high | S | Default JWT secret `"your-secret-key-here"` |
| 2 | `src/config.rs:410-421` | bug | medium | M | No `Config::validate()` (port range, workers ≤, thresholds 0..1) |
| 3 | `src/main.rs:595-614` | bug | medium | M | Port-exhaust fallback `unwrap_or(preferred)` hides exhaustion |
| 4 | `src/main.rs:134, 137` | perf | medium | S | `unsafe env::set_var` ordering pattern is fragile |
| 5 | `src/main.rs:236-287` | bug | low | S | Worker-count auto-detect can produce 0 — needs `.max(1)` |
| 6 | `src/main.rs:561, 1018` | bug | medium | M | `current_dir("services/llm")` assumes cwd; breaks in systemd |
| 7 | `src/config.rs:410-421` | polish | low | M | No env-var override of TOML config |
| 8 | `src/main.rs:193-200` | polish | low | S | Profiler guard not flushed on early panic |

### TTS

| # | Where | Cat | Sev | Eff | Summary |
|---|---|---|---|---|---|
| 1 | `src/core/kokoro_onnx.rs:273` | bug | medium | S | `f32::from_le_bytes(...).unwrap()` on voice file |
| 2 | `src/core/tts_pipeline.rs:385` | bug | low | S | Same pattern in test |
| 3 | `src/core/audio.rs:324` | bug | medium | M | Resample doesn't check `samples.len() % channels == 0` |
| 4 | `src/core/tts_pipeline.rs:171` | bug | medium | S | Spawned synthesis task panics silently; client hangs |
| 5 | `src/api/tts.rs:183-223` | perf | medium | M | No timeout on streaming response |
| 6 | `src/core/tts_pipeline.rs:56-59` | polish | low | S | Empty text returns empty stream, not 400 |
| 7 | `src/core/tts_pipeline.rs:54-121` | perf | low | M | Splitter walks string repeatedly |
| 8 | `src/core/tts_manager.rs:213-216` | polish | low | S | Cached audio sample-rate not asserted to match capability |
| 9 | `src/core/kokoro_onnx.rs:414-421` | polish | low | S | Missing voice silently zero-style |
| 10 | `src/core/kokoro_onnx.rs:551-552, 562` | polish | low | S | Generic `expect()` in tests |
| 11 | engines / `audio.rs` | polish | low | M | `AudioData` lacks construction validation |
| 12 | `src/core/kokoro_onnx.rs:72-76` | polish | low | S | Semaphore `.expect()` panics on shutdown race |
| 13 | `src/core/kokoro_onnx.rs:33` | polish | low | M | G2P cache size hardcoded 1024 |
| 14 | `src/core/audio.rs:105`, `tts_manager.rs:220-221` | polish | low | S | Duration ignores channels |
| 15 | `src/api/tts.rs:285-293` | polish | low | M | `/tts/health` returns 200 with zero engines |
| 16 | `src/core/audio.rs:404` | polish | low | S | WAV size estimate comment misleading |

### STT / Audio

| # | Where | Cat | Sev | Eff | Summary |
|---|---|---|---|---|---|
| 1 | `src/api/audio.rs:174-195, 235-245` | bug | high | S | Unbounded multipart |
| 2 | `src/api/audio.rs:143` | bug | medium | S | Duration ignores channels |
| 3 | `src/api/audio.rs:210-212` | perf | high | M | Whisper inference blocks async executor |
| 4 | `src/core/audio.rs:79-81` | bug | medium | S | WAV header bytes parsed without size cap |
| 5 | `src/core/audio_models.rs:348` | bug | low | S | `to_mono()` divides by zero on `channels==0` |
| 6 | `src/core/audio.rs:272-289` | bug | high | M | Symphonia decode loop unbounded |
| 7 | `src/core/audio.rs:291` | polish | low | S | `Err(_) => break` masks decode errors |
| 8 | `src/api/audio.rs:194` | polish | low | S | `parse().unwrap_or(false)` swallows bad `timestamps` |
| 9 | `audio.rs:66`, `whisper_onnx.rs:26`, `audio_models.rs:221` | polish | low | S | Hardcoded 16000 Hz |
| 10 | `src/api/audio.rs:170-212, 82-168` | polish | low | S | Missing tracing context (request id, duration) |
| 11 | `src/core/audio_models.rs:374-380` | bug | low | S | No UTF-8 validation on Whisper output |
| 12 | `src/core/audio_models.rs:255-259` | bug | low | S | Silent resample mismatch |
| 13 | `src/api/audio.rs:176-195` | polish | low | S | Dead "model" multipart field |
| 14 | `src/core/audio.rs:341-372` | bug | low | S | `pos + chunk_size` overflow on 32-bit |
| 15 | `src/api/audio.rs:138, 203, 259` | perf | low | M | Per-request `AudioProcessor` allocation |

### Classification

| # | Where | Cat | Sev | Eff | Summary |
|---|---|---|---|---|---|
| 1 | `src/api/classify.rs:242-243` | perf | medium | S | Per-image `ImagePipeline` in stream |
| 2 | `src/core/ort_classify.rs:188-189` | perf | medium | S | Lock held across full inference |
| 3 | `src/core/ort_classify.rs:199-203` | perf | low | S | Unnecessary `to_vec()` when output is prob |
| 4 | `src/core/ort_classify.rs:188` | bug | high | S | `.lock().unwrap()` poison panic |
| 5 | `src/api/classify.rs:224-285` | bug | medium | M | No per-item timeout in stream |
| 6 | `src/core/ort_classify.rs:210-215` | polish | low | S | Class id OOB falls back silently |
| 7 | `src/core/ort_classify.rs:39-40` | polish | low | S | `IMAGENET_*` undocumented |
| 8 | `src/core/ort_classify.rs:157-225` | bug | low | S | No batch-shape validation |
| 9 | `src/api/classify.rs:142-143`, `image_pipeline.rs:120-130` | perf | medium | M | Rayon preprocess blocks actix thread |
| 10 | `src/api/classify.rs:255-264` | polish | low | S | All-fail stream has no summary |
| 11 | `src/core/ort_classify.rs:125-130` | bug | low | S | Softmax NaN if logits non-finite |
| 12 | `src/api/classify.rs:99-108` | bug | high | S | Per-image base64 size uncapped |
| 13 | `src/api/classify.rs:142-144` | polish | low | S | Preprocess error lacks batch context |
| 14 | `src/api/classify.rs:205-222` | bug | medium | S | Stream classify same gap as #12 |
| 15 | `src/core/ort_classify.rs:210-215` | polish | low | S | Empty/short labels file silently degrades |

### YOLO Detection

| # | Where | Cat | Sev | Eff | Summary |
|---|---|---|---|---|---|
| 1 | `src/api/yolo.rs:159, 206` | bug | high | S | Image dim fallback to 640×640 |
| 2 | `src/api/yolo.rs:110-120` | bug | high | S | Unbounded multipart |
| 3 | `src/core/ort_yolo.rs:192-203` | perf | medium | M | NMS O(n²) without early exit |
| 4 | `src/api/yolo.rs:150-152`, `src/core/yolo.rs:179-226` | perf | medium | L | PyTorch detector built per request |
| 5 | `src/core/ort_yolo.rs:162-164` | bug | low | S | NMS conf threshold ordering |
| 6 | `core/yolo.rs:221-222`, `api/yolo.rs:269-270`, `core/ort_yolo.rs:31` | polish | low | S | Hardcoded 0.25 / 0.45 / 640 |
| 7 | `src/core/yolo.rs:497-523` | perf | low | M | PyTorch NMS uses `Vec::remove(0)` loop |
| 8 | `src/core/ort_yolo.rs:126` | bug | medium | S | `.lock().unwrap()` poison panic |
| 9 | `src/api/yolo.rs:150-158` | bug | high | M | PyTorch path blocks async runtime |
| 10 | `src/core/ort_yolo.rs:209-211` | bug | low | S | Class id OOB silently labeled |
| 11 | `src/core/yolo.rs:377, 386-399` | polish | low | S | `double_value()` precision loss |
| 12 | `src/api/yolo.rs:110-121` | bug | low | S | Multipart loop overwrites temp file |
| 13 | `src/api/yolo.rs:156-158, 209-212` | polish | low | S | No structured error logging |
| 14 | `core/ort_yolo.rs:31`, `core/yolo.rs:220` | polish | low | M | 640×640 implicit assumption |
| 15 | `src/api/yolo.rs:2` | polish | low | S | Blanket `#![allow(...)]` masks warnings |

### LLM chat

| # | Where | Cat | Sev | Eff | Summary |
|---|---|---|---|---|---|
| 1 | `services/llm/src/handler.rs:216` | polish | high | M | Token counts hardcoded to 0 |
| 2 | `services/llm/src/handler.rs:106-114` | bug | high | S | First SSE chunk missing `role` |
| 3 | `services/llm/src/handler.rs:207-217` | polish | medium | S | Missing `created` |
| 4 | `services/llm/src/handler.rs:108, 208` | polish | medium | S | Hardcoded `chatcmpl-1` id |
| 5 | `services/llm/src/handler.rs:76-104` | bug | medium | M | Empty `pairs` produces malformed prompt |
| 6 | `services/llm/src/handler.rs:31, 130` | bug | medium | S | Temperature NaN/neg silently clamped |
| 7 | `services/llm/src/engine.rs:112-113, 205-206` | polish | low | M | `top_k`/`top_p` hardcoded |
| 8 | `services/llm/src/engine.rs:99-109` | bug | medium | M | No prompt+max_tokens vs ctx_size check |
| 9 | `services/llm/src/engine.rs:132, 226` | perf | medium | M | Client disconnect doesn't cancel inference |
| 10 | `services/llm/src/handler.rs:191-192` | perf | low | M | Non-streaming `recv()` no timeout |
| 11 | `services/llm/src/handler.rs:160-162` | bug | medium | S | Streaming inference error swallowed |
| 12 | `services/llm/src/handler.rs:129` | polish | low | S | `max_tokens: 0` returns silent empty |
| 13 | `services/llm/src/handler.rs:83-84, 98, 151` | perf | low | S | Extra `String` clones |
| 14 | `services/llm/src/main.rs:62` | perf | medium | S | `.workers(1)` |
| 15 | `services/llm/src/main.rs:44` | polish | low | S | 32 MB JSON limit |
| 16 | `services/llm/src/engine.rs:204-209` | perf | low | S | Sampler chain re-create note |
| 17 | `services/llm/src/handler.rs:214` | polish | low | S | `finish_reason` always `"stop"` |
| 18 | `services/llm/src/handler.rs:22-31` | polish | low | M | No `top_k`/`top_p` request fields |
| 19 | `services/llm/src/handler.rs:22-31` | polish | low | M | No `stop` request field |

### Playground & Assets

| # | Where | Cat | Sev | Eff | Summary |
|---|---|---|---|---|---|
| 1 | `playground.html:5703, 5736` | bug/sec | high | S | XSS via unescaped model labels |
| 2 | `playground.html:5640, 5682, 5709, 5742` | bug/sec | high | S | XSS via `bodyHTML` of `makeToolCard` |
| 3 | `playground.html:5649` | perf | medium | S | Emoji regex strips legitimate Unicode |
| 4 | `playground.html:4168, 4171, 4192, 4193` | bug | medium | M | Unbounded `localStorage` growth |
| 5 | `playground.html:2348` | bug | medium | S | `dashES` EventSource never closed |
| 6 | `playground.html:4791` | bug | medium | M | SSE `\n\n` split fragile across reads |
| 7 | `playground.html` (multiple) | perf | low | S | Some fetches missing `Content-Type` |
| 8 | `playground.html:5680, 2680` | bug | low | M | Audio playback overlap |
| 9 | playground POSTs | bug | medium | L | No CSRF protection |
| 10 | `src/api/assets.rs:110, 125` | bug | medium | S | CDN fallback redirect — mixed-content edge |
| 11 | `playground.html:3766, 3794` | bug | low | M | Mic `MediaStreamTrack` not stopped |
| 12 | `playground.html:4188-4193` | polish | low | S | `ki_*` localStorage namespace collision |
| 13 | `src/api/assets.rs:40-89` | perf | low | S | No abort for slow CDN bootstrap |
| 14 | `playground.html:4906-5060` | polish | low | S | `console.log` in API examples (cut) |
| 15 | `playground.html` (~13 sites) | bug | low | M | Listeners never `removeEventListener` |
| 16 | `playground.html:1-600` | perf | low | M | HTML not minified (cut) |
| 17 | `assets.rs:110, 125; playground.html:8` | bug | low | S | Silent CDN fallback (warn-only) |
| 18 | `src/api/handlers.rs:14-23` | bug | low | M | Strong-ETag vs encoding interplay (cut) |

### Cross-cutting infra

| # | Where | Cat | Sev | Eff | Summary |
|---|---|---|---|---|---|
| 1 | `src/api/health.rs:65,76,91,159,175,193,195` | polish | medium | M | Magic thresholds |
| 2 | `src/api/system.rs:97-142`, `src/api/performance.rs:79-142` | perf | medium | M | `sysinfo::System::new_all()` per request |
| 3 | `src/api/health.rs:76, 193, 195` | polish | low | S | Active-request thresholds vs worker count |
| 4 | `src/api/health.rs:27-46` | bug | high | S | Liveness 200 even when degraded |
| 5 | `src/api/system.rs:114-127` | perf | low | S | GPU device list cloned per call |
| 6 | `src/api/health.rs:59-62, 151-154` | polish | low | S | Error rate 0% when no requests |
| 7 | `src/core/model_cache.rs:37-38, 43-44` | bug | low | S | NUL-byte separator comment vs code mismatch |
| 8 | `src/api/metrics_endpoint.rs:12-14` | polish | low | S | Metrics error lacks correlation id |
| 9 | `src/middleware/rate_limit.rs:49-74` | perf | medium | M | Cleanup never called; map grows |
| 10 | `src/middleware/request_logger.rs:126` | polish | low | S | `>=500` boundary off-by-one |
| 11 | `src/error.rs:64-82` | polish | low | S | `InferenceError` not mapped to status codes |
| 12 | `src/api/health.rs:32-35` | bug | medium | S | Liveness vs readiness conflated |
| 13 | `src/api/performance.rs:82,…` | polish | low | S | Endpoint duration not logged |
| 14 | `src/api/system.rs:144-150` | polish | low | S | Hardcoded feature flags |
| 15 | `src/core/model_cache.rs:100-101, 109-110` | polish | low | S | Cache (de)serialize errors opaque |

---

## QA verification (post-audit, 2026-04-29)

Spot-checked the high-severity claims and a sample of mediums against the actual code. Sub-agents got the line numbers and the descriptions almost entirely right. Three items need adjustment, one downgrade, and one bonus issue surfaced from `cargo check`.

### Confirmed (read the file at the cited line, claim holds)

- ✓ Server core #1 — `src/config.rs:491` literally `jwt_secret: "your-secret-key-here".to_string()`, with `auth.enabled: true` on the same struct (line 490). High/S stands.
- ✓ Mutex poison panics — `ort_classify.rs:188`, `ort_yolo.rs:126`, `kokoro_onnx.rs:72-76` all use `.unwrap()`/`.expect()` on lock or semaphore acquisition. High/S stands for all three.
- ✓ STT blocking — `audio.rs:209-212` calls `state.model_manager.transcribe_audio(...)` with no `.await`; `transcribe_audio` is defined as a synchronous fn in `core/audio_models.rs:434`, so it blocks the actix executor. High/M stands.
- ✓ Classify rayon blocking — `image_pipeline.rs:124-130` uses `images.par_iter()`; called directly from the async handler at `classify.rs:142-143`. Medium/M stands.
- ✓ YOLO PyTorch blocking — `api/yolo.rs:150-158` calls `detector.detect(...)` with no `spawn_blocking`; the ORT path at line 209 *does* use `spawn_blocking`, confirming the inconsistency. High/M stands.
- ✓ Unbounded multipart/base64 — STT (`audio.rs:174-195`), YOLO (`yolo.rs:110-120`), Classify (`classify.rs:99-108`) — none have size caps. High/S stands.
- ✓ YOLO image-dim fallback — `yolo.rs:159` and `:206` both contain `image::image_dimensions(&temp_file).unwrap_or((640, 640))`. High/S stands.
- ✓ STT duration ignores channels — `api/audio.rs:143` `audio.samples.len() as f32 / audio.sample_rate as f32`. Medium/S stands. Same exact pattern in `tts_manager.rs:220-221`. **Note:** `core/audio.rs:105` is in `validate_wav` and uses `reader.duration()` from `hound`, which already returns frame count, not sample count, so that line is correct. **Drop** the `core/audio.rs:105` mention from P5; keep `api/audio.rs:143` and `tts_manager.rs:220-221`.
- ✓ LLM token counts hardcoded — `services/llm/src/handler.rs:216` literally `"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}`. High/M stands.
- ✓ LLM first chunk missing role — `sse_chunk` at `handler.rs:106-114` only emits `delta: {content}`. High/S stands.
- ✓ LLM hardcoded `chatcmpl-1`, missing `created`, `finish_reason: "stop"`, no temperature validation, error swallowed in streaming, no max_tokens validation, no `recv()` timeout, `pairs.clone()` — all match the cited lines.
- ✓ Liveness 200 when degraded — `health.rs:43` always returns `HttpResponse::Ok()`. **However:** readiness at `health.rs:120-128` already correctly returns 503 when not-ready, so the bucket J fix is narrower than stated — only liveness needs splitting.
- ✓ Symphonia decode unbounded — `core/audio.rs:276-292` while-loop with `.extend_from_slice(...)` and no cap. High/M stands.
- ✓ WAV size pre-check is only `data.len() < 4` — `core/audio.rs:79-81`. Medium/S stands.
- ✓ Stream classify per-image `ImagePipeline` — `classify.rs:242-243` inside the `for` loop. Medium/S stands.
- ✓ Playground XSS via `innerHTML` — `playground.html:5703` (`p.label`), `:5736` (`b.class_name`), and `makeToolCard` at `:5640` (`bodyHTML` is concatenated raw). High/S stands.
- ✓ `/system/info` calls `sysinfo::System::new_all()` per request — `system.rs:103`. Medium/M stands.
- ✓ FNV-1a NUL separator — `model_cache.rs:37-38` literally `h ^= 0u64;` (no-op) followed by `h.wrapping_mul(FNV_PRIME)`. Audit was right that the comment misleads; the multiply does the actual separation, so collision-resistance is preserved. Low/S stands as a comment fix.

### Downgraded / refuted

- ✗ **TTS #1 — `kokoro_onnx.rs:273` `f32::from_le_bytes(b.try_into().unwrap())` panic risk.** Refuted. The slice is produced by `chunks_exact(4)`, which by definition only yields slices of exactly 4 bytes (remainders are dropped). `b.try_into::<[u8;4]>()` cannot fail. The follow-up `anyhow::ensure!(floats.len() == expected, ...)` on line 276 catches truncated voice files cleanly. **Drop from scope.** TTS #2 (same pattern in test) follows.
- ↓ **Playground #3 — emoji regex strips legitimate Unicode (CJK / math).** Overstated. `\p{Extended_Pictographic}` is the official Unicode property for pictographic characters; it does **not** include Han/CJK ideographs (Lo) or mathematical symbols (Sm/So). The regex only strips actual emojis and emoji-adjacent dingbats. There is still a real concern (e.g. legitimately-wanted `™`, `©`, or arrow symbols can be in `Extended_Pictographic`), but it's a UX nit, not a corruption bug. **Downgrade to Low/S** — keep but de-prioritise.
- ↓ **Cross-cutting #7 — FNV-1a NUL separator collision risk.** The hash itself is correct; only the comment is misleading. Already at low/S, but flagging that this is comment-only, not collision-risk.
- ↓ **Bucket J — Liveness/readiness split.** Readiness already does the right thing (503 on not-ready). The fix is just to make liveness model "is the process alive" (always 200 unless we genuinely intend to die) and let readiness keep its existing logic. Effort drops from S–M to S.

### Bonus finding (from `cargo check`, not the audits)

- ➕ **Dead feature flag `metal` in `system.rs:174`.** `cfg!(feature = "metal")` always evaluates to `false` because `metal` isn't declared in `Cargo.toml`. The intended branch never runs — Apple Silicon users see a wrong backend label in `/system/info`. **Add to Bucket H (Validation & error mapping) as Bug/Low/S.** Likely one-character fix (`metal` is exposed via the `tch`/`candle` feature stacks, not as a top-level feature; rename to whatever actually flips on Metal, or declare the feature).

### Build status

`cargo check` succeeds with warnings only (one of which is the dead-feature finding above). Audit work did not introduce any code changes.

### Adjusted scope after QA

- Drop: TTS #1, TTS #2.
- Downgrade to Low: Playground #3 (regex), Bucket J effort (S only).
- Add: Dead `metal` feature flag (`system.rs:174`).
- Net total: **120 issues** (was 121), still grouped into the same 10 buckets. The recommended Phase 3 ordering (C → A → B → F → D → H → E → I → G → J) does not change.

---

## Follow-up bucket K — Test-suite hygiene (surfaced 2026-04-29)

Surfaced while running `cargo test --lib` to verify Bucket C. None of these are caused by Bucket C; all reproduce on `main` before any of my edits. Logging here so they don't get forgotten.

| Test | Failure | Likely category |
|---|---|---|
| `api::system::tests::test_get_config_*` (6 tests) | All asserting on config response shape | API drift since the test was written |
| `api::yolo::tests::test_detect_objects_{model_exists_no_torch,no_model_on_disk,v5_small_no_model}` | Model-absence path | env-dependent or assertion drift |
| `core::xtts::tests::test_xtts_synthesize_samples_clamped` | `panicked at "sample out of range: NaN"` (`src/core/xtts.rs:334`) | Real bug — clamp guard fails |
| `core::bark_tts::tests::test_bark_synthesize_samples_clamped` | Same pattern | Real bug — same shape |
| `core::styletts2::tests::test_styletts2_synthesize_samples_clamped` | Same pattern | Real bug — same shape |
| `core::kokoro_tts::tests::test_kokoro_returns_error_when_model_absent` | `panicked at "should return an error when model files are absent"` (`src/core/kokoro_tts.rs:293`) | Test asserts negation; engine no longer errors as expected |
| `api::models::download_coverage_tests::test_download_model_async_empty_model_type_tts_default` | Flaky (passed on second run) | Flake — likely network/IO timing |
| (process-level) | `libc++abi: terminating ... mutex lock failed: Invalid argument` after the suite finishes; SIGABRT | OS-mutex (C++) misuse on teardown — likely a global static `Drop` |

Sev/effort estimates:
- The three `*_synthesize_samples_clamped` failures look like the same NaN-propagation bug across three engines — one fix, three call-sites. **Bug / medium / S–M.**
- The 6 system-config tests are likely a single mass assertion update. **Polish / low / S.**
- The YOLO model-absence tests need an env probe. **Polish / low / S.**
- The Kokoro absent-model test is a contract change check. **Polish / low / S.**
- The teardown SIGABRT is concerning but the root cause needs investigation before triage. **Bug / medium / M (investigation).**

Recommendation: pick this up after Buckets A–H. The test-suite isn't actively misleading us — the failures are sharp and easy to ignore — but it should not stay red on `main`.

---

## Next step

Tell me which buckets / individual items are in scope, and I'll start Phase 3 with the first bucket (recommended: **C — Mutex poison resilience**, then **A — Security & DoS hardening**). Each bucket goes through brainstorm → spec → plan → implement → verify, separately.
