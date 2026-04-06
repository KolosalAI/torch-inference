# Logging Improvement Design
**Date:** 2026-03-30
**Status:** Approved
**Scope:** `src/main.rs`, `src/telemetry/`, `src/middleware/request_logger.rs`, `src/core/engine.rs`, `src/models/manager.rs`

---

## Goals

- Replace all `println!` and `log::info!`/`log::warn!` calls with `tracing::info!`/`tracing::warn!` so every event goes through the structured logging pipeline
- Wrap each startup phase in a tracing span to capture live timing per phase
- Extend the request logger to emit full live request state (query string, user-agent, content length, response size, status class, slow-request warnings)
- Add structured inference and model lifecycle events so every inference call is visible with its current state

## Non-Goals

- No per-stage inference breakdown (tokenize/forward/decode) — that is Option C scope
- No log aggregation platform integration changes
- No changes to `structured_logging.rs` initialization logic
- No new dependencies

---

## Section 1: Logging Unification

### Problem
`main.rs` uses three different logging mechanisms simultaneously:
- `println!` — bypasses the tracing pipeline entirely; invisible to log aggregators
- `log::info!` / `log::warn!` — goes through the `log` facade (bridged to tracing, but inconsistent)
- `tracing::info!` — correct path

Additionally, `actix_middleware::Logger` runs alongside the custom `RequestLogger`, producing two log lines per request.

### Changes
1. Remove all `println!` from `main.rs` and `log_system_info()`. Replace each with a `tracing::info!` call with equivalent structured fields.
2. Replace all `log::info!`, `log::warn!`, `log::debug!` in `main.rs` with `tracing::info!`, `tracing::warn!`, `tracing::debug!`.
3. Remove the `.wrap(actix_middleware::Logger::new(...))` line from the HTTP server configuration. The custom `RequestLogger` already covers this.
4. The startup banner (border lines, server URL, health URL) becomes structured tracing events:
   ```rust
   tracing::info!(server_url = %format!("http://{}", display_addr), health_url = %format!("http://{}/health", display_addr), "server ready");
   ```

---

## Section 2: Startup Phase Spans

### Problem
The ~15 startup phases produce single log lines with no timing. A slow GPU init or TTS engine load is invisible — you can't distinguish a 5ms config load from a 5000ms ORT library scan.

### Design
Wrap each startup phase in a `tracing::info_span!` and enter it before the phase work begins. The subscriber's `FmtSpan::CLOSE` setting (already configured in `structured_logging.rs`) emits an automatic close event with `time.busy` and `time.idle` when the span guard drops.

### Phases and Fields

| Span name | Key fields | Location |
|---|---|---|
| `config_load` | `device_type`, `host`, `port` | `main()` after `Config::load()` |
| `device_detect` | `backend`, `device_count`, `device_ids` | `main()` after GPU detection block |
| `system_info` | `cpu_cores`, `ram_total_gb`, `ram_avail_gb` | `log_system_info()` |
| `components_init` | `cache_size_mb`, `workers_min`, `workers_max` | after component init block |
| `gpu_init` | `backend`, `gpu_count`, `device_names` | after `GpuManager::new()` block |
| `download_init` | `cache_dir` | after `ModelDownloadManager` init |
| `audio_init` | `model_dir` | after `AudioModelManager` init |
| `tts_init` | `engine_count`, `engine_ids` | after `TTSManager` init |
| `model_preload` | `model_name`, `status` | inside preload loop |
| `warmup` | `status`, `error` | inside warmup `tokio::spawn` |
| `server_bind` | `addr`, `workers` | before `HttpServer::new` |

### Example Output (plain mode)
```
INFO config_load{device_type="cuda" host="0.0.0.0" port=8080}: close time.busy=1ms
INFO gpu_init{backend="cuda" gpu_count=2 device_names="RTX 4090,RTX 3080"}: close time.busy=38ms
INFO tts_init{engine_count=1 engine_ids="kokoro-onnx"}: close time.busy=210ms
INFO server_bind{addr="0.0.0.0:8080" workers=4}: close time.busy=2ms
```

---

## Section 3: Request Logger — Live Request State

### Problem
`RequestLoggerMiddleware` emits `request_received` and `request_completed` but is missing:
- Query string (what parameters were sent)
- User-Agent (which client/SDK is calling)
- Content-Length (how large was the request body)
- Response size (how large was the response)
- Status class (`2xx`/`4xx`/`5xx`) for easy log filtering
- Slow request detection

### Changes to `src/middleware/request_logger.rs`

**`request_received` — add fields:**
```rust
query        = %req.query_string(),          // "" if none
user_agent   = %user_agent,                  // from User-Agent header, "unknown" if absent
content_len  = %content_length,              // from Content-Length header, 0 if absent
```

**`request_completed` — add fields:**
```rust
status_class    = %status_class(status),     // "2xx", "4xx", "5xx"
response_bytes  = %response_bytes,           // from Content-Length response header; 0 if absent or streaming
```
Note: actix `ServiceResponse<B>` body cannot be read without consuming it. Use `res.headers().get("content-length")` parsed as `u64`, defaulting to `0` for streaming or unknown responses.

**New: slow request warning** — emitted after `request_completed` if `duration_ms >= 500`:
```rust
tracing::warn!(
    correlation_id = %metrics.correlation_id.as_str(),
    method         = %method,
    path           = %path,
    duration_ms    = %metrics.duration_ms(),
    threshold_ms   = 500,
    event          = "slow_request",
);
```

**`request_error` — add fields:**
```rust
status_class = "5xx",
```

### Helper function
```rust
fn status_class(status: u16) -> &'static str {
    match status {
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _         => "other",
    }
}
```

---

## Section 4: Inference & Model Lifecycle Logging

### Problem
No visibility into inference operations. A request to `/api/v1/inference` produces only the HTTP-level request_received/request_completed lines — nothing about which model ran, with what batch size, or how long the forward pass took.

### Changes to `src/core/engine.rs`

**`InferenceEngine::infer()`** — wrap in a span:
```rust
let span = tracing::info_span!(
    "inference",
    model        = %model_name,
    correlation_id = %correlation_id,
    batch_size   = %batch_size,
);
let _guard = span.enter();

tracing::info!(input_shape = %input_shape, "inference start");
// ... inference work ...
tracing::info!(
    elapsed_ms   = %elapsed.as_millis(),
    output_shape = %output_shape,
    "inference complete"
);
if elapsed.as_millis() > 500 {
    tracing::warn!(elapsed_ms = %elapsed.as_millis(), threshold_ms = 500, "slow inference");
}
```

### Changes to `src/models/manager.rs`

**`ModelManager::load_model()`**:
```rust
tracing::info!(model = %name, backend = %backend, "model load start");
// ... load work ...
tracing::info!(model = %name, backend = %backend, elapsed_ms = %elapsed, device = %device, "model load complete");
```

**`ModelManager::unload_model()`** (or LRU eviction path):
```rust
tracing::info!(model = %name, reason = %reason, "model unload");
```

**`InferenceEngine::warmup()`**:
```rust
tracing::info!(model = %name, "warmup start");
// on success:
tracing::info!(model = %name, elapsed_ms = %elapsed, status = "ok", "warmup complete");
// on failure:
tracing::warn!(model = %name, error = %e, status = "failed", "warmup failed");
```

---

## Files Changed

| File | Change type |
|---|---|
| `src/main.rs` | Replace all `println!`, `log::`, unify to `tracing::`, add phase spans, remove actix Logger |
| `src/telemetry/structured_logging.rs` | No change needed |
| `src/middleware/request_logger.rs` | Add query, user_agent, content_len, response_bytes, status_class, slow_request warn |
| `src/core/engine.rs` | Add inference span, start/complete events, slow inference warn |
| `src/models/manager.rs` | Add model load/unload/warmup structured events |

---

## Acceptance Criteria

- No `println!` calls remain in `main.rs` or `log_system_info()`
- No `log::info!` / `log::warn!` calls remain in `main.rs`
- `actix_middleware::Logger` is removed from the server config
- Each startup phase span closes with `time.busy` visible in logs
- Every request log includes `query`, `user_agent`, `content_len`, `response_bytes`, `status_class`
- Requests over 500ms emit a `slow_request` warn event
- Every inference call emits `inference start` and `inference complete` with structured fields
- Inference over 500ms emits a `slow_inference` warn event
- Model load/unload emits structured events with timing
- All existing tests pass (`cargo test`)
