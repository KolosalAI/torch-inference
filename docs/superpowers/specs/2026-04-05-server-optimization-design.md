# Server Optimization Design

**Date:** 2026-04-05  
**Status:** Approved  
**Scope:** Full-stack optimization — latency, throughput, memory, feature gaps, benchmarks  
**Targets:** Apple Silicon (Metal), NVIDIA CUDA, CPU-only, mixed deployments

---

## Background

Phase A (lock-free state, priority queue batching) and Phase C (zero-copy cache, tensor pooling, request deduplication) have been merged. This spec covers the next three phases:

- **Phase D** — serialization & memory hot-paths (low risk, high certainty)
- **Phase E** — feature gap closure: TTS streaming, LLM backend, connection pooling
- **Phase F** — expanded benchmarks and flamegraph-guided final pass

Success criteria: measurable before/after benchmark data, feature gaps closed, no regressions.

---

## Phase D: Serialization & Memory Hot-Paths

### D1. simd-json on the Request Path

**File:** `src/api/handlers.rs` and all handler files that call `serde_json::from_slice` / `serde_json::from_str`

Replace `serde_json` deserialization on the inbound request path with `simd_json::from_slice` (AVX2 on x86_64, NEON on ARM/Apple Silicon). `simd-json` is a drop-in replacement for serde_json deserialization and delivers 2-4× parse throughput on typical JSON payloads.

- Add `simd-json = { version = "0.13", features = ["serde_impl"] }` to `Cargo.toml`
- Response serialization stays as-is (`serde_json::to_string`) since responses already go through `Arc<Bytes>` via `BytesCacheEntry`; for non-cached responses, route through a per-request reusable `Vec<u8>` buffer (pre-allocated at 4KB, grown as needed) instead of a fresh heap allocation per response

**Expected impact:** 30–40% reduction in JSON parse CPU at 10k+ RPS.

### D2. Segmented LRU (SLRU)

**File:** `src/cache.rs`

Replace the current adaptive-sampling LRU with a two-segment SLRU:

- **Probationary segment** — new entries land here (capacity: 20% of total cache size)
- **Protected segment** — entries promoted after a second access (capacity: 80%)

Eviction order: probationary tail first, then protected tail. Promotion: on cache hit in probationary, move to protected head. This eliminates the adaptive sampling scan entirely — all operations are O(1) pointer manipulations on doubly-linked lists backed by `DashMap`.

**Expected impact:** O(1) eviction (vs. current O(sample_size)), ~5–10% improvement in cache hit rate on repeated inference workloads.

### D3. Image Buffer Pooling

**File:** `src/tensor_pool.rs` (extend existing pool)

Add a `BufferPool` for `Vec<u8>` scratch buffers used in the image preprocessing pipeline (JPEG decode → resize → normalize). Strategy:

- Pool keyed by buffer capacity bucket (1KB, 4KB, 16KB, 64KB, 256KB, 1MB)
- On acquire: return exact-fit or next-larger bucket
- On release: return to pool if pool depth < 32; drop otherwise
- Pool is `Arc<Mutex<HashMap<usize, Vec<Vec<u8>>>>>` — contention is low since acquire/release is brief

**Expected impact:** Eliminate 1–3 heap allocations per image request; reduced GC pressure under sustained vision workloads.

### D4. HTTP & Actix Server Tuning

**File:** `src/main.rs` (HttpServer builder)

Apply the following to the `HttpServer` builder:

```rust
.keep_alive(Duration::from_secs(75))
.client_request_timeout(Duration::from_secs(5))
.client_disconnect_timeout(Duration::from_secs(1))
```

And on the TCP listener:
```rust
.tcp_nodelay(true)
.tcp_keepalive(Some(Duration::from_secs(30)))
```

Wire `server.workers` to `num_cpus::get()` at startup when the config value is `0` or absent, rather than defaulting to a hardcoded 16. Add `num_cpus = "1.16"` to dependencies.

**Expected impact:** Reduced connection setup overhead on sustained traffic; better CPU utilization on non-16-core machines.

---

## Phase E: Feature Gap Closure

### E1. TTS Mid-Synthesis Streaming

**New endpoint:** `POST /synthesize/stream`  
**Files:** `src/api/handlers/tts.rs`, `src/tts/pipeline.rs`

The TTS pipeline already processes text sentence-by-sentence. Replace the current "accumulate all → return" pattern with a streaming response:

- Handler creates a `tokio::sync::mpsc::channel::<Bytes>(8)` (bounded at 8 — applies backpressure if client reads slowly)
- TTS pipeline sends each sentence's encoded audio (WAV or Ogg/Opus per `output_format`) as a `Bytes` chunk to the sender
- Handler returns `HttpResponse::Ok().content_type("audio/wav").streaming(ReceiverStream::new(rx))`
- Existing `/synthesize` endpoint is unchanged (still accumulates for clients that need a complete file)

**TTFA target:** First audio chunk delivered within 80–150ms (first sentence synthesis time) rather than full synthesis time (500ms–2s+).

Error handling: if synthesis fails mid-stream, send a final empty chunk and close the channel; the client detects end-of-stream.

### E2. LLM Candle Backend

**Files:** `src/llm/backend.rs` (new), `src/llm/mod.rs` (wire up)

Implement the `InferenceBackend` trait for Candle. Scope:

- **Decoding:** greedy (argmax) and basic beam search (beam width 1–8, configurable)
- **Device abstraction:** `candle_core::Device` — CPU, Metal (macOS), CUDA — selected at startup via existing `auto_detect_backends()` logic
- **KV-cache:** per-request KV-cache stored in `InflightRequest` metadata; reused across multi-turn completions by passing the cache handle through the priority queue
- **Model loading:** hook into the existing `ModelManager` registry; LLM models loaded on demand, unloaded on eviction
- **Token streaming:** emit tokens via `mpsc::Sender<String>` for streaming completions (mirrors TTS streaming pattern)

Out of scope for this phase: speculative decoding, LoRA adapters, quantization beyond what Candle provides natively (GGUF f16/q4 via `candle-transformers`).

**Supported model families:** LLaMA 2/3, Mistral, Phi-2/3 (via `candle-transformers` model zoo).

### E3. Model Download Connection Pool

**File:** `src/model_manager.rs` (or wherever download logic lives)

Replace per-request `reqwest::Client::new()` with a shared `reqwest::Client` stored in `AppState`:

```rust
reqwest::Client::builder()
    .pool_max_idle_per_host(5)
    .tcp_keepalive(Duration::from_secs(30))
    .timeout(Duration::from_secs(300))  // large models
    .build()
```

Pass the shared client into the download handler. This prevents repeated TCP handshakes to the same model registry host (HuggingFace, custom registries).

---

## Phase F: Benchmarks & Flamegraph-Guided Final Pass

### F1. Expanded Benchmark Suite

Add to `benches/`:

**`latency_bench.rs`**
- p50/p95/p99 latency histograms for `/predict`, `/synthesize`, `/detect`
- Concurrency levels: 1, 8, 32 concurrent clients
- Uses Criterion's async bencher with `tokio::runtime`

**`tts_streaming_bench.rs`**
- Measures TTFA (time to first audio chunk) vs. full-synthesis latency
- Text lengths: short (10 words), medium (50 words), long (200 words)
- Compares `/synthesize` vs. `/synthesize/stream`

**`llm_bench.rs`**
- Tokens/sec throughput: greedy vs. beam search (width 4)
- Batch sizes: 1, 4, 8
- Devices: CPU, Metal (where available), CUDA (where available)

**`memory_bench.rs`**
- Peak RSS and allocation rate under sustained load
- Uses `jemalloc_ctl` stats API (`epoch`, `stats.allocated`, `stats.resident`)
- Measures: baseline idle, single-request spike, sustained 100 RPS for 60s

### F2. Flamegraph-Guided Final Pass

After Phase D+E land:

1. Run `make flamegraph` under mixed-modality load: concurrent TTS + image detection + LLM completion
2. Identify any new hot spots consuming >5% of CPU
3. Apply targeted fixes
4. Re-run flamegraph to confirm improvement

The flamegraph infrastructure is already in place (`pprof` feature flag, shutdown handler writes `flamegraph.svg`).

### F3. Before/After Baseline Report

- Run full benchmark suite **before Phase D** begins; commit results to `benches/baseline/`
- Run full benchmark suite **after Phase F** completes; append results to `docs/performance_optimization_implementation.md`

Report sections: throughput delta (req/s), p99 latency delta, TTFA delta (TTS), peak RSS delta, tokens/sec (LLM).

---

## Architecture Impact Summary

| Component | Change | Phase |
|-----------|--------|-------|
| `src/api/handlers.rs` | simd-json deserialization | D |
| `src/cache.rs` | SLRU replacement | D |
| `src/tensor_pool.rs` | Image buffer pooling | D |
| `src/main.rs` | HTTP/TCP tuning, dynamic worker count | D |
| `src/api/handlers/tts.rs` | Streaming endpoint | E |
| `src/tts/pipeline.rs` | mpsc sender per sentence | E |
| `src/llm/backend.rs` | Candle backend (new file) | E |
| `src/llm/mod.rs` | Wire up backend | E |
| `src/model_manager.rs` | Shared reqwest client | E |
| `benches/` | 4 new benchmark files | F |
| `docs/performance_optimization_implementation.md` | Append results | F |

## Error Handling

- SLRU: on capacity overflow, evict probationary tail atomically; if probationary is empty, evict protected tail
- TTS streaming: mid-stream error closes the channel; client receives a partial audio stream and an HTTP trailer error (or connection close)
- LLM backend: if Candle device unavailable (e.g., CUDA not found), fall back to CPU automatically via existing `auto_detect_backends()` logic
- Buffer pool: on borrow failure (pool empty), allocate fresh — never block

## Testing

- SLRU: unit tests for promotion, eviction order, capacity boundaries
- TTS streaming: integration test asserting first chunk arrives before full synthesis completes
- LLM backend: unit test with a small model (TinyLlama or equivalent) checking greedy output is deterministic; integration test for multi-turn KV-cache reuse
- Buffer pool: property test (proptest) verifying no buffer is returned to pool while still borrowed
- HTTP tuning: existing integration tests cover correctness; no new tests needed
