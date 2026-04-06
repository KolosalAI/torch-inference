# Post-Processing & Response Envelope — Design Spec
**Date:** 2026-04-03  
**Scope:** Audio, Classification, YOLO — excludes LLM  
**Status:** Approved

---

## 1. Goal

Add a consistent post-processing layer that improves signal quality per model type and wraps every response in a uniform metadata envelope. Post-processing is **opt-in, defaults on** — callers pass `"skip_postprocess": true` to bypass.

---

## 2. Module Structure

New module: `src/postprocess/`

```
src/postprocess/
├── mod.rs          — re-exports, shared PostprocessConfig, PostprocessResult<T>
├── audio.rs        — DC offset → peak normalize → silence trim
├── classify.rs     — temperature scale → round → min_confidence filter
├── yolo.rs         — bbox_rel, area, label_confidence bucket, confidence rounding
└── envelope.rs     — Envelope<T> and ResponseMeta types
```

Registered in `src/lib.rs` as `pub mod postprocess;`.

---

## 3. Data Flow

```
Handler receives request
  → runs inference (unchanged)
  → calls postprocess::<domain>::process(raw_output, &config.postprocess.<domain>)
  → receives (processed_output, Vec<String> steps, Vec<String> warnings)
  → calls Envelope::new(processed_output, meta)
  → HttpResponse::Ok().json(envelope)
```

---

## 4. Configuration

New `[postprocess]` section added to `Config` in `src/config.rs`:

```toml
[postprocess.audio]
enabled = true
target_peak = 0.95
silence_threshold = 0.01
pad_ms = 50

[postprocess.classify]
enabled = true
temperature = 1.0
min_confidence = 0.01

[postprocess.yolo]
enabled = true
high_confidence_threshold = 0.7
medium_confidence_threshold = 0.4
```

`PostprocessConfig` derives `Deserialize`, `Serialize`, `Clone`, `Debug`. Each sub-config implements `Default` matching the values above.

Per-request opt-out: any request body may include `"skip_postprocess": true`; this field is added to the relevant request structs and bypasses the postprocess call (envelope still applied, `postprocessing_applied: false`).

---

## 5. Audio Post-Processing (`postprocess/audio.rs`)

**Input:** `Vec<f32>` samples, `u32` sample_rate, `AudioPostprocessConfig`  
**Output:** `(Vec<f32>, Vec<String> steps, Vec<String> warnings)`

### Stage 1 — DC Offset Removal
```
mean = samples.sum() / samples.len()
samples = samples.map(|s| s - mean)
```
- Warning emitted: `"dc_offset_removed:{mean:.4}"` when `|mean| > 0.001`
- Step recorded: `"dc_offset"`

### Stage 2 — Peak Normalization
```
peak = samples.map(|s| s.abs()).max()
if peak > 0.0: samples = samples.map(|s| s / peak * target_peak)
```
- Warning emitted: `"clipping_detected"` when any `|sample| > 1.0` before normalization
- Step recorded: `"normalize:{target_peak}"`
- Default `target_peak = 0.95` (~−0.44 dBFS)

### Stage 3 — Silence Trimming
Scan from start and end, drop samples where `|sample| < silence_threshold`. Preserve `pad_ms` milliseconds of silence at each boundary.
- Step recorded: `"trim:pad={pad_ms}ms"`
- Warning emitted: `"all_silence"` if entire buffer is below threshold (no trim applied)

**Wiring:**
- `api/tts.rs::synthesize` — call after backend synthesis, before `AudioProcessor::save_wav()`
- `api/audio.rs::synthesize_speech` — same insertion point

---

## 6. Classification Post-Processing (`postprocess/classify.rs`)

**Input:** `Vec<Prediction>`, `ClassifyPostprocessConfig`  
**Output:** `(Vec<Prediction>, Vec<String> steps, Vec<String> warnings)`

`Prediction` fields: `label: String`, `confidence: f32`, `class_id: usize`

### Stage 1 — Temperature Scaling
```
logit_i = ln(confidence_i + 1e-10) / temperature
scaled = softmax(logits)
```
- Only applied when `temperature != 1.0`
- Step recorded: `"temperature_scaled:{temperature}"`

### Stage 2 — Round + Filter
```
confidence = round(confidence, 4 decimal places)
keep if confidence >= min_confidence
```
- Warning emitted: `"predictions_filtered:{n}"` when n > 0 dropped
- Step recorded: `"rounded"`, `"filtered:min={min_confidence}"`

**Wiring:**
- `api/classify.rs::classify_batch` — call after NoOpClassificationBackend returns predictions
- `api/classification.rs` handlers — call after `ImageClassifier::classify()` returns `TopKResults`

---

## 7. YOLO Post-Processing (`postprocess/yolo.rs`)

**Input:** `YoloResults` (contains `Vec<Detection>`), image `(width: u32, height: u32)`, `YoloPostprocessConfig`  
**Output:** `(EnrichedYoloResults, Vec<String> steps, Vec<String> warnings)`

`Detection` gains three new fields:
```rust
pub bbox_rel: BboxRel,        // normalized [0,1] coordinates
pub area: f32,                // pixel area
pub confidence_bucket: String, // "high" | "medium" | "low"
```

### Stage 1 — Relative Bounding Box
```
bbox_rel = { x1/w, y1/h, x2/w, y2/h }.clamp(0.0, 1.0)
```
- Step recorded: `"bbox_rel"`

### Stage 2 — Area + Confidence Rounding
```
area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
confidence = round(confidence, 4dp)
```
- Step recorded: `"bbox_enriched"`

### Stage 3 — Confidence Bucket
```
"high"   if confidence >= high_threshold  (default 0.7)
"medium" if confidence >= medium_threshold (default 0.4)
"low"    otherwise
```
- Step recorded: `"confidence_bucketed"`

**Wiring:**
- `api/yolo.rs::detect_objects` — image dimensions captured during multipart parsing; pass to `postprocess::yolo::process()` before building `YoloDetectResponse`

---

## 8. Response Envelope (`postprocess/envelope.rs`)

```rust
pub struct Envelope<T: Serialize> {
    pub data: T,
    pub meta: ResponseMeta,
}

pub struct ResponseMeta {
    pub latency_ms: f64,
    pub model_id: String,
    pub postprocessing_applied: bool,
    pub postprocess_steps: Vec<String>,
    pub warnings: Vec<String>,
    pub version: &'static str,   // env!("CARGO_PKG_VERSION")
    pub request_id: String,      // echoed from X-Correlation-ID extension
}
```

`Envelope<T>` implements `Serialize`. `ResponseMeta` derives `Serialize`, `Debug`.

**Helper:** `Envelope::new(data, meta)` — constructor.  
**Latency:** Handlers capture `Instant::now()` at the start; `elapsed().as_secs_f64() * 1000.0` before building envelope.  
**Request ID:** Read from `req.extensions().get::<CorrelationId>()` — already set by `CorrelationIdMiddleware`.

**Applied to:** `api/tts.rs`, `api/audio.rs`, `api/classify.rs`, `api/classification.rs`, `api/yolo.rs`  
**Not applied to:** `api/llm.rs` (excluded by request), `api/image.rs` (different ownership model), `api/inference.rs` (generic JSON)

---

## 9. Error Handling

Post-processing failures must never fail the request. Each `process()` function returns `Result<..., PostprocessError>`. On `Err`, the handler logs a warning, returns the raw unprocessed output, and sets `postprocessing_applied: false` with warning `"postprocess_failed:{reason}"`.

---

## 10. Testing

Each sub-module gets a `#[cfg(test)]` block covering:
- **audio:** DC offset removed correctly; clipping warning triggered; silence trimmed within `pad_ms`; all-silence no-crash
- **classify:** Temperature scaling changes distribution; min_confidence filter drops correct predictions; round-trip precision
- **yolo:** `bbox_rel` values clamped to [0,1]; area calculation; bucket thresholds; confidence rounding
- **envelope:** Serializes correctly; `request_id` propagated; `postprocessing_applied` false when skipped
