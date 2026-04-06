# Post-Processing & Response Envelope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `src/postprocess/` module that improves signal quality for Audio, Classification, and YOLO outputs and wraps all five endpoint families in a uniform `Envelope<T>` response.

**Architecture:** A new `src/postprocess/` module with five files (`mod.rs`, `audio.rs`, `classify.rs`, `yolo.rs`, `envelope.rs`). Each domain module exposes a pure `process()` function that takes raw inference output and returns enriched output plus `steps: Vec<String>` and `warnings: Vec<String>`. Handlers call `process()`, then wrap the result with `Envelope::new()` before returning.

**Tech Stack:** Rust, actix-web 4, serde/serde_json, existing `crate::config::Config`, `crate::middleware::correlation_id::get_correlation_id`, `crate::core::audio::AudioData`, `crate::core::yolo::{Detection, BoundingBox, YoloResults}`, `crate::api::classify::Prediction`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/postprocess/mod.rs` | Module re-exports, registers sub-modules |
| Create | `src/postprocess/envelope.rs` | `Envelope<T>`, `ResponseMeta` types |
| Create | `src/postprocess/audio.rs` | DC offset → normalize → trim pipeline |
| Create | `src/postprocess/classify.rs` | Temperature scale → round → filter pipeline |
| Create | `src/postprocess/yolo.rs` | bbox_rel, area, bucket enrichment |
| Modify | `src/lib.rs` | Add `pub mod postprocess;` |
| Modify | `src/config.rs` | Add `PostprocessConfig` and sub-configs |
| Modify | `src/api/tts.rs` | Add `skip_postprocess`, wire audio + envelope |
| Modify | `src/api/audio.rs` | Add `skip_postprocess`, wire audio + envelope |
| Modify | `src/api/classify.rs` | Add `skip_postprocess`, wire classify + envelope |
| Modify | `src/api/yolo.rs` | Add `skip_postprocess`, wire yolo + envelope |

---

## Task 1: Add PostprocessConfig to config.rs and register module in lib.rs

**Files:**
- Modify: `src/config.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Add the config structs to `src/config.rs`**

Open `src/config.rs`. Append these structs before the `impl Config` block (find it by searching for `pub fn load`). Add after the last existing `#[derive]` struct:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPostprocessConfig {
    #[serde(default = "AudioPostprocessConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default = "AudioPostprocessConfig::default_target_peak")]
    pub target_peak: f32,
    #[serde(default = "AudioPostprocessConfig::default_silence_threshold")]
    pub silence_threshold: f32,
    #[serde(default = "AudioPostprocessConfig::default_pad_ms")]
    pub pad_ms: u32,
}

impl AudioPostprocessConfig {
    fn default_enabled() -> bool { true }
    fn default_target_peak() -> f32 { 0.95 }
    fn default_silence_threshold() -> f32 { 0.01 }
    fn default_pad_ms() -> u32 { 50 }
}

impl Default for AudioPostprocessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_peak: 0.95,
            silence_threshold: 0.01,
            pad_ms: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyPostprocessConfig {
    #[serde(default = "ClassifyPostprocessConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default = "ClassifyPostprocessConfig::default_temperature")]
    pub temperature: f32,
    #[serde(default = "ClassifyPostprocessConfig::default_min_confidence")]
    pub min_confidence: f32,
}

impl ClassifyPostprocessConfig {
    fn default_enabled() -> bool { true }
    fn default_temperature() -> f32 { 1.0 }
    fn default_min_confidence() -> f32 { 0.01 }
}

impl Default for ClassifyPostprocessConfig {
    fn default() -> Self {
        Self { enabled: true, temperature: 1.0, min_confidence: 0.01 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloPostprocessConfig {
    #[serde(default = "YoloPostprocessConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default = "YoloPostprocessConfig::default_high")]
    pub high_confidence_threshold: f32,
    #[serde(default = "YoloPostprocessConfig::default_medium")]
    pub medium_confidence_threshold: f32,
}

impl YoloPostprocessConfig {
    fn default_enabled() -> bool { true }
    fn default_high() -> f32 { 0.7 }
    fn default_medium() -> f32 { 0.4 }
}

impl Default for YoloPostprocessConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            high_confidence_threshold: 0.7,
            medium_confidence_threshold: 0.4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessConfig {
    #[serde(default)]
    pub audio: AudioPostprocessConfig,
    #[serde(default)]
    pub classify: ClassifyPostprocessConfig,
    #[serde(default)]
    pub yolo: YoloPostprocessConfig,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            audio: AudioPostprocessConfig::default(),
            classify: ClassifyPostprocessConfig::default(),
            yolo: YoloPostprocessConfig::default(),
        }
    }
}
```

- [ ] **Step 2: Add `postprocess` field to the main `Config` struct**

In `src/config.rs`, find the `pub struct Config` block and add the new field:

```rust
pub struct Config {
    // ... existing fields ...
    #[serde(default)]
    pub sanitizer: SanitizerConfig,
    #[serde(default)]                   // ← add this
    pub postprocess: PostprocessConfig, // ← add this
}
```

- [ ] **Step 3: Register the module in `src/lib.rs`**

Open `src/lib.rs`. Add after the last `pub mod` line:

```rust
pub mod postprocess;
```

- [ ] **Step 4: Check it compiles**

```bash
cargo check 2>&1 | grep -E "^error"
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs src/lib.rs
git commit -m "feat(postprocess): add PostprocessConfig to config and register module"
```

---

## Task 2: Create the envelope module

**Files:**
- Create: `src/postprocess/mod.rs`
- Create: `src/postprocess/envelope.rs`

- [ ] **Step 1: Create `src/postprocess/mod.rs`**

```rust
pub mod audio;
pub mod classify;
pub mod envelope;
pub mod yolo;

pub use envelope::{Envelope, ResponseMeta};
```

- [ ] **Step 2: Write the failing test for Envelope in `src/postprocess/envelope.rs`**

Create `src/postprocess/envelope.rs` with tests first:

```rust
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Envelope<T: Serialize> {
    pub data: T,
    pub meta: ResponseMeta,
}

#[derive(Debug, Serialize)]
pub struct ResponseMeta {
    pub latency_ms: f64,
    pub model_id: String,
    pub postprocessing_applied: bool,
    pub postprocess_steps: Vec<String>,
    pub warnings: Vec<String>,
    pub version: &'static str,
    pub request_id: String,
}

impl<T: Serialize> Envelope<T> {
    pub fn new(data: T, meta: ResponseMeta) -> Self {
        Self { data, meta }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_serializes_data_and_meta() {
        #[derive(Serialize)]
        struct Payload { value: u32 }

        let meta = ResponseMeta {
            latency_ms: 42.0,
            model_id: "test-model".into(),
            postprocessing_applied: true,
            postprocess_steps: vec!["normalize".into()],
            warnings: vec![],
            version: "1.0.0",
            request_id: "req-123".into(),
        };
        let envelope = Envelope::new(Payload { value: 7 }, meta);
        let json = serde_json::to_string(&envelope).unwrap();
        assert!(json.contains("\"value\":7"));
        assert!(json.contains("\"latency_ms\":42.0"));
        assert!(json.contains("\"model_id\":\"test-model\""));
        assert!(json.contains("\"postprocessing_applied\":true"));
        assert!(json.contains("\"normalize\""));
    }

    #[test]
    fn test_envelope_postprocessing_false_when_steps_empty() {
        #[derive(Serialize)]
        struct Payload { ok: bool }

        let meta = ResponseMeta {
            latency_ms: 1.0,
            model_id: "m".into(),
            postprocessing_applied: false,
            postprocess_steps: vec![],
            warnings: vec![],
            version: "1.0.0",
            request_id: "r".into(),
        };
        let env = Envelope::new(Payload { ok: true }, meta);
        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains("\"postprocessing_applied\":false"));
    }

    #[test]
    fn test_envelope_warnings_propagated() {
        #[derive(Serialize)]
        struct Payload {}

        let meta = ResponseMeta {
            latency_ms: 0.0,
            model_id: "m".into(),
            postprocessing_applied: true,
            postprocess_steps: vec![],
            warnings: vec!["clipping_detected".into()],
            version: "1.0.0",
            request_id: "r".into(),
        };
        let env = Envelope::new(Payload {}, meta);
        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains("clipping_detected"));
    }
}
```

- [ ] **Step 3: Run the tests**

```bash
cargo test postprocess::envelope --lib 2>&1 | tail -10
```

Expected: all 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/postprocess/mod.rs src/postprocess/envelope.rs
git commit -m "feat(postprocess): add Envelope<T> and ResponseMeta"
```

---

## Task 3: Audio post-processor

**Files:**
- Create: `src/postprocess/audio.rs`

- [ ] **Step 1: Write the failing tests**

Create `src/postprocess/audio.rs`:

```rust
use crate::config::AudioPostprocessConfig;

pub struct AudioPostprocessResult {
    pub samples: Vec<f32>,
    pub steps: Vec<String>,
    pub warnings: Vec<String>,
}

/// DC offset removal → peak normalization → silence trim.
/// Returns the processed samples plus audit steps and warnings.
/// Never panics; if all samples are silent, returns them unchanged with a warning.
pub fn process(
    samples: Vec<f32>,
    sample_rate: u32,
    config: &AudioPostprocessConfig,
) -> AudioPostprocessResult {
    if !config.enabled || samples.is_empty() {
        return AudioPostprocessResult { samples, steps: vec![], warnings: vec![] };
    }

    let mut samples = samples;
    let mut steps = Vec::new();
    let mut warnings = Vec::new();

    // Stage 1 — DC offset removal
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    if mean.abs() > 0.001 {
        for s in &mut samples { *s -= mean; }
        warnings.push(format!("dc_offset_removed:{mean:.4}"));
    }
    steps.push("dc_offset".to_string());

    // Stage 2 — Peak normalization
    let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    let had_clipping = samples.iter().any(|s| s.abs() > 1.0);
    if had_clipping {
        warnings.push("clipping_detected".to_string());
    }
    if peak > 0.0 {
        let scale = config.target_peak / peak;
        for s in &mut samples { *s *= scale; }
    }
    steps.push(format!("normalize:{}", config.target_peak));

    // Stage 3 — Silence trimming
    let threshold = config.silence_threshold;
    let pad_samples = (config.pad_ms as f32 / 1000.0 * sample_rate as f32) as usize;

    let first_loud = samples.iter().position(|s| s.abs() >= threshold);
    let last_loud = samples.iter().rposition(|s| s.abs() >= threshold);

    match (first_loud, last_loud) {
        (Some(start), Some(end)) => {
            let trim_start = start.saturating_sub(pad_samples);
            let trim_end = (end + 1 + pad_samples).min(samples.len());
            samples = samples[trim_start..trim_end].to_vec();
            steps.push(format!("trim:pad={}ms", config.pad_ms));
        }
        _ => {
            // All silence
            warnings.push("all_silence".to_string());
        }
    }

    AudioPostprocessResult { samples, steps, warnings }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AudioPostprocessConfig;

    fn cfg() -> AudioPostprocessConfig { AudioPostprocessConfig::default() }

    #[test]
    fn test_dc_offset_removed() {
        let samples = vec![0.5f32; 100];
        let result = process(samples, 22050, &cfg());
        let mean: f32 = result.samples.iter().sum::<f32>() / result.samples.len() as f32;
        assert!(mean.abs() < 1e-4, "DC offset not removed, mean={mean}");
        assert!(result.warnings.iter().any(|w| w.starts_with("dc_offset_removed")));
        assert!(result.steps.contains(&"dc_offset".to_string()));
    }

    #[test]
    fn test_peak_normalized_to_target() {
        let mut c = cfg();
        c.silence_threshold = 0.0; // disable trim
        c.pad_ms = 0;
        let samples = vec![0.5f32, -0.5f32, 0.3f32, -0.3f32];
        let result = process(samples, 22050, &c);
        let peak = result.samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        assert!((peak - 0.95).abs() < 1e-5, "peak={peak}");
    }

    #[test]
    fn test_clipping_warning_emitted() {
        let samples = vec![1.5f32, -0.5f32];
        let result = process(samples, 22050, &cfg());
        assert!(result.warnings.contains(&"clipping_detected".to_string()));
    }

    #[test]
    fn test_silence_trimmed() {
        let mut c = cfg();
        c.pad_ms = 0;
        c.silence_threshold = 0.01;
        // 10 silent, 10 loud, 10 silent
        let mut samples = vec![0.001f32; 10];
        samples.extend(vec![0.5f32; 10]);
        samples.extend(vec![0.001f32; 10]);
        let result = process(samples, 22050, &c);
        assert!(result.samples.len() < 30, "len={}", result.samples.len());
        assert!(result.steps.iter().any(|s| s.starts_with("trim")));
    }

    #[test]
    fn test_all_silence_returns_warning_no_crash() {
        let samples = vec![0.0f32; 100];
        let result = process(samples, 22050, &cfg());
        assert!(result.warnings.contains(&"all_silence".to_string()));
    }

    #[test]
    fn test_disabled_returns_input_unchanged() {
        let mut c = cfg();
        c.enabled = false;
        let samples = vec![2.0f32, -2.0f32]; // would be clipped/normalized if enabled
        let result = process(samples.clone(), 22050, &c);
        assert_eq!(result.samples, samples);
        assert!(result.steps.is_empty());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_empty_input_returns_empty_no_crash() {
        let result = process(vec![], 22050, &cfg());
        assert!(result.samples.is_empty());
    }

    #[test]
    fn test_pad_ms_preserves_silence_at_edges() {
        let mut c = cfg();
        c.pad_ms = 10; // 10ms at 8000 Hz = 80 samples
        c.silence_threshold = 0.01;
        // 100 silent, 10 loud, 100 silent
        let mut samples = vec![0.001f32; 100];
        samples.extend(vec![0.5f32; 10]);
        samples.extend(vec![0.001f32; 100]);
        let result = process(samples, 8000, &c);
        // After trim with 80-sample pad, should keep some silence
        assert!(result.samples.len() > 10, "pad samples missing");
        assert!(result.samples.len() < 210, "trim didn't remove anything");
    }
}
```

- [ ] **Step 2: Run tests (expect pass — pure functions, no external deps)**

```bash
cargo test postprocess::audio --lib 2>&1 | tail -15
```

Expected: 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/postprocess/audio.rs
git commit -m "feat(postprocess): add audio post-processor (DC offset, normalize, trim)"
```

---

## Task 4: Classification post-processor

**Files:**
- Create: `src/postprocess/classify.rs`

- [ ] **Step 1: Write the full module with tests**

Create `src/postprocess/classify.rs`:

```rust
use crate::api::classify::Prediction;
use crate::config::ClassifyPostprocessConfig;

pub struct ClassifyPostprocessResult {
    pub predictions: Vec<Vec<Prediction>>,
    pub steps: Vec<String>,
    pub warnings: Vec<String>,
}

/// Temperature scaling → round to 4dp → drop below min_confidence.
pub fn process(
    predictions: Vec<Vec<Prediction>>,
    config: &ClassifyPostprocessConfig,
) -> ClassifyPostprocessResult {
    if !config.enabled {
        return ClassifyPostprocessResult { predictions, steps: vec![], warnings: vec![] };
    }

    let mut steps = Vec::new();
    let mut warnings = Vec::new();

    let predictions = predictions
        .into_iter()
        .map(|batch| {
            // Stage 1 — Temperature scaling (only when temperature != 1.0)
            let batch = if (config.temperature - 1.0).abs() > f32::EPSILON {
                apply_temperature(batch, config.temperature)
            } else {
                batch
            };

            // Stage 2 — Round to 4 decimal places
            let batch: Vec<Prediction> = batch
                .into_iter()
                .map(|mut p| {
                    p.confidence = (p.confidence * 10_000.0).round() / 10_000.0;
                    p
                })
                .collect();

            // Stage 3 — Filter below min_confidence
            batch
                .into_iter()
                .filter(|p| p.confidence >= config.min_confidence)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    if (config.temperature - 1.0).abs() > f32::EPSILON {
        steps.push(format!("temperature_scaled:{}", config.temperature));
    }
    steps.push("rounded".to_string());
    steps.push(format!("filtered:min={}", config.min_confidence));

    // Count total filtered predictions for warning
    let total: usize = predictions.iter().map(|b| b.len()).sum();
    if total == 0 {
        warnings.push("all_predictions_filtered".to_string());
    }

    ClassifyPostprocessResult { predictions, steps, warnings }
}

fn apply_temperature(predictions: Vec<Prediction>, temperature: f32) -> Vec<Prediction> {
    // Convert confidences back to logits via log, scale by temperature, re-softmax
    let logits: Vec<f32> = predictions
        .iter()
        .map(|p| (p.confidence + 1e-10).ln() / temperature)
        .collect();

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    predictions
        .into_iter()
        .zip(exps)
        .map(|(mut p, e)| {
            p.confidence = if sum > 0.0 { e / sum } else { 0.0 };
            p
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClassifyPostprocessConfig;

    fn cfg() -> ClassifyPostprocessConfig { ClassifyPostprocessConfig::default() }

    fn preds(confs: &[f32]) -> Vec<Vec<Prediction>> {
        vec![confs
            .iter()
            .enumerate()
            .map(|(i, &c)| Prediction {
                label: format!("class_{i}"),
                confidence: c,
                class_id: i,
            })
            .collect()]
    }

    #[test]
    fn test_min_confidence_filters_low_predictions() {
        let result = process(preds(&[0.9, 0.005]), &cfg()); // 0.005 < 0.01
        assert_eq!(result.predictions[0].len(), 1);
        assert_eq!(result.predictions[0][0].label, "class_0");
    }

    #[test]
    fn test_filtered_warning_when_all_filtered() {
        let mut c = cfg();
        c.min_confidence = 1.0; // filter everything
        let result = process(preds(&[0.5, 0.5]), &c);
        assert!(result.warnings.contains(&"all_predictions_filtered".to_string()));
    }

    #[test]
    fn test_rounding_to_4_decimal_places() {
        let mut c = cfg();
        c.min_confidence = 0.0;
        let result = process(preds(&[0.123456789]), &c);
        let conf = result.predictions[0][0].confidence;
        // 0.123456789 rounded to 4dp = 0.1235
        assert!((conf - 0.1235).abs() < 1e-5, "conf={conf}");
    }

    #[test]
    fn test_temperature_above_1_softens_distribution() {
        let mut c = cfg();
        c.temperature = 2.0;
        c.min_confidence = 0.0;
        let result = process(preds(&[0.9, 0.1]), &c);
        // Top class should be less confident than 0.9 after softening
        assert!(result.predictions[0][0].confidence < 0.9);
        assert!(result.steps.iter().any(|s| s.contains("temperature")));
    }

    #[test]
    fn test_temperature_1_not_recorded_in_steps() {
        let result = process(preds(&[0.9, 0.1]), &cfg());
        assert!(!result.steps.iter().any(|s| s.contains("temperature")));
    }

    #[test]
    fn test_temperature_softmax_sums_to_1() {
        let mut c = cfg();
        c.temperature = 3.0;
        c.min_confidence = 0.0;
        let result = process(preds(&[0.7, 0.2, 0.1]), &c);
        let sum: f32 = result.predictions[0].iter().map(|p| p.confidence).sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
    }

    #[test]
    fn test_disabled_passes_through_unchanged() {
        let mut c = cfg();
        c.enabled = false;
        let input = preds(&[0.0001]); // would be filtered if enabled
        let result = process(input.clone(), &c);
        assert_eq!(result.predictions[0].len(), 1);
        assert!(result.steps.is_empty());
    }

    #[test]
    fn test_steps_always_include_rounded_and_filtered() {
        let result = process(preds(&[0.5]), &cfg());
        assert!(result.steps.contains(&"rounded".to_string()));
        assert!(result.steps.iter().any(|s| s.starts_with("filtered")));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test postprocess::classify --lib 2>&1 | tail -15
```

Expected: 8 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/postprocess/classify.rs
git commit -m "feat(postprocess): add classification post-processor (temperature, round, filter)"
```

---

## Task 5: YOLO post-processor

**Files:**
- Create: `src/postprocess/yolo.rs`

- [ ] **Step 1: Write the full module with tests**

Create `src/postprocess/yolo.rs`:

```rust
use serde::Serialize;

use crate::config::YoloPostprocessConfig;
use crate::core::yolo::{BoundingBox, Detection, YoloResults};

#[derive(Debug, Clone, Serialize)]
pub struct BboxRel {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnrichedDetection {
    pub class_id: usize,
    pub class_name: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
    pub bbox_rel: BboxRel,
    pub area: f32,
    pub confidence_bucket: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnrichedYoloResults {
    pub detections: Vec<EnrichedDetection>,
    pub inference_time_ms: f64,
    pub preprocessing_time_ms: f64,
    pub postprocessing_time_ms: f64,
    pub total_time_ms: f64,
}

pub struct YoloPostprocessResult {
    pub results: EnrichedYoloResults,
    pub steps: Vec<String>,
    pub warnings: Vec<String>,
}

/// Enrich YOLO detections with relative bbox, area, and confidence bucket.
/// `img_w` and `img_h` are the original image dimensions in pixels.
pub fn process(
    results: YoloResults,
    img_w: u32,
    img_h: u32,
    config: &YoloPostprocessConfig,
) -> YoloPostprocessResult {
    let mut steps = Vec::new();
    let warnings = Vec::new();

    if !config.enabled {
        // Return enriched type with identity values when disabled
        let detections = results
            .detections
            .into_iter()
            .map(|d| enrich(d, img_w, img_h, config))
            .collect();
        return YoloPostprocessResult {
            results: EnrichedYoloResults {
                detections,
                inference_time_ms: results.inference_time_ms,
                preprocessing_time_ms: results.preprocessing_time_ms,
                postprocessing_time_ms: results.postprocessing_time_ms,
                total_time_ms: results.total_time_ms,
            },
            steps,
            warnings,
        };
    }

    let detections = results
        .detections
        .into_iter()
        .map(|d| enrich(d, img_w, img_h, config))
        .collect();

    steps.push("bbox_rel".to_string());
    steps.push("bbox_enriched".to_string());
    steps.push("confidence_bucketed".to_string());

    YoloPostprocessResult {
        results: EnrichedYoloResults {
            detections,
            inference_time_ms: results.inference_time_ms,
            preprocessing_time_ms: results.preprocessing_time_ms,
            postprocessing_time_ms: results.postprocessing_time_ms,
            total_time_ms: results.total_time_ms,
        },
        steps,
        warnings,
    }
}

fn enrich(d: Detection, img_w: u32, img_h: u32, config: &YoloPostprocessConfig) -> EnrichedDetection {
    let w = img_w as f32;
    let h = img_h as f32;

    let bbox_rel = BboxRel {
        x1: (d.bbox.x1 / w).clamp(0.0, 1.0),
        y1: (d.bbox.y1 / h).clamp(0.0, 1.0),
        x2: (d.bbox.x2 / w).clamp(0.0, 1.0),
        y2: (d.bbox.y2 / h).clamp(0.0, 1.0),
    };

    let area = (d.bbox.x2 - d.bbox.x1).max(0.0) * (d.bbox.y2 - d.bbox.y1).max(0.0);

    let confidence = (d.confidence * 10_000.0).round() / 10_000.0;

    let confidence_bucket = if confidence >= config.high_confidence_threshold {
        "high".to_string()
    } else if confidence >= config.medium_confidence_threshold {
        "medium".to_string()
    } else {
        "low".to_string()
    };

    EnrichedDetection {
        class_id: d.class_id,
        class_name: d.class_name,
        confidence,
        bbox: d.bbox,
        bbox_rel,
        area,
        confidence_bucket,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::YoloPostprocessConfig;
    use crate::core::yolo::{BoundingBox, Detection, YoloResults};

    fn cfg() -> YoloPostprocessConfig { YoloPostprocessConfig::default() }

    fn make_results(detections: Vec<Detection>) -> YoloResults {
        YoloResults {
            detections,
            inference_time_ms: 10.0,
            preprocessing_time_ms: 2.0,
            postprocessing_time_ms: 1.0,
            total_time_ms: 13.0,
        }
    }

    fn detection(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32) -> Detection {
        Detection {
            class_id: 0,
            class_name: "person".into(),
            confidence: conf,
            bbox: BoundingBox { x1, y1, x2, y2 },
        }
    }

    #[test]
    fn test_bbox_rel_normalized_correctly() {
        let d = detection(100.0, 50.0, 300.0, 200.0, 0.8);
        let r = process(make_results(vec![d]), 400, 400, &cfg());
        let rel = &r.results.detections[0].bbox_rel;
        assert!((rel.x1 - 0.25).abs() < 1e-5);
        assert!((rel.y1 - 0.125).abs() < 1e-5);
        assert!((rel.x2 - 0.75).abs() < 1e-5);
        assert!((rel.y2 - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_bbox_rel_clamped_to_0_1() {
        let d = detection(-10.0, -10.0, 500.0, 500.0, 0.8);
        let r = process(make_results(vec![d]), 400, 400, &cfg());
        let rel = &r.results.detections[0].bbox_rel;
        assert!(rel.x1 >= 0.0 && rel.x1 <= 1.0);
        assert!(rel.y1 >= 0.0 && rel.y1 <= 1.0);
        assert!(rel.x2 >= 0.0 && rel.x2 <= 1.0);
        assert!(rel.y2 >= 0.0 && rel.y2 <= 1.0);
    }

    #[test]
    fn test_area_calculated_correctly() {
        let d = detection(0.0, 0.0, 100.0, 50.0, 0.8);
        let r = process(make_results(vec![d]), 640, 480, &cfg());
        assert!((r.results.detections[0].area - 5000.0).abs() < 1e-2);
    }

    #[test]
    fn test_confidence_rounded_to_4dp() {
        let d = detection(0.0, 0.0, 10.0, 10.0, 0.756789);
        let r = process(make_results(vec![d]), 100, 100, &cfg());
        let conf = r.results.detections[0].confidence;
        assert!((conf - 0.7568).abs() < 1e-5, "conf={conf}");
    }

    #[test]
    fn test_confidence_bucket_high() {
        let d = detection(0.0, 0.0, 10.0, 10.0, 0.9);
        let r = process(make_results(vec![d]), 100, 100, &cfg());
        assert_eq!(r.results.detections[0].confidence_bucket, "high");
    }

    #[test]
    fn test_confidence_bucket_medium() {
        let d = detection(0.0, 0.0, 10.0, 10.0, 0.55);
        let r = process(make_results(vec![d]), 100, 100, &cfg());
        assert_eq!(r.results.detections[0].confidence_bucket, "medium");
    }

    #[test]
    fn test_confidence_bucket_low() {
        let d = detection(0.0, 0.0, 10.0, 10.0, 0.3);
        let r = process(make_results(vec![d]), 100, 100, &cfg());
        assert_eq!(r.results.detections[0].confidence_bucket, "low");
    }

    #[test]
    fn test_steps_recorded() {
        let r = process(make_results(vec![detection(0.0, 0.0, 1.0, 1.0, 0.9)]), 100, 100, &cfg());
        assert!(r.steps.contains(&"bbox_rel".to_string()));
        assert!(r.steps.contains(&"bbox_enriched".to_string()));
        assert!(r.steps.contains(&"confidence_bucketed".to_string()));
    }

    #[test]
    fn test_timing_fields_preserved() {
        let results = make_results(vec![]);
        let r = process(results, 100, 100, &cfg());
        assert!((r.results.inference_time_ms - 10.0).abs() < 1e-5);
        assert!((r.results.total_time_ms - 13.0).abs() < 1e-5);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test postprocess::yolo --lib 2>&1 | tail -15
```

Expected: 10 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/postprocess/yolo.rs
git commit -m "feat(postprocess): add YOLO post-processor (bbox_rel, area, bucket)"
```

---

## Task 6: Wire TTS handler (`api/tts.rs`)

**Files:**
- Modify: `src/api/tts.rs`

The `synthesize()` handler returns `SynthesisResponse` which wraps `AudioData` (has `.samples: Vec<f32>` and `.sample_rate: u32`).

- [ ] **Step 1: Add `skip_postprocess` to `SynthesisRequest`**

In `src/api/tts.rs`, find `pub struct SynthesisRequest` and add the new field:

```rust
#[derive(Debug, Deserialize)]
pub struct SynthesisRequest {
    pub text: String,
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default = "default_pitch")]
    pub pitch: f32,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub skip_postprocess: bool,   // ← add this
}
```

- [ ] **Step 2: Add imports at top of `src/api/tts.rs`**

Add after the existing imports:

```rust
use actix_web::HttpRequest;
use std::time::Instant;
use crate::config::Config;
use crate::postprocess::{self, envelope::ResponseMeta, Envelope};
use crate::middleware::correlation_id::get_correlation_id;
```

- [ ] **Step 3: Update the `synthesize()` handler signature and body**

Replace the entire `synthesize()` function:

```rust
pub async fn synthesize(
    req: web::Json<SynthesisRequest>,
    state: web::Data<TTSState>,
    config: web::Data<Config>,
    http_req: HttpRequest,
) -> Result<HttpResponse, ApiError> {
    let start = Instant::now();

    if req.text.is_empty() {
        return Err(ApiError::BadRequest("Text cannot be empty".to_string()));
    }
    if req.text.len() > 50000 {
        return Err(ApiError::BadRequest("Text too long (max 50000 characters)".to_string()));
    }

    let params = SynthesisParams {
        speed: req.speed.max(0.25).min(4.0),
        pitch: req.pitch.max(0.5).min(2.0),
        voice: req.voice.clone(),
        language: req.language.clone(),
    };

    let mut audio = state.manager.synthesize(
        &req.text,
        req.engine.as_deref(),
        params,
    ).await.map_err(|e| ApiError::InternalError(format!("Synthesis failed: {}", e)))?;

    let engine_used = if let Some(engine_id) = req.engine.as_deref() {
        engine_id.to_string()
    } else {
        state.manager.get_default_engine()
            .map(|e| e.name().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    };

    // Post-process samples (DC offset → normalize → trim)
    let (pp_steps, pp_warnings) = if !req.skip_postprocess {
        let result = postprocess::audio::process(
            std::mem::take(&mut audio.samples),
            audio.sample_rate,
            &config.postprocess.audio,
        );
        audio.samples = result.samples;
        (result.steps, result.warnings)
    } else {
        (vec![], vec![])
    };

    let wav_data = AudioProcessor::default()
        .save_wav(&audio)
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    use base64::Engine as _;
    let audio_base64 = base64::engine::general_purpose::STANDARD.encode(&wav_data);
    let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;

    let data = SynthesisResponse {
        audio_base64,
        sample_rate: audio.sample_rate,
        duration_secs,
        format: "wav".to_string(),
        engine_used: engine_used.clone(),
    };

    let envelope = Envelope::new(data, ResponseMeta {
        latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        model_id: engine_used,
        postprocessing_applied: !req.skip_postprocess && !pp_steps.is_empty(),
        postprocess_steps: pp_steps,
        warnings: pp_warnings,
        version: env!("CARGO_PKG_VERSION"),
        request_id: get_correlation_id(&http_req).as_str().to_string(),
    });

    Ok(HttpResponse::Ok().json(envelope))
}
```

- [ ] **Step 4: Check it compiles**

```bash
cargo check 2>&1 | grep -E "^error"
```

Expected: no errors. If `audio.samples` is not pub-mut, the compiler will tell you — in that case replace `std::mem::take(&mut audio.samples)` with `audio.samples.clone()` and assign back: `audio.samples = result.samples;`.

- [ ] **Step 5: Commit**

```bash
git add src/api/tts.rs
git commit -m "feat(postprocess): wire audio post-processing and envelope into TTS handler"
```

---

## Task 7: Wire Audio handler (`api/audio.rs`)

**Files:**
- Modify: `src/api/audio.rs`

- [ ] **Step 1: Add `skip_postprocess` to `SynthesizeRequest`**

In `src/api/audio.rs`, find `pub struct SynthesizeRequest` and add:

```rust
#[derive(Debug, Deserialize)]
pub struct SynthesizeRequest {
    pub text: String,
    pub model: Option<String>,
    pub voice: Option<String>,
    pub speed: Option<f32>,
    pub pitch: Option<f32>,
    #[serde(default)]
    pub skip_postprocess: bool,  // ← add this
}
```

- [ ] **Step 2: Add imports to `src/api/audio.rs`**

Add after the existing imports:

```rust
use actix_web::HttpRequest;
use std::time::Instant;
use crate::config::Config;
use crate::postprocess::{self, envelope::ResponseMeta, Envelope};
use crate::middleware::correlation_id::get_correlation_id;
```

- [ ] **Step 3: Update `synthesize_speech()` handler signature**

Find `pub async fn synthesize_speech(` and update its signature to accept `config` and `http_req`:

```rust
pub async fn synthesize_speech(
    req: web::Json<SynthesizeRequest>,
    state: web::Data<AudioState>,
    config: web::Data<Config>,
    http_req: HttpRequest,
) -> Result<HttpResponse, ApiError> {
    let start = Instant::now();
    // ... existing body unchanged until the point where audio samples are produced ...
```

- [ ] **Step 4: Add post-processing before WAV encoding**

Find the section in `synthesize_speech()` that calls `AudioProcessor` (after the model produces samples). Insert post-processing between synthesis and encoding. Look for the pattern:

```rust
// BEFORE (existing):
let wav_data = AudioProcessor::default()
    .save_wav(&audio_data)
    ...;

// AFTER (replace with):
let mut audio_data = audio_data; // ensure mutability

let (pp_steps, pp_warnings) = if !req.skip_postprocess {
    let result = postprocess::audio::process(
        std::mem::take(&mut audio_data.samples),
        audio_data.sample_rate,
        &config.postprocess.audio,
    );
    audio_data.samples = result.samples;
    (result.steps, result.warnings)
} else {
    (vec![], vec![])
};

let wav_data = AudioProcessor::default()
    .save_wav(&audio_data)
    .map_err(|e| ApiError::InternalError(e.to_string()))?;
```

- [ ] **Step 5: Wrap response in Envelope**

Find the final `Ok(HttpResponse::Ok().json(SynthesizeResponse { ... }))` and replace with:

```rust
let model_name = req.model.as_deref().unwrap_or("default").to_string();
let data = SynthesizeResponse {
    audio_base64,
    sample_rate: audio_data.sample_rate,
    duration_secs,
    format: "wav".to_string(),
};

let envelope = Envelope::new(data, ResponseMeta {
    latency_ms: start.elapsed().as_secs_f64() * 1000.0,
    model_id: model_name,
    postprocessing_applied: !req.skip_postprocess && !pp_steps.is_empty(),
    postprocess_steps: pp_steps,
    warnings: pp_warnings,
    version: env!("CARGO_PKG_VERSION"),
    request_id: get_correlation_id(&http_req).as_str().to_string(),
});

Ok(HttpResponse::Ok().json(envelope))
```

- [ ] **Step 6: Check it compiles**

```bash
cargo check 2>&1 | grep -E "^error"
```

- [ ] **Step 7: Commit**

```bash
git add src/api/audio.rs
git commit -m "feat(postprocess): wire audio post-processing and envelope into audio handler"
```

---

## Task 8: Wire Classification handler (`api/classify.rs`)

**Files:**
- Modify: `src/api/classify.rs`

- [ ] **Step 1: Add `skip_postprocess` to `BatchClassifyRequest`**

In `src/api/classify.rs`, find `pub struct BatchClassifyRequest` and add:

```rust
#[derive(Debug, Deserialize)]
pub struct BatchClassifyRequest {
    pub images: Vec<String>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_model_dim")]
    pub model_width: u32,
    #[serde(default = "default_model_dim")]
    pub model_height: u32,
    #[serde(default)]
    pub skip_postprocess: bool,  // ← add this
}
```

- [ ] **Step 2: Add imports to `src/api/classify.rs`**

Add after the existing imports:

```rust
use actix_web::HttpRequest;
use std::time::Instant;
use crate::config::Config;
use crate::postprocess::{self, envelope::ResponseMeta, Envelope};
use crate::middleware::correlation_id::get_correlation_id;
```

- [ ] **Step 3: Update `batch_classify()` handler**

Replace the entire `batch_classify()` function:

```rust
pub async fn batch_classify(
    req: web::Json<BatchClassifyRequest>,
    state: web::Data<ClassifyState>,
    config: web::Data<Config>,
    http_req: HttpRequest,
) -> Result<HttpResponse, ApiError> {
    let start = Instant::now();

    // Validation (unchanged from before)
    if req.images.is_empty() {
        return Err(ApiError::BadRequest("images array must not be empty".to_string()));
    }
    if req.images.len() > 128 {
        return Err(ApiError::BadRequest("batch too large (max 128 images)".to_string()));
    }
    if req.top_k == 0 {
        return Err(ApiError::BadRequest("top_k must be >= 1".to_string()));
    }
    if req.top_k > 1000 {
        return Err(ApiError::BadRequest("top_k must be <= 1000".to_string()));
    }
    if req.model_width == 0 || req.model_height == 0 {
        return Err(ApiError::BadRequest("model_width and model_height must be >= 1".to_string()));
    }
    if req.model_width > 4096 || req.model_height > 4096 {
        return Err(ApiError::BadRequest("model_width and model_height must be <= 4096".to_string()));
    }

    use base64::Engine as _;
    let raw_images: Vec<Vec<u8>> = req
        .images
        .iter()
        .enumerate()
        .map(|(i, b64)| {
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| ApiError::BadRequest(format!("image[{}]: {}", i, e)))
        })
        .collect::<Result<_, _>>()?;

    let cfg = crate::core::image_pipeline::PreprocessConfig::imagenet(req.model_width, req.model_height);
    let pipeline = crate::core::image_pipeline::ImagePipeline::new(cfg);
    let batch = pipeline
        .preprocess_batch(&raw_images)
        .map_err(|e| ApiError::BadRequest(format!("preprocess failed: {}", e)))?;

    let results = state
        .backend
        .classify_nchw(batch, req.top_k)
        .await
        .map_err(|e| ApiError::InternalError(format!("inference failed: {}", e)))?;

    // Post-process
    let (results, pp_steps, pp_warnings) = if !req.skip_postprocess {
        let pp = postprocess::classify::process(results, &config.postprocess.classify);
        (pp.predictions, pp.steps, pp.warnings)
    } else {
        (results, vec![], vec![])
    };

    let data = BatchClassifyResponse {
        batch_size: results.len(),
        results,
    };

    let envelope = Envelope::new(data, ResponseMeta {
        latency_ms: start.elapsed().as_secs_f64() * 1000.0,
        model_id: "classification-backend".to_string(),
        postprocessing_applied: !req.skip_postprocess && !pp_steps.is_empty(),
        postprocess_steps: pp_steps,
        warnings: pp_warnings,
        version: env!("CARGO_PKG_VERSION"),
        request_id: get_correlation_id(&http_req).as_str().to_string(),
    });

    Ok(HttpResponse::Ok().json(envelope))
}
```

- [ ] **Step 4: Check it compiles**

```bash
cargo check 2>&1 | grep -E "^error"
```

- [ ] **Step 5: Commit**

```bash
git add src/api/classify.rs
git commit -m "feat(postprocess): wire classification post-processing and envelope into classify handler"
```

---

## Task 9: Wire YOLO handler (`api/yolo.rs`)

**Files:**
- Modify: `src/api/yolo.rs`

The YOLO handler uses `web::Query<YoloDetectRequest>` (not JSON body) and multipart upload, so `skip_postprocess` goes in the query params. Image dimensions are obtained by decoding the temp file with the `image` crate (already in Cargo.toml).

- [ ] **Step 1: Add `skip_postprocess` to `YoloDetectRequest`**

In `src/api/yolo.rs`, find `pub struct YoloDetectRequest` and add:

```rust
#[derive(Debug, Deserialize)]
pub struct YoloDetectRequest {
    pub model_version: String,
    pub model_size: String,
    #[serde(default = "default_conf_threshold")]
    pub conf_threshold: f32,
    #[serde(default = "default_iou_threshold")]
    pub iou_threshold: f32,
    #[serde(default)]
    pub skip_postprocess: bool,  // ← add this
}
```

- [ ] **Step 2: Add imports to `src/api/yolo.rs`**

Add after the existing imports:

```rust
use actix_web::HttpRequest;
use std::time::Instant;
use crate::config::Config;
use crate::postprocess::{self, envelope::ResponseMeta, Envelope};
use crate::postprocess::yolo::EnrichedYoloResults;
use crate::middleware::correlation_id::get_correlation_id;
```

- [ ] **Step 3: Add `EnrichedYoloDetectResponse`**

Add this new struct to `src/api/yolo.rs` (alongside the existing `YoloDetectResponse`):

```rust
#[derive(Debug, Serialize)]
pub struct EnrichedYoloDetectResponse {
    pub success: bool,
    pub results: Option<EnrichedYoloResults>,
    pub error: Option<String>,
}
```

- [ ] **Step 4: Update `detect_objects()` handler**

Replace the entire `detect_objects()` function with:

```rust
pub async fn detect_objects(
    mut payload: Multipart,
    query: web::Query<YoloDetectRequest>,
    state: web::Data<YoloState>,
    config: web::Data<Config>,
    http_req: HttpRequest,
) -> Result<HttpResponse, ApiError> {
    let start = Instant::now();

    let version = YoloVersion::from_str(&query.model_version)
        .ok_or_else(|| ApiError::BadRequest(format!("Invalid YOLO version: {}", query.model_version)))?;
    let size = YoloSize::from_suffix(&query.model_size)
        .ok_or_else(|| ApiError::BadRequest(format!("Invalid model size: {}", query.model_size)))?;

    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join(format!("yolo_input_{}.jpg", uuid::Uuid::new_v4()));

    while let Some(Ok(mut field)) = payload.next().await {
        let mut file = fs::File::create(&temp_file).await
            .map_err(|e| ApiError::InternalError(e.to_string()))?;
        while let Some(chunk) = field.next().await {
            let data = chunk.map_err(|e| ApiError::InternalError(e.to_string()))?;
            file.write_all(&data).await
                .map_err(|e| ApiError::InternalError(e.to_string()))?;
        }
    }

    // Get image dimensions for bbox_rel calculation
    let (img_w, img_h) = image::image_dimensions(&temp_file)
        .unwrap_or((640, 640)); // graceful fallback: can't fail the request over this

    let model_name = format!("yolo{}{}",
        version.as_str().to_lowercase().replace("yolo", ""),
        size.suffix()
    );
    let model_path = state.models_dir.join(&model_name).join(format!("{}.pt", model_name));

    if !model_path.exists() {
        let _ = fs::remove_file(&temp_file).await;
        return Err(ApiError::NotFound(format!(
            "Model not found: {}. Please download it first.", model_name
        )));
    }

    let class_names = load_coco_names();

    #[cfg(feature = "torch")]
    let detector = {
        use tch::Device;
        let mut detector = YoloDetector::new(
            &model_path,
            version,
            size,
            class_names,
            Some(Device::Cpu),
        ).map_err(|e| ApiError::InternalError(e.to_string()))?;
        detector.set_conf_threshold(query.conf_threshold);
        detector.set_iou_threshold(query.iou_threshold);
        detector
    };

    #[cfg(not(feature = "torch"))]
    {
        let _ = fs::remove_file(&temp_file).await;
        return Err(ApiError::InternalError(
            "PyTorch feature not enabled. Compile with --features torch".to_string()
        ));
    }

    #[cfg(feature = "torch")]
    let raw_results = detector.detect(&temp_file)
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    let _ = fs::remove_file(&temp_file).await;

    #[cfg(feature = "torch")]
    {
        // Post-process
        let (enriched, pp_steps, pp_warnings) = if !query.skip_postprocess {
            let pp = postprocess::yolo::process(raw_results, img_w, img_h, &config.postprocess.yolo);
            (pp.results, pp.steps, pp.warnings)
        } else {
            let pp = postprocess::yolo::process(raw_results, img_w, img_h, &config.postprocess.yolo);
            (pp.results, vec![], vec![])
        };

        let data = EnrichedYoloDetectResponse {
            success: true,
            results: Some(enriched),
            error: None,
        };

        let envelope = Envelope::new(data, ResponseMeta {
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            model_id: model_name,
            postprocessing_applied: !query.skip_postprocess && !pp_steps.is_empty(),
            postprocess_steps: pp_steps,
            warnings: pp_warnings,
            version: env!("CARGO_PKG_VERSION"),
            request_id: get_correlation_id(&http_req).as_str().to_string(),
        });

        return Ok(HttpResponse::Ok().json(envelope));
    }

    #[cfg(not(feature = "torch"))]
    Ok(HttpResponse::Ok().json(EnrichedYoloDetectResponse {
        success: false,
        results: None,
        error: Some("PyTorch not enabled".to_string()),
    }))
}
```

- [ ] **Step 5: Check it compiles**

```bash
cargo check 2>&1 | grep -E "^error"
```

- [ ] **Step 6: Run all tests to confirm nothing broke**

```bash
cargo test --lib 2>&1 | grep -E "test result:|FAILED"
```

Expected: `test result: ok.` with 0 failures.

- [ ] **Step 7: Final commit**

```bash
git add src/api/yolo.rs
git commit -m "feat(postprocess): wire YOLO post-processing and envelope into detect handler"
```
