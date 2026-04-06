// Items in this module are only called from api/yolo.rs, which is gated on
// #[cfg(feature = "torch")]. Without that feature rustc sees them as dead code —
// suppress the false positive.
#![cfg_attr(not(feature = "torch"), allow(dead_code))]

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
/// Zero-dimension images are handled safely (clamped to 1 pixel to avoid NaN).
/// When `config.enabled = false`, enrichment still runs (return type requires it)
/// but steps are not recorded — callers see `steps: []`.
pub fn process(
    results: YoloResults,
    img_w: u32,
    img_h: u32,
    config: &YoloPostprocessConfig,
) -> YoloPostprocessResult {
    let mut steps = Vec::new();
    let warnings: Vec<String> = Vec::new(); // populated by future warning stages

    // Capture timing fields before partial move of `results.detections`
    let inference_time_ms = results.inference_time_ms;
    let preprocessing_time_ms = results.preprocessing_time_ms;
    let postprocessing_time_ms = results.postprocessing_time_ms;
    let total_time_ms = results.total_time_ms;

    let detections = results
        .detections
        .into_iter()
        .map(|d| enrich(d, img_w, img_h, config))
        .collect();

    if config.enabled {
        steps.push("bbox_rel".to_string());
        steps.push("bbox_enriched".to_string());
        steps.push("confidence_bucketed".to_string());
    }

    YoloPostprocessResult {
        results: EnrichedYoloResults {
            detections,
            inference_time_ms,
            preprocessing_time_ms,
            postprocessing_time_ms,
            total_time_ms,
        },
        steps,
        warnings,
    }
}

fn enrich(
    d: Detection,
    img_w: u32,
    img_h: u32,
    config: &YoloPostprocessConfig,
) -> EnrichedDetection {
    // Clamp to 1.0 to avoid NaN from division by zero on zero-dimension images
    let w = (img_w as f32).max(1.0);
    let h = (img_h as f32).max(1.0);

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

    fn cfg() -> YoloPostprocessConfig {
        YoloPostprocessConfig::default()
    }

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
        let r = process(
            make_results(vec![detection(0.0, 0.0, 1.0, 1.0, 0.9)]),
            100,
            100,
            &cfg(),
        );
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

    #[test]
    fn test_zero_dimension_image_no_nan() {
        let d = detection(10.0, 10.0, 50.0, 50.0, 0.8);
        let r = process(make_results(vec![d]), 0, 0, &cfg());
        let rel = &r.results.detections[0].bbox_rel;
        assert!(rel.x1.is_finite(), "x1 is NaN/Inf for zero-dim image");
        assert!(rel.y1.is_finite());
        assert!(rel.x2.is_finite());
        assert!(rel.y2.is_finite());
    }

    #[test]
    fn test_disabled_steps_empty() {
        let mut c = cfg();
        c.enabled = false;
        let d = detection(0.0, 0.0, 10.0, 10.0, 0.9);
        let r = process(make_results(vec![d]), 100, 100, &c);
        assert!(r.steps.is_empty(), "steps should be empty when disabled");
    }
}
