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
        return ClassifyPostprocessResult {
            predictions,
            steps: vec![],
            warnings: vec![],
        };
    }

    let mut steps = Vec::new();
    let mut warnings = Vec::new();

    let mut dropped: usize = 0;

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
            let before = batch.len();
            let filtered: Vec<_> = batch
                .into_iter()
                .filter(|p| p.confidence >= config.min_confidence)
                .collect();
            dropped += before - filtered.len();
            filtered
        })
        .collect::<Vec<_>>();

    if (config.temperature - 1.0).abs() > f32::EPSILON {
        steps.push(format!("temperature_scaled:{}", config.temperature));
    }
    steps.push("rounded".to_string());
    steps.push(format!("filtered:min={}", config.min_confidence));

    if dropped > 0 {
        warnings.push(format!("predictions_filtered:{dropped}"));
    }

    ClassifyPostprocessResult {
        predictions,
        steps,
        warnings,
    }
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

    fn cfg() -> ClassifyPostprocessConfig {
        ClassifyPostprocessConfig::default()
    }

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
        assert!(result
            .warnings
            .iter()
            .any(|w| w.starts_with("predictions_filtered:")));
    }

    #[test]
    fn test_filtered_warning_on_partial_drop() {
        let result = process(preds(&[0.9, 0.005]), &cfg()); // 0.005 < 0.01 min
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w == "predictions_filtered:1"),
            "expected warning for 1 dropped prediction"
        );
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
