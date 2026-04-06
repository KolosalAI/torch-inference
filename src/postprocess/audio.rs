use crate::config::AudioPostprocessConfig;

pub struct AudioPostprocessResult {
    pub samples: Vec<f32>,
    pub steps: Vec<String>,
    pub warnings: Vec<String>,
}

/// Silence detection on raw signal → DC offset removal → peak normalization → trim slice.
/// Silence is detected before DC removal to avoid DC bias masking silence boundaries.
/// Returns the processed samples plus audit steps and warnings.
/// Never panics; if all samples are silent, returns them unchanged with a warning.
pub fn process(
    mut samples: Vec<f32>,
    sample_rate: u32,
    config: &AudioPostprocessConfig,
) -> AudioPostprocessResult {
    if !config.enabled || samples.is_empty() {
        return AudioPostprocessResult {
            samples,
            steps: vec![],
            warnings: vec![],
        };
    }
    let mut steps = Vec::new();
    let mut warnings = Vec::new();

    // Check for clipping on raw input before any processing
    let had_clipping = samples.iter().any(|s| s.abs() > 1.0);

    // Stage 3 (detection pass) — Silence trimming on original signal
    // Silence detection is performed first on the raw samples so that DC offset does not
    // cause originally-silent samples to appear loud after mean subtraction.
    let threshold = config.silence_threshold;
    let pad_samples = (config.pad_ms as f32 / 1000.0 * sample_rate as f32) as usize;

    let first_loud = samples.iter().position(|s| s.abs() >= threshold);
    let last_loud = samples.iter().rposition(|s| s.abs() >= threshold);

    match (first_loud, last_loud) {
        (Some(start), Some(end)) => {
            let trim_start = start.saturating_sub(pad_samples);
            let trim_end = (end + 1 + pad_samples).min(samples.len());
            samples = samples[trim_start..trim_end].to_vec();
        }
        _ => {
            // All silence — return immediately with warning; no further processing needed
            warnings.push("all_silence".to_string());
            return AudioPostprocessResult {
                samples,
                steps: vec![],
                warnings,
            };
        }
    }

    // Stage 1 — DC offset removal
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    if mean.abs() > 0.001 {
        for s in &mut samples {
            *s -= mean;
        }
        warnings.push(format!("dc_offset_removed:{mean:.4}"));
    }
    steps.push("dc_offset".to_string());

    // Stage 2 — Peak normalization
    let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if had_clipping {
        warnings.push("clipping_detected".to_string());
    }
    if peak > 0.0 {
        let scale = config.target_peak / peak;
        for s in &mut samples {
            *s *= scale;
        }
    } else {
        warnings.push("zero_peak_after_dc".to_string());
    }
    steps.push(format!("normalize:{}", config.target_peak));
    steps.push(format!("trim:pad={}ms", config.pad_ms));

    AudioPostprocessResult {
        samples,
        steps,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AudioPostprocessConfig;

    fn cfg() -> AudioPostprocessConfig {
        AudioPostprocessConfig::default()
    }

    #[test]
    fn test_dc_offset_removed() {
        let samples = vec![0.5f32; 100];
        let result = process(samples, 22050, &cfg());
        let mean: f32 = result.samples.iter().sum::<f32>() / result.samples.len() as f32;
        assert!(mean.abs() < 1e-4, "DC offset not removed, mean={mean}");
        assert!(result
            .warnings
            .iter()
            .any(|w| w.starts_with("dc_offset_removed")));
        assert!(result.steps.contains(&"dc_offset".to_string()));
    }

    #[test]
    fn test_peak_normalized_to_target() {
        let mut c = cfg();
        c.silence_threshold = 0.0; // disable trim
        c.pad_ms = 0;
        let samples = vec![0.5f32, -0.5f32, 0.3f32, -0.3f32];
        let result = process(samples, 22050, &c);
        let peak = result
            .samples
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
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
        assert!(
            result.steps.is_empty(),
            "no steps should be recorded when all silent"
        );
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
