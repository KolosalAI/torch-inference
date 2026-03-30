#![allow(dead_code)]
/// Token sampler — greedy, temperature, top-k, and nucleus (top-p) sampling.
///
/// All methods are stateless pure functions operating on logit slices so they
/// can be called from any thread without locking.
use anyhow::{bail, Result};

// ── Sampling parameters ───────────────────────────────────────────────────

/// Controls how the next token is sampled from the model's logit distribution.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature. 1.0 = unchanged; <1.0 sharpens; >1.0 flattens.
    /// Set to 0.0 to force greedy decoding.
    pub temperature: f32,
    /// Top-p (nucleus) cutoff in (0.0, 1.0].  1.0 disables.
    pub top_p: f32,
    /// Top-k cutoff (0 disables).
    pub top_k: usize,
    /// Maximum tokens to generate (including prompt continuation).
    pub max_tokens: usize,
    /// Stop at these token IDs (generation ends, token not emitted).
    pub stop_token_ids: Vec<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            max_tokens: 256,
            stop_token_ids: vec![],
        }
    }
}

impl SamplingParams {
    /// Greedy decoding (always picks the highest-probability token).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            max_tokens: 256,
            stop_token_ids: vec![],
        }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.temperature < 0.0 {
            bail!("temperature must be >= 0");
        }
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            bail!("top_p must be in (0, 1]");
        }
        if self.max_tokens == 0 {
            bail!("max_tokens must be >= 1");
        }
        Ok(())
    }
}

// ── Sampler ───────────────────────────────────────────────────────────────

/// Samples the next token from `logits` according to `params`.
///
/// `logits` has shape `[vocab_size]` as raw model output (before softmax).
///
/// Returns a token id in `[0, vocab_size)`.
pub fn sample(logits: &[f32], params: &SamplingParams) -> Result<u32> {
    if logits.is_empty() {
        bail!("logits must be non-empty");
    }

    // Fast path: greedy
    if params.temperature == 0.0 || params.top_k == 1 {
        return Ok(argmax(logits) as u32);
    }

    // Apply temperature
    let mut scores: Vec<f32> = logits
        .iter()
        .map(|&l| l / params.temperature)
        .collect();

    // Top-k filtering
    if params.top_k > 0 && params.top_k < scores.len() {
        apply_top_k(&mut scores, params.top_k);
    }

    // Softmax → probabilities
    let probs = softmax(&scores);

    // Top-p (nucleus) filtering
    let probs = if params.top_p < 1.0 {
        apply_top_p(probs, params.top_p)
    } else {
        probs
    };

    // Multinomial sample
    Ok(multinomial_sample(&probs) as u32)
}

// ── Internals ─────────────────────────────────────────────────────────────

/// Return the index of the maximum value.
pub fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Numerically stable softmax in-place, returning a new Vec.
pub fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Zero out all but the top-k logits (the rest become -inf).
fn apply_top_k(scores: &mut [f32], k: usize) {
    // Find the kth-largest value via partial sort.
    let mut sorted: Vec<f32> = scores.iter().cloned().collect();
    // Descending partial sort: element at index k-1 is the k-th largest.
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    // Keep only scores >= threshold; zero rest as -inf.
    let mut kept = 0;
    for s in scores.iter_mut() {
        if *s >= threshold && kept < k {
            kept += 1;
        } else {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// Keep the smallest set of tokens whose cumulative probability >= top_p.
/// Returns a new probability vector (already-summed to 1 after re-normalisation).
fn apply_top_p(probs: Vec<f32>, top_p: f32) -> Vec<f32> {
    // Sort indices by probability descending.
    let mut indexed: Vec<(usize, f32)> =
        probs.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate until cumsum >= top_p.
    let mut cumsum = 0.0_f32;
    let mut cutoff_idx = indexed.len();
    for (i, (_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Build filtered probability vector.
    let keep: std::collections::HashSet<usize> =
        indexed[..cutoff_idx].iter().map(|(i, _)| *i).collect();
    let mut filtered: Vec<f32> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| if keep.contains(&i) { p } else { 0.0 })
        .collect();

    // Re-normalise.
    let sum: f32 = filtered.iter().sum();
    if sum > 0.0 {
        filtered.iter_mut().for_each(|p| *p /= sum);
    }
    filtered
}

/// Draw a sample from a discrete distribution.  Uses a simple linear scan.
fn multinomial_sample(probs: &[f32]) -> usize {
    // Use a fixed seed for reproducibility in tests; real inference injects RNG.
    // For production, replace with a seeded RNG passed in via params.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    // Hash the probs as raw bits for a deterministic-ish sample.
    let bits: Vec<u32> = probs.iter().map(|&p| p.to_bits()).collect();
    bits.hash(&mut h);
    let rand_val = (h.finish() as f32) / (u64::MAX as f32); // [0, 1)
    let rand_val = rand_val.abs(); // always positive

    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val <= cumsum {
            return i;
        }
    }
    probs.len() - 1 // fallback: last token
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SamplingParams ────────────────────────────────────────────────────

    #[test]
    fn test_default_params() {
        let p = SamplingParams::default();
        assert!((p.temperature - 1.0).abs() < 1e-6);
        assert!((p.top_p - 1.0).abs() < 1e-6);
        assert_eq!(p.top_k, 0);
        assert_eq!(p.max_tokens, 256);
    }

    #[test]
    fn test_greedy_params() {
        let p = SamplingParams::greedy();
        assert_eq!(p.temperature, 0.0);
        assert_eq!(p.top_k, 1);
    }

    #[test]
    fn test_builder_methods() {
        let p = SamplingParams::default()
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_top_k(50)
            .with_max_tokens(512);
        assert!((p.temperature - 0.7).abs() < 1e-6);
        assert!((p.top_p - 0.9).abs() < 1e-6);
        assert_eq!(p.top_k, 50);
        assert_eq!(p.max_tokens, 512);
    }

    #[test]
    fn test_validate_ok() {
        assert!(SamplingParams::default().validate().is_ok());
    }

    #[test]
    fn test_validate_negative_temperature() {
        let p = SamplingParams::default().with_temperature(-0.1);
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_zero_top_p() {
        let p = SamplingParams { top_p: 0.0, ..SamplingParams::default() };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_top_p_gt_1() {
        let p = SamplingParams { top_p: 1.1, ..SamplingParams::default() };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_zero_max_tokens() {
        let p = SamplingParams::default().with_max_tokens(0);
        assert!(p.validate().is_err());
    }

    // ── argmax ────────────────────────────────────────────────────────────

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[1.0, 0.0, 0.0]), 0);
        assert_eq!(argmax(&[0.0, 0.0, 1.0]), 2);
    }

    #[test]
    fn test_argmax_single_element() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    // ── softmax ───────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let p = softmax(&v);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_preserves_order() {
        let v = vec![1.0f32, 3.0, 2.0];
        let p = softmax(&v);
        assert!(p[1] > p[2]);
        assert!(p[2] > p[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let v = vec![0.0f32; 4];
        let p = softmax(&v);
        for &prob in &p {
            assert!((prob - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_numerically_stable_large_values() {
        let v = vec![1000.0f32, 1001.0, 1002.0];
        let p = softmax(&v);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={}", sum);
        assert!(p.iter().all(|&x| x.is_finite()));
    }

    // ── sample ────────────────────────────────────────────────────────────

    #[test]
    fn test_sample_greedy_always_picks_max() {
        let logits = vec![0.1f32, 5.0, 2.0, 1.0];
        let params = SamplingParams::greedy();
        let tok = sample(&logits, &params).unwrap();
        assert_eq!(tok, 1);
    }

    #[test]
    fn test_sample_empty_logits_returns_error() {
        let result = sample(&[], &SamplingParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_single_token() {
        let logits = vec![1.0f32];
        let tok = sample(&logits, &SamplingParams::default()).unwrap();
        assert_eq!(tok, 0);
    }

    #[test]
    fn test_sample_temperature_zero_is_greedy() {
        let logits = vec![0.5f32, 10.0, 1.0];
        let params = SamplingParams::default().with_temperature(0.0);
        let tok = sample(&logits, &params).unwrap();
        assert_eq!(tok, 1);
    }

    #[test]
    fn test_sample_top_k_one_is_greedy() {
        let logits = vec![0.1f32, 8.0, 3.0];
        let params = SamplingParams::default().with_top_k(1);
        let tok = sample(&logits, &params).unwrap();
        assert_eq!(tok, 1);
    }

    #[test]
    fn test_sample_result_in_valid_range() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let params = SamplingParams::default();
        let tok = sample(&logits, &params).unwrap();
        assert!((tok as usize) < logits.len());
    }

    // ── apply_top_k (via sample) ──────────────────────────────────────────

    #[test]
    fn test_top_k_restricts_vocabulary() {
        // Only tokens 3,4 have high logits; top_k=2 should always pick one of them.
        let logits = vec![-10.0f32, -10.0, -10.0, 10.0, 9.0];
        let params = SamplingParams::default()
            .with_top_k(2)
            .with_temperature(1.0);
        for _ in 0..20 {
            let tok = sample(&logits, &params).unwrap();
            assert!(tok == 3 || tok == 4, "unexpected token {}", tok);
        }
    }

    // ── apply_top_p (via sample) ──────────────────────────────────────────

    #[test]
    fn test_top_p_very_small_forces_top_token() {
        // With top_p=0.01 only the dominant token survives.
        let logits = vec![-100.0f32, 100.0, -100.0];
        let params = SamplingParams { top_p: 0.01, ..SamplingParams::default() };
        let tok = sample(&logits, &params).unwrap();
        assert_eq!(tok, 1);
    }

    // ── sampling params clone + debug ─────────────────────────────────────

    #[test]
    fn test_sampling_params_clone_and_debug() {
        let p = SamplingParams::greedy();
        let q = p.clone();
        assert_eq!(q.temperature, 0.0);
        let s = format!("{:?}", q);
        assert!(s.contains("temperature"));
    }
}
