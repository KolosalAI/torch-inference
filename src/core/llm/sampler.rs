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
    let mut scores: Vec<f32> = logits.iter().map(|&l| l / params.temperature).collect();

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
///
/// Uses a plain `>` comparison so the loop is auto-vectorizable by LLVM.
/// The previous iterator chain (`max_by` with `partial_cmp`) emitted an
/// `Ordering` enum per comparison and could not be vectorized.
pub fn argmax(v: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best_idx = i;
        }
    }
    best_idx
}

/// Sample from an already-normalized probability distribution.
///
/// Unlike [`sample`], this does **not** apply temperature, top-k, top-p, or
/// softmax — it expects `probs` to already sum to 1.  Use this when you hold
/// probabilities directly (e.g. in speculative decoding's rejection step) to
/// avoid a needless `log(p) → softmax(log(p))` roundtrip.
pub fn sample_from_probs(probs: &[f32]) -> usize {
    multinomial_sample(probs)
}

/// Numerically stable softmax, returning a new Vec.
///
/// Single allocation: exp values are written in-place and then normalised by
/// multiplying with the reciprocal of the sum (one multiply vs one divide per
/// element — measurably faster on modern CPUs).
pub fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut result: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = result.iter().sum();
    let inv_sum = 1.0 / sum;
    result.iter_mut().for_each(|e| *e *= inv_sum);
    result
}

/// Zero out all but the top-k logits (the rest become -inf).
///
/// Uses `select_nth_unstable_by` — O(n) average (Floyd-Rivest / introselect)
/// instead of a full O(n log n) sort.  For typical LLM vocab sizes (32k–128k)
/// this is ~10–15× faster than the sort-based approach.
fn apply_top_k(scores: &mut [f32], k: usize) {
    // Partition a scratch copy so that scratch[k-1] is the k-th largest value.
    // Elements in scratch[..k] are all >= scratch[k-1] (unordered within that
    // range — we only need the threshold, not a sorted prefix).
    let mut scratch: Vec<f32> = scores.iter().cloned().collect();
    scratch.select_nth_unstable_by(k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = scratch[k - 1];
    // Mask out every entry below the threshold, respecting exact ties.
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
/// Returns a new probability vector (already re-normalised to sum to 1).
///
/// Uses a `Vec<bool>` bitmask instead of `HashSet<usize>` to track kept tokens:
/// O(1) lookup vs O(1) amortised hash, but with much lower constant factor and
/// no heap allocation for the hash table.  The input `Vec` is reused in-place to
/// avoid a second heap allocation for the output.
fn apply_top_p(mut probs: Vec<f32>, top_p: f32) -> Vec<f32> {
    // Sort indices by probability descending (probs is only read here).
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Walk the sorted indices, accumulating probability mass until >= top_p.
    // Record kept positions in a flat bool bitmask (cache-line friendly).
    let mut keep = vec![false; probs.len()];
    let mut cumsum = 0.0_f32;
    for &idx in &indices {
        keep[idx] = true;
        cumsum += probs[idx];
        if cumsum >= top_p {
            break;
        }
    }

    // Zero non-kept entries and collect the new sum — single pass, in-place.
    let mut sum = 0.0_f32;
    for (p, &k) in probs.iter_mut().zip(keep.iter()) {
        if !k {
            *p = 0.0;
        } else {
            sum += *p;
        }
    }

    // Re-normalise with a multiply (cheaper than divide on modern CPUs).
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        probs.iter_mut().for_each(|p| *p *= inv_sum);
    }
    probs
}

/// Draw a sample from a discrete distribution via the inverse-CDF method.
///
/// Uses `rand::thread_rng()` for proper stochastic sampling.  The previous
/// `DefaultHasher`-based approach was deterministic within a run (same input
/// logits always produced the same token), which defeated the purpose of
/// temperature / top-p / top-k sampling.
fn multinomial_sample(probs: &[f32]) -> usize {
    use rand::Rng;
    let rand_val: f32 = rand::thread_rng().gen();
    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val <= cumsum {
            return i;
        }
    }
    probs.len() - 1 // fallback: numerical precision guard
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
        let p = SamplingParams {
            top_p: 0.0,
            ..SamplingParams::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_top_p_gt_1() {
        let p = SamplingParams {
            top_p: 1.1,
            ..SamplingParams::default()
        };
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
        let params = SamplingParams {
            top_p: 0.01,
            ..SamplingParams::default()
        };
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
