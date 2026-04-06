#![allow(dead_code)]
/// Token sampler — greedy, temperature, top-k, and nucleus (top-p) sampling.
///
/// All methods are stateless pure functions operating on logit slices so they
/// can be called from any thread without locking.
use anyhow::{bail, Result};
use std::cell::RefCell;

// ── Thread-local scratch buffers ──────────────────────────────────────────
//
// LLM decode calls `sample()` once per generated token.  The naive
// implementation allocates 3–4 `Vec<f32>` per call (scores, softmax result,
// top-k scratch, top-p indices + keep mask).  For a 32k vocabulary that is
// ~500 kB of heap traffic per token; at 100 tokens/s that is 50 MB/s of
// malloc pressure.
//
// Instead, each thread keeps a single reusable workspace.  `with_scratch` is
// the only entry point — it hands a mutable reference that is guaranteed to be
// reclaimed even on early returns / `?` propagation.

thread_local! {
    /// Temperature-scaled logits → top-k-masked scores → softmax probabilities.
    static SCORES: RefCell<Vec<f32>> = RefCell::new(Vec::with_capacity(32_768));
    /// Scratch copy used by `apply_top_k` to find the k-th-largest threshold.
    static TOP_K_SCRATCH: RefCell<Vec<f32>> = RefCell::new(Vec::with_capacity(32_768));
    /// Sorted index list used by `apply_top_p`.
    static TOP_P_IDX: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(32_768));
    /// Boolean keep-mask used by `apply_top_p`.
    static TOP_P_KEEP: RefCell<Vec<bool>> = RefCell::new(Vec::with_capacity(32_768));
}

/// Borrow a thread-local `Vec<f32>` scratch buffer, clear it, run `f`, then
/// return the borrow.  The buffer retains its capacity across calls so future
/// `extend`/`push` operations never reallocate after the first warm-up call.
#[inline]
fn with_f32_scratch<T>(tls: &'static std::thread::LocalKey<RefCell<Vec<f32>>>, f: impl FnOnce(&mut Vec<f32>) -> T) -> T {
    tls.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        f(&mut buf)
    })
}

#[inline]
fn with_usize_scratch<T>(tls: &'static std::thread::LocalKey<RefCell<Vec<usize>>>, f: impl FnOnce(&mut Vec<usize>) -> T) -> T {
    tls.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        f(&mut buf)
    })
}

#[inline]
fn with_bool_scratch<T>(tls: &'static std::thread::LocalKey<RefCell<Vec<bool>>>, f: impl FnOnce(&mut Vec<bool>) -> T) -> T {
    tls.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        f(&mut buf)
    })
}

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

    // Apply temperature into a thread-local scratch buffer (no heap alloc).
    let result = SCORES.with(|cell| {
        let mut scores = cell.borrow_mut();
        scores.clear();
        scores.extend(logits.iter().map(|&l| l / params.temperature));

        // Top-k filtering (in-place on scores).
        if params.top_k > 0 && params.top_k < scores.len() {
            apply_top_k(&mut scores, params.top_k);
        }

        // Softmax in-place (no extra allocation).
        softmax_inplace(&mut scores);

        // Top-p: returns the token index directly, consuming scores in-place.
        let token = if params.top_p < 1.0 {
            apply_top_p_sample(&scores, params.top_p)
        } else {
            multinomial_sample(&scores)
        };
        token as u32
    });

    Ok(result)
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
/// Kept as a public API for callers outside the hot `sample()` path
/// (e.g. speculative decoding's rejection step which needs an owned Vec).
/// Internal `sample()` uses `softmax_inplace` to avoid allocation.
pub fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut result: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = result.iter().sum();
    let inv_sum = 1.0 / sum;
    result.iter_mut().for_each(|e| *e *= inv_sum);
    result
}

/// In-place numerically stable softmax — zero allocations.
///
/// Normalises `v` in-place: after return `v[i] = exp(v[i]-max) / sum_j exp(v[j]-max)`.
#[inline]
pub fn softmax_inplace(v: &mut [f32]) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv_sum = 1.0 / sum;
    v.iter_mut().for_each(|x| *x *= inv_sum);
}

/// Zero out all but the top-k logits (the rest become -inf).
///
/// Uses `select_nth_unstable_by` — O(n) average (Floyd-Rivest / introselect)
/// instead of a full O(n log n) sort.  For typical LLM vocab sizes (32k–128k)
/// this is ~10–15× faster than the sort-based approach.
///
/// The scratch copy needed for partitioning is taken from a thread-local buffer
/// so no heap allocation occurs on the hot path.
fn apply_top_k(scores: &mut [f32], k: usize) {
    let threshold = with_f32_scratch(&TOP_K_SCRATCH, |scratch| {
        scratch.extend_from_slice(scores);
        scratch.select_nth_unstable_by(k - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        scratch[k - 1]
    });

    let mut kept = 0;
    for s in scores.iter_mut() {
        if *s >= threshold && kept < k {
            kept += 1;
        } else {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// Keep the smallest set of tokens whose cumulative probability >= top_p,
/// then draw a multinomial sample from the re-normalised distribution.
///
/// Returns the sampled token index directly to avoid a second pass and an
/// extra allocation.  Scratch index and bool-mask Vecs come from TLS buffers.
fn apply_top_p_sample(probs: &[f32], top_p: f32) -> usize {
    // Build a sorted-by-probability index list in a TLS scratch Vec.
    let token = with_usize_scratch(&TOP_P_IDX, |indices| {
        indices.extend(0..probs.len());
        indices.sort_unstable_by(|&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Walk sorted indices accumulating probability until >= top_p.
        // Build a bool-mask from a TLS Vec to avoid another allocation.
        with_bool_scratch(&TOP_P_KEEP, |keep| {
            keep.resize(probs.len(), false);
            let mut cumsum = 0.0_f32;
            for &idx in indices.iter() {
                keep[idx] = true;
                cumsum += probs[idx];
                if cumsum >= top_p {
                    break;
                }
            }

            // Normalise and sample in one pass.
            let sum: f32 = probs.iter().zip(keep.iter()).filter(|(_, &k)| k).map(|(&p, _)| p).sum();
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };

            use rand::Rng;
            let rand_val: f32 = rand::thread_rng().gen();
            let mut cumsum2 = 0.0_f32;
            for (i, (&p, &k)) in probs.iter().zip(keep.iter()).enumerate() {
                if k {
                    cumsum2 += p * inv_sum;
                    if rand_val <= cumsum2 {
                        return i;
                    }
                }
            }
            probs.len() - 1
        })
    });
    token
}

/// Keep the smallest set of tokens whose cumulative probability >= top_p.
/// Returns a new probability vector (already re-normalised to sum to 1).
///
/// **This allocating variant is kept for tests and external callers.**
/// Hot-path `sample()` uses `apply_top_p_sample` instead.
#[cfg(test)]
fn apply_top_p(mut probs: Vec<f32>, top_p: f32) -> Vec<f32> {
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![false; probs.len()];
    let mut cumsum = 0.0_f32;
    for &idx in &indices {
        keep[idx] = true;
        cumsum += probs[idx];
        if cumsum >= top_p {
            break;
        }
    }

    let mut sum = 0.0_f32;
    for (p, &k) in probs.iter_mut().zip(keep.iter()) {
        if !k {
            *p = 0.0;
        } else {
            sum += *p;
        }
    }

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
