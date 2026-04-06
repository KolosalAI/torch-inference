#![allow(dead_code)]
/// Speculative decoding — draft-model proposes K tokens, target verifies in one pass.
///
/// # Algorithm
///
/// 1. Draft model proposes `K` candidate tokens `[d_1, …, d_K]` autoregressively.
/// 2. Target model runs one forward pass over the original context +
///    the K draft tokens, producing K+1 logit distributions.
/// 3. For each position i (1..=K), compute acceptance ratio
///    `α_i = min(1, p_target(d_i) / p_draft(d_i))`.
///    - Accept with probability `α_i`.
///    - On first rejection at position j, resample token j from an adjusted
///      distribution and discard positions j+1..K.
/// 4. The final token is always sampled from position K+1 of the target.
///
/// In the best case all K draft tokens are accepted → K+1 tokens generated
/// per target forward pass instead of 1 → theoretical ~K-fold speedup
/// (bounded by draft latency).
use anyhow::Result;

use crate::core::llm::sampler::{sample, sample_from_probs, softmax, SamplingParams};

// ── Traits ────────────────────────────────────────────────────────────────

/// A draft model that quickly proposes candidate tokens.
/// Typically a smaller, cheaper model (e.g. 1B vs 8B parameters).
pub trait DraftModel: Send + Sync {
    /// Propose `num_speculative` candidate tokens given the current `context`.
    /// Returns `(tokens, log_probs_per_token)` where `log_probs` has shape
    /// `[num_speculative, vocab_size]`.
    fn propose(&self, context: &[u32], num_speculative: usize)
        -> Result<(Vec<u32>, Vec<Vec<f32>>)>;
}

/// A target model that evaluates sequences.
pub trait TargetModel: Send + Sync {
    /// Run a single forward pass on `tokens` and return logits for each
    /// position: shape `[tokens.len(), vocab_size]`.
    fn forward(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>>;
}

// ── Speculative decoder ───────────────────────────────────────────────────

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpecConfig {
    /// Number of draft tokens to propose per step.
    pub num_speculative: usize,
    /// Sampling params used for the final (bonus) target token.
    pub sampling: SamplingParams,
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            num_speculative: 5,
            sampling: SamplingParams::greedy(),
        }
    }
}

/// Output of one speculative decoding step.
pub struct SpecStepOutput {
    /// Accepted (and possibly corrected) tokens from this step.
    pub tokens: Vec<u32>,
    /// How many of the K draft tokens were accepted (before the bonus token).
    pub num_accepted: usize,
}

/// Speculative decoder that orchestrates draft and target models.
pub struct SpeculativeDecoder {
    config: SpecConfig,
}

impl SpeculativeDecoder {
    pub fn new(config: SpecConfig) -> Self {
        Self { config }
    }

    /// Run one speculative decoding step, extending `context` by at least one
    /// token.
    ///
    /// Returns the accepted output tokens (1 to K+1 tokens).
    pub fn step(
        &self,
        context: &[u32],
        draft: &dyn DraftModel,
        target: &dyn TargetModel,
    ) -> Result<SpecStepOutput> {
        let k = self.config.num_speculative;

        // 1. Draft proposes K tokens.
        let (draft_tokens, draft_probs) = draft.propose(context, k)?;

        // 2. Target evaluates context + draft tokens (K+1 positions).
        let mut eval_tokens: Vec<u32> = context.to_vec();
        eval_tokens.extend_from_slice(&draft_tokens);
        let target_logits = target.forward(&eval_tokens)?;
        // We need positions [context.len()-1 .. context.len()-1+K+1] if
        // the model returns logits for all positions; here we take the last K+1.
        let relevant = &target_logits[context.len().saturating_sub(1)..];

        // 3. Accept / reject each draft token.
        let mut accepted_tokens: Vec<u32> = Vec::with_capacity(draft_tokens.len());
        let mut num_accepted = 0;

        for (i, &dt) in draft_tokens.iter().enumerate() {
            // target_probs at position i (predicting token after context+draft[0..i])
            let target_pos_logits = if i < relevant.len() {
                &relevant[i]
            } else {
                break;
            };
            let target_probs = softmax(target_pos_logits);
            let draft_pos_probs = softmax(&draft_probs[i]);

            let p_target = *target_probs.get(dt as usize).unwrap_or(&0.0);
            let p_draft = *draft_pos_probs.get(dt as usize).unwrap_or(&1e-9_f32);

            let alpha = (p_target / p_draft.max(1e-9)).min(1.0);

            // Stochastic acceptance.
            let accept = acceptance_roll(alpha);
            if accept {
                accepted_tokens.push(dt);
                num_accepted += 1;
            } else {
                // Rejection: sample corrected token from the adjusted distribution.
                // `sample_from_probs` does a direct multinomial draw without any
                // logits_from_probs → softmax roundtrip (which introduced numerical
                // imprecision and allocated an extra Vec).
                let adjusted = adjusted_distribution(&target_probs, &draft_pos_probs);
                let corrected = sample_from_probs(&adjusted) as u32;
                accepted_tokens.push(corrected);
                break; // Discard remaining draft tokens.
            }
        }

        // 4. Bonus token from the last target position.
        if let Some(bonus_logits) = relevant.get(num_accepted) {
            let bonus = sample(bonus_logits, &self.config.sampling)?;
            accepted_tokens.push(bonus);
        }

        Ok(SpecStepOutput {
            tokens: accepted_tokens,
            num_accepted,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Stochastic acceptance roll.
///
/// The previous implementation seeded `DefaultHasher` with the loop position
/// `i`, making acceptance/rejection purely deterministic — the same draft
/// token at position 0 was *always* accepted or *always* rejected regardless
/// of `alpha`.  A real uniform draw is required for the algorithm to converge
/// to the same distribution as the target model.
fn acceptance_roll(alpha: f32) -> bool {
    use rand::Rng;
    rand::thread_rng().gen::<f32>() < alpha
}

/// Adjusted distribution for the rejection case: max(0, p_target − p_draft),
/// re-normalised.
fn adjusted_distribution(p_target: &[f32], p_draft: &[f32]) -> Vec<f32> {
    let mut adj: Vec<f32> = p_target
        .iter()
        .zip(p_draft.iter())
        .map(|(&pt, &pd)| (pt - pd).max(0.0))
        .collect();
    let sum: f32 = adj.iter().sum();
    if sum > 1e-9 {
        let inv = 1.0 / sum;
        adj.iter_mut().for_each(|v| *v *= inv);
    } else {
        // Degenerate case: fall back to uniform.
        let n = adj.len();
        adj.iter_mut().for_each(|v| *v = 1.0 / n as f32);
    }
    adj
}

/// Convert probabilities to "logit-like" values for passing to `sample`.
/// Since `sample` applies softmax, using log(p) gives back the original
/// distribution (softmax of log(p) ≈ p for well-formed distributions).
fn logits_from_probs(probs: &[f32]) -> Vec<f32> {
    probs.iter().map(|&p| p.max(1e-45).ln()).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Mock models ───────────────────────────────────────────────────────

    const VOCAB: usize = 8;

    /// Always proposes token `0` with 100 % probability.
    struct MockDraftAlwaysZero;
    impl DraftModel for MockDraftAlwaysZero {
        fn propose(&self, _ctx: &[u32], n: usize) -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
            let tokens = vec![0u32; n];
            let mut probs = vec![vec![0.0f32; VOCAB]; n];
            for row in &mut probs {
                row[0] = 1.0;
            }
            Ok((tokens, probs))
        }
    }

    /// Target always returns uniform distribution (no preference).
    struct MockTargetUniform;
    impl TargetModel for MockTargetUniform {
        fn forward(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![1.0 / VOCAB as f32; VOCAB]; tokens.len()])
        }
    }

    /// Target always prefers token `0` (matches draft perfectly).
    struct MockTargetPerfectMatch;
    impl TargetModel for MockTargetPerfectMatch {
        fn forward(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
            let mut logits = vec![vec![-100.0f32; VOCAB]; tokens.len()];
            for row in &mut logits {
                row[0] = 100.0;
            }
            Ok(logits)
        }
    }

    /// Target always prefers token `1` (mismatches draft token 0).
    struct MockTargetAlwaysOne;
    impl TargetModel for MockTargetAlwaysOne {
        fn forward(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
            let mut logits = vec![vec![-100.0f32; VOCAB]; tokens.len()];
            for row in &mut logits {
                row[1] = 100.0;
            }
            Ok(logits)
        }
    }

    // ── SpecConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_spec_config_default() {
        let cfg = SpecConfig::default();
        assert_eq!(cfg.num_speculative, 5);
    }

    #[test]
    fn test_spec_config_clone() {
        let cfg = SpecConfig::default();
        let c2 = cfg.clone();
        assert_eq!(c2.num_speculative, 5);
    }

    // ── SpecStepOutput ────────────────────────────────────────────────────

    #[test]
    fn test_spec_step_output_fields() {
        let out = SpecStepOutput {
            tokens: vec![1, 2],
            num_accepted: 1,
        };
        assert_eq!(out.tokens.len(), 2);
        assert_eq!(out.num_accepted, 1);
    }

    // ── acceptance_roll ───────────────────────────────────────────────────

    #[test]
    fn test_acceptance_roll_always_rejects_zero_alpha() {
        assert!(!acceptance_roll(0.0));
        assert!(!acceptance_roll(0.0));
    }

    #[test]
    fn test_acceptance_roll_always_accepts_one_alpha() {
        // alpha = 1.0 → any u < 1.0 accepts
        assert!(acceptance_roll(1.0));
        assert!(acceptance_roll(1.0));
    }

    // ── adjusted_distribution ─────────────────────────────────────────────

    #[test]
    fn test_adjusted_distribution_sums_to_one() {
        let pt = vec![0.3, 0.5, 0.2];
        let pd = vec![0.1, 0.6, 0.3];
        let adj = adjusted_distribution(&pt, &pd);
        let sum: f32 = adj.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={}", sum);
    }

    #[test]
    fn test_adjusted_distribution_no_negatives() {
        let pt = vec![0.1, 0.9];
        let pd = vec![0.8, 0.2];
        let adj = adjusted_distribution(&pt, &pd);
        assert!(adj.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_adjusted_distribution_degenerate_fallback() {
        // When p_target = p_draft everywhere, result should be uniform.
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let adj = adjusted_distribution(&p, &p);
        // All zeroes → uniform fallback
        let sum: f32 = adj.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    // ── logits_from_probs ─────────────────────────────────────────────────

    #[test]
    fn test_logits_from_probs_recovers_distribution() {
        let probs = vec![0.1f32, 0.6, 0.3];
        let logits = logits_from_probs(&probs);
        let recovered = softmax(&logits);
        for (a, b) in probs.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "{} vs {}", a, b);
        }
    }

    // ── SpeculativeDecoder::step ──────────────────────────────────────────

    #[test]
    fn test_step_perfect_match_accepts_all_k_tokens() {
        // Draft proposes token 0, target strongly prefers token 0 → all accepted.
        let dec = SpeculativeDecoder::new(SpecConfig {
            num_speculative: 3,
            sampling: SamplingParams::greedy(),
        });
        let out = dec
            .step(&[10u32, 20], &MockDraftAlwaysZero, &MockTargetPerfectMatch)
            .unwrap();
        // All 3 draft tokens accepted + 1 bonus = 4 tokens
        assert_eq!(out.num_accepted, 3);
        assert_eq!(out.tokens.len(), 4);
        // All should be token 0 (draft agrees with target).
        assert!(out.tokens.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_step_mismatch_rejects_first_draft_token() {
        // Draft proposes 0, target prefers 1 → first token rejected.
        let dec = SpeculativeDecoder::new(SpecConfig {
            num_speculative: 4,
            sampling: SamplingParams::greedy(),
        });
        let out = dec
            .step(&[5u32], &MockDraftAlwaysZero, &MockTargetAlwaysOne)
            .unwrap();
        // 0 draft tokens accepted; corrected token is 1; bonus token is 1.
        assert_eq!(out.num_accepted, 0);
        // Should emit: [corrected(1), (maybe bonus)], so at least 1 token.
        assert!(!out.tokens.is_empty());
    }

    #[test]
    fn test_step_returns_at_least_one_token() {
        let dec = SpeculativeDecoder::new(SpecConfig::default());
        let out = dec
            .step(&[1u32], &MockDraftAlwaysZero, &MockTargetUniform)
            .unwrap();
        assert!(!out.tokens.is_empty());
    }

    #[test]
    fn test_step_tokens_in_valid_range() {
        let dec = SpeculativeDecoder::new(SpecConfig {
            num_speculative: 3,
            sampling: SamplingParams::greedy(),
        });
        let out = dec
            .step(&[1u32, 2, 3], &MockDraftAlwaysZero, &MockTargetUniform)
            .unwrap();
        for &t in &out.tokens {
            assert!((t as usize) < VOCAB, "token {} out of vocab", t);
        }
    }
}
