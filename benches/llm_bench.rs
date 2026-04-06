//! LLM inference benchmarks: sampler throughput (argmax, softmax) and
//! batch token generation across batch sizes 1, 4, 8.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

/// Argmax over a logit vector — the greedy decoding hot-path.
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Softmax in-place — used before top-k/top-p sampling.
fn softmax(logits: &mut [f32]) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in logits.iter_mut() {
        *x /= sum;
    }
}

/// Simulate generating `num_tokens` tokens greedily for a single sequence.
fn greedy_generate(vocab_size: usize, num_tokens: usize) -> Vec<usize> {
    let mut logits: Vec<f32> = (0..vocab_size).map(|i| i as f32 * 0.001).collect();
    let mut tokens = Vec::with_capacity(num_tokens);
    for step in 0..num_tokens {
        // Perturb logits slightly to simulate different steps
        logits[step % vocab_size] += 0.1;
        tokens.push(argmax(&logits));
    }
    tokens
}

fn bench_argmax_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("llm_argmax");
    group.measurement_time(Duration::from_secs(5));

    for vocab_size in [32_000usize, 50_257, 128_256] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| i as f32 * 0.001).collect();
        group.bench_with_input(
            BenchmarkId::new("vocab_size", vocab_size),
            &logits,
            |b, l| {
                b.iter(|| argmax(l));
            },
        );
    }
    group.finish();
}

fn bench_softmax_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("llm_softmax");
    group.measurement_time(Duration::from_secs(5));

    for vocab_size in [32_000usize, 50_257, 128_256] {
        group.bench_with_input(
            BenchmarkId::new("vocab_size", vocab_size),
            &vocab_size,
            |b, &vs| {
                b.iter(|| {
                    let mut logits: Vec<f32> = (0..vs).map(|i| i as f32 * 0.001).collect();
                    softmax(&mut logits);
                    logits[0]
                });
            },
        );
    }
    group.finish();
}

fn bench_batch_token_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("llm_batch_generation");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    let vocab_size = 32_000;
    let num_tokens = 32;

    for batch_size in [1usize, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    (0..bs)
                        .map(|_| greedy_generate(vocab_size, num_tokens).len())
                        .sum::<usize>()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    llm_benches,
    bench_argmax_throughput,
    bench_softmax_throughput,
    bench_batch_token_generation
);
criterion_main!(llm_benches);
