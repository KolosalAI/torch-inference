//! TTS streaming benchmarks: sentence-split cost and mpsc channel overhead.
//! Compares time-to-first-chunk (TTFA) simulation vs full-synthesis accumulation.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use tokio::sync::mpsc;

/// Simulate splitting text into sentences (the split done before streaming).
fn split_sentences(text: &str) -> Vec<&str> {
    text.split(|c| c == '.' || c == '!' || c == '?')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// Simulate per-sentence audio encoding cost (just allocates bytes proportional to length).
fn encode_sentence(sentence: &str) -> Vec<u8> {
    // Approx 22050 Hz * 2 bytes/sample * 0.1s per word / 5 chars per word
    let words = sentence.split_whitespace().count().max(1);
    vec![0u8; words * 882]
}

fn bench_sentence_split(c: &mut Criterion) {
    let short = "Hello world. How are you?";
    let medium = "The quick brown fox jumps over the lazy dog. \
                  It was a dark and stormy night. \
                  All happy families are alike. \
                  Call me Ishmael. \
                  It is a truth universally acknowledged.";
    let long_text: String =
        std::iter::repeat("The model processed the input and returned a confident prediction. ")
            .take(40)
            .collect();

    let mut group = c.benchmark_group("tts_sentence_split");
    group.measurement_time(Duration::from_secs(3));

    for (label, text) in [("short_10w", short), ("medium_50w", medium)] {
        group.bench_with_input(BenchmarkId::new("text", label), text, |b, t| {
            b.iter(|| split_sentences(t));
        });
    }
    group.bench_with_input(
        BenchmarkId::new("text", "long_200w"),
        long_text.as_str(),
        |b, t| {
            b.iter(|| split_sentences(t));
        },
    );
    group.finish();
}

fn bench_streaming_channel_overhead(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let sentences = vec![
        "Hello world.",
        "This is a test sentence.",
        "And another one here.",
    ];

    let mut group = c.benchmark_group("tts_channel_overhead");
    group.measurement_time(Duration::from_secs(5));

    // Streaming: send each sentence chunk through mpsc as it's encoded
    group.bench_function("streaming_mpsc", |b| {
        b.iter(|| {
            rt.block_on(async {
                let (tx, mut rx) = mpsc::channel::<Vec<u8>>(8);
                let sentences_clone = sentences.clone();
                tokio::spawn(async move {
                    for s in sentences_clone {
                        let chunk = encode_sentence(s);
                        let _ = tx.send(chunk).await;
                    }
                });
                let mut total = 0usize;
                while let Some(chunk) = rx.recv().await {
                    total += chunk.len();
                }
                total
            })
        });
    });

    // Accumulating: encode all sentences, collect into one buffer, return
    group.bench_function("accumulating_collect", |b| {
        b.iter(|| {
            sentences
                .iter()
                .flat_map(|s| encode_sentence(s))
                .collect::<Vec<u8>>()
                .len()
        });
    });

    group.finish();
}

criterion_group!(
    tts_benches,
    bench_sentence_split,
    bench_streaming_channel_overhead
);
criterion_main!(tts_benches);
