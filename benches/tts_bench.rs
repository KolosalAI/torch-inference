//! TTS pipeline benchmarks.
//!
//! Covers:
//! - Sentence splitting latency across short/medium/long inputs
//! - Audio chunk accumulation and WAV header construction
//! - TTFA (time-to-first-audio) simulation via mpsc channel timing
//! - Voice parameter serialisation overhead
//!
//! Run: cargo bench --bench tts_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Helpers mirroring TTS pipeline internals
// ---------------------------------------------------------------------------

/// Split text into sentence chunks the same way the TTS streaming handler does.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    for (i, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?') {
            let s = text[start..=i].trim();
            if !s.is_empty() {
                sentences.push(s);
            }
            start = i + ch.len_utf8();
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

/// Simulate per-sentence PCM encoding cost (proportional to word count).
/// 22 050 Hz × 2 bytes/sample × 0.12 s/word ≈ 5 292 bytes/word.
fn encode_sentence(sentence: &str) -> Vec<u8> {
    let words = sentence.split_whitespace().count().max(1);
    vec![0u8; words * 5_292]
}

/// Build a minimal WAV header + body from raw PCM bytes.
fn build_wav(pcm: &[u8], sample_rate: u32, channels: u16) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let mut wav = Vec::with_capacity(44 + pcm.len());
    let byte_rate = sample_rate * u32::from(channels) * 2;
    let block_align = channels * 2;

    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());   // PCM
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());  // bits/sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_sentence_split(c: &mut Criterion) {
    let inputs = [
        ("short_1s",   "Hello world."),
        ("medium_5s",  "The weather today is sunny with a high of 24 degrees. Pack light if you're heading out. Don't forget sunscreen!"),
        ("long_30s",   "Sentence-level streaming synthesis breaks long text into individual sentences before synthesis begins. \
                        This reduces the time-to-first-audio dramatically. \
                        Each sentence is synthesised independently. \
                        The resulting audio chunks are streamed back to the client as soon as they are ready. \
                        This approach works especially well for dialogue and narration use-cases. \
                        It also allows early cancellation if the user interrupts playback. \
                        The trade-off is slightly different prosody at sentence boundaries compared to full-text synthesis."),
    ];

    let mut group = c.benchmark_group("tts_sentence_split");
    for (name, text) in inputs {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(split_sentences(black_box(text))));
        });
    }
    group.finish();
}

fn bench_wav_build(c: &mut Criterion) {
    let durations_ms = [500u32, 1_000, 5_000, 10_000];
    let mut group = c.benchmark_group("tts_wav_build");
    for ms in durations_ms {
        let samples = (22_050 * ms / 1_000) as usize;
        let pcm: Vec<u8> = (0..samples * 2).map(|i| (i % 256) as u8).collect();
        group.throughput(Throughput::Bytes(pcm.len() as u64));
        group.bench_with_input(BenchmarkId::new("mono_22k", ms), &pcm, |b, pcm| {
            b.iter(|| black_box(build_wav(black_box(pcm), 22_050, 1)));
        });
    }
    group.finish();
}

fn bench_ttfa_channel(c: &mut Criterion) {
    let text = "The quick brown fox jumps over the lazy dog. \
                Pack my box with five dozen liquor jugs. \
                How vexingly quick daft zebras jump.";

    let mut group = c.benchmark_group("tts_ttfa_channel");
    group.measurement_time(Duration::from_secs(10));

    for buf_size in [1usize, 4, 16] {
        group.bench_with_input(BenchmarkId::new("buf", buf_size), &buf_size, |b, &buf| {
            let rt = Runtime::new().unwrap();
            b.iter(|| {
                rt.block_on(async {
                    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(buf);
                    let sentences: Vec<String> =
                        split_sentences(text).iter().map(|s| s.to_string()).collect();
                    let tx = Arc::new(tx);

                    let tx2 = Arc::clone(&tx);
                    tokio::spawn(async move {
                        for s in sentences {
                            let chunk = encode_sentence(&s);
                            let _ = tx2.send(chunk).await;
                        }
                    });

                    // Time-to-first-audio: receive only the first chunk.
                    black_box(rx.recv().await)
                })
            });
        });
    }
    group.finish();
}

fn bench_audio_accumulation(c: &mut Criterion) {
    let sentence_count = [3usize, 10, 30];
    let mut group = c.benchmark_group("tts_audio_accumulation");

    for &n in &sentence_count {
        let chunks: Vec<Vec<u8>> = (0..n)
            .map(|i| vec![((i * 17) % 256) as u8; 22_050 * 2 / n])
            .collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("chunks", n), &chunks, |b, chunks| {
            b.iter(|| {
                let total: usize = chunks.iter().map(|c| c.len()).sum();
                let mut buf = Vec::with_capacity(total);
                for chunk in chunks {
                    buf.extend_from_slice(chunk);
                }
                black_box(buf)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sentence_split,
    bench_wav_build,
    bench_ttfa_channel,
    bench_audio_accumulation,
);
criterion_main!(benches);
