//! Audio / STT preprocessing benchmarks.
//!
//! Covers:
//! - WAV decode and metadata extraction
//! - Linear-interpolation resampling (e.g. 44 100 Hz → 16 000 Hz for Whisper)
//! - Audio normalization (peak and RMS)
//! - Spectrogram feature extraction (mel filterbank simulation)
//! - Mono downmix from stereo
//!
//! Run: cargo bench --bench audio_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Audio helpers (mirrors core/audio.rs logic without the ORT dependency)
// ---------------------------------------------------------------------------

/// Generate a synthetic sine-wave PCM signal (i16, little-endian).
fn synthetic_pcm_i16(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> Vec<i16> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (f32::sin(2.0 * std::f32::consts::PI * freq_hz * t) * i16::MAX as f32) as i16
        })
        .collect()
}

/// Normalize i16 PCM to f32 in [-1.0, 1.0].
fn pcm_i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| s as f32 / i16::MAX as f32).collect()
}

/// Peak-normalise f32 audio to [-1.0, 1.0].
fn peak_normalize(samples: &mut Vec<f32>) {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        samples.iter_mut().for_each(|s| *s /= peak);
    }
}

/// RMS-normalise f32 audio to a target RMS level.
fn rms_normalize(samples: &mut Vec<f32>, target_rms: f32) {
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    if rms > 0.0 {
        let gain = target_rms / rms;
        samples.iter_mut().for_each(|s| *s *= gain);
    }
}

/// Linear-interpolation resample from `src_rate` to `dst_rate`.
fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate {
        return samples.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = (pos - idx as f64) as f32;
        let a = *samples.get(idx).unwrap_or(&0.0);
        let b = *samples.get(idx + 1).unwrap_or(&a);
        out.push(a + frac * (b - a));
    }
    out
}

/// Downmix stereo interleaved to mono by averaging channels.
fn stereo_to_mono(samples: &[f32]) -> Vec<f32> {
    samples.chunks_exact(2).map(|c| (c[0] + c[1]) * 0.5).collect()
}

/// Compute a simple energy-based mel-like spectrogram frame (no FFT — just
/// energy per band using overlapping windows). Used as a preprocessing cost proxy.
fn energy_frames(samples: &[f32], frame_len: usize, hop: usize, n_bands: usize) -> Vec<Vec<f32>> {
    let band_size = (frame_len / 2) / n_bands;
    samples
        .windows(frame_len)
        .step_by(hop)
        .map(|frame| {
            (0..n_bands)
                .map(|b| {
                    frame[b * band_size..(b + 1) * band_size]
                        .iter()
                        .map(|s| s * s)
                        .sum::<f32>()
                        / band_size as f32
                })
                .collect()
        })
        .collect()
}

/// Build a minimal 44-byte WAV header for the given PCM parameters.
fn build_wav_header(sample_rate: u32, channels: u16, num_samples: u32) -> [u8; 44] {
    let data_size = num_samples * u32::from(channels) * 2;
    let byte_rate = sample_rate * u32::from(channels) * 2;
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&(36 + data_size).to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes());
    h[22..24].copy_from_slice(&channels.to_le_bytes());
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&(channels * 2).to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes());
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_size.to_le_bytes());
    h
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_resample(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_resample");
    group.measurement_time(Duration::from_secs(8));

    // Whisper expects 16 kHz mono — typical input is 44.1 kHz or 48 kHz stereo.
    let cases: &[(&str, u32, u32, f32)] = &[
        ("44k_to_16k_1s",  44_100, 16_000, 1.0),
        ("48k_to_16k_1s",  48_000, 16_000, 1.0),
        ("44k_to_16k_5s",  44_100, 16_000, 5.0),
        ("44k_to_22k_5s",  44_100, 22_050, 5.0),
    ];

    for &(name, src, dst, secs) in cases {
        let samples = pcm_i16_to_f32(&synthetic_pcm_i16(440.0, src, secs));
        group.throughput(Throughput::Elements(samples.len() as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(resample_linear(black_box(&samples), src, dst)));
        });
    }
    group.finish();
}

fn bench_normalize(c: &mut Criterion) {
    let sizes = [16_000usize, 80_000, 480_000]; // 1s, 5s, 30s at 16 kHz
    let mut group = c.benchmark_group("audio_normalize");

    for &n in &sizes {
        let base: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 2.0 - 1.0).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("peak", n), &base, |b, buf| {
            b.iter(|| { let mut v = buf.clone(); peak_normalize(&mut v); black_box(v) });
        });
        group.bench_with_input(BenchmarkId::new("rms", n), &base, |b, buf| {
            b.iter(|| { let mut v = buf.clone(); rms_normalize(&mut v, 0.1); black_box(v) });
        });
    }
    group.finish();
}

fn bench_stereo_to_mono(c: &mut Criterion) {
    let durations = [1.0f32, 5.0, 30.0];
    let mut group = c.benchmark_group("audio_stereo_to_mono");

    for &secs in &durations {
        let stereo_l = pcm_i16_to_f32(&synthetic_pcm_i16(440.0, 44_100, secs));
        let stereo_r = pcm_i16_to_f32(&synthetic_pcm_i16(880.0, 44_100, secs));
        // Interleave L/R
        let interleaved: Vec<f32> = stereo_l
            .iter()
            .zip(stereo_r.iter())
            .flat_map(|(&l, &r)| [l, r])
            .collect();

        group.throughput(Throughput::Elements(interleaved.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("stereo_44k", (secs * 1000.0) as u64),
            &interleaved,
            |b, samples| {
                b.iter(|| black_box(stereo_to_mono(black_box(samples))));
            },
        );
    }
    group.finish();
}

fn bench_pcm_convert(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_pcm_i16_to_f32");

    for &secs in &[1.0f32, 5.0, 30.0] {
        let pcm = synthetic_pcm_i16(440.0, 16_000, secs);
        group.throughput(Throughput::Elements(pcm.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("16k_mono", (secs * 1000.0) as u64),
            &pcm,
            |b, pcm| {
                b.iter(|| black_box(pcm_i16_to_f32(black_box(pcm))));
            },
        );
    }
    group.finish();
}

fn bench_energy_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_energy_spectrogram");
    group.measurement_time(Duration::from_secs(10));

    // Whisper-style: 16 kHz, 25ms frames (400 samples), 10ms hop (160 samples), 80 bands
    let samples = pcm_i16_to_f32(&synthetic_pcm_i16(440.0, 16_000, 5.0));
    group.throughput(Throughput::Elements(samples.len() as u64));
    group.bench_function("whisper_style_5s", |b| {
        b.iter(|| black_box(energy_frames(black_box(&samples), 400, 160, 80)));
    });
    group.finish();
}

fn bench_wav_header(c: &mut Criterion) {
    c.bench_function("audio_wav_header_build", |b| {
        b.iter(|| black_box(build_wav_header(black_box(16_000), black_box(1), black_box(80_000))));
    });
}

criterion_group!(
    benches,
    bench_resample,
    bench_normalize,
    bench_stereo_to_mono,
    bench_pcm_convert,
    bench_energy_spectrogram,
    bench_wav_header,
);
criterion_main!(benches);
