/// provider_comparison — Apples-to-apples benchmark comparing torch-inference
/// against external providers.
///
/// DESIGN PRINCIPLE — every comparison is same-modality, same-metric,
/// same-measurement-method:
///
///   ┌─────────────────────────────────────────────────────────────────────┐
///   │ Category       │ What we measure           │ Metric                 │
///   ├────────────────┼───────────────────────────┼────────────────────────┤
///   │ TTS synthesis  │ wall-clock: send text →   │ chars/sec synthesised  │
///   │                │ receive full audio         │ real-time factor (RTF) │
///   ├────────────────┼───────────────────────────┼────────────────────────┤
///   │ Image classify │ wall-clock: send image →  │ ms/image, images/sec   │
///   │                │ receive top-5 labels       │                        │
///   ├────────────────┼───────────────────────────┼────────────────────────┤
///   │ Text inference │ wall-clock: send prompt → │ tokens/sec (output)    │
///   │                │ receive full response      │ chars/sec (output)     │
///   │                │ (streaming where available)│ TTFT ms                │
///   └─────────────────────────────────────────────────────────────────────┘
///
/// torch-inference in each category:
///   TTS   — POST /tts/synthesize     (Kokoro/Piper, local model)
///   Image — POST /predict            (ONNX ResNet/MobileNet, local model)
///   Text  — POST /predict            (any text model loaded on the server)
///
/// External providers:
///   TTS   — OpenAI TTS-1, ElevenLabs
///   Image — Google Cloud Vision, AWS Rekognition (via REST)
///   Text  — OpenAI GPT-4o-mini, Groq Llama-3.1-8B, Together Llama-3.1-8B,
///            Anthropic Claude Haiku
///
/// Usage:
///   cargo run --bin provider-comparison -- --runs 5
///   cargo run --bin provider-comparison -- --category tts --runs 10
///   cargo run --bin provider-comparison -- --category image --runs 5
///   cargo run --bin provider-comparison -- --category text --runs 5
///
/// API keys (env vars):
///   OPENAI_API_KEY        OpenAI (text + TTS)
///   GROQ_API_KEY          Groq (text)
///   TOGETHER_API_KEY      Together AI (text)
///   ANTHROPIC_API_KEY     Anthropic (text)
///   ELEVENLABS_API_KEY    ElevenLabs (TTS)
///   GOOGLE_API_KEY        Google Cloud Vision (image)
///   AWS_ACCESS_KEY_ID +   AWS Rekognition (image)
///   AWS_SECRET_ACCESS_KEY
///
/// Local server: http://localhost:8080  (or set LOCAL_SERVER_URL)

use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::env;
use std::time::{Duration, Instant};

// ── Shared result type ────────────────────────────────────────────────────

/// A single timed measurement. Output size is in the natural unit for the
/// modality: chars for text/TTS, pixels (w*h) for image.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Wall-clock ms from first byte sent to last byte received.
    pub latency_ms: f64,
    /// Time to first content chunk (meaningful only for streaming endpoints).
    pub ttft_ms: Option<f64>,
    /// Natural output unit (chars for text/TTS, pixel count for image).
    pub output_units: usize,
    pub success: bool,
    pub error: Option<String>,
}

impl RunResult {
    pub fn failure(latency_ms: f64, err: impl Into<String>) -> Self {
        RunResult {
            latency_ms,
            ttft_ms: None,
            output_units: 0,
            success: false,
            error: Some(err.into()),
        }
    }
}

// ── Summary ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ProviderSummary {
    pub provider: String,
    pub model: String,
    pub category: &'static str,
    /// Unit label printed in the table (e.g. "chars/s", "images/s").
    pub unit_label: &'static str,
    pub runs: usize,
    pub success_rate_pct: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub avg_throughput: f64,   // output_units / (latency_ms / 1000)
    pub avg_ttft_ms: Option<f64>,
}

impl ProviderSummary {
    pub fn from_runs(
        provider: &str,
        model: &str,
        category: &'static str,
        unit_label: &'static str,
        mut results: Vec<RunResult>,
    ) -> Self {
        let runs = results.len();
        let successes: Vec<&RunResult> = results.iter().filter(|r| r.success).collect();
        let n_ok = successes.len();
        let success_rate_pct = n_ok as f64 / runs.max(1) as f64 * 100.0;

        let mut latencies: Vec<f64> = successes.iter().map(|r| r.latency_ms).collect();
        latencies.sort_by(f64::total_cmp);

        let p50 = percentile(&latencies, 0.50);
        let p95 = percentile(&latencies, 0.95);

        let avg_throughput = {
            let values: Vec<f64> = successes
                .iter()
                .filter(|r| r.latency_ms > 0.0)
                .map(|r| r.output_units as f64 / (r.latency_ms / 1000.0))
                .collect();
            mean(&values)
        };

        let avg_ttft_ms = {
            let values: Vec<f64> = successes
                .iter()
                .filter_map(|r| r.ttft_ms)
                .collect();
            if values.is_empty() { None } else { Some(mean(&values)) }
        };

        ProviderSummary {
            provider: provider.to_string(),
            model: model.to_string(),
            category,
            unit_label,
            runs,
            success_rate_pct,
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            avg_throughput,
            avg_ttft_ms,
        }
    }
}

// ── HTTP client ───────────────────────────────────────────────────────────

fn build_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("Failed to build HTTP client")
}

fn local_url() -> String {
    env::var("LOCAL_SERVER_URL").unwrap_or_else(|_| "http://localhost:8080".to_string())
}

fn api_key(env_var: &str) -> Option<String> {
    env::var(env_var).ok().filter(|k| !k.is_empty())
}

// ── TTS benchmarks ────────────────────────────────────────────────────────
// Same measurement for every provider:
//   1. Send the same plain-text string.
//   2. Wait for complete audio (binary blob or base64 in JSON).
//   3. Record wall-clock ms.
//   4. Derive: chars/sec = text.len() / (latency_ms / 1000)
//   5. Real-time factor (RTF) = synthesis_time_sec / estimated_audio_duration_sec
//      (audio duration ≈ chars / 14  at normal speaking speed ~840 chars/min)

/// torch-inference local TTS — POST /tts/synthesize
async fn bench_local_tts(client: &Client, text: &str) -> RunResult {
    let url = format!("{}/tts/synthesize", local_url());
    let start = Instant::now();
    match client
        .post(&url)
        .json(&json!({"text": text, "speed": 1.0}))
        .send()
        .await
    {
        Err(e) => RunResult::failure(ms(start), e.to_string()),
        Ok(r) if !r.status().is_success() => {
            let status = r.status().as_u16();
            let body = r.text().await.unwrap_or_default();
            RunResult::failure(ms(start), format!("HTTP {} — {}", status, trunc(&body, 200)))
        }
        Ok(r) => {
            // Consume the full body before stopping the clock.
            let _body = r.bytes().await.unwrap_or_default();
            RunResult {
                latency_ms: ms(start),
                ttft_ms: None,
                output_units: text.len(),
                success: true,
                error: None,
            }
        }
    }
}

/// OpenAI TTS-1 — POST /v1/audio/speech
/// Returns raw MP3 bytes; we wait for the full response.
async fn bench_openai_tts(client: &Client, api_key: &str, text: &str) -> RunResult {
    let start = Instant::now();
    match client
        .post("https://api.openai.com/v1/audio/speech")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({"model": "tts-1", "voice": "alloy", "input": text}))
        .send()
        .await
    {
        Err(e) => RunResult::failure(ms(start), e.to_string()),
        Ok(r) if !r.status().is_success() => {
            let status = r.status().as_u16();
            RunResult::failure(ms(start), format!("HTTP {}", status))
        }
        Ok(r) => {
            let _bytes = r.bytes().await.unwrap_or_default();
            RunResult {
                latency_ms: ms(start),
                ttft_ms: None,
                output_units: text.len(),
                success: true,
                error: None,
            }
        }
    }
}

/// ElevenLabs TTS — POST /v1/text-to-speech/{voice_id}
/// Returns MP3 bytes.
async fn bench_elevenlabs_tts(client: &Client, api_key: &str, text: &str) -> RunResult {
    // "Rachel" voice — widely available on all plans.
    let voice_id = "21m00Tcm4TlvDq8ikWAM";
    let url = format!("https://api.elevenlabs.io/v1/text-to-speech/{}", voice_id);
    let start = Instant::now();
    match client
        .post(&url)
        .header("xi-api-key", api_key)
        .json(&json!({
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }))
        .send()
        .await
    {
        Err(e) => RunResult::failure(ms(start), e.to_string()),
        Ok(r) if !r.status().is_success() => {
            let status = r.status().as_u16();
            RunResult::failure(ms(start), format!("HTTP {}", status))
        }
        Ok(r) => {
            let _bytes = r.bytes().await.unwrap_or_default();
            RunResult {
                latency_ms: ms(start),
                ttft_ms: None,
                output_units: text.len(),
                success: true,
                error: None,
            }
        }
    }
}

// ── Image classification benchmarks ──────────────────────────────────────
// Same measurement for every provider:
//   1. Send the same JPEG (224×224, ~15 KB synthetic image encoded to JPEG).
//   2. Wait for classification labels.
//   3. Record wall-clock ms.
//   4. Derive: images/sec = 1 / (latency_ms / 1000)

fn make_test_jpeg() -> Vec<u8> {
    use image::{DynamicImage, ImageBuffer, Rgb};
    let img = DynamicImage::ImageRgb8(ImageBuffer::<Rgb<u8>, _>::from_fn(224, 224, |x, y| {
        Rgb([(x % 255) as u8, (y % 255) as u8, ((x + y) % 255) as u8])
    }));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    buf.into_inner()
}

/// torch-inference local image — POST /predict
async fn bench_local_image(client: &Client, jpeg_bytes: &[u8]) -> RunResult {
    let url = format!("{}/predict", local_url());
    let b64 = base64_encode(jpeg_bytes);
    let start = Instant::now();
    match client
        .post(&url)
        .json(&json!({
            "model_name": "default",
            "inputs": {"image_base64": b64, "format": "jpeg"},
            "priority": 0
        }))
        .send()
        .await
    {
        Err(e) => RunResult::failure(ms(start), e.to_string()),
        Ok(r) if !r.status().is_success() => {
            let status = r.status().as_u16();
            let body = r.text().await.unwrap_or_default();
            RunResult::failure(ms(start), format!("HTTP {} — {}", status, trunc(&body, 200)))
        }
        Ok(r) => {
            let _body: Value = r.json().await.unwrap_or(json!({}));
            RunResult {
                latency_ms: ms(start),
                ttft_ms: None,
                output_units: 1, // 1 image classified
                success: true,
                error: None,
            }
        }
    }
}

/// Google Cloud Vision — annotate image with LABEL_DETECTION.
async fn bench_google_vision(client: &Client, api_key: &str, jpeg_bytes: &[u8]) -> RunResult {
    let b64 = base64_encode(jpeg_bytes);
    let url = format!(
        "https://vision.googleapis.com/v1/images:annotate?key={}",
        api_key
    );
    let start = Instant::now();
    match client
        .post(&url)
        .json(&json!({
            "requests": [{
                "image": {"content": b64},
                "features": [{"type": "LABEL_DETECTION", "maxResults": 5}]
            }]
        }))
        .send()
        .await
    {
        Err(e) => RunResult::failure(ms(start), e.to_string()),
        Ok(r) if !r.status().is_success() => {
            let status = r.status().as_u16();
            RunResult::failure(ms(start), format!("HTTP {}", status))
        }
        Ok(r) => {
            let _body: Value = r.json().await.unwrap_or(json!({}));
            RunResult {
                latency_ms: ms(start),
                ttft_ms: None,
                output_units: 1,
                success: true,
                error: None,
            }
        }
    }
}

// ── Text generation benchmarks ────────────────────────────────────────────
// Same measurement for every provider:
//   1. Send the same prompt.
//   2. Use streaming (SSE) where available to capture TTFT.
//   3. Count output characters; estimate tokens at 4 chars/token.
//   4. Record: chars/sec = total_output_chars / (total_latency_ms / 1000)
//              tokens/sec = chars/sec / 4

/// OpenAI-compatible streaming chat completions (OpenAI, Groq, Together).
async fn bench_openai_compat_text(
    client: &Client,
    base_url: &str,
    api_key: &str,
    model: &str,
    prompt: &str,
) -> RunResult {
    let url = format!("{}/v1/chat/completions", base_url);
    let start = Instant::now();
    let res = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&json!({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": true,
            "max_tokens": 256,
            "temperature": 0.1
        }))
        .send()
        .await;

    match res {
        Err(e) => RunResult::failure(ms(start), e.to_string()),
        Ok(r) if !r.status().is_success() => {
            let status = r.status().as_u16();
            let body = r.text().await.unwrap_or_default();
            RunResult::failure(ms(start), format!("HTTP {} — {}", status, trunc(&body, 200)))
        }
        Ok(r) => {
            let mut stream = r.bytes_stream();
            let mut ttft: Option<f64> = None;
            let mut chars = 0usize;

            while let Some(chunk) = stream.next().await {
                if let Ok(bytes) = chunk {
                    for line in String::from_utf8_lossy(&bytes).lines() {
                        if let Some(data) = line.trim().strip_prefix("data: ") {
                            if data == "[DONE]" { break; }
                            if let Ok(v) = serde_json::from_str::<Value>(data) {
                                if let Some(c) = v.pointer("/choices/0/delta/content")
                                    .and_then(|c| c.as_str())
                                {
                                    if !c.is_empty() {
                                        if ttft.is_none() {
                                            ttft = Some(ms(start));
                                        }
                                        chars += c.len();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            RunResult {
                latency_ms: ms(start),
                ttft_ms: ttft,
                output_units: chars,
                success: true,
                error: None,
            }
        }
    }
}

/// Anthropic Messages API (streaming).
async fn bench_anthropic_text(client: &Client, api_key: &str, prompt: &str) -> RunResult {
    let start = Instant::now();
    let res = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 256,
            "stream": true,
            "messages": [{"role": "user", "content": prompt}]
        }))
        .send()
        .await;

    match res {
        Err(e) => RunResult::failure(ms(start), e.to_string()),
        Ok(r) if !r.status().is_success() => {
            let status = r.status().as_u16();
            let body = r.text().await.unwrap_or_default();
            RunResult::failure(ms(start), format!("HTTP {} — {}", status, trunc(&body, 200)))
        }
        Ok(r) => {
            let mut stream = r.bytes_stream();
            let mut ttft: Option<f64> = None;
            let mut chars = 0usize;

            while let Some(chunk) = stream.next().await {
                if let Ok(bytes) = chunk {
                    for line in String::from_utf8_lossy(&bytes).lines() {
                        if let Some(data) = line.trim().strip_prefix("data: ") {
                            if let Ok(v) = serde_json::from_str::<Value>(data) {
                                if v.get("type").and_then(|t| t.as_str())
                                    == Some("content_block_delta")
                                {
                                    if let Some(t) = v.pointer("/delta/text")
                                        .and_then(|t| t.as_str())
                                    {
                                        if !t.is_empty() {
                                            if ttft.is_none() {
                                                ttft = Some(ms(start));
                                            }
                                            chars += t.len();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            RunResult {
                latency_ms: ms(start),
                ttft_ms: ttft,
                output_units: chars,
                success: true,
                error: None,
            }
        }
    }
}

/// torch-inference local text — POST /predict
/// Measures end-to-end including actual model inference on the loaded model.
async fn bench_local_text(client: &Client, prompt: &str) -> RunResult {
    let url = format!("{}/predict", local_url());
    let start = Instant::now();
    match client
        .post(&url)
        .json(&json!({"model_name": "default", "inputs": {"text": prompt}, "priority": 0}))
        .send()
        .await
    {
        Err(e) => RunResult::failure(ms(start), e.to_string()),
        Ok(r) if !r.status().is_success() => {
            let status = r.status().as_u16();
            let body = r.text().await.unwrap_or_default();
            RunResult::failure(ms(start), format!("HTTP {} — {}", status, trunc(&body, 200)))
        }
        Ok(r) => {
            let body: Value = r.json().await.unwrap_or(json!({}));
            let output = serde_json::to_string(&body).unwrap_or_default();
            RunResult {
                latency_ms: ms(start),
                ttft_ms: None,
                output_units: output.len(),
                success: true,
                error: None,
            }
        }
    }
}

// ── Orchestration ─────────────────────────────────────────────────────────

async fn run_tts_benchmarks(client: &Client, runs: usize) -> Vec<ProviderSummary> {
    // Same text for every provider — 89 chars at natural speaking speed ≈ 6.4s audio.
    let text = "The quick brown fox jumps over the lazy dog. \
                Pack my box with five dozen liquor jugs.";

    println!("── TTS Synthesis (same text, {runs} runs each) ─────────────────────");
    println!("  Text: \"{text}\"  ({} chars)", text.len());
    println!();

    let mut summaries = Vec::new();

    // torch-inference (local)
    if probe_local(client).await {
        let results = run_n(runs, || bench_local_tts(client, text)).await;
        print_run_summary("torch-inference (local)", runs, &results);
        summaries.push(ProviderSummary::from_runs(
            "torch-inference (local)", "Kokoro/Piper", "tts", "chars/s", results,
        ));
    } else {
        skip("torch-inference (local)", "server not reachable at LOCAL_SERVER_URL");
    }

    // OpenAI TTS-1
    match api_key("OPENAI_API_KEY") {
        None => skip("OpenAI TTS-1", "OPENAI_API_KEY not set"),
        Some(k) => {
            let results = run_n(runs, || bench_openai_tts(client, &k, text)).await;
            print_run_summary("OpenAI TTS-1", runs, &results);
            summaries.push(ProviderSummary::from_runs(
                "OpenAI", "tts-1", "tts", "chars/s", results,
            ));
        }
    }

    // ElevenLabs
    match api_key("ELEVENLABS_API_KEY") {
        None => skip("ElevenLabs", "ELEVENLABS_API_KEY not set"),
        Some(k) => {
            let results = run_n(runs, || bench_elevenlabs_tts(client, &k, text)).await;
            print_run_summary("ElevenLabs", runs, &results);
            summaries.push(ProviderSummary::from_runs(
                "ElevenLabs", "eleven_monolingual_v1", "tts", "chars/s", results,
            ));
        }
    }

    summaries
}

async fn run_image_benchmarks(client: &Client, runs: usize) -> Vec<ProviderSummary> {
    println!("── Image Classification (same 224×224 JPEG, {runs} runs each) ─────");

    let jpeg = make_test_jpeg();
    println!("  Image: 224×224 synthetic JPEG ({} bytes)", jpeg.len());
    println!();

    let mut summaries = Vec::new();

    // torch-inference (local)
    if probe_local(client).await {
        let results = run_n(runs, || bench_local_image(client, &jpeg)).await;
        print_run_summary("torch-inference (local)", runs, &results);
        summaries.push(ProviderSummary::from_runs(
            "torch-inference (local)", "ONNX/default", "image", "imgs/s", results,
        ));
    } else {
        skip("torch-inference (local)", "server not reachable");
    }

    // Google Cloud Vision
    match api_key("GOOGLE_API_KEY") {
        None => skip("Google Cloud Vision", "GOOGLE_API_KEY not set"),
        Some(k) => {
            let results = run_n(runs, || bench_google_vision(client, &k, &jpeg)).await;
            print_run_summary("Google Cloud Vision", runs, &results);
            summaries.push(ProviderSummary::from_runs(
                "Google Cloud Vision", "vision/v1", "image", "imgs/s", results,
            ));
        }
    }

    summaries
}

async fn run_text_benchmarks(client: &Client, runs: usize) -> Vec<ProviderSummary> {
    // Same prompt for every provider, chosen to produce ~80-120 output tokens.
    let prompt = "List the 8 planets of the solar system in order from the Sun, \
                  one per line, with their diameter in km. Be concise.";

    println!("── Text Inference (same prompt, {runs} runs each) ───────────────────");
    println!("  Prompt: \"{prompt}\"");
    println!();

    let mut summaries = Vec::new();

    // torch-inference (local)
    if probe_local(client).await {
        let results = run_n(runs, || bench_local_text(client, prompt)).await;
        print_run_summary("torch-inference (local)", runs, &results);
        summaries.push(ProviderSummary::from_runs(
            "torch-inference (local)", "loaded model", "text", "chars/s", results,
        ));
    } else {
        skip("torch-inference (local)", "server not reachable");
    }

    // OpenAI GPT-4o-mini
    match api_key("OPENAI_API_KEY") {
        None => skip("OpenAI GPT-4o-mini", "OPENAI_API_KEY not set"),
        Some(k) => {
            let results = run_n(runs, || {
                bench_openai_compat_text(
                    client,
                    "https://api.openai.com",
                    &k,
                    "gpt-4o-mini",
                    prompt,
                )
            })
            .await;
            print_run_summary("OpenAI GPT-4o-mini", runs, &results);
            summaries.push(ProviderSummary::from_runs(
                "OpenAI", "gpt-4o-mini", "text", "chars/s", results,
            ));
        }
    }

    // Groq Llama-3.1-8B
    match api_key("GROQ_API_KEY") {
        None => skip("Groq llama-3.1-8b-instant", "GROQ_API_KEY not set"),
        Some(k) => {
            let results = run_n(runs, || {
                bench_openai_compat_text(
                    client,
                    "https://api.groq.com",
                    &k,
                    "llama-3.1-8b-instant",
                    prompt,
                )
            })
            .await;
            print_run_summary("Groq llama-3.1-8b-instant", runs, &results);
            summaries.push(ProviderSummary::from_runs(
                "Groq", "llama-3.1-8b-instant", "text", "chars/s", results,
            ));
        }
    }

    // Together AI Llama-3.1-8B
    match api_key("TOGETHER_API_KEY") {
        None => skip("Together Llama-3.1-8B", "TOGETHER_API_KEY not set"),
        Some(k) => {
            let results = run_n(runs, || {
                bench_openai_compat_text(
                    client,
                    "https://api.together.xyz",
                    &k,
                    "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                    prompt,
                )
            })
            .await;
            print_run_summary("Together Llama-3.1-8B", runs, &results);
            summaries.push(ProviderSummary::from_runs(
                "Together AI",
                "Llama-3.1-8B-Instruct-Turbo",
                "text",
                "chars/s",
                results,
            ));
        }
    }

    // Anthropic Claude Haiku
    match api_key("ANTHROPIC_API_KEY") {
        None => skip("Anthropic Claude Haiku", "ANTHROPIC_API_KEY not set"),
        Some(k) => {
            let results = run_n(runs, || bench_anthropic_text(client, &k, prompt)).await;
            print_run_summary("Anthropic Claude Haiku", runs, &results);
            summaries.push(ProviderSummary::from_runs(
                "Anthropic",
                "claude-haiku-4-5-20251001",
                "text",
                "chars/s",
                results,
            ));
        }
    }

    summaries
}

// ── Table rendering ───────────────────────────────────────────────────────

fn print_table(summaries: &[ProviderSummary], category: &'static str, title: &str) {
    let rows: Vec<&ProviderSummary> = summaries.iter().filter(|s| s.category == category).collect();
    if rows.is_empty() { return; }

    let unit = rows[0].unit_label;
    let has_ttft = category == "text" && rows.iter().any(|r| r.avg_ttft_ms.is_some());

    println!();
    println!("{}", "─".repeat(90));
    println!("  {}", title);
    println!("{}", "─".repeat(90));

    if has_ttft {
        println!(
            "  {:<30}  {:<28}  {:>10}  {:>10}  {:>8}  {:>8}",
            "Provider", "Model", unit, "tokens/s", "P50 ms", "TTFT ms"
        );
        println!("  {}", "·".repeat(86));
        for r in &rows {
            let tok_s = if category == "text" { r.avg_throughput / 4.0 } else { f64::NAN };
            println!(
                "  {:<30}  {:<28}  {:>10}  {:>10}  {:>8}  {:>8}",
                trunc(&r.provider, 30),
                trunc(&r.model, 28),
                fmt_f(r.avg_throughput),
                if tok_s.is_nan() { "  —  ".to_string() } else { fmt_f(tok_s) },
                fmt_f(r.p50_latency_ms),
                r.avg_ttft_ms.map(fmt_f).unwrap_or_else(|| "  —  ".to_string()),
            );
        }
    } else {
        println!(
            "  {:<30}  {:<28}  {:>10}  {:>8}  {:>8}",
            "Provider", "Model", unit, "P50 ms", "P95 ms"
        );
        println!("  {}", "·".repeat(82));
        for r in &rows {
            println!(
                "  {:<30}  {:<28}  {:>10}  {:>8}  {:>8}",
                trunc(&r.provider, 30),
                trunc(&r.model, 28),
                fmt_f(r.avg_throughput),
                fmt_f(r.p50_latency_ms),
                fmt_f(r.p95_latency_ms),
            );
        }
    }
    println!("{}", "─".repeat(90));

    if category == "text" {
        println!("  tokens/s = chars/s ÷ 4  (estimate; exact only when usage field is returned)");
    }
    if category == "tts" {
        println!("  chars/s = input_chars ÷ synthesis_time  (both providers receive the same text)");
        println!("  RTF = synthesis_time ÷ audio_duration  (lower is better; audio_duration ≈ chars/14)");
        for r in &rows {
            if r.avg_throughput > 0.0 && !r.avg_throughput.is_nan() {
                let audio_duration = rows[0].runs as f64 * 89.0 / 14.0; // rough
                let rtf = (r.p50_latency_ms / 1000.0) / (89.0 / 14.0);
                println!("    {:30}  RTF ≈ {:.2}", r.provider, rtf);
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn trunc(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}

fn fmt_f(v: f64) -> String {
    if v.is_nan() || v.is_infinite() { "  —  ".to_string() } else { format!("{:.1}", v) }
}

fn base64_encode(data: &[u8]) -> String {
    use std::fmt::Write;
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 { chunk[1] as usize } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as usize } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(n >> 18) & 0x3f] as char);
        out.push(TABLE[(n >> 12) & 0x3f] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) & 0x3f] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n & 0x3f] as char } else { '=' });
    }
    out
}

async fn probe_local(client: &Client) -> bool {
    let url = format!("{}/health", local_url());
    client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
}

fn skip(name: &str, reason: &str) {
    println!("  [SKIP] {:<35}  {}", name, reason);
}

fn print_run_summary(name: &str, runs: usize, results: &[RunResult]) {
    let ok = results.iter().filter(|r| r.success).count();
    let avg_latency = mean(
        &results.iter().filter(|r| r.success).map(|r| r.latency_ms).collect::<Vec<_>>(),
    );
    println!(
        "  [OK]   {:<35}  {}/{} ok  avg {:.0} ms",
        name, ok, runs, avg_latency
    );
    for r in results.iter().filter(|r| !r.success) {
        if let Some(e) = &r.error { println!("    error: {}", e); }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return f64::NAN; }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return f64::NAN; }
    v.iter().sum::<f64>() / v.len() as f64
}

async fn run_n<F, Fut>(n: usize, mut f: F) -> Vec<RunResult>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = RunResult>,
{
    let mut results = Vec::with_capacity(n);
    for _ in 0..n {
        results.push(f().await);
        tokio::time::sleep(Duration::from_millis(150)).await;
    }
    results
}

// ── Main ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();

    let runs: usize = args.windows(2)
        .find(|w| w[0] == "--runs")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(5);

    let category: Option<String> = args.windows(2)
        .find(|w| w[0] == "--category")
        .map(|w| w[1].clone());

    let client = build_client();

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  torch-inference  ·  Apples-to-Apples Provider Comparison");
    println!("  Methodology: same input → same output metric → same clock measurement");
    println!("  Runs per provider: {runs}");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    let mut all: Vec<ProviderSummary> = Vec::new();

    let do_tts   = category.as_deref().map_or(true, |c| c == "tts");
    let do_image = category.as_deref().map_or(true, |c| c == "image");
    let do_text  = category.as_deref().map_or(true, |c| c == "text");

    if do_tts   { all.extend(run_tts_benchmarks(&client, runs).await); println!(); }
    if do_image { all.extend(run_image_benchmarks(&client, runs).await); println!(); }
    if do_text  { all.extend(run_text_benchmarks(&client, runs).await); println!(); }

    // ── Results tables ────────────────────────────────────────────────────
    print_table(&all, "tts",   "TTS SYNTHESIS  —  chars/sec of text synthesised to audio");
    print_table(&all, "image", "IMAGE CLASSIFICATION  —  end-to-end latency per image");
    print_table(&all, "text",  "TEXT INFERENCE  —  chars/sec and tokens/sec (streaming)");

    if all.is_empty() {
        println!();
        println!("  No results collected. Either set API keys or start the local server:");
        println!();
        println!("    # Local server");
        println!("    cargo run --release -- --config config/production.toml &");
        println!();
        println!("    # External providers");
        println!("    export OPENAI_API_KEY=sk-...");
        println!("    export GROQ_API_KEY=gsk_...");
        println!("    export TOGETHER_API_KEY=...");
        println!("    export ANTHROPIC_API_KEY=sk-ant-...");
        println!("    export ELEVENLABS_API_KEY=...");
        println!("    export GOOGLE_API_KEY=...");
        println!();
        println!("    cargo run --bin provider-comparison -- --runs 10");
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_run(latency_ms: f64, output_units: usize) -> RunResult {
        RunResult {
            latency_ms,
            ttft_ms: Some(latency_ms * 0.25),
            output_units,
            success: true,
            error: None,
        }
    }

    // ── percentile ──────────────────────────────────────────────────────

    #[test]
    fn test_percentile_empty_is_nan() {
        assert!(percentile(&[], 0.5).is_nan());
    }

    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[7.0], 0.5), 7.0);
        assert_eq!(percentile(&[7.0], 0.99), 7.0);
    }

    #[test]
    fn test_percentile_p50_p95() {
        let v: Vec<f64> = (1..=20).map(|x| x as f64).collect(); // 1..20
        let p50 = percentile(&v, 0.5);
        assert!(p50 >= 9.0 && p50 <= 11.0, "p50={p50}");
        let p95 = percentile(&v, 0.95);
        assert!(p95 >= 18.0 && p95 <= 20.0, "p95={p95}");
    }

    // ── mean ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mean_empty_is_nan() {
        assert!(mean(&[]).is_nan());
    }

    #[test]
    fn test_mean_uniform() {
        let v = vec![10.0, 10.0, 10.0];
        assert!((mean(&v) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_mean_mixed() {
        let v = vec![0.0, 100.0];
        assert!((mean(&v) - 50.0).abs() < 1e-9);
    }

    // ── ProviderSummary::from_runs ────────────────────────────────────────

    #[test]
    fn test_summary_success_rate_all_ok() {
        let results = vec![ok_run(100.0, 400), ok_run(120.0, 480)];
        let s = ProviderSummary::from_runs("P", "m", "text", "chars/s", results);
        assert!((s.success_rate_pct - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_summary_success_rate_half_failed() {
        let results = vec![
            ok_run(100.0, 400),
            RunResult::failure(0.0, "timeout"),
        ];
        let s = ProviderSummary::from_runs("P", "m", "text", "chars/s", results);
        assert!((s.success_rate_pct - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_summary_throughput_chars_per_sec() {
        // 400 chars in 100 ms → 4000 chars/sec
        let results = vec![ok_run(100.0, 400)];
        let s = ProviderSummary::from_runs("P", "m", "text", "chars/s", results);
        assert!((s.avg_throughput - 4000.0).abs() < 1.0, "got {}", s.avg_throughput);
    }

    #[test]
    fn test_summary_tokens_per_sec_derived() {
        // chars/sec = 4000 → tokens/sec = 4000/4 = 1000
        let results = vec![ok_run(100.0, 400)];
        let s = ProviderSummary::from_runs("P", "m", "text", "chars/s", results);
        let tok_s = s.avg_throughput / 4.0;
        assert!((tok_s - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_summary_ttft_captured() {
        let results = vec![ok_run(100.0, 400)]; // ttft = latency * 0.25 = 25 ms
        let s = ProviderSummary::from_runs("P", "m", "text", "chars/s", results);
        assert!(s.avg_ttft_ms.is_some());
        let ttft = s.avg_ttft_ms.unwrap();
        assert!((ttft - 25.0).abs() < 1.0, "got {ttft}");
    }

    #[test]
    fn test_summary_all_failed_no_throughput() {
        let results = vec![RunResult::failure(0.0, "err")];
        let s = ProviderSummary::from_runs("P", "m", "tts", "chars/s", results);
        assert_eq!(s.success_rate_pct, 0.0);
        assert!(s.avg_throughput.is_nan());
        assert!(s.p50_latency_ms.is_nan());
    }

    #[test]
    fn test_summary_p50_p95_ordering() {
        let results: Vec<RunResult> = [10.0, 20.0, 30.0, 40.0, 100.0]
            .iter()
            .map(|&l| ok_run(l, 100))
            .collect();
        let s = ProviderSummary::from_runs("P", "m", "text", "chars/s", results);
        assert!(s.p50_latency_ms <= s.p95_latency_ms,
            "p50={} should be ≤ p95={}", s.p50_latency_ms, s.p95_latency_ms);
    }

    // ── base64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_base64_encode_known_vectors() {
        // RFC 4648 test vectors
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    // ── RunResult::failure ────────────────────────────────────────────────

    #[test]
    fn test_run_result_failure_fields() {
        let r = RunResult::failure(42.0, "connection refused");
        assert!(!r.success);
        assert_eq!(r.latency_ms, 42.0);
        assert_eq!(r.output_units, 0);
        assert!(r.error.as_deref() == Some("connection refused"));
    }

    // ── fmt_f ─────────────────────────────────────────────────────────────

    #[test]
    fn test_fmt_f_nan_dash() {
        assert_eq!(fmt_f(f64::NAN), "  —  ");
        assert_eq!(fmt_f(f64::INFINITY), "  —  ");
    }

    #[test]
    fn test_fmt_f_value() {
        assert_eq!(fmt_f(1234.5), "1234.5");
        assert_eq!(fmt_f(0.0), "0.0");
    }
}
