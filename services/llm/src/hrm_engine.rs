use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::config::HrmConfig;
use crate::tokenizer::HrmTokenizer;

#[derive(Debug, Clone, Deserialize)]
pub struct HrmRuntimeConfig {
    pub eos_token_id: u32,
    pub ctx_size: u32,
    pub slow_loops: u32,
    pub fast_loops: u32,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
}

#[derive(Debug)]
pub struct HrmEngine {
    /// `None` in stub mode (no ONNX weights loaded).
    pub session: Option<Arc<Mutex<Session>>>,
    /// `None` in stub mode (no tokenizer files loaded).
    pub tokenizer: Option<HrmTokenizer>,
    pub runtime: HrmRuntimeConfig,
    pub model_dir: PathBuf,
    /// When true, the engine emits canned output instead of running inference.
    stub: bool,
}

unsafe impl Send for HrmEngine {}
unsafe impl Sync for HrmEngine {}

impl HrmEngine {
    /// True when running as the lightweight stub (no weights loaded).
    pub fn is_stub(&self) -> bool {
        self.stub
    }

    pub fn load(cfg: &HrmConfig) -> Result<Self> {
        let model_dir = PathBuf::from(&cfg.model_dir);

        // Stub mode: boot with no model/tokenizer files at all.
        if cfg.stub.unwrap_or(false) {
            tracing::warn!("HRM-Text running in STUB mode — no weights loaded, canned responses only");
            return Ok(Self {
                session: None,
                tokenizer: None,
                runtime: HrmRuntimeConfig {
                    eos_token_id: 0,
                    ctx_size: 1024,
                    slow_loops: 1,
                    fast_loops: 1,
                    vocab_size: 1,
                    hidden_size: 1,
                    num_layers: 1,
                },
                model_dir,
                stub: true,
            });
        }

        let onnx_path = if cfg.use_quantized.unwrap_or(false) {
            model_dir.join("model.int8.onnx")
        } else {
            model_dir.join("model.onnx")
        };

        if !onnx_path.exists() {
            anyhow::bail!(
                "HRM-Text ONNX not found at {}. Run `make hrm-download` or `make hrm-export`.",
                onnx_path.display()
            );
        }

        tracing::info!(path = %onnx_path.display(), "Loading HRM-Text ONNX...");

        let session = Self::build_session(&onnx_path, cfg)
            .context("build ort session")?;

        let tokenizer = HrmTokenizer::load(&model_dir)
            .context("load HrmTokenizer")?;

        let runtime: HrmRuntimeConfig = {
            let text = std::fs::read_to_string(model_dir.join("config.json"))
                .context("read config.json")?;
            serde_json::from_str(&text).context("parse config.json")?
        };

        tracing::info!(
            ctx_size = runtime.ctx_size,
            slow_loops = runtime.slow_loops,
            fast_loops = runtime.fast_loops,
            "HRM-Text loaded"
        );

        Ok(Self {
            session: Some(Arc::new(Mutex::new(session))),
            tokenizer: Some(tokenizer),
            runtime,
            model_dir,
            stub: false,
        })
    }

    /// Greedy autoregressive decode. Returns the list of generated token IDs
    /// (not including the prompt, not including EOS).
    pub fn decode_greedy(&self, prompt_ids: &[i64], max_tokens: u32) -> Result<Vec<i64>> {
        let mut ids: Vec<i64> = prompt_ids.to_vec();
        let mut out: Vec<i64> = Vec::with_capacity(max_tokens as usize);

        for _ in 0..max_tokens {
            let logits = self.prefill(&ids)?;
            // argmax
            let (next_id, _) = logits.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                    if v > acc.1 { (i, v) } else { acc }
                });
            let next_id = next_id as i64;

            if next_id as u32 == self.runtime.eos_token_id {
                break;
            }
            if ids.len() as u32 >= self.runtime.ctx_size {
                tracing::warn!("decode hit ctx_size cap");
                break;
            }
            ids.push(next_id);
            out.push(next_id);
        }
        Ok(out)
    }

    /// Sample one token from `logits` using top-k, top-p, temperature.
    /// temperature <= 0 -> greedy argmax.
    fn sample(&self, logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> usize {
        if temperature <= 0.0 {
            return logits.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)|
                    if v > acc.1 { (i, v) } else { acc }).0;
        }
        let t = temperature.clamp(0.01, 2.0);

        // top-k
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i,&v)| (i, v/t)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k.max(1));

        // softmax
        let max = indexed[0].1;
        let mut probs: Vec<f32> = indexed.iter().map(|(_, l)| (l - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum; }

        // top-p (nucleus): keep smallest prefix with cumulative prob >= top_p
        let mut cum = 0.0_f32;
        let mut keep = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p { keep = i + 1; break; }
        }
        probs.truncate(keep);
        let renorm: f32 = probs.iter().sum();
        for p in &mut probs { *p /= renorm; }

        // weighted choice
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut acc = 0.0_f32;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r <= acc { return indexed[i].0; }
        }
        indexed.last().unwrap().0
    }

    /// Drop-in replacement for the old LlamaEngine::infer_text. Streams
    /// decoded token strings into `tx`. Blocking — wrap in spawn_blocking.
    pub fn infer_text(
        self: std::sync::Arc<Self>,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<()> {
        // Stub mode: stream a canned, deterministic reply (one chunk per word),
        // capped by max_tokens. No tokenizer or ONNX session is touched.
        if self.stub {
            let reply = format!(
                "[stub-llm] HRM-Text stub engine active — no weights loaded. \
                 Received a {}-char prompt; the chat pipeline is working end-to-end.",
                prompt.len()
            );
            for (i, word) in reply.split_inclusive(' ').enumerate() {
                if i as u32 >= max_tokens { break; }
                if tx.blocking_send(word.to_string()).is_err() { break; }
            }
            return Ok(());
        }

        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("infer_text called without a tokenizer"))?;
        let mut ids = tokenizer.encode(&prompt, true)?;
        for _ in 0..max_tokens {
            let logits = self.prefill(&ids)?;
            let next = self.sample(&logits, temperature, 40, 0.95);
            let next_i64 = next as i64;

            if next as u32 == self.runtime.eos_token_id { break; }
            if ids.len() as u32 >= self.runtime.ctx_size { break; }

            let piece = tokenizer.decode_single(next as u32).unwrap_or_default();
            if tx.blocking_send(piece).is_err() { break; }
            ids.push(next_i64);
        }
        Ok(())
    }

    /// Run a prefill pass on `input_ids` and return the next-token logits
    /// (over the full vocab) corresponding to the last position.
    ///
    /// HRM-Text exports logits in fp16. This method converts to fp32 once,
    /// so callers (sampler, etc.) can stay in standard f32.
    ///
    /// Returned shape: `Vec<f32>` of length `runtime.vocab_size`.
    pub fn prefill(&self, input_ids: &[i64]) -> Result<Vec<f32>> {
        use ort::value::Tensor;

        if input_ids.is_empty() {
            anyhow::bail!("prefill requires at least one input token");
        }

        // Hard ceiling on sequence length. The monolithic graph emits a
        // [1, seq, vocab] fp16 logits tensor and O(seq^2) attention scores, so an
        // unbounded sequence is the primary OOM/crash vector. Refuse here — before
        // building any tensor or locking the session — so EVERY caller (chat,
        // planner, decode_greedy) is protected, not just the ones that remembered
        // to check. ~vocab*2 bytes per position; report the would-be allocation.
        if input_ids.len() as u32 > self.runtime.ctx_size {
            let approx_mb =
                (input_ids.len() as u64 * self.runtime.vocab_size as u64 * 2) / (1024 * 1024);
            anyhow::bail!(
                "input sequence length {} exceeds ctx_size {} — refusing prefill \
                 (monolithic logits would allocate ~{} MB)",
                input_ids.len(),
                self.runtime.ctx_size,
                approx_mb
            );
        }

        let seq_len = input_ids.len();
        let input_tensor = Tensor::<i64>::from_array(
            ([1_usize, seq_len], input_ids.to_vec())
        ).context("build input_ids tensor")?;

        let session_arc = self.session.as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill called on stub engine (no ONNX session)"))?;
        let mut session = session_arc
            .lock()
            .map_err(|e| anyhow::anyhow!("session lock poisoned: {}", e))?;
        let outputs = session
            .run(ort::inputs!["input_ids" => input_tensor])
            .context("ort run prefill")?;

        // ort 2.0.0-rc.10: try_extract_tensor returns (&Shape, &[T]) tuple,
        // not an ndarray view. HRM-Text logits are fp16.
        let (shape, data) = outputs["logits"]
            .try_extract_tensor::<half::f16>()
            .context("extract fp16 logits")?;

        let dims = shape.as_ref();
        // Expected layout: [batch=1, seq, vocab]
        if dims.len() != 3 {
            anyhow::bail!("unexpected logits shape: {:?}", dims);
        }
        let vocab = dims[2] as usize;
        let last_pos = (dims[1] as usize) - 1;
        let row_start = last_pos * vocab;
        Ok(data[row_start..row_start + vocab]
            .iter()
            .map(|h| h.to_f32())
            .collect())
    }

    fn build_session(onnx_path: &Path, cfg: &HrmConfig) -> Result<Session> {
        let threads = cfg.n_threads.unwrap_or(4).max(1);
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads as usize)?;
        // EP selection per cfg.ep_preference. "auto" -> platform default.
        // Concrete EP wiring is omitted here; ort 2.0.0-rc.10 picks CPU by default.
        // Additional EPs (CoreML/CUDA) are a follow-up perf spec.
        Ok(builder.commit_from_file(onnx_path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_cfg() -> HrmConfig {
        HrmConfig {
            model_dir: format!("{}/models/hrm-text-1b", env!("CARGO_MANIFEST_DIR")),
            ep_preference: "cpu".into(),
            use_quantized: Some(false),
            n_threads: Some(2),
            stub: Some(false),
        }
    }

    fn skip_if_no_model() -> bool {
        !std::path::Path::new(&format!(
            "{}/models/hrm-text-1b/model.onnx",
            env!("CARGO_MANIFEST_DIR")
        )).exists()
    }

    #[test]
    fn load_errors_when_onnx_missing() {
        let cfg = HrmConfig {
            model_dir: "/nonexistent/path".into(),
            ep_preference: "cpu".into(),
            use_quantized: Some(false),
            n_threads: Some(2),
            stub: Some(false),
        };
        let err = HrmEngine::load(&cfg).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    // NOTE: the tests below load the real 2.3 GB ONNX model, so they create an
    // ORT environment. On macOS, ORT >= 1.21 crashes (SIGABRT) in OrtEnv's
    // static destructor at process exit — see `exit_skipping_ort_teardown` in
    // main.rs. libtest cannot bypass that exit, so these are `#[ignore]`d to keep
    // the default `cargo test` green and fast. Run them with `-- --ignored`
    // (model required); the trailing SIGABRT there is the known upstream artifact
    // and does not invalidate the assertions, which all run before exit.
    #[test]
    #[ignore = "loads real ORT model; triggers upstream macOS ORT exit SIGABRT (run with --ignored)"]
    fn load_succeeds_with_artifacts() {
        if skip_if_no_model() {
            eprintln!("skipping: run `make hrm-download` to enable HrmEngine load tests");
            return;
        }
        let cfg = fixture_cfg();
        let eng = HrmEngine::load(&cfg).unwrap();
        assert!(eng.runtime.ctx_size > 0);
        assert!(eng.runtime.slow_loops >= 1);
    }

    #[test]
    #[ignore = "loads real ORT model; triggers upstream macOS ORT exit SIGABRT (run with --ignored)"]
    fn prefill_returns_logits_for_last_position() {
        if skip_if_no_model() {
            eprintln!("skipping: requires hrm-text-1b artifacts");
            return;
        }
        let eng = HrmEngine::load(&fixture_cfg()).unwrap();
        let ids = eng.tokenizer.as_ref().unwrap().encode("The capital of France is", true).unwrap();
        let logits = eng.prefill(&ids).unwrap();
        assert_eq!(logits.len() as u32, eng.runtime.vocab_size);
        // Logits should be non-uniform (some variation between positions)
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max - min > 0.1, "logits look uniform: max-min={}", max-min);
    }

    #[test]
    #[ignore = "loads real ORT model; triggers upstream macOS ORT exit SIGABRT (run with --ignored)"]
    fn decode_greedy_produces_tokens_under_max() {
        if skip_if_no_model() {
            eprintln!("skipping");
            return;
        }
        let eng = HrmEngine::load(&fixture_cfg()).unwrap();
        let ids = eng.tokenizer.as_ref().unwrap().encode("Hello,", true).unwrap();
        let generated = eng.decode_greedy(&ids, 8).unwrap();
        assert!(!generated.is_empty(), "no tokens generated");
        assert!(generated.len() <= 8, "exceeded max_tokens");
        // None of the generated tokens should equal eos (decode stops on eos)
        let eos = eng.runtime.eos_token_id;
        assert!(!generated.iter().any(|&t| t == eos as i64));
    }

    #[tokio::test]
    #[ignore = "loads real ORT model; triggers upstream macOS ORT exit SIGABRT (run with --ignored)"]
    async fn infer_text_streams_tokens_via_channel() {
        if skip_if_no_model() {
            eprintln!("skipping");
            return;
        }
        let eng = std::sync::Arc::new(HrmEngine::load(&fixture_cfg()).unwrap());
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(32);

        let eng2 = eng.clone();
        let h = tokio::task::spawn_blocking(move || {
            eng2.infer_text("Hello,".to_string(), 8, 0.0, tx)
        });

        let mut received = Vec::new();
        while let Some(s) = rx.recv().await { received.push(s); }
        h.await.unwrap().unwrap();
        assert!(!received.is_empty(), "no streamed tokens");
    }

    // ──────────────────────────── stub mode ──────────────────────────────────
    // The stub engine lets the service boot and answer /v1/chat/completions with
    // near-zero memory — no 2.3 GB ONNX, no tokenizer files required.

    fn stub_cfg() -> HrmConfig {
        HrmConfig {
            model_dir: "/nonexistent/stub/path".into(),
            ep_preference: "cpu".into(),
            use_quantized: Some(false),
            n_threads: Some(2),
            stub: Some(true),
        }
    }

    #[test]
    fn stub_load_succeeds_without_any_model_files() {
        // No model.onnx, no tokenizer.json, no config.json — load must still succeed.
        let eng = HrmEngine::load(&stub_cfg())
            .expect("stub load should succeed without model files");
        assert!(eng.is_stub(), "engine should report stub mode");
        assert!(eng.runtime.ctx_size > 0, "stub should expose a usable ctx_size");
    }

    #[tokio::test]
    async fn stub_infer_text_streams_nonempty_output() {
        let eng = std::sync::Arc::new(HrmEngine::load(&stub_cfg()).unwrap());
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);

        let eng2 = eng.clone();
        let h = tokio::task::spawn_blocking(move || {
            eng2.infer_text("Hello there, are you working?".to_string(), 16, 0.0, tx)
        });

        let mut received = String::new();
        while let Some(s) = rx.recv().await { received.push_str(&s); }
        h.await.unwrap().unwrap();

        assert!(!received.is_empty(), "stub must stream non-empty output");
    }

    #[tokio::test]
    async fn stub_infer_text_respects_max_tokens() {
        // With max_tokens = 1, the stub must emit at most one chunk.
        let eng = std::sync::Arc::new(HrmEngine::load(&stub_cfg()).unwrap());
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);

        let eng2 = eng.clone();
        let h = tokio::task::spawn_blocking(move || {
            eng2.infer_text("Count the words in this prompt".to_string(), 1, 0.0, tx)
        });

        let mut chunks = 0usize;
        while rx.recv().await.is_some() { chunks += 1; }
        h.await.unwrap().unwrap();

        assert!(chunks >= 1, "should emit at least one chunk");
        assert!(chunks <= 1, "max_tokens=1 must cap the stub to one chunk, got {chunks}");
    }

    #[test]
    fn prefill_refuses_sequence_longer_than_ctx_size() {
        // A sequence longer than ctx_size would make the monolithic ONNX graph
        // allocate a [1, seq, vocab] fp16 logits tensor — for a huge prompt that
        // is tens to hundreds of GB and OOM-kills the host. prefill must refuse
        // BEFORE building any tensor or touching ORT, for every caller. We can
        // prove the guard fires ahead of the session lookup using a stub engine
        // (session = None): an oversized input must report the ctx-size refusal,
        // NOT the "stub engine has no session" error.
        let eng = HrmEngine::load(&stub_cfg()).unwrap();
        let ctx = eng.runtime.ctx_size as usize;
        let oversized = vec![0i64; ctx + 1];
        let err = eng.prefill(&oversized).unwrap_err().to_string();
        assert!(
            err.contains("ctx_size") || err.contains("exceeds"),
            "expected a ctx-size refusal, got: {err}"
        );
    }

    #[test]
    fn prefill_allows_sequence_at_ctx_size_boundary() {
        // Exactly ctx_size tokens is allowed (the guard is strictly greater-than).
        // On the stub engine that means we fall through to the "no session" error,
        // which proves the seq guard did NOT trip at the boundary.
        let eng = HrmEngine::load(&stub_cfg()).unwrap();
        let ctx = eng.runtime.ctx_size as usize;
        let at_boundary = vec![0i64; ctx];
        let err = eng.prefill(&at_boundary).unwrap_err().to_string();
        assert!(
            !(err.contains("ctx_size") || err.contains("exceeds")),
            "boundary length must not trip the ctx-size guard, got: {err}"
        );
    }

    #[test]
    fn non_stub_load_still_errors_when_onnx_missing() {
        // Guard: a non-stub config with a missing model must still fail loudly.
        let cfg = HrmConfig {
            model_dir: "/nonexistent/path".into(),
            ep_preference: "cpu".into(),
            use_quantized: Some(false),
            n_threads: Some(2),
            stub: Some(false),
        };
        let err = HrmEngine::load(&cfg).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }
}
