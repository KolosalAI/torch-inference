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
    pub session: Arc<Mutex<Session>>,
    pub tokenizer: HrmTokenizer,
    pub runtime: HrmRuntimeConfig,
    pub model_dir: PathBuf,
}

unsafe impl Send for HrmEngine {}
unsafe impl Sync for HrmEngine {}

impl HrmEngine {
    pub fn load(cfg: &HrmConfig) -> Result<Self> {

        let model_dir = PathBuf::from(&cfg.model_dir);
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
            session: Arc::new(Mutex::new(session)),
            tokenizer,
            runtime,
            model_dir,
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
        let mut ids = self.tokenizer.encode(&prompt, true)?;
        for _ in 0..max_tokens {
            let logits = self.prefill(&ids)?;
            let next = self.sample(&logits, temperature, 40, 0.95);
            let next_i64 = next as i64;

            if next as u32 == self.runtime.eos_token_id { break; }
            if ids.len() as u32 >= self.runtime.ctx_size { break; }

            let piece = self.tokenizer.decode_single(next as u32).unwrap_or_default();
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

        let seq_len = input_ids.len();
        let input_tensor = Tensor::<i64>::from_array(
            ([1_usize, seq_len], input_ids.to_vec())
        ).context("build input_ids tensor")?;

        let mut session = self.session
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
        };
        let err = HrmEngine::load(&cfg).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
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
    fn prefill_returns_logits_for_last_position() {
        if skip_if_no_model() {
            eprintln!("skipping: requires hrm-text-1b artifacts");
            return;
        }
        let eng = HrmEngine::load(&fixture_cfg()).unwrap();
        let ids = eng.tokenizer.encode("The capital of France is", true).unwrap();
        let logits = eng.prefill(&ids).unwrap();
        assert_eq!(logits.len() as u32, eng.runtime.vocab_size);
        // Logits should be non-uniform (some variation between positions)
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max - min > 0.1, "logits look uniform: max-min={}", max-min);
    }

    #[test]
    fn decode_greedy_produces_tokens_under_max() {
        if skip_if_no_model() {
            eprintln!("skipping");
            return;
        }
        let eng = HrmEngine::load(&fixture_cfg()).unwrap();
        let ids = eng.tokenizer.encode("Hello,", true).unwrap();
        let generated = eng.decode_greedy(&ids, 8).unwrap();
        assert!(!generated.is_empty(), "no tokens generated");
        assert!(generated.len() <= 8, "exceeded max_tokens");
        // None of the generated tokens should equal eos (decode stops on eos)
        let eos = eng.runtime.eos_token_id;
        assert!(!generated.iter().any(|&t| t == eos as i64));
    }

    #[tokio::test]
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
}
