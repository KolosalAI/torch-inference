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
}
