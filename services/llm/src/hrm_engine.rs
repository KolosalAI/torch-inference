use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    pub session: Arc<Session>,
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
            session: Arc::new(session),
            tokenizer,
            runtime,
            model_dir,
        })
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
}
