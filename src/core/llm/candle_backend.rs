//! Candle-backed LLM implementation of the [`LlmBackend`] trait.

use anyhow::{Context, Result};
use async_trait::async_trait;
use candle_core::Device;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc};
use parking_lot::Mutex;

use crate::api::llm::{LlmBackend, ModelInfo};
use crate::core::llm::sampler::{self, SamplingParams};

/// Select the best available device: Metal (macOS) > CPU.
/// NOTE: CUDA selection is not yet wired — add `Device::new_cuda(0)` here
/// when the `cuda` feature on candle-core is enabled.
pub fn best_device() -> Device {
    #[cfg(all(target_os = "macos", feature = "llm-metal"))]
    if let Ok(dev) = Device::new_metal(0) {
        return dev;
    }
    Device::Cpu
}

struct LoadedModel {
    info: ModelInfo,
    vocab_size: usize,
}

pub struct CandleLlmBackend {
    device: Device,
    models: Arc<Mutex<HashMap<String, LoadedModel>>>,
}

impl CandleLlmBackend {
    pub fn new() -> Self {
        Self {
            device: best_device(),
            models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn new_cpu() -> Self {
        Self {
            device: Device::Cpu,
            models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn list_models_sync(&self) -> Vec<ModelInfo> {
        self.models
            .lock()
            .unwrap()
            .values()
            .map(|m| m.info.clone())
            .collect()
    }

    pub fn load_model(&self, model_id: String, model_dir: PathBuf) -> Result<()> {
        let config_path = model_dir.join("config.json");
        let config_bytes = std::fs::read(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let config: serde_json::Value =
            serde_json::from_slice(&config_bytes).context("parsing config.json")?;

        let vocab_size = config["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let architecture = config["model_type"].as_str().unwrap_or("llama").to_string();

        tracing::info!(model_id = %model_id, architecture = %architecture, vocab_size, "loading candle model");

        let model = LoadedModel {
            info: ModelInfo {
                id: model_id.clone(),
                object: "model".to_string(),
                created: chrono::Utc::now().timestamp(),
                owned_by: "local".to_string(),
            },
            vocab_size,
        };

        self.models.lock().insert(model_id, model);
        Ok(())
    }

    fn greedy_decode(&self, params: &SamplingParams, vocab_size: usize) -> Result<Vec<u32>> {
        let mut output_ids: Vec<u32> = Vec::with_capacity(params.max_tokens);
        let logits = vec![0.0f32; vocab_size];

        for _ in 0..params.max_tokens {
            let next_token = sampler::sample(&logits, params).context("sampling next token")?;
            if params.stop_token_ids.contains(&next_token) {
                break;
            }
            output_ids.push(next_token);
        }
        Ok(output_ids)
    }

    /// STUB — replace with a real BPE/SentencePiece tokeniser when model weights are wired in.
    fn tokenize(prompt: &str) -> Vec<u32> {
        prompt
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| (i % 32000) as u32)
            .collect()
    }

    /// STUB — replace with vocabulary lookup once a real tokeniser is integrated.
    fn detokenize(token_ids: &[u32]) -> String {
        token_ids
            .iter()
            .map(|id| format!("[{}]", id))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for CandleLlmBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmBackend for CandleLlmBackend {
    fn list_models(&self) -> Vec<ModelInfo> {
        self.list_models_sync()
    }

    async fn complete(
        &self,
        model: &str,
        prompt: &str,
        params: SamplingParams,
    ) -> Result<(String, usize)> {
        let vocab_size = {
            let guard = self.models.lock();
            guard
                .get(model)
                .map(|m| m.vocab_size)
                .ok_or_else(|| anyhow::anyhow!("model '{}' is not loaded", model))?
        };

        params.validate().context("invalid sampling params")?;
        // Stub: token IDs will be passed to the model forward pass once
        // real tokenisation and model weights are integrated.
        let _token_ids = Self::tokenize(prompt);
        let output_ids = self.greedy_decode(&params, vocab_size)?;
        let completion_tokens = output_ids.len();
        let text = Self::detokenize(&output_ids);
        Ok((text, completion_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cpu_backend_has_no_models() {
        let b = CandleLlmBackend::new_cpu();
        assert!(b.list_models_sync().is_empty());
    }

    #[test]
    fn tokenize_produces_bounded_ids() {
        let ids = CandleLlmBackend::tokenize("hello world foo");
        assert_eq!(ids.len(), 3);
        assert!(ids.iter().all(|&id| id < 32000));
    }

    #[test]
    fn detokenize_is_deterministic() {
        let a = CandleLlmBackend::detokenize(&[1, 2, 3]);
        let b = CandleLlmBackend::detokenize(&[1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn greedy_decode_respects_max_tokens() {
        let b = CandleLlmBackend::new_cpu();
        let params = SamplingParams::greedy().with_max_tokens(5);
        let output = b.greedy_decode(&params, 32000).unwrap();
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn greedy_decode_stops_at_stop_token() {
        let b = CandleLlmBackend::new_cpu();
        // With a vocab_size of 1, argmax always returns token 0.
        // Make token 0 a stop token -> should stop immediately.
        let params = SamplingParams {
            temperature: 0.0,
            top_k: 1,
            max_tokens: 100,
            stop_token_ids: vec![0],
            ..SamplingParams::default()
        };
        let output = b.greedy_decode(&params, 1).unwrap();
        assert!(output.is_empty(), "should stop immediately at token 0");
    }

    #[tokio::test]
    async fn complete_fails_for_unloaded_model() {
        let b = CandleLlmBackend::new_cpu();
        let result = b
            .complete("nonexistent", "hello", SamplingParams::greedy())
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not loaded"));
    }
}
