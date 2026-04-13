use anyhow::{Context, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    mtmd::{
        mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
    },
    sampling::LlamaSampler,
};
use std::ffi::CString;
use std::num::NonZeroU32;
use tokio::sync::mpsc;

use crate::config::LlmConfig;

pub struct LlamaEngine {
    pub backend: LlamaBackend,
    pub model: LlamaModel,
    pub config: LlmConfig,
    /// Set only when mmproj file exists at startup.
    pub mmproj_path: Option<String>,
}

// LlamaBackend and LlamaModel are marked Send+Sync (unsafe impl) in llama-cpp-2.
// We serialise inference via a single tokio worker (workers(1) in main.rs).
unsafe impl Send for LlamaEngine {}
unsafe impl Sync for LlamaEngine {}

impl LlamaEngine {
    /// Load the GGUF model. Exits with a clear message if the model file is missing.
    pub fn load(config: LlmConfig) -> Result<Self> {
        let model_path = std::path::Path::new(&config.model_path);
        if !model_path.exists() {
            anyhow::bail!(
                "Model file not found: {}  — run `bash scripts/download_llm_model.sh`",
                config.model_path
            );
        }

        tracing::info!(path = %config.model_path, "Loading GGUF model...");
        let backend = LlamaBackend::init().context("init llama backend")?;

        let n_gpu = u32::try_from(config.n_gpu_layers.max(0)).unwrap_or(0);
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu);

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("load model from file")?;
        tracing::info!("Model loaded");

        let mmproj_path = config.effective_mmproj().map(str::to_owned);

        Ok(Self { backend, model, config, mmproj_path })
    }

    /// Build a ChatML-formatted prompt string.
    /// If `image_marker` is Some, it is prepended to the user message content.
    pub fn build_prompt(messages: &[(String, String)], image_marker: Option<&str>) -> String {
        let mut buf = String::new();
        for (role, content) in messages {
            buf.push_str(&format!("<|im_start|>{}\n", role));
            if role == "user" {
                if let Some(marker) = image_marker {
                    buf.push_str(marker);
                    buf.push('\n');
                }
            }
            buf.push_str(content);
            buf.push_str("<|im_end|>\n");
        }
        buf.push_str("<|im_start|>assistant\n");
        buf
    }

    /// Run text-only inference. Sends generated tokens via `tx`.
    /// Must be called from a blocking thread (use tokio::task::spawn_blocking).
    pub fn infer_text(
        &self,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        tx: mpsc::Sender<String>,
    ) -> Result<()> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.config.ctx_size).unwrap()))
            .with_n_threads(self.config.n_threads)
            .with_n_threads_batch(self.config.n_threads);

        let mut ctx = self.model
            .new_context(&self.backend, ctx_params)
            .context("create llama context")?;

        let tokens = self.model
            .str_to_token(&prompt, AddBos::Always)
            .context("tokenize prompt")?;

        let n_prompt = tokens.len();
        let mut batch = LlamaBatch::new(n_prompt.max(512), 1);
        for (i, &tok) in tokens.iter().enumerate() {
            batch.add(tok, i as i32, &[0], i == n_prompt - 1)
                .context("add token to batch")?;
        }
        ctx.decode(&mut batch).context("decode prompt")?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(temperature.clamp(0.01, 2.0)),
            LlamaSampler::dist(0),
        ]);

        let mut n_past = n_prompt as i32;

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(new_token);

            if self.model.is_eog_token(new_token) {
                break;
            }

            let token_str = self.model
                .token_to_str(new_token, Special::Tokenize)
                .unwrap_or_default();

            if tx.blocking_send(token_str).is_err() {
                break;
            }

            batch.clear();
            batch.add(new_token, n_past, &[0], true)
                .context("add generated token")?;
            ctx.decode(&mut batch).context("decode token")?;
            n_past += 1;
        }

        Ok(())
    }

    /// Run multimodal inference with one image.
    /// `image_bytes` are raw JPEG/PNG bytes.
    pub fn infer_multimodal(
        &self,
        messages: &[(String, String)],
        image_bytes: Vec<u8>,
        max_tokens: u32,
        temperature: f32,
        tx: mpsc::Sender<String>,
    ) -> Result<()> {
        let mmproj = self.mmproj_path.as_deref()
            .context("multimodal not configured: mmproj_path missing or file not found")?;

        // Decode image bytes → RGB pixels
        let img = image::load_from_memory(&image_bytes).context("decode image")?;
        let rgb = img.to_rgb8();
        let (w, h) = (img.width(), img.height());
        let rgb_data = rgb.into_raw();

        let bitmap = MtmdBitmap::from_image_data(w, h, &rgb_data)
            .map_err(|e| anyhow::anyhow!("create bitmap: {:?}", e))?;

        let mtmd_params = MtmdContextParams {
            use_gpu: self.config.n_gpu_layers > 0,
            print_timings: false,
            n_threads: self.config.n_threads,
            media_marker: CString::new(mtmd_default_marker())
                .context("build media_marker CString")?,
        };
        let mtmd_ctx = MtmdContext::init_from_file(mmproj, &self.model, &mtmd_params)
            .map_err(|e| anyhow::anyhow!("init mtmd context: {:?}", e))?;

        let marker = mtmd_default_marker();
        let prompt = Self::build_prompt(messages, Some(marker));

        let input_text = MtmdInputText {
            text: prompt,
            add_special: true,
            parse_special: true,
        };

        let chunks = mtmd_ctx
            .tokenize(input_text, &[&bitmap])
            .map_err(|e| anyhow::anyhow!("mtmd tokenize: {:?}", e))?;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(self.config.ctx_size).unwrap()))
            .with_n_threads(self.config.n_threads)
            .with_n_threads_batch(self.config.n_threads);
        let mut ctx = self.model
            .new_context(&self.backend, ctx_params)
            .context("create llama context")?;

        // eval_chunks with logits_last=true so we can sample immediately after
        let n_past = chunks
            .eval_chunks(&mtmd_ctx, &ctx, 0, 0, 512, true)
            .map_err(|e| anyhow::anyhow!("eval chunks: {:?}", e))?;

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(temperature.clamp(0.01, 2.0)),
            LlamaSampler::dist(0),
        ]);

        let mut batch = LlamaBatch::new(512, 1);
        let mut n_cur = n_past;

        for _ in 0..max_tokens {
            let new_token = sampler.sample(&ctx, (n_cur - 1) as i32);
            sampler.accept(new_token);

            if self.model.is_eog_token(new_token) {
                break;
            }

            let token_str = self.model
                .token_to_str(new_token, Special::Tokenize)
                .unwrap_or_default();

            if tx.blocking_send(token_str).is_err() {
                break;
            }

            batch.clear();
            batch.add(new_token, n_cur, &[0], true)
                .context("add generated token")?;
            ctx.decode(&mut batch).context("decode token")?;
            n_cur += 1;
        }

        Ok(())
    }
}
