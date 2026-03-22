#[cfg(feature = "torch")]
use tch::{nn, nn::Module, Device, Tensor, Kind};
#[cfg(feature = "torch")]
use anyhow::Result;
#[cfg(feature = "torch")]
use anyhow::Context;
#[cfg(feature = "torch")]
use serde::Deserialize;

#[cfg(feature = "torch")]
#[derive(Debug, Clone, Deserialize)]
pub struct StyleTTS2Config {
    pub dim_in: i64,       // 64  — initial token embedding size
    pub hidden_dim: i64,   // 512 — working dimension
    pub n_token: i64,      // 178
    pub style_dim: i64,    // 128
    pub n_mels: i64,       // 80
    pub n_layer: i64,      // 3
    pub dropout: f64,      // 0.2
    pub max_dur: i64,      // 50
    pub multispeaker: bool,
    pub n_speakers: i64,   // 54
}

#[cfg(feature = "torch")]
impl Default for StyleTTS2Config {
    fn default() -> Self {
        Self {
            dim_in: 64,
            hidden_dim: 512,
            n_token: 178,
            style_dim: 128,
            n_mels: 80,
            n_layer: 3,
            dropout: 0.2,
            max_dur: 50,
            multispeaker: true,
            n_speakers: 54,
        }
    }
}

#[cfg(feature = "torch")]
pub struct TextEncoder {
    embed: nn::Embedding,
    proj:  nn::Linear,
    convs: Vec<nn::Conv1D>,
    norms: Vec<nn::LayerNorm>,
}

#[cfg(feature = "torch")]
impl TextEncoder {
    pub fn new(vs: &nn::Path, cfg: &StyleTTS2Config) -> Self {
        let embed = nn::embedding(vs / "embed", cfg.n_token, cfg.dim_in, Default::default());
        let proj  = nn::linear(vs / "proj", cfg.dim_in, cfg.hidden_dim, Default::default());
        let mut convs = Vec::new();
        let mut norms = Vec::new();
        for i in 0..cfg.n_layer {
            convs.push(nn::conv1d(
                vs / format!("conv_{}", i),
                cfg.hidden_dim,
                cfg.hidden_dim,
                5,
                nn::ConvConfig { padding: 2, ..Default::default() },
            ));
            norms.push(nn::layer_norm(
                vs / format!("norm_{}", i),
                vec![cfg.hidden_dim],
                Default::default(),
            ));
        }
        Self { embed, proj, convs, norms }
    }

    pub fn forward(&self, tokens: &Tensor) -> Tensor {
        // tokens: [batch, seq]
        let x = self.embed.forward(tokens);      // [batch, seq, dim_in]
        let mut x = self.proj.forward(&x);        // [batch, seq, hidden_dim]
        // Conv1d expects [batch, channels, seq]
        x = x.transpose(1, 2);
        for (conv, norm) in self.convs.iter().zip(self.norms.iter()) {
            let residual = x.shallow_clone();
            x = conv.forward(&x).relu();
            // LayerNorm over channel dim — transpose to [batch, seq, hidden]
            x = norm.forward(&x.transpose(1, 2)).transpose(1, 2);
            x = x + residual;
        }
        x.transpose(1, 2) // [batch, seq, hidden_dim]
    }
}

#[cfg(feature = "torch")]
pub struct DurationPredictor {
    conv1: nn::Conv1D,
    conv2: nn::Conv1D,
    proj:  nn::Linear,
}

#[cfg(feature = "torch")]
impl DurationPredictor {
    pub fn new(vs: &nn::Path, cfg: &StyleTTS2Config) -> Self {
        Self {
            conv1: nn::conv1d(
                vs / "conv1",
                cfg.hidden_dim,
                256,
                3,
                nn::ConvConfig { padding: 1, ..Default::default() },
            ),
            conv2: nn::conv1d(
                vs / "conv2",
                256,
                256,
                3,
                nn::ConvConfig { padding: 1, ..Default::default() },
            ),
            proj: nn::linear(vs / "proj", 256, 1, Default::default()),
        }
    }

    pub fn forward(&self, hidden: &Tensor) -> Tensor {
        // hidden: [batch, seq, hidden_dim]
        let x = hidden.transpose(1, 2); // [batch, hidden_dim, seq]
        let x = self.conv1.forward(&x).relu();
        let x = self.conv2.forward(&x).relu();
        let x = x.transpose(1, 2); // [batch, seq, 256]
        self.proj.forward(&x).squeeze_dim(-1).softplus() // [batch, seq]
    }
}

#[cfg(feature = "torch")]
pub struct SpeakerEmbedding {
    embed: nn::Embedding,
}

#[cfg(feature = "torch")]
impl SpeakerEmbedding {
    pub fn new(vs: &nn::Path, n_speakers: i64, style_dim: i64) -> Self {
        Self {
            embed: nn::embedding(vs / "embed", n_speakers, style_dim, Default::default()),
        }
    }

    pub fn forward(&self, speaker_id: i64, device: Device) -> Tensor {
        let id = Tensor::of_slice(&[speaker_id]).to_device(device);
        self.embed.forward(&id) // [1, style_dim]
    }
}

#[cfg(feature = "torch")]
pub struct MelDecoder {
    style_proj: nn::Linear,
    convs: Vec<nn::Conv1D>,
    out_proj: nn::Conv1D,
}

#[cfg(feature = "torch")]
impl MelDecoder {
    pub fn new(vs: &nn::Path, cfg: &StyleTTS2Config) -> Self {
        Self {
            style_proj: nn::linear(
                vs / "style_proj",
                cfg.style_dim,
                cfg.hidden_dim,
                Default::default(),
            ),
            convs: (0..3)
                .map(|i| {
                    nn::conv1d(
                        vs / format!("conv_{}", i),
                        cfg.hidden_dim,
                        cfg.hidden_dim,
                        3,
                        nn::ConvConfig { padding: 1, ..Default::default() },
                    )
                })
                .collect(),
            out_proj: nn::conv1d(
                vs / "out_proj",
                cfg.hidden_dim,
                cfg.n_mels,
                1,
                Default::default(),
            ),
        }
    }

    pub fn forward(&self, hidden_expanded: &Tensor, style: &Tensor) -> Tensor {
        // hidden_expanded: [batch, frames, hidden_dim]
        // style: [batch, style_dim]
        let style_bias = self.style_proj.forward(style).unsqueeze(1); // [batch, 1, hidden]
        let mut x = (hidden_expanded + style_bias).transpose(1, 2);   // [batch, hidden, frames]
        for conv in &self.convs {
            let res = x.shallow_clone();
            x = (conv.forward(&x) + res).relu();
        }
        self.out_proj.forward(&x) // [batch, n_mels, frames]
    }
}

/// Duration expansion: repeat each hidden state by its predicted integer duration.
///
/// # Note
/// This function is only correct for `batch_size == 1`. For batch > 1, use separate
/// per-item calls or a padded implementation.
#[cfg(feature = "torch")]
pub fn duration_expand(hidden: &Tensor, durations: &Tensor) -> Tensor {
    debug_assert_eq!(hidden.size()[0], 1, "duration_expand only supports batch_size=1");
    // hidden: [batch, seq, hidden_dim], durations: [batch, seq] (float, round to int)
    let batch = hidden.size()[0];
    let hidden_dim = hidden.size()[2];
    let dur_int = durations.round().to_kind(Kind::Int64).clamp(1, 50);
    let total_frames = dur_int.sum(Kind::Int64).int64_value(&[]);

    let mut frames = Vec::new();
    let seq_len = hidden.size()[1];
    for b in 0..batch {
        for s in 0..seq_len {
            let d = dur_int.i((b, s)).int64_value(&[]);
            let h = hidden.i((b, s, ..)).unsqueeze(0); // [1, hidden_dim]
            for _ in 0..d {
                frames.push(h.shallow_clone());
            }
        }
    }
    Tensor::stack(&frames, 0).view([batch, total_frames, hidden_dim])
}

/// Full inference: tokens -> mel spectrogram
#[cfg(feature = "torch")]
pub struct StyleTTS2Inference {
    pub text_encoder: TextEncoder,
    pub dur_predictor: DurationPredictor,
    pub speaker_embed: SpeakerEmbedding,
    pub mel_decoder:   MelDecoder,
    pub(crate) vs: tch::nn::VarStore,
    pub config: StyleTTS2Config,
    pub device: Device,
}

#[cfg(feature = "torch")]
impl StyleTTS2Inference {
    pub fn new(safetensors_path: &std::path::Path, device: Device) -> Result<Self> {
        let config = StyleTTS2Config::default();
        let mut vs = tch::nn::VarStore::new(device);
        let root = vs.root();

        let text_encoder  = TextEncoder::new(&(&root / "text_encoder"),  &config);
        let dur_predictor = DurationPredictor::new(&(&root / "duration_predictor"), &config);
        let speaker_embed = SpeakerEmbedding::new(
            &(&root / "speaker_embedding"),
            config.n_speakers,
            config.style_dim,
        );
        let mel_decoder = MelDecoder::new(&(&root / "decoder"), &config);

        let unmatched = vs
            .load_partial(safetensors_path)
            .with_context(|| format!("Failed to load {:?}", safetensors_path))?;
        if !unmatched.is_empty() {
            log::warn!("Safetensors file contained {} keys not registered in model: {:?}",
                       unmatched.len(), &unmatched[..unmatched.len().min(5)]);
        }

        log::info!("StyleTTS2 loaded from {:?}", safetensors_path);
        Ok(Self {
            text_encoder,
            dur_predictor,
            speaker_embed,
            mel_decoder,
            vs,
            config,
            device,
        })
    }

    pub fn tokens_to_mel(&self, tokens: &[i64], speaker: Option<i64>) -> Result<Vec<Vec<f32>>> {
        tch::no_grad(|| {
            let tok_tensor = Tensor::of_slice(tokens)
                .to_device(self.device)
                .unsqueeze(0); // [1, seq]

            let hidden    = self.text_encoder.forward(&tok_tensor);                           // [1, seq, 512]
            let durations = self.dur_predictor.forward(&hidden);                              // [1, seq]
            let style     = self.speaker_embed.forward(speaker.unwrap_or(0), self.device);   // [1, 128]
            let expanded  = duration_expand(&hidden, &durations);                             // [1, frames, 512]
            let mel       = self.mel_decoder.forward(&expanded, &style);                      // [1, 80, frames]

            // Convert to Vec<Vec<f32>> [n_mels, frames]
            let mel_cpu = mel.squeeze_dim(0).to_device(Device::Cpu); // [80, frames]
            let shape = mel_cpu.size();
            let n_mels   = shape[0] as usize;
            let n_frames = shape[1] as usize;
            let flat: Vec<f32> = mel_cpu.view([-1]).try_into().context("mel tensor to vec")?;
            let mut result = vec![vec![0.0f32; n_frames]; n_mels];
            for i in 0..n_mels {
                for j in 0..n_frames {
                    result[i][j] = flat[i * n_frames + j];
                }
            }
            Ok(result)
        })
    }
}

// Non-torch stubs for builds without --features torch
#[cfg(not(feature = "torch"))]
pub struct StyleTTS2Config;
#[cfg(not(feature = "torch"))]
impl Default for StyleTTS2Config {
    fn default() -> Self {
        Self
    }
}
#[cfg(not(feature = "torch"))]
pub struct StyleTTS2Inference;
#[cfg(not(feature = "torch"))]
impl StyleTTS2Inference {
    pub fn new(_: &std::path::Path, _: i32) -> anyhow::Result<Self> {
        anyhow::bail!("StyleTTS2 requires --features torch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "torch")]
    #[test]
    fn test_text_encoder_output_shape() {
        use tch::{Device, Tensor, Kind};
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let config = StyleTTS2Config::default();
        let encoder = TextEncoder::new(&vs.root(), &config);
        // batch=1, seq_len=5 token IDs all in [0, n_token)
        let tokens = Tensor::of_slice(&[10i64, 20, 30, 40, 50]).unsqueeze(0);
        let out = encoder.forward(&tokens);
        // expect [1, seq_len=5, hidden_dim=512]
        assert_eq!(out.size(), vec![1, 5, 512]);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn test_duration_predictor_output_shape() {
        use tch::{Device, Tensor};
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let config = StyleTTS2Config::default();
        let pred = DurationPredictor::new(&vs.root(), &config);
        let hidden = Tensor::zeros(&[1, 5, 512], (tch::Kind::Float, Device::Cpu));
        let durs = pred.forward(&hidden);
        // expect [1, 5] durations
        assert_eq!(durs.size(), vec![1, 5]);
        // durations must be positive (Softplus output)
        assert!(durs.min().double_value(&[]) > 0.0);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn test_mel_decoder_output_shape() {
        use tch::{Device, Tensor};
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let config = StyleTTS2Config::default();
        let decoder = MelDecoder::new(&vs.root(), &config);
        // expanded hidden: [1, 50 frames, 512]
        let hidden = Tensor::zeros(&[1, 50, 512], (tch::Kind::Float, Device::Cpu));
        let style = Tensor::zeros(&[1, 128], (tch::Kind::Float, Device::Cpu));
        let mel = decoder.forward(&hidden, &style);
        // expect [1, n_mels=80, 50]
        assert_eq!(mel.size(), vec![1, 80, 50]);
    }
}
