/// ISTFTNet Vocoder - Convert mel-spectrogram to waveform
/// Implements a real tch::nn ISTFTNet vocoder for Kokoro TTS.
#[cfg(feature = "torch")]
use tch::{nn, nn::Module, Device, Tensor, Kind};
#[cfg(feature = "torch")]
use anyhow::Result;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[cfg(feature = "torch")]
#[derive(Debug, Clone)]
pub struct ISTFTNetConfig {
    pub upsample_rates: Vec<i64>,
    pub upsample_kernel_sizes: Vec<i64>,
    pub resblock_kernel_sizes: Vec<i64>,
    pub resblock_dilation_sizes: Vec<Vec<i64>>,
    pub upsample_initial_channel: i64,
    pub gen_istft_n_fft: i64,
    pub gen_istft_hop_size: i64,
}

#[cfg(feature = "torch")]
impl Default for ISTFTNetConfig {
    fn default() -> Self {
        Self {
            upsample_rates: vec![10, 6],
            upsample_kernel_sizes: vec![20, 12],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![
                vec![1, 3, 5],
                vec![1, 3, 5],
                vec![1, 3, 5],
            ],
            upsample_initial_channel: 512,
            gen_istft_n_fft: 20,
            gen_istft_hop_size: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// ResBlock — one residual block for the Multi-Receptive-Field Fusion (MRF)
// ---------------------------------------------------------------------------

#[cfg(feature = "torch")]
struct ResBlock {
    convs: Vec<nn::Conv1D>,
}

#[cfg(feature = "torch")]
impl ResBlock {
    fn new(vs: &nn::Path, channels: i64, kernel: i64, dilations: &[i64]) -> Self {
        let convs = dilations
            .iter()
            .map(|&d| {
                nn::conv1d(
                    vs,
                    channels,
                    channels,
                    kernel,
                    nn::ConvConfig {
                        padding: (kernel - 1) * d / 2,
                        dilation: d,
                        ..Default::default()
                    },
                )
            })
            .collect();
        Self { convs }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.shallow_clone();
        for conv in &self.convs {
            out = conv.forward(&out.leaky_relu()) + &out;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// ISTFTNet — the neural vocoder
// ---------------------------------------------------------------------------

#[cfg(feature = "torch")]
pub struct ISTFTNet {
    input_proj: nn::Conv1D,
    upsamplers: Vec<nn::ConvTranspose1D>,
    mrf_blocks: Vec<Vec<ResBlock>>,
    output_proj: nn::Conv1D,
    config: ISTFTNetConfig,
    device: Device,
}

#[cfg(feature = "torch")]
impl ISTFTNet {
    pub fn new(vs: &nn::Path, cfg: &ISTFTNetConfig) -> Self {
        let device = vs.device();

        // Input projection: n_mels=80 → upsample_initial_channel
        let input_proj = nn::conv1d(
            &(vs / "input_proj"),
            80,
            cfg.upsample_initial_channel,
            7,
            nn::ConvConfig {
                padding: 3,
                ..Default::default()
            },
        );

        let mut ch = cfg.upsample_initial_channel;
        let mut upsamplers = Vec::new();
        let mut mrf_blocks = Vec::new();

        for (i, (&rate, &ksize)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let out_ch = ch / 2;
            upsamplers.push(nn::conv_transpose1d(
                &(vs / format!("up_{}", i)),
                ch,
                out_ch,
                ksize,
                nn::ConvTransposeConfig {
                    stride: rate,
                    padding: (ksize - rate) / 2,
                    ..Default::default()
                },
            ));

            // MRF: one ResBlock per kernel size
            let blocks: Vec<ResBlock> = cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
                .enumerate()
                .map(|(j, (&k, dilations))| {
                    ResBlock::new(
                        &(vs / format!("mrf_{}_{}", i, j)),
                        out_ch,
                        k,
                        dilations,
                    )
                })
                .collect();
            mrf_blocks.push(blocks);
            ch = out_ch;
        }

        // ch is now 128 after two upsamplers (512 → 256 → 128)
        // Output channels: (n_fft/2 + 1) * 2 for real + imag
        let out_channels = (cfg.gen_istft_n_fft / 2 + 1) * 2;
        let output_proj = nn::conv1d(
            &(vs / "output_proj"),
            ch,
            out_channels,
            1,
            Default::default(),
        );

        Self {
            input_proj,
            upsamplers,
            mrf_blocks,
            output_proj,
            config: cfg.clone(),
            device,
        }
    }

    pub fn forward(&self, mel: &Tensor) -> Tensor {
        // mel: [batch, 80, mel_frames]
        let mut x = self.input_proj.forward(mel).leaky_relu();

        for (up, mrf_stack) in self.upsamplers.iter().zip(self.mrf_blocks.iter()) {
            x = up.forward(&x.leaky_relu());
            // MRF: average across all residual block kernels
            let sum: Tensor = mrf_stack
                .iter()
                .map(|b| b.forward(&x))
                .reduce(|a, b| a + b)
                .unwrap();
            x = sum / (mrf_stack.len() as f64);
        }

        // Output projection → complex STFT representation
        let x = self.output_proj.forward(&x.leaky_relu());
        // x: [batch, (n_fft/2+1)*2, upsampled_frames]
        let batch = x.size()[0];
        let frames = x.size()[2];
        let n_bins = self.config.gen_istft_n_fft / 2 + 1;

        // Reshape to [batch, n_bins, frames, 2] then apply tanh
        let stft = x.view([batch, n_bins, frames, 2]).tanh();

        // Extract real and imag components
        let real = stft.i((.., .., .., 0i64));
        let imag = stft.i((.., .., .., 1i64));
        let complex = Tensor::stack(&[real, imag], -1); // [batch, n_bins, frames, 2]

        // ISTFT
        let n_fft = self.config.gen_istft_n_fft;
        let hop = self.config.gen_istft_hop_size;
        let window = Tensor::hann_window(n_fft, (Kind::Float, self.device));

        complex.istft(
            n_fft,
            hop.into(),
            n_fft.into(),
            Some(&window),
            false,
            false,
            true,
            None::<i64>,
            false,
        )
        // → waveform [batch, n_samples]
    }
}

// ---------------------------------------------------------------------------
// ISTFTNetVocoder — public wrapper that owns the VarStore
// ---------------------------------------------------------------------------

#[cfg(feature = "torch")]
pub struct ISTFTNetVocoder {
    net: ISTFTNet,
    _vs: tch::nn::VarStore,
    device: Device,
}

#[cfg(feature = "torch")]
impl ISTFTNetVocoder {
    /// Create a new vocoder.
    ///
    /// * `weights_path` – optional path to a safetensors file; weights will be
    ///   partially loaded when provided and the file exists.
    /// * `device`       – computation device.
    /// * `_sample_rate` – kept for API compatibility; not used internally.
    pub fn new(
        weights_path: Option<&std::path::Path>,
        device: Device,
        _sample_rate: i64,
    ) -> Result<Self> {
        let config = ISTFTNetConfig::default();
        let mut vs = tch::nn::VarStore::new(device);
        let net = ISTFTNet::new(&(vs.root() / "vocoder"), &config);

        if let Some(path) = weights_path {
            if path.exists() {
                let unmatched = vs.load_partial(path)?;
                if !unmatched.is_empty() {
                    log::warn!(
                        "Vocoder: some registered vars not found in weights: {:?}",
                        unmatched
                    );
                }
            } else {
                log::warn!("Vocoder weights not found at {:?}, using random weights", path);
            }
        }

        Ok(Self {
            net,
            _vs: vs,
            device,
        })
    }

    /// Convert a mel-spectrogram to audio samples.
    ///
    /// `mel` is `[n_mels][n_frames]` (row-major: each inner Vec is one mel bin).
    pub fn mel_to_audio(&self, mel: &Vec<Vec<f32>>, device: Device) -> Result<Vec<f32>> {
        let n_mels = mel.len();
        let n_frames = mel
            .first()
            .ok_or_else(|| anyhow::anyhow!("empty mel-spectrogram"))?
            .len();

        let flat: Vec<f32> = mel.iter().flat_map(|r| r.iter().copied()).collect();
        let mel_tensor = Tensor::of_slice(&flat)
            .to_device(device)
            .view([1, n_mels as i64, n_frames as i64]);

        let audio = tch::no_grad(|| self.net.forward(&mel_tensor));
        let audio_cpu = audio.squeeze_dim(0).to_device(Device::Cpu);
        Ok(audio_cpu.try_into()?)
    }
}

// ---------------------------------------------------------------------------
// Non-torch stubs
// ---------------------------------------------------------------------------

#[cfg(not(feature = "torch"))]
pub struct ISTFTNetConfig;

#[cfg(not(feature = "torch"))]
impl Default for ISTFTNetConfig {
    fn default() -> Self {
        Self
    }
}

#[cfg(not(feature = "torch"))]
pub struct ISTFTNetVocoder;

#[cfg(not(feature = "torch"))]
impl ISTFTNetVocoder {
    pub fn new(
        _weights_path: Option<&std::path::Path>,
        _device: i32,
        _sample_rate: i64,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("ISTFTNet requires --features torch")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "torch")]
    #[test]
    fn test_istftnet_output_is_1d_audio() {
        use tch::{Device, Tensor};
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let config = ISTFTNetConfig::default();
        let net = ISTFTNet::new(&vs.root(), &config);
        // mel: [1, 80, 10 frames]
        let mel = Tensor::zeros(&[1, 80, 10], (tch::Kind::Float, Device::Cpu));
        let audio = net.forward(&mel);
        // Should be [1, n_samples]
        assert_eq!(audio.size().len(), 2);
        assert_eq!(audio.size()[0], 1);
        assert!(audio.size()[1] > 0);
    }

    #[cfg(feature = "torch")]
    #[test]
    fn test_istftnet_no_nan_output() {
        use tch::{Device, Tensor};
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let config = ISTFTNetConfig::default();
        let net = ISTFTNet::new(&vs.root(), &config);
        let mel = Tensor::randn(&[1, 80, 20], (tch::Kind::Float, Device::Cpu));
        let audio = net.forward(&mel);
        let has_nan = audio.isnan().any().int64_value(&[]) != 0;
        assert!(!has_nan, "ISTFTNet output contains NaN");
    }
}
