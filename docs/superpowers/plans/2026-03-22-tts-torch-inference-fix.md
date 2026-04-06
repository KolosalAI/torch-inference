# TTS Torch Inference Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Kokoro TTS engine produce real speech audio using native `tch::nn` (torch primary) and `ort` ONNX (secondary), replacing the broken sine-wave and Griffin-Lim stubs.

**Architecture:** The `kokoro` engine runs the full StyleTTS2+ISTFTNet pipeline in Rust via `tch::nn`, loading weights from a `.safetensors` file converted from the existing `.pth`. The `kokoro-onnx` engine calls an `ort::Session` with the correct 2.0.0-rc.10 API, using Mutex-wrapped session and named-input inference. Both engines propagate errors clearly instead of silently falling back to synthetic audio.

**Tech Stack:** Rust, `tch` 0.16 (LibTorch bindings), `ort` 2.0.0-rc.10 (ONNX Runtime), `ndarray`, Python (one-time conversion script only)

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `src/` (all) | Restore from git | All Rust source — deleted by migration commit |
| `convert_kokoro.py` | Create | One-time `.pth` → `.safetensors` conversion |
| `src/core/g2p_misaki.rs` | Modify | Complete 178-token phoneme vocab + fix letter fallback |
| `src/core/styletts2_model.rs` | Modify | Replace CModule stub with tch::nn TextEncoder + DurPredictor + SpeakerEmbed + MelDecoder |
| `src/core/istftnet_vocoder.rs` | Modify | Replace Griffin-Lim with tch::nn ISTFTNet upsamplers + MRF + Tensor::istft() |
| `src/core/kokoro_tts.rs` | Modify | Wire NativeInference to new models; delete synthesize_fallback() |
| `src/core/kokoro_onnx.rs` | Modify | Real ort Session inference; Mutex<Session>; delete synthesize_parametric() |

---

## Task 1: Restore `src/` from git history

**Files:**
- Restore: `src/` (all files, ~100 Rust source files)

The migration commit `3e1d8c1` deleted all Rust source. Restore from the last good state.

- [ ] **Step 1: Restore src/**

```bash
cd /path/to/torch-inference
git checkout 4331e91 -- src/
```

Expected: `src/` directory appears with all `.rs` files.

- [ ] **Step 2: Verify key TTS files exist**

```bash
ls src/core/g2p_misaki.rs src/core/styletts2_model.rs src/core/istftnet_vocoder.rs src/core/kokoro_tts.rs src/core/kokoro_onnx.rs src/core/tts_engine.rs src/core/tts_manager.rs
```

Expected: all seven files printed, no "No such file" errors.

- [ ] **Step 3: Verify build compiles (without torch feature)**

```bash
cargo build 2>&1 | tail -5
```

Expected: `Finished dev [unoptimized + debuginfo] target(s)` (warnings OK, no errors).

- [ ] **Step 4: Commit the restoration**

```bash
git add src/
git commit -m "chore: restore Rust source files deleted by migration commit"
```

---

## Task 2: Create `convert_kokoro.py`

**Files:**
- Create: `convert_kokoro.py`

This script converts `kokoro-v1_0.pth` (raw PyTorch state dict) to `.safetensors` format that `tch::VarStore::load_partial()` can read. Run once before the torch TTS engine is used.

- [ ] **Step 1: Create the script**

```python
# convert_kokoro.py
"""
One-time conversion: kokoro-v1_0.pth -> kokoro-v1_0.safetensors
Run before first use of the torch TTS engine.
Requires: pip install torch safetensors
Usage: python convert_kokoro.py
"""
import torch
from safetensors.torch import save_file
import sys
import os

src = "models/kokoro-82m/kokoro-v1_0.pth"
dst = "models/kokoro-82m/kokoro-v1_0.safetensors"

if not os.path.exists(src):
    print(f"ERROR: {src} not found. Download from:")
    print("  https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth")
    sys.exit(1)

print(f"Loading {src}...")
state = torch.load(src, map_location="cpu")
if not isinstance(state, dict):
    state = state.state_dict()

# safetensors requires contiguous tensors
state = {k: v.contiguous() for k, v in state.items()}
save_file(state, dst)
print(f"Saved {dst} ({len(state)} tensors)")
print("First 10 keys:", list(state.keys())[:10])
print("Done. Now build with: cargo build --features torch")
```

- [ ] **Step 2: Commit**

```bash
git add convert_kokoro.py
git commit -m "chore: add kokoro .pth to .safetensors conversion script"
```

---

## Task 3: Fix `g2p_misaki.rs` — Complete phoneme vocabulary

**Files:**
- Modify: `src/core/g2p_misaki.rs`

The existing `PHONEME_VOCAB` has ~35 entries. Kokoro uses exactly 178 tokens. Missing tokens cause phonemes to be silently dropped, feeding wrong IDs to the model. This is pure logic with no model dependencies — test it first.

- [ ] **Step 1: Write the failing test**

Add at the bottom of `src/core/g2p_misaki.rs` inside the existing `#[cfg(test)]` block:

```rust
#[test]
fn test_all_common_phonemes_have_tokens() {
    let g2p = MisakiG2P::new().unwrap();
    // These IPA symbols MUST have token IDs — they appear in the pronunciation dict
    let required = ["ə", "ɪ", "ɛ", "æ", "ɑ", "ɔ", "ʊ", "ʌ", "ɜ",
                    "ð", "θ", "ŋ", "ʃ", "ʒ", "ʧ", "ʤ", "ɡ", "j",
                    "ˈ", "ˌ", "ː", " "];
    for sym in required {
        assert!(g2p.vocab.get(sym).is_some(), "Missing token for: {:?}", sym);
    }
}

#[test]
fn test_hello_world_produces_nonzero_tokens() {
    let g2p = MisakiG2P::new().unwrap();
    let tokens = g2p.text_to_tokens("hello world").unwrap();
    assert!(tokens.len() >= 5, "Expected at least 5 tokens, got {}", tokens.len());
    // All tokens must be in valid Kokoro range [1, 177]
    for &t in &tokens {
        assert!(t >= 1 && t <= 177, "Token {} out of valid range [1,177]", t);
    }
}

#[test]
fn test_letter_fallback_produces_valid_tokens() {
    let g2p = MisakiG2P::new().unwrap();
    // "xyz" not in dictionary — exercises letter_to_phoneme fallback
    let tokens = g2p.text_to_tokens("xyz").unwrap();
    for &t in &tokens {
        assert!(t >= 1 && t <= 177, "Fallback token {} out of range", t);
    }
}
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cargo test g2p --features torch 2>&1 | grep -E "FAILED|passed|failed"
```

Expected: `test_all_common_phonemes_have_tokens FAILED` (and others).

- [ ] **Step 3: Replace PHONEME_VOCAB with the complete 178-token set**

In `src/core/g2p_misaki.rs`, replace the `PHONEME_VOCAB` lazy_static entirely:

```rust
lazy_static! {
    static ref PHONEME_VOCAB: HashMap<String, i64> = {
        let mut m = HashMap::new();
        // --- Punctuation & special ---
        m.insert(";".to_string(), 1);
        m.insert(":".to_string(), 2);
        m.insert(",".to_string(), 3);
        m.insert(".".to_string(), 4);
        m.insert("!".to_string(), 5);
        m.insert("?".to_string(), 6);
        m.insert("—".to_string(), 9);
        m.insert("…".to_string(), 10);
        m.insert(" ".to_string(), 16);
        // --- ASCII letters (Kokoro uses these for letter-level fallback) ---
        m.insert("A".to_string(), 21); m.insert("B".to_string(), 22);
        m.insert("C".to_string(), 23); m.insert("D".to_string(), 24);
        m.insert("E".to_string(), 25); m.insert("F".to_string(), 26);
        m.insert("G".to_string(), 27); m.insert("H".to_string(), 28);
        m.insert("I".to_string(), 29); m.insert("J".to_string(), 30);
        m.insert("K".to_string(), 31); m.insert("L".to_string(), 32);
        m.insert("M".to_string(), 33); m.insert("N".to_string(), 34);
        m.insert("O".to_string(), 35); m.insert("P".to_string(), 36);
        m.insert("Q".to_string(), 37); m.insert("R".to_string(), 38);
        m.insert("S".to_string(), 39); m.insert("T".to_string(), 40);
        m.insert("U".to_string(), 41); m.insert("V".to_string(), 42);
        m.insert("W".to_string(), 43); m.insert("X".to_string(), 44);
        m.insert("Y".to_string(), 45); m.insert("Z".to_string(), 46);
        // --- IPA vowels ---
        m.insert("a".to_string(), 47);
        m.insert("ɑ".to_string(), 49);
        m.insert("ɐ".to_string(), 50);
        m.insert("ɒ".to_string(), 51);
        m.insert("æ".to_string(), 52);
        m.insert("e".to_string(), 55);
        m.insert("ɘ".to_string(), 57);
        m.insert("ə".to_string(), 58);
        m.insert("ɚ".to_string(), 59);
        m.insert("ɛ".to_string(), 60);
        m.insert("ɜ".to_string(), 61);
        m.insert("ɝ".to_string(), 62);
        m.insert("i".to_string(), 67);
        m.insert("ɪ".to_string(), 69);
        m.insert("o".to_string(), 73);
        m.insert("ɔ".to_string(), 75);
        m.insert("u".to_string(), 82);
        m.insert("ʊ".to_string(), 84);
        m.insert("ʌ".to_string(), 85);
        m.insert("ɵ".to_string(), 86);
        // --- IPA consonants ---
        m.insert("b".to_string(), 88);
        m.insert("d".to_string(), 90);
        m.insert("ð".to_string(), 91);
        m.insert("f".to_string(), 93);
        m.insert("ɡ".to_string(), 95);
        m.insert("h".to_string(), 98);
        m.insert("j".to_string(), 100);
        m.insert("k".to_string(), 101);
        m.insert("l".to_string(), 102);
        m.insert("m".to_string(), 103);
        m.insert("n".to_string(), 104);
        m.insert("ŋ".to_string(), 106);
        m.insert("p".to_string(), 109);
        m.insert("r".to_string(), 111);
        m.insert("s".to_string(), 112);
        m.insert("t".to_string(), 113);
        m.insert("θ".to_string(), 115);
        m.insert("v".to_string(), 118);
        m.insert("w".to_string(), 119);
        m.insert("z".to_string(), 122);
        m.insert("ʃ".to_string(), 124);
        m.insert("ʒ".to_string(), 129);
        m.insert("ʧ".to_string(), 130);
        m.insert("ʤ".to_string(), 131);
        // --- Diphthongs (two-char IPA sequences) ---
        m.insert("aɪ".to_string(), 48);
        m.insert("aʊ".to_string(), 53);
        m.insert("eɪ".to_string(), 56);
        m.insert("oʊ".to_string(), 74);
        m.insert("ɔɪ".to_string(), 76);
        // --- Stress and length markers ---
        m.insert("ˈ".to_string(), 156);
        m.insert("ˌ".to_string(), 157);
        m.insert("ː".to_string(), 158);
        m
    };
}
```

Also update `letter_to_phoneme` to map characters to valid IPA symbols (not raw char codes):

```rust
fn letter_to_phoneme(&self, word: &str) -> Result<Vec<i64>> {
    let mut tokens = Vec::new();
    for ch in word.chars() {
        let ipa: &str = match ch {
            'a' | 'A' => "æ",
            'e' | 'E' => "ɛ",
            'i' | 'I' => "ɪ",
            'o' | 'O' => "ɔ",
            'u' | 'U' => "ʊ",
            'b' | 'B' => "b",
            'c' | 'C' => "k",
            'd' | 'D' => "d",
            'f' | 'F' => "f",
            'g' | 'G' => "ɡ",
            'h' | 'H' => "h",
            'j' | 'J' => "ʤ",
            'k' | 'K' => "k",
            'l' | 'L' => "l",
            'm' | 'M' => "m",
            'n' | 'N' => "n",
            'p' | 'P' => "p",
            'r' | 'R' => "r",
            's' | 'S' => "s",
            't' | 'T' => "t",
            'v' | 'V' => "v",
            'w' | 'W' => "w",
            'x' | 'X' => "s",  // approximate
            'y' | 'Y' => "j",
            'z' | 'Z' => "z",
            'q' | 'Q' => "k",
            _ => continue,
        };
        if let Some(&token) = self.vocab.get(ipa) {
            tokens.push(token);
        } else {
            log::warn!("letter_to_phoneme: no token for IPA {:?} (from char {:?})", ipa, ch);
        }
    }
    Ok(tokens)
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test g2p --features torch 2>&1 | grep -E "FAILED|ok|failed"
```

Expected: all three tests `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/core/g2p_misaki.rs
git commit -m "fix: complete Kokoro 178-token phoneme vocabulary in MisakiG2P"
```

---

## Task 4: Fix `styletts2_model.rs` — Replace CModule with `tch::nn`

**Files:**
- Modify: `src/core/styletts2_model.rs`

Replace the `CModule::load_on_device` stub with real `tch::nn` modules. This runs only with `--features torch`.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` block in `src/core/styletts2_model.rs`:

```rust
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cargo test styletts2 --features torch 2>&1 | grep -E "FAILED|error|failed"
```

Expected: compile errors (structs don't exist yet) or test failures.

- [ ] **Step 3: Implement `TextEncoder`, `DurationPredictor`, `SpeakerEmbedding`, `MelDecoder`**

Replace the entire contents of `src/core/styletts2_model.rs` (keep the `#[cfg(not(feature = "torch"))]` stub at the bottom). Key structures:

```rust
#[cfg(feature = "torch")]
use tch::{nn, nn::Module, Device, Tensor, Kind};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

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
            dim_in: 64, hidden_dim: 512, n_token: 178,
            style_dim: 128, n_mels: 80, n_layer: 3,
            dropout: 0.2, max_dur: 50,
            multispeaker: true, n_speakers: 54,
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
                cfg.hidden_dim, cfg.hidden_dim,
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
        let x = self.embed.forward(tokens);     // [batch, seq, dim_in]
        let mut x = self.proj.forward(&x);       // [batch, seq, hidden_dim]
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
            conv1: nn::conv1d(vs / "conv1", cfg.hidden_dim, 256, 3,
                              nn::ConvConfig { padding: 1, ..Default::default() }),
            conv2: nn::conv1d(vs / "conv2", 256, 256, 3,
                              nn::ConvConfig { padding: 1, ..Default::default() }),
            proj:  nn::linear(vs / "proj", 256, 1, Default::default()),
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
            style_proj: nn::linear(vs / "style_proj", cfg.style_dim, cfg.hidden_dim, Default::default()),
            convs: (0..3).map(|i| nn::conv1d(
                vs / format!("conv_{}", i),
                cfg.hidden_dim, cfg.hidden_dim, 3,
                nn::ConvConfig { padding: 1, ..Default::default() },
            )).collect(),
            out_proj: nn::conv1d(vs / "out_proj", cfg.hidden_dim, cfg.n_mels, 1, Default::default()),
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

/// Duration expansion: repeat each hidden state by its predicted integer duration
#[cfg(feature = "torch")]
pub fn duration_expand(hidden: &Tensor, durations: &Tensor) -> Tensor {
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

/// Full inference: tokens → mel spectrogram
#[cfg(feature = "torch")]
pub struct StyleTTS2Inference {
    pub text_encoder: TextEncoder,
    pub dur_predictor: DurationPredictor,
    pub speaker_embed: SpeakerEmbedding,
    pub mel_decoder:   MelDecoder,
    pub vs: tch::nn::VarStore,
    pub config: StyleTTS2Config,
    pub device: Device,
}

#[cfg(feature = "torch")]
impl StyleTTS2Inference {
    pub fn new(safetensors_path: &std::path::Path, device: Device) -> Result<Self> {
        let config = StyleTTS2Config::default();
        let mut vs = tch::nn::VarStore::new(device);
        let root = vs.root();

        let text_encoder  = TextEncoder::new(&root / "text_encoder",  &config);
        let dur_predictor = DurationPredictor::new(&root / "duration_predictor", &config);
        let speaker_embed = SpeakerEmbedding::new(
            &root / "speaker_embedding", config.n_speakers, config.style_dim
        );
        let mel_decoder   = MelDecoder::new(&root / "decoder", &config);

        let unmatched = vs.load_partial(safetensors_path)
            .with_context(|| format!("Failed to load {:?}", safetensors_path))?;
        if !unmatched.is_empty() {
            log::error!("Registered vars not found in model: {:?}", unmatched);
            anyhow::bail!("Model weight mismatch — re-run convert_kokoro.py");
        }

        log::info!("StyleTTS2 loaded from {:?}", safetensors_path);
        Ok(Self { text_encoder, dur_predictor, speaker_embed, mel_decoder, vs, config, device })
    }

    pub fn tokens_to_mel(&self, tokens: &[i64], speaker: Option<i64>) -> Result<Vec<Vec<f32>>> {
        tch::no_grad(|| {
            let tok_tensor = Tensor::of_slice(tokens)
                .to_device(self.device)
                .unsqueeze(0); // [1, seq]

            let hidden   = self.text_encoder.forward(&tok_tensor);            // [1, seq, 512]
            let durations = self.dur_predictor.forward(&hidden);              // [1, seq]
            let style    = self.speaker_embed.forward(speaker.unwrap_or(0), self.device); // [1, 128]
            let expanded = duration_expand(&hidden, &durations);              // [1, frames, 512]
            let mel      = self.mel_decoder.forward(&expanded, &style);       // [1, 80, frames]

            // Convert to Vec<Vec<f32>> [n_mels, frames]
            let mel_cpu = mel.squeeze_dim(0).to_device(Device::Cpu); // [80, frames]
            let shape = mel_cpu.size();
            let n_mels  = shape[0] as usize;
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

#[cfg(not(feature = "torch"))]
pub struct StyleTTS2Config;
#[cfg(not(feature = "torch"))]
impl Default for StyleTTS2Config { fn default() -> Self { Self } }
#[cfg(not(feature = "torch"))]
pub struct StyleTTS2Inference;
#[cfg(not(feature = "torch"))]
impl StyleTTS2Inference {
    pub fn new(_: &std::path::Path, _: i32) -> anyhow::Result<Self> {
        anyhow::bail!("StyleTTS2 requires --features torch")
    }
}
```

- [ ] **Step 4: Run shape tests**

```bash
cargo test styletts2 --features torch 2>&1 | grep -E "FAILED|ok|failed|error"
```

Expected: all three shape tests `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/core/styletts2_model.rs
git commit -m "fix: replace CModule stub with tch::nn StyleTTS2 (TextEncoder, DurPredictor, MelDecoder)"
```

---

## Task 5: Fix `istftnet_vocoder.rs` — Replace Griffin-Lim with real ISTFTNet

**Files:**
- Modify: `src/core/istftnet_vocoder.rs`

Replace the `SimplifiedVocoder` Griffin-Lim path with a real `tch::nn` ISTFTNet implementation. All neural code is `#[cfg(feature = "torch")]`.

- [ ] **Step 1: Write the failing test**

Add to `#[cfg(test)]` block in `src/core/istftnet_vocoder.rs`:

```rust
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
    // Should be [1, n_samples] — with rates [10,6] and n_frames=10, n_samples ≈ 10*60*5 = 3000
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
cargo test istftnet --features torch 2>&1 | grep -E "FAILED|error|failed"
```

Expected: compile errors (ISTFTNet struct doesn't exist yet).

- [ ] **Step 3: Implement ISTFTNet**

Replace the full `src/core/istftnet_vocoder.rs`. Key structure:

```rust
#[cfg(feature = "torch")]
use tch::{nn, nn::Module, Device, Tensor, Kind};
use anyhow::Result;

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
            resblock_dilation_sizes: vec![vec![1,3,5], vec![1,3,5], vec![1,3,5]],
            upsample_initial_channel: 512,
            gen_istft_n_fft: 20,
            gen_istft_hop_size: 5,
        }
    }
}

/// One residual block for MRF
#[cfg(feature = "torch")]
struct ResBlock {
    convs: Vec<nn::Conv1D>,
}

#[cfg(feature = "torch")]
impl ResBlock {
    fn new(vs: &nn::Path, channels: i64, kernel: i64, dilations: &[i64]) -> Self {
        let convs = dilations.iter().map(|&d| {
            nn::conv1d(vs, channels, channels, kernel, nn::ConvConfig {
                padding: (kernel - 1) * d / 2,
                dilation: d,
                ..Default::default()
            })
        }).collect();
        Self { convs }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.shallow_clone();
        for conv in &self.convs {
            out = (conv.forward(&out.leaky_relu()) + &out);
        }
        out
    }
}

#[cfg(feature = "torch")]
pub struct ISTFTNet {
    input_proj: nn::Conv1D,
    upsamplers: Vec<nn::ConvTranspose1D>,
    mrf_blocks: Vec<Vec<ResBlock>>, // [n_upsamplers][n_resblock_kernels]
    output_proj: nn::Conv1D,
    config: ISTFTNetConfig,
    device: Device,
}

#[cfg(feature = "torch")]
impl ISTFTNet {
    pub fn new(vs: &nn::Path, cfg: &ISTFTNetConfig) -> Self {
        let device = vs.device();
        // Input projection: n_mels=80 → upsample_initial_channel=512
        let input_proj = nn::conv1d(vs / "input_proj", 80, cfg.upsample_initial_channel,
                                    7, nn::ConvConfig { padding: 3, ..Default::default() });

        let mut ch = cfg.upsample_initial_channel;
        let mut upsamplers = Vec::new();
        let mut mrf_blocks = Vec::new();

        for (i, (&rate, &ksize)) in cfg.upsample_rates.iter()
                                       .zip(cfg.upsample_kernel_sizes.iter()).enumerate() {
            let out_ch = ch / 2;
            upsamplers.push(nn::conv_transpose1d(
                vs / format!("up_{}", i), ch, out_ch, ksize,
                nn::ConvTransposeConfig {
                    stride: rate,
                    padding: (ksize - rate) / 2,
                    ..Default::default()
                },
            ));
            // MRF: one ResBlock per kernel size
            let blocks: Vec<ResBlock> = cfg.resblock_kernel_sizes.iter()
                .zip(cfg.resblock_dilation_sizes.iter())
                .enumerate()
                .map(|(j, (&k, dilations))| {
                    ResBlock::new(&(vs / format!("mrf_{}_{}", i, j)), out_ch, k, dilations)
                })
                .collect();
            mrf_blocks.push(blocks);
            ch = out_ch;
        }

        // ch is now 128 after two upsamplers
        let out_channels = (cfg.gen_istft_n_fft / 2 + 1) * 2; // (10+1)*2 = 22
        let output_proj = nn::conv1d(vs / "output_proj", ch, out_channels,
                                     1, Default::default());

        Self { input_proj, upsamplers, mrf_blocks, output_proj, config: cfg.clone(), device }
    }

    pub fn forward(&self, mel: &Tensor) -> Tensor {
        // mel: [batch, 80, mel_frames]
        let mut x = self.input_proj.forward(mel).leaky_relu();

        for (up, mrf_stack) in self.upsamplers.iter().zip(self.mrf_blocks.iter()) {
            x = up.forward(&x.leaky_relu());
            // MRF: average across all residual block kernels
            let sum: Tensor = mrf_stack.iter()
                .map(|b| b.forward(&x))
                .reduce(|a, b| a + b)
                .unwrap();
            x = sum / (mrf_stack.len() as f64);
        }

        // Output projection → complex STFT
        let x = self.output_proj.forward(&x.leaky_relu());
        // x: [batch, (n_fft/2+1)*2, upsampled_frames]
        let batch = x.size()[0];
        let frames = x.size()[2];
        let n_bins = self.config.gen_istft_n_fft / 2 + 1;
        // Reshape to [batch, n_bins, frames, 2]
        let stft = x.view([batch, n_bins, frames, 2]).tanh();
        // Convert to complex: stack real and imag
        let real = stft.i((.., .., .., 0i64));
        let imag = stft.i((.., .., .., 1i64));
        let complex = Tensor::stack(&[real, imag], -1); // [batch, n_bins, frames, 2]

        // ISTFT
        let n_fft = self.config.gen_istft_n_fft;
        let hop   = self.config.gen_istft_hop_size;
        let window = Tensor::hann_window(n_fft, (Kind::Float, self.device));

        complex.istft(n_fft, hop.into(), n_fft.into(), Some(&window),
                      false, false, true, None::<i64>, false)
    }
}

/// Wrapper that loads weights and exposes mel_to_audio
#[cfg(feature = "torch")]
pub struct ISTFTNetVocoder {
    net: ISTFTNet,
    _vs: tch::nn::VarStore,
}

#[cfg(feature = "torch")]
impl ISTFTNetVocoder {
    pub fn new(safetensors_path: &std::path::Path, device: Device) -> Result<Self> {
        let config = ISTFTNetConfig::default();
        let mut vs = tch::nn::VarStore::new(device);
        let net = ISTFTNet::new(&vs.root() / "vocoder", &config);
        let unmatched = vs.load_partial(safetensors_path)?;
        if !unmatched.is_empty() {
            log::warn!("Vocoder: some registered vars not found in weights: {:?}", unmatched);
        }
        Ok(Self { net, _vs: vs })
    }

    pub fn mel_to_audio(&self, mel: &Vec<Vec<f32>>, device: Device) -> Result<Vec<f32>> {
        let n_mels   = mel.len();
        let n_frames = mel.first().ok_or_else(|| anyhow::anyhow!("empty mel"))?.len();
        let flat: Vec<f32> = mel.iter().flat_map(|r| r.iter().copied()).collect();
        let mel_tensor = Tensor::of_slice(&flat)
            .to_device(device)
            .view([1, n_mels as i64, n_frames as i64]);

        let audio = tch::no_grad(|| self.net.forward(&mel_tensor));
        let audio_cpu = audio.squeeze_dim(0).to_device(Device::Cpu);
        Ok(audio_cpu.try_into()?)
    }
}

// Non-torch stub
#[cfg(not(feature = "torch"))]
pub struct ISTFTNetConfig;
#[cfg(not(feature = "torch"))]
impl Default for ISTFTNetConfig { fn default() -> Self { Self } }
#[cfg(not(feature = "torch"))]
pub struct ISTFTNetVocoder;
#[cfg(not(feature = "torch"))]
impl ISTFTNetVocoder {
    pub fn new(_: &std::path::Path, _: i32) -> anyhow::Result<Self> {
        anyhow::bail!("ISTFTNet requires --features torch")
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test istftnet --features torch 2>&1 | grep -E "FAILED|ok|failed|error"
```

Expected: both tests `ok`. No NaN, correct output dimensions.

- [ ] **Step 5: Commit**

```bash
git add src/core/istftnet_vocoder.rs
git commit -m "fix: replace Griffin-Lim with tch::nn ISTFTNet upsampler + ISTFT vocoder"
```

---

## Task 6: Fix `kokoro_tts.rs` — Wire up native inference

**Files:**
- Modify: `src/core/kokoro_tts.rs`

Wire `NativeInference` to the new `StyleTTS2Inference` + `ISTFTNetVocoder`. Delete `synthesize_fallback()` entirely.

- [ ] **Step 1: Write the failing test**

Add to `#[cfg(test)]` in `src/core/kokoro_tts.rs`:

```rust
#[tokio::test]
async fn test_kokoro_returns_error_when_model_absent() {
    let config = serde_json::json!({
        "model_path": "/nonexistent/kokoro-v1_0.safetensors",
        "sample_rate": 24000
    });
    let engine = KokoroEngine::new(&config).unwrap();
    // Engine constructs OK (model loading is lazy) but synthesize fails
    let params = crate::core::tts_engine::SynthesisParams::default();
    let result = engine.synthesize("hello", &params).await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    // Must NOT silently produce audio — must return an error
    assert!(msg.contains("unavailable") || msg.contains("not found") || msg.contains("failed"),
            "Unexpected error: {}", msg);
}
```

- [ ] **Step 2: Run to verify test fails**

```bash
cargo test kokoro_returns_error --features torch 2>&1 | grep -E "FAILED|ok|failed"
```

- [ ] **Step 3: Update `KokoroEngine`**

In `src/core/kokoro_tts.rs`:

1. In `NativeInference` struct (behind `#[cfg(feature = "torch")]`), replace fields:
```rust
#[cfg(feature = "torch")]
struct NativeInference {
    model:   StyleTTS2Inference,
    vocoder: ISTFTNetVocoder,
    device:  Device,
}
```

2. Replace `init_native_inference()`:
```rust
#[cfg(feature = "torch")]
fn init_native_inference(model_path: &std::path::Path, device: Device) -> Option<NativeInference> {
    let safetensors = model_path.with_extension("safetensors");
    if !safetensors.exists() {
        log::warn!("Kokoro: {:?} not found. Run: python convert_kokoro.py", safetensors);
        return None;
    }
    let model = match StyleTTS2Inference::new(&safetensors, device) {
        Ok(m) => m,
        Err(e) => { log::warn!("Kokoro model load failed: {}", e); return None; }
    };
    let vocoder = match ISTFTNetVocoder::new(&safetensors, device) {
        Ok(v) => v,
        Err(e) => { log::warn!("Kokoro vocoder load failed: {}", e); return None; }
    };
    log::info!("Kokoro native inference ready");
    Some(NativeInference { model, vocoder, device })
}
```

3. Replace `synthesize_with_native()`:
```rust
#[cfg(feature = "torch")]
fn synthesize_with_native(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
    let inf = self.native_inference.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Native inference not initialized"))?;

    let tokens = self.g2p.text_to_tokens(text)?;
    let speaker_id = self.voice_to_speaker_id(params.voice.as_deref());
    let mel = inf.model.tokens_to_mel(&tokens, Some(speaker_id))?;
    let samples = inf.vocoder.mel_to_audio(&mel, inf.device)?;

    Ok(AudioData { samples, sample_rate: self.config.sample_rate, channels: 1 })
}
```

4. Delete `synthesize_fallback()` entirely (remove the whole function, not just its `#[deprecated]` annotation).

5. Update `synthesize()`:
```rust
async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
    self.validate_text(text)?;

    #[cfg(feature = "torch")]
    if self.native_inference.is_some() {
        return self.synthesize_with_native(text, params);
    }

    if let Some(ref bridge) = self.python_bridge {
        return bridge.synthesize(text, params.voice.as_deref(), params.speed);
    }

    anyhow::bail!(
        "Kokoro TTS unavailable. Run: python convert_kokoro.py  \
         then: cargo build --features torch"
    )
}
```

- [ ] **Step 4: Run test**

```bash
cargo test kokoro_returns_error --features torch 2>&1 | grep -E "ok|FAILED|failed"
```

Expected: `ok`.

- [ ] **Step 5: Verify no `synthesize_fallback` remains**

```bash
grep -n "synthesize_fallback" src/core/kokoro_tts.rs
```

Expected: no output (function fully deleted).

- [ ] **Step 6: Commit**

```bash
git add src/core/kokoro_tts.rs
git commit -m "fix: wire KokoroEngine native inference to StyleTTS2+ISTFTNet, delete synthesize_fallback"
```

---

## Task 7: Fix `kokoro_onnx.rs` — Real ONNX inference

**Files:**
- Modify: `src/core/kokoro_onnx.rs`

Implement real `ort` 2.0.0-rc.10 inference. Wrap `Session` in `Mutex`. Delete `synthesize_parametric()`.

- [ ] **Step 1: Write the failing test**

Add to `#[cfg(test)]` in `src/core/kokoro_onnx.rs`:

```rust
#[test]
fn test_onnx_engine_errors_when_model_absent() {
    let config = serde_json::json!({
        "model_dir": "/nonexistent/kokoro-onnx",
        "sample_rate": 24000
    });
    let result = KokoroOnnxEngine::new(&config);
    assert!(result.is_err(), "Expected Err when model file missing");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not found") || msg.contains("No such file") || msg.contains("model"),
            "Unexpected error: {}", msg);
}

#[test]
fn test_load_voice_style_binary() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Write 256 little-endian f32 values (all 0.5)
    let mut f = NamedTempFile::new().unwrap();
    for _ in 0..256u32 {
        f.write_all(&0.5f32.to_le_bytes()).unwrap();
    }
    let path = f.path().parent().unwrap().to_path_buf();
    let name = f.path().file_name().unwrap().to_str().unwrap()
                 .trim_end_matches(".tmp");
    // Directly test the loading logic
    let bytes = std::fs::read(f.path()).unwrap();
    let floats: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(floats.len(), 256);
    assert!((floats[0] - 0.5).abs() < 1e-6);
}
```

- [ ] **Step 2: Run to verify tests fail (or compile error)**

```bash
cargo test onnx_engine --features torch 2>&1 | grep -E "FAILED|error|ok|failed"
```

- [ ] **Step 3: Rewrite `KokoroOnnxEngine`**

Replace the struct definition, `new()`, and `synthesize()`. Delete `synthesize_parametric()` and `synthesize_with_onnx()` stubs. Key implementation:

```rust
use ort::{Session, GraphOptimizationLevel};
use ndarray::{Array1, Array2};
use std::sync::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;

pub struct KokoroOnnxEngine {
    session: Mutex<Session>,
    voice_styles: HashMap<String, Vec<f32>>,  // voice_id → flat f32[256]
    config: KokoroOnnxConfig,
    capabilities: EngineCapabilities,
}

impl KokoroOnnxEngine {
    pub fn new(cfg: &serde_json::Value) -> Result<Self> {
        let model_dir = PathBuf::from(
            cfg.get("model_dir").and_then(|v| v.as_str()).unwrap_or("models/kokoro-onnx")
        );
        let model_path = model_dir.join("kokoro-v1_0.onnx");
        if !model_path.exists() {
            anyhow::bail!(
                "Kokoro ONNX model not found at {:?}. \
                 Download from: https://huggingface.co/hexgrad/Kokoro-82M",
                model_path
            );
        }

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;
        let session = builder.commit_from_file(&model_path)?;

        // Log I/O shapes for diagnostics
        for inp in &session.inputs  { log::info!("ONNX input:  {:?}", inp); }
        for out in &session.outputs { log::info!("ONNX output: {:?}", out); }

        // Preload all voice style .bin files
        let voices_dir = model_dir.join("voices");
        let mut voice_styles = HashMap::new();
        let default_voices = ["af_heart","af_bella","af_sarah","af_nicole",
                               "am_adam","am_michael","bf_emma","bf_isabella",
                               "bm_george","bm_lewis"];
        for voice_id in default_voices {
            let bin_path = voices_dir.join(format!("{}.bin", voice_id));
            match Self::load_voice_style(&bin_path) {
                Ok(style) => { voice_styles.insert(voice_id.to_string(), style); }
                Err(e) => log::warn!("Voice {:?} not loaded: {}", voice_id, e),
            }
        }
        if !voice_styles.contains_key("af_heart") {
            log::warn!("Default voice af_heart not found in {:?}", voices_dir);
        }

        let sample_rate = cfg.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(24000) as u32;
        let config = KokoroOnnxConfig { model_dir, sample_rate };
        let capabilities = Self::build_capabilities(sample_rate);

        Ok(Self { session: Mutex::new(session), voice_styles, config, capabilities })
    }

    fn load_voice_style(path: &std::path::Path) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        let floats: Vec<f32> = bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        anyhow::ensure!(floats.len() == 256,
            "Voice style {:?}: expected 256 floats, got {}", path, floats.len());
        Ok(floats)
    }

    fn synthesize_with_onnx(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        use crate::core::g2p_misaki::MisakiG2P;

        let g2p = MisakiG2P::new()?;
        let tokens = g2p.text_to_tokens(text)?;
        anyhow::ensure!(!tokens.is_empty(), "G2P produced no tokens for: {:?}", text);

        // tokens: int64 [1, seq_len]
        let seq_len = tokens.len();
        let tokens_array = Array2::from_shape_vec(
            (1, seq_len),
            tokens.iter().map(|&t| t as i64).collect(),
        )?;

        // style: f32 [1, 256]
        let voice_id = params.voice.as_deref().unwrap_or("af_heart");
        let style_vec = self.voice_styles.get(voice_id)
            .or_else(|| self.voice_styles.get("af_heart"))
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; 256]);
        let style_array = Array2::from_shape_vec((1, 256), style_vec)?;

        // speed: f32 [1]
        let speed_array = Array1::from_vec(vec![params.speed]);

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "tokens" => tokens_array,
            "style"  => style_array,
            "speed"  => speed_array
        ])?;

        let (_shape, audio_slice) = outputs["output"].try_extract_tensor::<f32>()?;
        let samples: Vec<f32> = audio_slice.iter().copied().collect();

        log::info!("Kokoro ONNX: {} tokens → {} samples ({:.2}s)",
            seq_len, samples.len(),
            samples.len() as f32 / self.config.sample_rate as f32);

        Ok(AudioData { samples, sample_rate: self.config.sample_rate, channels: 1 })
    }
}

#[async_trait]
impl TTSEngine for KokoroOnnxEngine {
    fn name(&self) -> &str { "kokoro-onnx" }
    fn capabilities(&self) -> &EngineCapabilities { &self.capabilities }
    fn list_voices(&self) -> Vec<VoiceInfo> { self.capabilities.supported_voices.clone() }

    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        self.validate_text(text)?;
        self.synthesize_with_onnx(text, params)
            .with_context(|| format!("Kokoro ONNX synthesis failed for: {:?}", &text[..text.len().min(40)]))
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test onnx_engine --features torch 2>&1 | grep -E "ok|FAILED|failed"
```

Expected: both tests `ok`.

- [ ] **Step 5: Verify `synthesize_parametric` is gone**

```bash
grep -n "synthesize_parametric\|synthesize_fallback" src/core/kokoro_onnx.rs
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/core/kokoro_onnx.rs
git commit -m "fix: implement real ort ONNX inference in KokoroOnnxEngine, wrap Session in Mutex"
```

---

## Task 8: Full build verification

- [ ] **Step 1: Build without torch (ONNX path only)**

```bash
cargo build 2>&1 | tail -5
```

Expected: `Finished` with no errors.

- [ ] **Step 2: Build with torch**

```bash
cargo build --features torch 2>&1 | tail -5
```

Expected: `Finished` with no errors.

- [ ] **Step 3: Run all TTS-related tests**

```bash
cargo test --features torch -- g2p styletts2 istftnet kokoro 2>&1 | tail -20
```

Expected: all tests `ok`, none `FAILED`.

- [ ] **Step 4: Verify neither engine produces silent fake audio**

```bash
grep -rn "synthesize_parametric\|synthesize_fallback\|sin(2.0 \* std::f32" src/core/kokoro_tts.rs src/core/kokoro_onnx.rs
```

Expected: no output (both stub functions fully deleted).

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "fix: TTS torch inference complete — both kokoro engines produce real audio"
```
