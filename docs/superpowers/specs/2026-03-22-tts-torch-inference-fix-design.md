# TTS Torch Inference Fix — Design Spec
Date: 2026-03-22

## Problem

The TTS system produces random sounds or sine-wave tones instead of real speech due to three compounding bugs:

1. `KokoroOnnxEngine::synthesize()` is hardcoded to `synthesize_parametric()`, which generates harmonic sine waves — never real ONNX inference.
2. `ISTFTNetVocoder::mel_to_linear()` skips both the `exp()` de-log step and proper mel filterbank inversion, feeding garbage magnitudes to Griffin-Lim (which also starts with random phase).
3. `MisakiG2P::PHONEME_VOCAB` defines only ~35 of the 178 Kokoro phoneme tokens, silently dropping most phonemes or substituting raw ASCII codes as token IDs.

Additionally, the entire `src/` directory was deleted by commit `3e1d8c1` ("migrate from Python to Rust"), so restoration is required before any fixes can be applied.

## Goals

- Torch native inference (`tch::nn`) is the primary TTS path using `kokoro-v1_0.pth`
- ONNX inference (`ort`) is the secondary path using `kokoro-v1_0.onnx`
- No sine-wave or parametric fallback in production paths — errors propagate clearly
- The `src/` directory is restored from git history

## Non-Goals

- Python bridge (`KokoroPythonBridge`) is kept as last-resort fallback but not improved
- Other TTS engines (Piper, Bark, VITS, etc.) are not touched
- STT, image, or other inference paths are not touched

---

## Architecture

```
Request → TTSManager
            |
            +-- "kokoro" engine (PRIMARY, requires --features torch)
            |     KokoroEngine::synthesize()
            |       1. MisakiG2P::text_to_tokens()        <- fixed 178-token vocab
            |       2. KokoroNativeModel::forward()        <- tch::nn StyleTTS2+ISTFTNet
            |            TextEncoder → DurPredictor → MelDecoder → ISTFTNetVocoder
            |       3. AudioData { samples, sample_rate: 24000, channels: 1 }
            |       Fallback: KokoroPythonBridge (if torch model absent)
            |
            +-- "kokoro-onnx" engine (SECONDARY, always compiled via ort dep)
                  KokoroOnnxEngine::synthesize()
                    1. MisakiG2P::text_to_tokens()        <- same fixed vocab
                    2. ort::Session forward pass           <- real ONNX inference
                         inputs: tokens [int64,1,seq], style [f32,1,256], speed [f32,1]
                         output: raw audio f32[1, n_samples]
                    Fallback: error 503 (no sine wave)
```

---

## Components

### 0. Restore `src/`

```
git checkout 4331e91 -- src/
```

Restores all Rust source files deleted in the migration commit.

### 1. Fix `g2p_misaki.rs` — Complete phoneme vocabulary

Replace the partial `PHONEME_VOCAB` lazy_static with a complete const array covering all 178 Kokoro token IDs. The full set includes:

- Punctuation (indices 1–16): `;`, `:`, `,`, `.`, `!`, `?`, `—`, `…`, space
- ASCII letters (indices 21–46): a–z mapped to their Kokoro positions
- IPA vowels (indices 47–90): `ɑ`, `æ`, `ɐ`, `ɒ`, `ə`, `ɚ`, `ɛ`, `ɜ`, `ɪ`, `ɔ`, `ʊ`, `ʌ`, `ɵ`, `ɘ`, `e`, `i`, `o`, `u`, `a`
- IPA consonants (indices 91–145): `b`, `d`, `f`, `ɡ`, `h`, `j`, `k`, `l`, `m`, `n`, `p`, `r`, `s`, `t`, `v`, `w`, `z`, `ð`, `θ`, `ŋ`, `ʃ`, `ʒ`, `ʧ`, `ʤ`
- Diphthongs represented as digraphs: `aɪ`, `eɪ`, `oʊ`, `aʊ`, `ɔɪ`
- Stress/prosody markers (indices 156–158): `ˈ`, `ˌ`, `ː`

The `letter_to_phoneme` fallback is updated to map to correct IPA symbols that exist in the complete vocab, not raw char codes.

### 2. Fix `styletts2_model.rs` — Replace `CModule` with `tch::nn`

**Weight loading strategy:**

`tch::VarStore::load()` cannot read raw PyTorch state dict `.pth` files (only files saved with `VarStore::save()`). Use `VarStore::load_partial()` which returns a list of unmatched keys rather than erroring on the first mismatch — this lets us log all missing keys before failing.

The `.pth` is first converted to `.safetensors` (which `tch 0.16` supports natively via `VarStore::load()` on `.safetensors` extension):

```python
# convert_kokoro.py (run once, see Conversion Script section)
```

In Rust, load via:
```rust
// load_partial returns Vec<String> — names of VarStore vars NOT found in the file
let unmatched_local = vs.load_partial(&safetensors_path)?;
if !unmatched_local.is_empty() {
    log::error!("Registered vars not found in model: {:?}", unmatched_local);
    anyhow::bail!("Model weight mismatch — check convert_kokoro.py output");
}
// Extra keys in the file that are not registered in the VarStore are silently ignored by tch
```

**Architecture (all gated behind `#[cfg(feature = "torch")]`):**

```
TextEncoder:
  Embedding(n_token=178, dim_in=64)     <- initial token embedding (dim_in from config)
  Linear(dim_in=64, hidden_dim=512)     <- up-projection to working dimension
  3x [Conv1d(512,512,k=5,pad=2) + LayerNorm(512) + ReLU + Dropout(0.2)]

DurationPredictor:
  Conv1d(512,256,k=3,pad=1) + ReLU
  Conv1d(256,256,k=3,pad=1) + ReLU
  Linear(256,1) + Softplus → per-phoneme duration (frames)

SpeakerEmbedding:
  nn::Embedding(n_speakers=54, style_dim=128)

MelDecoder (AdaIN-conditioned):
  Duration-expand: repeat each hidden state by its predicted duration
  Conv blocks with AdaIN(style_dim=128) conditioning
  Output Conv1d → n_mels=80
```

`dim_in=64` is the initial token embedding dimension before up-projection to `hidden_dim=512`.

State dict key prefixes mapped to sub-modules:
- `text_encoder.*` → TextEncoder
- `duration_predictor.*` → DurationPredictor
- `speaker_embedding.*` → SpeakerEmbedding
- `decoder.*` → MelDecoder

### 3. Fix `istftnet_vocoder.rs` — Replace Griffin-Lim with real ISTFTNet

All code gated behind `#[cfg(feature = "torch")]`.

**ISTFTNet architecture:**

```
Input: mel [batch, n_mels=80, mel_frames]

Input projection (REQUIRED — bridges n_mels=80 to upsample_initial_channel=512):
  Conv1d(in=80, out=512, kernel=7, padding=3)
  LeakyReLU(negative_slope=0.1)

ConvTranspose1d upsamplers (rates [10, 6]):
  ConvTranspose1d(512, 256, kernel=20, stride=10, padding=5)
  LeakyReLU(negative_slope=0.1)
  ConvTranspose1d(256, 128, kernel=12, stride=6,  padding=3)
  LeakyReLU(negative_slope=0.1)

Multi-Receptive Field Fusion (kernel sizes [3, 7, 11], dilations [1, 3, 5]):
  For each kernel size k in [3, 7, 11]:
    3x [LeakyReLU + dilated Conv1d(128, 128, k, dilation=d) + residual]
  Output: mean of all three stacks / 3

Output projection:
  LeakyReLU(negative_slope=0.1)
  Conv1d(128, (gen_istft_n_fft/2 + 1) * 2, kernel=1)
  -> shape [batch, (n_fft/2+1)*2, upsampled_frames]
  Reshape to [batch, n_fft/2+1, upsampled_frames, 2] (real + imaginary components)
  Tanh activation

ISTFT (via tch method form — NOT a free function):
  // hann_window signature: (window_length: i64, options: (Kind, Device)) -> Tensor
  let window = Tensor::hann_window(gen_istft_n_fft, (kind, device));
  let audio = stft_complex.istft(
      gen_istft_n_fft,      // n_fft = 20
      gen_istft_hop_size,   // hop_length = 5
      gen_istft_n_fft,      // win_length = 20
      &window,              // window tensor (REQUIRED parameter)
      /*center=*/ false,    // center (REQUIRED parameter)
      /*normalized=*/ false,
      /*onesided=*/ true,
      /*length=*/ None,
      /*return_complex=*/ false,
  );
  -> waveform [batch, n_samples]
```

Weights loaded from the same `.safetensors` file under `vocoder.*` key namespace via `load_partial()`. If weights aren't loaded, the vocoder returns `bail!("ISTFTNet requires loaded weights")` — no Griffin-Lim fallback.

### 4. Fix `kokoro_tts.rs` — Wire up native inference

`synthesize_fallback()` body is **deleted entirely** (not just `#[deprecated]`-annotated — the function is removed). The `synthesize()` method priority chain:

```
1. native_inference (tch::nn) if Some(_)   <- requires --features torch + .safetensors
2. python_bridge if Some(_)                 <- requires pip install kokoro
3. bail!("Kokoro TTS unavailable. Run convert_kokoro.py then rebuild with --features torch")
```

`init_native_inference()` updated to load from `.safetensors` (not `CModule::load_on_device`). If the `.safetensors` file is absent, returns `None` with a log warning pointing to `convert_kokoro.py`.

### 5. Fix `kokoro_onnx.rs` — Real ONNX inference via `ort` 2.0.0-rc.10

`ort` is an unconditional dependency (not feature-gated) so this path always compiles.

**Session mutability:** `Session::run()` requires `&mut self`. The `session` field must be wrapped in `Mutex<Session>`:

```rust
pub struct KokoroOnnxEngine {
    session: Mutex<ort::Session>,   // run() requires &mut self
    voice_styles: HashMap<String, Vec<f32>>,  // preloaded, [256] per voice
    config: KokoroOnnxConfig,
    capabilities: EngineCapabilities,
}
```

**Correct `ort` 2.0.0-rc.10 API:**

```rust
// Construction — commit_from_file takes &mut self so must bind builder first
let mut builder = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?;
let session = builder.commit_from_file(&model_path)?;

// Log input/output shapes at startup (fields, not methods)
for input in &session.inputs {
    log::info!("ONNX input: {:?}", input);
}
for output in &session.outputs {
    log::info!("ONNX output: {:?}", output);
}

// Voice style loading: flat f32[256] little-endian binary
fn load_voice_style(voices_dir: &Path, voice_id: &str) -> Result<Vec<f32>> {
    let bytes = std::fs::read(voices_dir.join(format!("{}.bin", voice_id)))?;
    let floats: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    anyhow::ensure!(floats.len() == 256, "Voice style must be 256 floats");
    Ok(floats)
}

// Inference (lock session, shape style as [1, 256])
let mut session = self.session.lock().unwrap();
let style = style_floats; // Vec<f32> len=256, reshape to [1,256] via ndarray
let outputs = session.run(ort::inputs![
    "tokens" => tokens_array,    // int64 ndarray Array2<i64> shape [1, seq_len]
    "style"  => style_array,     // f32 ndarray Array2<f32>  shape [1, 256]
    "speed"  => speed_array,     // f32 ndarray Array1<f32>  shape [1]
])?;                             // NOTE: no ? after ] — inputs![] returns Vec not Result

// Output extraction — try_extract_tensor returns Result<(&Shape, &[f32])>
let (_shape, audio_slice) = outputs["output"].try_extract_tensor::<f32>()?;
let audio: Vec<f32> = audio_slice.iter().copied().collect();
```

`synthesize_parametric()` body is **deleted entirely**. `synthesize_with_onnx()` is the sole synthesis path. If the ONNX model file is absent, `KokoroOnnxEngine::new()` returns `Err` immediately with a clear message.

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| `.safetensors` missing | `init_native_inference()` returns `None`, logs path + instruction to run `convert_kokoro.py` |
| `.safetensors` key mismatch | `load_partial()` collects all mismatches, logs them, returns `Err` |
| `.onnx` model missing | `KokoroOnnxEngine::new()` returns `Err` immediately |
| ONNX input shape mismatch | `session.run()` returns `Err`, propagated to API as 503 |
| Voice style `.bin` missing | Falls back to `af_heart` (speaker 0) with `warn!()` |
| Token not in phoneme vocab | `warn!()` logged per missing phoneme, token silently skipped |
| tch CUDA OOM | `Err` propagated, no retry |

No path produces fake audio silently. Every failure surfaces to the API as `503 Service Unavailable` with the reason string.

---

## File Change Summary

| File | Change |
|------|--------|
| `src/` (all) | Restored from git history (commit `4331e91`) |
| `src/core/g2p_misaki.rs` | Complete 178-token `PHONEME_VOCAB`, fix letter fallback |
| `src/core/styletts2_model.rs` | Replace `CModule` with `tch::nn` TextEncoder+DurPredictor+Decoder; load via `.safetensors` + `load_partial()` |
| `src/core/istftnet_vocoder.rs` | Replace Griffin-Lim with `tch::nn` ISTFTNet (add missing input projection Conv1d); use `Tensor::istft()` method form with `window` and `center` args |
| `src/core/kokoro_tts.rs` | Delete `synthesize_fallback()` body; wire `NativeInference` to new models |
| `src/core/kokoro_onnx.rs` | Wrap session in `Mutex<Session>`; implement `ort` 2.0.0-rc.10 inference; delete `synthesize_parametric()` body |
| `convert_kokoro.py` | New: one-time `.pth` → `.safetensors` conversion script |

---

## Conversion Script

`convert_kokoro.py` (committed to repo root):

```python
"""
One-time conversion: kokoro-v1_0.pth -> kokoro-v1_0.safetensors
Run before first use of the torch TTS engine.
Requires: pip install torch safetensors
"""
import torch
from safetensors.torch import save_file

src = "models/kokoro-82m/kokoro-v1_0.pth"
dst = "models/kokoro-82m/kokoro-v1_0.safetensors"

state = torch.load(src, map_location="cpu")
if not isinstance(state, dict):
    state = state.state_dict()

state = {k: v.contiguous() for k, v in state.items()}
save_file(state, dst)
print(f"Saved {dst} ({len(state)} tensors)")
print("Keys:", list(state.keys())[:10], "...")
```

---

## Success Criteria

- `POST /tts/synthesize` with `engine: "kokoro"` returns intelligible speech audio (after running `convert_kokoro.py`)
- `POST /tts/synthesize` with `engine: "kokoro-onnx"` returns intelligible speech audio (after placing `kokoro-v1_0.onnx` and voice `.bin` files)
- Both engines return a clear error (not sine waves) when model files are absent
- `cargo build --features torch` compiles without errors
- `cargo build` (without torch, ONNX path only) compiles without errors
