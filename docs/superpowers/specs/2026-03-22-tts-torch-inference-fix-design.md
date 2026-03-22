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
            +-- "kokoro" engine (PRIMARY)
            |     KokoroEngine::synthesize()
            |       1. MisakiG2P::text_to_tokens()        <- fixed 178-token vocab
            |       2. KokoroNativeModel::forward()        <- tch::nn StyleTTS2+ISTFTNet
            |            TextEncoder → DurPredictor → MelDecoder → ISTFTNetVocoder
            |       3. AudioData { samples, sample_rate: 24000, channels: 1 }
            |       Fallback: KokoroPythonBridge (if torch model absent)
            |
            +-- "kokoro-onnx" engine (SECONDARY)
                  KokoroOnnxEngine::synthesize()
                    1. MisakiG2P::text_to_tokens()        <- same fixed vocab
                    2. ort::Session forward pass           <- real ONNX inference
                         inputs: tokens, style, speed
                         output: raw audio f32[1, n_samples]
                    Fallback: error 503 (no sine wave)
```

---

## Components

### 1. Restore `src/`

```
git checkout 4331e91 -- src/
```

Restores all Rust source files deleted in the migration commit. No other changes needed for restoration.

### 2. Fix `g2p_misaki.rs` — Complete phoneme vocabulary

Replace the partial `PHONEME_VOCAB` lazy_static with a complete const array covering all 178 Kokoro token IDs. The full set includes:

- Punctuation (indices 1–16): `;`, `:`, `,`, `.`, `!`, `?`, `—`, `…`, space
- ASCII letters (indices 21–46): a–z mapped to their Kokoro positions
- IPA vowels (indices 47–90): `ɑ`, `æ`, `ɐ`, `ɒ`, `ə`, `ɚ`, `ɛ`, `ɜ`, `ɪ`, `ɔ`, `ʊ`, `ʌ`, `ɵ`, `ɘ`, `e`, `i`, `o`, `u`, `a`
- IPA consonants (indices 91–145): `b`, `d`, `f`, `ɡ`, `h`, `j`, `k`, `l`, `m`, `n`, `p`, `r`, `s`, `t`, `v`, `w`, `z`, `ð`, `θ`, `ŋ`, `ʃ`, `ʒ`, `ʧ`, `ʤ`
- Diphthongs represented as digraphs: `aɪ`, `eɪ`, `oʊ`, `aʊ`, `ɔɪ`
- Stress/prosody markers (indices 156–158): `ˈ`, `ˌ`, `ː`

The `letter_to_phoneme` fallback is updated to map to correct IPA symbols that exist in the complete vocab, not raw char codes.

### 3. Fix `styletts2_model.rs` — Replace CModule with `tch::nn`

Replace `CModule::load_on_device()` with a proper `nn::Sequential`-style graph:

**TextEncoder**
```
Embedding(n_token=178, d_model=512)
+ positional encoding
+ 3x [Conv1d(512,512,k=5,pad=2) + LayerNorm + ReLU + Dropout(0.2)]
```

**DurationPredictor**
```
Conv1d(512,256,k=3,pad=1) + ReLU
Conv1d(256,256,k=3,pad=1) + ReLU
Linear(256,1) + Softplus → per-phoneme duration (frames)
```

**SpeakerEmbedding**
```
Embedding(n_speakers=54, style_dim=128)
```

**MelDecoder** (AdaIN-conditioned)
```
Duration-expand: repeat each hidden state by its predicted duration
Conv blocks with AdaIN(style_dim=128) conditioning
Output Conv1d(→ n_mels=80)
```

Weights loaded from `kokoro-v1_0.pth` via `VarStore::load()`. Key names in the state dict follow the pattern `text_encoder.*`, `duration_predictor.*`, `decoder.*`, `speaker_embedding.*` — verified against the model file at load time with clear error messages for missing keys.

### 4. Fix `istftnet_vocoder.rs` — Replace Griffin-Lim with real ISTFTNet

Replace the `SimplifiedVocoder` Griffin-Lim path with an actual `tch::nn` upsampling network:

**ISTFTNet architecture**
```
Input: mel [batch, 80, mel_frames]

ConvTranspose1d upsamplers (rates [10, 6]):
  ConvTranspose1d(512, 256, k=20, stride=10)
  ConvTranspose1d(256, 128, k=12, stride=6)

Multi-Receptive Field Fusion (3 residual stacks):
  Each stack: 3x [dilated Conv1d(k=3, d=1/3/5) + residual]

Output projection:
  Conv1d(128, gen_istft_n_fft+2, k=1)   <- complex STFT coefficients
  Tanh activation

torch::istft(n_fft=20, hop_size=5) → waveform [batch, n_samples]
```

Weights loaded from same `kokoro-v1_0.pth` under `decoder.*` or `vocoder.*` key namespace.

If torch weights aren't loaded, the vocoder returns an error rather than running Griffin-Lim.

### 5. Fix `kokoro_tts.rs` — Wire up native inference

`NativeInference` struct updated to hold `KokoroNativeModel` (text encoder + dur predictor + decoder) and `ISTFTNetVocoder` as separate `tch::nn` structs. `synthesize_with_native()` calls them in order. The deprecated `synthesize_fallback()` is removed. The `synthesize()` method priority:

```
1. native_inference (tch::nn) if loaded
2. python_bridge if available
3. bail!("No working TTS method. Model: models/kokoro-82m/kokoro-v1_0.pth")
```

### 6. Fix `kokoro_onnx.rs` — Real ONNX inference

Replace the `synthesize_with_onnx()` stub and the `synthesize()` hardcoding:

```rust
// Load session once at construction time
let session = ort::Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_model_from_file(model_path)?;

// Load voice style vectors from voices/<voice_id>.bin (f32 arrays, shape [256])
let style = load_voice_style(voice_id)?;  // [1, 1, 256]

// Inputs
let tokens_tensor = ...;  // int64 [1, seq_len]
let style_tensor = ...;   // f32  [1, 1, 256]
let speed_tensor = ...;   // f32  [1]

// Run
let outputs = session.run(inputs![tokens_tensor, style_tensor, speed_tensor])?;
let audio: Vec<f32> = outputs[0].try_extract_tensor()?.view([-1]).to_vec();
```

Voice style files for the 10 default voices are bundled in `models/kokoro-onnx/voices/`. The ONNX model file path is `models/kokoro-onnx/kokoro-v1_0.onnx` — if absent, construction returns an informative error.

`synthesize_parametric()` is removed. `synthesize()` calls `synthesize_with_onnx()` only.

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| `.pth` model file missing | `KokoroEngine::new()` succeeds, `is_ready()` returns false, `synthesize()` returns `bail!` with model path |
| `.pth` weight key mismatch | Load logs which keys are missing, returns `Err` |
| `.onnx` model file missing | `KokoroOnnxEngine::new()` returns `Err` immediately |
| Voice style file missing | Falls back to `af_heart` (speaker 0) with a warning |
| Token not in phoneme vocab | Warning logged, token skipped (not silent) |
| tch CUDA OOM | Error propagated, no retry |

No path produces fake audio silently. Every failure returns a structured error that surfaces to the API as `503 Service Unavailable` with the reason.

---

## File Change Summary

| File | Change |
|------|--------|
| `src/` (all) | Restored from git history (commit `4331e91`) |
| `src/core/g2p_misaki.rs` | Complete 178-token PHONEME_VOCAB, fix letter fallback |
| `src/core/styletts2_model.rs` | Replace CModule with tch::nn TextEncoder+DurPredictor+Decoder |
| `src/core/istftnet_vocoder.rs` | Replace Griffin-Lim with tch::nn ISTFTNet upsampler |
| `src/core/kokoro_tts.rs` | Wire NativeInference to new models, remove parametric fallback |
| `src/core/kokoro_onnx.rs` | Implement real ort::Session inference, remove parametric fallback |

---

## Success Criteria

- `POST /tts/synthesize` with `engine: "kokoro"` returns intelligible speech audio
- `POST /tts/synthesize` with `engine: "kokoro-onnx"` returns intelligible speech audio
- Both engines return a clear error (not sine waves) when model files are absent
- `cargo build --features torch` compiles without errors
- `cargo build` (without torch) compiles without errors (ONNX path still works)
