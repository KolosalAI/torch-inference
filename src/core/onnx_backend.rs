/// Shared Kokoro ONNX backend — loaded once, reused by all engines that lack
/// their own model weights (vits, bark, styletts2, xtts, kokoro-pth).
///
/// The singleton is initialised on first call and kept alive for the
/// lifetime of the process.  If the model files are missing the backend
/// resolves to `None` and the caller must handle the absence gracefully.
use std::sync::{Arc, OnceLock};

use crate::core::kokoro_onnx::KokoroOnnxEngine;

static SHARED_BACKEND: OnceLock<Option<Arc<KokoroOnnxEngine>>> = OnceLock::new();

/// Return the shared Kokoro ONNX engine, initialising it on first call.
///
/// Returns `None` if the model files are not present or could not be loaded.
pub fn get_kokoro_onnx_backend() -> Option<Arc<KokoroOnnxEngine>> {
    SHARED_BACKEND
        .get_or_init(|| {
            let cfg = serde_json::json!({
                "model_dir": "models/kokoro-82m",
                "sample_rate": 24000,
                // Single session for the shared backend; dedicated engines keep their own pool.
                "pool_size": 1
            });
            match KokoroOnnxEngine::new(&cfg) {
                Ok(engine) => {
                    log::info!("Shared Kokoro ONNX backend ready (models/kokoro-82m)");
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    log::warn!("Shared Kokoro ONNX backend unavailable: {}", e);
                    None
                }
            }
        })
        .clone()
}

/// Map an arbitrary engine-specific voice name to a Kokoro voice id.
///
/// Falls back to `"af_heart"` for anything unrecognised.
pub fn map_voice(engine_voice: Option<&str>) -> &'static str {
    match engine_voice {
        // Kokoro native ids pass straight through
        Some("af_heart") => "af_heart",
        Some("af_bella") => "af_bella",
        Some("af_sarah") => "af_sarah",
        Some("af_nicole") => "af_nicole",
        Some("am_adam") => "am_adam",
        Some("am_michael") => "am_michael",
        Some("bf_emma") => "bf_emma",
        Some("bf_isabella") => "bf_isabella",
        Some("bm_george") => "bm_george",
        Some("bm_lewis") => "bm_lewis",

        // Kokoro-pth short ids (af / am / bf)
        Some("af") => "af_heart",
        Some("am") => "am_adam",
        Some("bf") => "bf_emma",

        // VITS voices
        Some("vits_en_female") => "af_bella",
        Some("vits_en_male") => "am_adam",

        // StyleTTS2 voices
        Some("styletts2_expressive") => "af_heart",
        Some("styletts2_natural") => "bf_emma",

        // Bark voices
        Some("bark_v2_en_speaker_0") => "bm_george",
        Some("bark_v2_en_speaker_1") => "af_sarah",
        Some("bark_v2_en_speaker_6") => "am_michael",

        // XTTS voices
        Some("xtts_v2_en_female") => "af_nicole",
        Some("xtts_v2_en_male") => "bm_lewis",
        Some("xtts_v2_multilingual") => "bf_isabella",

        _ => "af_heart",
    }
}
