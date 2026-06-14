#![allow(dead_code)]
/// Python TTS Bridge - FFI bridge to Python TTS libraries
/// Enables calling Python TTS packages (kokoro, piper, etc.) from Rust
/// Requires: --features python (which links against libpython)

#[cfg(feature = "python")]
mod inner {
    use anyhow::{Context, Result};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};
    use std::ffi::CString;
    use std::sync::Mutex;

    use super::super::audio::AudioData;

    /// Convert a Rust `&str` to an owned `CString` for pyo3 APIs that
    /// took `&str` in 0.20 but now require `&CStr` in 0.24+. Panics on
    /// embedded NULs — none of our inline Python snippets have any.
    #[inline]
    fn cstr(s: &str) -> CString {
        CString::new(s).expect("inline Python source must be NUL-free")
    }

    /// Global Python interpreter guard
    static PYTHON_INIT: Mutex<bool> = Mutex::new(false);

    /// Initialize Python interpreter (called once)
    pub fn ensure_python_initialized() -> Result<()> {
        let mut initialized = PYTHON_INIT.lock().unwrap();
        if !*initialized {
            pyo3::prepare_freethreaded_python();
            *initialized = true;
            log::info!("Python interpreter initialized for TTS bridge");
        }
        Ok(())
    }

    /// Kokoro TTS Python Bridge
    pub struct KokoroPythonBridge {
        pub initialized: bool,
    }

    impl KokoroPythonBridge {
        pub fn new() -> Result<Self> {
            ensure_python_initialized()?;

            Python::with_gil(|py| {
                py.run(
                    &cstr(
                        r#"
import sys
try:
    import kokoro
    print(f"Kokoro version: {kokoro.__version__ if hasattr(kokoro, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"Error: kokoro not installed. Install with: pip install kokoro")
    raise
"#,
                    ),
                    None,
                    None,
                )
            })
            .context("Failed to import kokoro. Run: pip install kokoro")?;

            log::info!("Kokoro Python bridge initialized successfully");
            Ok(Self { initialized: true })
        }

        /// Synthesize speech using Kokoro Python library
        pub fn synthesize(&self, text: &str, voice: Option<&str>, speed: f32) -> Result<AudioData> {
            if !self.initialized {
                anyhow::bail!("Kokoro bridge not initialized");
            }

            let voice = voice.unwrap_or("af_heart");

            log::info!(
                "Synthesizing with Kokoro: '{}' (voice: {}, speed: {})",
                &text[..text.len().min(50)],
                voice,
                speed
            );

            Python::with_gil(|py| -> Result<AudioData> {
                let kokoro = py
                    .import("kokoro")
                    .map_err(|e| anyhow::anyhow!("Failed to import kokoro: {}", e))?;

                let pipeline_class = kokoro
                    .getattr("KPipeline")
                    .map_err(|e| anyhow::anyhow!("Failed to get KPipeline class: {}", e))?;
                let kwargs = PyDict::new(py);
                kwargs
                    .set_item("lang_code", "a")
                    .map_err(|e| anyhow::anyhow!("Failed to set lang_code: {}", e))?;
                let pipeline = pipeline_class
                    .call((), Some(&kwargs))
                    .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {}", e))?;

                let gen_kwargs = PyDict::new(py);
                gen_kwargs
                    .set_item("voice", voice)
                    .map_err(|e| anyhow::anyhow!("Failed to set voice: {}", e))?;
                let generator = pipeline
                    .call((text,), Some(&gen_kwargs))
                    .map_err(|e| anyhow::anyhow!("Failed to call pipeline: {}", e))?;

                let iterator = generator
                    .call_method0("__iter__")
                    .map_err(|e| anyhow::anyhow!("Failed to get iterator: {}", e))?;
                let result = iterator
                    .call_method0("__next__")
                    .map_err(|e| anyhow::anyhow!("Failed to get next result: {}", e))?;

                // result is tuple: (graphemes, phonemes, audio).
                // pyo3 0.24 returns `Bound<'_, PyAny>` from get_item;
                // dropping the explicit `&PyAny` annotation lets the
                // compiler infer the new type.
                let audio_array = result
                    .get_item(2)
                    .map_err(|e| anyhow::anyhow!("Failed to get audio data at index 2: {}", e))?;

                let audio_flat = audio_array
                    .call_method0("flatten")
                    .map_err(|e| anyhow::anyhow!("Failed to flatten audio: {}", e))?;
                let audio_list = audio_flat
                    .call_method0("tolist")
                    .map_err(|e| anyhow::anyhow!("Failed to convert to list: {}", e))?;

                let samples: Vec<f32> = audio_list
                    .extract()
                    .map_err(|e| anyhow::anyhow!("Failed to extract samples: {}", e))?;

                log::info!(
                    "Kokoro generated {} samples ({:.2}s)",
                    samples.len(),
                    samples.len() as f32 / 24000.0
                );

                Ok(AudioData {
                    samples,
                    sample_rate: 24000,
                    channels: 1,
                })
            })
            .context("Failed to synthesize with Kokoro Python library")
        }

        pub fn list_voices(&self) -> Result<Vec<String>> {
            Python::with_gil(|py| {
                let code = r#"
voices = [
    'af_heart', 'af_bella', 'af_sarah', 'af_nicole',
    'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella',
    'bm_george', 'bm_lewis'
]
voices
"#;
                let locals = PyDict::new(py);
                py.run(&cstr(code), None, Some(&locals))?;
                let voices: Vec<String> = locals
                    .get_item("voices")
                    .context("Failed to get voices")?
                    .ok_or_else(|| anyhow::anyhow!("voices not found in locals"))?
                    .extract()?;
                Ok(voices)
            })
        }
    }

    pub fn check_python_tts_dependencies() -> Result<Vec<String>> {
        ensure_python_initialized()?;

        Python::with_gil(|py| {
            let code = r#"
available = []
missing = []
for pkg in ['kokoro', 'soundfile', 'numpy']:
    try:
        __import__(pkg)
        available.append(pkg)
    except ImportError:
        missing.append(pkg)
result = (available, missing)
"#;
            let locals = PyDict::new(py);
            py.run(&cstr(code), None, Some(&locals))?;
            let (available, missing): (Vec<String>, Vec<String>) = locals
                .get_item("result")
                .context("no result")?
                .ok_or_else(|| anyhow::anyhow!("result not in locals"))?
                .extract()?;
            if !missing.is_empty() {
                log::warn!("Missing Python TTS dependencies: {:?}", missing);
            }
            Ok(available)
        })
    }
}

// Public re-exports — only available with the "python" feature
#[cfg(feature = "python")]
pub use inner::{check_python_tts_dependencies, KokoroPythonBridge};

// Stub when python feature is disabled
#[cfg(not(feature = "python"))]
pub struct KokoroPythonBridge;

#[cfg(not(feature = "python"))]
impl KokoroPythonBridge {
    pub fn new() -> anyhow::Result<Self> {
        anyhow::bail!("Python bridge unavailable (build with --features python)")
    }

    pub fn synthesize(
        &self,
        _text: &str,
        _voice: Option<&str>,
        _speed: f32,
    ) -> anyhow::Result<super::audio::AudioData> {
        anyhow::bail!("Python bridge unavailable (build with --features python)")
    }

    pub fn list_voices(&self) -> anyhow::Result<Vec<String>> {
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Line 183-185: new() bails without the python feature.
    #[test]
    fn test_kokoro_bridge_new_fails_without_python_feature() {
        let result = KokoroPythonBridge::new();
        assert!(result.is_err());
        // Use .err().unwrap() instead of .unwrap_err() because KokoroPythonBridge
        // doesn't implement Debug (it's a unit struct stub).
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("Python bridge unavailable") || msg.contains("python"),
            "unexpected error: {}",
            msg
        );
    }

    /// Lines 187 / 193: synthesize() bails without the python feature.
    #[test]
    #[cfg(not(feature = "python"))]
    fn test_kokoro_bridge_synthesize_bails_without_python_feature() {
        // Construct the unit struct directly (bypassing new()).
        let bridge = KokoroPythonBridge;
        let result = bridge.synthesize("hello", Some("af_heart"), 1.0);
        assert!(result.is_err());
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("Python bridge unavailable") || msg.contains("python"),
            "unexpected error: {}",
            msg
        );
    }

    /// Lines 196-197: list_voices() returns an empty vec without the python feature.
    #[test]
    #[cfg(not(feature = "python"))]
    fn test_kokoro_bridge_list_voices_empty_without_python_feature() {
        let bridge = KokoroPythonBridge;
        let result = bridge.list_voices();
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
