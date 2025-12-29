/// Python TTS Bridge - FFI bridge to Python TTS libraries
/// Enables calling Python TTS packages (kokoro, piper, etc.) from Rust
/// 
/// NOTE: This module requires the `python` feature to be enabled.
/// It can conflict with libtorch on some systems.

use anyhow::{Result, Context};

// Re-export AudioData for use by callers
use super::audio::AudioData;

// ============================================================================
// PYTHON FEATURE ENABLED
// ============================================================================

#[cfg(feature = "python")]
mod python_impl {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Global Python interpreter guard
    static PYTHON_INIT: Mutex<bool> = Mutex::new(false);
    /// Flag to indicate if Python initialization should be skipped
    static PYTHON_SKIP: AtomicBool = AtomicBool::new(false);

    /// Check if Python bridge is enabled via environment variable
    fn is_python_bridge_enabled() -> bool {
        if std::env::var("DISABLE_PYTHON_TTS").map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false) {
            return false;
        }
        if PYTHON_SKIP.load(Ordering::SeqCst) {
            return false;
        }
        true
    }

    /// Check if Python is actually available on the system
    fn check_python_available() -> bool {
        std::process::Command::new("python3")
            .arg("-c")
            .arg("import sys; print(sys.version_info.major)")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Initialize Python interpreter (called once)
    pub fn ensure_python_initialized() -> Result<()> {
        if !is_python_bridge_enabled() {
            anyhow::bail!("Python TTS bridge is disabled");
        }
        
        if !check_python_available() {
            PYTHON_SKIP.store(true, Ordering::SeqCst);
            anyhow::bail!("Python is not available on this system");
        }
        
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
        initialized: bool,
    }

    impl KokoroPythonBridge {
        pub fn new() -> Result<Self> {
            ensure_python_initialized()?;
            
            Python::with_gil(|py| {
                py.run(
                    r#"
import sys
try:
    import kokoro
    print(f"Kokoro version: {kokoro.__version__ if hasattr(kokoro, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"Error: kokoro not installed. Install with: pip install kokoro")
    raise
"#,
                    None,
                    None,
                )
            })
            .context("Failed to import kokoro. Run: pip install kokoro")?;
            
            log::info!("Kokoro Python bridge initialized successfully");
            Ok(Self { initialized: true })
        }
        
        pub fn synthesize(&self, text: &str, voice: Option<&str>, speed: f32) -> Result<AudioData> {
            if !self.initialized {
                anyhow::bail!("Kokoro bridge not initialized");
            }
            
            let voice = voice.unwrap_or("af_heart");
            
            log::info!("Synthesizing with Kokoro: '{}' (voice: {}, speed: {})", 
                       &text[..text.len().min(50)], voice, speed);
            
            Python::with_gil(|py| -> Result<AudioData> {
                let kokoro = py.import("kokoro")
                    .map_err(|e| anyhow::anyhow!("Failed to import kokoro: {}", e))?;
                
                let pipeline_class = kokoro.getattr("KPipeline")
                    .map_err(|e| anyhow::anyhow!("Failed to get KPipeline class: {}", e))?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("lang_code", "a")
                    .map_err(|e| anyhow::anyhow!("Failed to set lang_code: {}", e))?;
                let pipeline = pipeline_class.call((), Some(kwargs))
                    .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {}", e))?;
                
                let gen_kwargs = PyDict::new(py);
                gen_kwargs.set_item("voice", voice)
                    .map_err(|e| anyhow::anyhow!("Failed to set voice: {}", e))?;
                let generator = pipeline.call((text,), Some(gen_kwargs))
                    .map_err(|e| anyhow::anyhow!("Failed to call pipeline: {}", e))?;
                
                let iterator = generator.call_method0("__iter__")
                    .map_err(|e| anyhow::anyhow!("Failed to get iterator: {}", e))?;
                let result = iterator.call_method0("__next__")
                    .map_err(|e| anyhow::anyhow!("Failed to get next result: {}", e))?;
                
                let audio_array = result.get_item(2)
                    .map_err(|e| anyhow::anyhow!("Failed to get audio data: {}", e))?;
                
                let audio_flat = audio_array.call_method0("flatten")
                    .map_err(|e| anyhow::anyhow!("Failed to flatten audio: {}", e))?;
                let audio_list = audio_flat.call_method0("tolist")
                    .map_err(|e| anyhow::anyhow!("Failed to convert to list: {}", e))?;
                
                let samples: Vec<f32> = audio_list.extract()
                    .map_err(|e| anyhow::anyhow!("Failed to extract samples: {}", e))?;
                
                log::info!("Kokoro generated {} samples ({:.2}s)", 
                          samples.len(), samples.len() as f32 / 24000.0);
                
                Ok(AudioData {
                    samples,
                    sample_rate: 24000,
                    channels: 1,
                })
            })
            .context("Failed to synthesize with Kokoro Python library")
        }
        
        pub fn list_voices(&self) -> Result<Vec<String>> {
            Ok(vec![
                "af_heart".to_string(), "af_bella".to_string(), "af_sarah".to_string(),
                "am_adam".to_string(), "am_michael".to_string(),
                "bf_emma".to_string(), "bf_isabella".to_string(),
                "bm_george".to_string(), "bm_lewis".to_string(),
            ])
        }
    }
}

// ============================================================================
// PYTHON FEATURE DISABLED (STUB IMPLEMENTATION)
// ============================================================================

#[cfg(not(feature = "python"))]
mod python_impl {
    use super::*;

    /// Stub Kokoro TTS Python Bridge when python feature is disabled
    pub struct KokoroPythonBridge;

    impl KokoroPythonBridge {
        pub fn new() -> Result<Self> {
            anyhow::bail!("Python TTS bridge not available (compile with --features python)")
        }
        
        pub fn synthesize(&self, _text: &str, _voice: Option<&str>, _speed: f32) -> Result<AudioData> {
            anyhow::bail!("Python TTS bridge not available")
        }
        
        pub fn list_voices(&self) -> Result<Vec<String>> {
            anyhow::bail!("Python TTS bridge not available")
        }
    }
}

// Re-export the implementation
pub use python_impl::KokoroPythonBridge;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(feature = "python")]
    fn test_kokoro_bridge_creation() {
        // This test only runs if python feature is enabled
        let result = KokoroPythonBridge::new();
        // Just check it doesn't panic - may fail if kokoro not installed
        let _ = result;
    }
}
