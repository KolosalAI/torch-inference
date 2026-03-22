/// Python TTS Bridge - FFI bridge to Python TTS libraries
/// Enables calling Python TTS packages (kokoro, piper, etc.) from Rust
use anyhow::{Result, Context};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Mutex;

use super::audio::AudioData;

/// Global Python interpreter guard
static PYTHON_INIT: Mutex<bool> = Mutex::new(false);

/// Initialize Python interpreter (called once)
fn ensure_python_initialized() -> Result<()> {
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
        
        // Check if kokoro is available
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
    
    /// Synthesize speech using Kokoro Python library
    pub fn synthesize(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: f32,
    ) -> Result<AudioData> {
        if !self.initialized {
            anyhow::bail!("Kokoro bridge not initialized");
        }
        
        let voice = voice.unwrap_or("af_heart");
        
        log::info!("Synthesizing with Kokoro: '{}' (voice: {}, speed: {})", 
                   &text[..text.len().min(50)], voice, speed);
        
        Python::with_gil(|py| -> Result<AudioData> {
            // Import kokoro
            let kokoro = py.import("kokoro")
                .map_err(|e| anyhow::anyhow!("Failed to import kokoro: {}", e))?;
            
            // Create pipeline with American English
            let pipeline_class = kokoro.getattr("KPipeline")
                .map_err(|e| anyhow::anyhow!("Failed to get KPipeline class: {}", e))?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("lang_code", "a")
                .map_err(|e| anyhow::anyhow!("Failed to set lang_code: {}", e))?;
            let pipeline = pipeline_class.call((), Some(kwargs))
                .map_err(|e| anyhow::anyhow!("Failed to create pipeline: {}", e))?;
            
            // Generate audio - call pipeline with text and voice
            let gen_kwargs = PyDict::new(py);
            gen_kwargs.set_item("voice", voice)
                .map_err(|e| anyhow::anyhow!("Failed to set voice: {}", e))?;
            let generator = pipeline.call((text,), Some(gen_kwargs))
                .map_err(|e| anyhow::anyhow!("Failed to call pipeline: {}", e))?;
            
            // Get iterator and first result
            let iterator = generator.call_method0("__iter__")
                .map_err(|e| anyhow::anyhow!("Failed to get iterator: {}", e))?;
            let result = iterator.call_method0("__next__")
                .map_err(|e| anyhow::anyhow!("Failed to get next result: {}", e))?;
            
            // Extract audio data (result is tuple: (graphemes, phonemes, audio))
            let audio_array: &PyAny = result.get_item(2)
                .map_err(|e| anyhow::anyhow!("Failed to get audio data at index 2: {}", e))?;
            
            // Convert numpy array to Vec<f32>
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
    
    /// List available voices
    pub fn list_voices(&self) -> Result<Vec<String>> {
        Python::with_gil(|py| {
            let code = r#"
# Kokoro voices from VOICES.md
voices = [
    'af_heart', 'af_bella', 'af_sarah', 'af_nicole',
    'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella',
    'bm_george', 'bm_lewis'
]
voices
"#;
            let locals = PyDict::new(py);
            py.run(code, None, Some(locals))?;
            let voices: Vec<String> = locals.get_item("voices")
                .context("Failed to get voices")?
                .ok_or_else(|| anyhow::anyhow!("voices not found in locals"))?
                .extract()?;
            Ok(voices)
        })
    }
}

/// Check if Python TTS dependencies are installed
pub fn check_python_tts_dependencies() -> Result<Vec<String>> {
    ensure_python_initialized()?;
    
    Python::with_gil(|py| {
        let code = r#"
import sys
available = []
missing = []

# Check kokoro
try:
    import kokoro
    available.append('kokoro')
except ImportError:
    missing.append('kokoro')

# Check soundfile (required by kokoro)
try:
    import soundfile
    available.append('soundfile')
except ImportError:
    missing.append('soundfile')

# Check numpy
try:
    import numpy
    available.append('numpy')
except ImportError:
    missing.append('numpy')

(available, missing)
"#;
        let locals = PyDict::new(py);
        py.run(code, None, Some(locals))?;
        
        let result: (&PyAny, &PyAny) = locals.get_item("(available, missing)")
            .context("Failed to get dependency check results")?
            .ok_or_else(|| anyhow::anyhow!("dependency check results not found"))?
            .extract()?;
        
        let available: Vec<String> = result.0.extract()?;
        let missing: Vec<String> = result.1.extract()?;
        
        if !missing.is_empty() {
            log::warn!("Missing Python TTS dependencies: {:?}", missing);
            log::warn!("Install with: pip install {}", missing.join(" "));
        }
        
        if !available.is_empty() {
            log::info!("Available Python TTS dependencies: {:?}", available);
        }
        
        Ok(available)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_python_init() {
        let result = ensure_python_initialized();
        assert!(result.is_ok());
    }
    
    #[test]
    #[ignore] // Only run if kokoro is installed
    fn test_kokoro_bridge() {
        let bridge = KokoroPythonBridge::new();
        if let Ok(b) = bridge {
            let result = b.synthesize("Hello world", Some("af_heart"), 1.0);
            assert!(result.is_ok());
            
            let audio = result.unwrap();
            assert!(audio.samples.len() > 0);
            assert_eq!(audio.sample_rate, 24000);
        }
    }
}
