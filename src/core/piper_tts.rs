/// Piper TTS Engine - Neural TTS using ONNX Runtime
use anyhow::{Result, Context, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use std::collections::HashMap;

use super::tts_engine::{TTSEngine, EngineCapabilities, VoiceInfo, VoiceGender, VoiceQuality, SynthesisParams};
use super::audio::AudioData;

/// Piper TTS Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiperConfig {
    pub model_path: PathBuf,
    pub config_path: PathBuf,
    pub sample_rate: u32,
    pub noise_scale: f32,
    pub length_scale: f32,
    pub noise_w: f32,
}

/// Piper TTS Engine - High-quality neural TTS
pub struct PiperTTSEngine {
    config: PiperConfig,
    capabilities: EngineCapabilities,
    phoneme_id_map: HashMap<String, Vec<i64>>,
    has_onnx: bool,
}

impl PiperTTSEngine {
    pub fn new(cfg: &serde_json::Value) -> Result<Self> {
        let model_path = cfg.get("model_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing model_path"))?;
        
        let config_path = cfg.get("config_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing config_path"))?;
        
        let model_path = PathBuf::from(model_path);
        let config_path = PathBuf::from(config_path);
        
        // Verify files exist
        if !model_path.exists() {
            bail!("Model file not found: {:?}", model_path);
        }
        
        // Load config if it exists
        let (sample_rate, noise_scale, length_scale, noise_w, phoneme_id_map) = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
            
            let sample_rate = config_json.get("audio")
                .and_then(|a| a.get("sample_rate"))
                .and_then(|s| s.as_u64())
                .unwrap_or(22050) as u32;
            
            let noise_scale = config_json.get("inference")
                .and_then(|i| i.get("noise_scale"))
                .and_then(|n| n.as_f64())
                .unwrap_or(0.667) as f32;
            
            let length_scale = config_json.get("inference")
                .and_then(|i| i.get("length_scale"))
                .and_then(|l| l.as_f64())
                .unwrap_or(1.0) as f32;
            
            let noise_w = config_json.get("inference")
                .and_then(|i| i.get("noise_w"))
                .and_then(|n| n.as_f64())
                .unwrap_or(0.8) as f32;
            
            // Load phoneme_id_map from config
            let mut phoneme_id_map = HashMap::new();
            if let Some(map) = config_json.get("phoneme_id_map").and_then(|m| m.as_object()) {
                for (key, value) in map {
                    if let Some(ids) = value.as_array() {
                        let id_vec: Vec<i64> = ids.iter()
                            .filter_map(|v| v.as_i64())
                            .collect();
                        phoneme_id_map.insert(key.clone(), id_vec);
                    }
                }
            }
            
            (sample_rate, noise_scale, length_scale, noise_w, phoneme_id_map)
        } else {
            (22050, 0.667, 1.0, 0.8, HashMap::new())
        };
        
        let config = PiperConfig {
            model_path: model_path.clone(),
            config_path,
            sample_rate,
            noise_scale,
            length_scale,
            noise_w,
        };
        
        let capabilities = EngineCapabilities {
            name: "Piper Neural TTS".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "en_US_lessac".to_string(),
                    name: "English US (Lessac)".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Female,
                    quality: VoiceQuality::High,
                },
            ],
            max_text_length: 10000,
            sample_rate,
            supports_ssml: false,
            supports_streaming: false,
        };
        
        let capabilities = EngineCapabilities {
            name: "Piper Neural TTS".to_string(),
            version: "1.0.0".to_string(),
            supported_languages: vec!["en".to_string()],
            supported_voices: vec![
                VoiceInfo {
                    id: "lessac".to_string(),
                    name: "Lessac (High Quality)".to_string(),
                    language: "en".to_string(),
                    gender: VoiceGender::Male,
                    quality: VoiceQuality::Neural,
                },
            ],
            max_text_length: 5000,
            sample_rate,
            supports_ssml: false,
            supports_streaming: false,
        };
        
        // Check if ONNX model exists
        let has_onnx = model_path.exists();
        if has_onnx {
            log::info!("✅ Piper ONNX model found at: {:?}", model_path);
            #[cfg(not(feature = "onnx"))]
            {
                bail!("ONNX feature not enabled. Compile with --features onnx to use Piper TTS");
            }
        } else {
            bail!("Piper model not found at {:?}", model_path);
        }
        
        Ok(Self {
            config,
            capabilities,
            phoneme_id_map,
            has_onnx,
        })
    }
    
    fn text_to_phoneme_ids(&self, text: &str) -> Vec<i64> {
        let mut phoneme_ids = vec![1]; // Start token
        
        let text_lower = text.to_lowercase();
        let mut i = 0;
        let chars: Vec<char> = text_lower.chars().collect();
        
        while i < chars.len() {
            let ch = chars[i];
            
            // Try to match multi-character phonemes first
            if i + 1 < chars.len() {
                let two_char = format!("{}{}", ch, chars[i + 1]);
                if let Some(ids) = self.phoneme_id_map.get(&two_char) {
                    phoneme_ids.extend(ids);
                    i += 2;
                    continue;
                }
            }
            
            // Try single character
            let single_char = ch.to_string();
            if let Some(ids) = self.phoneme_id_map.get(&single_char) {
                phoneme_ids.extend(ids);
            } else {
                // Default to space if unknown
                phoneme_ids.push(3);
            }
            
            i += 1;
        }
        
        phoneme_ids.push(2); // End token
        phoneme_ids
    }
    
    
    
    fn synthesize_with_phonemes(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        log::info!("Synthesizing with enhanced phoneme synthesis: '{}'", text);
        
        // Convert text to phoneme IDs using the Piper phoneme map
        let phoneme_ids = self.text_to_phoneme_ids(text);
        log::debug!("Generated {} phoneme IDs", phoneme_ids.len());
        
        // Generate audio based on phoneme sequence with better quality
        let duration_per_phoneme = 0.08 / params.speed;  // 80ms per phoneme
        let total_duration = phoneme_ids.len() as f32 * duration_per_phoneme;
        let sample_count = (self.config.sample_rate as f32 * total_duration) as usize;
        
        let mut samples = Vec::with_capacity(sample_count);
        let samples_per_phoneme = (self.config.sample_rate as f32 * duration_per_phoneme) as usize;
        
        for (idx, &phoneme_id) in phoneme_ids.iter().enumerate() {
            let phoneme_samples = self.generate_phoneme_audio(
                phoneme_id, 
                samples_per_phoneme,
                params.pitch
            );
            samples.extend(phoneme_samples);
        }
        
        // Apply smoothing to reduce artifacts
        self.apply_smoothing(&mut samples);
        
        log::info!("✅ Generated {} samples ({:.2}s) using enhanced phoneme synthesis", 
            samples.len(), 
            samples.len() as f32 / self.config.sample_rate as f32
        );
        
        Ok(AudioData {
            samples,
            sample_rate: self.config.sample_rate,
            channels: 1,
        })
    }
    
    fn generate_phoneme_audio(&self, phoneme_id: i64, sample_count: usize, pitch: f32) -> Vec<f32> {
        let mut samples = Vec::with_capacity(sample_count);
        let sample_rate = self.config.sample_rate as f32;
        
        // Map phoneme IDs to approximate frequency patterns
        let (base_freq, formant1, formant2, noise_level) = match phoneme_id {
            0 => (0.0, 0.0, 0.0, 0.0),  // Silence/padding
            1 | 2 => (0.0, 0.0, 0.0, 0.0),  // Start/end tokens
            3 => (0.0, 0.0, 0.0, 0.0),  // Space
            // Vowels (lower IDs in piper map)
            14..=21 | 39 | 59 | 61 | 74 => {  // a, e, i, o, u and variations
                (220.0 * pitch, 500.0, 1500.0, 0.1)
            },
            // Consonants
            15 | 28 | 25 | 26 => {  // b, p, m, n
                (150.0 * pitch, 800.0, 1200.0, 0.3)
            },
            19 | 31 | 96 | 108 => {  // f, s, sh, zh (fricatives)
                (0.0, 4000.0, 6000.0, 0.8)
            },
            _ => (180.0 * pitch, 1000.0, 2000.0, 0.2),  // Default
        };
        
        for i in 0..sample_count {
            let t = i as f32 / sample_rate;
            let phase = 2.0 * std::f32::consts::PI * t;
            
            let mut sample = 0.0;
            
            // Fundamental frequency
            if base_freq > 0.0 {
                sample += (phase * base_freq).sin() * 0.3;
            }
            
            // Formants (resonance frequencies)
            if formant1 > 0.0 {
                sample += (phase * formant1).sin() * 0.15;
            }
            if formant2 > 0.0 {
                sample += (phase * formant2).sin() * 0.1;
            }
            
            // Noise component for consonants
            if noise_level > 0.0 {
                let noise = ((i as f32 * 0.1).sin() * (i as f32 * 0.13).cos()) * noise_level;
                sample += noise * 0.2;
            }
            
            // ADSR envelope
            let attack = 0.01;
            let decay = 0.02;
            let sustain = 0.7;
            let release = 0.02;
            
            let duration = sample_count as f32 / sample_rate;
            let envelope = if t < attack {
                t / attack
            } else if t < attack + decay {
                1.0 - (1.0 - sustain) * ((t - attack) / decay)
            } else if t < duration - release {
                sustain
            } else {
                sustain * ((duration - t) / release)
            };
            
            sample *= envelope;
            samples.push(sample * 0.5);
        }
        
        samples
    }
    
    fn apply_smoothing(&self, samples: &mut [f32]) {
        if samples.len() < 3 {
            return;
        }
        
        // Simple moving average for smoothing
        let window_size = 5;
        let mut smoothed = vec![0.0; samples.len()];
        
        for i in 0..samples.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(samples.len());
            let sum: f32 = samples[start..end].iter().sum();
            smoothed[i] = sum / (end - start) as f32;
        }
        
        samples.copy_from_slice(&smoothed);
    }
    
    fn synthesize_fallback(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        log::warn!("ORT not available, using fallback synthesis");
        
        // Simple fallback - generate tone based on text length
        let duration = (text.len() as f32 * 0.05 / params.speed).max(0.5);
        let sample_count = (self.config.sample_rate as f32 * duration) as usize;
        
        let mut samples = Vec::with_capacity(sample_count);
        let frequency = 440.0 * params.pitch;
        
        for i in 0..sample_count {
            let t = i as f32 / self.config.sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.3;
            samples.push(sample);
        }
        
        Ok(AudioData {
            samples,
            sample_rate: self.config.sample_rate,
            channels: 1,
        })
    }
}

#[async_trait]
impl TTSEngine for PiperTTSEngine {
    fn name(&self) -> &str {
        "piper"
    }
    
    fn capabilities(&self) -> &EngineCapabilities {
        &self.capabilities
    }
    
    async fn synthesize(&self, text: &str, params: &SynthesisParams) -> Result<AudioData> {
        self.validate_text(text)?;
        bail!("Piper TTS requires ONNX Runtime. Please use Windows SAPI engine or compile with --features onnx")
    }
    
    fn list_voices(&self) -> Vec<VoiceInfo> {
        self.capabilities.supported_voices.clone()
    }
}
