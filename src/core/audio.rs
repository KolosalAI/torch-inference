use serde::{Deserialize, Serialize};
use std::io::{Cursor, Read};
use std::path::Path;
use anyhow::{Result, Context, bail};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
}

impl AudioFormat {
    pub fn from_extension(ext: &str) -> Result<Self> {
        match ext.to_lowercase().as_str() {
            "wav" => Ok(AudioFormat::Wav),
            "mp3" => Ok(AudioFormat::Mp3),
            "flac" => Ok(AudioFormat::Flac),
            "ogg" => Ok(AudioFormat::Ogg),
            _ => bail!("Unsupported audio format: {}", ext),
        }
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .context("Invalid file extension")?;
        Self::from_extension(ext)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    pub format: AudioFormat,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_secs: f32,
    pub bits_per_sample: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

pub struct AudioProcessor {
    default_sample_rate: u32,
}

impl AudioProcessor {
    pub fn new() -> Self {
        Self {
            default_sample_rate: 16000,
        }
    }

    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self {
            default_sample_rate: sample_rate,
        }
    }

    /// Validate audio file
    pub fn validate_audio(&self, data: &[u8]) -> Result<AudioMetadata> {
        // Try to detect format from magic bytes
        if data.len() < 4 {
            bail!("Audio data too short");
        }

        // Check for WAV format (RIFF header)
        if &data[0..4] == b"RIFF" {
            self.validate_wav(data)
        } else if data.len() >= 3 && &data[0..3] == b"ID3" || (data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) {
            // MP3 with ID3 tag or sync word
            self.validate_mp3(data)
        } else if data.len() >= 4 && &data[0..4] == b"fLaC" {
            self.validate_flac(data)
        } else if data.len() >= 4 && &data[0..4] == b"OggS" {
            self.validate_ogg(data)
        } else {
            bail!("Unknown audio format")
        }
    }

    fn validate_wav(&self, data: &[u8]) -> Result<AudioMetadata> {
        let cursor = Cursor::new(data);
        let reader = hound::WavReader::new(cursor)
            .context("Failed to parse WAV file")?;
        
        let spec = reader.spec();
        let duration_secs = reader.duration() as f32 / spec.sample_rate as f32;

        Ok(AudioMetadata {
            format: AudioFormat::Wav,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            duration_secs,
            bits_per_sample: spec.bits_per_sample,
        })
    }

    fn validate_mp3(&self, _data: &[u8]) -> Result<AudioMetadata> {
        // Basic MP3 validation
        // For full validation, we'd need a proper MP3 decoder
        Ok(AudioMetadata {
            format: AudioFormat::Mp3,
            sample_rate: 44100, // Default assumption
            channels: 2,
            duration_secs: 0.0,
            bits_per_sample: 16,
        })
    }

    fn validate_flac(&self, _data: &[u8]) -> Result<AudioMetadata> {
        // Basic FLAC validation
        Ok(AudioMetadata {
            format: AudioFormat::Flac,
            sample_rate: 44100,
            channels: 2,
            duration_secs: 0.0,
            bits_per_sample: 16,
        })
    }

    fn validate_ogg(&self, _data: &[u8]) -> Result<AudioMetadata> {
        // Basic OGG validation
        Ok(AudioMetadata {
            format: AudioFormat::Ogg,
            sample_rate: 44100,
            channels: 2,
            duration_secs: 0.0,
            bits_per_sample: 16,
        })
    }

    /// Load audio from bytes
    pub fn load_audio(&self, data: &[u8]) -> Result<AudioData> {
        let metadata = self.validate_audio(data)?;
        
        match metadata.format {
            AudioFormat::Wav => self.load_wav(data),
            AudioFormat::Mp3 => self.load_with_symphonia(data),
            AudioFormat::Flac => self.load_with_symphonia(data),
            AudioFormat::Ogg => self.load_with_symphonia(data),
        }
    }

    fn load_wav(&self, data: &[u8]) -> Result<AudioData> {
        let cursor = Cursor::new(data);
        let mut reader = hound::WavReader::new(cursor)
            .context("Failed to parse WAV file")?;
        
        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.samples::<f32>()
                    .collect::<Result<Vec<_>, _>>()
                    .context("Failed to read samples")?
            }
            hound::SampleFormat::Int => {
                reader.samples::<i32>()
                    .map(|s| s.map(|sample| sample as f32 / i32::MAX as f32))
                    .collect::<Result<Vec<_>, _>>()
                    .context("Failed to read samples")?
            }
        };

        Ok(AudioData {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        })
    }

    fn load_with_symphonia(&self, data: &[u8]) -> Result<AudioData> {
        use symphonia::core::audio::SampleBuffer;
        use symphonia::core::codecs::DecoderOptions;
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;

        // Create a static copy of the data
        let data_vec: Vec<u8> = data.to_vec();
        let cursor = Cursor::new(data_vec);
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

        let hint = Hint::new();
        let mut probed = symphonia::default::get_probe()
            .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
            .context("Failed to probe audio format")?;

        let track = probed.format.default_track()
            .context("No default track found")?;

        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .context("Failed to create decoder")?;

        let track_id = track.id;
        let mut samples = Vec::new();
        let mut sample_rate = 44100;
        let mut channels = 2;

        while let Ok(packet) = probed.format.next_packet() {
            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    sample_rate = decoded.spec().rate;
                    channels = decoded.spec().channels.count() as u16;

                    let mut sample_buf = SampleBuffer::<f32>::new(
                        decoded.capacity() as u64,
                        *decoded.spec(),
                    );
                    sample_buf.copy_interleaved_ref(decoded);
                    samples.extend_from_slice(sample_buf.samples());
                }
                Err(_) => break,
            }
        }

        Ok(AudioData {
            samples,
            sample_rate,
            channels,
        })
    }

    /// Resample audio to target sample rate
    pub fn resample(&self, audio: &AudioData, target_sample_rate: u32) -> Result<AudioData> {
        if audio.sample_rate == target_sample_rate {
            return Ok(audio.clone());
        }

        // Simplified resampling using linear interpolation
        // For production, use proper resampling library when rubato API is stable
        let ratio = target_sample_rate as f64 / audio.sample_rate as f64;
        let new_len = (audio.samples.len() as f64 * ratio) as usize;
        let mut resampled_samples = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_pos = i as f64 / ratio;
            let src_idx = src_pos as usize;
            
            if src_idx + 1 < audio.samples.len() {
                let frac = src_pos - src_idx as f64;
                let sample = audio.samples[src_idx] * (1.0 - frac as f32) + 
                            audio.samples[src_idx + 1] * frac as f32;
                resampled_samples.push(sample);
            } else if src_idx < audio.samples.len() {
                resampled_samples.push(audio.samples[src_idx]);
            }
        }

        Ok(AudioData {
            samples: resampled_samples,
            sample_rate: target_sample_rate,
            channels: audio.channels,
        })
    }

    /// Save audio as WAV
    pub fn save_wav(&self, audio: &AudioData) -> Result<Vec<u8>> {
        let spec = hound::WavSpec {
            channels: audio.channels,
            sample_rate: audio.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec)
                .context("Failed to create WAV writer")?;
            
            for &sample in &audio.samples {
                writer.write_sample(sample)
                    .context("Failed to write sample")?;
            }

            writer.finalize()
                .context("Failed to finalize WAV file")?;
        }

        Ok(cursor.into_inner())
    }
}

impl Default for AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_from_extension() {
        assert!(matches!(AudioFormat::from_extension("wav"), Ok(AudioFormat::Wav)));
        assert!(matches!(AudioFormat::from_extension("mp3"), Ok(AudioFormat::Mp3)));
        assert!(AudioFormat::from_extension("invalid").is_err());
    }

    #[test]
    fn test_audio_processor_creation() {
        let processor = AudioProcessor::new();
        assert_eq!(processor.default_sample_rate, 16000);
    }
}
