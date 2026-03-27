use serde::{Deserialize, Serialize};
use std::io::Cursor;
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

    fn probe_metadata_with_symphonia(&self, data: &[u8], format: AudioFormat) -> Result<AudioMetadata> {
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;

        let cursor = Cursor::new(data.to_vec());
        let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

        let mut hint = Hint::new();
        match &format {
            AudioFormat::Mp3 => { hint.with_extension("mp3"); }
            AudioFormat::Flac => { hint.with_extension("flac"); }
            AudioFormat::Ogg => { hint.with_extension("ogg"); }
            AudioFormat::Wav => { hint.with_extension("wav"); }
        }

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
            .context("Failed to probe audio format")?;

        let format_reader = probed.format;
        let track = format_reader.default_track()
            .context("No default track found in audio data")?;

        let codec_params = &track.codec_params;

        let sample_rate = codec_params.sample_rate
            .context("No sample rate in codec params")?;

        let channels = codec_params.channels
            .map(|c| c.count() as u32)
            .unwrap_or(1);

        let duration_secs = match (codec_params.n_frames, codec_params.sample_rate) {
            (Some(n_frames), Some(sr)) if sr > 0 => n_frames as f32 / sr as f32,
            _ => 0.0,
        };

        Ok(AudioMetadata {
            format,
            sample_rate,
            channels: channels as u16,
            duration_secs,
            bits_per_sample: codec_params.bits_per_sample.unwrap_or(16) as u16,
        })
    }

    fn validate_mp3(&self, data: &[u8]) -> Result<AudioMetadata> {
        self.probe_metadata_with_symphonia(data, AudioFormat::Mp3)
    }

    fn validate_flac(&self, data: &[u8]) -> Result<AudioMetadata> {
        self.probe_metadata_with_symphonia(data, AudioFormat::Flac)
    }

    fn validate_ogg(&self, data: &[u8]) -> Result<AudioMetadata> {
        self.probe_metadata_with_symphonia(data, AudioFormat::Ogg)
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

    /// Resample audio to `target_sample_rate` using rubato's FFT resampler.
    ///
    /// rubato's `FftFixedInOut` is SIMD-accelerated and alias-free, unlike the
    /// previous linear-interpolation fallback.  It operates on per-channel
    /// slices so multi-channel audio is handled correctly.
    pub fn resample(&self, audio: &AudioData, target_sample_rate: u32) -> Result<AudioData> {
        if audio.sample_rate == target_sample_rate {
            return Ok(audio.clone());
        }

        use rubato::{FftFixedInOut, Resampler};

        let channels  = audio.channels as usize;
        let in_rate   = audio.sample_rate as usize;
        let out_rate  = target_sample_rate as usize;

        // Process in chunks of 1024 input frames.
        let chunk_size = 1024usize;
        let mut resampler = FftFixedInOut::<f32>::new(
            in_rate,
            out_rate,
            chunk_size,
            channels,
        ).context("Failed to create FFT resampler")?;

        // De-interleave: rubato expects [channel][frame] layout.
        let total_frames = audio.samples.len() / channels;
        let mut channel_bufs: Vec<Vec<f32>> = (0..channels)
            .map(|c| {
                (0..total_frames)
                    .map(|f| audio.samples[f * channels + c])
                    .collect()
            })
            .collect();

        // Process full chunks, then flush the remainder.
        let mut out_channels: Vec<Vec<f32>> = vec![Vec::new(); channels];
        let mut pos = 0usize;

        while pos + chunk_size <= total_frames {
            let in_chunk: Vec<&[f32]> = channel_bufs.iter()
                .map(|ch| &ch[pos..pos + chunk_size])
                .collect();
            let out = resampler.process(&in_chunk, None)
                .context("Resampler error during full-chunk processing")?;
            for (c, ch_out) in out.iter().enumerate() {
                out_channels[c].extend_from_slice(ch_out);
            }
            pos += chunk_size;
        }

        // Flush remaining samples (zero-padded by rubato internally).
        if pos < total_frames {
            let in_partial: Vec<Vec<f32>> = channel_bufs.iter()
                .map(|ch| ch[pos..].to_vec())
                .collect();
            let in_refs: Vec<&[f32]> = in_partial.iter().map(|v| v.as_slice()).collect();
            let out = resampler.process_partial(Some(&in_refs), None)
                .context("Resampler error during partial-chunk flush")?;
            for (c, ch_out) in out.iter().enumerate() {
                out_channels[c].extend_from_slice(ch_out);
            }
        } else {
            // Flush resampler internal delay line even when input was chunk-aligned.
            if let Ok(out) = resampler.process_partial(None::<&[&[f32]]>, None) {
                for (c, ch_out) in out.iter().enumerate() {
                    out_channels[c].extend_from_slice(ch_out);
                }
            }
        }

        // Re-interleave output frames.
        let out_frames = out_channels[0].len();
        let mut resampled = Vec::with_capacity(out_frames * channels);
        for f in 0..out_frames {
            for c in 0..channels {
                resampled.push(out_channels[c][f]);
            }
        }

        Ok(AudioData {
            samples: resampled,
            sample_rate: target_sample_rate,
            channels: audio.channels,
        })
    }

    /// Save audio as WAV (16-bit PCM for maximum compatibility).
    ///
    /// The output buffer is pre-allocated to the exact expected size:
    ///   44-byte RIFF/WAV header + 2 bytes per sample (16-bit PCM).
    /// This avoids the default `Vec::new()` growth / realloc sequence.
    pub fn save_wav(&self, audio: &AudioData) -> Result<Vec<u8>> {
        let spec = hound::WavSpec {
            channels: audio.channels,
            sample_rate: audio.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        // 44-byte WAV header + 2 bytes per sample × channels
        let estimated = 44 + audio.samples.len() * 2;
        let mut cursor = Cursor::new(Vec::with_capacity(estimated));
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec)
                .context("Failed to create WAV writer")?;

            // Convert f32 samples (-1.0 to 1.0) to i16 (-32768 to 32767)
            for &sample in &audio.samples {
                let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                writer.write_sample(sample_i16)
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

    #[test]
    fn test_validate_mp3_rejects_invalid_data() {
        let audio_processor = AudioProcessor::new();
        let result = audio_processor.validate_mp3(&[0u8; 16]);
        assert!(result.is_err(), "validate_mp3 should reject invalid data, got Ok");
    }

    #[test]
    fn test_validate_flac_rejects_invalid_data() {
        let audio_processor = AudioProcessor::new();
        let result = audio_processor.validate_flac(&[0u8; 16]);
        assert!(result.is_err(), "validate_flac should reject invalid data, got Ok");
    }

    #[test]
    fn test_validate_ogg_rejects_invalid_data() {
        let audio_processor = AudioProcessor::new();
        let result = audio_processor.validate_ogg(&[0u8; 16]);
        assert!(result.is_err(), "validate_ogg should reject invalid data, got Ok");
    }
}
