use actix_web::{web, HttpResponse, Result};
use actix_multipart::Multipart;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::core::audio::AudioProcessor;
use crate::core::audio_models::{AudioModelManager, TTSParameters};
use crate::error::ApiError;
use crate::security::sanitizer::Sanitizer;

#[derive(Debug, Deserialize)]
pub struct SynthesizeRequest {
    pub text: String,
    pub model: Option<String>,
    pub voice: Option<String>,
    pub speed: Option<f32>,
    pub pitch: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct SynthesizeResponse {
    pub audio_base64: String,
    pub sample_rate: u32,
    pub duration_secs: f32,
    pub format: String,
}

#[derive(Debug, Deserialize)]
pub struct TranscribeRequest {
    pub language: Option<String>,
    pub timestamps: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    pub language: Option<String>,
    pub confidence: f32,
    pub segments: Option<Vec<TranscriptSegment>>,
}

#[derive(Debug, Serialize)]
pub struct TranscriptSegment {
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub confidence: f32,
}

#[derive(Debug, Serialize)]
pub struct AudioValidationResponse {
    pub valid: bool,
    pub format: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_secs: f32,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct AudioHealthResponse {
    pub status: String,
    pub audio_backend: String,
    pub supported_formats: Vec<String>,
    pub models_available: Vec<String>,
}

pub struct AudioState {
    pub model_manager: Arc<AudioModelManager>,
    pub sanitizer: Sanitizer,
}

pub async fn synthesize_speech(
    req: web::Json<SynthesizeRequest>,
    state: web::Data<AudioState>,
) -> Result<HttpResponse, ApiError> {
    if req.text.is_empty() {
        return Err(ApiError::BadRequest("Text cannot be empty".to_string()));
    }

    // Sanitize text
    let sanitized_text = state.sanitizer.sanitize_input(&serde_json::json!(req.text))
        .map_err(|e| ApiError::BadRequest(format!("Invalid input: {}", e)))?
        .as_str()
        .ok_or_else(|| ApiError::BadRequest("Sanitized text is not a string".to_string()))?
        .to_string();

    // Get model name (default to "default")
    let model_name = req.model.as_deref().unwrap_or("default");
    
    // Get TTS model
    let model = state.model_manager.get_tts_model(model_name)
        .ok_or_else(|| ApiError::NotFound(format!("TTS model '{}' not found", model_name)))?;

    // Prepare parameters
    let params = TTSParameters {
        speed: req.speed.unwrap_or(1.0),
        pitch: req.pitch.unwrap_or(1.0),
        energy: 1.0,
    };

    // Synthesize audio
    let audio = model.synthesize(&sanitized_text, &params)
        .map_err(|e| ApiError::InternalError(format!("TTS synthesis failed: {}", e)))?;

    // Convert to WAV
    let processor = AudioProcessor::new();
    let wav_data = processor.save_wav(&audio)
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
    let audio_base64 = base64_encode(&wav_data);

    Ok(HttpResponse::Ok().json(SynthesizeResponse {
        audio_base64,
        sample_rate: audio.sample_rate,
        duration_secs,
        format: "wav".to_string(),
    }))
}

pub async fn transcribe_audio(
    mut payload: Multipart,
    state: web::Data<AudioState>,
) -> Result<HttpResponse, ApiError> {
    let mut audio_data = Vec::new();
    let mut model_name = "default".to_string();
    let mut return_timestamps = false;

    // Extract audio file and parameters from multipart
    while let Some(item) = payload.next().await {
        let mut field = item.map_err(|e| ApiError::BadRequest(e.to_string()))?;
        let content_disposition = field.content_disposition();
        let field_name = content_disposition.get_name().unwrap_or("");

        if field_name == "audio" || field_name == "file" {
            while let Some(chunk) = field.next().await {
                let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                audio_data.extend_from_slice(&data);
            }
        } else if field_name == "model" {
            let mut model_str = String::new();
            while let Some(chunk) = field.next().await {
                let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                model_str.push_str(&String::from_utf8_lossy(&data));
            }
            model_name = model_str;
        } else if field_name == "timestamps" {
            let mut ts_str = String::new();
            while let Some(chunk) = field.next().await {
                let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
                ts_str.push_str(&String::from_utf8_lossy(&data));
            }
            return_timestamps = ts_str.trim().parse().unwrap_or(false);
        }
    }

    if audio_data.is_empty() {
        return Err(ApiError::BadRequest("No audio data provided".to_string()));
    }

    // Load and validate audio
    let processor = AudioProcessor::new();
    let audio = processor.load_audio(&audio_data)
        .map_err(|e| ApiError::BadRequest(format!("Invalid audio: {}", e)))?;

    // Get STT model
    let model = state.model_manager.get_stt_model(&model_name)
        .ok_or_else(|| ApiError::NotFound(format!("STT model '{}' not found", model_name)))?;

    // Transcribe
    let result = model.transcribe(&audio, return_timestamps)
        .map_err(|e| ApiError::InternalError(format!("Transcription failed: {}", e)))?;

    // Convert to response format
    let segments = result.segments.map(|segs| {
        segs.into_iter().map(|seg| TranscriptSegment {
            text: seg.text,
            start: seg.start_time,
            end: seg.end_time,
            confidence: seg.confidence,
        }).collect()
    });

    Ok(HttpResponse::Ok().json(TranscribeResponse {
        text: result.text,
        language: result.language,
        confidence: result.confidence,
        segments,
    }))
}

pub async fn validate_audio(
    mut payload: Multipart,
) -> Result<HttpResponse, ApiError> {
    let mut audio_data = Vec::new();
    let mut errors = Vec::new();

    while let Some(item) = payload.next().await {
        let mut field = item.map_err(|e| ApiError::BadRequest(e.to_string()))?;
        
        while let Some(chunk) = field.next().await {
            let data = chunk.map_err(|e| ApiError::BadRequest(e.to_string()))?;
            audio_data.extend_from_slice(&data);
        }
    }

    if audio_data.is_empty() {
        errors.push("No audio data provided".to_string());
        return Ok(HttpResponse::Ok().json(AudioValidationResponse {
            valid: false,
            format: "unknown".to_string(),
            sample_rate: 0,
            channels: 0,
            duration_secs: 0.0,
            errors,
        }));
    }

    let processor = AudioProcessor::new();
    
    match processor.validate_audio(&audio_data) {
        Ok(metadata) => {
            Ok(HttpResponse::Ok().json(AudioValidationResponse {
                valid: true,
                format: format!("{:?}", metadata.format),
                sample_rate: metadata.sample_rate,
                channels: metadata.channels,
                duration_secs: metadata.duration_secs,
                errors: vec![],
            }))
        }
        Err(e) => {
            errors.push(e.to_string());
            Ok(HttpResponse::Ok().json(AudioValidationResponse {
                valid: false,
                format: "unknown".to_string(),
                sample_rate: 0,
                channels: 0,
                duration_secs: 0.0,
                errors,
            }))
        }
    }
}

pub async fn audio_health(state: web::Data<AudioState>) -> Result<HttpResponse, ApiError> {
    let tts_models = state.model_manager.list_tts_models();
    let stt_models = state.model_manager.list_stt_models();
    
    let mut models_available = Vec::new();
    for model in tts_models {
        models_available.push(format!("TTS: {}", model));
    }
    for model in stt_models {
        models_available.push(format!("STT: {}", model));
    }

    let backend = if cfg!(feature = "onnx") {
        "ONNX Runtime + Symphonia"
    } else {
        "Symphonia (fallback mode)"
    };

    Ok(HttpResponse::Ok().json(AudioHealthResponse {
        status: "ok".to_string(),
        audio_backend: backend.to_string(),
        supported_formats: vec![
            "wav".to_string(),
            "mp3".to_string(),
            "flac".to_string(),
            "ogg".to_string(),
        ],
        models_available,
    }))
}

// Helper function to encode base64
fn base64_encode(data: &[u8]) -> String {
    use base64::{Engine as _, engine::general_purpose};
    general_purpose::STANDARD.encode(data)
}
