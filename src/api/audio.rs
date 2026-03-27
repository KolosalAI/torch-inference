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

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web, App};
    use actix_web::http::StatusCode;
    use crate::core::audio_models::AudioModelManager;

    fn make_audio_state() -> web::Data<AudioState> {
        use crate::config::SanitizerConfig;
        web::Data::new(AudioState {
            model_manager: Arc::new(AudioModelManager::new("/tmp")),
            sanitizer: crate::security::sanitizer::Sanitizer::new(SanitizerConfig::default()),
        })
    }

    // ── audio_health ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_audio_health_returns_200() {
        let state = make_audio_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/health", web::get().to(audio_health))
        ).await;
        let req = test::TestRequest::get().uri("/audio/health").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_audio_health_response_body() {
        let state = make_audio_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/health", web::get().to(audio_health))
        ).await;
        let req = test::TestRequest::get().uri("/audio/health").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["status"], "ok");
        assert!(body["supported_formats"].is_array());
        assert!(body["models_available"].is_array());
        let formats: Vec<&str> = body["supported_formats"].as_array().unwrap()
            .iter().filter_map(|v| v.as_str()).collect();
        assert!(formats.contains(&"wav"));
    }

    #[tokio::test]
    async fn test_audio_health_empty_model_manager() {
        let state = make_audio_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/health", web::get().to(audio_health))
        ).await;
        let req = test::TestRequest::get().uri("/audio/health").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        // No models loaded, so models_available should be empty
        let models = body["models_available"].as_array().unwrap();
        assert!(models.is_empty());
    }

    // ── synthesize_speech ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_synthesize_speech_empty_text_returns_400() {
        let state = make_audio_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/synthesize", web::post().to(synthesize_speech))
        ).await;
        let req = test::TestRequest::post()
            .uri("/audio/synthesize")
            .set_json(&serde_json::json!({"text": ""}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_synthesize_speech_model_not_found_returns_404() {
        let state = make_audio_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/synthesize", web::post().to(synthesize_speech))
        ).await;
        let req = test::TestRequest::post()
            .uri("/audio/synthesize")
            .set_json(&serde_json::json!({"text": "hello world", "model": "nonexistent"}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_synthesize_speech_default_model_not_found_returns_404() {
        let state = make_audio_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/synthesize", web::post().to(synthesize_speech))
        ).await;
        // No model field — defaults to "default" which doesn't exist
        let req = test::TestRequest::post()
            .uri("/audio/synthesize")
            .set_json(&serde_json::json!({"text": "hello"}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── synthesize_request serde ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_synthesize_request_serde_all_fields() {
        let json = r#"{"text":"hello","model":"default","voice":"speaker1","speed":1.2,"pitch":0.9}"#;
        let req: SynthesizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "hello");
        assert_eq!(req.model.as_deref(), Some("default"));
        assert_eq!(req.voice.as_deref(), Some("speaker1"));
        assert!((req.speed.unwrap() - 1.2).abs() < 1e-5);
        assert!((req.pitch.unwrap() - 0.9).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_synthesize_request_serde_minimal() {
        let json = r#"{"text":"test"}"#;
        let req: SynthesizeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "test");
        assert!(req.model.is_none());
        assert!(req.voice.is_none());
        assert!(req.speed.is_none());
        assert!(req.pitch.is_none());
    }

    #[tokio::test]
    async fn test_synthesize_response_serialization() {
        let resp = SynthesizeResponse {
            audio_base64: "base64data".to_string(),
            sample_rate: 22050,
            duration_secs: 2.5,
            format: "wav".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["sample_rate"], 22050);
        assert_eq!(back["format"], "wav");
    }

    #[tokio::test]
    async fn test_transcribe_request_serde() {
        let json = r#"{"language":"en","timestamps":true}"#;
        let req: TranscribeRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.language.as_deref(), Some("en"));
        assert_eq!(req.timestamps, Some(true));
    }

    #[tokio::test]
    async fn test_transcribe_request_serde_minimal() {
        let json = r#"{}"#;
        let req: TranscribeRequest = serde_json::from_str(json).unwrap();
        assert!(req.language.is_none());
        assert!(req.timestamps.is_none());
    }

    #[tokio::test]
    async fn test_transcribe_response_serialization() {
        let resp = TranscribeResponse {
            text: "hello world".to_string(),
            language: Some("en".to_string()),
            confidence: 0.95,
            segments: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["text"], "hello world");
        assert!((back["confidence"].as_f64().unwrap() - 0.95).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_transcript_segment_serialization() {
        let seg = TranscriptSegment {
            text: "hi".to_string(),
            start: 0.0,
            end: 1.5,
            confidence: 0.9,
        };
        let json = serde_json::to_string(&seg).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["text"], "hi");
        assert!((back["end"].as_f64().unwrap() - 1.5).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_audio_validation_response_serialization() {
        let resp = AudioValidationResponse {
            valid: true,
            format: "wav".to_string(),
            sample_rate: 44100,
            channels: 2,
            duration_secs: 3.0,
            errors: vec![],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["valid"], true);
        assert_eq!(back["channels"], 2);
    }

    #[tokio::test]
    async fn test_audio_health_response_serialization() {
        let resp = AudioHealthResponse {
            status: "ok".to_string(),
            audio_backend: "Symphonia".to_string(),
            supported_formats: vec!["wav".to_string(), "mp3".to_string()],
            models_available: vec!["TTS: default".to_string()],
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["status"], "ok");
        assert_eq!(back["supported_formats"][0], "wav");
    }

    #[tokio::test]
    async fn test_base64_encode_helper() {
        let data = b"hello";
        let encoded = base64_encode(data);
        use base64::{Engine as _, engine::general_purpose};
        let expected = general_purpose::STANDARD.encode(data);
        assert_eq!(encoded, expected);
    }

    // ── additional helpers ────────────────────────────────────────────────────

    /// Build a minimal valid WAV file in memory (silence, 16-bit PCM).
    fn make_wav_bytes_for_test(sample_rate: u32, channels: u16, samples: u32) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut buf, spec).unwrap();
        for _ in 0..samples {
            for _ in 0..channels {
                writer.write_sample(0i16).unwrap();
            }
        }
        writer.finalize().unwrap();
        buf.into_inner()
    }

    /// Build a multipart/form-data body with a single file field.
    fn make_multipart_body(boundary: &str, field_name: &str, data: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        let header = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        );
        body.extend_from_slice(header.as_bytes());
        body.extend_from_slice(data);
        let footer = format!("\r\n--{boundary}--\r\n");
        body.extend_from_slice(footer.as_bytes());
        body
    }

    /// Build an AudioState that has a TTS model named "default" pre-loaded.
    async fn make_audio_state_with_tts() -> web::Data<AudioState> {
        use crate::config::SanitizerConfig;
        use crate::core::audio_models::{TTSConfig, AudioModelManager};
        use std::path::PathBuf;

        let manager = Arc::new(AudioModelManager::new("/tmp"));
        let config = TTSConfig {
            model_path: PathBuf::from("/tmp/fake_tts.onnx"),
            vocoder_path: None,
            sample_rate: 16000,
            max_text_length: 500,
        };
        manager.load_tts_model("default", config).await.unwrap();

        web::Data::new(AudioState {
            model_manager: manager,
            sanitizer: crate::security::sanitizer::Sanitizer::new(SanitizerConfig::default()),
        })
    }

    /// Build an AudioState that has an STT model named "default" pre-loaded.
    async fn make_audio_state_with_stt() -> web::Data<AudioState> {
        use crate::config::SanitizerConfig;
        use crate::core::audio_models::{STTConfig, AudioModelManager};
        use std::path::PathBuf;

        let manager = Arc::new(AudioModelManager::new("/tmp"));
        let config = STTConfig {
            model_path: PathBuf::from("/tmp/fake_stt.onnx"),
            sample_rate: 16000,
            chunk_length_secs: 30.0,
        };
        manager.load_stt_model("default", config).await.unwrap();

        web::Data::new(AudioState {
            model_manager: manager,
            sanitizer: crate::security::sanitizer::Sanitizer::new(SanitizerConfig::default()),
        })
    }

    // ── audio_health with loaded models ───────────────────────────────────────

    #[tokio::test]
    async fn test_audio_health_with_loaded_tts_model() {
        let state = make_audio_state_with_tts().await;
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/health", web::get().to(audio_health))
        ).await;
        let req = test::TestRequest::get().uri("/audio/health").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let models = body["models_available"].as_array().unwrap();
        assert!(!models.is_empty());
        let model_strs: Vec<&str> = models.iter().filter_map(|v| v.as_str()).collect();
        assert!(model_strs.iter().any(|s| s.starts_with("TTS:")));
    }

    #[tokio::test]
    async fn test_audio_health_with_loaded_stt_model() {
        let state = make_audio_state_with_stt().await;
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/health", web::get().to(audio_health))
        ).await;
        let req = test::TestRequest::get().uri("/audio/health").to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;
        let models = body["models_available"].as_array().unwrap();
        assert!(!models.is_empty());
        let model_strs: Vec<&str> = models.iter().filter_map(|v| v.as_str()).collect();
        assert!(model_strs.iter().any(|s| s.starts_with("STT:")));
    }

    // ── synthesize_speech success path ────────────────────────────────────────

    #[tokio::test]
    async fn test_synthesize_speech_success_returns_200() {
        let state = make_audio_state_with_tts().await;
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/synthesize", web::post().to(synthesize_speech))
        ).await;

        let req = test::TestRequest::post()
            .uri("/audio/synthesize")
            .set_json(&serde_json::json!({"text": "hello world", "model": "default"}))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let body: serde_json::Value = test::read_body_json(resp).await;
        assert!(body["audio_base64"].as_str().is_some());
        assert_eq!(body["format"], "wav");
        assert!(body["sample_rate"].as_u64().unwrap() > 0);
        assert!(body["duration_secs"].as_f64().unwrap() > 0.0);
    }

    #[tokio::test]
    async fn test_synthesize_speech_audio_base64_decodes_to_wav() {
        let state = make_audio_state_with_tts().await;
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/synthesize", web::post().to(synthesize_speech))
        ).await;

        let req = test::TestRequest::post()
            .uri("/audio/synthesize")
            .set_json(&serde_json::json!({"text": "verify base64", "model": "default"}))
            .to_request();
        let body: serde_json::Value = test::call_and_read_body_json(&app, req).await;

        use base64::{Engine as _, engine::general_purpose};
        let decoded = general_purpose::STANDARD
            .decode(body["audio_base64"].as_str().unwrap())
            .unwrap();
        assert!(decoded.len() > 4);
        assert_eq!(&decoded[0..4], b"RIFF", "decoded audio should be a WAV RIFF file");
    }

    #[tokio::test]
    async fn test_synthesize_speech_with_speed_and_pitch() {
        let state = make_audio_state_with_tts().await;
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/synthesize", web::post().to(synthesize_speech))
        ).await;

        let req = test::TestRequest::post()
            .uri("/audio/synthesize")
            .set_json(&serde_json::json!({
                "text": "testing speed and pitch",
                "model": "default",
                "speed": 1.5,
                "pitch": 0.8
            }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── validate_audio handler ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_validate_audio_no_audio_field_returns_invalid() {
        let app = test::init_service(
            App::new()
                .route("/audio/validate", web::post().to(validate_audio))
        ).await;

        // Multipart with only a non-audio field — audio_data stays empty
        let boundary = "noaudiobnd";
        let mut body = Vec::new();
        let part = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"other_field\"\r\n\r\nsome value"
        );
        body.extend_from_slice(part.as_bytes());
        let footer = format!("\r\n--{boundary}--\r\n");
        body.extend_from_slice(footer.as_bytes());

        let req = test::TestRequest::post()
            .uri("/audio/validate")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        // audio_data is empty → handler returns 200 with valid=false
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(resp_body["valid"], false);
        assert_eq!(resp_body["format"], "unknown");
        assert_eq!(resp_body["sample_rate"], 0);
    }

    #[tokio::test]
    async fn test_validate_audio_valid_wav_returns_valid_true() {
        let wav_bytes = make_wav_bytes_for_test(16000, 1, 16000);

        let app = test::init_service(
            App::new()
                .route("/audio/validate", web::post().to(validate_audio))
        ).await;

        let boundary = "wavboundary123";
        let body = make_multipart_body(boundary, "audio", &wav_bytes);

        let req = test::TestRequest::post()
            .uri("/audio/validate")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(resp_body["valid"], true);
        assert_eq!(resp_body["sample_rate"], 16000);
        assert_eq!(resp_body["channels"], 1);
    }

    #[tokio::test]
    async fn test_validate_audio_invalid_bytes_returns_valid_false() {
        let invalid_audio = b"this is not audio data at all".to_vec();

        let app = test::init_service(
            App::new()
                .route("/audio/validate", web::post().to(validate_audio))
        ).await;

        let boundary = "invalidboundary";
        let body = make_multipart_body(boundary, "file", &invalid_audio);

        let req = test::TestRequest::post()
            .uri("/audio/validate")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(resp_body["valid"], false);
        assert!(!resp_body["errors"].as_array().unwrap().is_empty());
    }

    // ── transcribe_audio handler ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_transcribe_audio_empty_payload_returns_400() {
        let state = make_audio_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/transcribe", web::post().to(transcribe_audio))
        ).await;

        let boundary = "emptyboundary";
        let empty_body = format!("--{boundary}--\r\n");
        let req = test::TestRequest::post()
            .uri("/audio/transcribe")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(empty_body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_transcribe_audio_stt_model_not_found_returns_404() {
        let state = make_audio_state(); // no STT model loaded
        let wav_bytes = make_wav_bytes_for_test(16000, 1, 16000);

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/transcribe", web::post().to(transcribe_audio))
        ).await;

        let boundary = "notfoundbnd";
        let body = make_multipart_body(boundary, "audio", &wav_bytes);
        let req = test::TestRequest::post()
            .uri("/audio/transcribe")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_transcribe_audio_success_without_timestamps() {
        let state = make_audio_state_with_stt().await;
        let wav_bytes = make_wav_bytes_for_test(16000, 1, 16000);

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/transcribe", web::post().to(transcribe_audio))
        ).await;

        let boundary = "successbnd";
        let body = make_multipart_body(boundary, "audio", &wav_bytes);
        let req = test::TestRequest::post()
            .uri("/audio/transcribe")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: serde_json::Value = test::read_body_json(resp).await;
        assert!(resp_body["text"].as_str().is_some());
        assert!(resp_body["confidence"].as_f64().is_some());
    }

    #[tokio::test]
    async fn test_transcribe_audio_success_with_timestamps() {
        let state = make_audio_state_with_stt().await;
        let wav_bytes = make_wav_bytes_for_test(16000, 1, 48000); // 3 seconds

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/transcribe", web::post().to(transcribe_audio))
        ).await;

        let boundary = "tsboundary";
        let mut body = Vec::new();
        let audio_part = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        );
        body.extend_from_slice(audio_part.as_bytes());
        body.extend_from_slice(&wav_bytes);
        let ts_part = format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"timestamps\"\r\n\r\ntrue"
        );
        body.extend_from_slice(ts_part.as_bytes());
        let footer = format!("\r\n--{boundary}--\r\n");
        body.extend_from_slice(footer.as_bytes());

        let req = test::TestRequest::post()
            .uri("/audio/transcribe")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: serde_json::Value = test::read_body_json(resp).await;
        assert!(resp_body["text"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_transcribe_audio_invalid_audio_data_returns_400() {
        let state = make_audio_state_with_stt().await;

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/transcribe", web::post().to(transcribe_audio))
        ).await;

        let boundary = "invalidaudiobnd";
        let body = make_multipart_body(boundary, "audio", b"not valid audio data");
        let req = test::TestRequest::post()
            .uri("/audio/transcribe")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    // ── Additional gap-closing tests ─────────────────────────────────────────

    /// Exercises the multipart "model" field parsing in transcribe_audio
    /// (lines 141-147): audio + model fields are both sent; the model name
    /// is read into model_name which then drives the STT model lookup.
    #[tokio::test]
    async fn test_transcribe_audio_with_explicit_model_field() {
        let state = make_audio_state_with_stt().await;
        let wav_bytes = make_wav_bytes_for_test(16000, 1, 16000);

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/transcribe", web::post().to(transcribe_audio))
        ).await;

        let boundary = "modelfieldtest";
        let mut body = Vec::new();

        // audio part
        let audio_part = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        );
        body.extend_from_slice(audio_part.as_bytes());
        body.extend_from_slice(&wav_bytes);

        // model part — uses the "default" model which IS loaded
        let model_part = format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ndefault"
        );
        body.extend_from_slice(model_part.as_bytes());

        let footer = format!("\r\n--{boundary}--\r\n");
        body.extend_from_slice(footer.as_bytes());

        let req = test::TestRequest::post()
            .uri("/audio/transcribe")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// Exercises transcribe_audio when model field names a non-existent STT model,
    /// confirming the 404 path (line 169) is reached.
    #[tokio::test]
    async fn test_transcribe_audio_explicit_model_not_found() {
        let state = make_audio_state_with_stt().await;
        let wav_bytes = make_wav_bytes_for_test(16000, 1, 16000);

        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/transcribe", web::post().to(transcribe_audio))
        ).await;

        let boundary = "nomodelbnd";
        let mut body = Vec::new();

        let audio_part = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        );
        body.extend_from_slice(audio_part.as_bytes());
        body.extend_from_slice(&wav_bytes);

        let model_part = format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nnonexistent-stt-model"
        );
        body.extend_from_slice(model_part.as_bytes());

        let footer = format!("\r\n--{boundary}--\r\n");
        body.extend_from_slice(footer.as_bytes());

        let req = test::TestRequest::post()
            .uri("/audio/transcribe")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    /// Exercises the validate_audio handler with a "file" field name (line 136:
    /// `field_name == "audio" || field_name == "file"`).
    #[tokio::test]
    async fn test_validate_audio_with_file_field_name() {
        let wav_bytes = make_wav_bytes_for_test(22050, 2, 22050);

        let app = test::init_service(
            App::new()
                .route("/audio/validate", web::post().to(validate_audio))
        ).await;

        let boundary = "filenametest";
        // Use "file" instead of "audio" as field name
        let body = make_multipart_body(boundary, "file", &wav_bytes);

        let req = test::TestRequest::post()
            .uri("/audio/validate")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: serde_json::Value = test::read_body_json(resp).await;
        // Valid WAV provided via "file" field name
        assert_eq!(resp_body["valid"], true);
    }

    /// Exercises synthesize_speech with voice and pitch optional fields present.
    #[tokio::test]
    async fn test_synthesize_speech_with_all_optional_fields() {
        let state = make_audio_state_with_tts().await;
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/synthesize", web::post().to(synthesize_speech))
        ).await;

        let req = test::TestRequest::post()
            .uri("/audio/synthesize")
            .set_json(&serde_json::json!({
                "text": "test with voice and pitch",
                "model": "default",
                "voice": "speaker_0",
                "speed": 0.8,
                "pitch": 1.1
            }))
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ── validate_audio: empty multipart body (lines 208-217) ──────────────────

    /// Exercises the `validate_audio` handler with a small invalid audio body
    /// so that `audio_data` is non-empty but fails validation (covers lines 208-244).
    #[tokio::test]
    async fn test_validate_audio_truly_empty_multipart_returns_invalid() {
        let app = test::init_service(
            App::new()
                .route("/audio/validate", web::post().to(validate_audio))
        ).await;

        let boundary = "emptyvalidatebnd";
        // Send a multipart with one field containing garbage bytes (not valid audio).
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"\r\n\r\nNOTAUDIO\r\n--{boundary}--\r\n"
        );

        let req = test::TestRequest::post()
            .uri("/audio/validate")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        // Handler returns 200 with valid=false when validation fails.
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(resp_body["valid"], false);
    }

    // ── transcribe_audio: empty audio_data after loop (line 159) ─────────────

    /// Exercises the `if audio_data.is_empty()` check in `transcribe_audio`
    /// (line 159) by sending a multipart that has a field name unrelated to
    /// "audio" or "file", so audio_data stays empty and the 400 error fires.
    #[tokio::test]
    async fn test_transcribe_audio_no_audio_field_returns_400() {
        let state = make_audio_state();
        let app = test::init_service(
            App::new()
                .app_data(state.clone())
                .route("/audio/transcribe", web::post().to(transcribe_audio))
        ).await;

        let boundary = "notaudiofieldbnd";
        // Send a multipart with a "metadata" field (not "audio"/"file") so
        // audio_data stays empty, hitting the is_empty() check at line 158-159.
        let mut body = Vec::new();
        let part = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"metadata\"\r\n\r\nsome text"
        );
        body.extend_from_slice(part.as_bytes());
        let footer = format!("\r\n--{boundary}--\r\n");
        body.extend_from_slice(footer.as_bytes());

        let req = test::TestRequest::post()
            .uri("/audio/transcribe")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Sends a multipart field with a zero-byte body so audio_data stays empty
    /// after the loop → covers lines 208-217 (the `if audio_data.is_empty()` branch).
    #[tokio::test]
    async fn test_validate_audio_empty_field_body_returns_valid_false() {
        let app = test::init_service(
            App::new().route("/audio/validate", web::post().to(validate_audio))
        ).await;

        let boundary = "emptybody123";
        // make_multipart_body with empty data → field has 0 bytes of content
        let body = make_multipart_body(boundary, "audio", &[]);

        let req = test::TestRequest::post()
            .uri("/audio/validate")
            .insert_header(("content-type", format!("multipart/form-data; boundary={boundary}")))
            .set_payload(body)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let resp_body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(resp_body["valid"], false);
        assert_eq!(resp_body["format"], "unknown");
        assert_eq!(resp_body["sample_rate"], 0);
        let errors = resp_body["errors"].as_array().unwrap();
        assert!(errors.iter().any(|e| e.as_str().unwrap_or("").contains("No audio data provided")));
    }
}
