/// Production TTS API - Clean, modular design
use actix_web::{web, HttpResponse};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::tts_manager::{TTSManager, TTSManagerStats};
use crate::core::tts_engine::{SynthesisParams, VoiceInfo};
use crate::core::audio::AudioProcessor;
use crate::error::ApiError;

/// API State
pub struct TTSState {
    pub manager: Arc<TTSManager>,
}

/// Synthesis request
#[derive(Debug, Deserialize)]
pub struct SynthesisRequest {
    pub text: String,
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default = "default_speed")]
    pub speed: f32,
    #[serde(default = "default_pitch")]
    pub pitch: f32,
    #[serde(default)]
    pub language: Option<String>,
}

fn default_speed() -> f32 { 1.0 }
fn default_pitch() -> f32 { 1.0 }

/// Synthesis response
#[derive(Debug, Serialize)]
pub struct SynthesisResponse {
    pub audio_base64: String,
    pub sample_rate: u32,
    pub duration_secs: f32,
    pub format: String,
    pub engine_used: String,
}

/// Engine list response
#[derive(Debug, Serialize)]
pub struct EngineListResponse {
    pub engines: Vec<EngineInfoResponse>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct EngineInfoResponse {
    pub id: String,
    pub name: String,
    pub version: String,
    pub languages: Vec<String>,
    pub voices_count: usize,
}

/// Voice list response
#[derive(Debug, Serialize)]
pub struct VoiceListResponse {
    pub voices: Vec<VoiceInfo>,
    pub total: usize,
    pub engine: String,
}

/// Manager stats response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub stats: TTSManagerStats,
}

/// POST /tts/synthesize - Synthesize speech
pub async fn synthesize(
    req: web::Json<SynthesisRequest>,
    state: web::Data<TTSState>,
) -> Result<HttpResponse, ApiError> {
    // Validate input
    if req.text.is_empty() {
        return Err(ApiError::BadRequest("Text cannot be empty".to_string()));
    }
    
    if req.text.len() > 50000 {
        return Err(ApiError::BadRequest("Text too long (max 50000 characters)".to_string()));
    }
    
    // Prepare synthesis parameters
    let params = SynthesisParams {
        speed: req.speed.max(0.25).min(4.0), // Clamp speed
        pitch: req.pitch.max(0.5).min(2.0),  // Clamp pitch
        voice: req.voice.clone(),
        language: req.language.clone(),
    };
    
    // Synthesize
    let audio = state.manager.synthesize(
        &req.text,
        req.engine.as_deref(),
        params
    ).await.map_err(|e| ApiError::InternalError(format!("Synthesis failed: {}", e)))?;
    
    // Convert to WAV — AudioProcessor is stateless; use Default to avoid a
    // per-request heap allocation for the struct itself.
    let wav_data = AudioProcessor::default()
        .save_wav(&audio)
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    // Encode to base64 using the modern Engine API (base64 0.21+).
    use base64::Engine as _;
    let audio_base64 = base64::engine::general_purpose::STANDARD.encode(&wav_data);
    let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
    
    let engine_used = if let Some(engine_id) = req.engine.as_deref() {
        engine_id.to_string()
    } else {
        state.manager.get_default_engine()
            .map(|e| e.name().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    };
    
    Ok(HttpResponse::Ok().json(SynthesisResponse {
        audio_base64,
        sample_rate: audio.sample_rate,
        duration_secs,
        format: "wav".to_string(),
        engine_used,
    }))
}

/// GET /tts/engines - List available engines
pub async fn list_engines(
    state: web::Data<TTSState>,
) -> Result<HttpResponse, ApiError> {
    let engine_ids = state.manager.list_engines();
    
    let mut engines = Vec::new();
    for id in engine_ids {
        if let Some(caps) = state.manager.get_capabilities(&id) {
            engines.push(EngineInfoResponse {
                id: id.clone(),
                name: caps.name,
                version: caps.version,
                languages: caps.supported_languages,
                voices_count: caps.supported_voices.len(),
            });
        }
    }
    
    Ok(HttpResponse::Ok().json(EngineListResponse {
        total: engines.len(),
        engines,
    }))
}

/// GET /tts/engines/{engine_id}/capabilities - Get engine capabilities
pub async fn get_capabilities(
    engine_id: web::Path<String>,
    state: web::Data<TTSState>,
) -> Result<HttpResponse, ApiError> {
    let caps = state.manager.get_capabilities(&engine_id)
        .ok_or_else(|| ApiError::NotFound(format!("Engine '{}' not found", engine_id)))?;
    
    Ok(HttpResponse::Ok().json(caps))
}

/// GET /tts/engines/{engine_id}/voices - List voices for an engine
pub async fn list_voices(
    engine_id: web::Path<String>,
    state: web::Data<TTSState>,
) -> Result<HttpResponse, ApiError> {
    let voices = state.manager
        .with_engine(&engine_id, |e| e.list_voices())
        .ok_or_else(|| ApiError::NotFound(format!("Engine '{}' not found", engine_id)))?;

    Ok(HttpResponse::Ok().json(VoiceListResponse {
        total: voices.len(),
        voices,
        engine: engine_id.to_string(),
    }))
}

/// GET /tts/stats - Get TTS manager statistics
pub async fn get_stats(
    state: web::Data<TTSState>,
) -> Result<HttpResponse, ApiError> {
    let stats = state.manager.get_stats();
    Ok(HttpResponse::Ok().json(StatsResponse { stats }))
}

/// GET /tts/health - Health check
pub async fn health_check(
    state: web::Data<TTSState>,
) -> Result<HttpResponse, ApiError> {
    let engines_count = state.manager.list_engines().len();
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "engines_loaded": engines_count,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    })))
}

/// Configure TTS routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/tts")
            .route("/synthesize", web::post().to(synthesize))
            .route("/engines", web::get().to(list_engines))
            .route("/engines/{engine_id}/capabilities", web::get().to(get_capabilities))
            .route("/engines/{engine_id}/voices", web::get().to(list_voices))
            .route("/stats", web::get().to(get_stats))
            .route("/health", web::get().to(health_check))
    );
}
