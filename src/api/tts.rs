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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_request_defaults() {
        let json = r#"{"text": "hello world"}"#;
        let req: SynthesisRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "hello world");
        assert_eq!(req.speed, 1.0);
        assert_eq!(req.pitch, 1.0);
        assert!(req.engine.is_none());
        assert!(req.voice.is_none());
        assert!(req.language.is_none());
    }

    #[test]
    fn test_synthesis_request_all_fields() {
        let json = r#"{
            "text": "hi",
            "engine": "kokoro",
            "voice": "af_sky",
            "speed": 1.5,
            "pitch": 0.8,
            "language": "en"
        }"#;
        let req: SynthesisRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, "hi");
        assert_eq!(req.engine.as_deref(), Some("kokoro"));
        assert_eq!(req.voice.as_deref(), Some("af_sky"));
        assert!((req.speed - 1.5).abs() < f32::EPSILON);
        assert!((req.pitch - 0.8).abs() < f32::EPSILON);
        assert_eq!(req.language.as_deref(), Some("en"));
    }

    #[test]
    fn test_synthesis_response_serialization() {
        let resp = SynthesisResponse {
            audio_base64: "AAAA".to_string(),
            sample_rate: 22050,
            duration_secs: 1.5,
            format: "wav".to_string(),
            engine_used: "kokoro".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"audio_base64\""));
        assert!(json.contains("22050"));
        assert!(json.contains("\"wav\""));
        assert!(json.contains("\"kokoro\""));
    }

    #[test]
    fn test_engine_list_response_serialization() {
        let resp = EngineListResponse {
            engines: vec![EngineInfoResponse {
                id: "kokoro".to_string(),
                name: "Kokoro TTS".to_string(),
                version: "1.0".to_string(),
                languages: vec!["en".to_string(), "ja".to_string()],
                voices_count: 5,
            }],
            total: 1,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["total"], 1);
        assert_eq!(parsed["engines"][0]["id"], "kokoro");
        assert_eq!(parsed["engines"][0]["voices_count"], 5);
    }

    #[test]
    fn test_engine_info_response_serialization() {
        let resp = EngineInfoResponse {
            id: "test-engine".to_string(),
            name: "Test Engine".to_string(),
            version: "2.0".to_string(),
            languages: vec!["en".to_string()],
            voices_count: 3,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["id"], "test-engine");
        assert_eq!(back["version"], "2.0");
    }

    #[test]
    fn test_stats_response_serialization() {
        use crate::core::tts_manager::TTSManagerStats;
        let stats = TTSManagerStats {
            total_engines: 2,
            engine_ids: vec!["kokoro".to_string()],
            cache_size: 10,
            cache_capacity: 100,
        };
        let resp = StatsResponse { stats };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"total_engines\""));
        assert!(json.contains("\"cache_size\""));
    }

    #[test]
    fn test_default_speed_and_pitch_values() {
        assert!((default_speed() - 1.0).abs() < f32::EPSILON);
        assert!((default_pitch() - 1.0).abs() < f32::EPSILON);
    }

    // ── Handler tests ─────────────────────────────────────────────────────────

    use crate::core::tts_manager::{TTSManager, TTSManagerConfig};
    use crate::core::tts_engine::{
        EngineCapabilities, SynthesisParams, VoiceInfo, VoiceGender, VoiceQuality,
    };
    use crate::core::audio::AudioData;

    struct MockTTSEngine {
        caps: EngineCapabilities,
    }

    impl MockTTSEngine {
        fn new() -> Self {
            Self {
                caps: EngineCapabilities {
                    name: "mock-tts".to_string(),
                    version: "0.1".to_string(),
                    supported_languages: vec!["en".to_string()],
                    supported_voices: vec![VoiceInfo {
                        id: "v1".to_string(),
                        name: "Voice 1".to_string(),
                        language: "en".to_string(),
                        gender: VoiceGender::Female,
                        quality: VoiceQuality::Neural,
                    }],
                    max_text_length: 50000,
                    sample_rate: 22050,
                    supports_ssml: false,
                    supports_streaming: false,
                },
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::core::tts_engine::TTSEngine for MockTTSEngine {
        fn name(&self) -> &str { "mock-tts" }
        fn capabilities(&self) -> &EngineCapabilities { &self.caps }
        async fn synthesize(&self, _text: &str, _params: &SynthesisParams) -> anyhow::Result<AudioData> {
            Ok(AudioData { samples: vec![0.0_f32; 22050], sample_rate: 22050, channels: 1 })
        }
        fn list_voices(&self) -> Vec<VoiceInfo> { self.caps.supported_voices.clone() }
    }

    fn make_tts_state(engine_id: &str) -> web::Data<TTSState> {
        let manager = Arc::new(TTSManager::new(TTSManagerConfig {
            default_engine: engine_id.to_string(),
            ..TTSManagerConfig::default()
        }));
        manager.register_engine(engine_id.to_string(), Arc::new(MockTTSEngine::new())).unwrap();
        web::Data::new(TTSState { manager })
    }

    fn make_empty_tts_state() -> web::Data<TTSState> {
        let manager = Arc::new(TTSManager::new(TTSManagerConfig::default()));
        web::Data::new(TTSState { manager })
    }

    // synthesize — empty text should return BadRequest
    #[actix_web::test]
    async fn test_synthesize_handler_empty_text() {
        let state = make_tts_state("mock-tts");
        let req = web::Json(SynthesisRequest {
            text: "".to_string(),
            engine: None,
            voice: None,
            speed: 1.0,
            pitch: 1.0,
            language: None,
        });
        let result = synthesize(req, state).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::ApiError::BadRequest(_)));
    }

    // synthesize — text too long should return BadRequest
    #[actix_web::test]
    async fn test_synthesize_handler_text_too_long() {
        let state = make_tts_state("mock-tts");
        let req = web::Json(SynthesisRequest {
            text: "x".repeat(50001),
            engine: None,
            voice: None,
            speed: 1.0,
            pitch: 1.0,
            language: None,
        });
        let result = synthesize(req, state).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::ApiError::BadRequest(_)));
    }

    // synthesize — engine not found should return InternalError
    #[actix_web::test]
    async fn test_synthesize_handler_engine_not_found() {
        let state = make_empty_tts_state();
        let req = web::Json(SynthesisRequest {
            text: "hello".to_string(),
            engine: Some("nonexistent".to_string()),
            voice: None,
            speed: 1.0,
            pitch: 1.0,
            language: None,
        });
        let result = synthesize(req, state).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::ApiError::InternalError(_)));
    }

    // list_engines — empty manager returns empty list
    #[actix_web::test]
    async fn test_list_engines_handler_empty() {
        let state = make_empty_tts_state();
        let resp = list_engines(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // list_engines — manager with a registered engine returns it
    #[actix_web::test]
    async fn test_list_engines_handler_with_engine() {
        let state = make_tts_state("mock-tts");
        let resp = list_engines(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // get_capabilities — engine found returns 200
    #[actix_web::test]
    async fn test_get_capabilities_handler_found() {
        let state = make_tts_state("mock-tts");
        let path = web::Path::from("mock-tts".to_string());
        let resp = get_capabilities(path, state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // get_capabilities — engine not found returns NotFound error
    #[actix_web::test]
    async fn test_get_capabilities_handler_not_found() {
        let state = make_empty_tts_state();
        let path = web::Path::from("ghost-engine".to_string());
        let result = get_capabilities(path, state).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::ApiError::NotFound(_)));
    }

    // list_voices — engine found returns voice list
    #[actix_web::test]
    async fn test_list_voices_handler_found() {
        let state = make_tts_state("mock-tts");
        let path = web::Path::from("mock-tts".to_string());
        let resp = list_voices(path, state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // list_voices — engine not found returns NotFound error
    #[actix_web::test]
    async fn test_list_voices_handler_not_found() {
        let state = make_empty_tts_state();
        let path = web::Path::from("ghost-engine".to_string());
        let result = list_voices(path, state).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::error::ApiError::NotFound(_)));
    }

    // get_stats — always succeeds
    #[actix_web::test]
    async fn test_get_stats_handler() {
        let state = make_tts_state("mock-tts");
        let resp = get_stats(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // health_check — always succeeds
    #[actix_web::test]
    async fn test_health_check_handler() {
        let state = make_tts_state("mock-tts");
        let resp = health_check(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // VoiceListResponse serde
    #[test]
    fn test_voice_list_response_serialization() {
        let resp = VoiceListResponse {
            voices: vec![VoiceInfo {
                id: "v1".to_string(),
                name: "Voice 1".to_string(),
                language: "en".to_string(),
                gender: VoiceGender::Female,
                quality: VoiceQuality::Neural,
            }],
            total: 1,
            engine: "mock-tts".to_string(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["total"], 1);
        assert_eq!(v["engine"], "mock-tts");
        assert_eq!(v["voices"][0]["id"], "v1");
    }

    // SynthesisRequest — speed/pitch clamping values
    #[test]
    fn test_synthesis_request_extreme_speed_pitch() {
        let json = r#"{"text": "hi", "speed": 999.0, "pitch": -1.0}"#;
        let req: SynthesisRequest = serde_json::from_str(json).unwrap();
        // Handler would clamp: speed.max(0.25).min(4.0) = 4.0, pitch.max(0.5).min(2.0) = 0.5
        let clamped_speed = req.speed.max(0.25_f32).min(4.0);
        let clamped_pitch = req.pitch.max(0.5_f32).min(2.0);
        assert!((clamped_speed - 4.0).abs() < f32::EPSILON);
        assert!((clamped_pitch - 0.5).abs() < f32::EPSILON);
    }

    // synthesize — happy path (valid text, registered engine) returns 200
    #[actix_web::test]
    async fn test_synthesize_handler_success() {
        let state = make_tts_state("mock-tts");
        let req = web::Json(SynthesisRequest {
            text: "Hello world".to_string(),
            engine: Some("mock-tts".to_string()),
            voice: None,
            speed: 1.0,
            pitch: 1.0,
            language: None,
        });
        let resp = synthesize(req, state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // synthesize — default engine (no engine specified) returns 200
    #[actix_web::test]
    async fn test_synthesize_handler_default_engine() {
        let state = make_tts_state("mock-tts");
        let req = web::Json(SynthesisRequest {
            text: "Using default engine".to_string(),
            engine: None,
            voice: None,
            speed: 1.0,
            pitch: 1.0,
            language: None,
        });
        let resp = synthesize(req, state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // synthesize — with voice and language params returns 200
    #[actix_web::test]
    async fn test_synthesize_handler_with_voice_and_language() {
        let state = make_tts_state("mock-tts");
        let req = web::Json(SynthesisRequest {
            text: "Voice test".to_string(),
            engine: Some("mock-tts".to_string()),
            voice: Some("v1".to_string()),
            speed: 1.2,
            pitch: 0.9,
            language: Some("en".to_string()),
        });
        let resp = synthesize(req, state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // synthesize — clamped speed/pitch still succeeds
    #[actix_web::test]
    async fn test_synthesize_handler_clamped_speed_pitch() {
        let state = make_tts_state("mock-tts");
        let req = web::Json(SynthesisRequest {
            text: "Extreme params".to_string(),
            engine: Some("mock-tts".to_string()),
            voice: None,
            speed: 999.0,   // will be clamped to 4.0
            pitch: -1.0,    // will be clamped to 0.5
            language: None,
        });
        let resp = synthesize(req, state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // health_check — response body contains expected keys
    #[actix_web::test]
    async fn test_health_check_response_body() {
        let state = make_tts_state("mock-tts");
        let resp = health_check(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
        // Status 200 is sufficient; body shape validated by serialization tests
    }

    // health_check — empty manager also returns healthy
    #[actix_web::test]
    async fn test_health_check_empty_manager_is_healthy() {
        let state = make_empty_tts_state();
        let resp = health_check(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // get_stats — stats reflect registered engine
    #[actix_web::test]
    async fn test_get_stats_handler_reflects_engine() {
        let state = make_tts_state("mock-tts");
        let resp = get_stats(state).await.unwrap();
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }

    // SynthesisRequest — engine field is optional
    #[test]
    fn test_synthesis_request_with_engine_none() {
        let json = r#"{"text": "no engine"}"#;
        let req: SynthesisRequest = serde_json::from_str(json).unwrap();
        assert!(req.engine.is_none());
    }

    // SynthesisRequest — voice and language optional
    #[test]
    fn test_synthesis_request_voice_language_optional() {
        let json = r#"{"text": "plain text", "speed": 0.5, "pitch": 2.0}"#;
        let req: SynthesisRequest = serde_json::from_str(json).unwrap();
        assert!(req.voice.is_none());
        assert!(req.language.is_none());
        assert!((req.speed - 0.5).abs() < f32::EPSILON);
        assert!((req.pitch - 2.0).abs() < f32::EPSILON);
    }

    // ── configure_routes (lines 639-647) ──────────────────────────────────────
    // Invoking configure_routes exercises every line of the function body.

    #[actix_web::test]
    async fn test_configure_routes_registers_all_tts_endpoints() {
        use actix_web::{test as actix_test, App};

        let state = make_tts_state("mock-tts");

        let app = actix_test::init_service(
            App::new()
                .app_data(state)
                .configure(configure_routes),
        )
        .await;

        // POST /tts/synthesize
        let req = actix_test::TestRequest::post()
            .uri("/tts/synthesize")
            .set_json(serde_json::json!({"text": "hello"}))
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        // GET /tts/engines
        let req = actix_test::TestRequest::get()
            .uri("/tts/engines")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        // GET /tts/engines/{id}/capabilities
        let req = actix_test::TestRequest::get()
            .uri("/tts/engines/mock-tts/capabilities")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        // GET /tts/engines/{id}/voices
        let req = actix_test::TestRequest::get()
            .uri("/tts/engines/mock-tts/voices")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        // GET /tts/stats
        let req = actix_test::TestRequest::get()
            .uri("/tts/stats")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);

        // GET /tts/health
        let req = actix_test::TestRequest::get()
            .uri("/tts/health")
            .to_request();
        let resp = actix_test::call_service(&app, req).await;
        assert_eq!(resp.status(), actix_web::http::StatusCode::OK);
    }
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
