use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model_name: String,
    pub inputs: serde_json::Value,
    pub priority: i32,
    pub timeout: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub success: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub processing_time: Option<f64>,
    pub model_info: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub healthy: bool,
    pub checks: serde_json::Value,
    pub timestamp: f64,
    pub engine_stats: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSRequest {
    pub model_name: String,
    pub text: String,
    pub voice: Option<String>,
    pub speed: f32,
    pub pitch: f32,
    pub volume: f32,
    pub language: String,
    pub emotion: Option<String>,
    pub output_format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSResponse {
    pub success: bool,
    pub audio_data: Option<String>,
    pub audio_format: Option<String>,
    pub duration: Option<f64>,
    pub sample_rate: Option<u32>,
    pub processing_time: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STTRequest {
    pub model_name: String,
    pub audio_data: String,
    pub language: String,
    pub enable_timestamps: bool,
    pub beam_size: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STTResponse {
    pub success: bool,
    pub text: Option<String>,
    pub segments: Option<Vec<serde_json::Value>>,
    pub language: Option<String>,
    pub confidence: Option<f32>,
    pub processing_time: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDownloadRequest {
    pub source: String,
    pub model_id: String,
    pub name: String,
    pub task: String,
    pub auto_convert_tts: bool,
    pub include_vocoder: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDownloadResponse {
    pub success: bool,
    pub download_id: Option<String>,
    pub message: String,
    pub model_name: String,
    pub status: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessRequest {
    pub model_name: String,
    pub security_level: String,
    pub enable_sanitization: bool,
    pub enable_adversarial_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessResponse {
    pub success: bool,
    pub processed_image: Option<String>,
    pub threats_detected: Vec<String>,
    pub processing_time: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DownloadJob {
    pub id: String,
    pub model_id: String,
    pub name: String,
    pub source: String,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl DownloadJob {
    pub fn new(model_id: String, name: String, source: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            model_id,
            name,
            source,
            status: "pending".to_string(),
            created_at: chrono::Utc::now(),
        }
    }
}
