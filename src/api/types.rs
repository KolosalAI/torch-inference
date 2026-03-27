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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_job_new() {
        let job = DownloadJob::new(
            "model-123".to_string(),
            "my-model".to_string(),
            "huggingface".to_string(),
        );
        assert_eq!(job.model_id, "model-123");
        assert_eq!(job.name, "my-model");
        assert_eq!(job.source, "huggingface");
        assert_eq!(job.status, "pending");
        assert!(!job.id.is_empty());
    }

    #[test]
    fn test_download_job_unique_ids() {
        let job1 = DownloadJob::new("m1".to_string(), "n1".to_string(), "s1".to_string());
        let job2 = DownloadJob::new("m2".to_string(), "n2".to_string(), "s2".to_string());
        assert_ne!(job1.id, job2.id);
    }

    #[test]
    fn test_inference_request_serde_roundtrip() {
        let req = InferenceRequest {
            model_name: "test-model".to_string(),
            inputs: serde_json::json!({"input": [1.0, 2.0, 3.0]}),
            priority: 1,
            timeout: Some(30.0),
        };
        let json = serde_json::to_string(&req).expect("serialization failed");
        let deserialized: InferenceRequest =
            serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(deserialized.model_name, "test-model");
        assert_eq!(deserialized.priority, 1);
        assert_eq!(deserialized.timeout, Some(30.0));
    }

    #[test]
    fn test_inference_request_no_timeout() {
        let req = InferenceRequest {
            model_name: "m".to_string(),
            inputs: serde_json::json!(null),
            priority: 0,
            timeout: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert!(deserialized.timeout.is_none());
    }

    #[test]
    fn test_inference_response_serde_roundtrip() {
        let resp = InferenceResponse {
            success: true,
            result: Some(serde_json::json!({"output": 42})),
            error: None,
            processing_time: Some(0.123),
            model_info: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: InferenceResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert!(deserialized.error.is_none());
        assert_eq!(deserialized.processing_time, Some(0.123));
    }

    #[test]
    fn test_health_check_response_serde_roundtrip() {
        let resp = HealthCheckResponse {
            healthy: true,
            checks: serde_json::json!({"db": "ok"}),
            timestamp: 1234567890.0,
            engine_stats: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: HealthCheckResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.healthy);
        assert_eq!(deserialized.timestamp, 1234567890.0);
    }

    #[test]
    fn test_tts_request_serde_roundtrip() {
        let req = TTSRequest {
            model_name: "kokoro".to_string(),
            text: "Hello world".to_string(),
            voice: Some("af_sky".to_string()),
            speed: 1.0,
            pitch: 1.0,
            volume: 1.0,
            language: "en-us".to_string(),
            emotion: None,
            output_format: "wav".to_string(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: TTSRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_name, "kokoro");
        assert_eq!(deserialized.text, "Hello world");
        assert_eq!(deserialized.voice, Some("af_sky".to_string()));
        assert_eq!(deserialized.output_format, "wav");
    }

    #[test]
    fn test_tts_response_serde_roundtrip() {
        let resp = TTSResponse {
            success: true,
            audio_data: Some("base64encodeddata".to_string()),
            audio_format: Some("wav".to_string()),
            duration: Some(2.5),
            sample_rate: Some(24000),
            processing_time: Some(0.5),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: TTSResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.sample_rate, Some(24000));
        assert!(deserialized.error.is_none());
    }

    #[test]
    fn test_stt_request_serde_roundtrip() {
        let req = STTRequest {
            model_name: "whisper".to_string(),
            audio_data: "base64audio".to_string(),
            language: "en".to_string(),
            enable_timestamps: true,
            beam_size: 5,
            temperature: 0.0,
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: STTRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_name, "whisper");
        assert!(deserialized.enable_timestamps);
        assert_eq!(deserialized.beam_size, 5);
    }

    #[test]
    fn test_stt_response_serde_roundtrip() {
        let resp = STTResponse {
            success: false,
            text: None,
            segments: None,
            language: None,
            confidence: None,
            processing_time: None,
            error: Some("model not loaded".to_string()),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: STTResponse = serde_json::from_str(&json).unwrap();
        assert!(!deserialized.success);
        assert_eq!(deserialized.error, Some("model not loaded".to_string()));
    }

    #[test]
    fn test_model_download_request_serde_roundtrip() {
        let req = ModelDownloadRequest {
            source: "huggingface".to_string(),
            model_id: "bert-base".to_string(),
            name: "bert".to_string(),
            task: "text-classification".to_string(),
            auto_convert_tts: false,
            include_vocoder: false,
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: ModelDownloadRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_id, "bert-base");
        assert!(!deserialized.auto_convert_tts);
    }

    #[test]
    fn test_model_download_response_serde_roundtrip() {
        let resp = ModelDownloadResponse {
            success: true,
            download_id: Some("dl-abc".to_string()),
            message: "Download started".to_string(),
            model_name: "bert".to_string(),
            status: "pending".to_string(),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: ModelDownloadResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.download_id, Some("dl-abc".to_string()));
        assert_eq!(deserialized.status, "pending");
    }

    #[test]
    fn test_image_process_request_serde_roundtrip() {
        let req = ImageProcessRequest {
            model_name: "security-model".to_string(),
            security_level: "high".to_string(),
            enable_sanitization: true,
            enable_adversarial_detection: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        let deserialized: ImageProcessRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.security_level, "high");
        assert!(deserialized.enable_sanitization);
    }

    #[test]
    fn test_image_process_response_serde_roundtrip() {
        let resp = ImageProcessResponse {
            success: true,
            processed_image: Some("imgdata".to_string()),
            threats_detected: vec!["adversarial".to_string()],
            processing_time: Some(0.1),
            error: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: ImageProcessResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.success);
        assert_eq!(deserialized.threats_detected.len(), 1);
        assert_eq!(deserialized.threats_detected[0], "adversarial");
    }
}
