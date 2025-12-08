/// Model Registry API Module
/// Provides REST API endpoints for model management
use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub score: f32,
    pub rank: i32,
    pub size: String,
    pub url: String,
    pub architecture: String,
    pub voices: String,
    pub quality: String,
    pub status: String,
    pub note: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub version: String,
    pub updated: String,
    pub models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut models = HashMap::new();
        
        // Kokoro v1.0
        models.insert("kokoro-v1.0".to_string(), ModelInfo {
            name: "Kokoro v1.0".to_string(),
            score: 58.1,
            rank: 7,
            size: "312 MB".to_string(),
            url: "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth".to_string(),
            architecture: "StyleTTS2 + ISTFTNet".to_string(),
            voices: "54".to_string(),
            quality: "High".to_string(),
            status: "Downloaded".to_string(),
            note: None,
        });
        
        // Kokoro v0.19
        models.insert("kokoro-v0.19".to_string(), ModelInfo {
            name: "Kokoro v0.19".to_string(),
            score: 59.0,
            rank: 6,
            size: "312 MB".to_string(),
            url: "https://huggingface.co/hexgrad/kLegacy/resolve/main/v0.19/kokoro-v0_19.pth".to_string(),
            architecture: "StyleTTS2 + ISTFTNet".to_string(),
            voices: "10".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: None,
        });
        
        // Fish Speech v1.5
        models.insert("fish-speech".to_string(), ModelInfo {
            name: "Fish Speech v1.5".to_string(),
            score: 57.1,
            rank: 8,
            size: "~1 GB".to_string(),
            url: "https://huggingface.co/fishaudio/fish-speech-1.5".to_string(),
            architecture: "VQGAN + Llama".to_string(),
            voices: "Multi".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Requires git-lfs".to_string()),
        });
        
        // XTTS v2
        models.insert("xtts-v2".to_string(), ModelInfo {
            name: "XTTS v2".to_string(),
            score: 56.1,
            rank: 9,
            size: "~2 GB".to_string(),
            url: "https://huggingface.co/coqui/XTTS-v2".to_string(),
            architecture: "Transformer + HiFiGAN".to_string(),
            voices: "Voice Cloning".to_string(),
            quality: "High".to_string(),
            status: "Available".to_string(),
            note: Some("Coqui TTS".to_string()),
        });
        
        // StyleTTS 2
        models.insert("styletts2".to_string(), ModelInfo {
            name: "StyleTTS 2".to_string(),
            score: 49.0,
            rank: 12,
            size: "~500 MB".to_string(),
            url: "https://huggingface.co/yl4579/StyleTTS2-LJSpeech".to_string(),
            architecture: "StyleTTS2".to_string(),
            voices: "Single".to_string(),
            quality: "Medium-High".to_string(),
            status: "Available".to_string(),
            note: Some("Base model for Kokoro".to_string()),
        });
        
        // MetaVoice
        models.insert("metavoice".to_string(), ModelInfo {
            name: "MetaVoice".to_string(),
            score: 49.1,
            rank: 11,
            size: "~1 GB".to_string(),
            url: "https://huggingface.co/metavoiceio/metavoice-1B-v0.1".to_string(),
            architecture: "Transformer".to_string(),
            voices: "Multi".to_string(),
            quality: "Medium-High".to_string(),
            status: "Available".to_string(),
            note: None,
        });
        
        // OpenVoice
        models.insert("openvoice".to_string(), ModelInfo {
            name: "OpenVoice".to_string(),
            score: 43.1,
            rank: 14,
            size: "~600 MB".to_string(),
            url: "https://huggingface.co/myshell-ai/OpenVoice".to_string(),
            architecture: "VITS".to_string(),
            voices: "Voice Cloning".to_string(),
            quality: "Medium".to_string(),
            status: "Available".to_string(),
            note: None,
        });
        
        // MeloTTS
        models.insert("melotts".to_string(), ModelInfo {
            name: "MeloTTS".to_string(),
            score: 41.3,
            rank: 15,
            size: "~200 MB".to_string(),
            url: "https://huggingface.co/myshell-ai/MeloTTS-English".to_string(),
            architecture: "VITS".to_string(),
            voices: "Multi-language".to_string(),
            quality: "Medium".to_string(),
            status: "Available".to_string(),
            note: None,
        });
        
        // Windows SAPI
        models.insert("windows-sapi".to_string(), ModelInfo {
            name: "Windows SAPI".to_string(),
            score: 0.0,
            rank: 0,
            size: "Built-in".to_string(),
            url: "Built-in".to_string(),
            architecture: "Neural TTS".to_string(),
            voices: "3".to_string(),
            quality: "High".to_string(),
            status: "Active".to_string(),
            note: Some("Currently used for real speech".to_string()),
        });
        
        // Piper ONNX
        models.insert("piper-onnx".to_string(), ModelInfo {
            name: "Piper ONNX".to_string(),
            score: 0.0,
            rank: 0,
            size: "60 MB".to_string(),
            url: "https://huggingface.co/rhasspy/piper-voices".to_string(),
            architecture: "VITS (ONNX)".to_string(),
            voices: "1".to_string(),
            quality: "Medium".to_string(),
            status: "Downloaded".to_string(),
            note: None,
        });
        
        Self {
            version: "1.0".to_string(),
            updated: chrono::Utc::now().to_rfc3339(),
            models,
        }
    }
    
    pub fn get_model(&self, model_id: &str) -> Option<&ModelInfo> {
        self.models.get(model_id)
    }
    
    pub fn list_models(&self) -> Vec<(&String, &ModelInfo)> {
        let mut models: Vec<_> = self.models.iter().collect();
        models.sort_by_key(|(_, info)| info.rank);
        models
    }
    
    pub fn get_downloaded_models(&self) -> Vec<(&String, &ModelInfo)> {
        self.models.iter()
            .filter(|(_, info)| info.status == "Downloaded" || info.status == "Active")
            .collect()
    }
}

// API Endpoints

/// GET /api/models - List all models
pub async fn list_models() -> Result<HttpResponse> {
    let registry = ModelRegistry::new();
    Ok(HttpResponse::Ok().json(registry))
}

/// GET /api/models/{model_id} - Get specific model info
pub async fn get_model(path: web::Path<String>) -> Result<HttpResponse> {
    let model_id = path.into_inner();
    let registry = ModelRegistry::new();
    
    match registry.get_model(&model_id) {
        Some(model) => Ok(HttpResponse::Ok().json(model)),
        None => Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Model not found",
            "model_id": model_id
        })))
    }
}

/// GET /api/models/downloaded - List downloaded models
pub async fn list_downloaded_models() -> Result<HttpResponse> {
    let registry = ModelRegistry::new();
    let downloaded: HashMap<_, _> = registry.get_downloaded_models()
        .into_iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "count": downloaded.len(),
        "models": downloaded
    })))
}

#[derive(Debug, Deserialize)]
pub struct DownloadRequest {
    pub model_id: String,
}

/// POST /api/models/download - Download a model
pub async fn download_model(req: web::Json<DownloadRequest>) -> Result<HttpResponse> {
    let registry = ModelRegistry::new();
    
    match registry.get_model(&req.model_id) {
        Some(model) => {
            // Check if already downloaded
            if model.status == "Downloaded" || model.status == "Active" {
                return Ok(HttpResponse::Ok().json(serde_json::json!({
                    "status": "already_downloaded",
                    "model": model
                })));
            }
            
            // Return download information
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "status": "download_initiated",
                "model": model,
                "message": "Use PowerShell script: .\\Download-Models.ps1 -Model {}",
                "note": "Direct download from API not yet implemented"
            })))
        },
        None => Ok(HttpResponse::NotFound().json(serde_json::json!({
            "error": "Model not found",
            "model_id": req.model_id
        })))
    }
}

/// GET /api/models/comparison - Get model comparison
pub async fn get_model_comparison() -> Result<HttpResponse> {
    let registry = ModelRegistry::new();
    let models = registry.list_models();
    
    let comparison: Vec<_> = models.iter()
        .map(|(id, info)| {
            serde_json::json!({
                "id": id,
                "name": info.name,
                "score": info.score,
                "rank": info.rank,
                "size": info.size,
                "quality": info.quality,
                "status": info.status,
                "architecture": info.architecture
            })
        })
        .collect();
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "models": comparison,
        "total_count": comparison.len()
    })))
}

// Configure routes
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/models")
            .route("", web::get().to(list_models))
            .route("/downloaded", web::get().to(list_downloaded_models))
            .route("/comparison", web::get().to(get_model_comparison))
            .route("/download", web::post().to(download_model))
            .route("/{model_id}", web::get().to(get_model))
    );
}
