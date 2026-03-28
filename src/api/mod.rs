pub mod handlers;
pub mod types;
pub mod audio;
pub mod image;
pub mod model_download;
pub mod system;
pub mod logging;
pub mod performance;
pub mod tts;
pub mod registry;
pub mod models;
pub mod health;
pub mod metrics_endpoint;

// ML inference modules
pub mod classification;
pub mod classify;  // batched image classification (ImagePipeline + ORT backend)
pub mod llm;       // OpenAI-compatible LLM completions API
pub mod inference;
pub mod yolo;  // YOLO object detection
