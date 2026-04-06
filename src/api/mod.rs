pub mod audio;
pub mod dashboard;
pub mod handlers;
pub mod health;
pub mod image;
pub mod logging;
pub mod metrics_endpoint;
pub mod model_download;
pub mod models;
pub mod performance;
pub mod registry;
pub mod system;
pub mod tts;
pub mod types;

// ML inference modules
pub mod classification;
pub mod classify; // batched image classification (ImagePipeline + ORT backend)
pub mod inference;
pub mod llm; // OpenAI-compatible LLM completions API
pub mod yolo; // YOLO object detection
