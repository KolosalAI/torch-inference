// Core infrastructure
pub mod engine;
pub mod gpu;
pub mod audio;
pub mod audio_models;
pub mod image_security;

// Production TTS engines
pub mod tts_engine;
pub mod tts_manager;
pub mod piper_tts;
pub mod windows_sapi_tts;

// Neural network and ML modules
pub mod neural_network;
pub mod image_classifier;

// PyTorch auto-detection
pub mod torch_autodetect;
