// Core infrastructure
pub mod engine;
pub mod gpu;
pub mod audio;
pub mod audio_models;
pub mod image_security;

// CUDA and TensorRT optimization
pub mod cuda_optimizer;

// Python bridge for neural TTS (fallback)
pub mod python_tts_bridge;

// Native Rust TTS components
pub mod phoneme_converter;
pub mod g2p_misaki;
pub mod styletts2_model;
pub mod istftnet_vocoder;

// Production TTS engines
pub mod tts_engine;
pub mod tts_manager;
pub mod kokoro_tts;
pub mod kokoro_onnx;
pub mod piper_tts;
pub mod windows_sapi_tts;
pub mod vits_tts;
pub mod styletts2;
pub mod bark_tts;
pub mod xtts;

// Speech-to-Text (STT) engine
pub mod whisper_stt;

// Neural network and ML modules
pub mod neural_network;
pub mod image_classifier;
pub mod yolo;  // YOLO object detection (v5, v8, v10, v11, v12)

// PyTorch auto-detection
pub mod torch_autodetect;
