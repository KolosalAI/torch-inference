// Core infrastructure
pub mod engine;
pub mod gpu;
pub mod audio;
pub mod audio_models;
pub mod image_security;

// Python bridge for neural TTS (fallback)
pub mod python_tts_bridge;

// Native Rust TTS components
pub mod phoneme_converter;
pub mod g2p_misaki;
pub mod styletts2_model;
pub mod istftnet_vocoder;

// Streaming TTS pipeline (sentence-level parallelism → low TTFA)
pub mod tts_pipeline;

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

// SIMD-fused image preprocessing pipeline (decode → resize → normalize)
pub mod image_pipeline;

// CPU core-affinity utilities for P-core worker pinning
pub mod affinity;

// Neural network and ML modules
pub mod neural_network;
pub mod image_classifier;
pub mod yolo;  // YOLO object detection (v5, v8, v10, v11, v12)

// PyTorch auto-detection
pub mod torch_autodetect;

// LLM inference subsystem (PagedAttention + continuous batching + speculative decoding)
#[cfg(feature = "llm")]
pub mod llm;
// Always compile the LLM subsystem in test/dev builds so tests run without feature flags.
#[cfg(not(feature = "llm"))]
pub mod llm;
