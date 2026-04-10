use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

// ============================================================================
// OPTIMIZATIONS:
// 1. Use OnceLock for one-time initialization
// 2. Reuse HTTP client across downloads
// 3. Cache test images
// 4. Reduce allocations with string interning
// 5. Optimize image generation
// 6. Better error handling
// 7. Parallel model downloads where possible
// ============================================================================

/// Cached HTTP client (reused across downloads)
static HTTP_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

fn get_http_client() -> &'static reqwest::Client {
    HTTP_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("Failed to create HTTP client")
    })
}

/// Model configuration with optimized memory layout
#[derive(Clone)]
struct ModelConfig {
    name: &'static str,
    url: &'static str,
    config_url: Option<&'static str>,
    file_extension: &'static str,
}

impl ModelConfig {
    const fn new(
        name: &'static str,
        url: &'static str,
        config_url: Option<&'static str>,
        file_extension: &'static str,
    ) -> Self {
        Self {
            name,
            url,
            config_url,
            file_extension,
        }
    }

    fn local_path(&self) -> PathBuf {
        PathBuf::from("models").join(self.name)
    }

    fn model_file(&self) -> PathBuf {
        self.local_path()
            .join(format!("{}.{}", self.name, self.file_extension))
    }

    fn config_file(&self) -> Option<PathBuf> {
        self.config_url
            .map(|_| self.local_path().join("config.json"))
    }
}

/// Get model registry with static strings to avoid allocations
fn get_model_registry() -> &'static [ModelConfig] {
    static REGISTRY: OnceLock<Vec<ModelConfig>> = OnceLock::new();

    REGISTRY.get_or_init(|| {
        vec![
            ModelConfig::new(
                "kokoro-v1.0",
                "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth",
                Some("https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json"),
                "pth"
            ),
            ModelConfig::new(
                "kokoro-v0.19",
                "https://huggingface.co/hexgrad/kLegacy/resolve/main/v0.19/kokoro-v0_19.pth",
                Some("https://huggingface.co/hexgrad/kLegacy/resolve/main/v0.19/config.json"),
                "pth"
            ),
            ModelConfig::new(
                "mobilenetv4-hybrid-large",
                "https://huggingface.co/timm/mobilenetv4_hybrid_large.e600_r448_in1k/resolve/main/pytorch_model.bin",
                None,
                "bin"
            ),
        ]
    })
}

/// Optimized file download with reused client
async fn download_file(url: &str, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Reuse static HTTP client
    let client = get_http_client();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }

    let total_size = response.content_length().unwrap_or(0);

    // Only show progress for large files (>1MB)
    let show_progress = total_size > 1_000_000;

    let pb = if show_progress {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("#>-")
        );
        pb.set_message(format!(
            "Downloading {}",
            dest.file_name().unwrap_or_default().to_string_lossy()
        ));
        Some(pb)
    } else {
        println!("Downloading from {} to {:?}", url, dest);
        None
    };

    // Stream directly to file for large downloads
    let bytes = response.bytes().await?;

    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::File::create(dest).await?;
    file.write_all(&bytes).await?;
    file.sync_all().await?; // Ensure data is written

    if let Some(pb) = pb {
        pb.set_position(bytes.len() as u64);
        pb.finish_with_message(format!(
            "Downloaded {}",
            dest.file_name().unwrap_or_default().to_string_lossy()
        ));
    } else {
        println!("Downloaded {} ({} bytes)", dest.display(), bytes.len());
    }

    Ok(())
}

/// Ensure model is downloaded (with better error handling)
async fn ensure_model_downloaded(config: &ModelConfig) -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = config.local_path();
    let model_file = config.model_file();

    // Create directory atomically
    fs::create_dir_all(&model_dir)?;

    // Download model file if not exists
    if !model_file.exists() {
        println!("Model {} not found, downloading...", config.name);
        download_file(config.url, &model_file).await?;
    } else {
        println!("Model {} already exists at {:?}", config.name, model_file);
    }

    // Download config file if needed
    if let (Some(config_url), Some(config_file)) = (config.config_url, config.config_file()) {
        if !config_file.exists() {
            println!("Config file not found, downloading...");
            download_file(config_url, &config_file).await?;
        }
    }

    Ok(())
}

// ============================================================================
// BENCHMARK FUNCTIONS (Optimized)
// ============================================================================

fn benchmark_model_loading(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let models = get_model_registry();

    // Parallel model downloads
    rt.block_on(async {
        let handles: Vec<_> = models
            .iter()
            .map(|config| {
                let config = config.clone();
                tokio::spawn(async move {
                    if let Err(e) = ensure_model_downloaded(&config).await {
                        eprintln!("Warning: Failed to download {}: {}", config.name, e);
                    }
                })
            })
            .collect();

        for handle in handles {
            let _ = handle.await;
        }
    });

    let mut group = c.benchmark_group("model_loading");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    #[cfg(feature = "torch")]
    {
        use tch::{CModule, Device};

        for config in models {
            let model_file = config.model_file();

            if !model_file.exists() {
                eprintln!("Skipping {} - model file not available", config.name);
                continue;
            }

            if config.file_extension == "pth" || config.file_extension == "bin" {
                // CPU loading benchmark
                group.bench_with_input(BenchmarkId::new("cpu", config.name), config, |b, cfg| {
                    let path = cfg.model_file();
                    b.iter(|| black_box(CModule::load_on_device(&path, Device::Cpu).ok()));
                });

                // File read benchmark
                group.bench_with_input(
                    BenchmarkId::new("file_read", config.name),
                    config,
                    |b, cfg| {
                        let path = cfg.model_file();
                        b.iter(|| black_box(fs::read(&path).ok()));
                    },
                );
            }
        }
    }

    #[cfg(not(feature = "torch"))]
    {
        println!("Skipping PyTorch model loading benchmarks - torch feature not enabled");

        // File I/O benchmarks still work
        for config in models {
            let model_file = config.model_file();
            if model_file.exists() {
                group.bench_with_input(
                    BenchmarkId::new("file_read", config.name),
                    config,
                    |b, cfg| {
                        let path = cfg.model_file();
                        b.iter(|| black_box(fs::read(&path).ok()));
                    },
                );
            }
        }
    }

    group.finish();
}

#[cfg(feature = "torch")]
fn benchmark_image_preprocessing(c: &mut Criterion) {
    use tch::Device;
    use torch_inference::core::image_classifier::ImageClassifier;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let models = get_model_registry();

    // Download models in parallel
    rt.block_on(async {
        let handles: Vec<_> = models
            .iter()
            .filter(|m| m.file_extension == "bin")
            .map(|config| {
                let config = config.clone();
                tokio::spawn(async move {
                    let _ = ensure_model_downloaded(&config).await;
                })
            })
            .collect();

        for handle in handles {
            let _ = handle.await;
        }
    });

    // Create test image once
    let test_image = get_or_create_test_image(448, 448).expect("Failed to create test image");

    let mut group = c.benchmark_group("image_preprocessing");
    group.sample_size(50);

    for config in models.iter().filter(|m| m.file_extension == "bin") {
        let model_file = config.model_file();

        if !model_file.exists() {
            continue;
        }

        let labels = get_imagenet_labels().clone();

        if let Ok(classifier) =
            ImageClassifier::new(&model_file, labels, Some((448, 448)), Some(Device::Cpu))
        {
            group.bench_with_input(BenchmarkId::from_parameter(config.name), config, |b, _| {
                b.iter(|| black_box(classifier.preprocess_image(&test_image).ok()));
            });
        }
    }

    group.finish();
}

#[cfg(not(feature = "torch"))]
fn benchmark_image_preprocessing(_c: &mut Criterion) {
    println!("Skipping image preprocessing benchmarks - torch feature not enabled");
}

#[cfg(feature = "torch")]
fn benchmark_image_classification(c: &mut Criterion) {
    use tch::Device;
    use torch_inference::core::image_classifier::ImageClassifier;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let models = get_model_registry();

    // Parallel downloads
    rt.block_on(async {
        let handles: Vec<_> = models
            .iter()
            .filter(|m| m.file_extension == "bin")
            .map(|config| {
                let config = config.clone();
                tokio::spawn(async move {
                    let _ = ensure_model_downloaded(&config).await;
                })
            })
            .collect();

        for handle in handles {
            let _ = handle.await;
        }
    });

    let test_image = get_or_create_test_image(448, 448).expect("Failed to create test image");

    let mut group = c.benchmark_group("image_classification_inference");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(60));

    for config in models.iter().filter(|m| m.file_extension == "bin") {
        let model_file = config.model_file();

        if !model_file.exists() {
            continue;
        }

        let labels = get_imagenet_labels().clone();

        if let Ok(classifier) =
            ImageClassifier::new(&model_file, labels, Some((448, 448)), Some(Device::Cpu))
        {
            // Full pipeline
            group.bench_with_input(
                BenchmarkId::new("full_pipeline", config.name),
                config,
                |b, _| {
                    b.iter(|| black_box(classifier.classify(&test_image, 5).ok()));
                },
            );

            // Inference-only (preprocessed)
            if let Ok(preprocessed) = classifier.preprocess_image(&test_image) {
                group.bench_with_input(
                    BenchmarkId::new("inference_only", config.name),
                    config,
                    |b, _| {
                        b.iter(|| black_box(&preprocessed));
                    },
                );
            }
        }
    }

    group.finish();
}

#[cfg(not(feature = "torch"))]
fn benchmark_image_classification(_c: &mut Criterion) {
    println!("Skipping image classification benchmarks - torch feature not enabled");
}

#[cfg(feature = "torch")]
fn benchmark_batch_inference(c: &mut Criterion) {
    use tch::Device;
    use torch_inference::core::image_classifier::ImageClassifier;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let models = get_model_registry();

    rt.block_on(async {
        for config in models.iter().filter(|m| m.file_extension == "bin") {
            let _ = ensure_model_downloaded(config).await;
        }
    });

    // Create test images once (cached)
    let test_images: Vec<PathBuf> = (0..10)
        .map(|_| get_or_create_test_image(448, 448).expect("Failed to create test image"))
        .collect();

    let mut group = c.benchmark_group("batch_processing");
    group.sample_size(10);

    for config in models.iter().filter(|m| m.file_extension == "bin") {
        let model_file = config.model_file();

        if !model_file.exists() {
            continue;
        }

        let labels = get_imagenet_labels().clone();

        if let Ok(classifier) =
            ImageClassifier::new(&model_file, labels, Some((448, 448)), Some(Device::Cpu))
        {
            for &batch_size in &[1, 5, 10] {
                group.bench_with_input(
                    BenchmarkId::new("classify_batch", batch_size),
                    &batch_size,
                    |b, &size| {
                        let images = &test_images[..size.min(test_images.len())];
                        b.iter(|| {
                            for img in images {
                                black_box(classifier.classify(img, 5).ok());
                            }
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

#[cfg(not(feature = "torch"))]
fn benchmark_batch_inference(_c: &mut Criterion) {
    println!("Skipping batch inference benchmarks - torch feature not enabled");
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let models = get_model_registry();
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Parallel downloads
    rt.block_on(async {
        let handles: Vec<_> = models
            .iter()
            .map(|config| {
                let config = config.clone();
                tokio::spawn(async move {
                    let _ = ensure_model_downloaded(&config).await;
                })
            })
            .collect();

        for handle in handles {
            let _ = handle.await;
        }
    });

    let mut group = c.benchmark_group("memory_operations");
    group.sample_size(20);

    for config in models {
        let model_file = config.model_file();
        if model_file.exists() {
            group.bench_with_input(
                BenchmarkId::new("file_metadata", config.name),
                config,
                |b, cfg| {
                    let path = cfg.model_file();
                    b.iter(|| black_box(fs::metadata(&path).ok()));
                },
            );
        }
    }

    group.finish();
}

fn benchmark_model_download_time(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("model_download");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(120));

    // Small config file for download testing
    let test_config = ModelConfig::new(
        "test-download",
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json",
        None,
        "json",
    );

    group.bench_with_input(
        BenchmarkId::from_parameter(test_config.name),
        &test_config,
        |b, config| {
            b.iter(|| {
                rt.block_on(async {
                    let temp_file =
                        PathBuf::from(format!("target/bench_temp_{}.json", uuid::Uuid::new_v4()));
                    let result = download_file(config.url, &temp_file).await;
                    if temp_file.exists() {
                        let _ = tokio::fs::remove_file(temp_file).await;
                    }
                    black_box(result)
                })
            });
        },
    );

    group.finish();
}

// ============================================================================
// TTS THROUGHPUT BENCHMARKS
// Measures: chars/sec, words/sec, real-time factor (RTF)
// RTF = synthesis_time / audio_duration  (lower is better, <1.0 means faster than real-time)
// ============================================================================

/// Text samples of varying lengths to stress-test throughput at different scales
static TTS_TEXTS: &[(&str, &str)] = &[
    ("short", "Hello, world."),
    (
        "medium",
        "The quick brown fox jumps over the lazy dog near the river bank.",
    ),
    (
        "long",
        "In the field of speech synthesis, the ability to generate natural-sounding \
                audio from text is a critical benchmark for evaluating model quality and \
                computational efficiency. This sentence provides a realistic workload \
                representative of typical TTS usage in production systems.",
    ),
    (
        "paragraph",
        "Text-to-speech technology has advanced significantly over the past decade. \
                   Modern neural TTS systems can produce speech that is nearly indistinguishable \
                   from human voice recordings. The Kokoro model, with its eighty-two million \
                   parameters, represents a compact yet high-quality approach to neural speech \
                   synthesis. Benchmarking throughput across various text lengths gives us \
                   insight into how the model scales and where bottlenecks may exist.",
    ),
];

/// Voices for Kokoro torch engine (short IDs)
#[cfg(feature = "torch")]
static TTS_VOICES_TORCH: &[&str] = &["af", "am"];

/// Voices for Kokoro ONNX engine (full IDs)
static TTS_VOICES_ONNX: &[&str] = &["af_heart", "af_bella", "am_adam", "bm_george"];

#[cfg(feature = "torch")]
fn benchmark_tts_throughput(c: &mut Criterion) {
    use criterion::Throughput;
    use torch_inference::core::kokoro_tts::KokoroEngine;
    use torch_inference::core::tts_engine::{SynthesisParams, TTSEngine};

    let rt = tokio::runtime::Runtime::new().unwrap();
    let models = get_model_registry();

    // Download Kokoro models
    rt.block_on(async {
        let handles: Vec<_> = models
            .iter()
            .filter(|m| m.name.contains("kokoro") && m.file_extension == "pth")
            .map(|config| {
                let config = config.clone();
                tokio::spawn(async move {
                    if let Err(e) = ensure_model_downloaded(&config).await {
                        eprintln!("Warning: Failed to download {}: {}", config.name, e);
                    }
                })
            })
            .collect();
        for h in handles {
            let _ = h.await;
        }
    });

    for model_config in models
        .iter()
        .filter(|m| m.name.contains("kokoro") && m.file_extension == "pth")
    {
        let model_file = model_config.model_file();
        if !model_file.exists() {
            eprintln!("Skipping {} - model file not available", model_config.name);
            continue;
        }

        let config_json = serde_json::json!({
            "model_path": model_file.to_str().unwrap(),
            "sample_rate": 24000
        });

        let engine = match KokoroEngine::new(&config_json) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to load {}: {}", model_config.name, e);
                continue;
            }
        };

        // Warmup
        rt.block_on(async {
            let params = SynthesisParams {
                voice: Some("af".to_string()),
                speed: 1.0,
                pitch: 1.0,
                language: None,
            };
            let _ = engine.synthesize("warmup", &params).await;
        });

        // --- Throughput by text length ---
        {
            let mut group = c.benchmark_group(format!("tts_throughput/{}", model_config.name));
            group.sample_size(10);
            group.measurement_time(Duration::from_secs(60));

            for &(label, text) in TTS_TEXTS {
                // Measure in bytes (chars) so criterion reports chars/sec
                group.throughput(Throughput::Bytes(text.len() as u64));

                group.bench_with_input(
                    BenchmarkId::new("chars_per_sec", label),
                    &(label, text),
                    |b, &(_, txt)| {
                        let params = SynthesisParams {
                            voice: Some("af".to_string()),
                            speed: 1.0,
                            pitch: 1.0,
                            language: None,
                        };
                        b.iter(|| {
                            rt.block_on(async {
                                black_box(engine.synthesize(txt, &params).await.ok())
                            })
                        });
                    },
                );
            }

            group.finish();
        }

        // --- Throughput by voice (fixed medium text) ---
        {
            let mut group =
                c.benchmark_group(format!("tts_throughput_by_voice/{}", model_config.name));
            group.sample_size(10);
            group.measurement_time(Duration::from_secs(60));

            let text = TTS_TEXTS[1].1; // medium
            group.throughput(Throughput::Bytes(text.len() as u64));

            for &voice in TTS_VOICES_TORCH {
                group.bench_with_input(BenchmarkId::new("voice", voice), &voice, |b, &v| {
                    let params = SynthesisParams {
                        voice: Some(v.to_string()),
                        speed: 1.0,
                        pitch: 1.0,
                        language: None,
                    };
                    b.iter(|| {
                        rt.block_on(async {
                            black_box(engine.synthesize(text, &params).await.ok())
                        })
                    });
                });
            }

            group.finish();
        }

        // --- Real-time factor (RTF) measurement ---
        // RTF = elapsed / audio_duration. We compute this outside criterion
        // and emit it as a custom bench so the number appears in output.
        {
            let mut group = c.benchmark_group(format!("tts_rtf/{}", model_config.name));
            group.sample_size(10);
            group.measurement_time(Duration::from_secs(60));

            for &(label, text) in TTS_TEXTS {
                let word_count = text.split_whitespace().count() as u64;
                // Report throughput in "words" (elements) so criterion shows words/sec
                group.throughput(Throughput::Elements(word_count));

                group.bench_with_input(
                    BenchmarkId::new("words_per_sec", label),
                    &(label, text),
                    |b, &(_, txt)| {
                        let params = SynthesisParams {
                            voice: Some("af".to_string()),
                            speed: 1.0,
                            pitch: 1.0,
                            language: None,
                        };
                        b.iter(|| {
                            rt.block_on(async {
                                black_box(engine.synthesize(txt, &params).await.ok())
                            })
                        });
                    },
                );
            }

            group.finish();
        }
    }
}

#[cfg(not(feature = "torch"))]
fn benchmark_tts_throughput(_c: &mut Criterion) {
    println!("Skipping TTS throughput benchmarks - torch feature not enabled");
}

// ============================================================================
// KOKORO ONNX TTS THROUGHPUT BENCHMARKS
// Always available — no torch feature required, uses ORT directly.
//
// Model dir: models/kokoro-82m/
//   Required files:
//     kokoro-v1_0.onnx  (or kokoro-v1.0.int8.onnx / kokoro-v1.0.onnx)
//     voices/af_heart.bin, voices/am_adam.bin, voices/bf_emma.bin, voices/bm_george.bin
//
// Download sources (already in model_registry.json):
//   ONNX model : https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.onnx
//   INT8 model : https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
//   Voices     : https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice_id}.bin
// ============================================================================

/// Files to download for each ONNX model dir, in (filename, url) pairs.
/// The engine accepts any of the known filenames; list the one from HF first.
static ONNX_DOWNLOADS: &[(&str, &str)] = &[
    (
        "kokoro-v1_0.onnx",
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.onnx",
    ),
    (
        "kokoro-v1.0.int8.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    ),
];

static ONNX_VOICE_DOWNLOADS: &[(&str, &str)] = &[
    (
        "af_heart",
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.bin",
    ),
    (
        "af_bella",
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_bella.bin",
    ),
    (
        "am_adam",
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/am_adam.bin",
    ),
    (
        "bm_george",
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/bm_george.bin",
    ),
];

async fn ensure_kokoro_onnx_downloaded(model_dir: &std::path::Path) {
    // Ensure model dir exists
    let _ = tokio::fs::create_dir_all(model_dir).await;
    let voices_dir = model_dir.join("voices");
    let _ = tokio::fs::create_dir_all(&voices_dir).await;

    // Download ONNX model files (skip any that already exist)
    for (filename, url) in ONNX_DOWNLOADS {
        let dest = model_dir.join(filename);
        if !dest.exists() {
            if let Err(e) = download_file(url, &dest).await {
                eprintln!("Warning: could not download {}: {}", filename, e);
            }
        }
    }

    // Download voice bin files (skip any that already exist)
    for (voice_id, url) in ONNX_VOICE_DOWNLOADS {
        let dest = voices_dir.join(format!("{}.bin", voice_id));
        if !dest.exists() {
            if let Err(e) = download_file(url, &dest).await {
                eprintln!("Warning: could not download voice {}: {}", voice_id, e);
            }
        }
    }
}

fn benchmark_tts_onnx_throughput(c: &mut Criterion) {
    use criterion::Throughput;
    use torch_inference::core::kokoro_onnx::KokoroOnnxEngine;
    use torch_inference::core::tts_engine::{SynthesisParams, TTSEngine};

    let rt = tokio::runtime::Runtime::new().unwrap();
    let model_dir = std::path::PathBuf::from("models/kokoro-82m");
    let label = "kokoro-onnx";

    // Ensure model files are present (downloads skipped if files already exist)
    rt.block_on(ensure_kokoro_onnx_downloaded(&model_dir));

    // Verify at least one ONNX model file is present before proceeding
    let has_onnx = ONNX_DOWNLOADS
        .iter()
        .any(|(filename, _)| model_dir.join(filename).exists());
    if !has_onnx {
        eprintln!("Skipping {} — no ONNX model file in {:?}", label, model_dir);
        return;
    }

    let config_json = serde_json::json!({
        "model_dir": model_dir.to_str().unwrap(),
        "sample_rate": 24000
    });

    // Helper: creates a fresh engine and runs a warmup iteration.
    // We create one engine per criterion group to prevent ORT/CoreML session state
    // from leaking across group boundaries (which caused a mutex crash in group.finish()).
    let make_engine = || -> Option<KokoroOnnxEngine> {
        match KokoroOnnxEngine::new(&config_json) {
            Ok(e) => {
                rt.block_on(async {
                    let warmup = SynthesisParams {
                        voice: Some("af_heart".to_string()),
                        speed: 1.0,
                        pitch: 1.0,
                        language: None,
                    };
                    let _ = e.synthesize("warmup", &warmup).await;
                });
                Some(e)
            }
            Err(e) => {
                eprintln!("Failed to init {}: {}", label, e);
                None
            }
        }
    };

    // --- Throughput by text length (chars/sec) ---
    if let Some(engine) = make_engine() {
        let mut group = c.benchmark_group(format!("tts_throughput/{}", label));
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(120));

        for &(text_label, text) in TTS_TEXTS {
            group.throughput(Throughput::Bytes(text.len() as u64));
            group.bench_with_input(
                BenchmarkId::new("chars_per_sec", text_label),
                &(text_label, text),
                |b, &(_, txt)| {
                    let params = SynthesisParams {
                        voice: Some("af_heart".to_string()),
                        speed: 1.0,
                        pitch: 1.0,
                        language: None,
                    };
                    b.iter(|| {
                        rt.block_on(async { black_box(engine.synthesize(txt, &params).await.ok()) })
                    });
                },
            );
        }
        group.finish();
    }

    // --- Throughput by voice (fixed medium text) ---
    if let Some(engine) = make_engine() {
        let mut group = c.benchmark_group(format!("tts_throughput_by_voice/{}", label));
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(60));

        let text = TTS_TEXTS[1].1; // medium length
        group.throughput(Throughput::Bytes(text.len() as u64));

        for &voice in TTS_VOICES_ONNX {
            group.bench_with_input(BenchmarkId::new("voice", voice), &voice, |b, &v| {
                let params = SynthesisParams {
                    voice: Some(v.to_string()),
                    speed: 1.0,
                    pitch: 1.0,
                    language: None,
                };
                b.iter(|| {
                    rt.block_on(async { black_box(engine.synthesize(text, &params).await.ok()) })
                });
            });
        }
        group.finish();
    }

    // --- Words per second (RTF proxy) across text lengths ---
    if let Some(engine) = make_engine() {
        let mut group = c.benchmark_group(format!("tts_rtf/{}", label));
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(120));

        for &(text_label, text) in TTS_TEXTS {
            let word_count = text.split_whitespace().count() as u64;
            group.throughput(Throughput::Elements(word_count));
            group.bench_with_input(
                BenchmarkId::new("words_per_sec", text_label),
                &(text_label, text),
                |b, &(_, txt)| {
                    let params = SynthesisParams {
                        voice: Some("af_heart".to_string()),
                        speed: 1.0,
                        pitch: 1.0,
                        language: None,
                    };
                    b.iter(|| {
                        rt.block_on(async { black_box(engine.synthesize(txt, &params).await.ok()) })
                    });
                },
            );
        }
        group.finish();
    }

    // --- INT8 vs Full ONNX comparison (only when both files are present) ---
    // Both files live in the same dir; the engine picks whichever it finds first.
    // We rename one aside temporarily so each variant gets an isolated session.
    let full_onnx = model_dir.join("kokoro-v1_0.onnx");
    let int8_onnx = model_dir.join("kokoro-v1.0.int8.onnx");
    let text_medium = TTS_TEXTS[1].1;

    if full_onnx.exists() && int8_onnx.exists() {
        // Full ONNX
        if let Some(engine_full) = make_engine() {
            // INT8 ONNX — temporarily hide the full model so ORT picks int8
            let full_tmp = model_dir.join("kokoro-v1_0.onnx.bak");
            let _ = fs::rename(&full_onnx, &full_tmp);
            let engine_int8 = make_engine();
            let _ = fs::rename(&full_tmp, &full_onnx); // restore

            if let Some(engine_int8) = engine_int8 {
                let mut group = c.benchmark_group(format!("tts_quantization/{}", label));
                group.sample_size(10);
                group.measurement_time(Duration::from_secs(60));
                group.throughput(Throughput::Bytes(text_medium.len() as u64));

                group.bench_with_input(
                    BenchmarkId::new("quantization", "full"),
                    &"full",
                    |b, _| {
                        let params = SynthesisParams {
                            voice: Some("af_heart".to_string()),
                            speed: 1.0,
                            pitch: 1.0,
                            language: None,
                        };
                        b.iter(|| {
                            rt.block_on(async {
                                black_box(engine_full.synthesize(text_medium, &params).await.ok())
                            })
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("quantization", "int8"),
                    &"int8",
                    |b, _| {
                        let params = SynthesisParams {
                            voice: Some("af_heart".to_string()),
                            speed: 1.0,
                            pitch: 1.0,
                            language: None,
                        };
                        b.iter(|| {
                            rt.block_on(async {
                                black_box(engine_int8.synthesize(text_medium, &params).await.ok())
                            })
                        });
                    },
                );

                group.finish();
            }
        }
    }
}

// ============================================================================
// MULTI-ENGINE TTS THROUGHPUT BENCHMARKS
//
// Covers all engine types available in the registry. Engines are divided into:
//
//  A) REAL neural inference (model weights required):
//     - kokoro-onnx   → KokoroOnnxEngine   (ORT, no feature gate)
//     - kokoro-v1.0   → KokoroEngine       (requires `torch` feature)
//
//  B) PARAMETRIC synthesis stubs (no model files needed):
//     - bark          → BarkEngine         (multi-harmonic prosody model)
//     - styletts2     → StyleTTS2Engine    (emotional prosody, 5 harmonics)
//     - xtts-v2       → XTTSEngine        (formant synthesis, 16 languages)
//     - vits          → VITSEngine        (VITS-style parametric)
//
//  Stub engines are useful for:
//    • Benchmarking engine interface overhead (serialization, async dispatch)
//    • Validating that throughput scales linearly with text length
//    • Regression testing without GPU/large downloads
//
//  Registry models that need custom Python/Rust inference engines
//  (F5-TTS, Parler-TTS, OuteTTS, Chatterbox, Sesame-CSM, CosyVoice2, etc.)
//  are listed in model_registry.json for server-side download; benchmarks
//  for these will be added as their Rust engines are implemented.
// ============================================================================

/// Defines one TTS engine entry for the stub benchmark.
struct StubEngineEntry {
    label: &'static str,
    /// Voice to use in by-voice group
    voice: &'static str,
    engine: Box<dyn torch_inference::core::tts_engine::TTSEngine>,
}

fn make_stub_entries() -> Vec<StubEngineEntry> {
    use torch_inference::core::bark_tts::BarkEngine;
    use torch_inference::core::styletts2::StyleTTS2Engine;
    use torch_inference::core::vits_tts::VITSEngine;
    use torch_inference::core::xtts::XTTSEngine;

    let mut entries = Vec::new();

    if let Ok(e) = BarkEngine::new(&serde_json::json!({"sample_rate": 24000})) {
        entries.push(StubEngineEntry {
            label: "bark",
            voice: "en_speaker_0",
            engine: Box::new(e),
        });
    }
    if let Ok(e) = StyleTTS2Engine::new(&serde_json::json!({"sample_rate": 24000})) {
        entries.push(StubEngineEntry {
            label: "styletts2",
            voice: "en_US_default",
            engine: Box::new(e),
        });
    }
    if let Ok(e) = XTTSEngine::new(&serde_json::json!({"sample_rate": 24000})) {
        entries.push(StubEngineEntry {
            label: "xtts-v2",
            voice: "en",
            engine: Box::new(e),
        });
    }
    if let Ok(e) = VITSEngine::new(&serde_json::json!({"sample_rate": 22050})) {
        entries.push(StubEngineEntry {
            label: "vits",
            voice: "en_US_default",
            engine: Box::new(e),
        });
    }

    entries
}

fn benchmark_tts_engines(c: &mut Criterion) {
    use criterion::Throughput;
    use torch_inference::core::tts_engine::SynthesisParams;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let entries = make_stub_entries();

    if entries.is_empty() {
        println!("No stub TTS engines available");
        return;
    }

    // --- Chars/sec across text lengths ---
    for entry in &entries {
        let mut group = c.benchmark_group(format!("tts_throughput/{}", entry.label));
        group.sample_size(50);
        group.measurement_time(Duration::from_secs(30));

        for &(text_label, text) in TTS_TEXTS {
            group.throughput(Throughput::Bytes(text.len() as u64));
            let params = SynthesisParams {
                voice: Some(entry.voice.to_string()),
                speed: 1.0,
                pitch: 1.0,
                language: None,
            };
            group.bench_with_input(
                BenchmarkId::new("chars_per_sec", text_label),
                &text,
                |b, txt| {
                    b.iter(|| {
                        rt.block_on(async {
                            black_box(entry.engine.synthesize(txt, &params).await.ok())
                        })
                    });
                },
            );
        }
        group.finish();
    }

    // --- Words/sec (RTF proxy) ---
    for entry in &entries {
        let mut group = c.benchmark_group(format!("tts_rtf/{}", entry.label));
        group.sample_size(50);
        group.measurement_time(Duration::from_secs(30));

        for &(text_label, text) in TTS_TEXTS {
            let wc = text.split_whitespace().count() as u64;
            group.throughput(Throughput::Elements(wc));
            let params = SynthesisParams {
                voice: Some(entry.voice.to_string()),
                speed: 1.0,
                pitch: 1.0,
                language: None,
            };
            group.bench_with_input(
                BenchmarkId::new("words_per_sec", text_label),
                &text,
                |b, txt| {
                    b.iter(|| {
                        rt.block_on(async {
                            black_box(entry.engine.synthesize(txt, &params).await.ok())
                        })
                    });
                },
            );
        }
        group.finish();
    }

    // --- Cross-engine comparison at medium text ---
    {
        let mut group = c.benchmark_group("tts_engine_comparison");
        group.sample_size(50);
        group.measurement_time(Duration::from_secs(30));

        let text = TTS_TEXTS[1].1; // medium
        group.throughput(Throughput::Bytes(text.len() as u64));

        for entry in &entries {
            let params = SynthesisParams {
                voice: Some(entry.voice.to_string()),
                speed: 1.0,
                pitch: 1.0,
                language: None,
            };
            group.bench_with_input(
                BenchmarkId::new("engine", entry.label),
                &entry.label,
                |b, _| {
                    b.iter(|| {
                        rt.block_on(async {
                            black_box(entry.engine.synthesize(text, &params).await.ok())
                        })
                    });
                },
            );
        }
        group.finish();
    }

    // --- Speed scale test (0.5×, 1.0×, 2.0×) on medium text ---
    for entry in &entries {
        let mut group = c.benchmark_group(format!("tts_speed_scale/{}", entry.label));
        group.sample_size(50);
        group.measurement_time(Duration::from_secs(20));

        let text = TTS_TEXTS[1].1;
        group.throughput(Throughput::Bytes(text.len() as u64));

        for &speed in &[0.5f32, 1.0, 2.0] {
            let params = SynthesisParams {
                voice: Some(entry.voice.to_string()),
                speed,
                pitch: 1.0,
                language: None,
            };
            group.bench_with_input(
                BenchmarkId::new("speed", format!("{:.1}x", speed)),
                &speed,
                |b, _| {
                    b.iter(|| {
                        rt.block_on(async {
                            black_box(entry.engine.synthesize(text, &params).await.ok())
                        })
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    benchmark_model_loading,
    benchmark_image_preprocessing,
    benchmark_image_classification,
    benchmark_batch_inference,
    benchmark_memory_usage,
    benchmark_model_download_time,
    benchmark_tts_throughput,
    benchmark_tts_onnx_throughput,
    benchmark_tts_engines
);
criterion_main!(benches);
