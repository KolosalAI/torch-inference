use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use image::{ImageBuffer, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone)]
struct ModelConfig {
    name: String,
    url: String,
    config_url: Option<String>,
    local_path: String,
    file_extension: String,
    task: Option<String>,
}

impl ModelConfig {
    fn new(
        name: &str,
        url: &str,
        config_url: Option<&str>,
        file_extension: &str,
        task: Option<&str>,
    ) -> Self {
        Self {
            name: name.to_string(),
            url: url.to_string(),
            config_url: config_url.map(|s| s.to_string()),
            local_path: format!("models/{}", name),
            file_extension: file_extension.to_string(),
            task: task.map(|s| s.to_string()),
        }
    }

    fn model_file(&self) -> PathBuf {
        Path::new(&self.local_path).join(format!("{}.{}", self.name, self.file_extension))
    }

    fn config_file(&self) -> Option<PathBuf> {
        self.config_url
            .as_ref()
            .map(|_| Path::new(&self.local_path).join("config.json"))
    }
}

fn get_model_registry() -> Vec<ModelConfig> {
    let file =
        std::fs::File::open("model_registry.json").expect("Failed to open model_registry.json");
    let reader = std::io::BufReader::new(file);
    let registry: serde_json::Value =
        serde_json::from_reader(reader).expect("Failed to parse model_registry.json");

    let mut models = Vec::new();

    if let Some(models_map) = registry.get("models").and_then(|m| m.as_object()) {
        for (key, value) in models_map {
            let url = value.get("url").and_then(|v| v.as_str()).unwrap_or("");

            // Skip built-in or invalid URLs
            if url == "Built-in" || url.is_empty() {
                continue;
            }

            let name = key.clone();
            let config_url = value.get("config_url").and_then(|v| v.as_str());
            let task = value.get("task").and_then(|v| v.as_str());

            // Determine extension from URL
            let file_extension = if url.ends_with(".pth") {
                "pth"
            } else if url.ends_with(".bin") {
                "bin"
            } else if url.ends_with(".onnx") {
                "onnx"
            } else {
                // Fallback based on name/type
                if name.contains("kokoro") {
                    "pth"
                } else {
                    "bin"
                }
            };

            models.push(ModelConfig::new(
                &name,
                url,
                config_url,
                file_extension,
                task,
            ));
        }
    }

    // Sort for consistent benchmark order
    models.sort_by(|a, b| a.name.cmp(&b.name));

    models
}

async fn download_file(url: &str, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("Downloading from {} to {:?}", url, dest);

    let client = reqwest::Client::new();
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
        .unwrap()
        .progress_chars("#>-"));
    pb.set_message(format!(
        "Downloading {}",
        dest.file_name().unwrap().to_string_lossy()
    ));

    let mut file = tokio::fs::File::create(dest).await?;
    let bytes = response.bytes().await?;

    use tokio::io::AsyncWriteExt;
    file.write_all(&bytes).await?;
    pb.set_position(bytes.len() as u64);

    pb.finish_with_message(format!(
        "Downloaded {}",
        dest.file_name().unwrap().to_string_lossy()
    ));
    Ok(())
}

async fn ensure_model_downloaded(config: &ModelConfig) -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = Path::new(&config.local_path);
    let model_file = config.model_file();

    // Create directory if needed
    if !model_dir.exists() {
        fs::create_dir_all(model_dir)?;
    }

    // Download model file if not exists
    if !model_file.exists() {
        println!("Model {} not found, downloading...", config.name);
        download_file(&config.url, &model_file).await?;
    } else {
        println!("Model {} already exists at {:?}", config.name, model_file);
    }

    // Download config file if needed
    if let Some(config_url) = &config.config_url {
        if let Some(config_file) = config.config_file() {
            if !config_file.exists() {
                println!("Config file not found, downloading...");
                download_file(config_url, &config_file).await?;
            }
        }
    }

    Ok(())
}

fn load_imagenet_labels() -> Vec<String> {
    // Return a subset of ImageNet labels for testing
    // In production, load from synset_labels.txt
    (0..1000).map(|i| format!("class_{}", i)).collect()
}

fn create_test_image(
    path: &Path,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create parent directory if needed
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Create a simple test image with gradient pattern
    let mut img: RgbImage = ImageBuffer::new(width, height);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (x * 255 / width) as u8;
        let g = (y * 255 / height) as u8;
        let b = ((x + y) * 255 / (width + height)) as u8;
        *pixel = image::Rgb([r, g, b]);
    }

    img.save(path)?;
    Ok(())
}

#[cfg(feature = "torch")]
fn benchmark_image_preprocessing(c: &mut Criterion) {
    use tch::Device;
    use torch_inference::core::image_classifier::ImageClassifier;

    let rt = tokio::runtime::Runtime::new().unwrap();

    let all_models = get_model_registry();
    let models: Vec<&ModelConfig> = all_models
        .iter()
        .filter(|m| m.task.as_deref() == Some("Image Classification"))
        .collect();

    // Download and create test image
    for model_config in &models {
        rt.block_on(async {
            if let Err(e) = ensure_model_downloaded(model_config).await {
                eprintln!("Warning: Failed to download {}: {}", model_config.name, e);
            }
        });
    }

    let test_image = PathBuf::from("target/bench_test_image.jpg");
    if !test_image.exists() {
        let _ = create_test_image(&test_image, 448, 448);
    }

    let mut group = c.benchmark_group("image_preprocessing");
    group.sample_size(50);

    for model_config in models {
        let model_file = model_config.model_file();

        if !model_file.exists() {
            eprintln!("Skipping {} - model file not available", model_config.name);
            continue;
        }

        let labels = load_imagenet_labels();

        if let Ok(classifier) =
            ImageClassifier::new(&model_file, labels, Some((448, 448)), Some(Device::Cpu))
        {
            group.bench_with_input(
                BenchmarkId::from_parameter(&model_config.name),
                model_config,
                |b, _config| {
                    b.iter(|| black_box(classifier.preprocess_image(&test_image).ok()));
                },
            );
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

    let all_models = get_model_registry();
    let models: Vec<&ModelConfig> = all_models
        .iter()
        .filter(|m| m.task.as_deref() == Some("Image Classification"))
        .collect();

    // Download models if needed
    for model_config in &models {
        rt.block_on(async {
            if let Err(e) = ensure_model_downloaded(model_config).await {
                eprintln!("Warning: Failed to download {}: {}", model_config.name, e);
            }
        });
    }

    let test_image = PathBuf::from("target/bench_test_image.jpg");
    if !test_image.exists() {
        let _ = create_test_image(&test_image, 448, 448);
    }

    let mut group = c.benchmark_group("image_classification_inference");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(60));

    for model_config in models {
        let model_file = model_config.model_file();

        if !model_file.exists() {
            eprintln!("Skipping {} - model file not available", model_config.name);
            continue;
        }

        let labels = load_imagenet_labels();

        match ImageClassifier::new(&model_file, labels, Some((448, 448)), Some(Device::Cpu)) {
            Ok(classifier) => {
                // Full pipeline benchmark (preprocessing + inference)
                group.bench_with_input(
                    BenchmarkId::new("full_pipeline", &model_config.name),
                    model_config,
                    |b, _config| {
                        b.iter(|| black_box(classifier.classify(&test_image, 5).ok()));
                    },
                );

                // Inference-only benchmark (preprocessed input)
                if let Ok(preprocessed) = classifier.preprocess_image(&test_image) {
                    group.bench_with_input(
                        BenchmarkId::new("inference_only", &model_config.name),
                        model_config,
                        |b, _config| {
                            b.iter(|| {
                                // Simulate inference without preprocessing
                                black_box(&preprocessed)
                            });
                        },
                    );
                }
            }
            Err(e) => {
                eprintln!("Failed to load classifier {}: {}", model_config.name, e);
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
fn benchmark_tts_inference(c: &mut Criterion) {
    use torch_inference::core::kokoro_tts::KokoroEngine;
    use torch_inference::core::tts_engine::{SynthesisParams, TTSEngine};

    let rt = tokio::runtime::Runtime::new().unwrap();
    let models = get_model_registry();

    // Filter for Kokoro models
    let kokoro_models: Vec<&ModelConfig> = models
        .iter()
        .filter(|m| m.name.contains("kokoro") && m.file_extension == "pth")
        .collect();

    for model_config in &kokoro_models {
        rt.block_on(async {
            if let Err(e) = ensure_model_downloaded(model_config).await {
                eprintln!("Warning: Failed to download {}: {}", model_config.name, e);
            }
        });
    }

    let mut group = c.benchmark_group("tts_inference");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    for model_config in kokoro_models {
        let model_file = model_config.model_file();

        if !model_file.exists() {
            continue;
        }

        let config_json = json!({
            "model_path": model_file.to_str().unwrap(),
            "sample_rate": 24000
        });

        if let Ok(engine) = KokoroEngine::new(&config_json) {
            let params = SynthesisParams {
                voice: Some("af".to_string()),
                speed: 1.0,
                pitch: 1.0,
                language: None,
            };

            let text = "Hello, this is a benchmark test for Kokoro TTS.";

            group.bench_with_input(
                BenchmarkId::new("synthesize", &model_config.name),
                &model_config,
                |b, _config| {
                    b.iter(|| {
                        rt.block_on(async {
                            black_box(engine.synthesize(text, &params).await.ok())
                        })
                    });
                },
            );
        }
    }
    group.finish();
}

#[cfg(not(feature = "torch"))]
fn benchmark_tts_inference(_c: &mut Criterion) {
    println!("Skipping TTS benchmarks - torch feature not enabled");
}

fn benchmark_model_loading(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let models = get_model_registry();

    // Download all models first
    for model_config in &models {
        rt.block_on(async {
            if let Err(e) = ensure_model_downloaded(model_config).await {
                eprintln!("Warning: Failed to download {}: {}", model_config.name, e);
            }
        });
    }

    let mut group = c.benchmark_group("model_loading");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    #[cfg(feature = "torch")]
    {
        use tch::{CModule, Device};

        for model_config in &models {
            let model_file = model_config.model_file();

            if !model_file.exists() {
                eprintln!("Skipping {} - model file not available", model_config.name);
                continue;
            }

            if model_config.file_extension == "pth" || model_config.file_extension == "bin" {
                // Benchmark loading to CPU
                group.bench_with_input(
                    BenchmarkId::new("cpu", &model_config.name),
                    &model_config,
                    |b, config| {
                        b.iter(|| {
                            black_box(
                                CModule::load_on_device(&config.model_file(), Device::Cpu).ok(),
                            )
                        });
                    },
                );

                // Also benchmark the file reading time
                group.bench_with_input(
                    BenchmarkId::new("file_read", &model_config.name),
                    &model_config,
                    |b, config| {
                        b.iter(|| black_box(std::fs::read(&config.model_file()).ok()));
                    },
                );
            }
        }
    }

    #[cfg(not(feature = "torch"))]
    {
        println!("Skipping PyTorch model loading benchmarks - torch feature not enabled");

        // Still benchmark file I/O
        for model_config in &models {
            let model_file = model_config.model_file();

            if !model_file.exists() {
                continue;
            }

            group.bench_with_input(
                BenchmarkId::new("file_read", &model_config.name),
                &model_config,
                |b, config| {
                    b.iter(|| black_box(std::fs::read(&config.model_file()).ok()));
                },
            );
        }
    }

    group.finish();
}

fn benchmark_batch_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    group.sample_size(10);

    #[cfg(feature = "torch")]
    {
        use tch::Device;
        use torch_inference::core::image_classifier::ImageClassifier;

        let rt = tokio::runtime::Runtime::new().unwrap();

        let model_config = ModelConfig::new(
            "mobilenetv4-hybrid-large",
            "https://huggingface.co/timm/mobilenetv4_hybrid_large.e600_r448_in1k/resolve/main/pytorch_model.bin",
            None,
            "bin",
            Some("Image Classification")
        );

        rt.block_on(async {
            let _ = ensure_model_downloaded(&model_config).await;
        });

        // Create multiple test images
        let test_images: Vec<PathBuf> = (0..10)
            .map(|i| {
                let path = PathBuf::from(format!("target/bench_test_image_{}.jpg", i));
                if !path.exists() {
                    let _ = create_test_image(&path, 448, 448);
                }
                path
            })
            .collect();

        let model_file = model_config.model_file();
        if model_file.exists() {
            let labels = load_imagenet_labels();

            if let Ok(classifier) =
                ImageClassifier::new(&model_file, labels, Some((448, 448)), Some(Device::Cpu))
            {
                // Benchmark different batch sizes
                for batch_size in [1, 5, 10] {
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
    }

    #[cfg(not(feature = "torch"))]
    {
        println!("Skipping batch inference benchmarks - torch feature not enabled");
    }

    group.finish();
}

fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    group.sample_size(20);

    // Benchmark model size estimation
    let models = get_model_registry();
    let rt = tokio::runtime::Runtime::new().unwrap();

    for model_config in &models {
        rt.block_on(async {
            let _ = ensure_model_downloaded(model_config).await;
        });

        let model_file = model_config.model_file();
        if model_file.exists() {
            group.bench_with_input(
                BenchmarkId::new("file_metadata", &model_config.name),
                &model_config,
                |b, config| {
                    b.iter(|| black_box(std::fs::metadata(&config.model_file()).ok()));
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

    // Only benchmark small models for download
    let small_models = vec![ModelConfig::new(
        "test-download",
        "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json",
        None,
        "json",
        None,
    )];

    for model_config in &small_models {
        group.bench_with_input(
            BenchmarkId::from_parameter(&model_config.name),
            &model_config,
            |b, config| {
                b.iter(|| {
                    rt.block_on(async {
                        let temp_file = PathBuf::from(format!(
                            "target/bench_temp_{}.json",
                            uuid::Uuid::new_v4()
                        ));
                        let result = download_file(&config.url, &temp_file).await;
                        if temp_file.exists() {
                            let _ = tokio::fs::remove_file(temp_file).await;
                        }
                        black_box(result)
                    })
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_model_loading,
    benchmark_image_preprocessing,
    benchmark_image_classification,
    benchmark_tts_inference,
    benchmark_batch_inference,
    benchmark_memory_usage,
    benchmark_model_download_time
);
criterion_main!(benches);
