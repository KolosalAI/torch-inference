use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Duration;
use std::sync::{Arc, Mutex};
use reqwest;
use indicatif::{ProgressBar, ProgressStyle};
use image::{RgbImage, ImageBuffer};
use serde_json::json;

mod benchmark_reporter;
use benchmark_reporter::{BenchmarkReporter, BenchmarkResult};

lazy_static::lazy_static! {
    static ref REPORTER: Arc<Mutex<BenchmarkReporter>> = Arc::new(Mutex::new(
        BenchmarkReporter::new("benches/data")
    ));
}

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
    fn new(name: &str, url: &str, config_url: Option<&str>, file_extension: &str, task: Option<&str>) -> Self {
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
        self.config_url.as_ref().map(|_| {
            Path::new(&self.local_path).join("config.json")
        })
    }
}

fn get_model_registry() -> Vec<ModelConfig> {
    let file = std::fs::File::open("model_registry.json").expect("Failed to open model_registry.json");
    let reader = std::io::BufReader::new(file);
    let registry: serde_json::Value = serde_json::from_reader(reader).expect("Failed to parse model_registry.json");
    
    let mut models = Vec::new();
    
    if let Some(models_map) = registry.get("models").and_then(|m| m.as_object()) {
        for (key, value) in models_map {
            let url = value.get("url").and_then(|v| v.as_str()).unwrap_or("");
            
            // Skip built-in or invalid URLs, and skip TTS models
            if url == "Built-in" || url.is_empty() || key.contains("kokoro") {
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
                "bin"
            };

            models.push(ModelConfig::new(
                &name,
                url,
                config_url,
                file_extension,
                task
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
    pb.set_message(format!("Downloading {}", dest.file_name().unwrap().to_string_lossy()));

    let mut file = tokio::fs::File::create(dest).await?;
    let bytes = response.bytes().await?;
    
    use tokio::io::AsyncWriteExt;
    file.write_all(&bytes).await?;
    pb.set_position(bytes.len() as u64);

    pb.finish_with_message(format!("Downloaded {}", dest.file_name().unwrap().to_string_lossy()));
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
    (0..1000).map(|i| format!("class_{}", i)).collect()
}

fn create_test_image(path: &Path, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

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
    use torch_inference::core::image_classifier::ImageClassifier;
    use tch::Device;
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let all_models = get_model_registry();
    let models: Vec<&ModelConfig> = all_models.iter()
        .filter(|m| m.task.as_deref() == Some("Image Classification"))
        .collect();

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
        
        if let Ok(classifier) = ImageClassifier::new(
            &model_file,
            labels,
            Some((448, 448)),
            Some(Device::Cpu),
        ) {
            let bench_id = BenchmarkId::from_parameter(&model_config.name);
            group.bench_with_input(
                bench_id,
                model_config,
                |b, _config| {
                    b.iter(|| {
                        black_box(classifier.preprocess_image(&test_image).ok())
                    });
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
    use torch_inference::core::image_classifier::ImageClassifier;
    use tch::Device;
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let all_models = get_model_registry();
    let models: Vec<&ModelConfig> = all_models.iter()
        .filter(|m| m.task.as_deref() == Some("Image Classification"))
        .collect();

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
        
        match ImageClassifier::new(
            &model_file,
            labels,
            Some((448, 448)),
            Some(Device::Cpu),
        ) {
            Ok(classifier) => {
                group.bench_with_input(
                    BenchmarkId::new("full_pipeline", &model_config.name),
                    model_config,
                    |b, _config| {
                        b.iter(|| {
                            black_box(classifier.classify(&test_image, 5).ok())
                        });
                    },
                );

                if let Ok(preprocessed) = classifier.preprocess_image(&test_image) {
                    group.bench_with_input(
                        BenchmarkId::new("inference_only", &model_config.name),
                        model_config,
                        |b, _config| {
                            b.iter(|| {
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

fn benchmark_model_loading(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let models = get_model_registry();

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
                group.bench_with_input(
                    BenchmarkId::new("cpu", &model_config.name),
                    &model_config,
                    |b, config| {
                        b.iter(|| {
                            black_box(CModule::load_on_device(&config.model_file(), Device::Cpu).ok())
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("file_read", &model_config.name),
                    &model_config,
                    |b, config| {
                        b.iter(|| {
                            black_box(std::fs::read(&config.model_file()).ok())
                        });
                    },
                );
            }
        }
    }

    #[cfg(not(feature = "torch"))]
    {
        println!("Skipping PyTorch model loading benchmarks - torch feature not enabled");
        
        for model_config in &models {
            let model_file = model_config.model_file();
            
            if !model_file.exists() {
                continue;
            }

            group.bench_with_input(
                BenchmarkId::new("file_read", &model_config.name),
                &model_config,
                |b, config| {
                    b.iter(|| {
                        black_box(std::fs::read(&config.model_file()).ok())
                    });
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
        use torch_inference::core::image_classifier::ImageClassifier;
        use tch::Device;
        
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

        let test_images: Vec<PathBuf> = (0..10).map(|i| {
            let path = PathBuf::from(format!("target/bench_test_image_{}.jpg", i));
            if !path.exists() {
                let _ = create_test_image(&path, 448, 448);
            }
            path
        }).collect();

        let model_file = model_config.model_file();
        if model_file.exists() {
            let labels = load_imagenet_labels();
            
            if let Ok(classifier) = ImageClassifier::new(
                &model_file,
                labels,
                Some((448, 448)),
                Some(Device::Cpu),
            ) {
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
                    b.iter(|| {
                        black_box(std::fs::metadata(&config.model_file()).ok())
                    });
                },
            );
        }
    }

    group.finish();
}

fn save_reports(_: &mut Criterion) {
    if let Ok(reporter) = REPORTER.lock() {
        reporter.print_summary();
        if let Err(e) = reporter.save_all("model_inference_benchmark") {
            eprintln!("Failed to save benchmark reports: {}", e);
        }
    }
}

criterion_group!(
    benches,
    benchmark_model_loading,
    benchmark_image_preprocessing,
    benchmark_image_classification,
    benchmark_batch_inference,
    benchmark_memory_usage,
    save_reports
);
criterion_main!(benches);
