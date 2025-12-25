use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Duration;
use std::sync::{Arc, Mutex};
use reqwest;
use indicatif::{ProgressBar, ProgressStyle};
use image::{RgbImage, ImageBuffer};
use serde_json::json;
use tokio::runtime::Runtime;

mod benchmark_reporter;
use benchmark_reporter::{BenchmarkReporter, BenchmarkResult, SystemInfo};

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
    let file = match std::fs::File::open("model_registry.json") {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = std::io::BufReader::new(file);
    let registry: serde_json::Value = match serde_json::from_reader(reader) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    
    let mut models = Vec::new();
    
    if let Some(models_map) = registry.get("models").and_then(|m| m.as_object()) {
        for (key, value) in models_map {
            let url = value.get("url").and_then(|v| v.as_str()).unwrap_or("");
            
            // Skip built-in, invalid URLs, and TTS models
            if url == "Built-in" || url.is_empty() || key.contains("kokoro") || key.contains("tts") {
                continue;
            }
            
            let name = key.clone();
            let config_url = value.get("config_url").and_then(|v| v.as_str());
            let task = value.get("task").and_then(|v| v.as_str());
            
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
    
    if !model_dir.exists() {
        fs::create_dir_all(model_dir)?;
    }

    if !model_file.exists() {
        println!("Model {} not found, downloading...", config.name);
        download_file(&config.url, &model_file).await?;
    } else {
        println!("Model {} already exists at {:?}", config.name, model_file);
    }

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

fn record_result(bench_name: &str, model_name: &str, param: &str, elapsed: Duration, iterations: u64) {
    let nanos = elapsed.as_nanos() as f64 / iterations as f64;
    
    let result = BenchmarkResult {
        timestamp: chrono::Utc::now().to_rfc3339(),
        benchmark_name: bench_name.to_string(),
        model_name: Some(model_name.to_string()),
        parameter: Some(param.to_string()),
        mean_time_ns: nanos,
        mean_time_ms: nanos / 1_000_000.0,
        std_dev_ns: 0.0,
        median_ns: nanos,
        min_ns: nanos,
        max_ns: nanos,
        sample_count: 10,
        iterations,
        throughput_ops_per_sec: Some(1_000_000_000.0 / nanos),
        system_info: SystemInfo::new(),
    };
    
    if let Ok(mut reporter) = REPORTER.lock() {
        reporter.add_result(result);
    }
}

// Batch sizes to test: powers of 2 from 1 to 1024
fn get_batch_sizes() -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut size = 1;
    while size <= 1024 {
        sizes.push(size);
        size *= 2;
    }
    sizes
}

// Concurrent request counts: powers of 2 from 1 to 1024
fn get_concurrent_counts() -> Vec<usize> {
    get_batch_sizes() // Same pattern
}

#[cfg(feature = "torch")]
fn benchmark_batch_inference_scaling(c: &mut Criterion) {
    use torch_inference::core::image_classifier::ImageClassifier;
    use tch::Device;
    
    let rt = Runtime::new().unwrap();
    let all_models = get_model_registry();
    let models: Vec<&ModelConfig> = all_models.iter()
        .filter(|m| m.task.as_deref() == Some("Image Classification"))
        .take(2) // Test with first 2 models to keep benchmark time reasonable
        .collect();

    if models.is_empty() {
        println!("No image classification models found");
        return;
    }

    // Download models
    for model_config in &models {
        rt.block_on(async {
            if let Err(e) = ensure_model_downloaded(model_config).await {
                eprintln!("Warning: Failed to download {}: {}", model_config.name, e);
            }
        });
    }

    // Create test images
    let test_images: Vec<PathBuf> = (0..1024).map(|i| {
        let path = PathBuf::from(format!("target/bench_batch_image_{}.jpg", i));
        if !path.exists() {
            let _ = create_test_image(&path, 224, 224);
        }
        path
    }).collect();

    let mut group = c.benchmark_group("batch_inference_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

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
            Some((224, 224)),
            Some(Device::Cpu),
        ) {
            for batch_size in get_batch_sizes() {
                let param = format!("batch_{}", batch_size);
                
                group.bench_with_input(
                    BenchmarkId::new(&model_config.name, &param),
                    &batch_size,
                    |b, &size| {
                        let images = &test_images[..size.min(test_images.len())];
                        b.iter_custom(|iters| {
                            let start = std::time::Instant::now();
                            for _ in 0..iters {
                                for img in images {
                                    black_box(classifier.classify(img, 5).ok());
                                }
                            }
                            let elapsed = start.elapsed();
                            
                            record_result(
                                "batch_inference_scaling",
                                &model_config.name,
                                &param,
                                elapsed,
                                iters * size as u64
                            );
                            
                            elapsed
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

#[cfg(not(feature = "torch"))]
fn benchmark_batch_inference_scaling(_c: &mut Criterion) {
    println!("Skipping batch inference scaling benchmarks - torch feature not enabled");
}

#[cfg(feature = "torch")]
fn benchmark_concurrent_inference_scaling(c: &mut Criterion) {
    use torch_inference::core::image_classifier::ImageClassifier;
    use tch::Device;
    use std::sync::Arc;
    
    let rt = Runtime::new().unwrap();
    let all_models = get_model_registry();
    let models: Vec<&ModelConfig> = all_models.iter()
        .filter(|m| m.task.as_deref() == Some("Image Classification"))
        .take(2)
        .collect();

    if models.is_empty() {
        println!("No image classification models found");
        return;
    }

    for model_config in &models {
        rt.block_on(async {
            if let Err(e) = ensure_model_downloaded(model_config).await {
                eprintln!("Warning: Failed to download {}: {}", model_config.name, e);
            }
        });
    }

    let test_image = PathBuf::from("target/bench_concurrent_image.jpg");
    if !test_image.exists() {
        let _ = create_test_image(&test_image, 224, 224);
    }

    let mut group = c.benchmark_group("concurrent_inference_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for model_config in models {
        let model_file = model_config.model_file();
        
        if !model_file.exists() {
            eprintln!("Skipping {} - model file not available", model_config.name);
            continue;
        }

        let labels = load_imagenet_labels();
        
        if let Ok(classifier) = ImageClassifier::new(
            &model_file,
            labels.clone(),
            Some((224, 224)),
            Some(Device::Cpu),
        ) {
            let classifier = Arc::new(classifier);
            
            for concurrent_count in get_concurrent_counts() {
                let param = format!("concurrent_{}", concurrent_count);
                
                group.bench_with_input(
                    BenchmarkId::new(&model_config.name, &param),
                    &concurrent_count,
                    |b, &count| {
                        let test_img = test_image.clone();
                        let classifier_clone = Arc::clone(&classifier);
                        
                        b.iter_custom(|iters| {
                            let start = std::time::Instant::now();
                            
                            for _ in 0..iters {
                                rt.block_on(async {
                                    let mut handles = Vec::new();
                                    
                                    for _ in 0..count {
                                        let img = test_img.clone();
                                        let clf = Arc::clone(&classifier_clone);
                                        
                                        let handle = tokio::spawn(async move {
                                            black_box(clf.classify(&img, 5).ok())
                                        });
                                        
                                        handles.push(handle);
                                    }
                                    
                                    for handle in handles {
                                        let _ = handle.await;
                                    }
                                });
                            }
                            
                            let elapsed = start.elapsed();
                            
                            record_result(
                                "concurrent_inference_scaling",
                                &model_config.name,
                                &param,
                                elapsed,
                                iters * count as u64
                            );
                            
                            elapsed
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

#[cfg(not(feature = "torch"))]
fn benchmark_concurrent_inference_scaling(_c: &mut Criterion) {
    println!("Skipping concurrent inference scaling benchmarks - torch feature not enabled");
}

fn save_reports(_: &mut Criterion) {
    if let Ok(reporter) = REPORTER.lock() {
        reporter.print_summary();
        if let Err(e) = reporter.save_all("batch_concurrent_benchmark") {
            eprintln!("Failed to save benchmark reports: {}", e);
        }
    }
}

criterion_group!(
    benches,
    benchmark_batch_inference_scaling,
    benchmark_concurrent_inference_scaling,
    save_reports
);
criterion_main!(benches);
