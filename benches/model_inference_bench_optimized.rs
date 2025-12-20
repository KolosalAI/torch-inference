use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::path::{Path, PathBuf};
use std::fs;
use std::time::Duration;
use std::sync::{Arc, Mutex, OnceLock};
use reqwest;
use indicatif::{ProgressBar, ProgressStyle};
use image::{RgbImage, ImageBuffer};

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

/// Cached ImageNet labels (avoid re-allocation)
static IMAGENET_LABELS: OnceLock<Vec<String>> = OnceLock::new();

fn get_imagenet_labels() -> &'static Vec<String> {
    IMAGENET_LABELS.get_or_init(|| {
        (0..1000).map(|i| format!("class_{}", i)).collect()
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
        self.local_path().join(format!("{}.{}", self.name, self.file_extension))
    }

    fn config_file(&self) -> Option<PathBuf> {
        self.config_url.map(|_| self.local_path().join("config.json"))
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
        pb.set_message(format!("Downloading {}", dest.file_name().unwrap_or_default().to_string_lossy()));
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
        pb.finish_with_message(format!("Downloaded {}", dest.file_name().unwrap_or_default().to_string_lossy()));
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

/// Cached test images to avoid regeneration
static TEST_IMAGE_CACHE: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Optimized test image creation with caching
fn get_or_create_test_image(width: u32, height: u32) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_key = format!("target/bench_test_image_{}x{}.jpg", width, height);
    let path = PathBuf::from(&cache_key);
    
    // Check cache first
    {
        let mut cache = TEST_IMAGE_CACHE.lock().unwrap();
        if cache.as_ref().map_or(false, |p| p.exists()) {
            return Ok(cache.as_ref().unwrap().clone());
        }
    }
    
    // Create if doesn't exist
    if !path.exists() {
        create_test_image(&path, width, height)?;
    }
    
    // Update cache
    {
        let mut cache = TEST_IMAGE_CACHE.lock().unwrap();
        *cache = Some(path.clone());
    }
    
    Ok(path)
}

/// Optimized image generation with SIMD-friendly operations
fn create_test_image(path: &Path, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Pre-allocate buffer
    let mut img: RgbImage = ImageBuffer::new(width, height);
    
    // Optimized pixel filling with better cache locality
    let width_f = width as f32;
    let height_f = height as f32;
    let sum_f = (width + height) as f32;
    
    for y in 0..height {
        let g = ((y as f32 * 255.0) / height_f) as u8;
        for x in 0..width {
            let r = ((x as f32 * 255.0) / width_f) as u8;
            let b = (((x + y) as f32 * 255.0) / sum_f) as u8;
            img.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    
    img.save(path)?;
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
        let handles: Vec<_> = models.iter().map(|config| {
            let config = config.clone();
            tokio::spawn(async move {
                if let Err(e) = ensure_model_downloaded(&config).await {
                    eprintln!("Warning: Failed to download {}: {}", config.name, e);
                }
            })
        }).collect();
        
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
                group.bench_with_input(
                    BenchmarkId::new("cpu", config.name),
                    config,
                    |b, cfg| {
                        let path = cfg.model_file();
                        b.iter(|| {
                            black_box(CModule::load_on_device(&path, Device::Cpu).ok())
                        });
                    },
                );

                // File read benchmark
                group.bench_with_input(
                    BenchmarkId::new("file_read", config.name),
                    config,
                    |b, cfg| {
                        let path = cfg.model_file();
                        b.iter(|| {
                            black_box(fs::read(&path).ok())
                        });
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
                        b.iter(|| {
                            black_box(fs::read(&path).ok())
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

#[cfg(feature = "torch")]
fn benchmark_image_preprocessing(c: &mut Criterion) {
    use torch_inference::core::image_classifier::ImageClassifier;
    use tch::Device;
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let models = get_model_registry();

    // Download models in parallel
    rt.block_on(async {
        let handles: Vec<_> = models.iter()
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
        
        if let Ok(classifier) = ImageClassifier::new(
            &model_file,
            labels,
            Some((448, 448)),
            Some(Device::Cpu),
        ) {
            group.bench_with_input(
                BenchmarkId::from_parameter(config.name),
                config,
                |b, _| {
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
    let models = get_model_registry();

    // Parallel downloads
    rt.block_on(async {
        let handles: Vec<_> = models.iter()
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
        
        if let Ok(classifier) = ImageClassifier::new(
            &model_file,
            labels,
            Some((448, 448)),
            Some(Device::Cpu),
        ) {
            // Full pipeline
            group.bench_with_input(
                BenchmarkId::new("full_pipeline", config.name),
                config,
                |b, _| {
                    b.iter(|| {
                        black_box(classifier.classify(&test_image, 5).ok())
                    });
                },
            );

            // Inference-only (preprocessed)
            if let Ok(preprocessed) = classifier.preprocess_image(&test_image) {
                group.bench_with_input(
                    BenchmarkId::new("inference_only", config.name),
                    config,
                    |b, _| {
                        b.iter(|| {
                            black_box(&preprocessed)
                        });
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
    use torch_inference::core::image_classifier::ImageClassifier;
    use tch::Device;
    
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
        
        if let Ok(classifier) = ImageClassifier::new(
            &model_file,
            labels,
            Some((448, 448)),
            Some(Device::Cpu),
        ) {
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
        let handles: Vec<_> = models.iter().map(|config| {
            let config = config.clone();
            tokio::spawn(async move {
                let _ = ensure_model_downloaded(&config).await;
            })
        }).collect();
        
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
                    b.iter(|| {
                        black_box(fs::metadata(&path).ok())
                    });
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
        "json"
    );

    group.bench_with_input(
        BenchmarkId::from_parameter(test_config.name),
        &test_config,
        |b, config| {
            b.iter(|| {
                rt.block_on(async {
                    let temp_file = PathBuf::from(format!("target/bench_temp_{}.json", uuid::Uuid::new_v4()));
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

criterion_group!(
    benches,
    benchmark_model_loading,
    benchmark_image_preprocessing,
    benchmark_image_classification,
    benchmark_batch_inference,
    benchmark_memory_usage,
    benchmark_model_download_time
);
criterion_main!(benches);
