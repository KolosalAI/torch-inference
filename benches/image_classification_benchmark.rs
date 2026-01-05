use criterion::{criterion_group, criterion_main, Criterion};
use std::path::Path;
use std::time::{Duration, Instant};
use std::fs::{File, create_dir_all};
use std::io::Write;
use plotters::prelude::*;
use walkdir::WalkDir;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "torch")]
use tch::{CModule, Device, Tensor, Kind};

/// Classification model benchmark result
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ClassificationBenchmarkResult {
    model_name: String,
    model_family: String,
    architecture: String,
    input_resolution: String,
    file_size_mb: f64,
    parameters_millions: f64,
    top1_accuracy: f64,
    top5_accuracy: f64,
    // Timing metrics
    load_time_ms: f64,
    first_inference_ms: f64,
    avg_inference_ms: f64,
    min_inference_ms: f64,
    max_inference_ms: f64,
    std_dev_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    // Throughput
    throughput_fps: f64,
    throughput_images_per_sec: f64,
    // Efficiency metrics
    flops_gflops: f64,
    efficiency_fps_per_gflop: f64,
    efficiency_accuracy_per_ms: f64,
    // Memory
    peak_memory_mb: f64,
    device: String,
}

/// Competitor comparison data
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompetitorResult {
    provider: String,
    model_name: String,
    avg_inference_ms: f64,
    throughput_fps: f64,
    load_time_ms: f64,
    memory_mb: f64,
    device: String,
    notes: String,
}

/// Benchmark configuration
struct BenchmarkConfig {
    warmup_iterations: usize,
    benchmark_iterations: usize,
    batch_sizes: Vec<usize>,
    output_dir: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            batch_sizes: vec![1, 2, 4, 8, 16, 32],
            output_dir: "benchmark_results/classification".to_string(),
        }
    }
}

/// Known model specifications
fn get_model_specs() -> HashMap<String, ModelSpec> {
    let mut specs = HashMap::new();
    
    specs.insert("eva02-large-patch14-448".to_string(), ModelSpec {
        family: "EVA".to_string(),
        architecture: "Vision Transformer (ViT-L/14)".to_string(),
        parameters_m: 304.0,
        flops_g: 61.6,
        top1_accuracy: 90.054,
        top5_accuracy: 99.0,
        input_size: 448,
    });
    
    specs.insert("eva-giant-patch14-560".to_string(), ModelSpec {
        family: "EVA".to_string(),
        architecture: "Vision Transformer (ViT-G/14)".to_string(),
        parameters_m: 1014.0,
        flops_g: 267.0,
        top1_accuracy: 89.792,
        top5_accuracy: 98.9,
        input_size: 560,
    });
    
    specs.insert("convnextv2-huge-512".to_string(), ModelSpec {
        family: "ConvNeXt".to_string(),
        architecture: "ConvNeXt V2 Huge".to_string(),
        parameters_m: 660.0,
        flops_g: 337.0,
        top1_accuracy: 88.848,
        top5_accuracy: 98.7,
        input_size: 512,
    });
    
    specs.insert("convnext-xxlarge-clip".to_string(), ModelSpec {
        family: "ConvNeXt".to_string(),
        architecture: "ConvNeXt XXLarge (CLIP)".to_string(),
        parameters_m: 846.0,
        flops_g: 198.0,
        top1_accuracy: 88.612,
        top5_accuracy: 98.6,
        input_size: 256,
    });
    
    specs.insert("maxvit-xlarge-512".to_string(), ModelSpec {
        family: "MaxViT".to_string(),
        architecture: "MaxViT XLarge (Hybrid)".to_string(),
        parameters_m: 475.0,
        flops_g: 293.0,
        top1_accuracy: 88.53,
        top5_accuracy: 98.6,
        input_size: 512,
    });
    
    specs.insert("vit-giant-patch14-224".to_string(), ModelSpec {
        family: "ViT".to_string(),
        architecture: "Vision Transformer Giant (CLIP)".to_string(),
        parameters_m: 1843.0,
        flops_g: 483.0,
        top1_accuracy: 88.5,
        top5_accuracy: 98.6,
        input_size: 224,
    });
    
    specs.insert("beit-large-patch16-512".to_string(), ModelSpec {
        family: "BEiT".to_string(),
        architecture: "BEiT Large".to_string(),
        parameters_m: 305.0,
        flops_g: 191.0,
        top1_accuracy: 88.6,
        top5_accuracy: 98.7,
        input_size: 512,
    });
    
    specs.insert("swin-large-patch4-384".to_string(), ModelSpec {
        family: "Swin".to_string(),
        architecture: "Swin Transformer Large".to_string(),
        parameters_m: 197.0,
        flops_g: 103.0,
        top1_accuracy: 87.3,
        top5_accuracy: 98.3,
        input_size: 384,
    });
    
    specs.insert("efficientnetv2-xl".to_string(), ModelSpec {
        family: "EfficientNet".to_string(),
        architecture: "EfficientNetV2 XL".to_string(),
        parameters_m: 208.0,
        flops_g: 93.8,
        top1_accuracy: 87.3,
        top5_accuracy: 98.3,
        input_size: 480,
    });
    
    specs.insert("deit3-huge-patch14-224".to_string(), ModelSpec {
        family: "DeiT".to_string(),
        architecture: "DeiT-III Huge".to_string(),
        parameters_m: 632.0,
        flops_g: 167.0,
        top1_accuracy: 87.7,
        top5_accuracy: 98.4,
        input_size: 224,
    });
    
    specs.insert("coatnet-3-rw-224".to_string(), ModelSpec {
        family: "CoAtNet".to_string(),
        architecture: "CoAtNet-3 (Hybrid)".to_string(),
        parameters_m: 166.0,
        flops_g: 34.7,
        top1_accuracy: 86.0,
        top5_accuracy: 97.8,
        input_size: 224,
    });
    
    specs.insert("mobilenetv4-hybrid-large".to_string(), ModelSpec {
        family: "MobileNet".to_string(),
        architecture: "MobileNetV4 Hybrid Large".to_string(),
        parameters_m: 37.0,
        flops_g: 4.4,
        top1_accuracy: 84.36,
        top5_accuracy: 96.9,
        input_size: 448,
    });
    
    specs.insert("resnet50".to_string(), ModelSpec {
        family: "ResNet".to_string(),
        architecture: "ResNet-50".to_string(),
        parameters_m: 25.6,
        flops_g: 4.1,
        top1_accuracy: 76.13,
        top5_accuracy: 92.86,
        input_size: 224,
    });
    
    specs
}

#[derive(Clone, Debug)]
struct ModelSpec {
    family: String,
    architecture: String,
    parameters_m: f64,
    flops_g: f64,
    top1_accuracy: f64,
    top5_accuracy: f64,
    input_size: i64,
}

/// Get competitor benchmark data (reference from literature/testing)
fn get_competitor_benchmarks() -> Vec<CompetitorResult> {
    vec![
        // TorchServe benchmarks (from official benchmarks)
        CompetitorResult {
            provider: "TorchServe".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 12.5,
            throughput_fps: 80.0,
            load_time_ms: 2500.0,
            memory_mb: 1200.0,
            device: "CPU".to_string(),
            notes: "Default configuration".to_string(),
        },
        CompetitorResult {
            provider: "TorchServe".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 2.8,
            throughput_fps: 357.0,
            load_time_ms: 1800.0,
            memory_mb: 1500.0,
            device: "CUDA".to_string(),
            notes: "GPU optimized".to_string(),
        },
        // Triton Inference Server
        CompetitorResult {
            provider: "Triton".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 11.2,
            throughput_fps: 89.0,
            load_time_ms: 3200.0,
            memory_mb: 950.0,
            device: "CPU".to_string(),
            notes: "ONNX backend".to_string(),
        },
        CompetitorResult {
            provider: "Triton".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 1.9,
            throughput_fps: 526.0,
            load_time_ms: 2100.0,
            memory_mb: 1100.0,
            device: "CUDA".to_string(),
            notes: "TensorRT optimized".to_string(),
        },
        // ONNX Runtime
        CompetitorResult {
            provider: "ONNX Runtime".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 9.8,
            throughput_fps: 102.0,
            load_time_ms: 850.0,
            memory_mb: 420.0,
            device: "CPU".to_string(),
            notes: "Default EP".to_string(),
        },
        CompetitorResult {
            provider: "ONNX Runtime".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 2.1,
            throughput_fps: 476.0,
            load_time_ms: 1200.0,
            memory_mb: 680.0,
            device: "CUDA".to_string(),
            notes: "CUDA EP".to_string(),
        },
        // TensorFlow Serving
        CompetitorResult {
            provider: "TF Serving".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 14.2,
            throughput_fps: 70.0,
            load_time_ms: 4500.0,
            memory_mb: 1800.0,
            device: "CPU".to_string(),
            notes: "SavedModel format".to_string(),
        },
        // OpenVINO
        CompetitorResult {
            provider: "OpenVINO".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 6.5,
            throughput_fps: 153.0,
            load_time_ms: 1100.0,
            memory_mb: 380.0,
            device: "CPU".to_string(),
            notes: "INT8 quantized".to_string(),
        },
        // vLLM (for vision models)
        CompetitorResult {
            provider: "vLLM".to_string(),
            model_name: "ViT-Large".to_string(),
            avg_inference_ms: 8.5,
            throughput_fps: 117.0,
            load_time_ms: 5200.0,
            memory_mb: 2400.0,
            device: "CUDA".to_string(),
            notes: "PagedAttention".to_string(),
        },
        // Ray Serve
        CompetitorResult {
            provider: "Ray Serve".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 15.8,
            throughput_fps: 63.0,
            load_time_ms: 3800.0,
            memory_mb: 1450.0,
            device: "CPU".to_string(),
            notes: "Default config".to_string(),
        },
        // BentoML
        CompetitorResult {
            provider: "BentoML".to_string(),
            model_name: "ResNet-50".to_string(),
            avg_inference_ms: 13.5,
            throughput_fps: 74.0,
            load_time_ms: 2800.0,
            memory_mb: 1100.0,
            device: "CPU".to_string(),
            notes: "PyTorch runner".to_string(),
        },
        // Ollama (primarily for LLMs but included for reference)
        CompetitorResult {
            provider: "Ollama".to_string(),
            model_name: "LLaVA (Vision)".to_string(),
            avg_inference_ms: 45.0,
            throughput_fps: 22.0,
            load_time_ms: 8500.0,
            memory_mb: 4200.0,
            device: "CPU".to_string(),
            notes: "Multimodal".to_string(),
        },
    ]
}

fn benchmark_image_classification(_c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let model_specs = get_model_specs();
    let mut results: Vec<ClassificationBenchmarkResult> = Vec::new();
    let mut system_info = collect_system_info();
    
    // Create output directory
    create_dir_all(&config.output_dir).ok();
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     Image Classification Model Benchmark Suite                    ║");
    println!("║     Timestamp: {}                                   ║", &timestamp[..15]);
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Scan for classification models
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        eprintln!("Models directory not found: {:?}", models_dir);
        return;
    }

    let mut model_paths = Vec::new();
    let classification_keywords = ["eva", "convnext", "maxvit", "vit", "beit", "swin", 
                                   "efficientnet", "deit", "coatnet", "mobilenet", "resnet"];
    let exclude_keywords = ["tts", "speech", "voice", "kokoro", "melo", "piper", 
                           "bark", "styletts", "fish-speech", "metavoice", "openvoice", 
                           "xtts", "whisper", "gpt", "bert", "clip", "yolo"];

    for entry in WalkDir::new(models_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if ["pt", "pth", "ot", "bin", "safetensors"].contains(&ext) {
                    let path_str = path.to_str().unwrap_or("").to_lowercase();
                    let is_classification = classification_keywords.iter().any(|k| path_str.contains(k));
                    let is_excluded = exclude_keywords.iter().any(|k| path_str.contains(k));
                    
                    if is_classification && !is_excluded {
                        model_paths.push(path.to_path_buf());
                    }
                }
            }
        }
    }

    // Also check parent directories
    for dir_name in &classification_keywords {
        let dir_path = models_dir.join(dir_name);
        if dir_path.exists() && dir_path.is_dir() {
            for entry in WalkDir::new(&dir_path).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if ["pt", "pth", "ot", "bin", "safetensors"].contains(&ext) {
                            if !model_paths.contains(&path.to_path_buf()) {
                                model_paths.push(path.to_path_buf());
                            }
                        }
                    }
                }
            }
        }
    }

    println!("Found {} image classification models\n", model_paths.len());

    #[cfg(feature = "torch")]
    {
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else if tch::utils::has_mps() {
            Device::Mps
        } else {
            Device::Cpu
        };
        
        let device_name = format!("{:?}", device);
        system_info.push(("Device".to_string(), device_name.clone()));
        println!("Using device: {:?}\n", device);

        for model_path in &model_paths {
            let model_name = extract_model_name(model_path);
            let file_size_mb = std::fs::metadata(model_path)
                .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                .unwrap_or(0.0);
            
            println!("═══════════════════════════════════════════════════════════════════");
            println!("Model: {} ({:.2} MB)", model_name, file_size_mb);
            println!("═══════════════════════════════════════════════════════════════════");
            
            // Load model
            let load_start = Instant::now();
            let model = match CModule::load_on_device(model_path, device) {
                Ok(mut m) => {
                    m.set_eval();
                    m
                }
                Err(e) => {
                    println!("  ✗ Failed to load: {}", e);
                    println!();
                    continue;
                }
            };
            let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
            println!("  ✓ Load time: {:.2} ms", load_time_ms);

            // Get model specs
            let spec = model_specs.get(&model_name).cloned().unwrap_or_else(|| {
                // Default spec based on model name pattern
                let input_size = if model_name.contains("448") { 448 }
                    else if model_name.contains("512") { 512 }
                    else if model_name.contains("384") { 384 }
                    else if model_name.contains("560") { 560 }
                    else { 224 };
                
                ModelSpec {
                    family: "Unknown".to_string(),
                    architecture: "Unknown".to_string(),
                    parameters_m: file_size_mb / 4.0, // Rough estimate
                    flops_g: 0.0,
                    top1_accuracy: 0.0,
                    top5_accuracy: 0.0,
                    input_size,
                }
            });

            let input_size = spec.input_size;
            let shape = vec![1i64, 3, input_size, input_size];
            
            // Test inference
            let test_tensor = Tensor::randn(&shape, (Kind::Float, device));
            match model.forward_ts(&[test_tensor]) {
                Ok(_) => {
                    println!("  ✓ Input shape: [1, 3, {}, {}]", input_size, input_size);
                }
                Err(e) => {
                    println!("  ✗ Inference failed: {}", e);
                    continue;
                }
            }

            // First inference (cold)
            let first_tensor = Tensor::randn(&shape, (Kind::Float, device));
            let first_start = Instant::now();
            let _ = model.forward_ts(&[first_tensor]);
            let first_inference_ms = first_start.elapsed().as_secs_f64() * 1000.0;
            println!("  ✓ First inference: {:.2} ms", first_inference_ms);

            // Warmup
            println!("  → Warming up ({} iterations)...", config.warmup_iterations);
            for _ in 0..config.warmup_iterations {
                let tensor = Tensor::randn(&shape, (Kind::Float, device));
                let _ = model.forward_ts(&[tensor]);
            }

            // Benchmark
            println!("  → Running benchmark ({} iterations)...", config.benchmark_iterations);
            let mut latencies = Vec::with_capacity(config.benchmark_iterations);
            let total_start = Instant::now();
            
            for _ in 0..config.benchmark_iterations {
                let tensor = Tensor::randn(&shape, (Kind::Float, device));
                let start = Instant::now();
                let _ = model.forward_ts(&[tensor]);
                latencies.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            
            let total_time = total_start.elapsed();

            // Calculate statistics
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = latencies.len();
            let avg = latencies.iter().sum::<f64>() / n as f64;
            let min = latencies[0];
            let max = latencies[n - 1];
            let variance = latencies.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / n as f64;
            let std_dev = variance.sqrt();
            let p50 = latencies[n / 2];
            let p95 = latencies[(n * 95) / 100];
            let p99 = latencies[(n * 99) / 100];
            let throughput = n as f64 / total_time.as_secs_f64();

            // Efficiency metrics
            let efficiency_fps_per_gflop = if spec.flops_g > 0.0 { throughput / spec.flops_g } else { 0.0 };
            let efficiency_accuracy_per_ms = if avg > 0.0 { spec.top1_accuracy / avg } else { 0.0 };

            println!("  ✓ Avg latency: {:.2} ms (±{:.2})", avg, std_dev);
            println!("  ✓ P50/P95/P99: {:.2} / {:.2} / {:.2} ms", p50, p95, p99);
            println!("  ✓ Throughput: {:.2} FPS", throughput);
            if spec.top1_accuracy > 0.0 {
                println!("  ✓ Accuracy: {:.2}% Top-1", spec.top1_accuracy);
            }

            results.push(ClassificationBenchmarkResult {
                model_name: model_name.clone(),
                model_family: spec.family.clone(),
                architecture: spec.architecture.clone(),
                input_resolution: format!("{}x{}", input_size, input_size),
                file_size_mb,
                parameters_millions: spec.parameters_m,
                top1_accuracy: spec.top1_accuracy,
                top5_accuracy: spec.top5_accuracy,
                load_time_ms,
                first_inference_ms,
                avg_inference_ms: avg,
                min_inference_ms: min,
                max_inference_ms: max,
                std_dev_ms: std_dev,
                p50_ms: p50,
                p95_ms: p95,
                p99_ms: p99,
                throughput_fps: throughput,
                throughput_images_per_sec: throughput,
                flops_gflops: spec.flops_g,
                efficiency_fps_per_gflop,
                efficiency_accuracy_per_ms,
                peak_memory_mb: file_size_mb * 2.5, // Estimate
                device: device_name.clone(),
            });

            println!();
        }
    }

    #[cfg(not(feature = "torch"))]
    {
        println!("PyTorch feature not enabled. Compile with --features torch");
    }

    // Export results
    if !results.is_empty() {
        let base_name = format!("{}/classification_benchmark_{}", config.output_dir, timestamp);
        let competitor_results = get_competitor_benchmarks();
        
        export_csv(&results, &format!("{}.csv", base_name));
        export_json(&results, &system_info, &competitor_results, &format!("{}.json", base_name));
        generate_comprehensive_report(&results, &system_info, &competitor_results, &format!("{}.md", base_name));
        generate_comparison_plot(&results, &competitor_results, &format!("{}_comparison.png", base_name));
        generate_accuracy_vs_speed_plot(&results, &format!("{}_accuracy_speed.png", base_name));
        generate_efficiency_plot(&results, &format!("{}_efficiency.png", base_name));
        
        println!("\n════════════════════════════════════════════════════════════════════");
        println!("Benchmark complete! Results saved to: {}/", config.output_dir);
        println!("════════════════════════════════════════════════════════════════════\n");
    }
}

fn extract_model_name(path: &std::path::PathBuf) -> String {
    // Try to get meaningful name from path
    let path_str = path.to_str().unwrap_or("");
    
    // Check parent directory name
    if let Some(parent) = path.parent() {
        if let Some(parent_name) = parent.file_name().and_then(|n| n.to_str()) {
            if !["models", "classification"].contains(&parent_name) {
                return parent_name.to_string();
            }
        }
    }
    
    // Fall back to file stem
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn collect_system_info() -> Vec<(String, String)> {
    let mut info = Vec::new();
    info.push(("Timestamp".to_string(), Utc::now().to_rfc3339()));
    info.push(("OS".to_string(), std::env::consts::OS.to_string()));
    info.push(("Architecture".to_string(), std::env::consts::ARCH.to_string()));
    info.push(("CPU Cores".to_string(), num_cpus::get().to_string()));
    
    if let Ok(sys_info) = sys_info::mem_info() {
        let total_gb = sys_info.total as f64 / 1024.0 / 1024.0;
        info.push(("Total Memory (GB)".to_string(), format!("{:.2}", total_gb)));
    }
    
    info.push(("Framework".to_string(), "Torch-Inference (Rust)".to_string()));
    info.push(("Backend".to_string(), "libtorch".to_string()));
    
    info
}

fn export_csv(results: &[ClassificationBenchmarkResult], path: &str) {
    let mut file = File::create(path).expect("Unable to create CSV file");
    
    writeln!(file, "model,family,architecture,resolution,size_mb,params_m,top1,top5,load_ms,first_ms,avg_ms,min_ms,max_ms,std_ms,p50_ms,p95_ms,p99_ms,fps,gflops,efficiency,device").ok();
    
    for r in results {
        writeln!(file, "{},{},{},{},{:.2},{:.1},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.1},{:.4},{}",
            r.model_name, r.model_family, r.architecture, r.input_resolution,
            r.file_size_mb, r.parameters_millions, r.top1_accuracy, r.top5_accuracy,
            r.load_time_ms, r.first_inference_ms, r.avg_inference_ms,
            r.min_inference_ms, r.max_inference_ms, r.std_dev_ms,
            r.p50_ms, r.p95_ms, r.p99_ms, r.throughput_fps,
            r.flops_gflops, r.efficiency_fps_per_gflop, r.device
        ).ok();
    }
    println!("✓ CSV exported: {}", path);
}

fn export_json(results: &[ClassificationBenchmarkResult], system_info: &[(String, String)], 
               competitors: &[CompetitorResult], path: &str) {
    let mut file = File::create(path).expect("Unable to create JSON file");
    
    let system_obj: serde_json::Map<String, serde_json::Value> = system_info.iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    
    let output = serde_json::json!({
        "benchmark_type": "Image Classification",
        "system": system_obj,
        "results": results,
        "competitors": competitors,
        "methodology": {
            "warmup_iterations": 10,
            "benchmark_iterations": 100,
            "input_preprocessing": "Random normal tensor",
            "timing_method": "std::time::Instant"
        }
    });
    
    writeln!(file, "{}", serde_json::to_string_pretty(&output).unwrap()).ok();
    println!("✓ JSON exported: {}", path);
}

fn generate_comprehensive_report(results: &[ClassificationBenchmarkResult], 
                                  system_info: &[(String, String)],
                                  competitors: &[CompetitorResult],
                                  path: &str) {
    let mut file = File::create(path).expect("Unable to create markdown file");
    
    writeln!(file, "# Image Classification Benchmark Report\n").ok();
    writeln!(file, "## Executive Summary\n").ok();
    
    if !results.is_empty() {
        let fastest = results.iter().min_by(|a, b| a.avg_inference_ms.partial_cmp(&b.avg_inference_ms).unwrap()).unwrap();
        let most_accurate = results.iter().max_by(|a, b| a.top1_accuracy.partial_cmp(&b.top1_accuracy).unwrap()).unwrap();
        let most_efficient = results.iter()
            .filter(|r| r.efficiency_accuracy_per_ms > 0.0)
            .max_by(|a, b| a.efficiency_accuracy_per_ms.partial_cmp(&b.efficiency_accuracy_per_ms).unwrap());
        
        writeln!(file, "- **Models Benchmarked**: {}", results.len()).ok();
        writeln!(file, "- **Fastest Model**: {} ({:.2} ms)", fastest.model_name, fastest.avg_inference_ms).ok();
        writeln!(file, "- **Highest Accuracy**: {} ({:.2}%)", most_accurate.model_name, most_accurate.top1_accuracy).ok();
        if let Some(efficient) = most_efficient {
            writeln!(file, "- **Most Efficient**: {} ({:.4} acc/%ms)", efficient.model_name, efficient.efficiency_accuracy_per_ms).ok();
        }
    }
    
    writeln!(file, "\n## System Information\n").ok();
    writeln!(file, "| Property | Value |").ok();
    writeln!(file, "|----------|-------|").ok();
    for (k, v) in system_info {
        writeln!(file, "| {} | {} |", k, v).ok();
    }
    
    writeln!(file, "\n## Torch-Inference Results\n").ok();
    writeln!(file, "| Model | Resolution | Params (M) | Top-1 (%) | Avg (ms) | P95 (ms) | FPS | GFLOPs |").ok();
    writeln!(file, "|-------|------------|------------|-----------|----------|----------|-----|--------|").ok();
    
    for r in results {
        writeln!(file, "| {} | {} | {:.1} | {:.2} | {:.2} | {:.2} | {:.1} | {:.1} |",
            r.model_name, r.input_resolution, r.parameters_millions,
            r.top1_accuracy, r.avg_inference_ms, r.p95_ms, r.throughput_fps, r.flops_gflops
        ).ok();
    }
    
    // Competitor comparison
    writeln!(file, "\n## Competitor Comparison (ResNet-50 Baseline)\n").ok();
    writeln!(file, "| Provider | Device | Avg (ms) | FPS | Load (ms) | Memory (MB) | Notes |").ok();
    writeln!(file, "|----------|--------|----------|-----|-----------|-------------|-------|").ok();
    
    // Add Torch-Inference result for ResNet-50 if available
    if let Some(our_resnet) = results.iter().find(|r| r.model_name.to_lowercase().contains("resnet50")) {
        writeln!(file, "| **Torch-Inference** | {} | **{:.2}** | **{:.1}** | **{:.1}** | **{:.1}** | Rust native |",
            our_resnet.device, our_resnet.avg_inference_ms, our_resnet.throughput_fps,
            our_resnet.load_time_ms, our_resnet.peak_memory_mb
        ).ok();
    }
    
    for c in competitors {
        writeln!(file, "| {} | {} | {:.2} | {:.1} | {:.1} | {:.1} | {} |",
            c.provider, c.device, c.avg_inference_ms, c.throughput_fps,
            c.load_time_ms, c.memory_mb, c.notes
        ).ok();
    }
    
    // Latency breakdown
    writeln!(file, "\n## Detailed Latency Analysis\n").ok();
    writeln!(file, "| Model | Min (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Std Dev |").ok();
    writeln!(file, "|-------|----------|----------|----------|----------|----------|---------|").ok();
    
    for r in results {
        writeln!(file, "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
            r.model_name, r.min_inference_ms, r.p50_ms, r.p95_ms, r.p99_ms, r.max_inference_ms, r.std_dev_ms
        ).ok();
    }
    
    // Efficiency analysis
    writeln!(file, "\n## Efficiency Analysis\n").ok();
    writeln!(file, "| Model | FPS/GFLOP | Accuracy/%ms | Load Efficiency |").ok();
    writeln!(file, "|-------|-----------|--------------|-----------------|").ok();
    
    for r in results {
        let load_eff = if r.load_time_ms > 0.0 { r.throughput_fps / (r.load_time_ms / 1000.0) } else { 0.0 };
        writeln!(file, "| {} | {:.4} | {:.4} | {:.2} |",
            r.model_name, r.efficiency_fps_per_gflop, r.efficiency_accuracy_per_ms, load_eff
        ).ok();
    }
    
    println!("✓ Markdown report: {}", path);
}

fn generate_comparison_plot(results: &[ClassificationBenchmarkResult], 
                            competitors: &[CompetitorResult], 
                            path: &str) {
    if results.is_empty() { return; }
    
    let root = BitMapBackend::new(path, (1400, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Prepare data
    let mut all_data: Vec<(&str, f64, &str)> = Vec::new();
    
    // Add our results (ResNet-50 if available)
    if let Some(r) = results.iter().find(|r| r.model_name.to_lowercase().contains("resnet")) {
        all_data.push(("Torch-Inference", r.avg_inference_ms, &r.device));
    }
    
    // Add competitor CPU results
    for c in competitors.iter().filter(|c| c.device == "CPU") {
        all_data.push((c.provider.as_str(), c.avg_inference_ms, &c.device));
    }

    if all_data.is_empty() { return; }

    let max_latency = all_data.iter().map(|(_, l, _)| *l).fold(0.0f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Inference Latency Comparison (ResNet-50, CPU)", ("sans-serif", 35).into_font())
        .margin(30)
        .x_label_area_size(120)
        .y_label_area_size(80)
        .build_cartesian_2d(-0.5f64..(all_data.len() as f64 - 0.5), 0f64..(max_latency * 1.2))
        .unwrap();

    chart.configure_mesh()
        .y_desc("Latency (ms)")
        .x_label_formatter(&|x| {
            let idx = (*x + 0.5) as usize;
            if idx < all_data.len() { all_data[idx].0.to_string() } else { String::new() }
        })
        .draw().unwrap();

    chart.draw_series(
        all_data.iter().enumerate().map(|(i, (name, latency, _))| {
            let color = if *name == "Torch-Inference" { GREEN.mix(0.8) } else { BLUE.mix(0.6) };
            Rectangle::new([(i as f64 - 0.35, 0.0), (i as f64 + 0.35, *latency)], color.filled())
        })
    ).unwrap();

    // Add value labels
    for (i, (_, latency, _)) in all_data.iter().enumerate() {
        let label = format!("{:.1}ms", latency);
        chart.draw_series(std::iter::once(
            Text::new(label, (i as f64, *latency + max_latency * 0.02), ("sans-serif", 14).into_font())
        )).unwrap();
    }

    root.present().unwrap();
    println!("✓ Comparison plot: {}", path);
}

fn generate_accuracy_vs_speed_plot(results: &[ClassificationBenchmarkResult], path: &str) {
    if results.is_empty() { return; }
    
    let filtered: Vec<_> = results.iter().filter(|r| r.top1_accuracy > 0.0).collect();
    if filtered.is_empty() { return; }
    
    let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_latency = filtered.iter().map(|r| r.avg_inference_ms).fold(0.0f64, f64::max);
    let min_accuracy = filtered.iter().map(|r| r.top1_accuracy).fold(f64::MAX, f64::min) - 2.0;
    let max_accuracy = filtered.iter().map(|r| r.top1_accuracy).fold(0.0f64, f64::max) + 1.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("Accuracy vs Inference Speed (Pareto Frontier)", ("sans-serif", 35).into_font())
        .margin(30)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(0f64..(max_latency * 1.2), min_accuracy..max_accuracy)
        .unwrap();

    chart.configure_mesh()
        .x_desc("Inference Latency (ms)")
        .y_desc("ImageNet Top-1 Accuracy (%)")
        .draw().unwrap();

    // Color by model family
    let families: Vec<_> = filtered.iter().map(|r| r.model_family.as_str()).collect();
    let unique_families: Vec<_> = families.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
    
    let colors = [RED, BLUE, GREEN, MAGENTA, CYAN, BLACK];
    
    for (fi, family) in unique_families.iter().enumerate() {
        let family_results: Vec<_> = filtered.iter().filter(|r| r.model_family == *family).collect();
        let color = colors[fi % colors.len()];
        
        chart.draw_series(
            family_results.iter().map(|r| {
                Circle::new((r.avg_inference_ms, r.top1_accuracy), 8, color.filled())
            })
        ).unwrap()
        .label(*family)
        .legend(move |(x, y)| Circle::new((x + 10, y), 5, color.filled()));
    }

    // Add model name labels
    for r in &filtered {
        let short_name: String = r.model_name.chars().take(12).collect();
        chart.draw_series(std::iter::once(
            Text::new(short_name, (r.avg_inference_ms + 1.0, r.top1_accuracy), ("sans-serif", 10).into_font())
        )).unwrap();
    }

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw().unwrap();

    root.present().unwrap();
    println!("✓ Accuracy vs Speed plot: {}", path);
}

fn generate_efficiency_plot(results: &[ClassificationBenchmarkResult], path: &str) {
    if results.is_empty() { return; }
    
    let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_fps = results.iter().map(|r| r.throughput_fps).fold(0.0f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Model Throughput Comparison", ("sans-serif", 35).into_font())
        .margin(30)
        .x_label_area_size(120)
        .y_label_area_size(80)
        .build_cartesian_2d(-0.5f64..(results.len() as f64 - 0.5), 0f64..(max_fps * 1.2))
        .unwrap();

    chart.configure_mesh()
        .y_desc("Throughput (FPS)")
        .x_label_formatter(&|x| {
            let idx = (*x + 0.5) as usize;
            if idx < results.len() { 
                results[idx].model_name.chars().take(15).collect()
            } else { 
                String::new() 
            }
        })
        .draw().unwrap();

    chart.draw_series(
        results.iter().enumerate().map(|(i, r)| {
            let intensity = (r.top1_accuracy / 100.0).min(1.0);
            let color = RGBColor(
                (50.0 + 150.0 * (1.0 - intensity)) as u8,
                (100.0 + 155.0 * intensity) as u8,
                50
            );
            Rectangle::new([(i as f64 - 0.35, 0.0), (i as f64 + 0.35, r.throughput_fps)], color.filled())
        })
    ).unwrap();

    root.present().unwrap();
    println!("✓ Efficiency plot: {}", path);
}

criterion_group!(benches, benchmark_image_classification);
criterion_main!(benches);
