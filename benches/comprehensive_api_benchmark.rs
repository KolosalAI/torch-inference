use criterion::{criterion_group, criterion_main, Criterion};
use std::path::Path;
use std::time::Instant;
use serde_json::json;
use std::fs::{File, create_dir_all};
use std::io::Write;
use plotters::prelude::*;
use walkdir::WalkDir;
use chrono::Utc;

#[cfg(feature = "torch")]
use tch::Device;

/// Struct to hold benchmark results
#[derive(Clone, Debug)]
struct BenchmarkResult {
    model_name: String,
    model_type: String,
    model_path: String,
    file_size_mb: f64,
    load_time_ms: f64,
    warmup_time_ms: f64,
    latency_ms: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
    std_dev_ms: f64,
    throughput_req_per_sec: f64,
    input_shape: String,
    device: String,
}

/// Benchmark configuration
struct BenchmarkConfig {
    warmup_iterations: usize,
    benchmark_iterations: usize,
    output_dir: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 5,
            benchmark_iterations: 20,
            output_dir: "benchmark_results".to_string(),
        }
    }
}

fn benchmark_models(_c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let mut results = Vec::new();
    let mut system_info = collect_system_info();
    
    // Create output directory
    create_dir_all(&config.output_dir).ok();
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     Model Benchmark Suite                                ║");
    println!("║     Timestamp: {}                            ║", &timestamp[..15]);
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Scan models
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        eprintln!("Models directory not found: {:?}", models_dir);
        return;
    }

    let mut model_paths = Vec::new();
    for entry in WalkDir::new(models_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if ["pt", "pth", "onnx", "safetensors", "ot", "bin"].contains(&ext) {
                    let path_str = path.to_str().unwrap_or("");
                    let tts_keywords = ["tts", "speech", "voice", "kokoro", "melo", "piper", "bark", "styletts", "fish-speech", "metavoice", "openvoice", "xtts"];
                    if !tts_keywords.iter().any(|k| path_str.to_lowercase().contains(k)) {
                        model_paths.push(path.to_path_buf());
                    }
                }
            }
        }
    }

    println!("Found {} non-TTS models\n", model_paths.len());

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
            let model_name = model_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            
            let file_size_mb = std::fs::metadata(model_path)
                .map(|m| m.len() as f64 / 1024.0 / 1024.0)
                .unwrap_or(0.0);
            
            println!("═══ {} ({:.2} MB) ═══", model_name, file_size_mb);
            
            if let Some(ext) = model_path.extension().and_then(|e| e.to_str()) {
                if ["pt", "pth", "ot"].contains(&ext) {
                    let load_start = Instant::now();
                    
                    match tch::CModule::load_on_device(model_path, device) {
                        Ok(mut model) => {
                            model.set_eval();
                            let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
                            println!("  ✓ Loaded in {:.2} ms", load_time_ms);
                            
                            let test_inputs = vec![
                                ("image_224", vec![1i64, 3, 224, 224]),
                                ("image_384", vec![1, 3, 384, 384]),
                                ("text_128", vec![1, 128]),
                                ("audio_16k", vec![1, 16000]),
                                ("small", vec![1, 10]),
                            ];
                            
                            for (input_name, shape) in test_inputs {
                                let tensor = tch::Tensor::zeros(&shape, (tch::Kind::Float, device));
                                
                                if model.forward_ts(&[tensor]).is_ok() {
                                    println!("  ✓ Input shape: {} {:?}", input_name, shape);
                                    
                                    // Warmup
                                    let warmup_start = Instant::now();
                                    for _ in 0..config.warmup_iterations {
                                        let t = tch::Tensor::zeros(&shape, (tch::Kind::Float, device));
                                        let _ = model.forward_ts(&[t]);
                                    }
                                    let warmup_time_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
                                    
                                    // Benchmark
                                    let mut latencies = Vec::new();
                                    for _ in 0..config.benchmark_iterations {
                                        let t = tch::Tensor::zeros(&shape, (tch::Kind::Float, device));
                                        let start = Instant::now();
                                        let _ = model.forward_ts(&[t]);
                                        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
                                    }
                                    
                                    let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
                                    let min = latencies.iter().cloned().fold(f64::MAX, f64::min);
                                    let max = latencies.iter().cloned().fold(0.0, f64::max);
                                    let variance = latencies.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / latencies.len() as f64;
                                    let std_dev = variance.sqrt();
                                    let throughput = 1000.0 / avg;
                                    
                                    println!("  ✓ Latency: {:.2} ms (±{:.2}), Throughput: {:.2} req/s", avg, std_dev, throughput);
                                    
                                    results.push(BenchmarkResult {
                                        model_name: model_name.clone(),
                                        model_type: "PyTorch".to_string(),
                                        model_path: model_path.to_string_lossy().to_string(),
                                        file_size_mb,
                                        load_time_ms,
                                        warmup_time_ms,
                                        latency_ms: avg,
                                        min_latency_ms: min,
                                        max_latency_ms: max,
                                        std_dev_ms: std_dev,
                                        throughput_req_per_sec: throughput,
                                        input_shape: format!("{:?}", shape),
                                        device: device_name.clone(),
                                    });
                                    
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            println!("  ✗ Failed to load: {}", e);
                        }
                    }
                }
            }
            println!();
        }
    }

    #[cfg(not(feature = "torch"))]
    {
        println!("PyTorch feature not enabled. Compile with --features torch");
    }

    if !results.is_empty() {
        let base_name = format!("{}/model_benchmark_{}", config.output_dir, timestamp);
        
        export_csv(&results, &format!("{}.csv", base_name));
        export_json(&results, &system_info, &format!("{}.json", base_name));
        generate_markdown_report(&results, &system_info, &format!("{}.md", base_name));
        generate_throughput_plot(&results, &format!("{}_throughput.png", base_name));
        generate_load_time_plot(&results, &format!("{}_loadtime.png", base_name));
        
        println!("════════════════════════════════════════════════════════════");
        println!("Benchmark complete! Results saved to: {}/", config.output_dir);
        println!("════════════════════════════════════════════════════════════\n");
    }
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
    info
}

fn export_csv(results: &[BenchmarkResult], path: &str) {
    let mut file = File::create(path).expect("Unable to create CSV file");
    writeln!(file, "model,type,path,size_mb,load_time_ms,warmup_ms,latency_ms,min_ms,max_ms,std_dev_ms,throughput_rps,input_shape,device").ok();
    
    for r in results {
        writeln!(file, "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}",
            r.model_name, r.model_type, r.model_path, r.file_size_mb, r.load_time_ms,
            r.warmup_time_ms, r.latency_ms, r.min_latency_ms, r.max_latency_ms,
            r.std_dev_ms, r.throughput_req_per_sec, r.input_shape, r.device
        ).ok();
    }
    println!("✓ CSV exported: {}", path);
}

fn export_json(results: &[BenchmarkResult], system_info: &[(String, String)], path: &str) {
    let mut file = File::create(path).expect("Unable to create JSON file");
    
    let system_obj: serde_json::Map<String, serde_json::Value> = system_info.iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    
    let results_arr: Vec<serde_json::Value> = results.iter().map(|r| {
        json!({
            "model": r.model_name,
            "type": r.model_type,
            "path": r.model_path,
            "file_size_mb": r.file_size_mb,
            "device": r.device,
            "input_shape": r.input_shape,
            "timing": {
                "load_time_ms": r.load_time_ms,
                "warmup_time_ms": r.warmup_time_ms
            },
            "latency": {
                "avg_ms": r.latency_ms,
                "min_ms": r.min_latency_ms,
                "max_ms": r.max_latency_ms,
                "std_dev_ms": r.std_dev_ms
            },
            "throughput_rps": r.throughput_req_per_sec
        })
    }).collect();
    
    let output = json!({ "system": system_obj, "results": results_arr });
    writeln!(file, "{}", serde_json::to_string_pretty(&output).unwrap()).ok();
    println!("✓ JSON exported: {}", path);
}

fn generate_markdown_report(results: &[BenchmarkResult], system_info: &[(String, String)], path: &str) {
    let mut file = File::create(path).expect("Unable to create markdown file");
    
    writeln!(file, "# Model Benchmark Report\n").ok();
    
    writeln!(file, "## System Information\n").ok();
    writeln!(file, "| Property | Value |").ok();
    writeln!(file, "|----------|-------|").ok();
    for (k, v) in system_info {
        writeln!(file, "| {} | {} |", k, v).ok();
    }
    
    writeln!(file, "\n## Results\n").ok();
    writeln!(file, "| Model | Size (MB) | Load (ms) | Latency (ms) | Throughput (req/s) | Device |").ok();
    writeln!(file, "|-------|-----------|-----------|--------------|--------------------| -------|").ok();
    
    for r in results {
        writeln!(file, "| {} | {:.2} | {:.2} | {:.2} ± {:.2} | {:.2} | {} |",
            r.model_name, r.file_size_mb, r.load_time_ms, r.latency_ms, r.std_dev_ms, r.throughput_req_per_sec, r.device
        ).ok();
    }
    
    // Summary
    if !results.is_empty() {
        let fastest = results.iter().min_by(|a, b| a.latency_ms.partial_cmp(&b.latency_ms).unwrap()).unwrap();
        let highest_throughput = results.iter().max_by(|a, b| a.throughput_req_per_sec.partial_cmp(&b.throughput_req_per_sec).unwrap()).unwrap();
        
        writeln!(file, "\n## Summary\n").ok();
        writeln!(file, "- **Models Tested**: {}", results.len()).ok();
        writeln!(file, "- **Fastest Model**: {} ({:.2} ms)", fastest.model_name, fastest.latency_ms).ok();
        writeln!(file, "- **Highest Throughput**: {} ({:.2} req/s)", highest_throughput.model_name, highest_throughput.throughput_req_per_sec).ok();
    }
    
    println!("✓ Markdown report: {}", path);
}

fn generate_throughput_plot(results: &[BenchmarkResult], path: &str) {
    if results.is_empty() { return; }
    
    let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_throughput = results.iter().map(|r| r.throughput_req_per_sec).fold(0.0f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Model Throughput (req/sec)", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(100)
        .y_label_area_size(60)
        .build_cartesian_2d(-0.5f64..(results.len() as f64 - 0.5), 0f64..(max_throughput * 1.2))
        .unwrap();

    chart.configure_mesh()
        .y_desc("Throughput (req/sec)")
        .x_label_formatter(&|x| {
            let idx = (*x + 0.5) as usize;
            if idx < results.len() { results[idx].model_name.clone() } else { String::new() }
        })
        .draw().unwrap();

    chart.draw_series(
        results.iter().enumerate().map(|(i, r)| {
            Rectangle::new([(i as f64 - 0.4, 0.0), (i as f64 + 0.4, r.throughput_req_per_sec)], BLUE.mix(0.7).filled())
        })
    ).unwrap();
    
    root.present().unwrap();
    println!("✓ Throughput plot: {}", path);
}

fn generate_load_time_plot(results: &[BenchmarkResult], path: &str) {
    if results.is_empty() { return; }
    
    let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_load_time = results.iter().map(|r| r.load_time_ms).fold(0.0f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Model Load Time (ms)", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(100)
        .y_label_area_size(60)
        .build_cartesian_2d(-0.5f64..(results.len() as f64 - 0.5), 0f64..(max_load_time * 1.2))
        .unwrap();

    chart.configure_mesh()
        .y_desc("Load Time (ms)")
        .x_label_formatter(&|x| {
            let idx = (*x + 0.5) as usize;
            if idx < results.len() { results[idx].model_name.clone() } else { String::new() }
        })
        .draw().unwrap();

    chart.draw_series(
        results.iter().enumerate().map(|(i, r)| {
            Rectangle::new([(i as f64 - 0.4, 0.0), (i as f64 + 0.4, r.load_time_ms)], GREEN.mix(0.7).filled())
        })
    ).unwrap();
    
    root.present().unwrap();
    println!("✓ Load time plot: {}", path);
}

criterion_group!(benches, benchmark_models);
criterion_main!(benches);
