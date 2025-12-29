use criterion::{criterion_group, criterion_main, Criterion};
use std::path::Path;
use std::time::Instant;
use std::fs::{File, create_dir_all};
use std::io::Write;
use plotters::prelude::*;
use walkdir::WalkDir;
use tokio::runtime::Runtime;
use std::sync::Arc;
use chrono::Utc;

#[cfg(feature = "torch")]
use tch::{CModule, Device, Tensor, Kind};

/// Struct to hold concurrent benchmark results
#[derive(Clone, Debug)]
struct ConcurrentBenchmarkResult {
    model_name: String,
    concurrency_level: usize,
    total_requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    total_time_ms: f64,
    throughput_req_per_sec: f64,
    avg_latency_ms: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
    p50_latency_ms: f64,
    p75_latency_ms: f64,
    p90_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,
    std_dev_ms: f64,
}

/// Benchmark configuration
struct BenchmarkConfig {
    concurrency_levels: Vec<usize>,
    requests_per_level: usize,
    warmup_requests: usize,
    output_dir: String,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            concurrency_levels: vec![1, 2, 4, 8, 16, 32, 64],
            requests_per_level: 100,
            warmup_requests: 10,
            output_dir: "benchmark_results".to_string(),
        }
    }
}

fn benchmark_concurrent_throughput(_c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();
    
    // Create output directory
    create_dir_all(&config.output_dir).ok();
    
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     Concurrent Throughput Benchmark                      ║");
    println!("║     Timestamp: {}                            ║", &timestamp[..15]);
    println!("╚══════════════════════════════════════════════════════════╝\n");
    
    // Find models
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
                if ["pt", "pth", "ot"].contains(&ext) {
                    let path_str = path.to_str().unwrap_or("");
                    let tts_keywords = ["tts", "speech", "voice", "kokoro", "melo", "piper", "bark", "styletts", "fish-speech", "metavoice", "openvoice", "xtts"];
                    if !tts_keywords.iter().any(|k| path_str.to_lowercase().contains(k)) {
                        model_paths.push(path.to_path_buf());
                    }
                }
            }
        }
    }

    if model_paths.is_empty() {
        println!("⚠️  No compatible models found");
        return;
    }

    println!("Found {} non-TTS PyTorch models\n", model_paths.len());

    let mut all_results = Vec::new();
    let mut system_info = collect_system_info();

    #[cfg(feature = "torch")]
    {
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else if tch::utils::has_mps() {
            Device::Mps
        } else {
            Device::Cpu
        };
        
        system_info.push(("Device".to_string(), format!("{:?}", device)));
        println!("Using device: {:?}\n", device);

        for model_path in &model_paths {
            let is_ot = model_path.extension().and_then(|e| e.to_str()) == Some("ot");
            if !is_ot && !all_results.is_empty() {
                continue;
            }
            
            let model_name = model_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            
            println!("═══ Testing: {} ═══", model_name);
            
            let model = match CModule::load_on_device(model_path, device) {
                Ok(mut m) => {
                    m.set_eval();
                    println!("✓ Model loaded");
                    Arc::new(m)
                }
                Err(e) => {
                    println!("✗ Failed to load: {}", e);
                    continue;
                }
            };

            let test_shapes = vec![
                ("image_224", vec![1, 3, 224, 224]),
                ("image_384", vec![1, 3, 384, 384]),
                ("text_seq_128", vec![1, 128]),
                ("text_seq_256", vec![1, 256]),
                ("audio_16k", vec![1, 16000]),
                ("small_1d", vec![1, 10]),
            ];

            let mut working_shape = None;
            for (name, shape) in test_shapes {
                let tensor_f = Tensor::zeros(&shape, (Kind::Float, device));
                if model.forward_ts(&[tensor_f]).is_ok() {
                    println!("✓ Working input shape (float): {} {:?}", name, shape);
                    working_shape = Some((shape, Kind::Float));
                    break;
                }
                
                let tensor_i = Tensor::zeros(&shape, (Kind::Int64, device));
                if model.forward_ts(&[tensor_i]).is_ok() {
                    println!("✓ Working input shape (int64): {} {:?}", name, shape);
                    working_shape = Some((shape, Kind::Int64));
                    break;
                }
            }

            let (shape, kind) = match working_shape {
                Some(s) => s,
                None => {
                    println!("✗ No working input shape found\n");
                    continue;
                }
            };

            // Warmup
            println!("  Warming up ({} requests)...", config.warmup_requests);
            for _ in 0..config.warmup_requests {
                let tensor = Tensor::zeros(&shape, (kind, device));
                let _ = model.forward_ts(&[tensor]);
            }

            for &concurrency in &config.concurrency_levels {
                print!("  Concurrency {:>2}: ", concurrency);
                std::io::stdout().flush().ok();
                
                let model_clone = Arc::clone(&model);
                let shape_clone = shape.clone();
                let model_name_clone = model_name.clone();
                
                let result = rt.block_on(async {
                    benchmark_concurrent_requests(
                        model_clone,
                        shape_clone,
                        kind,
                        device,
                        concurrency,
                        config.requests_per_level,
                        model_name_clone,
                    ).await
                });

                match result {
                    Ok(res) => {
                        println!("{:>8.2} req/s | avg: {:>6.2}ms | p95: {:>6.2}ms | p99: {:>6.2}ms",
                            res.throughput_req_per_sec, res.avg_latency_ms, res.p95_latency_ms, res.p99_latency_ms);
                        all_results.push(res);
                    }
                    Err(e) => {
                        println!("FAILED: {}", e);
                    }
                }
            }

            println!();
            break; // Only benchmark first working model
        }
    }

    #[cfg(not(feature = "torch"))]
    {
        println!("PyTorch feature not enabled. Compile with --features torch");
    }

    if !all_results.is_empty() {
        let base_name = format!("{}/benchmark_{}", config.output_dir, timestamp);
        
        export_csv(&all_results, &format!("{}.csv", base_name));
        export_json(&all_results, &system_info, &format!("{}.json", base_name));
        generate_markdown_report(&all_results, &system_info, &format!("{}.md", base_name));
        generate_throughput_plot(&all_results, &format!("{}_throughput.png", base_name));
        generate_latency_plot(&all_results, &format!("{}_latency.png", base_name));
        generate_scaling_plot(&all_results, &format!("{}_scaling.png", base_name));
        
        println!("\n════════════════════════════════════════════════════════════");
        println!("Benchmark complete! Results saved to: {}/", config.output_dir);
        println!("  - {}.csv", base_name);
        println!("  - {}.json", base_name);
        println!("  - {}.md", base_name);
        println!("  - {}_throughput.png", base_name);
        println!("  - {}_latency.png", base_name);
        println!("  - {}_scaling.png", base_name);
        println!("════════════════════════════════════════════════════════════\n");
    } else {
        println!("⚠️  No successful benchmarks to export");
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

#[cfg(feature = "torch")]
async fn benchmark_concurrent_requests(
    model: Arc<CModule>,
    shape: Vec<i64>,
    kind: Kind,
    device: Device,
    concurrency: usize,
    total_requests: usize,
    model_name: String,
) -> Result<ConcurrentBenchmarkResult, Box<dyn std::error::Error + Send + Sync>> {
    use tokio::task;
    
    let start = Instant::now();
    let requests_per_worker = total_requests / concurrency;
    let mut handles = Vec::new();

    for _ in 0..concurrency {
        let model_clone = Arc::clone(&model);
        let shape_clone = shape.clone();
        
        let handle = task::spawn_blocking(move || {
            let mut worker_latencies = Vec::new();
            let mut failures = 0usize;
            
            for _ in 0..requests_per_worker {
                let tensor = Tensor::zeros(&shape_clone, (kind, device));
                let req_start = Instant::now();
                
                match model_clone.forward_ts(&[tensor]) {
                    Ok(_) => {
                        worker_latencies.push(req_start.elapsed().as_secs_f64() * 1000.0);
                    }
                    Err(_) => {
                        failures += 1;
                    }
                }
            }
            
            (worker_latencies, failures)
        });
        
        handles.push(handle);
    }

    let mut all_latencies = Vec::new();
    let mut total_failures = 0usize;
    
    for handle in handles {
        if let Ok((latencies, failures)) = handle.await {
            all_latencies.extend(latencies);
            total_failures += failures;
        }
    }

    let total_time = start.elapsed();
    let total_time_ms = total_time.as_secs_f64() * 1000.0;

    if all_latencies.is_empty() {
        return Err("No successful requests".into());
    }

    all_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = all_latencies.len();
    let sum: f64 = all_latencies.iter().sum();
    let avg = sum / n as f64;
    let min = all_latencies[0];
    let max = all_latencies[n - 1];
    
    // Standard deviation
    let variance = all_latencies.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    
    // Percentiles
    let p50 = all_latencies[n / 2];
    let p75 = all_latencies[(n * 75) / 100];
    let p90 = all_latencies[(n * 90) / 100];
    let p95 = all_latencies[(n * 95) / 100];
    let p99 = all_latencies[(n * 99) / 100];
    
    let throughput = n as f64 / total_time.as_secs_f64();

    Ok(ConcurrentBenchmarkResult {
        model_name,
        concurrency_level: concurrency,
        total_requests,
        successful_requests: n,
        failed_requests: total_failures,
        total_time_ms,
        throughput_req_per_sec: throughput,
        avg_latency_ms: avg,
        min_latency_ms: min,
        max_latency_ms: max,
        p50_latency_ms: p50,
        p75_latency_ms: p75,
        p90_latency_ms: p90,
        p95_latency_ms: p95,
        p99_latency_ms: p99,
        std_dev_ms: std_dev,
    })
}

fn export_csv(results: &[ConcurrentBenchmarkResult], path: &str) {
    let mut file = File::create(path).expect("Unable to create CSV file");
    
    writeln!(file, "model,concurrency,total_requests,successful_requests,failed_requests,total_time_ms,throughput_rps,avg_latency_ms,min_latency_ms,max_latency_ms,p50_ms,p75_ms,p90_ms,p95_ms,p99_ms,std_dev_ms")
        .expect("Unable to write header");
    
    for r in results {
        writeln!(file, "{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            r.model_name,
            r.concurrency_level,
            r.total_requests,
            r.successful_requests,
            r.failed_requests,
            r.total_time_ms,
            r.throughput_req_per_sec,
            r.avg_latency_ms,
            r.min_latency_ms,
            r.max_latency_ms,
            r.p50_latency_ms,
            r.p75_latency_ms,
            r.p90_latency_ms,
            r.p95_latency_ms,
            r.p99_latency_ms,
            r.std_dev_ms
        ).expect("Unable to write row");
    }
    
    println!("✓ CSV exported: {}", path);
}

fn export_json(results: &[ConcurrentBenchmarkResult], system_info: &[(String, String)], path: &str) {
    let mut file = File::create(path).expect("Unable to create JSON file");
    
    let system_obj: serde_json::Map<String, serde_json::Value> = system_info.iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    
    let results_arr: Vec<serde_json::Value> = results.iter().map(|r| {
        serde_json::json!({
            "model": r.model_name,
            "concurrency": r.concurrency_level,
            "requests": {
                "total": r.total_requests,
                "successful": r.successful_requests,
                "failed": r.failed_requests
            },
            "timing": {
                "total_time_ms": r.total_time_ms,
                "throughput_rps": r.throughput_req_per_sec
            },
            "latency": {
                "avg_ms": r.avg_latency_ms,
                "min_ms": r.min_latency_ms,
                "max_ms": r.max_latency_ms,
                "std_dev_ms": r.std_dev_ms,
                "percentiles": {
                    "p50_ms": r.p50_latency_ms,
                    "p75_ms": r.p75_latency_ms,
                    "p90_ms": r.p90_latency_ms,
                    "p95_ms": r.p95_latency_ms,
                    "p99_ms": r.p99_latency_ms
                }
            }
        })
    }).collect();
    
    let output = serde_json::json!({
        "system": system_obj,
        "results": results_arr
    });
    
    writeln!(file, "{}", serde_json::to_string_pretty(&output).unwrap()).ok();
    println!("✓ JSON exported: {}", path);
}

fn generate_markdown_report(results: &[ConcurrentBenchmarkResult], system_info: &[(String, String)], path: &str) {
    let mut file = File::create(path).expect("Unable to create markdown file");
    
    writeln!(file, "# Benchmark Report\n").ok();
    writeln!(file, "## System Information\n").ok();
    writeln!(file, "| Property | Value |").ok();
    writeln!(file, "|----------|-------|").ok();
    for (k, v) in system_info {
        writeln!(file, "| {} | {} |", k, v).ok();
    }
    
    writeln!(file, "\n## Throughput Results\n").ok();
    writeln!(file, "| Concurrency | Throughput (req/s) | Avg Latency (ms) | P95 (ms) | P99 (ms) | Success Rate |").ok();
    writeln!(file, "|-------------|-------------------|------------------|----------|----------|--------------|").ok();
    
    for r in results {
        let success_rate = (r.successful_requests as f64 / r.total_requests as f64) * 100.0;
        writeln!(file, "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1}% |",
            r.concurrency_level,
            r.throughput_req_per_sec,
            r.avg_latency_ms,
            r.p95_latency_ms,
            r.p99_latency_ms,
            success_rate
        ).ok();
    }
    
    // Scaling analysis
    if let Some(baseline) = results.first() {
        writeln!(file, "\n## Scaling Analysis\n").ok();
        writeln!(file, "| Concurrency | Speedup | Efficiency |").ok();
        writeln!(file, "|-------------|---------|------------|").ok();
        
        for r in results {
            let speedup = r.throughput_req_per_sec / baseline.throughput_req_per_sec;
            let efficiency = (speedup / r.concurrency_level as f64) * 100.0;
            writeln!(file, "| {} | {:.2}x | {:.1}% |",
                r.concurrency_level,
                speedup,
                efficiency
            ).ok();
        }
    }
    
    // Summary
    if let (Some(first), Some(last)) = (results.first(), results.last()) {
        writeln!(file, "\n## Summary\n").ok();
        writeln!(file, "- **Model**: {}", first.model_name).ok();
        writeln!(file, "- **Baseline Throughput** (1 concurrent): {:.2} req/s", first.throughput_req_per_sec).ok();
        writeln!(file, "- **Peak Throughput** ({} concurrent): {:.2} req/s", last.concurrency_level, 
            results.iter().map(|r| r.throughput_req_per_sec).fold(0.0f64, f64::max)).ok();
        writeln!(file, "- **Best P95 Latency**: {:.2} ms", 
            results.iter().map(|r| r.p95_latency_ms).fold(f64::MAX, f64::min)).ok();
    }
    
    println!("✓ Markdown report: {}", path);
}

fn generate_throughput_plot(results: &[ConcurrentBenchmarkResult], path: &str) {
    let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_concurrency = results.iter().map(|r| r.concurrency_level).max().unwrap_or(1) as f64;
    let max_throughput = results.iter().map(|r| r.throughput_req_per_sec).fold(0.0f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Throughput vs Concurrency", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(0f64..max_concurrency * 1.1, 0f64..max_throughput * 1.2)
        .unwrap();

    chart.configure_mesh()
        .x_desc("Concurrency Level")
        .y_desc("Throughput (req/sec)")
        .draw()
        .unwrap();

    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.concurrency_level as f64, r.throughput_req_per_sec)),
        &BLUE,
    )).unwrap();

    chart.draw_series(
        results.iter().map(|r| Circle::new((r.concurrency_level as f64, r.throughput_req_per_sec), 5, BLUE.filled()))
    ).unwrap();

    root.present().unwrap();
    println!("✓ Throughput plot: {}", path);
}

fn generate_latency_plot(results: &[ConcurrentBenchmarkResult], path: &str) {
    let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_concurrency = results.iter().map(|r| r.concurrency_level).max().unwrap_or(1) as f64;
    let max_latency = results.iter().map(|r| r.p99_latency_ms).fold(0.0f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Latency Percentiles vs Concurrency", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(0f64..max_concurrency * 1.1, 0f64..max_latency * 1.2)
        .unwrap();

    chart.configure_mesh()
        .x_desc("Concurrency Level")
        .y_desc("Latency (ms)")
        .draw()
        .unwrap();

    // P50
    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.concurrency_level as f64, r.p50_latency_ms)), &GREEN,
    )).unwrap().label("P50").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    // P95
    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.concurrency_level as f64, r.p95_latency_ms)), &BLUE,
    )).unwrap().label("P95").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // P99
    chart.draw_series(LineSeries::new(
        results.iter().map(|r| (r.concurrency_level as f64, r.p99_latency_ms)), &RED,
    )).unwrap().label("P99").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw().unwrap();
    root.present().unwrap();
    println!("✓ Latency plot: {}", path);
}

fn generate_scaling_plot(results: &[ConcurrentBenchmarkResult], path: &str) {
    let root = BitMapBackend::new(path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max_concurrency = results.iter().map(|r| r.concurrency_level).max().unwrap_or(1) as f64;

    let mut chart = ChartBuilder::on(&root)
        .caption("Scaling Efficiency", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(80)
        .build_cartesian_2d(0f64..max_concurrency * 1.1, 0f64..max_concurrency * 1.2)
        .unwrap();

    chart.configure_mesh()
        .x_desc("Concurrency Level")
        .y_desc("Speedup Factor")
        .draw()
        .unwrap();

    if let Some(baseline) = results.first() {
        let baseline_throughput = baseline.throughput_req_per_sec;
        
        chart.draw_series(LineSeries::new(
            results.iter().map(|r| (r.concurrency_level as f64, r.throughput_req_per_sec / baseline_throughput)),
            &BLUE,
        )).unwrap().label("Actual").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        chart.draw_series(LineSeries::new(
            results.iter().map(|r| (r.concurrency_level as f64, r.concurrency_level as f64)),
            &RED.mix(0.5),
        )).unwrap().label("Ideal Linear").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw().unwrap();
    }

    root.present().unwrap();
    println!("✓ Scaling plot: {}", path);
}

criterion_group!(benches, benchmark_concurrent_throughput);
criterion_main!(benches);
