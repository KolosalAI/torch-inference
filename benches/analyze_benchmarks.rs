use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use chrono::Utc;

mod benchmark_reporter;
use benchmark_reporter::{BenchmarkResult, SystemInfo};

fn find_latest_csv(pattern: &str) -> Option<PathBuf> {
    let data_dir = PathBuf::from("benches/data");
    
    if let Ok(entries) = std::fs::read_dir(&data_dir) {
        let csv_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                let path = e.path();
                path.is_file() && 
                path.extension().and_then(|s| s.to_str()) == Some("csv") &&
                path.file_name().and_then(|s| s.to_str()).map(|s| s.contains(pattern)).unwrap_or(false)
            })
            .collect();
        
        csv_files.iter()
            .max_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()))
            .map(|e| e.path())
    } else {
        None
    }
}

fn load_csv_data(csv_file: &PathBuf) -> Result<Vec<HashMap<String, String>>, Box<dyn std::error::Error>> {
    let file = File::open(csv_file)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // Read header
    let header = lines.next().ok_or("Empty CSV file")??;
    let headers: Vec<String> = header.split(',').map(|s| s.to_string()).collect();
    
    // Read data rows
    let mut data = Vec::new();
    for line in lines {
        let line = line?;
        let values: Vec<String> = line.split(',').map(|s| s.to_string()).collect();
        
        let mut row = HashMap::new();
        for (i, header) in headers.iter().enumerate() {
            if let Some(value) = values.get(i) {
                row.insert(header.clone(), value.clone());
            }
        }
        data.push(row);
    }
    
    println!("✓ Loaded {} benchmark results from {:?}", data.len(), csv_file);
    Ok(data)
}

fn print_summary(data: &[HashMap<String, String>]) {
    if data.is_empty() {
        println!("No data to display");
        return;
    }
    
    println!("\n{:=^80}", "");
    println!("{:^80}", "BENCHMARK SUMMARY");
    println!("{:=^80}\n", "");
    
    // System info from first record
    if let Some(first_row) = data.first() {
        println!("System Information:");
        println!("  OS:           {}", first_row.get("os").unwrap_or(&"N/A".to_string()));
        println!("  CPU:          {}", first_row.get("cpu_model").unwrap_or(&"N/A".to_string()));
        println!("  CPU Cores:    {}", first_row.get("cpu_count").unwrap_or(&"N/A".to_string()));
        
        if let Some(memory) = first_row.get("total_memory_mb") {
            if let Ok(mem) = memory.parse::<f64>() {
                println!("  Total Memory: {:.2} GB", mem / 1024.0);
            }
        }
        println!("  Hostname:     {}", first_row.get("hostname").unwrap_or(&"N/A".to_string()));
        println!();
    }
    
    // Group by benchmark name
    let mut grouped: HashMap<String, Vec<&HashMap<String, String>>> = HashMap::new();
    for row in data {
        if let Some(bench_name) = row.get("benchmark_name") {
            grouped.entry(bench_name.clone()).or_insert_with(Vec::new).push(row);
        }
    }
    
    for (bench_name, rows) in grouped.iter() {
        println!("\n{:─^80}", format!(" {} ", bench_name));
        println!("{:<40} {:>12} {:>12} {:>12}", "Model/Parameter", "Mean (ms)", "Median (ms)", "Std Dev (ms)");
        println!("{:─<80}", "");
        
        // Aggregate by parameter/model
        let mut aggregated: HashMap<String, Vec<f64>> = HashMap::new();
        for row in rows {
            let key = row.get("model_name")
                .or_else(|| row.get("parameter"))
                .unwrap_or(&"N/A".to_string())
                .clone();
            
            if let Some(mean_str) = row.get("mean_time_ms") {
                if let Ok(mean_ms) = mean_str.parse::<f64>() {
                    aggregated.entry(key).or_insert_with(Vec::new).push(mean_ms);
                }
            }
        }
        
        let mut sorted_keys: Vec<_> = aggregated.keys().collect();
        sorted_keys.sort();
        
        for label in sorted_keys {
            if let Some(times) = aggregated.get(label) {
                if !times.is_empty() {
                    let avg_mean = times.iter().sum::<f64>() / times.len() as f64;
                    let min_time = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_time = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    println!("{:<40} {:>12.4} {:>12.4} {:>12.4}", label, avg_mean, min_time, max_time);
                }
            }
        }
    }
    
    println!("\n{:=^80}\n", "");
}

fn compare_models(data: &[HashMap<String, String>]) {
    println!("\nMODEL COMPARISON");
    println!("{:=^80}", "");
    
    // Filter rows with model names
    let model_data: Vec<_> = data.iter()
        .filter(|row| row.get("model_name").map(|s| !s.is_empty()).unwrap_or(false))
        .collect();
    
    if model_data.is_empty() {
        println!("No model comparisons available");
        return;
    }
    
    // Group by benchmark and model
    let mut comparison: HashMap<String, HashMap<String, Vec<f64>>> = HashMap::new();
    for row in model_data {
        let benchmark = row.get("benchmark_name").unwrap_or(&"unknown".to_string()).clone();
        let model = row.get("model_name").unwrap_or(&"unknown".to_string()).clone();
        
        if let Some(mean_str) = row.get("mean_time_ms") {
            if let Ok(mean_ms) = mean_str.parse::<f64>() {
                comparison.entry(benchmark)
                    .or_insert_with(HashMap::new)
                    .entry(model)
                    .or_insert_with(Vec::new)
                    .push(mean_ms);
            }
        }
    }
    
    for (benchmark, models) in comparison.iter() {
        println!("\n{}:", benchmark);
        let mut sorted_models: Vec<_> = models.keys().collect();
        sorted_models.sort();
        
        for model in sorted_models {
            if let Some(times) = models.get(model) {
                if !times.is_empty() {
                    let avg = times.iter().sum::<f64>() / times.len() as f64;
                    println!("  {:<40} {:>12.4} ms", model, avg);
                }
            }
        }
    }
    
    println!();
}

fn export_json_summary(data: &[HashMap<String, String>], output_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    
    let mut summary = serde_json::json!({
        "timestamp": Utc::now().to_rfc3339(),
        "total_benchmarks": data.len(),
        "benchmarks": {}
    });
    
    let mut grouped: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
    for row in data {
        let benchmark_name = row.get("benchmark_name").unwrap_or(&"unknown".to_string()).clone();
        
        let entry = serde_json::json!({
            "model": row.get("model_name"),
            "parameter": row.get("parameter"),
            "mean_time_ms": row.get("mean_time_ms").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0),
            "median_ms": row.get("median_ms").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0),
            "std_dev_ms": row.get("std_dev_ms").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0),
            "throughput_ops_per_sec": row.get("throughput_ops_per_sec").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0)
        });
        
        grouped.entry(benchmark_name).or_insert_with(Vec::new).push(entry);
    }
    
    summary["benchmarks"] = serde_json::to_value(grouped)?;
    
    let mut file = File::create(output_file)?;
    file.write_all(serde_json::to_string_pretty(&summary)?.as_bytes())?;
    
    println!("✓ Summary exported to {}", output_file);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pattern = if args.len() > 1 { &args[1] } else { "benchmark" };
    
    let csv_file = match find_latest_csv(pattern) {
        Some(file) => {
            println!("Using latest CSV file: {:?}", file);
            file
        }
        None => {
            eprintln!("No CSV files found in benches/data/");
            std::process::exit(1);
        }
    };
    
    let data = match load_csv_data(&csv_file) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading CSV: {}", e);
            std::process::exit(1);
        }
    };
    
    print_summary(&data);
    compare_models(&data);
    
    if let Err(e) = export_json_summary(&data, "benches/data/latest_summary.json") {
        eprintln!("Failed to export JSON summary: {}", e);
    }
    
    println!("\n📊 Analysis complete!");
    println!("\nYou can also:");
    println!("  - Open CSV files in Excel/LibreOffice for charts");
    println!("  - Use the JSON for custom analysis");
}
