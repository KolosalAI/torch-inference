use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

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
    
    let header = lines.next().ok_or("Empty CSV file")??;
    let headers: Vec<String> = header.split(',').map(|s| s.to_string()).collect();
    
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

fn extract_batch_size(param: &str) -> Option<usize> {
    if param.starts_with("batch_") {
        param.strip_prefix("batch_").and_then(|s| s.parse().ok())
    } else if param.starts_with("concurrent_") {
        param.strip_prefix("concurrent_").and_then(|s| s.parse().ok())
    } else {
        None
    }
}

fn analyze_batch_scaling(data: &[HashMap<String, String>]) {
    println!("\n{:=^80}", "");
    println!("{:^80}", "BATCH INFERENCE SCALING ANALYSIS");
    println!("{:=^80}\n", "");
    
    let batch_data: Vec<_> = data.iter()
        .filter(|row| row.get("benchmark_name").map(|s| s == "batch_inference_scaling").unwrap_or(false))
        .collect();
    
    if batch_data.is_empty() {
        println!("No batch inference data found");
        return;
    }
    
    // Group by model
    let mut by_model: HashMap<String, Vec<(usize, f64, f64)>> = HashMap::new();
    for row in batch_data {
        let model = row.get("model_name").unwrap_or(&"unknown".to_string()).clone();
        let param = row.get("parameter").unwrap_or(&"".to_string());
        
        if let Some(batch_size) = extract_batch_size(param) {
            if let (Some(mean_str), Some(throughput_str)) = (row.get("mean_time_ms"), row.get("throughput_ops_per_sec")) {
                if let (Ok(mean_ms), Ok(throughput)) = (mean_str.parse::<f64>(), throughput_str.parse::<f64>()) {
                    by_model.entry(model).or_insert_with(Vec::new).push((batch_size, mean_ms, throughput));
                }
            }
        }
    }
    
    for (model, mut results) in by_model {
        println!("\nModel: {}", model);
        println!("{:<15} {:>18} {:>22} {:>20}", "Batch Size", "Mean Time (ms)", "Throughput (ops/s)", "Scaling Efficiency");
        println!("{:─<80}", "");
        
        results.sort_by_key(|(size, _, _)| *size);
        
        let mut baseline_time = None;
        let mut baseline_size = None;
        
        for (batch_size, mean_ms, throughput) in &results {
            let efficiency = if let (Some(bt), Some(bs)) = (baseline_time, baseline_size) {
                let expected = bt * (*batch_size as f64 / bs as f64);
                (expected / mean_ms) * 100.0
            } else {
                baseline_time = Some(*mean_ms);
                baseline_size = Some(*batch_size as f64);
                100.0
            };
            
            println!("{:<15} {:>18.4} {:>22.2} {:>19.1}%", batch_size, mean_ms, throughput, efficiency);
        }
        
        // Calculate average efficiency
        if results.len() > 1 {
            let bt = baseline_time.unwrap();
            let bs = baseline_size.unwrap();
            
            let mut total_efficiency = 0.0;
            let mut count = 0;
            
            for (batch_size, mean_ms, _) in results.iter().skip(1) {
                let expected = bt * (*batch_size as f64 / bs);
                let eff = (expected / mean_ms) * 100.0;
                total_efficiency += eff;
                count += 1;
            }
            
            if count > 0 {
                let avg_efficiency = total_efficiency / count as f64;
                println!("\nAverage Scaling Efficiency: {:.1}%", avg_efficiency);
                
                let best = results.iter().max_by(|(_, _, t1), (_, _, t2)| t1.partial_cmp(t2).unwrap());
                if let Some((size, _, _)) = best {
                    println!("Best Batch Size: {} (highest throughput)", size);
                }
            }
        }
    }
}

fn analyze_concurrent_scaling(data: &[HashMap<String, String>]) {
    println!("\n{:=^80}", "");
    println!("{:^80}", "CONCURRENT REQUEST SCALING ANALYSIS");
    println!("{:=^80}\n", "");
    
    let concurrent_data: Vec<_> = data.iter()
        .filter(|row| row.get("benchmark_name").map(|s| s == "concurrent_inference_scaling").unwrap_or(false))
        .collect();
    
    if concurrent_data.is_empty() {
        println!("No concurrent inference data found");
        return;
    }
    
    let mut by_model: HashMap<String, Vec<(usize, f64, f64)>> = HashMap::new();
    for row in concurrent_data {
        let model = row.get("model_name").unwrap_or(&"unknown".to_string()).clone();
        let param = row.get("parameter").unwrap_or(&"".to_string());
        
        if let Some(concurrent_count) = extract_batch_size(param) {
            if let (Some(mean_str), Some(throughput_str)) = (row.get("mean_time_ms"), row.get("throughput_ops_per_sec")) {
                if let (Ok(mean_ms), Ok(throughput)) = (mean_str.parse::<f64>(), throughput_str.parse::<f64>()) {
                    by_model.entry(model).or_insert_with(Vec::new).push((concurrent_count, mean_ms, throughput));
                }
            }
        }
    }
    
    for (model, mut results) in by_model {
        println!("\nModel: {}", model);
        println!("{:<15} {:>18} {:>22} {:>20}", "Concurrent Reqs", "Mean Time (ms)", "Throughput (ops/s)", "Parallel Efficiency");
        println!("{:─<80}", "");
        
        results.sort_by_key(|(count, _, _)| *count);
        
        let baseline_time = results.first().map(|(_, time, _)| *time);
        
        for (concurrent_count, mean_ms, throughput) in &results {
            let efficiency = if let Some(bt) = baseline_time {
                (bt / mean_ms) * 100.0
            } else {
                100.0
            };
            
            println!("{:<15} {:>18.4} {:>22.2} {:>19.1}%", concurrent_count, mean_ms, throughput, efficiency);
        }
        
        if results.len() > 1 {
            let bt = baseline_time.unwrap();
            let mut total_efficiency = 0.0;
            let mut count = 0;
            
            for (_, mean_ms, _) in results.iter().skip(1) {
                let eff = (bt / mean_ms) * 100.0;
                total_efficiency += eff;
                count += 1;
            }
            
            if count > 0 {
                let avg_efficiency = total_efficiency / count as f64;
                println!("\nAverage Parallel Efficiency: {:.1}%", avg_efficiency);
                
                let best = results.iter().max_by(|(_, _, t1), (_, _, t2)| t1.partial_cmp(t2).unwrap());
                if let Some((count, _, _)) = best {
                    println!("Best Concurrency: {} (highest throughput)", count);
                }
            }
        }
    }
}

fn generate_summary(data: &[HashMap<String, String>]) {
    println!("\n{:=^80}", "");
    println!("{:^80}", "SCALING SUMMARY");
    println!("{:=^80}\n", "");
    
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
        println!();
    }
    
    let batch_count = data.iter().filter(|r| r.get("benchmark_name").map(|s| s == "batch_inference_scaling").unwrap_or(false)).count();
    let concurrent_count = data.iter().filter(|r| r.get("benchmark_name").map(|s| s == "concurrent_inference_scaling").unwrap_or(false)).count();
    
    println!("Total Benchmarks: {}", data.len());
    println!("  Batch Inference Tests:      {}", batch_count);
    println!("  Concurrent Request Tests:   {}", concurrent_count);
    
    let models: std::collections::HashSet<_> = data.iter()
        .filter_map(|r| r.get("model_name"))
        .filter(|s| !s.is_empty())
        .collect();
    
    println!("\nModels Tested: {}", models.len());
    let mut model_list: Vec<_> = models.into_iter().collect();
    model_list.sort();
    for model in model_list {
        println!("  - {}", model);
    }
}

fn export_scaling_report(data: &[HashMap<String, String>], output_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = serde_json::json!({
        "timestamp": data.first().and_then(|r| r.get("timestamp")).cloned(),
        "system_info": {
            "os": data.first().and_then(|r| r.get("os")).cloned(),
            "cpu_model": data.first().and_then(|r| r.get("cpu_model")).cloned(),
            "cpu_count": data.first().and_then(|r| r.get("cpu_count")).cloned(),
        },
        "batch_scaling": {},
        "concurrent_scaling": {}
    });
    
    // Process batch data
    let mut batch_by_model: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
    for row in data.iter().filter(|r| r.get("benchmark_name").map(|s| s == "batch_inference_scaling").unwrap_or(false)) {
        let model = row.get("model_name").unwrap_or(&"unknown".to_string()).clone();
        let param = row.get("parameter").unwrap_or(&"".to_string());
        
        if let Some(batch_size) = extract_batch_size(param) {
            let entry = serde_json::json!({
                "batch_size": batch_size,
                "mean_time_ms": row.get("mean_time_ms").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0),
                "throughput_ops_per_sec": row.get("throughput_ops_per_sec").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0)
            });
            batch_by_model.entry(model).or_insert_with(Vec::new).push(entry);
        }
    }
    
    // Sort each model's results by batch size
    for results in batch_by_model.values_mut() {
        results.sort_by_key(|v| v["batch_size"].as_i64().unwrap_or(0));
    }
    
    report["batch_scaling"] = serde_json::to_value(batch_by_model)?;
    
    // Process concurrent data
    let mut concurrent_by_model: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
    for row in data.iter().filter(|r| r.get("benchmark_name").map(|s| s == "concurrent_inference_scaling").unwrap_or(false)) {
        let model = row.get("model_name").unwrap_or(&"unknown".to_string()).clone();
        let param = row.get("parameter").unwrap_or(&"".to_string());
        
        if let Some(concurrent_count) = extract_batch_size(param) {
            let entry = serde_json::json!({
                "concurrent_count": concurrent_count,
                "mean_time_ms": row.get("mean_time_ms").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0),
                "throughput_ops_per_sec": row.get("throughput_ops_per_sec").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0)
            });
            concurrent_by_model.entry(model).or_insert_with(Vec::new).push(entry);
        }
    }
    
    for results in concurrent_by_model.values_mut() {
        results.sort_by_key(|v| v["concurrent_count"].as_i64().unwrap_or(0));
    }
    
    report["concurrent_scaling"] = serde_json::to_value(concurrent_by_model)?;
    
    let mut file = File::create(output_file)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    
    println!("\n✓ Scaling analysis exported to {}", output_file);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pattern = if args.len() > 1 { &args[1] } else { "batch_concurrent_benchmark" };
    
    let csv_file = match find_latest_csv(pattern) {
        Some(file) => {
            println!("Using latest CSV file: {:?}", file);
            file
        }
        None => {
            eprintln!("No CSV files found matching '{}' in benches/data/", pattern);
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
    
    generate_summary(&data);
    analyze_batch_scaling(&data);
    analyze_concurrent_scaling(&data);
    
    if let Err(e) = export_scaling_report(&data, "benches/data/scaling_analysis.json") {
        eprintln!("Failed to export scaling report: {}", e);
    }
    
    println!("\n{:=^80}", "");
    println!("\n📊 Scaling analysis complete!");
    println!("\nInterpretation:");
    println!("  - Scaling Efficiency: 100% = perfect linear scaling");
    println!("  - Parallel Efficiency: 100% = no overhead from concurrency");
    println!("  - Throughput: Higher is better");
    println!("\nNext steps:");
    println!("  - Use scaling_analysis.json for custom visualizations");
    println!("  - Compare across different hardware/configurations");
    println!("  - Identify optimal batch size and concurrency levels");
}
