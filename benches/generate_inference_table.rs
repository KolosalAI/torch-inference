use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
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
    
    println!("Loading: {:?}", csv_file);
    Ok(data)
}

#[derive(Default)]
struct ModelInferenceData {
    loading: Option<f64>,
    preprocessing: Option<f64>,
    inference_only: Option<f64>,
    full_pipeline: Option<f64>,
    throughput: Option<f64>,
    batch_times: HashMap<usize, f64>,
    concurrent_times: HashMap<usize, f64>,
}

fn extract_model_inference_times(data: &[HashMap<String, String>]) -> HashMap<String, ModelInferenceData> {
    let mut results: HashMap<String, ModelInferenceData> = HashMap::new();
    
    for row in data {
        let benchmark_name = row.get("benchmark_name").unwrap_or(&String::new());
        let model_name = match row.get("model_name") {
            Some(name) if !name.is_empty() => name.clone(),
            _ => continue,
        };
        let parameter = row.get("parameter").unwrap_or(&String::new());
        let mean_time_ms = row.get("mean_time_ms").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
        let throughput = row.get("throughput_ops_per_sec").and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
        
        let model_data = results.entry(model_name).or_insert_with(ModelInferenceData::default);
        
        if benchmark_name.contains("loading") {
            model_data.loading = Some(mean_time_ms);
        } else if benchmark_name.contains("preprocessing") {
            model_data.preprocessing = Some(mean_time_ms);
        } else if benchmark_name.contains("inference") && benchmark_name.contains("full") {
            model_data.full_pipeline = Some(mean_time_ms);
            model_data.throughput = Some(throughput);
        } else if benchmark_name.contains("inference") {
            model_data.inference_only = Some(mean_time_ms);
        } else if benchmark_name.contains("batch") {
            if let Some(batch_size) = parameter.strip_prefix("batch_").and_then(|s| s.parse().ok()) {
                model_data.batch_times.insert(batch_size, mean_time_ms);
            }
        } else if benchmark_name.contains("concurrent") {
            if let Some(conc_count) = parameter.strip_prefix("concurrent_").and_then(|s| s.parse().ok()) {
                model_data.concurrent_times.insert(conc_count, mean_time_ms);
            }
        }
    }
    
    results
}

fn generate_inference_table(results: &HashMap<String, ModelInferenceData>) {
    println!("\n## Model Inference Performance Comparison\n");
    println!("| Model Name | Model Load | Preprocessing | Inference Only | Full Pipeline | Throughput (ops/s) |");
    println!("|------------|------------|---------------|----------------|---------------|-------------------|");
    
    let mut sorted_models: Vec<_> = results.keys().collect();
    sorted_models.sort();
    
    for model_name in sorted_models {
        if let Some(data) = results.get(*model_name) {
            let loading = data.loading.unwrap_or(0.0);
            let preprocessing = data.preprocessing.unwrap_or(0.0);
            let inference = data.inference_only.unwrap_or(0.0);
            let full = data.full_pipeline.unwrap_or(0.0);
            let throughput = data.throughput.unwrap_or(0.0);
            
            println!("| {:<30} | {:>8.2} ms | {:>11.2} ms | {:>12.2} ms | {:>11.2} ms | {:>15.2} |",
                model_name, loading, preprocessing, inference, full, throughput);
        }
    }
    
    println!("\n");
}

fn generate_batch_scaling_table(results: &HashMap<String, ModelInferenceData>) {
    let mut has_batch_data = false;
    for data in results.values() {
        if !data.batch_times.is_empty() {
            has_batch_data = true;
            break;
        }
    }
    
    if !has_batch_data {
        return;
    }
    
    println!("\n## Batch Scaling Performance\n");
    
    // Get all batch sizes
    let mut all_batch_sizes = std::collections::HashSet::new();
    for data in results.values() {
        for size in data.batch_times.keys() {
            all_batch_sizes.insert(*size);
        }
    }
    let mut batch_sizes: Vec<_> = all_batch_sizes.into_iter().collect();
    batch_sizes.sort();
    
    // Header
    print!("| Model Name |");
    for size in &batch_sizes {
        print!(" Batch {} |", size);
    }
    println!();
    
    print!("|------------|");
    for _ in &batch_sizes {
        print!("----------|");
    }
    println!();
    
    let mut sorted_models: Vec<_> = results.keys().collect();
    sorted_models.sort();
    
    for model_name in sorted_models {
        if let Some(data) = results.get(*model_name) {
            if data.batch_times.is_empty() {
                continue;
            }
            
            print!("| {:<30} |", model_name);
            for size in &batch_sizes {
                if let Some(time_ms) = data.batch_times.get(size) {
                    print!(" {:>7.2} ms |", time_ms);
                } else {
                    print!(" {:>9} |", "-");
                }
            }
            println!();
        }
    }
    
    println!("\n");
}

fn generate_summary_statistics(results: &HashMap<String, ModelInferenceData>) {
    println!("\n## Performance Statistics Summary\n");
    
    if results.is_empty() {
        println!("No inference data available yet.\n");
        return;
    }
    
    let all_loading_times: Vec<f64> = results.values().filter_map(|v| v.loading).collect();
    let all_inference_times: Vec<f64> = results.values().filter_map(|v| v.inference_only).collect();
    let all_full_times: Vec<f64> = results.values().filter_map(|v| v.full_pipeline).collect();
    
    if !all_loading_times.is_empty() {
        let min = all_loading_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = all_loading_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let avg = all_loading_times.iter().sum::<f64>() / all_loading_times.len() as f64;
        
        println!("**Model Loading:**");
        println!("- Fastest: {:.2} ms", min);
        println!("- Slowest: {:.2} ms", max);
        println!("- Average: {:.2} ms", avg);
        println!();
    }
    
    if !all_inference_times.is_empty() {
        let min = all_inference_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = all_inference_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let avg = all_inference_times.iter().sum::<f64>() / all_inference_times.len() as f64;
        
        println!("**Inference Only:**");
        println!("- Fastest: {:.2} ms", min);
        println!("- Slowest: {:.2} ms", max);
        println!("- Average: {:.2} ms", avg);
        println!();
    }
    
    if !all_full_times.is_empty() {
        let min = all_full_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = all_full_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let avg = all_full_times.iter().sum::<f64>() / all_full_times.len() as f64;
        
        println!("**Full Pipeline:**");
        println!("- Fastest: {:.2} ms", min);
        println!("- Slowest: {:.2} ms", max);
        println!("- Average: {:.2} ms", avg);
        println!();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pattern = if args.len() > 1 { &args[1] } else { "benchmark" };
    
    let csv_file = match find_latest_csv(pattern) {
        Some(file) => file,
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
    
    let results = extract_model_inference_times(&data);
    
    if results.is_empty() {
        println!("\nNo model inference data found in the CSV files.");
        println!("This might mean:");
        println!("- Benchmarks haven't completed yet");
        println!("- No models were tested");
        println!("- Check the CSV file format");
        return;
    }
    
    println!("\nFound inference data for {} models\n", results.len());
    println!("{:=^100}", "");
    
    generate_inference_table(&results);
    generate_batch_scaling_table(&results);
    generate_summary_statistics(&results);
    
    println!("\n{:=^100}", "");
    println!("\nTotal models benchmarked: {}", results.len());
}
