use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub timestamp: String,
    pub benchmark_name: String,
    pub model_name: Option<String>,
    pub parameter: Option<String>,
    pub mean_time_ns: f64,
    pub mean_time_ms: f64,
    pub std_dev_ns: f64,
    pub median_ns: f64,
    pub min_ns: f64,
    pub max_ns: f64,
    pub sample_count: usize,
    pub iterations: u64,
    pub throughput_ops_per_sec: Option<f64>,
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub cpu_model: String,
    pub cpu_count: usize,
    pub total_memory_mb: f64,
    pub rust_version: String,
    pub hostname: String,
}

impl SystemInfo {
    pub fn new() -> Self {
        use sysinfo::System;
        
        let mut sys = System::new_all();
        sys.refresh_all();
        
        let cpu_model = sys.cpus().first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        
        let total_memory_mb = sys.total_memory() as f64 / 1024.0 / 1024.0;
        
        let hostname = System::host_name().unwrap_or_else(|| "Unknown".to_string());
        
        Self {
            os: std::env::consts::OS.to_string(),
            cpu_model,
            cpu_count: num_cpus::get(),
            total_memory_mb,
            rust_version: rustc_version().unwrap_or_else(|| "Unknown".to_string()),
            hostname,
        }
    }
}

fn rustc_version() -> Option<String> {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|s| s.trim().to_string())
}

pub struct BenchmarkReporter {
    results: Vec<BenchmarkResult>,
    output_dir: PathBuf,
    system_info: SystemInfo,
}

impl BenchmarkReporter {
    pub fn new(output_dir: impl AsRef<Path>) -> Self {
        let output_dir = output_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&output_dir).ok();
        
        Self {
            results: Vec::new(),
            output_dir,
            system_info: SystemInfo::new(),
        }
    }
    
    pub fn add_result(&mut self, mut result: BenchmarkResult) {
        result.system_info = self.system_info.clone();
        self.results.push(result);
    }
    
    pub fn save_csv(&self, filename: &str) -> std::io::Result<()> {
        let path = self.output_dir.join(filename);
        let file_exists = path.exists();
        
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        
        // Write header if file is new
        if !file_exists {
            writeln!(file, "timestamp,benchmark_name,model_name,parameter,mean_time_ms,std_dev_ms,median_ms,min_ms,max_ms,sample_count,iterations,throughput_ops_per_sec,os,cpu_model,cpu_count,total_memory_mb,hostname")?;
        }
        
        // Write results
        for result in &self.results {
            writeln!(
                file,
                "{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.2},{},{},{},{:.2},{}",
                result.timestamp,
                result.benchmark_name,
                result.model_name.as_deref().unwrap_or(""),
                result.parameter.as_deref().unwrap_or(""),
                result.mean_time_ms,
                result.std_dev_ns / 1_000_000.0,
                result.median_ns / 1_000_000.0,
                result.min_ns / 1_000_000.0,
                result.max_ns / 1_000_000.0,
                result.sample_count,
                result.iterations,
                result.throughput_ops_per_sec.unwrap_or(0.0),
                result.system_info.os,
                result.system_info.cpu_model,
                result.system_info.cpu_count,
                result.system_info.total_memory_mb,
                result.system_info.hostname,
            )?;
        }
        
        Ok(())
    }
    
    pub fn save_json(&self, filename: &str) -> std::io::Result<()> {
        let path = self.output_dir.join(filename);
        
        let json = serde_json::to_string_pretty(&self.results)?;
        let mut file = File::create(&path)?;
        file.write_all(json.as_bytes())?;
        
        Ok(())
    }
    
    pub fn save_all(&self, base_name: &str) -> std::io::Result<()> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let csv_filename = format!("{}_{}.csv", base_name, timestamp);
        let json_filename = format!("{}_{}.json", base_name, timestamp);
        
        self.save_csv(&csv_filename)?;
        self.save_json(&json_filename)?;
        
        println!("\n✅ Benchmark results saved:");
        println!("   📊 CSV:  {}", self.output_dir.join(&csv_filename).display());
        println!("   📋 JSON: {}", self.output_dir.join(&json_filename).display());
        
        Ok(())
    }
    
    pub fn print_summary(&self) {
        if self.results.is_empty() {
            println!("No benchmark results to display.");
            return;
        }
        
        println!("\n{:=^80}", "");
        println!("{:^80}", "BENCHMARK SUMMARY");
        println!("{:=^80}\n", "");
        
        println!("System Information:");
        println!("  OS:           {}", self.system_info.os);
        println!("  CPU:          {}", self.system_info.cpu_model);
        println!("  CPU Cores:    {}", self.system_info.cpu_count);
        println!("  Total Memory: {:.2} GB", self.system_info.total_memory_mb / 1024.0);
        println!("  Hostname:     {}", self.system_info.hostname);
        println!("  Rust:         {}", self.system_info.rust_version);
        println!();
        
        // Group by benchmark name
        let mut grouped: std::collections::HashMap<String, Vec<&BenchmarkResult>> = std::collections::HashMap::new();
        for result in &self.results {
            grouped.entry(result.benchmark_name.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        for (bench_name, results) in grouped.iter() {
            println!("\n{:─^80}", format!(" {} ", bench_name));
            println!("{:<35} {:>12} {:>12} {:>12}", "Model/Parameter", "Mean (ms)", "Median (ms)", "Std Dev (ms)");
            println!("{:─<80}", "");
            
            for result in results {
                let label = result.model_name.as_deref()
                    .or(result.parameter.as_deref())
                    .unwrap_or("N/A");
                
                println!(
                    "{:<35} {:>12.4} {:>12.4} {:>12.4}",
                    label,
                    result.mean_time_ms,
                    result.median_ns / 1_000_000.0,
                    result.std_dev_ns / 1_000_000.0,
                );
            }
        }
        
        println!("\n{:=^80}\n", "");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_reporter() {
        let reporter = BenchmarkReporter::new("target/test_benchmarks");
        assert!(reporter.results.is_empty());
    }
    
    #[test]
    fn test_system_info() {
        let info = SystemInfo::new();
        assert!(!info.os.is_empty());
        assert!(info.cpu_count > 0);
        assert!(info.total_memory_mb > 0.0);
    }
}
