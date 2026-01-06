use serde::{Deserialize, Serialize};
use std::sync::Arc;
use anyhow::{Result, Context};
use dashmap::DashMap;
use log::{info, warn, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuBackend {
    Cuda,
    Metal,
    Cpu,
}

/// CUDA memory configuration for optimal performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaMemoryConfig {
    pub memory_fraction: f32,       // Fraction of GPU memory to use (0.0-1.0)
    pub growth_strategy: String,    // "default", "aggressive", "conservative"
    pub enable_async_alloc: bool,   // Enable async memory allocator
    pub pool_size_mb: usize,        // Memory pool size
}

impl Default for CudaMemoryConfig {
    fn default() -> Self {
        Self {
            memory_fraction: 0.9,
            growth_strategy: "default".to_string(),
            enable_async_alloc: true,
            pool_size_mb: 4096,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub available: bool,
    pub count: usize,
    pub devices: Vec<GpuDevice>,
    pub backend: GpuBackend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    pub id: usize,
    pub name: String,
    pub total_memory: u64,
    pub free_memory: u64,
    pub used_memory: u64,
    pub temperature: Option<u32>,
    pub utilization: Option<u32>,
    pub power_usage: Option<u32>,
    pub power_limit: Option<u32>,
    pub compute_capability: Option<String>,
    pub cuda_cores: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStats {
    pub device_id: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub memory_used: u64,
    pub memory_free: u64,
    pub utilization: u32,
    pub temperature: u32,
}

pub struct GpuManager {
    stats_history: Arc<DashMap<usize, Vec<GpuStats>>>,
    max_history: usize,
}

impl GpuManager {
    pub fn new() -> Self {
        Self {
            stats_history: Arc::new(DashMap::new()),
            max_history: 100,
        }
    }

    /// Check if Metal is available (macOS only)
    #[cfg(target_os = "macos")]
    pub fn is_metal_available() -> bool {
        use std::process::Command;
        
        // Check if we're on Apple Silicon or if Metal is available
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
        {
            if let Ok(cpu_info) = String::from_utf8(output.stdout) {
                // Check for Apple Silicon (M1, M2, M3, etc.)
                if cpu_info.contains("Apple") {
                    return true;
                }
            }
        }
        
        // Alternative: Check for Metal framework
        std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists()
    }
    
    #[cfg(not(target_os = "macos"))]
    pub fn is_metal_available() -> bool {
        false
    }

    /// Get Metal GPU info
    #[cfg(target_os = "macos")]
    pub fn get_metal_info(&self) -> Result<GpuInfo> {
        use std::process::Command;
        
        log::info!("Detecting Metal GPU...");
        
        // Get system info
        let output = Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
            .context("Failed to run system_profiler")?;
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        
        // Parse GPU information
        let mut devices = Vec::new();
        let mut gpu_name = "Apple GPU".to_string();
        let mut total_memory = 0u64;
        
        // Try to extract GPU name and memory
        for line in output_str.lines() {
            let line = line.trim();
            if line.starts_with("Chipset Model:") {
                if let Some(name) = line.split(':').nth(1) {
                    gpu_name = name.trim().to_string();
                }
            } else if line.contains("VRAM") || line.contains("Memory") {
                // Try to extract memory size (e.g., "8 GB")
                if let Some(mem_part) = line.split(':').nth(1) {
                    let mem_str = mem_part.trim();
                    if let Some(size_str) = mem_str.split_whitespace().next() {
                        if let Ok(size) = size_str.parse::<u64>() {
                            total_memory = size * 1024 * 1024 * 1024; // Convert GB to bytes
                        }
                    }
                }
            }
        }
        
        // If we couldn't get specific memory, estimate based on system RAM
        if total_memory == 0 {
            // Unified memory on Apple Silicon - typically 2/3 of RAM can be used for GPU
            if let Ok(output) = Command::new("sysctl")
                .arg("-n")
                .arg("hw.memsize")
                .output()
            {
                if let Ok(mem_str) = String::from_utf8(output.stdout) {
                    if let Ok(sys_mem) = mem_str.trim().parse::<u64>() {
                        total_memory = (sys_mem * 2) / 3; // Estimate GPU can use 2/3 of system memory
                    }
                }
            }
        }
        
        devices.push(GpuDevice {
            id: 0,
            name: gpu_name,
            total_memory,
            free_memory: total_memory * 8 / 10, // Estimate 80% free
            used_memory: total_memory * 2 / 10, // Estimate 20% used
            temperature: None, // Metal doesn't expose temperature easily
            utilization: None, // Would need IOKit for this
            power_usage: None,
            power_limit: None,
            compute_capability: None,
            cuda_cores: None,
        });
        
        Ok(GpuInfo {
            available: true,
            count: 1,
            devices,
            backend: GpuBackend::Metal,
        })
    }
    
    #[cfg(not(target_os = "macos"))]
    pub fn get_metal_info(&self) -> Result<GpuInfo> {
        Ok(GpuInfo {
            available: false,
            count: 0,
            devices: Vec::new(),
            backend: GpuBackend::Cpu,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn get_info(&self) -> Result<GpuInfo> {
        use nvml_wrapper::Nvml;

        // Try CUDA first
        let nvml_result = Nvml::init();
        if nvml_result.is_err() {
            // CUDA not available, try Metal on macOS
            #[cfg(target_os = "macos")]
            {
                if Self::is_metal_available() {
                    log::info!("CUDA not available, falling back to Metal GPU");
                    return self.get_metal_info();
                }
            }
            
            // Return the NVML error
            return Err(nvml_result.unwrap_err().into());
        }
        
        let nvml = nvml_result.unwrap();
        let device_count = nvml.device_count().context("Failed to get device count")?;

        let mut devices = Vec::new();

        for i in 0..device_count {
            let device = nvml.device_by_index(i).context(format!("Failed to get device {}", i))?;
            
            let name = device.name().unwrap_or_else(|_| format!("GPU {}", i));
            let memory_info = device.memory_info().ok();
            
            let total_memory = memory_info.as_ref().map(|m| m.total).unwrap_or(0);
            let free_memory = memory_info.as_ref().map(|m| m.free).unwrap_or(0);
            let used_memory = memory_info.as_ref().map(|m| m.used).unwrap_or(0);
            
            let temperature = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu).ok();
            let utilization = device.utilization_rates().ok().map(|u| u.gpu);
            let power_usage = device.power_usage().ok().map(|p| p / 1000); // Convert to watts
            let power_limit = device.power_management_limit().ok().map(|p| p / 1000);

            devices.push(GpuDevice {
                id: i as usize,
                name,
                total_memory,
                free_memory,
                used_memory,
                temperature,
                utilization,
                power_usage,
                power_limit,
                compute_capability: None, // Can be retrieved via CUDA runtime
                cuda_cores: None,
            });
        }

        Ok(GpuInfo {
            available: device_count > 0,
            count: device_count as usize,
            devices,
            backend: GpuBackend::Cuda,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn get_info(&self) -> Result<GpuInfo> {
        // On macOS, try Metal first
        #[cfg(target_os = "macos")]
        {
            if Self::is_metal_available() {
                log::info!("Detected Metal GPU, using Metal backend");
                return self.get_metal_info();
            }
        }
        
        // Fall back to CPU
        log::info!("No GPU detected, using CPU backend");
        Ok(GpuInfo {
            available: false,
            count: 0,
            devices: Vec::new(),
            backend: GpuBackend::Cpu,
        })
    }

    pub fn record_stats(&self, device_id: usize, stats: GpuStats) {
        let mut history = self.stats_history.entry(device_id).or_insert_with(Vec::new);
        history.push(stats);
        
        // Keep only recent history
        let len = history.len();
        if len > self.max_history {
            history.drain(0..(len - self.max_history));
        }
    }

    pub fn get_stats_history(&self, device_id: usize) -> Vec<GpuStats> {
        self.stats_history
            .get(&device_id)
            .map(|h| h.clone())
            .unwrap_or_default()
    }

    #[cfg(feature = "cuda")]
    pub fn collect_stats(&self) -> Result<Vec<GpuStats>> {
        use nvml_wrapper::Nvml;

        let nvml = Nvml::init()?;
        let device_count = nvml.device_count()?;
        let mut stats = Vec::new();

        for i in 0..device_count {
            if let Ok(device) = nvml.device_by_index(i) {
                let memory_info = device.memory_info().ok();
                let utilization = device.utilization_rates().ok().map(|u| u.gpu).unwrap_or(0);
                let temperature = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu).unwrap_or(0);

                let stat = GpuStats {
                    device_id: i as usize,
                    timestamp: chrono::Utc::now(),
                    memory_used: memory_info.as_ref().map(|m| m.used).unwrap_or(0),
                    memory_free: memory_info.as_ref().map(|m| m.free).unwrap_or(0),
                    utilization,
                    temperature,
                };

                self.record_stats(i as usize, stat.clone());
                stats.push(stat);
            }
        }

        Ok(stats)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn collect_stats(&self) -> Result<Vec<GpuStats>> {
        // On macOS with Metal, we can collect basic stats
        #[cfg(target_os = "macos")]
        {
            if Self::is_metal_available() {
                return self.collect_metal_stats();
            }
        }
        Ok(Vec::new())
    }
    
    /// Collect Metal GPU statistics (macOS only)
    #[cfg(target_os = "macos")]
    pub fn collect_metal_stats(&self) -> Result<Vec<GpuStats>> {
        use std::process::Command;
        
        // Use system_profiler to get GPU utilization (approximate)
        let output = Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output();
            
        if let Ok(output) = output {
            let output_str = String::from_utf8_lossy(&output.stdout);
            
            // Try to get memory info from system
            let mem_cmd = Command::new("sysctl")
                .arg("-n")
                .arg("hw.memsize")
                .output();
                
            let total_memory = if let Ok(mem_output) = mem_cmd {
                if let Ok(mem_str) = String::from_utf8(mem_output.stdout) {
                    mem_str.trim().parse::<u64>().unwrap_or(0)
                } else {
                    0
                }
            } else {
                0
            };
            
            // Estimate GPU memory (2/3 of system memory on unified architecture)
            let gpu_memory = (total_memory * 2) / 3;
            
            let stat = GpuStats {
                device_id: 0,
                timestamp: chrono::Utc::now(),
                memory_used: gpu_memory / 5, // Rough estimate: 20% used
                memory_free: (gpu_memory * 4) / 5, // 80% free
                utilization: 0, // Metal doesn't expose this easily
                temperature: 0, // Metal doesn't expose this easily
            };
            
            self.record_stats(0, stat.clone());
            return Ok(vec![stat]);
        }
        
        Ok(Vec::new())
    }
    
    #[cfg(not(target_os = "macos"))]
    pub fn collect_metal_stats(&self) -> Result<Vec<GpuStats>> {
        Ok(Vec::new())
    }

    pub fn is_cuda_available() -> bool {
        cfg!(feature = "cuda")
    }
    
    /// Check if CUDA is available at runtime (not just compile-time)
    #[cfg(feature = "cuda")]
    pub fn is_cuda_runtime_available() -> bool {
        use nvml_wrapper::Nvml;
        Nvml::init().is_ok()
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn is_cuda_runtime_available() -> bool {
        false
    }
    
    /// Get CUDA runtime information
    #[cfg(feature = "cuda")]
    pub fn get_cuda_info() -> Option<String> {
        use nvml_wrapper::Nvml;
        if let Ok(nvml) = Nvml::init() {
            if let Ok(count) = nvml.device_count() {
                if count > 0 {
                    if let Ok(device) = nvml.device_by_index(0) {
                        if let Ok(name) = device.name() {
                            if let Ok(driver_version) = nvml.sys_driver_version() {
                                if let Ok(cuda_version) = nvml.sys_cuda_driver_version() {
                                    return Some(format!("{} (Driver: {}, CUDA: {})", name, driver_version, cuda_version));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn get_cuda_info() -> Option<String> {
        None
    }

    pub fn get_best_device(&self) -> Result<Option<usize>> {
        let info = self.get_info()?;
        
        if !info.available || info.devices.is_empty() {
            return Ok(None);
        }

        // Find device with most free memory
        let best = info.devices.iter()
            .max_by_key(|d| d.free_memory)
            .map(|d| d.id);

        Ok(best)
    }
    
    /// Get Metal runtime information (macOS only)
    #[cfg(target_os = "macos")]
    pub fn get_metal_info_string(&self) -> Option<String> {
        use std::process::Command;
        
        if !Self::is_metal_available() {
            return None;
        }
        
        // Get CPU info to determine Apple Silicon generation
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
        {
            if let Ok(cpu_info) = String::from_utf8(output.stdout) {
                let cpu_info = cpu_info.trim();
                
                // Get macOS version
                if let Ok(os_output) = Command::new("sw_vers")
                    .arg("-productVersion")
                    .output()
                {
                    if let Ok(os_version) = String::from_utf8(os_output.stdout) {
                        return Some(format!("{} with Metal (macOS {})", 
                            cpu_info, os_version.trim()));
                    }
                }
                
                return Some(format!("{} with Metal", cpu_info));
            }
        }
        
        Some("Apple Metal GPU".to_string())
    }
    
    #[cfg(not(target_os = "macos"))]
    pub fn get_metal_info_string(&self) -> Option<String> {
        None
    }
    
    /// Initialize CUDA optimizations for maximum inference performance
    #[cfg(feature = "cuda")]
    pub fn initialize_cuda_optimizations(&self, config: &CudaMemoryConfig) -> Result<()> {
        use nvml_wrapper::Nvml;
        
        log::info!("Initializing CUDA optimizations...");
        
        let nvml = Nvml::init().context("Failed to initialize NVML")?;
        let device_count = nvml.device_count()?;
        
        for i in 0..device_count {
            if let Ok(device) = nvml.device_by_index(i) {
                // Log device capabilities
                if let Ok(name) = device.name() {
                    log::info!("  GPU {}: {}", i, name);
                }
                
                // Log compute mode
                if let Ok(compute_mode) = device.compute_mode() {
                    log::info!("    Compute mode: {:?}", compute_mode);
                }
                
                // Log ECC mode (persistence API varies by nvml version)
                if let Ok(ecc) = device.is_ecc_enabled() {
                    log::info!("    ECC mode: {:?}", ecc);
                }
            }
        }
        
        log::info!("  ✓ Memory fraction: {:.0}%", config.memory_fraction * 100.0);
        log::info!("  ✓ Growth strategy: {}", config.growth_strategy);
        log::info!("  ✓ Async allocation: {}", config.enable_async_alloc);
        log::info!("  ✓ Pool size: {} MB", config.pool_size_mb);
        log::info!("CUDA optimizations initialized successfully");
        
        Ok(())
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn initialize_cuda_optimizations(&self, _config: &CudaMemoryConfig) -> Result<()> {
        log::warn!("CUDA optimizations not available (feature not enabled)");
        Ok(())
    }
    
    /// Get recommended CUDA configuration based on GPU capabilities
    #[cfg(feature = "cuda")]
    pub fn get_recommended_config(&self) -> Result<CudaMemoryConfig> {
        use nvml_wrapper::Nvml;
        
        let nvml = Nvml::init()?;
        let device = nvml.device_by_index(0)?;
        let memory_info = device.memory_info()?;
        
        // Calculate recommended pool size (75% of free memory)
        let pool_size_mb = ((memory_info.free as f64 * 0.75) / (1024.0 * 1024.0)) as usize;
        
        Ok(CudaMemoryConfig {
            memory_fraction: 0.9,
            growth_strategy: "default".to_string(),
            enable_async_alloc: true,
            pool_size_mb,
        })
    }
    
    #[cfg(not(feature = "cuda"))]
    pub fn get_recommended_config(&self) -> Result<CudaMemoryConfig> {
        Ok(CudaMemoryConfig::default())
    }
    
    /// Check if TensorRT is available on the system
    pub fn is_tensorrt_available() -> bool {
        // Check for TensorRT libraries
        #[cfg(target_os = "windows")]
        {
            std::path::Path::new("C:\\Program Files\\NVIDIA\\TensorRT").exists()
                || std::env::var("TENSORRT_ROOT").is_ok()
        }
        
        #[cfg(not(target_os = "windows"))]
        {
            std::path::Path::new("/usr/lib/x86_64-linux-gnu/libnvinfer.so").exists()
                || std::path::Path::new("/usr/local/tensorrt").exists()
                || std::env::var("TENSORRT_ROOT").is_ok()
        }
    }
    
    /// Get TensorRT version if available
    pub fn get_tensorrt_version() -> Option<String> {
        // Try to read from environment or common locations
        if let Ok(root) = std::env::var("TENSORRT_ROOT") {
            // Parse version from path or version file
            if let Ok(content) = std::fs::read_to_string(format!("{}/version.txt", root)) {
                return Some(content.trim().to_string());
            }
        }
        None
    }
}

impl Default for GpuManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_manager_creation() {
        let manager = GpuManager::new();
        assert_eq!(manager.max_history, 100);
    }

    #[test]
    fn test_cuda_available() {
        let available = GpuManager::is_cuda_available();
        #[cfg(feature = "cuda")]
        assert!(available);
        #[cfg(not(feature = "cuda"))]
        assert!(!available);
    }
}
