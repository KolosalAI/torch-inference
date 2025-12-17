use serde::{Deserialize, Serialize};
use std::sync::Arc;
use anyhow::{Result, Context};
use dashmap::DashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuBackend {
    Cuda,
    Metal,
    Cpu,
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
