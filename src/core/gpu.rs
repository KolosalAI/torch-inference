use serde::{Deserialize, Serialize};
use std::sync::Arc;
use anyhow::{Result, Context};
use dashmap::DashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub available: bool,
    pub count: usize,
    pub devices: Vec<GpuDevice>,
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

    #[cfg(feature = "cuda")]
    pub fn get_info(&self) -> Result<GpuInfo> {
        use nvml_wrapper::Nvml;

        let nvml = Nvml::init().context("Failed to initialize NVML")?;
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
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn get_info(&self) -> Result<GpuInfo> {
        Ok(GpuInfo {
            available: false,
            count: 0,
            devices: Vec::new(),
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
