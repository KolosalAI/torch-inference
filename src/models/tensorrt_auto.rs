//! TensorRT Auto-Integration Module
//! 
//! Provides automatic detection and configuration of TensorRT for optimal inference performance.
//! Supports automatic fallback to CUDA or CPU when TensorRT is not available.

use anyhow::{Result, Context};
use log::{info, warn, debug};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use crate::tensor_pool::TensorPool;

/// TensorRT availability status
#[derive(Debug, Clone, PartialEq)]
pub enum TensorRTStatus {
    /// TensorRT is available and ready
    Available { version: String, cuda_version: String },
    /// TensorRT not installed but CUDA is available
    CudaOnly { cuda_version: String },
    /// Only CPU available
    CpuOnly,
    /// Status not yet checked
    Unknown,
}

/// Auto-configured execution provider with fallback chain
#[derive(Debug, Clone)]
pub struct AutoExecutionProvider {
    pub primary: ExecutionBackend,
    pub fallback_chain: Vec<ExecutionBackend>,
    pub config: AutoTensorRTConfig,
}

/// Execution backend options
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionBackend {
    TensorRT { device_id: usize, precision: TensorRTPrecision },
    Cuda { device_id: usize },
    CoreML,
    DirectML,
    Cpu,
}

/// TensorRT precision modes
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum TensorRTPrecision {
    FP32,
    FP16,
    INT8,
}

impl TensorRTPrecision {
    pub fn as_str(&self) -> &str {
        match self {
            TensorRTPrecision::FP32 => "fp32",
            TensorRTPrecision::FP16 => "fp16",
            TensorRTPrecision::INT8 => "int8",
        }
    }
    
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "fp32" | "float32" => TensorRTPrecision::FP32,
            "fp16" | "float16" | "half" => TensorRTPrecision::FP16,
            "int8" | "int" => TensorRTPrecision::INT8,
            _ => TensorRTPrecision::FP16, // Default to FP16 for best balance
        }
    }
}

/// Auto-configuration for TensorRT
#[derive(Debug, Clone)]
pub struct AutoTensorRTConfig {
    /// Enable automatic TensorRT detection
    pub auto_detect: bool,
    /// Preferred precision (will fallback if not supported)
    pub preferred_precision: TensorRTPrecision,
    /// TensorRT workspace size in MB
    pub workspace_size_mb: usize,
    /// Maximum batch size for optimization
    pub max_batch_size: usize,
    /// Engine cache directory
    pub cache_dir: PathBuf,
    /// Enable timing cache for faster optimization
    pub use_timing_cache: bool,
    /// Enable engine serialization/caching
    pub use_engine_cache: bool,
    /// Optimization level (0-5, higher = more optimization time)
    pub optimization_level: u32,
    /// Enable dynamic shapes
    pub use_dynamic_shapes: bool,
    /// Minimum batch size for dynamic batching
    pub min_batch_size: usize,
    /// Optimal batch size for dynamic batching
    pub optimal_batch_size: usize,
}

impl Default for AutoTensorRTConfig {
    fn default() -> Self {
        Self {
            auto_detect: true,
            preferred_precision: TensorRTPrecision::FP16,
            workspace_size_mb: 2048, // 2GB default
            max_batch_size: 32,
            cache_dir: PathBuf::from("./tensorrt_cache"),
            use_timing_cache: true,
            use_engine_cache: true,
            optimization_level: 3,
            use_dynamic_shapes: true,
            min_batch_size: 1,
            optimal_batch_size: 8,
        }
    }
}

/// TensorRT Auto-Integration Manager
pub struct TensorRTAutoManager {
    status: TensorRTStatus,
    config: AutoTensorRTConfig,
    execution_provider: Option<AutoExecutionProvider>,
}

impl TensorRTAutoManager {
    /// Create a new auto-manager with automatic detection
    pub fn new() -> Self {
        let mut manager = Self {
            status: TensorRTStatus::Unknown,
            config: AutoTensorRTConfig::default(),
            execution_provider: None,
        };
        manager.detect_and_configure();
        manager
    }
    
    /// Create with custom configuration
    pub fn with_config(config: AutoTensorRTConfig) -> Self {
        let mut manager = Self {
            status: TensorRTStatus::Unknown,
            config,
            execution_provider: None,
        };
        manager.detect_and_configure();
        manager
    }
    
    /// Detect available backends and auto-configure
    pub fn detect_and_configure(&mut self) {
        info!("Auto-detecting TensorRT and CUDA availability...");
        
        // Check TensorRT
        let tensorrt_available = Self::check_tensorrt();
        let cuda_available = Self::check_cuda();
        let tensorrt_version = Self::get_tensorrt_version();
        let cuda_version = Self::get_cuda_version();
        
        // Update status
        self.status = if tensorrt_available {
            TensorRTStatus::Available {
                version: tensorrt_version.unwrap_or_else(|| "unknown".to_string()),
                cuda_version: cuda_version.clone().unwrap_or_else(|| "unknown".to_string()),
            }
        } else if cuda_available {
            TensorRTStatus::CudaOnly {
                cuda_version: cuda_version.clone().unwrap_or_else(|| "unknown".to_string()),
            }
        } else {
            TensorRTStatus::CpuOnly
        };
        
        // Log detection results
        match &self.status {
            TensorRTStatus::Available { version, cuda_version } => {
                info!("✓ TensorRT {} detected (CUDA {})", version, cuda_version);
                info!("  → Using TensorRT for maximum performance");
            }
            TensorRTStatus::CudaOnly { cuda_version } => {
                info!("✓ CUDA {} detected (TensorRT not found)", cuda_version);
                info!("  → Using CUDA execution provider");
                info!("  → Install TensorRT for better performance");
            }
            TensorRTStatus::CpuOnly => {
                info!("✓ CPU-only mode (no GPU acceleration)");
            }
            TensorRTStatus::Unknown => {
                warn!("Detection incomplete");
            }
        }
        
        // Build execution provider with fallback chain
        self.execution_provider = Some(self.build_execution_provider());
    }
    
    /// Build execution provider with appropriate fallback chain
    fn build_execution_provider(&self) -> AutoExecutionProvider {
        let (primary, fallback_chain) = match &self.status {
            TensorRTStatus::Available { .. } => {
                let primary = ExecutionBackend::TensorRT {
                    device_id: 0,
                    precision: self.config.preferred_precision,
                };
                let fallback = vec![
                    ExecutionBackend::Cuda { device_id: 0 },
                    ExecutionBackend::Cpu,
                ];
                (primary, fallback)
            }
            TensorRTStatus::CudaOnly { .. } => {
                let primary = ExecutionBackend::Cuda { device_id: 0 };
                let fallback = vec![ExecutionBackend::Cpu];
                (primary, fallback)
            }
            TensorRTStatus::CpuOnly | TensorRTStatus::Unknown => {
                // On macOS, try CoreML first
                #[cfg(target_os = "macos")]
                {
                    let primary = ExecutionBackend::CoreML;
                    let fallback = vec![ExecutionBackend::Cpu];
                    (primary, fallback)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let primary = ExecutionBackend::Cpu;
                    let fallback = vec![];
                    (primary, fallback)
                }
            }
        };
        
        AutoExecutionProvider {
            primary,
            fallback_chain,
            config: self.config.clone(),
        }
    }
    
    /// Check if TensorRT is available
    fn check_tensorrt() -> bool {
        // Check for TensorRT libraries
        #[cfg(target_os = "windows")]
        {
            // Check common Windows TensorRT locations
            let paths = vec![
                "C:\\Program Files\\NVIDIA\\TensorRT",
                "C:\\TensorRT",
            ];
            
            // Also check environment variable
            if std::env::var("TENSORRT_ROOT").is_ok() {
                return true;
            }
            
            for path in paths {
                if Path::new(path).exists() {
                    return true;
                }
            }
            
            // Check if nvinfer.dll exists in PATH
            if let Ok(path) = std::env::var("PATH") {
                for dir in path.split(';') {
                    if Path::new(dir).join("nvinfer.dll").exists() {
                        return true;
                    }
                }
            }
            
            false
        }
        
        #[cfg(target_os = "linux")]
        {
            let paths = vec![
                "/usr/lib/x86_64-linux-gnu/libnvinfer.so",
                "/usr/local/tensorrt/lib/libnvinfer.so",
                "/opt/tensorrt/lib/libnvinfer.so",
            ];
            
            if std::env::var("TENSORRT_ROOT").is_ok() {
                return true;
            }
            
            for path in paths {
                if Path::new(path).exists() {
                    return true;
                }
            }
            
            false
        }
        
        #[cfg(target_os = "macos")]
        {
            false // TensorRT not available on macOS
        }
    }
    
    /// Check if CUDA is available
    fn check_cuda() -> bool {
        #[cfg(target_os = "windows")]
        {
            // Check CUDA environment
            if std::env::var("CUDA_PATH").is_ok() {
                return true;
            }
            
            // Check common CUDA location
            if Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA").exists() {
                return true;
            }
            
            // Check for nvml.dll (NVIDIA Management Library)
            if let Ok(path) = std::env::var("PATH") {
                for dir in path.split(';') {
                    if Path::new(dir).join("nvml.dll").exists() {
                        return true;
                    }
                }
            }
            
            false
        }
        
        #[cfg(target_os = "linux")]
        {
            Path::new("/usr/local/cuda").exists()
                || std::env::var("CUDA_PATH").is_ok()
                || Path::new("/usr/lib/x86_64-linux-gnu/libcudart.so").exists()
        }
        
        #[cfg(target_os = "macos")]
        {
            false // CUDA not available on macOS
        }
    }
    
    /// Get TensorRT version
    fn get_tensorrt_version() -> Option<String> {
        if let Ok(root) = std::env::var("TENSORRT_ROOT") {
            let version_file = Path::new(&root).join("version.txt");
            if let Ok(content) = std::fs::read_to_string(version_file) {
                return Some(content.trim().to_string());
            }
            
            // Try to extract version from path
            if root.contains("TensorRT-") {
                if let Some(version) = root.split("TensorRT-").nth(1) {
                    return Some(version.split(['/', '\\']).next().unwrap_or("").to_string());
                }
            }
        }
        
        // Check for version in common locations
        #[cfg(target_os = "windows")]
        {
            let base_path = Path::new("C:\\Program Files\\NVIDIA\\TensorRT");
            if base_path.exists() {
                if let Ok(entries) = std::fs::read_dir(base_path) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if name.starts_with("TensorRT-") {
                            return Some(name.trim_start_matches("TensorRT-").to_string());
                        }
                    }
                }
            }
        }
        
        None
    }
    
    /// Get CUDA version
    fn get_cuda_version() -> Option<String> {
        // Check CUDA_VERSION environment variable
        if let Ok(version) = std::env::var("CUDA_VERSION") {
            return Some(version);
        }
        
        #[cfg(target_os = "windows")]
        {
            // Check CUDA path for version
            if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                // Path typically ends with version like "v11.8"
                if let Some(version) = cuda_path.split(['/', '\\']).last() {
                    if version.starts_with('v') {
                        return Some(version.trim_start_matches('v').to_string());
                    }
                }
            }
            
            // Check common CUDA location
            let cuda_base = Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
            if cuda_base.exists() {
                if let Ok(entries) = std::fs::read_dir(cuda_base) {
                    for entry in entries.filter_map(|e| e.ok()) {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if name.starts_with('v') {
                            return Some(name.trim_start_matches('v').to_string());
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            // Check /usr/local/cuda/version.txt
            if let Ok(content) = std::fs::read_to_string("/usr/local/cuda/version.txt") {
                // Format: "CUDA Version 11.8.0"
                if let Some(version) = content.split_whitespace().last() {
                    return Some(version.to_string());
                }
            }
        }
        
        None
    }
    
    /// Get current status
    pub fn status(&self) -> &TensorRTStatus {
        &self.status
    }
    
    /// Get configured execution provider
    pub fn execution_provider(&self) -> Option<&AutoExecutionProvider> {
        self.execution_provider.as_ref()
    }
    
    /// Get configuration
    pub fn config(&self) -> &AutoTensorRTConfig {
        &self.config
    }
    
    /// Update configuration and reconfigure
    pub fn update_config(&mut self, config: AutoTensorRTConfig) {
        self.config = config;
        self.execution_provider = Some(self.build_execution_provider());
    }
    
    /// Create engine cache directory if needed
    pub fn ensure_cache_dir(&self) -> Result<()> {
        if !self.config.cache_dir.exists() {
            std::fs::create_dir_all(&self.config.cache_dir)
                .context("Failed to create TensorRT cache directory")?;
            info!("Created TensorRT cache directory: {:?}", self.config.cache_dir);
        }
        Ok(())
    }
    
    /// Get recommended precision based on hardware
    pub fn recommended_precision(&self) -> TensorRTPrecision {
        match &self.status {
            TensorRTStatus::Available { .. } => {
                // TensorRT supports all precisions, recommend FP16 for best balance
                TensorRTPrecision::FP16
            }
            TensorRTStatus::CudaOnly { cuda_version } => {
                // Check CUDA version for FP16 support (requires compute capability 5.3+)
                // Most modern GPUs support FP16
                if cuda_version.starts_with("10") || cuda_version.starts_with("11") || cuda_version.starts_with("12") {
                    TensorRTPrecision::FP16
                } else {
                    TensorRTPrecision::FP32
                }
            }
            _ => TensorRTPrecision::FP32,
        }
    }
    
    /// Get execution provider summary string
    pub fn get_summary(&self) -> String {
        let mut summary = String::new();
        
        summary.push_str("TensorRT Auto-Integration Status:\n");
        summary.push_str("═══════════════════════════════════\n");
        
        match &self.status {
            TensorRTStatus::Available { version, cuda_version } => {
                summary.push_str(&format!("  Status: TensorRT Available\n"));
                summary.push_str(&format!("  TensorRT Version: {}\n", version));
                summary.push_str(&format!("  CUDA Version: {}\n", cuda_version));
                summary.push_str(&format!("  Precision: {:?}\n", self.config.preferred_precision));
                summary.push_str(&format!("  Workspace: {} MB\n", self.config.workspace_size_mb));
                summary.push_str(&format!("  Max Batch: {}\n", self.config.max_batch_size));
            }
            TensorRTStatus::CudaOnly { cuda_version } => {
                summary.push_str(&format!("  Status: CUDA Only (TensorRT not installed)\n"));
                summary.push_str(&format!("  CUDA Version: {}\n", cuda_version));
                summary.push_str("  Tip: Install TensorRT for 2-5x better performance\n");
            }
            TensorRTStatus::CpuOnly => {
                summary.push_str("  Status: CPU Only\n");
                summary.push_str("  Tip: Install CUDA for GPU acceleration\n");
            }
            TensorRTStatus::Unknown => {
                summary.push_str("  Status: Not Detected\n");
            }
        }
        
        if let Some(ep) = &self.execution_provider {
            summary.push_str(&format!("\n  Primary Backend: {:?}\n", ep.primary));
            if !ep.fallback_chain.is_empty() {
                summary.push_str(&format!("  Fallback Chain: {:?}\n", ep.fallback_chain));
            }
        }
        
        summary
    }
}

impl Default for TensorRTAutoManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for AutoTensorRTConfig
pub struct AutoTensorRTConfigBuilder {
    config: AutoTensorRTConfig,
}

impl AutoTensorRTConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: AutoTensorRTConfig::default(),
        }
    }
    
    pub fn auto_detect(mut self, enable: bool) -> Self {
        self.config.auto_detect = enable;
        self
    }
    
    pub fn precision(mut self, precision: TensorRTPrecision) -> Self {
        self.config.preferred_precision = precision;
        self
    }
    
    pub fn workspace_size_mb(mut self, size: usize) -> Self {
        self.config.workspace_size_mb = size;
        self
    }
    
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }
    
    pub fn cache_dir<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.config.cache_dir = path.as_ref().to_path_buf();
        self
    }
    
    pub fn use_timing_cache(mut self, enable: bool) -> Self {
        self.config.use_timing_cache = enable;
        self
    }
    
    pub fn use_engine_cache(mut self, enable: bool) -> Self {
        self.config.use_engine_cache = enable;
        self
    }
    
    pub fn optimization_level(mut self, level: u32) -> Self {
        self.config.optimization_level = level.min(5);
        self
    }
    
    pub fn dynamic_shapes(mut self, enable: bool) -> Self {
        self.config.use_dynamic_shapes = enable;
        self
    }
    
    pub fn batch_sizes(mut self, min: usize, optimal: usize, max: usize) -> Self {
        self.config.min_batch_size = min;
        self.config.optimal_batch_size = optimal;
        self.config.max_batch_size = max;
        self
    }
    
    pub fn build(self) -> AutoTensorRTConfig {
        self.config
    }
}

impl Default for AutoTensorRTConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensorrt_auto_manager_creation() {
        let manager = TensorRTAutoManager::new();
        // Should not panic and should set a valid status
        assert_ne!(manager.status(), &TensorRTStatus::Unknown);
    }
    
    #[test]
    fn test_config_builder() {
        let config = AutoTensorRTConfigBuilder::new()
            .precision(TensorRTPrecision::INT8)
            .workspace_size_mb(4096)
            .max_batch_size(64)
            .optimization_level(5)
            .build();
        
        assert_eq!(config.preferred_precision, TensorRTPrecision::INT8);
        assert_eq!(config.workspace_size_mb, 4096);
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.optimization_level, 5);
    }
    
    #[test]
    fn test_precision_from_str() {
        assert_eq!(TensorRTPrecision::from_str("fp32"), TensorRTPrecision::FP32);
        assert_eq!(TensorRTPrecision::from_str("FP16"), TensorRTPrecision::FP16);
        assert_eq!(TensorRTPrecision::from_str("int8"), TensorRTPrecision::INT8);
        assert_eq!(TensorRTPrecision::from_str("half"), TensorRTPrecision::FP16);
    }
}
