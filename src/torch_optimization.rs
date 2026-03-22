use tch::{CModule, Device, Tensor, Kind};
use log::{info, debug, warn};
use std::sync::Arc;
use std::time::Instant;

/// PyTorch inference optimization configuration
#[derive(Debug, Clone)]
pub struct TorchOptimizationConfig {
    /// Number of intra-op threads (default: num_cpus)
    pub num_threads: usize,
    
    /// Number of inter-op threads (default: 1)
    pub num_interop_threads: usize,
    
    /// Enable cuDNN benchmark for optimal conv algorithms
    pub cudnn_benchmark: bool,
    
    /// Enable mixed precision (AMP)
    pub enable_autocast: bool,
    
    /// Warmup iterations before real inference
    pub warmup_iterations: usize,
    
    /// Input shape for warmup (batch, channels, height, width)
    pub warmup_shape: Vec<i64>,
    
    /// Enable inference mode (maximum performance)
    pub enable_inference_mode: bool,
    
    /// Device to use
    pub device: Device,
}

impl Default for TorchOptimizationConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            num_interop_threads: 1,
            cudnn_benchmark: true,
            enable_autocast: false,
            warmup_iterations: 5,
            warmup_shape: vec![1, 3, 224, 224],
            enable_inference_mode: true,
            device: Device::cuda_if_available(),
        }
    }
}

/// Optimized PyTorch model wrapper
pub struct OptimizedTorchModel {
    model: CModule,
    config: TorchOptimizationConfig,
    device: Device,
    warmed_up: bool,
}

impl OptimizedTorchModel {
    /// Create a new optimized model
    pub fn new(model_path: &str, config: TorchOptimizationConfig) -> Result<Self, String> {
        info!("Loading optimized PyTorch model from: {}", model_path);
        
        // Set thread configuration BEFORE loading model
        Self::configure_threads(&config);
        
        // Configure cuDNN
        Self::configure_cudnn(&config);
        
        // Load model
        let device = config.device;
        let mut model = CModule::load_on_device(model_path, device)
            .map_err(|e| format!("Failed to load model: {}", e))?;
        
        // Set to evaluation mode
        model.set_eval();
        info!("Model set to evaluation mode");
        
        Ok(Self {
            model,
            config,
            device,
            warmed_up: false,
        })
    }
    
    /// Configure CPU threading
    fn configure_threads(config: &TorchOptimizationConfig) {
        info!("Configuring PyTorch threading:");
        info!("  Intra-op threads: {}", config.num_threads);
        info!("  Inter-op threads: {}", config.num_interop_threads);
        
        tch::set_num_threads(config.num_threads as i32);
        tch::set_num_interop_threads(config.num_interop_threads as i32);
    }
    
    /// Configure cuDNN
    fn configure_cudnn(config: &TorchOptimizationConfig) {
        #[cfg(feature = "cuda")]
        {
            if tch::Cuda::is_available() {
                if config.cudnn_benchmark {
                    info!("Enabling cuDNN benchmark mode for optimal conv algorithms");
                    tch::Cuda::cudnn_set_benchmark(true);
                }
                
                info!("CUDA available: {}", tch::Cuda::device_count());
            } else {
                warn!("CUDA not available, running on CPU");
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            debug!("CUDA feature not enabled");
            
            // Metal-specific optimizations on macOS
            #[cfg(target_os = "macos")]
            {
                info!("Detected macOS - PyTorch will use Metal Performance Shaders (MPS) if available");
                debug!("For best Metal performance:");
                debug!("  - Use FP16 precision when possible (1.5-2x speedup)");
                debug!("  - Keep batch sizes moderate (Metal prefers smaller batches)");
                debug!("  - Enable shader caching for faster startup");
            }
        }
    }
    
    /// Warmup the model (important for stable latency)
    pub fn warmup(&mut self) -> Result<(), String> {
        if self.warmed_up {
            debug!("Model already warmed up, skipping");
            return Ok(());
        }
        
        info!("Warming up model with {} iterations", self.config.warmup_iterations);
        
        let _guard = tch::no_grad_guard();
        
        // Create dummy input with warmup shape
        let shape: Vec<i64> = self.config.warmup_shape.clone();
        let dummy_input = Tensor::randn(&shape, (Kind::Float, self.device));
        
        let start = Instant::now();
        
        for i in 0..self.config.warmup_iterations {
            let iter_start = Instant::now();
            
            let _output = if self.config.enable_autocast {
                self.forward_with_autocast(&dummy_input)?
            } else {
                self.forward(&dummy_input)?
            };
            
            let iter_time = iter_start.elapsed();
            debug!("Warmup iteration {}: {:.2}ms", i + 1, iter_time.as_secs_f64() * 1000.0);
        }
        
        let total_time = start.elapsed();
        info!("Warmup completed in {:.2}ms", total_time.as_secs_f64() * 1000.0);
        
        self.warmed_up = true;
        Ok(())
    }
    
    /// Forward pass without autocast
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let _guard = tch::no_grad_guard();
        
        self.model
            .forward_ts(&[input.shallow_clone()])
            .map_err(|e| format!("Forward pass failed: {}", e))
    }
    
    /// Forward pass with autocast (mixed precision)
    pub fn forward_with_autocast(&self, input: &Tensor) -> Result<Tensor, String> {
        let _guard = tch::no_grad_guard();
        
        #[cfg(feature = "cuda")]
        {
            if tch::Cuda::is_available() {
                return tch::autocast(true, || {
                    self.model.forward_ts(&[input.shallow_clone()])
                })
                .map_err(|e| format!("Autocast forward pass failed: {}", e));
            }
        }
        
        // Fallback to regular forward if CUDA not available
        self.forward(input)
    }
    
    /// Optimized inference (automatically chooses best path)
    pub fn infer(&self, input: &Tensor) -> Result<Tensor, String> {
        if self.config.enable_autocast {
            self.forward_with_autocast(input)
        } else {
            self.forward(input)
        }
    }
    
    /// Batch inference
    pub fn infer_batch(&self, inputs: Vec<&Tensor>) -> Result<Vec<Tensor>, String> {
        let _guard = tch::no_grad_guard();
        
        let mut outputs = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let output = self.infer(input)?;
            outputs.push(output);
        }
        
        Ok(outputs)
    }
    
    /// Get device
    pub fn device(&self) -> Device {
        self.device
    }
    
    /// Check if model is warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }
    
    /// Get configuration
    pub fn config(&self) -> &TorchOptimizationConfig {
        &self.config
    }
}

/// Builder for TorchOptimizationConfig
pub struct TorchOptimizationConfigBuilder {
    config: TorchOptimizationConfig,
}

impl TorchOptimizationConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: TorchOptimizationConfig::default(),
        }
    }
    
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = threads;
        self
    }
    
    pub fn num_interop_threads(mut self, threads: usize) -> Self {
        self.config.num_interop_threads = threads;
        self
    }
    
    pub fn cudnn_benchmark(mut self, enabled: bool) -> Self {
        self.config.cudnn_benchmark = enabled;
        self
    }
    
    pub fn autocast(mut self, enabled: bool) -> Self {
        self.config.enable_autocast = enabled;
        self
    }
    
    pub fn warmup_iterations(mut self, iterations: usize) -> Self {
        self.config.warmup_iterations = iterations;
        self
    }
    
    pub fn warmup_shape(mut self, shape: Vec<i64>) -> Self {
        self.config.warmup_shape = shape;
        self
    }
    
    pub fn device(mut self, device: Device) -> Self {
        self.config.device = device;
        self
    }
    
    pub fn cpu_only(mut self) -> Self {
        self.config.device = Device::Cpu;
        self.config.cudnn_benchmark = false;
        self.config.enable_autocast = false;
        self
    }
    
    pub fn gpu_optimized(mut self) -> Self {
        self.config.device = Device::cuda_if_available();
        self.config.cudnn_benchmark = true;
        self.config.enable_autocast = true;
        self
    }
    
    pub fn build(self) -> TorchOptimizationConfig {
        self.config
    }
}

impl Default for TorchOptimizationConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = TorchOptimizationConfigBuilder::new()
            .num_threads(8)
            .num_interop_threads(2)
            .cudnn_benchmark(true)
            .autocast(true)
            .warmup_iterations(10)
            .build();
        
        assert_eq!(config.num_threads, 8);
        assert_eq!(config.num_interop_threads, 2);
        assert!(config.cudnn_benchmark);
        assert!(config.enable_autocast);
        assert_eq!(config.warmup_iterations, 10);
    }

    #[test]
    fn test_config_cpu_only() {
        let config = TorchOptimizationConfigBuilder::new()
            .cpu_only()
            .build();
        
        assert_eq!(config.device, Device::Cpu);
        assert!(!config.cudnn_benchmark);
        assert!(!config.enable_autocast);
    }

    #[test]
    fn test_config_gpu_optimized() {
        let config = TorchOptimizationConfigBuilder::new()
            .gpu_optimized()
            .build();
        
        assert!(config.cudnn_benchmark);
        assert!(config.enable_autocast);
    }

    #[test]
    fn test_default_config() {
        let config = TorchOptimizationConfig::default();
        
        assert_eq!(config.num_threads, num_cpus::get());
        assert_eq!(config.num_interop_threads, 1);
        assert!(config.cudnn_benchmark);
        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.warmup_shape, vec![1, 3, 224, 224]);
    }
}
