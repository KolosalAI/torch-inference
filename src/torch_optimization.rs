use log::{debug, info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tch::{CModule, Device, Kind, Tensor};

/// Precision target for model weights/activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationDtype {
    F32,
    F16,
    Int8,
}

/// When quantization was applied.
/// `Dynamic` = apply FP16 at inference time (INT8 must be done at Python export time).
/// `Static`  = model was already quantized before loading; config is informational.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    Dynamic,
    Static,
}

/// Quantization configuration attached to a model.
/// INT8 + Dynamic: logs a warning — use `torch.quantization.quantize_dynamic` in Python
///   before exporting the TorchScript `.pt` file, then load the quantized file here.
/// F16 + Dynamic: handled automatically via `enable_fp16` at inference time.
/// Any  + Static:  informational — logged on load, no runtime action.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationConfig {
    pub dtype: QuantizationDtype,
    pub mode: QuantizationMode,
}

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

    /// Convert input tensors to f16 before forward pass (GPU/MPS only, no-op on CPU).
    pub enable_fp16: bool,

    /// Optional quantization config — informational for Static, drives FP16 for Dynamic/F16.
    pub quantization: Option<QuantizationConfig>,
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
            enable_fp16: false,
            quantization: None,
        }
    }
}

/// Returns true when all tensors share the same shape (required for stacking).
pub fn shapes_are_homogeneous(inputs: &[&Tensor]) -> bool {
    match inputs.first() {
        None => true,
        Some(first) => {
            let target = first.size();
            inputs.iter().all(|t| t.size() == target)
        }
    }
}

static INT8_DYNAMIC_WARNED: AtomicBool = AtomicBool::new(false);

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

        info!(
            "Warming up model with {} iterations",
            self.config.warmup_iterations
        );

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
            debug!(
                "Warmup iteration {}: {:.2}ms",
                i + 1,
                iter_time.as_secs_f64() * 1000.0
            );
        }

        let total_time = start.elapsed();
        info!(
            "Warmup completed in {:.2}ms",
            total_time.as_secs_f64() * 1000.0
        );

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
                return tch::autocast(true, || self.model.forward_ts(&[input.shallow_clone()]))
                    .map_err(|e| format!("Autocast forward pass failed: {}", e));
            }
        }

        // Fallback to regular forward if CUDA not available
        self.forward(input)
    }

    /// Optimized inference (automatically chooses best path, applies FP16 if configured).
    pub fn infer(&self, input: &Tensor) -> Result<Tensor, String> {
        // Warn once if Int8+Dynamic requested — must be done at export time in Python.
        if let Some(ref qcfg) = self.config.quantization {
            if matches!(qcfg.dtype, QuantizationDtype::Int8)
                && matches!(qcfg.mode, QuantizationMode::Dynamic)
                && !INT8_DYNAMIC_WARNED.swap(true, Ordering::Relaxed)
            {
                warn!(
                    "INT8 dynamic quantization is not supported at Rust inference time. \
                     Quantize the model in Python with torch.quantization.quantize_dynamic \
                     before exporting the .pt file."
                );
            }
        }

        let input = if self.config.enable_fp16 && self.device != Device::Cpu {
            input.to_kind(Kind::Half)
        } else {
            input.shallow_clone()
        };

        if self.config.enable_autocast {
            self.forward_with_autocast(&input)
        } else {
            self.forward(&input)
        }
    }

    /// Batch inference.
    ///
    /// When all inputs share the same shape, they are stacked into a single
    /// batched tensor and processed in **one** forward pass (GPU/CPU kernel
    /// fusion, no per-sample overhead).  FP16 conversion (if configured) is
    /// applied to the stacked tensor before the forward pass.  When shapes
    /// differ, falls back to N serial forward passes to preserve correctness.
    pub fn infer_batch(&self, inputs: Vec<&Tensor>) -> Result<Vec<Tensor>, String> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        if shapes_are_homogeneous(&inputs) {
            // Stack → [N, original_dims...], one forward pass, then unbind back.
            let batched = Tensor::stack(&inputs, 0);
            let batched_output = self.infer(&batched)?;
            Ok(batched_output.unbind(0))
        } else {
            // Heterogeneous shapes: serial fallback.
            debug!("infer_batch: heterogeneous shapes, falling back to serial inference");
            inputs.iter().map(|input| self.infer(input)).collect()
        }
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

    pub fn fp16(mut self, enabled: bool) -> Self {
        self.config.enable_fp16 = enabled;
        self
    }

    pub fn quantization(mut self, config: Option<QuantizationConfig>) -> Self {
        self.config.quantization = config;
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
        let config = TorchOptimizationConfigBuilder::new().cpu_only().build();

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

    #[test]
    fn test_same_shape_check_passes() {
        let t1 = tch::Tensor::zeros(&[3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
        let t2 = tch::Tensor::zeros(&[3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
        assert!(shapes_are_homogeneous(&[&t1, &t2]));
    }

    #[test]
    fn test_same_shape_check_fails_on_mismatch() {
        let t1 = tch::Tensor::zeros(&[3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
        let t2 = tch::Tensor::zeros(&[3, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
        assert!(!shapes_are_homogeneous(&[&t1, &t2]));
    }

    #[test]
    fn test_same_shape_check_single_input() {
        let t1 = tch::Tensor::zeros(&[1, 3, 224, 224], (tch::Kind::Float, tch::Device::Cpu));
        assert!(shapes_are_homogeneous(&[&t1]));
    }

    #[test]
    fn test_same_shape_check_empty() {
        assert!(shapes_are_homogeneous(&[]));
    }

    #[test]
    fn test_config_fp16_builder() {
        let config = TorchOptimizationConfigBuilder::new().fp16(true).build();
        assert!(config.enable_fp16);
    }

    #[test]
    fn test_config_fp16_default_is_false() {
        let config = TorchOptimizationConfig::default();
        assert!(!config.enable_fp16);
    }

    #[test]
    fn test_config_quantization_builder() {
        use QuantizationDtype::*;
        use QuantizationMode::*;
        let qcfg = QuantizationConfig {
            dtype: F16,
            mode: Dynamic,
        };
        let config = TorchOptimizationConfigBuilder::new()
            .quantization(Some(qcfg))
            .build();
        assert!(config.quantization.is_some());
        assert!(matches!(config.quantization.unwrap().dtype, F16));
    }

    #[test]
    fn test_quantization_dtype_variants() {
        use QuantizationDtype::*;
        let _ = format!("{:?}", F32);
        let _ = format!("{:?}", F16);
        let _ = format!("{:?}", Int8);
    }

    #[test]
    fn test_quantization_mode_variants() {
        use QuantizationMode::*;
        let _ = format!("{:?}", Dynamic);
        let _ = format!("{:?}", Static);
    }

    #[test]
    fn test_quantization_config_clone() {
        use QuantizationDtype::*;
        use QuantizationMode::*;
        let qcfg = QuantizationConfig {
            dtype: Int8,
            mode: Static,
        };
        let cloned = qcfg.clone();
        assert!(matches!(cloned.dtype, Int8));
        assert!(matches!(cloned.mode, Static));
    }
}
