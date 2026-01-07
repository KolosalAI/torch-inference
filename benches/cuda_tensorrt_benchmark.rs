//! CUDA and TensorRT Performance Benchmark Suite
//! 
//! This benchmark suite tests the optimized CUDA and TensorRT configurations
//! for maximum inference performance. Run with:
//! 
//! ```bash
//! cargo bench --features cuda --bench cuda_tensorrt_benchmark
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

// Import the library modules
use torch_inference::core::cuda_optimizer::{
    CudaOptimizer, CudaOptimizerBuilder, CudaOptimizationLevel, 
    ComputePrecision, CudnnAlgorithmStrategy,
};
use torch_inference::models::tensorrt_auto::{
    TensorRTAutoManager, AutoTensorRTConfigBuilder, TensorRTPrecision,
};
use torch_inference::models::onnx_loader::OnnxModelLoader;

/// Benchmark CUDA optimizer configuration creation
fn bench_cuda_optimizer_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda_optimizer_creation");
    group.measurement_time(Duration::from_secs(5));
    
    group.bench_function("default", |b| {
        b.iter(|| {
            black_box(CudaOptimizer::new())
        })
    });
    
    group.bench_function("for_throughput", |b| {
        b.iter(|| {
            black_box(CudaOptimizer::for_throughput())
        })
    });
    
    group.bench_function("for_latency", |b| {
        b.iter(|| {
            black_box(CudaOptimizer::for_latency())
        })
    });
    
    group.bench_function("for_memory_efficiency", |b| {
        b.iter(|| {
            black_box(CudaOptimizer::for_memory_efficiency())
        })
    });
    
    group.bench_function("builder_complex", |b| {
        b.iter(|| {
            black_box(
                CudaOptimizerBuilder::new()
                    .optimization_level(CudaOptimizationLevel::Maximum)
                    .device_id(0)
                    .memory_pool_size(2048, 16384)
                    .memory_fraction(0.9)
                    .compute_streams(8)
                    .copy_streams(4)
                    .enable_cuda_graphs(true)
                    .graph_warmup_iterations(20)
                    .precision(ComputePrecision::FP16)
                    .enable_amp(true)
                    .enable_tf32(true)
                    .cudnn_benchmark(true)
                    .cudnn_algorithm(CudnnAlgorithmStrategy::Exhaustive)
                    .enable_persistent_l2(true)
                    .build()
            )
        })
    });
    
    group.finish();
}

/// Benchmark TensorRT auto-detection
fn bench_tensorrt_auto_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensorrt_auto_detection");
    group.measurement_time(Duration::from_secs(5));
    
    group.bench_function("auto_manager_creation", |b| {
        b.iter(|| {
            black_box(TensorRTAutoManager::new())
        })
    });
    
    group.bench_function("config_builder", |b| {
        b.iter(|| {
            black_box(
                AutoTensorRTConfigBuilder::new()
                    .precision(TensorRTPrecision::FP16)
                    .workspace_size_mb(4096)
                    .max_batch_size(64)
                    .optimization_level(5)
                    .build()
            )
        })
    });
    
    group.finish();
}

/// Benchmark ONNX loader creation with different configurations
fn bench_onnx_loader_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("onnx_loader_creation");
    group.measurement_time(Duration::from_secs(5));
    
    group.bench_function("default_tensorrt", |b| {
        b.iter(|| {
            black_box(OnnxModelLoader::new(true, 0, None))
        })
    });
    
    group.bench_function("default_cuda_only", |b| {
        b.iter(|| {
            black_box(OnnxModelLoader::new(false, 0, None))
        })
    });
    
    group.bench_function("for_throughput", |b| {
        b.iter(|| {
            black_box(OnnxModelLoader::for_throughput(0))
        })
    });
    
    group.bench_function("for_latency", |b| {
        b.iter(|| {
            black_box(OnnxModelLoader::for_latency(0))
        })
    });
    
    group.bench_function("for_int8", |b| {
        b.iter(|| {
            black_box(OnnxModelLoader::for_int8(0))
        })
    });
    
    group.finish();
}

/// Benchmark stats recording performance
fn bench_stats_recording(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_recording");
    group.measurement_time(Duration::from_secs(5));
    
    let optimizer = CudaOptimizer::new();
    
    group.bench_function("record_batch_time", |b| {
        b.iter(|| {
            optimizer.record_batch_time(black_box(10.5))
        })
    });
    
    group.bench_function("record_allocation", |b| {
        b.iter(|| {
            optimizer.record_allocation(black_box(1024 * 1024))
        })
    });
    
    group.bench_function("record_deallocation", |b| {
        b.iter(|| {
            optimizer.record_deallocation(black_box(1024 * 1024))
        })
    });
    
    group.bench_function("get_stats", |b| {
        b.iter(|| {
            black_box(optimizer.stats())
        })
    });
    
    group.finish();
}

/// Benchmark option generation for ONNX Runtime
fn bench_ort_options_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ort_options_generation");
    group.measurement_time(Duration::from_secs(5));
    
    let optimizer = CudaOptimizer::new();
    
    group.bench_function("get_ort_cuda_options", |b| {
        b.iter(|| {
            black_box(optimizer.get_ort_cuda_options())
        })
    });
    
    group.bench_function("get_tensorrt_options", |b| {
        b.iter(|| {
            black_box(optimizer.get_tensorrt_options())
        })
    });
    
    group.finish();
}

/// Benchmark batch size recommendations
fn bench_batch_size_recommendations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_size_recommendations");
    group.measurement_time(Duration::from_secs(5));
    
    let optimizer = CudaOptimizer::new();
    let model_sizes = [100, 500, 1000, 2000, 4000];
    
    for size in model_sizes {
        group.bench_with_input(
            BenchmarkId::new("model_size_mb", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    black_box(optimizer.recommended_batch_size(size))
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark summary generation
fn bench_summary_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("summary_generation");
    group.measurement_time(Duration::from_secs(5));
    
    let cuda_optimizer = CudaOptimizer::new();
    let tensorrt_manager = TensorRTAutoManager::new();
    
    group.bench_function("cuda_optimizer_summary", |b| {
        b.iter(|| {
            black_box(cuda_optimizer.get_summary())
        })
    });
    
    group.bench_function("tensorrt_manager_summary", |b| {
        b.iter(|| {
            black_box(tensorrt_manager.get_summary())
        })
    });
    
    group.finish();
}

/// Benchmark precision configurations
fn bench_precision_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_configs");
    group.measurement_time(Duration::from_secs(5));
    
    let precisions = [
        ComputePrecision::FP32,
        ComputePrecision::FP16,
        ComputePrecision::BF16,
        ComputePrecision::INT8,
    ];
    
    for precision in precisions {
        group.bench_with_input(
            BenchmarkId::new("precision", precision.as_str()),
            &precision,
            |b, &precision| {
                b.iter(|| {
                    black_box(
                        CudaOptimizerBuilder::new()
                            .precision(precision)
                            .build()
                    )
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark different optimization levels
fn bench_optimization_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_levels");
    group.measurement_time(Duration::from_secs(5));
    
    let levels = [
        ("minimal", CudaOptimizationLevel::Minimal),
        ("standard", CudaOptimizationLevel::Standard),
        ("aggressive", CudaOptimizationLevel::Aggressive),
        ("maximum", CudaOptimizationLevel::Maximum),
    ];
    
    for (name, level) in levels {
        group.bench_with_input(
            BenchmarkId::new("level", name),
            &level,
            |b, &level| {
                b.iter(|| {
                    black_box(
                        CudaOptimizerBuilder::new()
                            .optimization_level(level)
                            .build()
                    )
                })
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_cuda_optimizer_creation,
    bench_tensorrt_auto_detection,
    bench_onnx_loader_creation,
    bench_stats_recording,
    bench_ort_options_generation,
    bench_batch_size_recommendations,
    bench_summary_generation,
    bench_precision_configs,
    bench_optimization_levels,
);

criterion_main!(benches);
