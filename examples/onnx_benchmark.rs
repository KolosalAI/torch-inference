//! ONNX Inference Benchmark Example
//! 
//! This example demonstrates actual ONNX model inference using the optimized
//! CUDA/TensorRT execution providers.
//!
//! Run with: cargo run --example onnx_benchmark --features cuda --no-default-features
//! Run optimized: cargo run --example onnx_benchmark --features cuda --no-default-features --release

use std::path::Path;
use std::time::{Duration, Instant};
use anyhow::Result;
use ort::execution_providers::cuda::CuDNNConvAlgorithmSearch;
use ort::memory::{Allocator, MemoryInfo, AllocatorType, MemoryType, AllocationDevice};

fn main() -> Result<()> {
    println!("{}", "=".repeat(70));
    println!("ONNX Inference Benchmark (Rust) - Optimized CUDA");
    println!("{}", "=".repeat(70));
    
    // Check for ONNX model
    let model_path = Path::new("models/benchmark/resnet18.onnx");
    if !model_path.exists() {
        println!("\nModel not found: {:?}", model_path);
        println!("Please run the Python benchmark first to download the model:");
        println!("  python benchmark_onnx_inference.py");
        return Ok(());
    }
    
    println!("\nModel: {:?}", model_path);
    
    // Test batched inference - OPTIMIZED VERSION
    println!("\n{}", "=".repeat(70));
    println!("Batch Size Comparison - OPTIMIZED (Pinned Memory + TF32)");
    println!("{}", "=".repeat(70));
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64] {
        benchmark_batch_optimized(model_path, batch_size)?;
    }
    
    // Compare with baseline
    println!("\n{}", "=".repeat(70));
    println!("Batch Size Comparison - BASELINE (Standard Allocation)");
    println!("{}", "=".repeat(70));
    
    for batch_size in [1, 2, 4, 8, 16, 32, 64] {
        benchmark_batch(model_path, batch_size, true, false)?;
    }
    
    // Test IoBinding optimization for fixed shapes
    println!("\n{}", "=".repeat(70));
    println!("IoBinding Optimization (Zero-copy, GPU-resident)");
    println!("{}", "=".repeat(70));
    
    for batch_size in [1, 4, 8, 16, 32, 64] {
        benchmark_iobinding(model_path, batch_size)?;
    }
    
    Ok(())
}

fn benchmark_batch(model_path: &Path, batch_size: i64, tf32: bool, nhwc: bool) -> Result<()> {
    // Build optimized CUDA EP
    let mut cuda_builder = ort::execution_providers::CUDAExecutionProvider::default()
        .with_device_id(0)
        .with_conv_max_workspace(true)
        .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Exhaustive);
    
    if tf32 {
        cuda_builder = cuda_builder.with_tf32(true);
    }
    if nhwc {
        cuda_builder = cuda_builder.with_prefer_nhwc(true);
    }
    
    let cuda_ep = cuda_builder.build();
    let cpu_ep = ort::execution_providers::CPUExecutionProvider::default().build();
    
    let mut session = ort::session::Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_execution_providers([cuda_ep, cpu_ep])?
        .commit_from_file(model_path)?;
    
    let input_shape = vec![batch_size, 3, 224, 224];
    let num_elements: usize = input_shape.iter().map(|&x| x as usize).product();
    
    // Pre-allocate input data ONCE (same as Python)
    let input_data: Vec<f32> = vec![0.5f32; num_elements];
    
    // Warmup
    for _ in 0..100 {
        let tensor = ort::value::Tensor::from_array((input_shape.clone(), input_data.clone().into_boxed_slice()))?;
        let _ = session.run(&[tensor.into()][..])?;
    }
    
    // Benchmark - use same iterations as Python (500)
    let num_iterations = 500;
    
    let mut latencies = Vec::with_capacity(num_iterations);
    let start_total = Instant::now();
    
    for _ in 0..num_iterations {
        // Clone the pre-allocated data (fast memcpy, same as Python re-using numpy)
        let start = Instant::now();
        let tensor = ort::value::Tensor::from_array((input_shape.clone(), input_data.clone().into_boxed_slice()))?;
        let _ = session.run(&[tensor.into()][..])?;
        latencies.push(start.elapsed());
    }
    
    let total_time = start_total.elapsed();
    
    latencies.sort();
    let avg_latency: Duration = latencies.iter().sum::<Duration>() / num_iterations as u32;
    let images_processed = (num_iterations * batch_size as usize) as f64;
    let throughput = images_processed / total_time.as_secs_f64();
    
    println!("  Batch {:<3}: {:>8.2} img/s, {:>6.2} ms/batch, {:>6.3} ms/img",
        batch_size,
        throughput,
        avg_latency.as_secs_f64() * 1000.0,
        avg_latency.as_secs_f64() * 1000.0 / batch_size as f64
    );
    
    Ok(())
}

/// Optimized batch benchmark using pinned memory for faster host-to-device transfer
fn benchmark_batch_optimized(model_path: &Path, batch_size: i64) -> Result<()> {
    // Build CUDA EP with all optimizations
    let cuda_ep = ort::execution_providers::CUDAExecutionProvider::default()
        .with_device_id(0)
        .with_conv_max_workspace(true)
        .with_tf32(true)
        .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Exhaustive)
        .build();
    let cpu_ep = ort::execution_providers::CPUExecutionProvider::default().build();
    
    let mut session = ort::session::Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_execution_providers([cuda_ep, cpu_ep])?
        .commit_from_file(model_path)?;
    
    let input_shape: Vec<usize> = vec![batch_size as usize, 3, 224, 224];
    
    // Create pinned memory allocator for faster transfers
    let pinned_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?
    )?;
    
    // Warmup with pinned memory
    for _ in 0..100 {
        let tensor = ort::value::Tensor::<f32>::new(&pinned_allocator, input_shape.clone())?;
        let _ = session.run(&[tensor.into()][..])?;
    }
    
    // Benchmark
    let num_iterations = 500;
    let mut latencies = Vec::with_capacity(num_iterations);
    let start_total = Instant::now();
    
    for _ in 0..num_iterations {
        let start = Instant::now();
        let tensor = ort::value::Tensor::<f32>::new(&pinned_allocator, input_shape.clone())?;
        let _ = session.run(&[tensor.into()][..])?;
        latencies.push(start.elapsed());
    }
    
    let total_time = start_total.elapsed();
    
    latencies.sort();
    let avg_latency: Duration = latencies.iter().sum::<Duration>() / num_iterations as u32;
    let images_processed = (num_iterations * batch_size as usize) as f64;
    let throughput = images_processed / total_time.as_secs_f64();
    
    println!("  Batch {:<3}: {:>8.2} img/s, {:>6.2} ms/batch, {:>6.3} ms/img",
        batch_size,
        throughput,
        avg_latency.as_secs_f64() * 1000.0,
        avg_latency.as_secs_f64() * 1000.0 / batch_size as f64
    );
    
    Ok(())
}

/// IoBinding benchmark - keeps data on GPU, zero-copy between runs
fn benchmark_iobinding(model_path: &Path, batch_size: i64) -> Result<()> {
    // Build CUDA EP with all optimizations
    let cuda_ep = ort::execution_providers::CUDAExecutionProvider::default()
        .with_device_id(0)
        .with_conv_max_workspace(true)
        .with_tf32(true)
        .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Exhaustive)
        .build();
    let cpu_ep = ort::execution_providers::CPUExecutionProvider::default().build();
    
    let mut session = ort::session::Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_execution_providers([cuda_ep, cpu_ep])?
        .commit_from_file(model_path)?;
    
    let input_shape: Vec<usize> = vec![batch_size as usize, 3, 224, 224];
    
    // Create input tensor with pinned memory
    let pinned_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?
    )?;
    
    let mut input_tensor = ort::value::Tensor::<f32>::new(&pinned_allocator, input_shape.clone())?;
    {
        let (_, data) = input_tensor.extract_tensor_mut();
        for x in data.iter_mut() {
            *x = 0.5f32;
        }
    }
    
    // Create IoBinding
    let mut io_binding = session.create_binding()?;
    
    // Get input/output names
    let input_name = &session.inputs[0].name;
    let output_name = &session.outputs[0].name;
    
    // Bind input
    io_binding.bind_input(input_name, &input_tensor)?;
    
    // Create output allocator for GPU memory
    let output_allocator = Allocator::new(
        &session,
        MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUOutput)?
    )?;
    
    // Pre-allocate output tensor (ResNet-18 output: [batch, 1000])
    let output_shape: Vec<usize> = vec![batch_size as usize, 1000];
    let output_tensor = ort::value::Tensor::<f32>::new(&output_allocator, output_shape)?;
    io_binding.bind_output(output_name, output_tensor)?;
    
    // Warmup with IoBinding
    for _ in 0..100 {
        let _ = session.run_binding(&io_binding)?;
    }
    
    // Benchmark with IoBinding
    let num_iterations = 500;
    let mut latencies = Vec::with_capacity(num_iterations);
    let start_total = Instant::now();
    
    for _ in 0..num_iterations {
        let start = Instant::now();
        let _ = session.run_binding(&io_binding)?;
        latencies.push(start.elapsed());
    }
    
    let total_time = start_total.elapsed();
    
    latencies.sort();
    let avg_latency: Duration = latencies.iter().sum::<Duration>() / num_iterations as u32;
    let images_processed = (num_iterations * batch_size as usize) as f64;
    let throughput = images_processed / total_time.as_secs_f64();
    
    println!("  Batch {:<3}: {:>8.2} img/s, {:>6.2} ms/batch, {:>6.3} ms/img (IoBinding)",
        batch_size,
        throughput,
        avg_latency.as_secs_f64() * 1000.0,
        avg_latency.as_secs_f64() * 1000.0 / batch_size as f64
    );
    
    Ok(())
}
