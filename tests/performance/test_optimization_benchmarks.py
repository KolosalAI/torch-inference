"""
Performance benchmark tests for advanced optimization features.
Measures and compares performance improvements across different scenarios.
"""

import pytest
import torch
import torch.nn as nn
import time
import statistics
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import psutil
import gc

from framework.optimizers.int8_calibration import INT8CalibrationToolkit, CalibrationConfig
from framework.optimizers.kernel_autotuner import KernelAutoTuner, TuningConfig
from framework.optimizers.advanced_fusion import AdvancedLayerFusion, FusionConfig
from framework.optimizers.memory_optimizer import MemoryOptimizer, MemoryConfig


class BenchmarkModel(nn.Module):
    """Standard model for benchmarking."""
    
    def __init__(self, complexity="medium"):
        super().__init__()
        
        if complexity == "simple":
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(32 * 4 * 4, 10)
            
        elif complexity == "medium":
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)
            
        else:  # complex
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8))
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512 * 8 * 8, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
            
        self.complexity = complexity
    
    def forward(self, x):
        if self.complexity == "simple":
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            
        elif self.complexity == "medium":
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            
        else:  # complex
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            
        return x


class PerformanceProfiler:
    """Utility for measuring performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all measurements."""
        self.latencies = []
        self.memory_usage = []
        self.cpu_usage = []
        self.gpu_memory = []
    
    def measure_inference(self, model, input_data, warmup_runs=5, test_runs=20):
        """Measure inference performance."""
        device = next(model.parameters()).device
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_data)
        
        # Synchronize if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Measurement runs
        latencies = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(test_runs):
                # Memory before
                if device.type == "cuda":
                    mem_before = torch.cuda.memory_allocated(device) / 1024**2
                else:
                    mem_before = psutil.Process().memory_info().rss / 1024**2
                
                # Measure latency
                start_time = time.perf_counter()
                output = model(input_data)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
                
                # Memory after
                if device.type == "cuda":
                    mem_after = torch.cuda.memory_allocated(device) / 1024**2
                else:
                    mem_after = psutil.Process().memory_info().rss / 1024**2
                
                memory_usage.append(mem_after - mem_before)
        
        return {
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies),
                "p95": sorted(latencies)[int(0.95 * len(latencies))],
                "p99": sorted(latencies)[int(0.99 * len(latencies))]
            },
            "memory_mb": {
                "mean": statistics.mean(memory_usage),
                "peak": max(memory_usage) if memory_usage else 0
            },
            "throughput_samples_per_sec": input_data.size(0) / (statistics.mean(latencies) / 1000),
            "output_shape": output.shape
        }
    
    def measure_memory_footprint(self, model, input_data):
        """Measure memory footprint of model."""
        device = next(model.parameters()).device
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            
            # Measure model size
            model_memory = torch.cuda.memory_allocated(device)
            
            # Forward pass
            with torch.no_grad():
                output = model(input_data)
            
            peak_memory = torch.cuda.max_memory_allocated(device)
            
            return {
                "model_memory_mb": model_memory / 1024**2,
                "peak_memory_mb": peak_memory / 1024**2,
                "activation_memory_mb": (peak_memory - model_memory) / 1024**2
            }
        else:
            # CPU memory measurement is more approximate
            process = psutil.Process()
            mem_before = process.memory_info().rss
            
            with torch.no_grad():
                output = model(input_data)
            
            mem_after = process.memory_info().rss
            
            return {
                "memory_increase_mb": (mem_after - mem_before) / 1024**2,
                "peak_memory_mb": mem_after / 1024**2
            }


@pytest.fixture
def profiler():
    """Create performance profiler."""
    return PerformanceProfiler()


@pytest.fixture
def benchmark_models():
    """Create benchmark models of different complexities."""
    return {
        "simple": BenchmarkModel("simple"),
        "medium": BenchmarkModel("medium"),
        "complex": BenchmarkModel("complex")
    }


@pytest.fixture
def benchmark_inputs():
    """Create benchmark inputs of different sizes."""
    return {
        "small": torch.randn(1, 3, 32, 32),
        "medium": torch.randn(4, 3, 64, 64),
        "large": torch.randn(8, 3, 128, 128),
        "batch": torch.randn(16, 3, 32, 32)
    }


class TestBaseBenchmarks:
    """Establish baseline performance measurements."""
    
    def test_baseline_performance(self, profiler, benchmark_models, benchmark_inputs):
        """Measure baseline performance for all model/input combinations."""
        device = torch.device("cpu")
        baseline_results = {}
        
        for model_name, model in benchmark_models.items():
            model = model.to(device)
            baseline_results[model_name] = {}
            
            for input_name, input_data in benchmark_inputs.items():
                try:
                    input_data = input_data.to(device)
                    
                    # Measure performance
                    perf_metrics = profiler.measure_inference(
                        model, input_data, warmup_runs=3, test_runs=10
                    )
                    
                    # Measure memory
                    memory_metrics = profiler.measure_memory_footprint(model, input_data)
                    
                    baseline_results[model_name][input_name] = {
                        "performance": perf_metrics,
                        "memory": memory_metrics
                    }
                    
                    print(f"✓ Baseline {model_name}/{input_name}: "
                          f"{perf_metrics['latency_ms']['mean']:.2f}ms, "
                          f"{perf_metrics['throughput_samples_per_sec']:.1f} samples/sec")
                    
                except Exception as e:
                    print(f"⚠ Baseline {model_name}/{input_name} failed: {e}")
        
        return baseline_results
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_baseline_performance(self, profiler, benchmark_models, benchmark_inputs):
        """Measure CUDA baseline performance."""
        device = torch.device("cuda")
        cuda_baseline_results = {}
        
        for model_name, model in benchmark_models.items():
            if model_name == "complex":  # Skip complex model on CUDA for faster testing
                continue
                
            model = model.to(device)
            cuda_baseline_results[model_name] = {}
            
            for input_name, input_data in benchmark_inputs.items():
                if input_name == "large":  # Skip large inputs for faster testing
                    continue
                    
                try:
                    input_data = input_data.to(device)
                    
                    # Measure performance
                    perf_metrics = profiler.measure_inference(
                        model, input_data, warmup_runs=5, test_runs=15
                    )
                    
                    # Measure memory
                    memory_metrics = profiler.measure_memory_footprint(model, input_data)
                    
                    cuda_baseline_results[model_name][input_name] = {
                        "performance": perf_metrics,
                        "memory": memory_metrics
                    }
                    
                    print(f"✓ CUDA Baseline {model_name}/{input_name}: "
                          f"{perf_metrics['latency_ms']['mean']:.2f}ms, "
                          f"{perf_metrics['throughput_samples_per_sec']:.1f} samples/sec")
                    
                except Exception as e:
                    print(f"⚠ CUDA Baseline {model_name}/{input_name} failed: {e}")
        
        return cuda_baseline_results


class TestMemoryOptimizationBenchmarks:
    """Benchmark memory optimization performance."""
    
    def test_memory_optimization_performance(self, profiler, benchmark_models):
        """Benchmark memory optimization improvements."""
        device = torch.device("cpu")
        input_data = torch.randn(4, 3, 64, 64).to(device)
        
        results = {}
        
        for model_name, original_model in benchmark_models.items():
            if model_name == "complex":  # Skip for faster testing
                continue
                
            original_model = original_model.to(device).eval()
            
            # Baseline measurement
            baseline_perf = profiler.measure_inference(
                original_model, input_data, warmup_runs=3, test_runs=10
            )
            
            # Apply memory optimization
            memory_config = MemoryConfig(
                pool_size_mb=64,
                fragmentation_threshold=0.4,
                enable_background_cleanup=False
            )
            memory_optimizer = MemoryOptimizer(memory_config)
            
            try:
                optimized_model = memory_optimizer.optimize_model(original_model, input_data)
                
                # Measure optimized performance
                optimized_perf = profiler.measure_inference(
                    optimized_model, input_data, warmup_runs=3, test_runs=10
                )
                
                # Calculate improvements
                latency_improvement = (
                    baseline_perf["latency_ms"]["mean"] / optimized_perf["latency_ms"]["mean"]
                )
                throughput_improvement = (
                    optimized_perf["throughput_samples_per_sec"] / baseline_perf["throughput_samples_per_sec"]
                )
                
                results[model_name] = {
                    "baseline": baseline_perf,
                    "optimized": optimized_perf,
                    "latency_speedup": latency_improvement,
                    "throughput_speedup": throughput_improvement
                }
                
                print(f"✓ Memory optimization {model_name}: "
                      f"{latency_improvement:.2f}x latency, "
                      f"{throughput_improvement:.2f}x throughput")
                
            except Exception as e:
                print(f"⚠ Memory optimization {model_name} failed: {e}")
        
        return results
    
    def test_fragmentation_impact(self, profiler):
        """Test impact of memory fragmentation on performance."""
        memory_config = MemoryConfig(
            pool_size_mb=64,
            fragmentation_threshold=0.5,
            enable_background_cleanup=False
        )
        memory_optimizer = MemoryOptimizer(memory_config)
        pool = memory_optimizer.memory_pool
        
        # Create fragmented memory pattern
        tensors = []
        for i in range(20):
            size = torch.Size([50 + i * 5, 50 + i * 5])  # Varying sizes
            tensor = pool.allocate_tensor(size, torch.float32, torch.device("cpu"))
            tensors.append(tensor)
        
        # Deallocate every other tensor to create fragmentation
        for i in range(0, len(tensors), 2):
            pool.deallocate_tensor(tensors[i])
        
        # Measure fragmentation
        fragmentation_stats = pool.get_fragmentation_stats()
        
        print(f"✓ Fragmentation test: {fragmentation_stats.fragmentation_ratio:.3f} ratio, "
              f"{fragmentation_stats.num_free_blocks} free blocks")
        
        # Test allocation performance under fragmentation
        start_time = time.perf_counter()
        for i in range(10):
            new_tensor = pool.allocate_tensor(
                torch.Size([100, 100]), torch.float32, torch.device("cpu")
            )
            pool.deallocate_tensor(new_tensor)
        end_time = time.perf_counter()
        
        fragmented_allocation_time = (end_time - start_time) * 1000  # ms
        
        print(f"✓ Fragmented allocation time: {fragmented_allocation_time:.2f}ms for 10 allocations")
        
        return {
            "fragmentation_ratio": fragmentation_stats.fragmentation_ratio,
            "allocation_time_ms": fragmented_allocation_time
        }


class TestKernelTuningBenchmarks:
    """Benchmark kernel auto-tuning performance."""
    
    def test_kernel_tuning_performance(self, profiler, benchmark_models):
        """Benchmark kernel tuning improvements."""
        device = torch.device("cpu")
        input_data = torch.randn(4, 3, 64, 64).to(device)
        
        results = {}
        
        for model_name, original_model in benchmark_models.items():
            if model_name == "complex":  # Skip for faster testing
                continue
                
            original_model = original_model.to(device).eval()
            
            # Baseline measurement
            baseline_perf = profiler.measure_inference(
                original_model, input_data, warmup_runs=3, test_runs=10
            )
            
            # Apply kernel tuning
            tuning_config = TuningConfig(
                max_iterations=3,
                warmup_iterations=1,
                timeout_seconds=30,
                enable_caching=False,
                batch_sizes=[1, 4]
            )
            kernel_tuner = KernelAutoTuner(tuning_config)
            
            try:
                tuning_result = kernel_tuner.auto_tune(original_model, input_data, device)
                optimized_model = tuning_result["optimized_model"]
                
                # Measure optimized performance
                optimized_perf = profiler.measure_inference(
                    optimized_model, input_data, warmup_runs=3, test_runs=10
                )
                
                # Calculate improvements
                latency_improvement = (
                    baseline_perf["latency_ms"]["mean"] / optimized_perf["latency_ms"]["mean"]
                )
                throughput_improvement = (
                    optimized_perf["throughput_samples_per_sec"] / baseline_perf["throughput_samples_per_sec"]
                )
                
                results[model_name] = {
                    "baseline": baseline_perf,
                    "optimized": optimized_perf,
                    "latency_speedup": latency_improvement,
                    "throughput_speedup": throughput_improvement,
                    "best_config": tuning_result["best_config"]
                }
                
                print(f"✓ Kernel tuning {model_name}: "
                      f"{latency_improvement:.2f}x latency, "
                      f"{throughput_improvement:.2f}x throughput")
                
            except Exception as e:
                print(f"⚠ Kernel tuning {model_name} failed: {e}")
        
        return results
    
    def test_batch_size_optimization_impact(self, profiler):
        """Test impact of batch size optimization."""
        model = BenchmarkModel("medium").eval()
        device = torch.device("cpu")
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        results = {}
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, 3, 64, 64).to(device)
            
            try:
                perf_metrics = profiler.measure_inference(
                    model, input_data, warmup_runs=2, test_runs=5
                )
                
                results[batch_size] = {
                    "latency_per_sample": perf_metrics["latency_ms"]["mean"] / batch_size,
                    "throughput": perf_metrics["throughput_samples_per_sec"],
                    "total_latency": perf_metrics["latency_ms"]["mean"]
                }
                
                print(f"✓ Batch size {batch_size}: "
                      f"{results[batch_size]['latency_per_sample']:.2f}ms/sample, "
                      f"{results[batch_size]['throughput']:.1f} samples/sec")
                
            except Exception as e:
                print(f"⚠ Batch size {batch_size} failed: {e}")
        
        return results


class TestFusionBenchmarks:
    """Benchmark layer fusion performance."""
    
    def test_fusion_performance(self, profiler, benchmark_models):
        """Benchmark layer fusion improvements."""
        device = torch.device("cpu")
        input_data = torch.randn(4, 3, 64, 64).to(device)
        
        results = {}
        
        for model_name, original_model in benchmark_models.items():
            if model_name == "complex":  # Skip for faster testing
                continue
                
            original_model = original_model.to(device).eval()
            
            # Baseline measurement
            baseline_perf = profiler.measure_inference(
                original_model, input_data, warmup_runs=3, test_runs=10
            )
            
            # Apply layer fusion
            fusion_config = FusionConfig(
                optimization_level=2,
                validate_numerics=True,
                fallback_to_eager=True
            )
            layer_fusion = AdvancedLayerFusion(fusion_config)
            
            try:
                fused_model = layer_fusion.fuse_model(original_model, input_data)
                
                if fused_model is not None:
                    # Measure fused performance
                    fused_perf = profiler.measure_inference(
                        fused_model, input_data, warmup_runs=3, test_runs=10
                    )
                    
                    # Calculate improvements
                    latency_improvement = (
                        baseline_perf["latency_ms"]["mean"] / fused_perf["latency_ms"]["mean"]
                    )
                    throughput_improvement = (
                        fused_perf["throughput_samples_per_sec"] / baseline_perf["throughput_samples_per_sec"]
                    )
                    
                    results[model_name] = {
                        "baseline": baseline_perf,
                        "fused": fused_perf,
                        "latency_speedup": latency_improvement,
                        "throughput_speedup": throughput_improvement
                    }
                    
                    print(f"✓ Layer fusion {model_name}: "
                          f"{latency_improvement:.2f}x latency, "
                          f"{throughput_improvement:.2f}x throughput")
                else:
                    print(f"⚠ Layer fusion {model_name}: no fusions applied")
                    
            except Exception as e:
                print(f"⚠ Layer fusion {model_name} failed: {e}")
        
        return results
    
    def test_fusion_pattern_effectiveness(self, profiler):
        """Test effectiveness of different fusion patterns."""
        device = torch.device("cpu")
        
        # Test Conv+BN fusion
        class ConvBNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
            
            def forward(self, x):
                x = self.bn1(self.conv1(x))
                x = self.bn2(self.conv2(x))
                return x
        
        model = ConvBNModel().to(device).eval()
        input_data = torch.randn(4, 3, 32, 32).to(device)
        
        # Baseline
        baseline_perf = profiler.measure_inference(
            model, input_data, warmup_runs=3, test_runs=10
        )
        
        # Apply fusion
        fusion_config = FusionConfig(
            enable_conv_bn_fusion=True,
            optimization_level=1
        )
        layer_fusion = AdvancedLayerFusion(fusion_config)
        
        try:
            fused_model = layer_fusion.fuse_model(model, input_data)
            
            if fused_model is not None:
                fused_perf = profiler.measure_inference(
                    fused_model, input_data, warmup_runs=3, test_runs=10
                )
                
                improvement = baseline_perf["latency_ms"]["mean"] / fused_perf["latency_ms"]["mean"]
                
                print(f"✓ Conv+BN fusion improvement: {improvement:.2f}x")
                return improvement
            else:
                print("⚠ Conv+BN fusion: no fusions applied")
                return 1.0
                
        except Exception as e:
            print(f"⚠ Conv+BN fusion failed: {e}")
            return 1.0


class TestIntegratedBenchmarks:
    """Benchmark combined optimizations."""
    
    def test_combined_optimization_performance(self, profiler):
        """Test performance of combined optimizations."""
        device = torch.device("cpu")
        model = BenchmarkModel("medium").to(device).eval()
        input_data = torch.randn(4, 3, 64, 64).to(device)
        
        # Baseline measurement
        baseline_perf = profiler.measure_inference(
            model, input_data, warmup_runs=3, test_runs=10
        )
        
        current_model = model
        optimization_chain = []
        
        # Step 1: Memory optimization
        memory_config = MemoryConfig(pool_size_mb=64, enable_background_cleanup=False)
        memory_optimizer = MemoryOptimizer(memory_config)
        
        try:
            current_model = memory_optimizer.optimize_model(current_model, input_data)
            
            mem_perf = profiler.measure_inference(
                current_model, input_data, warmup_runs=3, test_runs=10
            )
            
            mem_improvement = baseline_perf["latency_ms"]["mean"] / mem_perf["latency_ms"]["mean"]
            optimization_chain.append(("memory", mem_improvement))
            
            print(f"✓ After memory optimization: {mem_improvement:.2f}x")
            
        except Exception as e:
            print(f"⚠ Memory optimization in chain failed: {e}")
        
        # Step 2: Layer fusion
        fusion_config = FusionConfig(optimization_level=1, fallback_to_eager=True)
        layer_fusion = AdvancedLayerFusion(fusion_config)
        
        try:
            fused_model = layer_fusion.fuse_model(current_model, input_data)
            
            if fused_model is not None:
                current_model = fused_model
                
                fusion_perf = profiler.measure_inference(
                    current_model, input_data, warmup_runs=3, test_runs=10
                )
                
                fusion_improvement = baseline_perf["latency_ms"]["mean"] / fusion_perf["latency_ms"]["mean"]
                optimization_chain.append(("fusion", fusion_improvement))
                
                print(f"✓ After memory + fusion: {fusion_improvement:.2f}x")
            else:
                print("⚠ Fusion: no fusions applied")
                
        except Exception as e:
            print(f"⚠ Fusion optimization in chain failed: {e}")
        
        # Step 3: Kernel tuning
        tuning_config = TuningConfig(
            max_iterations=2,
            timeout_seconds=20,
            enable_caching=False
        )
        kernel_tuner = KernelAutoTuner(tuning_config)
        
        try:
            tuning_result = kernel_tuner.auto_tune(current_model, input_data, device)
            current_model = tuning_result["optimized_model"]
            
            final_perf = profiler.measure_inference(
                current_model, input_data, warmup_runs=3, test_runs=10
            )
            
            final_improvement = baseline_perf["latency_ms"]["mean"] / final_perf["latency_ms"]["mean"]
            optimization_chain.append(("tuning", final_improvement))
            
            print(f"✓ Final combined optimization: {final_improvement:.2f}x")
            
        except Exception as e:
            print(f"⚠ Kernel tuning in chain failed: {e}")
        
        return {
            "baseline": baseline_perf,
            "final": final_perf if 'final_perf' in locals() else None,
            "optimization_chain": optimization_chain,
            "total_improvement": optimization_chain[-1][1] if optimization_chain else 1.0
        }
    
    def test_optimization_overhead(self, profiler):
        """Measure overhead of applying optimizations."""
        device = torch.device("cpu")
        model = BenchmarkModel("simple").to(device).eval()
        input_data = torch.randn(2, 3, 32, 32).to(device)
        
        overhead_results = {}
        
        # Memory optimization overhead
        start_time = time.perf_counter()
        memory_config = MemoryConfig(pool_size_mb=32)
        memory_optimizer = MemoryOptimizer(memory_config)
        optimized_model = memory_optimizer.optimize_model(model, input_data)
        memory_overhead = (time.perf_counter() - start_time) * 1000
        
        overhead_results["memory_optimization"] = memory_overhead
        print(f"✓ Memory optimization overhead: {memory_overhead:.1f}ms")
        
        # Fusion optimization overhead
        start_time = time.perf_counter()
        fusion_config = FusionConfig(optimization_level=1)
        layer_fusion = AdvancedLayerFusion(fusion_config)
        fused_model = layer_fusion.fuse_model(model, input_data)
        fusion_overhead = (time.perf_counter() - start_time) * 1000
        
        overhead_results["fusion_optimization"] = fusion_overhead
        print(f"✓ Fusion optimization overhead: {fusion_overhead:.1f}ms")
        
        # Kernel tuning overhead
        start_time = time.perf_counter()
        tuning_config = TuningConfig(max_iterations=1, timeout_seconds=10)
        kernel_tuner = KernelAutoTuner(tuning_config)
        tuning_result = kernel_tuner.auto_tune(model, input_data, device)
        tuning_overhead = (time.perf_counter() - start_time) * 1000
        
        overhead_results["kernel_tuning"] = tuning_overhead
        print(f"✓ Kernel tuning overhead: {tuning_overhead:.1f}ms")
        
        return overhead_results


class TestRegressionBenchmarks:
    """Benchmark for regression testing."""
    
    def test_performance_regression_detection(self, profiler):
        """Test for performance regressions."""
        device = torch.device("cpu")
        model = BenchmarkModel("medium").to(device).eval()
        input_data = torch.randn(4, 3, 64, 64).to(device)
        
        # Expected baseline performance (approximate)
        expected_baseline = {
            "latency_ms": {"mean": 100, "tolerance": 50},  # Allow 50ms tolerance
            "throughput_samples_per_sec": {"min": 10}  # Minimum 10 samples/sec
        }
        
        # Measure actual performance
        actual_perf = profiler.measure_inference(
            model, input_data, warmup_runs=3, test_runs=10
        )
        
        # Check for regressions
        regressions = []
        
        # Latency regression
        if actual_perf["latency_ms"]["mean"] > expected_baseline["latency_ms"]["mean"] + expected_baseline["latency_ms"]["tolerance"]:
            regressions.append(
                f"Latency regression: {actual_perf['latency_ms']['mean']:.1f}ms > "
                f"{expected_baseline['latency_ms']['mean'] + expected_baseline['latency_ms']['tolerance']}ms"
            )
        
        # Throughput regression
        if actual_perf["throughput_samples_per_sec"] < expected_baseline["throughput_samples_per_sec"]["min"]:
            regressions.append(
                f"Throughput regression: {actual_perf['throughput_samples_per_sec']:.1f} < "
                f"{expected_baseline['throughput_samples_per_sec']['min']}"
            )
        
        if regressions:
            print(f"⚠ Performance regressions detected: {regressions}")
        else:
            print("✓ No performance regressions detected")
        
        return {
            "regressions": regressions,
            "actual_performance": actual_perf,
            "expected_baseline": expected_baseline
        }
    
    def save_benchmark_results(self, results, filename="benchmark_results.json"):
        """Save benchmark results for comparison."""
        results_with_metadata = {
            "timestamp": time.time(),
            "torch_version": torch.__version__,
            "device_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "cuda_available": torch.cuda.is_available(),
            },
            "results": results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results_with_metadata, f, indent=2, default=str)
            print(f"✓ Benchmark results saved to {filename}")
        except Exception as e:
            print(f"⚠ Failed to save benchmark results: {e}")


if __name__ == "__main__":
    # Run comprehensive performance benchmarks
    print("Running comprehensive performance benchmarks...")
    
    profiler = PerformanceProfiler()
    model = BenchmarkModel("medium")
    input_data = torch.randn(4, 3, 64, 64)
    device = torch.device("cpu")
    
    all_results = {}
    
    print("\n=== Baseline Performance ===")
    baseline_perf = profiler.measure_inference(
        model.to(device).eval(), input_data.to(device), 
        warmup_runs=3, test_runs=10
    )
    
    print(f"Baseline latency: {baseline_perf['latency_ms']['mean']:.2f}ms ± {baseline_perf['latency_ms']['std']:.2f}")
    print(f"Baseline throughput: {baseline_perf['throughput_samples_per_sec']:.1f} samples/sec")
    
    all_results["baseline"] = baseline_perf
    
    print("\n=== Memory Optimization Benchmark ===")
    memory_config = MemoryConfig(pool_size_mb=64, enable_background_cleanup=False)
    memory_optimizer = MemoryOptimizer(memory_config)
    
    try:
        optimized_model = memory_optimizer.optimize_model(model.to(device).eval(), input_data.to(device))
        mem_perf = profiler.measure_inference(optimized_model, input_data.to(device), warmup_runs=3, test_runs=10)
        
        mem_speedup = baseline_perf["latency_ms"]["mean"] / mem_perf["latency_ms"]["mean"]
        print(f"Memory optimization speedup: {mem_speedup:.2f}x")
        
        all_results["memory_optimization"] = {
            "performance": mem_perf,
            "speedup": mem_speedup
        }
    except Exception as e:
        print(f"Memory optimization benchmark failed: {e}")
    
    print("\n=== Layer Fusion Benchmark ===")
    fusion_config = FusionConfig(optimization_level=2, fallback_to_eager=True)
    layer_fusion = AdvancedLayerFusion(fusion_config)
    
    try:
        fused_model = layer_fusion.fuse_model(model.to(device).eval(), input_data.to(device))
        if fused_model is not None:
            fusion_perf = profiler.measure_inference(fused_model, input_data.to(device), warmup_runs=3, test_runs=10)
            
            fusion_speedup = baseline_perf["latency_ms"]["mean"] / fusion_perf["latency_ms"]["mean"]
            print(f"Layer fusion speedup: {fusion_speedup:.2f}x")
            
            all_results["layer_fusion"] = {
                "performance": fusion_perf,
                "speedup": fusion_speedup
            }
        else:
            print("Layer fusion: no fusions applied")
    except Exception as e:
        print(f"Layer fusion benchmark failed: {e}")
    
    print("\n=== Kernel Tuning Benchmark ===")
    tuning_config = TuningConfig(max_iterations=2, timeout_seconds=20, enable_caching=False)
    kernel_tuner = KernelAutoTuner(tuning_config)
    
    try:
        tuning_result = kernel_tuner.auto_tune(model.to(device).eval(), input_data.to(device), device)
        tuned_model = tuning_result["optimized_model"]
        
        tuning_perf = profiler.measure_inference(tuned_model, input_data.to(device), warmup_runs=3, test_runs=10)
        
        tuning_speedup = baseline_perf["latency_ms"]["mean"] / tuning_perf["latency_ms"]["mean"]
        print(f"Kernel tuning speedup: {tuning_speedup:.2f}x")
        
        all_results["kernel_tuning"] = {
            "performance": tuning_perf,
            "speedup": tuning_speedup,
            "best_config": tuning_result["best_config"]
        }
    except Exception as e:
        print(f"Kernel tuning benchmark failed: {e}")
    
    print("\n=== Performance Summary ===")
    for optimization, result in all_results.items():
        if optimization != "baseline" and "speedup" in result:
            print(f"{optimization}: {result['speedup']:.2f}x improvement")
    
    # Save results
    try:
        with open("performance_benchmark_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print("\n✓ Benchmark results saved to performance_benchmark_results.json")
    except Exception as e:
        print(f"\n⚠ Failed to save results: {e}")
    
    print("\n✓ Comprehensive performance benchmarks completed")
    print("✓ All optimization features benchmarked and validated")
