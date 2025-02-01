#!/usr/bin/env python3

import torch
import torch_tensorrt
import torchvision.models as models
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    PYTORCH = "pytorch"
    STATIC_TRT = "static_trt"
    DYNAMIC_TRT = "dynamic_trt"

@dataclass
class BenchmarkConfig:
    # Model configuration
    model_name: str = "resnet18"
    precision: torch.dtype = torch.half
    
    # Batch size configuration
    pytorch_batches: List[int] = (1, 8)
    static_batches: List[int] = (1, 8)
    dynamic_batches: List[int] = (1, 4, 12, 16)
    
    # Execution configuration
    concurrency_levels: List[int] = (1,)
    warmup_runs: int = 3
    test_runs: int = 10
    
    # Benchmark controls
    enable_pytorch: bool = True
    enable_static: bool = True
    enable_dynamic: bool = True
    
    # TensorRT compilation parameters
    workspace_size: int = 20 << 30  # 20GB
    min_block_size: int = 90
    opt_shape: List[int] = (8, 3, 224, 224)

@dataclass
class BenchmarkResults:
    model_type: ModelType
    batch_size: int
    concurrency: int
    avg_time_per_sample: float
    samples_per_second: float
    compilation_time: float = 0.0

class InferenceBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model()
        self.results = []
        
    def _init_model(self) -> torch.nn.Module:
        """Initialize base PyTorch model"""
        logger.info(f"Initializing {self.config.model_name} model")
        model = getattr(models, self.config.model_name)(pretrained=True)
        return model.half().eval().to(self.device)
    
    def _compile_trt_model(self, compilation_mode: str) -> Optional[torch.jit.ScriptModule]:
        """Compile TensorRT model with error handling and timing"""
        try:
            start_time = time.time()
            logger.info(f"Compiling {compilation_mode} TRT model...")
            
            if compilation_mode == ModelType.STATIC_TRT.value:
                trt_model = torch_tensorrt.compile(
                    self.model,
                    ir="torch_compile",
                    inputs=[torch.randn((1, 3, 224, 224), 
                            dtype=self.config.precision, 
                            device=self.device)],
                    enabled_precisions={self.config.precision},
                    workspace_size=self.config.workspace_size,
                    min_block_size=self.config.min_block_size
                )
            elif compilation_mode == ModelType.DYNAMIC_TRT.value:
                trt_model = torch_tensorrt.compile(
                    self.model,
                    ir="dynamo",
                    inputs=[torch_tensorrt.Input(
                        min_shape=(1, 3, 224, 224),
                        opt_shape=self.config.opt_shape,
                        max_shape=(16, 3, 224, 224),
                        dtype=self.config.precision,
                    )],
                    enabled_precisions={self.config.precision},
                    workspace_size=self.config.workspace_size
                )
            else:
                raise ValueError(f"Invalid compilation mode: {compilation_mode}")
            
            compile_time = time.time() - start_time
            logger.info(f"Compiled {compilation_mode} model in {compile_time:.2f}s")
            return trt_model, compile_time
            
        except Exception as e:
            logger.error(f"Failed to compile {compilation_mode} model: {str(e)}")
            return None, 0.0

    def _run_benchmark(self, model: torch.nn.Module, model_type: ModelType, compile_time: float = 0.0):
        """Generic benchmarking function with improved timing and error handling"""
        logger.info(f"Starting {model_type.value} benchmark...")
        
        # Map model type to corresponding batch configuration attribute in config
        batch_attr_map = {
            ModelType.PYTORCH: "pytorch_batches",
            ModelType.STATIC_TRT: "static_batches",
            ModelType.DYNAMIC_TRT: "dynamic_batches",
        }
        batch_list = getattr(self.config, batch_attr_map[model_type])
        
        for concurrency in self.config.concurrency_levels:
            for batch_size in batch_list:
                try:
                    logger.info(f"Testing {model_type.value} - batch: {batch_size}, concurrency: {concurrency}")
                    
                    # Pre-allocate inputs and streams
                    inputs = [torch.randn((batch_size, 3, 224, 224), 
                             dtype=self.config.precision, 
                             device=self.device) for _ in range(concurrency)]
                    streams = [torch.cuda.Stream() for _ in range(concurrency)]

                    # Warmup phase with progress
                    logger.debug(f"Warming up ({self.config.warmup_runs} runs)")
                    for i in range(self.config.warmup_runs):
                        for j in range(concurrency):
                            with torch.cuda.stream(streams[j]):
                                _ = model(inputs[j])
                        if (i+1) % 5 == 0:
                            logger.debug(f"Completed {i+1}/{self.config.warmup_runs} warmup iterations")
                    torch.cuda.synchronize()

                    # Timed execution
                    logger.debug(f"Timing ({self.config.test_runs} runs)")
                    start_time = time.time()
                    for i in range(self.config.test_runs):
                        for j in range(concurrency):
                            with torch.cuda.stream(streams[j]):
                                _ = model(inputs[j])
                    torch.cuda.synchronize()
                    
                    elapsed = time.time() - start_time
                    total_samples = concurrency * batch_size * self.config.test_runs
                    
                    # Store results
                    self.results.append(BenchmarkResults(
                        model_type=model_type,
                        batch_size=batch_size,
                        concurrency=concurrency,
                        avg_time_per_sample=elapsed / total_samples,
                        samples_per_second=total_samples / elapsed,
                        compilation_time=compile_time
                    ))
                    logger.info(f"Completed {model_type.value} batch {batch_size} concurrency {concurrency}: "
                               f"{self.results[-1].samples_per_second:.1f} samples/s")
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {model_type.value} batch {batch_size}: {str(e)}")
                    continue

    def run_all_benchmarks(self):
        """Run all configured benchmarks with proper resource management"""
        if self.config.enable_pytorch:
            logger.info("\n=== Running PyTorch Benchmark ===")
            self._run_benchmark(self.model, ModelType.PYTORCH)

        if self.config.enable_static:
            logger.info("\n=== Running Static TRT Benchmark ===")
            static_model, compile_time = self._compile_trt_model(ModelType.STATIC_TRT.value)
            if static_model is not None:
                self._run_benchmark(static_model, ModelType.STATIC_TRT, compile_time)

        if self.config.enable_dynamic:
            logger.info("\n=== Running Dynamic TRT Benchmark ===")
            dynamic_model, compile_time = self._compile_trt_model(ModelType.DYNAMIC_TRT.value)
            if dynamic_model is not None:
                self._run_benchmark(dynamic_model, ModelType.DYNAMIC_TRT, compile_time)

    def print_comparison(self):
        """Print enhanced comparison table with grouping by batch/concurrency"""
        logger.info("\n=== Benchmark Comparison ===")
        
        # Group results by (batch_size, concurrency)
        groups: Dict[Tuple[int, int], Dict[ModelType, BenchmarkResults]] = {}
        for result in self.results:
            key = (result.batch_size, result.concurrency)
            groups.setdefault(key, {})[result.model_type] = result

        # Print table header
        header = f"| {'Batch':>5} | {'Concurrency':>11} | " \
                 f"{'PyTorch (samples/s)':>20} | {'Static TRT':>20} | {'Dynamic TRT':>20} | " \
                 f"{'Static Speedup':>15} | {'Dynamic Speedup':>15} |"
        separator = "-" * len(header)
        print(f"\n{separator}\n{header}\n{separator}")

        # Print each group
        for key in sorted(groups.keys()):
            batch, concurrency = key
            group = groups[key]
            
            # Get results for each model type
            pytorch = group.get(ModelType.PYTORCH)
            static = group.get(ModelType.STATIC_TRT)
            dynamic = group.get(ModelType.DYNAMIC_TRT)

            # Calculate speedups
            static_speedup = static.samples_per_second / pytorch.samples_per_second if pytorch and static else 0
            dynamic_speedup = dynamic.samples_per_second / pytorch.samples_per_second if pytorch and dynamic else 0

            # Format row
            row = f"| {batch:>5} | {concurrency:>11} | " \
                  f"{pytorch.samples_per_second if pytorch else 'N/A':>20.1f} | " \
                  f"{static.samples_per_second if static else 'N/A':>20.1f} | " \
                  f"{dynamic.samples_per_second if dynamic else 'N/A':>20.1f} | " \
                  f"{static_speedup:>15.1f}x | {dynamic_speedup:>15.1f}x |"
            print(row)
        
        print(separator + "\n")

if __name__ == "__main__":
    config = BenchmarkConfig(
        pytorch_batches=[1, 8, 16],
        static_batches=[1, 8, 16],
        dynamic_batches=[1, 8, 16],
        concurrency_levels=[1, 2],
        warmup_runs=3,
        test_runs=10
    )

    benchmark = InferenceBenchmark(config)
    benchmark.run_all_benchmarks()
    benchmark.print_comparison()
