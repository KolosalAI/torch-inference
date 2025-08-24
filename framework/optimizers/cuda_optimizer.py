"""
CUDA optimization module for PyTorch inference.

This module provides CUDA-specific optimizations including CUDA graphs,
streams, and other GPU acceleration techniques.
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from contextlib import contextmanager
import threading

import torch
import torch.nn as nn

from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


class CUDAOptimizer:
    """
    CUDA optimization manager for PyTorch models.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize CUDA optimizer.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.device = None
        self.cuda_graphs = {}
        self.streams = {}
        self.events = {}
        
        self.logger = logging.getLogger(f"{__name__}.CUDAOptimizer")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Set device
        if config and hasattr(config.device, 'device_id') and config.device.device_id is not None:
            self.device = torch.device(f"cuda:{config.device.device_id}")
        else:
            self.device = torch.device("cuda:0")  # Default to first GPU
        
        # Only set device if CUDA is properly available
        try:
            if hasattr(torch.cuda, 'set_device'):
                torch.cuda.set_device(self.device)
        except (AttributeError, RuntimeError) as e:
            self.logger.warning(f"Could not set CUDA device: {e}")
            # Continue without setting device explicitly
        
        self.logger.info(f"CUDA optimizer initialized on device: {self.device}")
        self._apply_cuda_optimizations()
    
    def optimize(self, model: nn.Module) -> nn.Module:
        """
        Apply CUDA optimizations to model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        if not self.enabled:
            self.logger.warning("CUDA not available, returning original model")
            return model
        
        return self.optimize_model_for_cuda(model)
    
    def apply_cuda_optimizations(self) -> None:
        """Apply CUDA optimizations (public interface)."""
        self._apply_cuda_optimizations()
    
    def _apply_cuda_optimizations(self) -> None:
        """Apply general CUDA optimizations."""
        try:
            # Enable cuDNN benchmark mode
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory management
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "captures_underway" not in str(e):
                    logger.warning(f"Failed to clear CUDA cache: {e}")
            
            # Enable tensor cores if available (for RTX/V100+ GPUs)
            if torch.cuda.get_device_capability(self.device)[0] >= 7:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("TensorFloat-32 (TF32) enabled for faster computation")
            
            self.logger.info("CUDA optimizations applied")
            
        except Exception as e:
            self.logger.warning(f"Failed to apply some CUDA optimizations: {e}")
    
    def create_cuda_graph(self, 
                         model: nn.Module, 
                         example_inputs: torch.Tensor,
                         graph_name: str = "default") -> bool:
        """
        Create CUDA graph for model inference.
        
        CUDA graphs can provide significant performance improvements by
        eliminating CPU overhead for repetitive GPU operations.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs for graph capture
            graph_name: Name for the CUDA graph
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            self.logger.info(f"Creating CUDA graph: {graph_name}")
            
            # Move model and inputs to CUDA device
            model = model.to(self.device)
            example_inputs = example_inputs.to(self.device)
            
            # Set model to eval mode
            model.eval()
            
            # Warmup - CUDA graphs require consistent memory patterns
            with torch.no_grad():
                for _ in range(3):
                    _ = model(example_inputs)
            
            torch.cuda.synchronize()
            
            # Create graph
            graph = torch.cuda.CUDAGraph()
            
            # Capture graph
            with torch.cuda.graph(graph):
                output = model(example_inputs)
            
            # Store graph and related tensors
            self.cuda_graphs[graph_name] = {
                "graph": graph,
                "model": model,
                "input": example_inputs,
                "output": output
            }
            
            self.logger.info(f"CUDA graph '{graph_name}' created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create CUDA graph: {e}")
            return False
    
    def run_cuda_graph(self, 
                      graph_name: str, 
                      inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Run inference using CUDA graph.
        
        Args:
            graph_name: Name of the CUDA graph
            inputs: Input tensor (must match graph input shape exactly)
            
        Returns:
            Output tensor or None if failed
        """
        if not self.enabled or graph_name not in self.cuda_graphs:
            return None
        
        try:
            graph_info = self.cuda_graphs[graph_name]
            graph = graph_info["graph"]
            graph_input = graph_info["input"]
            graph_output = graph_info["output"]
            
            # Copy inputs to graph input tensor
            graph_input.copy_(inputs)
            
            # Replay graph
            graph.replay()
            
            # Return copy of output (graph tensors are reused)
            return graph_output.clone()
            
        except Exception as e:
            self.logger.error(f"CUDA graph execution failed: {e}")
            return None
    
    def create_cuda_stream(self, stream_name: str = "default") -> bool:
        """
        Create CUDA stream for asynchronous operations.
        
        Args:
            stream_name: Name for the CUDA stream
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            stream = torch.cuda.Stream()
            self.streams[stream_name] = stream
            self.logger.info(f"CUDA stream '{stream_name}' created")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create CUDA stream: {e}")
            return False
    
    @contextmanager
    def cuda_stream(self, stream_name: str = "default"):
        """
        Context manager for CUDA stream operations.
        
        Args:
            stream_name: Name of the CUDA stream
        """
        if not self.enabled or stream_name not in self.streams:
            yield
            return
        
        stream = self.streams[stream_name]
        with torch.cuda.stream(stream):
            yield stream
    
    def create_cuda_events(self, event_names: List[str]) -> bool:
        """
        Create CUDA events for synchronization.
        
        Args:
            event_names: Names for CUDA events
            
        Returns:
            Success status
        """
        if not self.enabled:
            return False
        
        try:
            for event_name in event_names:
                event = torch.cuda.Event(enable_timing=True)
                self.events[event_name] = event
            
            self.logger.info(f"CUDA events created: {event_names}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create CUDA events: {e}")
            return False
    
    def synchronize_events(self, start_event: str, end_event: str) -> float:
        """
        Synchronize between CUDA events and measure elapsed time.
        
        Args:
            start_event: Start event name
            end_event: End event name
            
        Returns:
            Elapsed time in milliseconds
        """
        if not self.enabled:
            return 0.0
        
        if start_event not in self.events or end_event not in self.events:
            self.logger.warning("Events not found")
            return 0.0
        
        start = self.events[start_event]
        end = self.events[end_event]
        
        end.synchronize()
        return start.elapsed_time(end)
    
    def optimize_model_for_cuda(self, model: nn.Module) -> nn.Module:
        """
        Apply CUDA-specific optimizations to model.
        
        Args:
            model: PyTorch model
            
        Returns:
            CUDA-optimized model
        """
        if not self.enabled:
            return model
        
        try:
            self.logger.info("Applying CUDA-specific model optimizations")
            
            # Move to CUDA device
            model = model.to(self.device)
            model.eval()
            
            # Enable mixed precision if configured
            if self.config and hasattr(self.config.device, 'use_fp16') and self.config.device.use_fp16:
                model = model.half()
                self.logger.info("FP16 mixed precision enabled")
            
            # Optimize for channels_last memory format (for conv models)
            if self._is_conv_model(model):
                model = model.to(memory_format=torch.channels_last)
                self.logger.info("Channels-last memory format applied")
            
            # JIT compile if requested
            if self.config and hasattr(self.config.device, 'use_torch_compile') and self.config.device.use_torch_compile:
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    self.logger.info("Torch compilation applied")
                except Exception as e:
                    self.logger.warning(f"Torch compilation failed: {e}")
            
            self.logger.info("CUDA model optimization completed")
            return model
            
        except Exception as e:
            self.logger.error(f"CUDA model optimization failed: {e}")
            return model
    
    def _is_conv_model(self, model: nn.Module) -> bool:
        """Check if model contains convolutional layers."""
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d)):
                return True
        return False
    
    def benchmark_cuda_optimizations(self,
                                   model: nn.Module,
                                   example_inputs: torch.Tensor,
                                   iterations: int = 100,
                                   use_cuda_graph: bool = True) -> Dict[str, Any]:
        """
        Benchmark CUDA optimizations.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs
            iterations: Number of iterations
            use_cuda_graph: Whether to test CUDA graphs
            
        Returns:
            Benchmark results
        """
        if not self.enabled:
            return {"error": "CUDA not enabled"}
        
        results = {}
        
        # Move to CUDA
        model = model.to(self.device)
        example_inputs = example_inputs.to(self.device)
        model.eval()
        
        # Benchmark standard inference
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(example_inputs)
        
        torch.cuda.synchronize()
        standard_time = time.time() - start_time
        
        results["standard_inference"] = {
            "time_s": standard_time,
            "fps": iterations / standard_time
        }
        
        # Benchmark with CUDA graph
        if use_cuda_graph:
            graph_name = "benchmark_graph"
            success = self.create_cuda_graph(model, example_inputs, graph_name)
            
            if success:
                torch.cuda.synchronize()
                start_time = time.time()
                
                for _ in range(iterations):
                    _ = self.run_cuda_graph(graph_name, example_inputs)
                
                torch.cuda.synchronize()
                graph_time = time.time() - start_time
                
                results["cuda_graph_inference"] = {
                    "time_s": graph_time,
                    "fps": iterations / graph_time,
                    "speedup": standard_time / graph_time
                }
                
                # Clean up graph
                del self.cuda_graphs[graph_name]
            else:
                results["cuda_graph_inference"] = {"error": "Failed to create CUDA graph"}
        
        # Memory usage
        results["memory_usage"] = {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(self.device) / (1024**2)
        }
        
        return results
    
    def get_cuda_info(self) -> Dict[str, Any]:
        """Get CUDA device information."""
        if not self.enabled:
            return {"cuda_available": False}
        
        props = torch.cuda.get_device_properties(self.device)
        
        info = {
            "cuda_available": True,
            "device": str(self.device),
            "device_name": props.name,
            "compute_capability": props.major * 10 + props.minor,
            "total_memory_mb": props.total_memory / (1024**2),
            "multiprocessor_count": props.multi_processor_count,
            "current_memory": {
                "allocated_mb": torch.cuda.memory_allocated(self.device) / (1024**2),
                "reserved_mb": torch.cuda.memory_reserved(self.device) / (1024**2)
            },
            "optimizations": {
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
                "cuda_graphs_available": torch.cuda.is_available()
            }
        }
        
        return info
    
    def cleanup(self) -> None:
        """Cleanup CUDA resources."""
        if not self.enabled:
            return
        
        self.logger.info("Cleaning up CUDA resources")
        
        # Clear graphs
        self.cuda_graphs.clear()
        
        # Clear streams and events
        self.streams.clear()
        self.events.clear()
        
        # Clear CUDA cache
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "captures_underway" not in str(e):
                logger.warning(f"Failed to clear CUDA cache during cleanup: {e}")
        torch.cuda.synchronize()
        
        self.logger.info("CUDA cleanup completed")


class CUDAModelWrapper:
    """
    Wrapper for CUDA-optimized models with graph support.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 cuda_optimizer: CUDAOptimizer,
                 use_cuda_graph: bool = False,
                 graph_input_shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize CUDA model wrapper.
        
        Args:
            model: PyTorch model
            cuda_optimizer: CUDA optimizer instance
            use_cuda_graph: Whether to use CUDA graphs
            graph_input_shape: Input shape for CUDA graph
        """
        self.model = model
        self.cuda_optimizer = cuda_optimizer
        self.use_cuda_graph = use_cuda_graph
        self.graph_name = "wrapper_graph"
        self.graph_ready = False
        
        self.logger = logging.getLogger(f"{__name__}.CUDAModelWrapper")
        
        # Optimize model
        if cuda_optimizer.enabled:
            self.model = cuda_optimizer.optimize_model_for_cuda(model)
            
            # Create CUDA graph if requested
            if use_cuda_graph and graph_input_shape:
                example_input = torch.randn(graph_input_shape, device=cuda_optimizer.device)
                self.graph_ready = cuda_optimizer.create_cuda_graph(
                    self.model, example_input, self.graph_name
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional CUDA graph acceleration.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        if not self.cuda_optimizer.enabled:
            return self.model(x)
        
        # Try CUDA graph first
        if self.use_cuda_graph and self.graph_ready:
            result = self.cuda_optimizer.run_cuda_graph(self.graph_name, x)
            if result is not None:
                return result
            else:
                self.logger.warning("CUDA graph failed, falling back to standard inference")
        
        # Standard inference
        return self.model(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make the wrapper callable."""
        return self.forward(x)
    
    def eval(self):
        """Set to evaluation mode."""
        self.model.eval()
        return self
    
    def to(self, device):
        """Move to device."""
        self.model = self.model.to(device)
        return self
    
    def cuda(self):
        """Move to CUDA."""
        return self.to(self.cuda_optimizer.device)


def enable_cuda_optimizations(config: Optional[InferenceConfig] = None) -> CUDAOptimizer:
    """
    Enable CUDA optimizations globally.
    
    Args:
        config: Inference configuration
        
    Returns:
        CUDA optimizer instance
    """
    # Import here to avoid issues with mocking in tests
    optimizer = CUDAOptimizer(config)
    optimizer.apply_cuda_optimizations()  # Actually apply some optimizations
    return optimizer


# Global CUDA optimizer instance
_global_cuda_optimizer: Optional[CUDAOptimizer] = None


def get_cuda_optimizer() -> CUDAOptimizer:
    """Get global CUDA optimizer instance."""
    global _global_cuda_optimizer
    if _global_cuda_optimizer is None:
        _global_cuda_optimizer = CUDAOptimizer()
    return _global_cuda_optimizer
