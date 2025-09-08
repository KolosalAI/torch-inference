"""
Base model interface and abstract classes for the PyTorch inference framework.

This module defines the core interfaces that all model implementations must follow,
ensuring consistency and interoperability across different model types.
Includes Numba JIT acceleration for computational optimizations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging
import os
import time
import asyncio
import pickle
import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from weakref import WeakValueDictionary
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from collections import deque, defaultdict

from ..core.config import InferenceConfig

# Import Numba optimizer for JIT acceleration
try:
    from ..optimizers.numba_optimizer import NumbaOptimizer
    NUMBA_OPTIMIZER_AVAILABLE = True
except ImportError:
    NUMBA_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import security mitigations
try:
    from .security import PyTorchSecurityMitigation, ECDSASecurityMitigation
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logger.warning("Security mitigations not available")

# Initialize security if available (lazy initialization to prevent hanging)
_pytorch_security = None

def _get_pytorch_security():
    """Lazy initialization of PyTorch security for base models."""
    global _pytorch_security
    if SECURITY_AVAILABLE and _pytorch_security is None:
        try:
            _pytorch_security = PyTorchSecurityMitigation()
            logger.info("Security mitigations initialized for base models")
        except Exception as e:
            logger.warning(f"Failed to initialize base model security: {e}")
            _pytorch_security = False  # Mark as failed to avoid retrying
    return _pytorch_security if _pytorch_security is not False else None


@dataclass
class ModelMetadata:
    """Metadata about a model."""
    name: str
    version: str
    model_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelInferenceError(Exception):
    """Exception raised when model inference fails."""
    pass


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.
    
    This class defines the interface that all model implementations must follow,
    ensuring consistency and proper resource management.
    Includes Numba JIT acceleration capabilities for computational optimizations.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = config.device.get_torch_device()
        self.model: Optional[nn.Module] = None
        self.metadata: Optional[ModelMetadata] = None
        self._is_loaded = False
        self._compiled_model = None
        
        # Setup logging first
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Advanced caching and optimization state
        self._tensor_cache = WeakValueDictionary()
        self._preprocessing_cache = {}
        self._postprocessing_cache = {}
        self._compiled_functions = {}
        self._channels_last_enabled = False
        
        # Performance monitoring
        self._performance_stats = defaultdict(list)
        self._error_counts = defaultdict(int)
        
        # Async processing
        self._batch_queue = deque()
        self._batch_size = config.batch.batch_size
        self._batch_timeout = getattr(config.batch, 'timeout_ms', 50) / 1000.0
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize Numba optimizer for JIT acceleration
        self.numba_optimizer = None
        self.numba_ops = {}
        self._numba_enabled = False
        if NUMBA_OPTIMIZER_AVAILABLE:
            try:
                self.numba_optimizer = NumbaOptimizer()
                if self.numba_optimizer.is_available():
                    self.numba_ops = self.numba_optimizer.create_optimized_operations()
                    self._numba_enabled = True
                    self.logger.debug("Numba JIT acceleration enabled for model")
                else:
                    self.logger.debug("Numba available but not functional")
            except Exception as e:
                self.logger.debug(f"Failed to initialize Numba optimizer: {e}")
    
    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load the model from the given path.
        
        Args:
            model_path: Path to the model file
            
        Raises:
            ModelLoadError: If model loading fails
        """
        pass
    
    @abstractmethod
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """
        Preprocess inputs before inference.
        
        Args:
            inputs: Raw inputs (images, text, etc.)
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        pass
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass through the model.
        
        Args:
            inputs: Preprocessed input tensor
            
        Returns:
            Raw model outputs
        """
        pass
    
    @abstractmethod
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed outputs in the desired format
        """
        pass
    
    # Enhanced caching and tensor management methods
    @lru_cache(maxsize=128)
    def _get_tensor_shape_info(self, shape_tuple: Tuple[int, ...]) -> Dict[str, Any]:
        """Cache tensor shape computations."""
        return {
            'numel': torch.Size(shape_tuple).numel(),
            'ndim': len(shape_tuple),
            'memory_size': torch.Size(shape_tuple).numel() * 4,  # Assume float32
        }
    
    def _get_cached_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Get or create cached tensor for common shapes."""
        cache_key = (shape, dtype, str(self.device))
        
        if cache_key in self._tensor_cache:
            return self._tensor_cache[cache_key]
        
        # Create new tensor and cache it
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        self._tensor_cache[cache_key] = tensor
        return tensor
    
    def _compute_input_hash(self, inputs: Any) -> str:
        """Compute hash for input caching."""
        try:
            if isinstance(inputs, torch.Tensor):
                return f"tensor_{inputs.shape}_{inputs.dtype}_{inputs.sum().item():.6f}"
            elif isinstance(inputs, (list, tuple)):
                return f"sequence_{len(inputs)}_{hash(str(inputs))}"
            else:
                return f"other_{hash(str(inputs))}"
        except Exception:
            # Fallback for unhashable inputs
            return f"fallback_{id(inputs)}"
    
    def _compute_tensor_hash(self, tensor: torch.Tensor) -> str:
        """Compute hash for tensor caching."""
        try:
            return f"tensor_{tensor.shape}_{tensor.dtype}_{tensor.sum().item():.6f}"
        except Exception:
            return f"tensor_fallback_{id(tensor)}"
    
    @contextmanager
    def _performance_monitor(self, operation: str):
        """Context manager for performance monitoring."""
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = torch.cuda.memory_allocated(self.device) if self.device.type == 'cuda' else 0
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self._performance_stats[f"{operation}_time"].append(duration)
            if self.device.type == 'cuda':
                self._performance_stats[f"{operation}_memory"].append(memory_delta)
    
    def _get_inference_context(self):
        """Get optimal inference context based on configuration."""
        if self.device.type == 'cuda' and self.config.device.use_fp16:
            # Use float16 for better compatibility - BFloat16 can cause issues with some operations
            dtype = torch.float16
            return torch.amp.autocast('cuda', dtype=dtype)
        else:
            return torch.inference_mode()
    
    def predict(self, inputs: Any) -> Any:
        """
        Complete prediction pipeline: preprocess -> forward -> postprocess.
        Enhanced with caching and performance monitoring.
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Final predictions
        """
        if not self._is_loaded:
            raise ModelInferenceError("Model not loaded. Call load_model() first.")
        
        with self._performance_monitor("predict"):
            try:
                # Cache preprocessing results for identical inputs
                input_hash = self._compute_input_hash(inputs)
                if input_hash in self._preprocessing_cache:
                    preprocessed_inputs = self._preprocessing_cache[input_hash]
                else:
                    with self._performance_monitor("preprocess"):
                        preprocessed_inputs = self.preprocess(inputs)
                    if len(self._preprocessing_cache) < 100:  # Limit cache size
                        self._preprocessing_cache[input_hash] = preprocessed_inputs
                
                # Optimized inference context
                with self._performance_monitor("forward"):
                    with self._get_inference_context():
                        # Channel-last conversion if enabled
                        if hasattr(self, '_channels_last_enabled') and self._channels_last_enabled:
                            if preprocessed_inputs.dim() == 4:  # NCHW tensor
                                preprocessed_inputs = preprocessed_inputs.to(memory_format=torch.channels_last)
                        
                        try:
                            raw_outputs = self.forward(preprocessed_inputs)
                        except Exception as e:
                            # Handle compilation errors by falling back to non-compiled model
                            error_msg = str(e)
                            compilation_errors = [
                                "CppCompileError", 
                                "TritonMissing", 
                                "Cannot find a working triton installation",
                                "triton is required",
                                "Dynamo failed"
                            ]
                            
                            # Handle BFloat16 compatibility issues
                            bfloat16_errors = [
                                "Got unsupported ScalarType BFloat16",
                                "BFloat16",
                                "bfloat16",
                                "BFLOAT16"
                            ]
                            
                            if any(error in error_msg for error in compilation_errors) and self._compiled_model is not None:
                                self.logger.warning(f"Torch compilation failed ({error_msg[:100]}...), falling back to non-compiled model")
                                self.config.device.use_torch_compile = False
                                self._compiled_model = None
                                raw_outputs = self.forward(preprocessed_inputs)
                            elif any(error in error_msg for error in bfloat16_errors):
                                self.logger.warning(f"BFloat16 compatibility issue detected ({error_msg[:100]}...), converting to float16/float32")
                                # Convert input to float16 or float32
                                if preprocessed_inputs.dtype == torch.bfloat16:
                                    if self.device.type == 'cuda':
                                        preprocessed_inputs = preprocessed_inputs.to(torch.float16)
                                    else:
                                        preprocessed_inputs = preprocessed_inputs.to(torch.float32)
                                raw_outputs = self.forward(preprocessed_inputs)
                            else:
                                raise
                
                # Cache postprocessing if applicable
                output_hash = self._compute_tensor_hash(raw_outputs)
                if output_hash in self._postprocessing_cache:
                    predictions = self._postprocessing_cache[output_hash]
                else:
                    with self._performance_monitor("postprocess"):
                        predictions = self.postprocess(raw_outputs)
                    if len(self._postprocessing_cache) < 50:
                        self._postprocessing_cache[output_hash] = predictions
                
                # Convert to dict for backward compatibility if needed
                if hasattr(predictions, 'to_dict'):
                    return predictions.to_dict()
                
                return predictions
                
            except Exception as e:
                self._error_counts[type(e).__name__] += 1
                self.logger.error(f"Prediction failed: {e}")
                raise ModelInferenceError(f"Prediction failed: {e}") from e
    
    def predict_batch(self, inputs: List[Any]) -> List[Any]:
        """
        Batch prediction with automatic batching.
        
        Args:
            inputs: List of raw inputs
            
        Returns:
            List of predictions
        """
        if not inputs:
            return []
        
        batch_size = self.config.batch.batch_size
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            if len(batch) == 1:
                # Single item
                result = self.predict(batch[0])
                results.append(result)
            else:
                # True batch processing
                batch_result = self.predict_batch_internal(batch)
                results.extend(batch_result)
        
        return results
    
    def predict_batch_optimized(self, inputs: List[Any]) -> List[Any]:
        """Optimized batch prediction with dynamic batching."""
        if not inputs:
            return []
        
        # Sort by input complexity for better batching
        sorted_inputs = self._sort_inputs_by_complexity(inputs)
        
        results = [None] * len(inputs)
        original_indices = [i for i in range(len(inputs))]
        
        # Process in optimal batch sizes
        optimal_batch_size = self._calculate_optimal_batch_size(inputs[0])
        
        for i in range(0, len(sorted_inputs), optimal_batch_size):
            batch = sorted_inputs[i:i + optimal_batch_size]
            batch_indices = original_indices[i:i + optimal_batch_size]
            
            # True batch processing
            batch_results = self._process_true_batch(batch)
            
            # Restore original order
            for idx, result in zip(batch_indices, batch_results):
                results[idx] = result
        
        return results
    
    async def predict_async(self, inputs: Any) -> Any:
        """Asynchronous prediction with automatic batching."""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor, self.predict, inputs
        )
    
    def _calculate_optimal_batch_size(self, sample_input: Any) -> int:
        """Calculate optimal batch size based on memory and input complexity."""
        base_batch_size = self.config.batch.batch_size
        
        if self.device.type == 'cuda':
            try:
                # Estimate memory usage per sample
                sample_tensor = self.preprocess(sample_input)
                memory_per_sample = sample_tensor.numel() * sample_tensor.element_size()
                
                # Available memory (conservative estimate)
                available_memory = torch.cuda.get_device_properties(self.device).total_memory * 0.3
                
                # Calculate memory-bound batch size
                memory_batch_size = max(1, int(available_memory / (memory_per_sample * 4)))  # 4x safety factor
                
                return min(base_batch_size, memory_batch_size)
            except Exception as e:
                self.logger.debug(f"Optimal batch size calculation failed: {e}")
        
        return base_batch_size
    
    def _sort_inputs_by_complexity(self, inputs: List[Any]) -> List[Any]:
        """Sort inputs by complexity for better batching efficiency."""
        try:
            # Simple complexity metric based on input size
            def complexity(inp):
                if hasattr(inp, 'size'):
                    return inp.size if isinstance(inp.size, int) else sum(inp.size)
                elif isinstance(inp, (list, tuple)):
                    return len(inp)
                else:
                    return len(str(inp))
            
            return sorted(inputs, key=complexity)
        except Exception:
            return inputs  # Return original order if sorting fails
    
    def _process_true_batch(self, batch: List[Any]) -> List[Any]:
        """Process a true batch. Override in subclasses for better batch processing."""
        return self.predict_batch_internal(batch)
    
    def predict_batch_internal(self, inputs: List[Any]) -> List[Any]:
        """
        Internal batch processing method. Override for true batch processing.
        
        Args:
            inputs: List of raw inputs (batch)
            
        Returns:
            List of predictions
        """
        # Default implementation: process individually
        return [self.predict(inp) for inp in inputs]
    
    def warmup(self, num_iterations: int = None) -> None:
        """
        Warmup the model with dummy inputs.
        
        Args:
            num_iterations: Number of warmup iterations
        """
        if num_iterations is None:
            num_iterations = self.config.performance.warmup_iterations
        
        if not self._is_loaded:
            self.logger.warning("Model not loaded, skipping warmup")
            return
        
        self.logger.info(f"Warming up model with {num_iterations} iterations")
        
        try:
            # Create dummy input based on preprocessing config
            dummy_input = self._create_dummy_input()
            
            for i in range(num_iterations):
                try:
                    with torch.no_grad():
                        _ = self.forward(dummy_input)
                except Exception as e:
                    error_msg = str(e)
                    self.logger.warning(f"Warmup iteration {i+1} failed: {e}")
                    
                    # If iteration fails due to compilation issues, disable compilation and retry
                    if ("triton" in error_msg.lower() or "inductor" in error_msg.lower() or 
                        "CppCompileError" in error_msg):
                        self.logger.warning("Disabling torch.compile due to compilation error")
                        self.config.device.use_torch_compile = False
                        self._compiled_model = None
                        try:
                            with torch.no_grad():
                                _ = self.forward(dummy_input)
                        except Exception as e2:
                            self.logger.error(f"Warmup failed even without compilation: {e2}")
                            break
                    else:
                        # For other errors, just continue
                        continue
            
            self.logger.info("Model warmup completed")
            
        except Exception as e:
            self.logger.warning(f"Warmup failed: {e}. Model may still work for inference.")
    
    def compile_model(self) -> None:
        """Compile the model using torch.compile for optimization."""
        if not self._is_loaded or not self.config.device.use_torch_compile:
            return
        
        if not hasattr(torch, 'compile'):
            self.logger.warning("torch.compile not available, skipping compilation")
            return
        
        try:
            self.logger.info("Compiling model with torch.compile")
            
            # ROI Optimization 2: Choose optimal compile mode based on config
            compile_mode = getattr(self.config.device, 'compile_mode', 'reduce-overhead')
            if compile_mode not in ['max-autotune', 'reduce-overhead', 'default']:
                compile_mode = 'reduce-overhead'  # Safe default
            
            # Enhanced compile options for better performance
            compile_options = {
                'mode': compile_mode,
                'fullgraph': False,  # More compatible than True
                'dynamic': False,    # Static shapes for better optimization
            }
            
            # Add backend-specific optimizations
            if self.device.type == 'cuda':
                compile_options['backend'] = 'inductor'
            
            self._compiled_model = torch.compile(self.model, **compile_options)
            self.logger.info(f"Model compilation completed with mode '{compile_mode}'")
            
        except Exception as e:
            error_msg = str(e)
            compilation_errors = [
                "triton", "inductor", "TritonMissing", 
                "Cannot find a working triton installation",
                "triton is required", "CppCompileError", "Dynamo failed"
            ]
            
            if any(error in error_msg.lower() for error in compilation_errors):
                self.logger.warning(f"Model compilation failed due to missing dependencies: {error_msg[:150]}... Disabling torch.compile.")
                # Disable torch.compile for this configuration
                self.config.device.use_torch_compile = False
            else:
                self.logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
            # Don't raise the exception, just continue without compilation
    
    def get_model_for_inference(self) -> nn.Module:
        """Get the model instance to use for inference (compiled or original)."""
        return self._compiled_model if self._compiled_model is not None else self.model
    
    def optimize_for_inference(self) -> None:
        """Apply comprehensive optimizations for maximum inference performance."""
        if not self._is_loaded:
            return
        
        # ROI Optimization 0: Set to evaluation mode (critical for BN/Dropout)
        self.model.eval()
        
        # Move to target device
        self.model.to(self.device)
        
        # ROI Optimization 1: Enhanced TF32 and Tensor Core optimizations
        if self.device.type == 'cuda':
            self._optimize_gpu_compute()
                
        # ROI Optimization 2: Advanced CUDNN optimizations
        if torch.backends.cudnn.is_available() and self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            # Enable additional CUDNN optimizations for newer versions
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            self.logger.info("Enhanced CUDNN optimizations enabled")
        
        # ROI Optimization 3: Memory format optimization with fallback
        if self._has_conv_layers():
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                self._channels_last_enabled = True
                self.logger.info("Enabled channels_last memory format")
            except Exception as e:
                self.logger.debug(f"Channels-last optimization failed: {e}")
                self._channels_last_enabled = False
        
        # ROI Optimization 4: Intelligent CPU thread management
        if self.device.type == 'cpu':
            self._optimize_cpu_threads()
        
        # ROI Optimization 5: Advanced memory management
        if self.device.type == 'cuda':
            self._optimize_cuda_memory()
        
        # ROI Optimization 6: Model fusion and optimization
        self._apply_model_fusion()
        
        # ROI Optimization 7: Compilation with better error handling
        if self.config.device.use_torch_compile:
            self.compile_model()
        
        # ROI Optimization 8: Apply quantization if requested
        if getattr(self.config.device, 'use_quantization', False):
            self._apply_quantization()
        
        self.logger.info("Comprehensive model optimization for inference completed")
    
    def _optimize_gpu_compute(self) -> None:
        """Enhanced GPU compute optimizations."""
        try:
            device_idx = self.device.index if self.device.index is not None else 0
            major, minor = torch.cuda.get_device_capability(device_idx)
            
            if major >= 8:  # Ampere+
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision('high')
                # Enable Flash Attention if available
                if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
                    torch.backends.cuda.flash_sdp_enabled = True
                self.logger.info(f"TF32 and Flash Attention enabled for Ampere+ GPU (CC {major}.{minor})")
            elif major >= 7:  # Volta/Turing
                torch.set_float32_matmul_precision('high')
                # Optimize for Tensor Cores
                torch.backends.cuda.matmul.allow_tf32 = False  # Use FP16 instead
                self.logger.info(f"Tensor Core optimizations enabled for Volta/Turing GPU (CC {major}.{minor})")
            else:
                torch.set_float32_matmul_precision('highest')
                self.logger.info(f"High precision mode for older GPU (CC {major}.{minor})")
                
        except Exception as e:
            self.logger.debug(f"GPU optimization setup failed: {e}")
    
    def _optimize_cpu_threads(self) -> None:
        """Optimize CPU threading for inference workloads."""
        cpu_count = os.cpu_count() or 4
        
        # Dynamic thread allocation based on model size
        if hasattr(self.model, 'parameters'):
            param_count = sum(p.numel() for p in self.model.parameters())
            if param_count > 100_000_000:  # Large models
                optimal_threads = max(2, min(16, cpu_count))
            else:  # Smaller models
                optimal_threads = max(1, min(8, int(cpu_count * 0.75)))
        else:
            optimal_threads = max(1, min(8, int(cpu_count * 0.75)))
        
        torch.set_num_threads(optimal_threads)
        try:
            torch.set_num_interop_threads(1)
            self.logger.info(f"Optimized CPU threads: {optimal_threads} intra, 1 inter")
        except RuntimeError as e:
            if "cannot set number of interop threads" in str(e):
                self.logger.debug("Interop threads already configured")
            else:
                raise
    
    def _optimize_cuda_memory(self) -> None:
        """Advanced CUDA memory optimizations."""
        try:
            # Memory pool configuration
            memory_fraction = getattr(self.config.device, 'memory_fraction', 0.85)  # More conservative
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=self.device)
            
            # Advanced memory allocator settings
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
                                'max_split_size_mb:512,roundup_power2_divisions:16')
            
            # Enable memory pool caching
            torch.cuda.empty_cache()
            
            # Pre-allocate small buffer to stabilize allocations
            _ = torch.zeros(1024, device=self.device)
            
            self.logger.info(f"CUDA memory optimized - fraction: {memory_fraction}")
        except Exception as e:
            self.logger.debug(f"CUDA memory optimization failed: {e}")
    
    def _apply_model_fusion(self) -> None:
        """Apply model-level fusion optimizations."""
        try:
            # Check for fusible conv-bn-relu patterns
            if self._has_fusible_patterns():
                # Modern PyTorch handles this through torch.compile
                self.logger.info("Model has fusible patterns - will benefit from compilation")
            
            # Apply layer-specific optimizations
            self._optimize_model_layers()
            
        except Exception as e:
            self.logger.debug(f"Model fusion failed: {e}")
    
    def _has_fusible_patterns(self) -> bool:
        """Detect common fusible patterns in the model."""
        if not self.model:
            return False
        
        modules = list(self.model.modules())
        for i, module in enumerate(modules[:-2]):
            if isinstance(module, nn.Conv2d):
                # Check for Conv->BN->ReLU pattern
                if (i + 2 < len(modules) and 
                    isinstance(modules[i+1], nn.BatchNorm2d) and
                    isinstance(modules[i+2], (nn.ReLU, nn.ReLU6))):
                    return True
        return False
    
    def _optimize_model_layers(self) -> None:
        """Apply layer-specific optimizations."""
        for module in self.model.modules():
            # Optimize dropout layers for inference
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.eval()  # Ensure dropout is disabled
            
            # Optimize batch norm layers
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                # Fuse batch norm if possible
                if hasattr(module, 'track_running_stats') and module.track_running_stats:
                    module.momentum = 0.0  # Freeze running statistics
    
    def _can_fuse_operations(self) -> bool:
        """Check if model has operations that can be fused."""
        try:
            fused_ops = ['Conv', 'BatchNorm', 'ReLU', 'Linear']
            model_ops = [str(type(m).__name__) for m in self.model.modules()]
            return any(op in ' '.join(model_ops) for op in fused_ops)
        except Exception:
            return False
    
    def _apply_quantization(self) -> None:
        """Apply dynamic quantization for CPU inference speedup."""
        try:
            if self.device.type == 'cpu':
                # Apply dynamic quantization for CPU
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, 
                    {torch.nn.Linear, torch.nn.Conv2d}, 
                    dtype=torch.qint8
                )
                self.model = quantized_model
                self.logger.info("Applied dynamic quantization for CPU inference")
            else:
                self.logger.debug("Quantization currently only supported for CPU inference")
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
    
    # Model state management and persistence
    def save_optimized_state(self, path: Union[str, Path]) -> None:
        """Save optimized model state for faster loading."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state': self.model.state_dict() if self.model else None,
            'config': self.config,
            'metadata': self.metadata,
            'optimization_flags': {
                'channels_last_enabled': getattr(self, '_channels_last_enabled', False),
                'numba_enabled': self._numba_enabled,
                'compiled': self._compiled_model is not None,
            },
            'cache_stats': {
                'preprocessing_cache_size': len(self._preprocessing_cache),
                'postprocessing_cache_size': len(self._postprocessing_cache),
            }
        }
        
        # Save state
        torch.save(state, path / 'optimized_state.pth')
        
        # Save human-readable info
        info = {
            'optimization_applied': True,
            'device': str(self.device),
            'memory_usage': self.get_memory_usage(),
        }
        with open(path / 'optimization_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Saved optimized state to {path}")
    
    def load_optimized_state(self, path: Union[str, Path]) -> bool:
        """Load previously optimized model state."""
        path = Path(path)
        state_file = path / 'optimized_state.pth'
        
        if not state_file.exists():
            return False
        
        try:
            state = torch.load(state_file, map_location=self.device, weights_only=False)
            
            # Restore optimization flags
            opt_flags = state.get('optimization_flags', {})
            self._channels_last_enabled = opt_flags.get('channels_last_enabled', False)
            self._numba_enabled = opt_flags.get('numba_enabled', False)
            
            # Restore model state if available
            if state.get('model_state') and self.model:
                self.model.load_state_dict(state['model_state'])
            
            self.logger.info(f"Loaded optimized state from {path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load optimized state: {e}")
            return False
    
    # Performance monitoring and statistics
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        stats = {}
        
        for operation, times in self._performance_stats.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'p95': sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times),
                }
        
        stats['error_counts'] = dict(self._error_counts)
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._performance_stats.clear()
        self._error_counts.clear()
        self.logger.info("Performance statistics reset")
    
    def _has_conv_layers(self) -> bool:
        """Check if model has convolutional layers that would benefit from channels_last."""
        try:
            return any(isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) for m in self.model.modules())
        except Exception:
            return False
    
    def _create_dummy_input(self) -> torch.Tensor:
        """Create dummy input for warmup. Override in subclasses."""
        # Determine the appropriate dtype based on model configuration
        target_dtype = torch.float32
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Try to get dtype from the first parameter of the model
                first_param = next(iter(self.model.parameters()), None)
                if first_param is not None:
                    target_dtype = first_param.dtype
            except Exception:
                # Fallback: check if FP16 is enabled in config
                if (hasattr(self.config, 'device') and 
                    getattr(self.config.device, 'use_fp16', False) and 
                    self.device.type == 'cuda'):
                    # Use float16 for better compatibility - avoid BFloat16 for input tensors
                    target_dtype = torch.float16
        
        # Try to infer from model structure if possible
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Look at first layer to infer input shape
                first_layer = None
                for module in self.model.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)):
                        first_layer = module
                        break
                
                if isinstance(first_layer, torch.nn.Conv2d):
                    # CNN model - create image-like input
                    in_channels = first_layer.in_channels
                    # Use reasonable default image size
                    return torch.randn(1, in_channels, 64, 64, device=self.device, dtype=target_dtype)
                elif isinstance(first_layer, torch.nn.Conv1d):
                    # 1D CNN - create sequence-like input
                    in_channels = first_layer.in_channels
                    return torch.randn(1, in_channels, 64, device=self.device, dtype=target_dtype)
                elif isinstance(first_layer, torch.nn.Linear):
                    # Linear model - create flat input
                    in_features = first_layer.in_features
                    return torch.randn(1, in_features, device=self.device, dtype=target_dtype)
            except Exception as e:
                logger.debug(f"Failed to infer input shape from model: {e}")
        
        # Default implementation for image models using config
        if hasattr(self.config, 'preprocessing') and hasattr(self.config.preprocessing, 'input_size'):
            height, width = self.config.preprocessing.input_size
            return torch.randn(1, 3, height, width, device=self.device, dtype=target_dtype)
        else:
            # Generic dummy input - use a safe shape that works for most models
            return torch.randn(1, 3, 64, 64, device=self.device, dtype=target_dtype)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        memory_info = {}
        
        if self.device.type == 'cuda':
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            memory_info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        
        # CPU memory would require psutil
        try:
            import psutil
            process = psutil.Process()
            memory_info['cpu_memory_mb'] = process.memory_info().rss / (1024 ** 2)
        except ImportError:
            pass
        
        # Add model parameter info if available
        if self._is_loaded and self.model:
            try:
                # Handle both real models and Mock objects
                if hasattr(self.model, 'parameters') and callable(self.model.parameters):
                    total_params = sum(p.numel() for p in self.model.parameters())
                    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    memory_info["total_params"] = total_params
                    memory_info["trainable_params"] = trainable_params
                    
                    # Estimate model size (very rough approximation)
                    # Assume 4 bytes per parameter (float32)
                    memory_info["model_size_mb"] = total_params * 4 / (1024 ** 2)
            except (TypeError, AttributeError):
                # Skip parameter counting for Mock objects or other types
                pass
        
        return memory_info
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "captures_underway" in str(e):
                    # Skip cache clearing if CUDA graph capture is active
                    logger.debug("Skipping CUDA cache clear due to active graph capture")
                else:
                    logger.warning(f"Failed to clear CUDA cache during cleanup: {e}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @is_loaded.setter
    def is_loaded(self, value: bool):
        """Set model loaded status."""
        self._is_loaded = value
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "loaded": self._is_loaded,
            "device": str(self.device),
            "config": self.config,
        }
        
        if self.metadata:
            # Convert metadata to dict for compatibility
            if hasattr(self.metadata, '__dict__'):
                info["metadata"] = self.metadata.__dict__.copy()
            else:
                info["metadata"] = {
                    "model_type": getattr(self.metadata, 'model_type', 'pytorch'),
                    "input_shape": getattr(self.metadata, 'input_shape', None),
                    "output_shape": getattr(self.metadata, 'output_shape', None),
                    "num_parameters": getattr(self.metadata, 'num_parameters', None),
                    "framework_version": getattr(self.metadata, 'framework_version', None)
                }
        
        if self._is_loaded:
            info["memory_usage"] = self.get_memory_usage()
            
            # Model parameters count
            if self.model:
                try:
                    # Handle both real models and Mock objects
                    if hasattr(self.model, 'parameters') and callable(self.model.parameters):
                        total_params = sum(p.numel() for p in self.model.parameters())
                        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                        info["total_parameters"] = total_params
                        info["trainable_parameters"] = trainable_params
                except (TypeError, AttributeError):
                    # Skip parameter counting for Mock objects or other types
                    pass
        
        return info


class ModelManager:
    """
    Manager class for handling multiple model instances and lifecycle.
    """
    
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self.logger = logging.getLogger(f"{__name__}.ModelManager")
        
        # Import downloader here to avoid circular imports
        self._downloader = None
    
    def register_model(self, name: str, model: BaseModel) -> None:
        """Register a model instance."""
        if name in self._models:
            self.logger.warning(f"Model '{name}' already exists, replacing")
        
        self._models[name] = model
        self.logger.info(f"Registered model '{name}'")
    
    def get_model(self, name: str) -> BaseModel:
        """Get a registered model."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
    
    def is_model_loaded(self, name: str) -> bool:
        """Check if a model is loaded."""
        return name in self._models and self._models[name].is_loaded
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded models."""
        return [name for name, model in self._models.items() if model.is_loaded]
    
    def load_model(self, model_path: Union[str, Path], config: InferenceConfig) -> BaseModel:
        """Load a model with security mitigations."""
        # Create appropriate model adapter
        from ..adapters.model_adapters import ModelAdapterFactory
        
        adapter = ModelAdapterFactory.create_adapter(model_path, config)
        
        # Register the model in the manager
        self.register_model(str(model_path), adapter)
        
        # Use secure model loading if security is available
        pytorch_security = _get_pytorch_security()
        if pytorch_security:
            with pytorch_security.secure_torch_context():
                adapter.load_model(model_path)
                adapter.optimize_for_inference()
                adapter.warmup()
            self.logger.info(f"Model '{model_path}' loaded with security mitigations")
        else:
            adapter.load_model(model_path)
            adapter.optimize_for_inference()
            adapter.warmup()
            self.logger.warning(f"Model '{model_path}' loaded without security mitigations")
        
        return adapter
    
    def load_registered_model(self, name: str, model_path: Union[str, Path]) -> None:
        """Load a registered model with security mitigations."""
        model = self.get_model(name)
        
        # Use secure model loading if security is available
        pytorch_security = _get_pytorch_security()
        if pytorch_security:
            with pytorch_security.secure_torch_context():
                model.load_model(model_path)
                model.optimize_for_inference()
                model.warmup()
            self.logger.info(f"Model '{name}' loaded with security mitigations")
        else:
            model.load_model(model_path)
            model.optimize_for_inference()
            model.warmup()
            self.logger.warning(f"Model '{name}' loaded without security mitigations")
    
    def unload_model(self, name: str) -> None:
        """Unload and cleanup a model."""
        if name in self._models:
            self._models[name].cleanup()
            del self._models[name]
            self.logger.info(f"Unloaded model '{name}'")
    
    def cleanup_all(self) -> None:
        """Cleanup all models."""
        for name in list(self._models.keys()):
            self.unload_model(name)
    
    def get_downloader(self):
        """Get the model downloader instance."""
        if self._downloader is None:
            from .model_downloader import get_model_downloader
            self._downloader = get_model_downloader()
        return self._downloader
    
    def download_and_load_model(
        self, 
        source: str, 
        model_id: str, 
        name: str,
        config: Optional['InferenceConfig'] = None,
        **kwargs
    ) -> None:
        """
        Download a model from a source and load it into the framework.
        
        Args:
            source: Model source ('pytorch_hub', 'torchvision', 'huggingface', 'url')
            model_id: Model identifier
            name: Name to register the model with
            config: Inference configuration
            **kwargs: Additional arguments for download
        """
        try:
            # Download the model
            downloader = self.get_downloader()
            
            if source == "pytorch_hub":
                if "/" not in model_id:
                    raise ValueError("PyTorch Hub model_id should be in format 'repo/model'")
                # Parse PyTorch Hub model_id: repo/model or repo:version/model
                parts = model_id.rsplit("/", 1)  # Split from right to get last part as model
                if len(parts) != 2:
                    raise ValueError(f"Invalid PyTorch Hub model_id format: {model_id}. Expected: repo/model or repo:version/model")
                repo = parts[0]
                model = parts[1]
                model_path, model_info = downloader.download_pytorch_hub_model(
                    repo, model, name, **kwargs
                )
            elif source == "torchvision":
                model_path, model_info = downloader.download_torchvision_model(
                    model_id, custom_name=name, **kwargs
                )
            elif source == "huggingface":
                # Filter out kwargs that are not supported by huggingface downloader
                hf_kwargs = {k: v for k, v in kwargs.items() if k in ['task']}
                model_path, model_info = downloader.download_huggingface_model(
                    model_id, custom_name=name, **hf_kwargs
                )
            elif source == "url":
                # Filter out kwargs that are not supported by url downloader
                url_kwargs = {k: v for k, v in kwargs.items() if k in ['task', 'description', 'expected_hash']}
                model_path, model_info = downloader.download_from_url(
                    model_id, name, **url_kwargs
                )
            else:
                raise ValueError(f"Unsupported source: {source}")
            
            # Create appropriate model adapter and load
            if config is None:
                from .config import get_global_config
                config = get_global_config()
            
            # Import here to avoid circular imports
            from ..adapters.model_adapters import ModelAdapterFactory
            
            adapter = ModelAdapterFactory.create_adapter(model_path, config)
            
            # Load the model
            adapter.load_model(model_path)
            
            # Apply post-download optimizations if enabled
            if config.post_download_optimization.enable_optimization:
                self.logger.info(f"Applying post-download optimizations to '{name}'")
                try:
                    from ..optimizers.post_download_optimizer import create_post_download_optimizer
                    
                    # Create post-download optimizer
                    optimizer = create_post_download_optimizer(
                        config.post_download_optimization,
                        config
                    )
                    
                    # Apply optimizations to the loaded model
                    optimized_model, optimization_report = optimizer.optimize_model(
                        adapter.model,
                        name,
                        example_inputs=None,  # Will auto-generate
                        save_path=model_path.parent / "optimized" if hasattr(model_path, 'parent') else None
                    )
                    
                    # Replace the model with the optimized version
                    adapter.model = optimized_model
                    
                    # Log optimization results
                    self.logger.info(f"Post-download optimization completed for '{name}'")
                    self.logger.info(f"Applied optimizations: {optimization_report.get('optimizations_applied', [])}")
                    if 'model_size_metrics' in optimization_report:
                        size_reduction = optimization_report['model_size_metrics'].get('size_reduction_percent', 0)
                        self.logger.info(f"Model size reduction: {size_reduction:.1f}%")
                    
                    # Store optimization report in model metadata
                    if hasattr(adapter, 'metadata') and adapter.metadata:
                        if hasattr(adapter.metadata, '__dict__'):
                            adapter.metadata.__dict__['optimization_report'] = optimization_report
                        else:
                            adapter.metadata.optimization_report = optimization_report
                    
                except Exception as e:
                    self.logger.warning(f"Post-download optimization failed for '{name}': {e}")
                    # Continue with non-optimized model
            
            # Apply standard optimizations and warmup
            adapter.optimize_for_inference()
            adapter.warmup()
            
            # Register the model
            self.register_model(name, adapter)
            
            self.logger.info(f"Successfully downloaded and loaded model '{name}' from {source}")
            
        except Exception as e:
            self.logger.error(f"Failed to download and load model '{name}': {e}")
            raise
    
    def list_available_downloads(self) -> Dict[str, Any]:
        """List available models that can be downloaded."""
        downloader = self.get_downloader()
        return downloader.list_available_models()
    
    def get_download_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a downloadable model."""
        downloader = self.get_downloader()
        info = downloader.get_model_info(model_name)
        return info.__dict__ if info else None


# Global model manager instance
_global_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the global model manager."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = ModelManager()
    return _global_model_manager
