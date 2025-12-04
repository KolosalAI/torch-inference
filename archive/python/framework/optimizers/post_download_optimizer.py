"""
Post-download model optimization module.

This module provides functionality to automatically optimize models after they are downloaded,
including quantization and low-rank tensor optimizations.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import asdict
import copy

import torch
import torch.nn as nn

from ..core.config import InferenceConfig, PostDownloadOptimizationConfig
from .quantization_optimizer import QuantizationOptimizer, quantize_model, QuantizedModelWrapper


logger = logging.getLogger(__name__)


class PostDownloadOptimizer:
    """
    Optimizer that applies various optimizations to models after download.
    
    This includes quantization, tensor factorization, and other compression techniques
    to improve inference performance and reduce model size.
    """
    
    def __init__(self, config: PostDownloadOptimizationConfig, inference_config: Optional[InferenceConfig] = None):
        """
        Initialize the post-download optimizer.
        
        Args:
            config: Post-download optimization configuration
            inference_config: Overall inference configuration
        """
        self.config = config
        self.inference_config = inference_config
        self.logger = logging.getLogger(f"{__name__}.PostDownloadOptimizer")
        
        # Initialize optimizers
        self.quantization_optimizer = QuantizationOptimizer(inference_config)
        
        # Initialize compression suite with compatibility
        try:
            from .model_compression_suite import ModelCompressionSuite, ModelCompressionConfig
            compression_config = ModelCompressionConfig()
            self.compression_suite = ModelCompressionSuite(compression_config)
        except ImportError:
            self.logger.warning("ModelCompressionSuite not available, some optimizations will be disabled")
            self.compression_suite = None
        
        self.logger.info(f"PostDownloadOptimizer initialized with config: {asdict(config)}")
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 * 1024)  # Convert to MB
    
    def _benchmark_optimization(self, model: nn.Module, example_inputs: torch.Tensor, iterations: int = 50) -> Dict[str, Any]:
        """
        Benchmark optimization quality.
        
        Args:
            model: Model to benchmark
            example_inputs: Test inputs
            iterations: Number of benchmark iterations
            
        Returns:
            Dictionary of benchmark metrics
        """
        device = next(model.parameters()).device
        model.eval()
        
        metrics = {}
        
        try:
            # Performance benchmark
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(example_inputs)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            total_time = time.time() - start_time
            
            fps = iterations / total_time
            metrics.update({
                "fps": fps,
                "avg_inference_time_ms": (total_time / iterations) * 1000,
                "speed_improvement": 1.0,  # Baseline, will be compared against original
                "size_reduction": 0.0,  # Will be calculated later
                "accuracy_preserved": True  # Assume true unless proven otherwise
            })
            
        except Exception as e:
            self.logger.warning(f"Benchmarking failed: {e}")
            metrics = {
                "fps": 0.0,
                "avg_inference_time_ms": 0.0,
                "speed_improvement": 1.0,
                "size_reduction": 0.0,
                "accuracy_preserved": True
            }
        
        return metrics
    
    def optimize_model(
        self,
        model: nn.Module,
        model_name: str,
        example_inputs: Optional[torch.Tensor] = None,
        save_path: Optional[Path] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply post-download optimizations to a model.
        
        Args:
            model: The PyTorch model to optimize
            model_name: Name of the model for logging
            example_inputs: Example inputs for optimization (optional)
            save_path: Path to save the optimized model (optional)
            
        Returns:
            Tuple of (optimized_model, optimization_report)
        """
        if not self.config.enable_optimization:
            self.logger.info(f"Post-download optimization disabled for {model_name}")
            return model, {
                "model_name": model_name,
                "optimizations_applied": [], 
                "status": "optimization_disabled",
                "performance_metrics": {},
                "model_size_metrics": {
                    "original_size_mb": self._get_model_size(model),
                    "optimized_size_mb": self._get_model_size(model),
                    "size_reduction_ratio": 0.0,
                    "size_reduction_percent": 0.0
                },
                "errors": [],
                "optimization_time_seconds": 0.0
            }
        
        self.logger.info(f"Starting post-download optimization for model: {model_name}")
        start_time = time.time()
        
        # Initialize optimization tracking
        optimization_report = {
            "model_name": model_name,
            "optimizations_applied": [],
            "performance_metrics": {},
            "model_size_metrics": {},
            "errors": []
        }
        
        # Create a copy of the original model for comparison
        original_model = copy.deepcopy(model)
        optimized_model = model
        
        # Get original model size
        original_size = self._get_model_size(original_model)
        optimization_report["model_size_metrics"]["original_size_mb"] = original_size / (1024 * 1024)
        
        # Generate example inputs if not provided
        if example_inputs is None:
            example_inputs = self._generate_example_inputs(optimized_model)
        
        try:
            # Check if target device is GPU to avoid quantization incompatibility
            target_device = None
            if self.inference_config and hasattr(self.inference_config, 'device'):
                target_device = self.inference_config.device.get_torch_device()
            
            # Apply optimizations in order of preference
            if self.config.auto_select_best_method:
                optimized_model = self._auto_optimize(optimized_model, model_name, example_inputs, optimization_report)
            else:
                # Apply individual optimizations based on configuration
                if self.config.enable_quantization:
                    # Skip quantization for GPU devices as quantized ops don't support CUDA
                    if target_device and target_device.type == 'cuda':
                        self.logger.info(f"Skipping quantization for {model_name} - target device is GPU (CUDA)")
                        optimization_report["errors"].append("Quantization skipped: not compatible with GPU inference")
                    else:
                        optimized_model = self._apply_quantization(optimized_model, model_name, example_inputs, optimization_report)
                
                if self.config.enable_low_rank_optimization or self.config.enable_tensor_factorization:
                    optimized_model = self._apply_tensor_optimization(optimized_model, model_name, optimization_report)
                
                if self.config.enable_structured_pruning:
                    optimized_model = self._apply_structured_pruning(optimized_model, model_name, optimization_report)
        
        except Exception as e:
            self.logger.error(f"Error during optimization of {model_name}: {e}")
            optimization_report["errors"].append(str(e))
            # Return original model if optimization fails
            optimized_model = original_model
        
        # Calculate final metrics
        optimized_size = self._get_model_size(optimized_model)
        optimization_report["model_size_metrics"]["optimized_size_mb"] = optimized_size / (1024 * 1024)
        optimization_report["model_size_metrics"]["size_reduction_ratio"] = (original_size - optimized_size) / original_size
        optimization_report["model_size_metrics"]["size_reduction_percent"] = optimization_report["model_size_metrics"]["size_reduction_ratio"] * 100
        
        # Benchmark if enabled
        if self.config.benchmark_optimizations and example_inputs is not None:
            try:
                benchmark_results = self._benchmark_models(original_model, optimized_model, example_inputs)
                optimization_report["performance_metrics"].update(benchmark_results)
            except Exception as e:
                self.logger.warning(f"Benchmarking failed for {model_name}: {e}")
                optimization_report["errors"].append(f"Benchmarking failed: {str(e)}")
        
        # Save optimized model if requested
        if save_path and self.config.save_optimized_model:
            try:
                self._save_optimized_model(optimized_model, save_path, optimization_report)
            except Exception as e:
                self.logger.warning(f"Failed to save optimized model {model_name}: {e}")
                optimization_report["errors"].append(f"Save failed: {str(e)}")
        
        optimization_time = time.time() - start_time
        optimization_report["optimization_time_seconds"] = optimization_time
        
        self.logger.info(f"Post-download optimization completed for {model_name} in {optimization_time:.2f}s")
        self.logger.info(f"Size reduction: {optimization_report['model_size_metrics']['size_reduction_percent']:.1f}%")
        
        return optimized_model, optimization_report
    
    def _auto_optimize(
        self,
        model: nn.Module,
        model_name: str,
        example_inputs: torch.Tensor,
        optimization_report: Dict[str, Any]
    ) -> nn.Module:
        """
        Automatically select and apply the best optimization method.
        
        Args:
            model: Model to optimize
            model_name: Name of the model
            example_inputs: Example inputs for testing
            optimization_report: Report to update
            
        Returns:
            Optimized model
        """
        self.logger.info(f"Auto-selecting best optimization method for {model_name}")
        
        # Try different optimization strategies and pick the best one
        optimization_candidates = []
        
        # Check if target device is GPU - skip quantization if so
        target_device = next(model.parameters()).device
        
        # Test quantization
        if self.config.enable_quantization and target_device.type != 'cuda':
            try:
                quantized_model, _ = self.quantization_optimizer.quantize_model(copy.deepcopy(model), method=self.config.quantization_method, example_inputs=example_inputs)
                candidate_metrics = self._evaluate_optimization(model, quantized_model, example_inputs)
                candidate_metrics["method"] = "quantization"
                candidate_metrics["model"] = quantized_model
                optimization_candidates.append(candidate_metrics)
            except Exception as e:
                self.logger.warning(f"Quantization evaluation failed: {e}")
        elif self.config.enable_quantization and target_device.type == 'cuda':
            self.logger.info(f"Skipping quantization for {model_name} - target device is CUDA")
        
        # Test tensor factorization
        if self.config.enable_tensor_factorization and self.compression_suite:
            try:
                factorized_model = self.compression_suite.compress_model(copy.deepcopy(model))
                candidate_metrics = self._evaluate_optimization(model, factorized_model, example_inputs)
                candidate_metrics["method"] = "tensor_factorization"
                candidate_metrics["model"] = factorized_model
                optimization_candidates.append(candidate_metrics)
            except Exception as e:
                self.logger.warning(f"Tensor factorization evaluation failed: {e}")
        
        # Test comprehensive compression
        if self.compression_suite:
            try:
                compressed_model = self.compression_suite.compress_model(copy.deepcopy(model))
                candidate_metrics = self._evaluate_optimization(model, compressed_model, example_inputs)
                candidate_metrics["method"] = "comprehensive_compression"
                candidate_metrics["model"] = compressed_model
                optimization_candidates.append(candidate_metrics)
            except Exception as e:
                self.logger.warning(f"Comprehensive compression evaluation failed: {e}")
        
        if not optimization_candidates:
            self.logger.warning(f"No optimization candidates available for {model_name}")
            return model
        
        # Select best candidate based on composite score
        best_candidate = self._select_best_optimization(optimization_candidates)
        
        optimization_report["optimizations_applied"].append(best_candidate["method"])
        optimization_report["auto_selection_metrics"] = {
            "candidates_evaluated": len(optimization_candidates),
            "selected_method": best_candidate["method"],
            "selection_criteria": best_candidate
        }
        
        self.logger.info(f"Selected {best_candidate['method']} as best optimization for {model_name}")
        
        return best_candidate["model"]
    
    def _apply_quantization(
        self,
        model: nn.Module,
        model_name: str,
        example_inputs: torch.Tensor,
        optimization_report: Dict[str, Any]
    ) -> nn.Module:
        """Apply quantization to the model."""
        self.logger.info(f"Applying quantization to {model_name} using method: {self.config.quantization_method}")
        
        try:
            # Use the quantize_model method which returns (model, report)
            quantized_model, quant_report = self.quantization_optimizer.quantize_model(
                model, method=self.config.quantization_method, example_inputs=example_inputs
            )
            
            if quant_report.get("success", False):
                optimization_report["optimizations_applied"].append(f"quantization_{self.config.quantization_method}")
                if "performance_metrics" in quant_report:
                    optimization_report["performance_metrics"].update(quant_report["performance_metrics"])
                self.logger.info(f"Quantization applied successfully to {model_name}")
                return quantized_model
            else:
                error_msg = quant_report.get("error", "Unknown error")
                self.logger.error(f"Quantization failed for {model_name}: {error_msg}")
                optimization_report["errors"].append(f"Quantization failed: {error_msg}")
                return model
            
        except Exception as e:
            self.logger.error(f"Quantization failed for {model_name}: {e}")
            optimization_report["errors"].append(f"Quantization failed: {str(e)}")
            return model
    
    def _apply_tensor_optimization(
        self,
        model: nn.Module,
        model_name: str,
        optimization_report: Dict[str, Any]
    ) -> nn.Module:
        """Apply tensor factorization/low-rank optimization to the model."""
        self.logger.info(f"Applying tensor optimization to {model_name} using method: {self.config.low_rank_method}")
        
        try:
            if self.compression_suite is None:
                self.logger.warning("Compression suite not available, skipping tensor optimization")
                return model
            
            # Use the compression suite's compress_model method
            compressed_model = self.compression_suite.compress_model(model)
            
            optimization_report["optimizations_applied"].append(f"tensor_factorization_{self.config.low_rank_method}")
            self.logger.info(f"Tensor optimization applied successfully to {model_name}")
            
            return compressed_model
            
        except Exception as e:
            self.logger.error(f"Tensor optimization failed for {model_name}: {e}")
            optimization_report["errors"].append(f"Tensor optimization failed: {str(e)}")
            return model
    
    def _apply_structured_pruning(
        self,
        model: nn.Module,
        model_name: str,
        optimization_report: Dict[str, Any]
    ) -> nn.Module:
        """Apply structured pruning to the model."""
        self.logger.info(f"Applying structured pruning to {model_name}")
        
        try:
            if self.compression_suite is None:
                self.logger.warning("Compression suite not available, skipping structured pruning")
                return model
            
            # Use the compression suite's compress_model method for structured pruning
            compressed_model = self.compression_suite.compress_model(model)
            
            optimization_report["optimizations_applied"].append("structured_pruning")
            self.logger.info(f"Structured pruning applied successfully to {model_name}")
            
            return compressed_model
            
        except Exception as e:
            self.logger.error(f"Structured pruning failed for {model_name}: {e}")
            optimization_report["errors"].append(f"Structured pruning failed: {str(e)}")
            return model
    
    def _apply_comprehensive_compression(
        self,
        model: nn.Module,
        model_name: str,
        optimization_report: Dict[str, Any]
    ) -> nn.Module:
        """Apply comprehensive compression using multiple methods."""
        self.logger.info(f"Applying comprehensive compression to {model_name}")
        
        try:
            if self.compression_suite is None:
                self.logger.warning("Compression suite not available, skipping comprehensive compression")
                return model
            
            # Use the compression suite's compress_model method
            compressed_model = self.compression_suite.compress_model(model)
            
            optimization_report["optimizations_applied"].append("comprehensive_compression")
            self.logger.info(f"Comprehensive compression applied successfully to {model_name}")
            
            return compressed_model
            
        except Exception as e:
            self.logger.error(f"Comprehensive compression failed for {model_name}: {e}")
            optimization_report["errors"].append(f"Comprehensive compression failed: {str(e)}")
            return model
    
    def _evaluate_optimization(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        example_inputs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate optimization quality.
        
        Args:
            original_model: Original model
            optimized_model: Optimized model
            example_inputs: Test inputs
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Size metrics
        original_size = self._get_model_size(original_model)
        optimized_size = self._get_model_size(optimized_model)
        metrics["size_reduction_ratio"] = (original_size - optimized_size) / original_size
        metrics["size_reduction_mb"] = (original_size - optimized_size) / (1024 * 1024)
        
        # Performance metrics (simplified)
        try:
            original_model.eval()
            optimized_model.eval()
            
            with torch.no_grad():
                # Time original model
                start_time = time.time()
                for _ in range(10):
                    original_output = original_model(example_inputs)
                original_time = time.time() - start_time
                
                # Time optimized model
                start_time = time.time()
                for _ in range(10):
                    optimized_output = optimized_model(example_inputs)
                optimized_time = time.time() - start_time
                
                metrics["speedup"] = original_time / optimized_time if optimized_time > 0 else 1.0
                
                # Accuracy preservation (MSE)
                mse = torch.mean((original_output - optimized_output) ** 2).item()
                metrics["mse_accuracy_loss"] = mse
                
        except Exception as e:
            self.logger.warning(f"Performance evaluation failed: {e}")
            metrics["speedup"] = 1.0
            metrics["mse_accuracy_loss"] = 0.0
        
        # Composite score (higher is better)
        metrics["composite_score"] = (
            metrics["size_reduction_ratio"] * 0.4 +  # 40% weight on size reduction
            (metrics["speedup"] - 1.0) * 0.4 +       # 40% weight on speedup
            (1.0 - min(metrics["mse_accuracy_loss"], 1.0)) * 0.2  # 20% weight on accuracy preservation
        )
        
        return metrics
    
    def _select_best_optimization(self, candidates: list) -> Dict[str, Any]:
        """Select the best optimization from candidates based on composite score."""
        if not candidates:
            raise ValueError("No optimization candidates available")
        
        # Sort by composite score (descending)
        candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        
        return candidates[0]
    
    def _benchmark_models(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        example_inputs: torch.Tensor,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark original vs optimized model performance."""
        self.logger.info("Benchmarking model performance...")
        
        original_model.eval()
        optimized_model.eval()
        
        with torch.no_grad():
            # Benchmark original model
            start_time = time.time()
            for _ in range(iterations):
                _ = original_model(example_inputs)
            original_time = time.time() - start_time
            
            # Benchmark optimized model
            start_time = time.time()
            for _ in range(iterations):
                _ = optimized_model(example_inputs)
            optimized_time = time.time() - start_time
        
        # Calculate metrics
        original_fps = iterations / original_time
        optimized_fps = iterations / optimized_time
        speedup = original_time / optimized_time
        
        return {
            "benchmark_iterations": iterations,
            "original_time_s": original_time,
            "optimized_time_s": optimized_time,
            "original_fps": original_fps,
            "optimized_fps": optimized_fps,
            "speedup": speedup,
            "performance_improvement_percent": (speedup - 1) * 100
        }
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def _generate_example_inputs(self, model: nn.Module) -> torch.Tensor:
        """Generate example inputs for the model."""
        # This is a simple heuristic - in practice, you might want more sophisticated input generation
        try:
            # Try to infer input shape from the first layer
            first_layer = next(iter(model.children()))
            if hasattr(first_layer, 'in_features'):
                # Linear layer
                return torch.randn(1, first_layer.in_features)
            elif hasattr(first_layer, 'in_channels'):
                # Convolutional layer
                return torch.randn(1, first_layer.in_channels, 224, 224)  # Assume 224x224 images
            else:
                # Default to common shapes
                return torch.randn(1, 3, 224, 224)  # RGB image
        except Exception:
            # Fallback to common input shape
            return torch.randn(1, 3, 224, 224)
    
    def _save_optimized_model(
        self,
        model: nn.Module,
        save_path: Path,
        optimization_report: Dict[str, Any]
    ) -> None:
        """Save optimized model and optimization report."""
        save_path = Path(save_path)
        
        # Save model
        model_path = save_path / "optimized_model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save full model as well
        full_model_path = save_path / "optimized_model_full.pt"
        torch.save(model, full_model_path)
        
        # Save optimization report
        import json
        report_path = save_path / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(optimization_report, f, indent=2)
        
        self.logger.info(f"Saved optimized model to {model_path}")
        self.logger.info(f"Saved optimization report to {report_path}")


def create_post_download_optimizer(
    config: Optional[PostDownloadOptimizationConfig] = None,
    inference_config: Optional[InferenceConfig] = None
) -> PostDownloadOptimizer:
    """
    Factory function to create a PostDownloadOptimizer.
    
    Args:
        config: Post-download optimization configuration
        inference_config: Overall inference configuration
        
    Returns:
        PostDownloadOptimizer instance
    """
    if config is None:
        config = PostDownloadOptimizationConfig()
    
    return PostDownloadOptimizer(config, inference_config)


def optimize_downloaded_model(
    model: nn.Module,
    model_name: str,
    config: Optional[PostDownloadOptimizationConfig] = None,
    inference_config: Optional[InferenceConfig] = None,
    example_inputs: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to optimize a downloaded model.
    
    Args:
        model: The PyTorch model to optimize
        model_name: Name of the model for logging
        config: Post-download optimization configuration
        inference_config: Overall inference configuration
        example_inputs: Example inputs for optimization
        save_path: Path to save the optimized model
        
    Returns:
        Tuple of (optimized_model, optimization_report)
    """
    optimizer = create_post_download_optimizer(config, inference_config)
    return optimizer.optimize_model(model, model_name, example_inputs, save_path)
