"""
INT8 calibration toolkit for quantization optimization.

This module provides comprehensive INT8 quantization calibration including
entropy-based calibration, percentile calibration, and KL-divergence optimization.
"""

import logging
import math
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import threading
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader, Dataset

from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    """Configuration for INT8 calibration."""
    method: str = "entropy"  # entropy, percentile, kl_divergence, minmax
    num_calibration_batches: int = 100
    percentile: float = 99.99
    histogram_bins: int = 2048
    collect_stats: bool = True
    cache_calibration: bool = True
    smooth_distribution: bool = True
    outlier_threshold: float = 0.001


@dataclass
class ActivationStats:
    """Statistics for activation tensors during calibration."""
    min_val: float
    max_val: float
    mean: float
    std: float
    histogram: np.ndarray
    bin_edges: np.ndarray
    shape: Tuple[int, ...]
    sample_count: int


class ActivationObserver:
    """Observer for collecting activation statistics during calibration."""
    
    def __init__(self, config: CalibrationConfig):
        """
        Initialize activation observer.
        
        Args:
            config: Calibration configuration
        """
        self.config = config
        self.stats: Dict[str, ActivationStats] = {}
        self.hooks: List[Any] = []
        self.layer_names: Dict[nn.Module, str] = {}
        self.activation_cache: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{__name__}.ActivationObserver")
    
    def register_hooks(self, model: nn.Module) -> None:
        """
        Register forward hooks to collect activation statistics.
        
        Args:
            model: PyTorch model to instrument
        """
        def create_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._collect_activation_stats(name, output)
                elif isinstance(output, (list, tuple)):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            self._collect_activation_stats(f"{name}_output_{i}", out)
            return hook
        
        # Register hooks for key layer types
        target_layers = (
            nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d,
            nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d,
            nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU,
            nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d
        )
        
        layer_counter = defaultdict(int)
        
        for name, module in model.named_modules():
            if isinstance(module, target_layers):
                layer_type = type(module).__name__
                layer_counter[layer_type] += 1
                
                if not name:
                    layer_name = f"{layer_type}_{layer_counter[layer_type]}"
                else:
                    layer_name = name
                
                self.layer_names[module] = layer_name
                hook = module.register_forward_hook(create_hook(layer_name))
                self.hooks.append(hook)
        
        self.logger.info(f"Registered hooks for {len(self.hooks)} layers")
    
    def _collect_activation_stats(self, layer_name: str, activation: torch.Tensor) -> None:
        """
        Collect statistics for a single activation tensor.
        
        Args:
            layer_name: Name of the layer
            activation: Activation tensor
        """
        with self.lock:
            # Convert to CPU and numpy for processing
            act_data = activation.detach().cpu().numpy().flatten()
            
            # Skip if tensor is empty or contains only zeros
            if act_data.size == 0 or np.all(act_data == 0):
                return
            
            # Remove outliers based on threshold
            if self.config.outlier_threshold > 0:
                q_low = np.percentile(act_data, self.config.outlier_threshold * 100 / 2)
                q_high = np.percentile(act_data, 100 - self.config.outlier_threshold * 100 / 2)
                act_data = act_data[(act_data >= q_low) & (act_data <= q_high)]
            
            if layer_name not in self.stats:
                # Initialize statistics
                hist, bin_edges = np.histogram(act_data, bins=self.config.histogram_bins)
                
                self.stats[layer_name] = ActivationStats(
                    min_val=float(np.min(act_data)),
                    max_val=float(np.max(act_data)),
                    mean=float(np.mean(act_data)),
                    std=float(np.std(act_data)),
                    histogram=hist.astype(np.float64),
                    bin_edges=bin_edges.astype(np.float64),
                    shape=activation.shape,
                    sample_count=1
                )
            else:
                # Update existing statistics
                stats = self.stats[layer_name]
                
                # Update min/max
                stats.min_val = min(stats.min_val, float(np.min(act_data)))
                stats.max_val = max(stats.max_val, float(np.max(act_data)))
                
                # Update mean and std (running average)
                n = stats.sample_count
                new_mean = float(np.mean(act_data))
                stats.mean = (stats.mean * n + new_mean) / (n + 1)
                
                new_std = float(np.std(act_data))
                stats.std = (stats.std * n + new_std) / (n + 1)
                
                # Update histogram
                new_hist, _ = np.histogram(act_data, bins=stats.bin_edges)
                stats.histogram += new_hist.astype(np.float64)
                
                stats.sample_count += 1
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.logger.info("Removed all hooks")
    
    def get_stats(self) -> Dict[str, ActivationStats]:
        """Get collected activation statistics."""
        return self.stats.copy()
    
    def clear_stats(self) -> None:
        """Clear all collected statistics."""
        with self.lock:
            self.stats.clear()
            self.activation_cache.clear()


class INT8CalibrationToolkit:
    """
    Comprehensive INT8 calibration toolkit for quantization optimization.
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize INT8 calibration toolkit.
        
        Args:
            config: Calibration configuration
        """
        self.config = config or CalibrationConfig()
        self.observer = ActivationObserver(self.config)
        self.calibration_cache: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(f"{__name__}.INT8CalibrationToolkit")
        self.logger.info(f"INT8 calibration toolkit initialized with method: {self.config.method}")
    
    def calibrate_model(self, 
                       model: nn.Module,
                       calibration_loader: DataLoader,
                       device: Optional[torch.device] = None) -> Dict[str, Tuple[float, float]]:
        """
        Calibrate model for INT8 quantization.
        
        Args:
            model: PyTorch model to calibrate
            calibration_loader: DataLoader with calibration data
            device: Target device for calibration
            
        Returns:
            Dictionary mapping layer names to (scale, zero_point) tuples
        """
        if device is None:
            device = next(model.parameters()).device
        
        model = model.to(device)
        model.eval()
        
        self.logger.info(f"Starting INT8 calibration with {self.config.method} method")
        
        # Register hooks to collect activation statistics
        self.observer.register_hooks(model)
        
        try:
            # Run calibration data through model
            self._run_calibration(model, calibration_loader, device)
            
            # Calculate quantization parameters
            quantization_params = self._calculate_quantization_parameters()
            
            # Cache results if enabled
            if self.config.cache_calibration:
                self._cache_calibration_results(quantization_params)
            
            self.logger.info(f"Calibration completed for {len(quantization_params)} layers")
            return quantization_params
            
        finally:
            self.observer.remove_hooks()
    
    def _run_calibration(self, 
                        model: nn.Module, 
                        calibration_loader: DataLoader,
                        device: torch.device) -> None:
        """Run calibration data through the model."""
        batch_count = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(calibration_loader):
                # Handle different batch formats
                if isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0]
                else:
                    inputs = batch_data
                
                inputs = inputs.to(device)
                
                # Forward pass to collect activations
                _ = model(inputs)
                
                batch_count += 1
                if batch_count >= self.config.num_calibration_batches:
                    break
                
                if batch_idx % 10 == 0:
                    self.logger.debug(f"Processed calibration batch {batch_idx}")
        
        calibration_time = time.time() - start_time
        self.logger.info(f"Calibration completed in {calibration_time:.2f}s ({batch_count} batches)")
    
    def _calculate_quantization_parameters(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate quantization parameters based on collected statistics.
        
        Returns:
            Dictionary mapping layer names to (scale, zero_point) tuples
        """
        quantization_params = {}
        stats = self.observer.get_stats()
        
        for layer_name, activation_stats in stats.items():
            if self.config.method == "entropy":
                scale, zero_point = self._entropy_calibration(activation_stats)
            elif self.config.method == "percentile":
                scale, zero_point = self._percentile_calibration(activation_stats)
            elif self.config.method == "kl_divergence":
                scale, zero_point = self._kl_divergence_calibration(activation_stats)
            elif self.config.method == "minmax":
                scale, zero_point = self._minmax_calibration(activation_stats)
            else:
                self.logger.warning(f"Unknown calibration method: {self.config.method}, using minmax")
                scale, zero_point = self._minmax_calibration(activation_stats)
            
            quantization_params[layer_name] = (scale, zero_point)
        
        return quantization_params
    
    def _entropy_calibration(self, stats: ActivationStats) -> Tuple[float, float]:
        """
        Entropy-based calibration for optimal quantization range.
        
        Args:
            stats: Activation statistics
            
        Returns:
            (scale, zero_point) tuple
        """
        histogram = stats.histogram
        bin_edges = stats.bin_edges
        
        # Smooth histogram if configured
        if self.config.smooth_distribution:
            histogram = self._smooth_histogram(histogram)
        
        # Find optimal threshold using entropy minimization
        best_threshold = None
        min_entropy = float('inf')
        
        # Search for threshold that minimizes quantization entropy
        for threshold_idx in range(len(histogram) // 4, 3 * len(histogram) // 4):
            # Calculate entropy for this threshold
            entropy = self._calculate_quantization_entropy(histogram, threshold_idx)
            
            if entropy < min_entropy:
                min_entropy = entropy
                best_threshold = bin_edges[threshold_idx]
        
        if best_threshold is None:
            # Fallback to percentile method
            return self._percentile_calibration(stats)
        
        # Calculate scale and zero_point using absolute value to ensure positive scale
        # For symmetric quantization, we need the maximum absolute value
        abs_max = max(abs(stats.min_val), abs(stats.max_val), abs(best_threshold))
        
        # Ensure scale is always positive
        scale = abs_max / 127.0
        zero_point = 0  # Symmetric quantization
        
        # Ensure minimum scale to avoid zero or very small values
        scale = max(scale, 1e-6)
        
        return scale, zero_point
    
    def _percentile_calibration(self, stats: ActivationStats) -> Tuple[float, float]:
        """
        Percentile-based calibration.
        
        Args:
            stats: Activation statistics
            
        Returns:
            (scale, zero_point) tuple
        """
        histogram = stats.histogram
        bin_edges = stats.bin_edges
        
        # Calculate percentile threshold
        total_samples = np.sum(histogram)
        percentile_samples = total_samples * (self.config.percentile / 100.0)
        
        cumsum = np.cumsum(histogram)
        threshold_idx = np.searchsorted(cumsum, percentile_samples)
        threshold_idx = min(threshold_idx, len(bin_edges) - 1)
        
        threshold = bin_edges[threshold_idx]
        
        # Handle symmetric vs asymmetric quantization
        # Always use the maximum absolute value to ensure positive scale
        abs_max = max(abs(stats.min_val), abs(stats.max_val), abs(threshold))
        
        if abs(stats.min_val) > abs(stats.max_val):
            # Asymmetric quantization
            qmin, qmax = -128, 127
            scale = abs_max / 127.0
            zero_point = int(-stats.min_val / scale)
            zero_point = max(min(zero_point, 127), -128)
        else:
            # Symmetric quantization
            scale = abs_max / 127.0
            zero_point = 0
        
        # Ensure minimum scale to avoid zero or very small values
        scale = max(scale, 1e-6)
        
        return scale, zero_point
    
    def _kl_divergence_calibration(self, stats: ActivationStats) -> Tuple[float, float]:
        """
        KL-divergence minimization calibration.
        
        Args:
            stats: Activation statistics
            
        Returns:
            (scale, zero_point) tuple
        """
        histogram = stats.histogram
        bin_edges = stats.bin_edges
        
        # Smooth and normalize histogram
        if self.config.smooth_distribution:
            histogram = self._smooth_histogram(histogram)
        
        # Normalize to probability distribution
        P = histogram / np.sum(histogram)
        P = P + 1e-10  # Add small epsilon to avoid log(0)
        
        best_threshold = None
        min_kl_divergence = float('inf')
        
        # Search for threshold that minimizes KL divergence
        for threshold_idx in range(len(histogram) // 8, 7 * len(histogram) // 8):
            # Create quantized distribution
            Q = self._create_quantized_distribution(P, threshold_idx)
            
            # Calculate KL divergence
            kl_div = self._calculate_kl_divergence(P, Q)
            
            if kl_div < min_kl_divergence:
                min_kl_divergence = kl_div
                best_threshold = bin_edges[threshold_idx]
        
        if best_threshold is None:
            return self._percentile_calibration(stats)
        
        # Calculate scale and zero_point using absolute values to ensure positive scale
        abs_max = max(abs(stats.min_val), abs(stats.max_val), abs(best_threshold))
        scale = abs_max / 127.0
        zero_point = 0
        
        # Ensure minimum scale to avoid zero or very small values
        scale = max(scale, 1e-6)
        
        return scale, zero_point
    
    def _minmax_calibration(self, stats: ActivationStats) -> Tuple[float, float]:
        """
        Simple min-max calibration.
        
        Args:
            stats: Activation statistics
            
        Returns:
            (scale, zero_point) tuple
        """
        # Use full range - ensure scale is always positive
        abs_max = max(abs(stats.min_val), abs(stats.max_val))
        scale = abs_max / 127.0
        zero_point = 0
        
        # Ensure minimum scale to avoid zero or very small values
        scale = max(scale, 1e-6)
        
        return scale, zero_point
    
    def _smooth_histogram(self, histogram: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply smoothing to histogram."""
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(histogram, kernel, mode='same')
        return smoothed
    
    def _calculate_quantization_entropy(self, histogram: np.ndarray, threshold_idx: int) -> float:
        """Calculate entropy of quantized distribution."""
        # Simulate quantization by binning
        quantized_bins = 256  # INT8 has 256 levels
        
        if threshold_idx >= len(histogram):
            return float('inf')
        
        # Get the relevant portion of histogram
        relevant_hist = histogram[:threshold_idx]
        if np.sum(relevant_hist) == 0:
            return float('inf')
        
        # Quantize into 256 bins
        bin_size = max(1, len(relevant_hist) // quantized_bins)
        quantized_hist = []
        
        for i in range(0, len(relevant_hist), bin_size):
            bin_sum = np.sum(relevant_hist[i:i + bin_size])
            if bin_sum > 0:
                quantized_hist.append(bin_sum)
        
        if not quantized_hist:
            return float('inf')
        
        # Calculate entropy
        quantized_hist = np.array(quantized_hist)
        probs = quantized_hist / np.sum(quantized_hist)
        probs = probs[probs > 0]  # Remove zeros
        
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    
    def _create_quantized_distribution(self, P: np.ndarray, threshold_idx: int) -> np.ndarray:
        """Create quantized probability distribution."""
        Q = np.zeros_like(P)
        
        # Quantize the distribution into 256 levels
        relevant_portion = P[:threshold_idx]
        if np.sum(relevant_portion) == 0:
            return P  # Return original if no relevant data
        
        # Simple uniform quantization
        bin_size = max(1, len(relevant_portion) // 256)
        for i in range(0, len(relevant_portion), bin_size):
            bin_sum = np.sum(relevant_portion[i:i + bin_size])
            avg_prob = bin_sum / min(bin_size, len(relevant_portion) - i)
            
            for j in range(i, min(i + bin_size, len(relevant_portion))):
                Q[j] = avg_prob
        
        # Normalize
        if np.sum(Q) > 0:
            Q = Q / np.sum(Q)
        else:
            Q = P  # Fallback
        
        return Q
    
    def _calculate_kl_divergence(self, P: np.ndarray, Q: np.ndarray) -> float:
        """Calculate KL divergence between two distributions."""
        # Ensure both distributions are valid
        P = P + 1e-10
        Q = Q + 1e-10
        
        # Calculate KL divergence
        kl_div = np.sum(P * np.log(P / Q))
        return kl_div
    
    def _cache_calibration_results(self, quantization_params: Dict[str, Tuple[float, float]]) -> None:
        """Cache calibration results to disk."""
        try:
            cache_dir = Path("calibration_cache")
            cache_dir.mkdir(exist_ok=True)
            
            cache_file = cache_dir / f"calibration_{self.config.method}_{hash(str(quantization_params))}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'config': self.config,
                    'quantization_params': quantization_params,
                    'timestamp': time.time()
                }, f)
            
            self.logger.info(f"Calibration results cached to {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache calibration results: {e}")
    
    def load_cached_calibration(self, cache_key: str) -> Optional[Dict[str, Tuple[float, float]]]:
        """Load cached calibration results."""
        try:
            cache_dir = Path("calibration_cache")
            cache_file = cache_dir / f"calibration_{cache_key}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self.logger.info(f"Loaded cached calibration from {cache_file}")
                return cached_data['quantization_params']
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached calibration: {e}")
        
        return None
    
    def validate_calibration_quality(self, 
                                   original_model: nn.Module,
                                   quantized_model: nn.Module,
                                   validation_loader: DataLoader,
                                   device: Optional[torch.device] = None) -> Dict[str, float]:
        """
        Validate calibration quality by comparing model outputs.
        
        Args:
            original_model: Original FP32 model
            quantized_model: INT8 quantized model
            validation_loader: Validation data loader
            device: Target device
            
        Returns:
            Quality metrics dictionary
        """
        if device is None:
            device = next(original_model.parameters()).device
        
        original_model = original_model.to(device)
        quantized_model = quantized_model.to(device)
        
        original_model.eval()
        quantized_model.eval()
        
        metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'cosine_similarity': 0.0,
            'snr': 0.0,
            'samples_processed': 0
        }
        
        total_mse = 0.0
        total_mae = 0.0
        total_cosine_sim = 0.0
        total_snr = 0.0
        sample_count = 0
        
        self.logger.info("Validating calibration quality")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(validation_loader):
                if isinstance(batch_data, (list, tuple)):
                    inputs = batch_data[0]
                else:
                    inputs = batch_data
                
                inputs = inputs.to(device)
                
                # Get outputs from both models
                original_output = original_model(inputs)
                quantized_output = quantized_model(inputs)
                
                # Calculate metrics
                batch_mse = torch.mean((original_output - quantized_output) ** 2).item()
                batch_mae = torch.mean(torch.abs(original_output - quantized_output)).item()
                
                # Cosine similarity
                orig_flat = original_output.flatten()
                quant_flat = quantized_output.flatten()
                batch_cosine = torch.nn.functional.cosine_similarity(
                    orig_flat, quant_flat, dim=0
                ).item()
                
                # Signal-to-noise ratio
                signal_power = torch.mean(original_output ** 2).item()
                noise_power = torch.mean((original_output - quantized_output) ** 2).item()
                batch_snr = 10 * math.log10(signal_power / max(noise_power, 1e-10))
                
                # Accumulate metrics
                total_mse += batch_mse
                total_mae += batch_mae
                total_cosine_sim += batch_cosine
                total_snr += batch_snr
                sample_count += 1
                
                if batch_idx % 10 == 0:
                    self.logger.debug(f"Validated batch {batch_idx}")
        
        # Calculate average metrics
        if sample_count > 0:
            metrics['mse'] = total_mse / sample_count
            metrics['mae'] = total_mae / sample_count
            metrics['cosine_similarity'] = total_cosine_sim / sample_count
            metrics['snr'] = total_snr / sample_count
            metrics['samples_processed'] = sample_count
        
        self.logger.info(f"Calibration quality validation completed:")
        self.logger.info(f"  MSE: {metrics['mse']:.6f}")
        self.logger.info(f"  MAE: {metrics['mae']:.6f}")
        self.logger.info(f"  Cosine Similarity: {metrics['cosine_similarity']:.4f}")
        self.logger.info(f"  SNR (dB): {metrics['snr']:.2f}")
        
        return metrics
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Generate comprehensive calibration report."""
        stats = self.observer.get_stats()
        
        report = {
            'config': {
                'method': self.config.method,
                'num_calibration_batches': self.config.num_calibration_batches,
                'percentile': self.config.percentile,
                'histogram_bins': self.config.histogram_bins
            },
            'statistics': {},
            'recommendations': []
        }
        
        for layer_name, activation_stats in stats.items():
            layer_report = {
                'min_val': activation_stats.min_val,
                'max_val': activation_stats.max_val,
                'mean': activation_stats.mean,
                'std': activation_stats.std,
                'dynamic_range': activation_stats.max_val - activation_stats.min_val,
                'sample_count': activation_stats.sample_count,
                'shape': activation_stats.shape
            }
            
            # Add recommendations based on statistics
            if activation_stats.std == 0:
                layer_report['warning'] = "Layer has zero variance - may not benefit from quantization"
            elif activation_stats.max_val - activation_stats.min_val < 0.001:
                layer_report['warning'] = "Very small dynamic range - quantization may cause accuracy loss"
            elif abs(activation_stats.mean) > 10 * activation_stats.std:
                layer_report['info'] = "Distribution is highly skewed - consider asymmetric quantization"
            
            report['statistics'][layer_name] = layer_report
        
        # Global recommendations
        if len(stats) == 0:
            report['recommendations'].append("No activation statistics collected - check calibration setup")
        else:
            avg_dynamic_range = np.mean([s.max_val - s.min_val for s in stats.values()])
            if avg_dynamic_range < 0.01:
                report['recommendations'].append("Small average dynamic range - consider FP16 instead of INT8")
            
            zero_variance_layers = [name for name, s in stats.items() if s.std == 0]
            if zero_variance_layers:
                report['recommendations'].append(f"Consider excluding zero-variance layers: {zero_variance_layers}")
        
        return report


def create_calibration_dataset(data_loader: DataLoader, 
                             num_samples: int = 1000) -> torch.utils.data.Dataset:
    """
    Create a subset dataset for calibration.
    
    Args:
        data_loader: Original data loader
        num_samples: Number of samples to use for calibration
        
    Returns:
        Calibration dataset
    """
    
    class CalibrationDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    samples = []
    sample_count = 0
    
    for batch_data in data_loader:
        if isinstance(batch_data, (list, tuple)):
            inputs = batch_data[0]
        else:
            inputs = batch_data
        
        # Add individual samples
        for i in range(inputs.size(0)):
            if sample_count >= num_samples:
                break
            samples.append(inputs[i])
            sample_count += 1
        
        if sample_count >= num_samples:
            break
    
    return CalibrationDataset(samples)


# Global calibration toolkit instance
_global_calibration_toolkit: Optional[INT8CalibrationToolkit] = None


def get_calibration_toolkit() -> INT8CalibrationToolkit:
    """Get global INT8 calibration toolkit instance."""
    global _global_calibration_toolkit
    if _global_calibration_toolkit is None:
        _global_calibration_toolkit = INT8CalibrationToolkit()
    return _global_calibration_toolkit
