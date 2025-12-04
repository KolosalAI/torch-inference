"""
Autoscaling module for PyTorch inference framework.

This module provides advanced autoscaling capabilities including:
- Zero scaling (scale to zero when no requests)
- Dynamic model loading/unloading
- Request-based scaling
- Performance-based scaling
- Resource-based scaling
"""

from .zero_scaler import ZeroScaler, ZeroScalingConfig
from .model_loader import (
    DynamicModelLoader, 
    ModelLoaderConfig, 
    ModelLoadingStrategy,
    LoadBalancer,
    LoadBalancingStrategy
)
from .autoscaler import Autoscaler, AutoscalerConfig
from .metrics import ScalingMetrics, MetricsCollector

__all__ = [
    "ZeroScaler",
    "ZeroScalingConfig", 
    "DynamicModelLoader",
    "ModelLoaderConfig",
    "ModelLoadingStrategy",
    "LoadBalancer",
    "LoadBalancingStrategy",
    "Autoscaler",
    "AutoscalerConfig",
    "ScalingMetrics",
    "MetricsCollector"
]
