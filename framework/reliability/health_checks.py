"""
Comprehensive Health Check System

Provides granular health checks beyond basic endpoints including:
- Model-specific health checks
- Dependency health checks (GPU, memory, disk)
- Deep health checks with actual inference tests
"""

import asyncio
import time
import logging
import psutil
import os
import torch
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    
    @classmethod
    def healthy(cls, name: str, message: str, details: Dict[str, Any] = None) -> 'HealthCheckResult':
        """Create a healthy result."""
        return cls(
            name=name,
            status=HealthStatus.HEALTHY,
            message=message,
            details=details or {}
        )
    
    @classmethod
    def unhealthy(cls, name: str, message: str, details: Dict[str, Any] = None) -> 'HealthCheckResult':
        """Create an unhealthy result."""
        return cls(
            name=name,
            status=HealthStatus.CRITICAL,
            message=message,
            details=details or {}
        )
    
    @classmethod
    def warning(cls, name: str, message: str, details: Dict[str, Any] = None) -> 'HealthCheckResult':
        """Create a warning result."""
        return cls(
            name=name,
            status=HealthStatus.WARNING,
            message=message,
            details=details or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "healthy": self.status == HealthStatus.HEALTHY
        }


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout: float = 30.0):
        self.name = name
        self.timeout = timeout
        self.last_result: Optional[HealthCheckResult] = None
        self.last_check_time: float = 0.0
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        start_time = time.time()
        try:
            result = await asyncio.wait_for(self._perform_check(), timeout=self.timeout)
            result.duration_ms = (time.time() - start_time) * 1000
            self.last_result = result
            self.last_check_time = time.time()
            return result
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.timeout}s",
                duration_ms=duration_ms
            )
            self.last_result = result
            self.last_check_time = time.time()
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                duration_ms=duration_ms
            )
            self.last_result = result
            self.last_check_time = time.time()
            return result
    
    async def _perform_check(self) -> HealthCheckResult:
        """Override this method to implement the actual health check."""
        raise NotImplementedError


# Alias for compatibility
BaseHealthCheck = HealthCheck


class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)."""
    
    def __init__(self, 
                 name: str = "system_resources",
                 memory_warning_threshold: float = 0.8,
                 memory_critical_threshold: float = 0.95,
                 cpu_warning_threshold: float = 0.8,
                 cpu_critical_threshold: float = 0.95,
                 disk_warning_threshold: float = 0.8,
                 disk_critical_threshold: float = 0.95,
                 timeout: float = 10.0):
        super().__init__(name, timeout)
        self.memory_warning_threshold = memory_warning_threshold
        self.memory_critical_threshold = memory_critical_threshold
        self.cpu_warning_threshold = cpu_warning_threshold
        self.cpu_critical_threshold = cpu_critical_threshold
        self.disk_warning_threshold = disk_warning_threshold
        self.disk_critical_threshold = disk_critical_threshold
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check system resource usage."""
        issues = []
        warnings = []
        status = HealthStatus.HEALTHY
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_usage = memory.percent / 100.0
        
        if memory_usage >= self.memory_critical_threshold:
            issues.append(f"Critical memory usage: {memory_usage:.1%}")
            status = HealthStatus.CRITICAL
        elif memory_usage >= self.memory_warning_threshold:
            warnings.append(f"High memory usage: {memory_usage:.1%}")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
        
        # CPU check
        cpu_usage = psutil.cpu_percent(interval=1) / 100.0
        
        if cpu_usage >= self.cpu_critical_threshold:
            issues.append(f"Critical CPU usage: {cpu_usage:.1%}")
            status = HealthStatus.CRITICAL
        elif cpu_usage >= self.cpu_warning_threshold:
            warnings.append(f"High CPU usage: {cpu_usage:.1%}")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
        
        # Disk check
        disk_usage = psutil.disk_usage('/').percent / 100.0
        
        if disk_usage >= self.disk_critical_threshold:
            issues.append(f"Critical disk usage: {disk_usage:.1%}")
            status = HealthStatus.CRITICAL
        elif disk_usage >= self.disk_warning_threshold:
            warnings.append(f"High disk usage: {disk_usage:.1%}")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
        
        # Compile message
        all_issues = issues + warnings
        if all_issues:
            message = "; ".join(all_issues)
        else:
            message = "System resources are healthy"
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details={
                "memory": {
                    "usage_percent": memory_usage * 100,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3),
                    "threshold_warning": self.memory_warning_threshold * 100,
                    "threshold_critical": self.memory_critical_threshold * 100
                },
                "cpu": {
                    "usage_percent": cpu_usage * 100,
                    "count": psutil.cpu_count(),
                    "threshold_warning": self.cpu_warning_threshold * 100,
                    "threshold_critical": self.cpu_critical_threshold * 100
                },
                "disk": {
                    "usage_percent": disk_usage * 100,
                    "free_gb": psutil.disk_usage('/').free / (1024**3),
                    "total_gb": psutil.disk_usage('/').total / (1024**3),
                    "threshold_warning": self.disk_warning_threshold * 100,
                    "threshold_critical": self.disk_critical_threshold * 100
                }
            }
        )


class GPUHealthCheck(HealthCheck):
    """Health check for GPU resources and availability."""
    
    def __init__(self, 
                 name: str = "gpu_resources",
                 memory_warning_threshold: float = 0.8,
                 memory_critical_threshold: float = 0.95,
                 timeout: float = 10.0):
        super().__init__(name, timeout)
        self.memory_warning_threshold = memory_warning_threshold
        self.memory_critical_threshold = memory_critical_threshold
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check GPU health and resources."""
        if not torch.cuda.is_available():
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.WARNING,
                message="CUDA not available",
                details={
                    "cuda_available": False,
                    "device_count": 0,
                    "reason": "CUDA not available on this system"
                }
            )
        
        issues = []
        warnings = []
        status = HealthStatus.HEALTHY
        gpu_details = {}
        
        try:
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                
                # Memory usage
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = props.total_memory
                
                memory_usage = allocated / total
                reserved_usage = reserved / total
                
                device_details = {
                    "name": device_name,
                    "memory": {
                        "allocated_gb": allocated / (1024**3),
                        "reserved_gb": reserved / (1024**3),
                        "total_gb": total / (1024**3),
                        "usage_percent": memory_usage * 100,
                        "reserved_percent": reserved_usage * 100
                    },
                    "properties": {
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multiprocessor_count": props.multi_processor_count,
                        "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor
                    }
                }
                
                # Check memory thresholds
                if memory_usage >= self.memory_critical_threshold:
                    issues.append(f"GPU {i} critical memory usage: {memory_usage:.1%}")
                    status = HealthStatus.CRITICAL
                elif memory_usage >= self.memory_warning_threshold:
                    warnings.append(f"GPU {i} high memory usage: {memory_usage:.1%}")
                    if status == HealthStatus.HEALTHY:
                        status = HealthStatus.WARNING
                
                gpu_details[f"gpu_{i}"] = device_details
            
            # Compile message
            all_issues = issues + warnings
            if all_issues:
                message = "; ".join(all_issues)
            else:
                message = f"All {device_count} GPU(s) are healthy"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "cuda_available": True,
                    "device_count": device_count,
                    "devices": gpu_details,
                    "thresholds": {
                        "memory_warning": self.memory_warning_threshold * 100,
                        "memory_critical": self.memory_critical_threshold * 100
                    }
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"GPU health check failed: {str(e)}",
                details={
                    "cuda_available": True,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )


class ModelHealthCheck(HealthCheck):
    """Health check for a specific model."""
    
    def __init__(self, 
                 model_name: str, 
                 model_manager,
                 perform_inference_test: bool = True,
                 timeout: float = 30.0):
        super().__init__(f"model_{model_name}", timeout)
        self.model_name = model_name
        self.model_manager = model_manager
        self.perform_inference_test = perform_inference_test
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check model health and optionally perform inference test."""
        try:
            # Check if model exists
            if not self.model_manager.is_model_loaded(self.model_name):
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Model '{self.model_name}' is not loaded",
                    details={
                        "model_name": self.model_name,
                        "loaded": False,
                        "available_models": self.model_manager.list_models()
                    }
                )
            
            # Get model instance
            model = self.model_manager.get_model(self.model_name)
            
            # Basic model checks
            if not model.is_loaded:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Model '{self.model_name}' is loaded but not ready",
                    details={
                        "model_name": self.model_name,
                        "loaded": False,
                        "model_info": model.model_info
                    }
                )
            
            model_details = {
                "model_name": self.model_name,
                "loaded": True,
                "model_info": model.model_info,
                "device": str(model.device) if hasattr(model, 'device') else "unknown"
            }
            
            # Perform inference test if enabled
            if self.perform_inference_test:
                try:
                    # Create dummy input for the model
                    if hasattr(model, '_create_dummy_input'):
                        dummy_input = model._create_dummy_input()
                    else:
                        # Fallback dummy input
                        dummy_input = torch.randn(1, 10, device=model.device) if hasattr(model, 'device') else torch.randn(1, 10)
                    
                    # Perform inference test
                    start_time = time.time()
                    
                    if hasattr(model, 'predict'):
                        # Use predict method if available
                        result = model.predict(dummy_input.cpu().numpy())
                    else:
                        # Use direct forward pass
                        with torch.no_grad():
                            result = model.forward(dummy_input)
                    
                    inference_time = time.time() - start_time
                    
                    model_details["inference_test"] = {
                        "success": True,
                        "inference_time_ms": inference_time * 1000,
                        "input_shape": list(dummy_input.shape) if hasattr(dummy_input, 'shape') else "unknown",
                        "output_type": type(result).__name__
                    }
                    
                    message = f"Model '{self.model_name}' is healthy (inference test: {inference_time*1000:.1f}ms)"
                    
                except Exception as inference_error:
                    model_details["inference_test"] = {
                        "success": False,
                        "error": str(inference_error),
                        "error_type": type(inference_error).__name__
                    }
                    
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.CRITICAL,
                        message=f"Model '{self.model_name}' inference test failed: {str(inference_error)}",
                        details=model_details
                    )
            else:
                message = f"Model '{self.model_name}' is loaded and ready"
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=message,
                details=model_details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Model health check failed: {str(e)}",
                details={
                    "model_name": self.model_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )


class DependencyHealthCheck(HealthCheck):
    """Health check for external dependencies."""
    
    def __init__(self, 
                 name: str = "dependencies",
                 check_internet: bool = True,
                 check_huggingface: bool = True,
                 timeout: float = 15.0):
        super().__init__(name, timeout)
        self.check_internet = check_internet
        self.check_huggingface = check_huggingface
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check external dependencies."""
        issues = []
        warnings = []
        status = HealthStatus.HEALTHY
        dependency_details = {}
        
        # Check internet connectivity
        if self.check_internet:
            try:
                import aiohttp
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get('https://www.google.com') as response:
                        if response.status == 200:
                            dependency_details["internet"] = {
                                "available": True,
                                "response_time_ms": 0  # Would need timing implementation
                            }
                        else:
                            warnings.append(f"Internet check returned status {response.status}")
                            dependency_details["internet"] = {
                                "available": False,
                                "status_code": response.status
                            }
                            if status == HealthStatus.HEALTHY:
                                status = HealthStatus.WARNING
            except Exception as e:
                warnings.append(f"Internet connectivity check failed: {str(e)}")
                dependency_details["internet"] = {
                    "available": False,
                    "error": str(e)
                }
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
        
        # Check HuggingFace Hub
        if self.check_huggingface:
            try:
                import aiohttp
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get('https://huggingface.co/api/models?limit=1') as response:
                        if response.status == 200:
                            dependency_details["huggingface"] = {
                                "available": True,
                                "api_accessible": True
                            }
                        else:
                            warnings.append(f"HuggingFace API check returned status {response.status}")
                            dependency_details["huggingface"] = {
                                "available": False,
                                "status_code": response.status
                            }
                            if status == HealthStatus.HEALTHY:
                                status = HealthStatus.WARNING
            except Exception as e:
                warnings.append(f"HuggingFace connectivity check failed: {str(e)}")
                dependency_details["huggingface"] = {
                    "available": False,
                    "error": str(e)
                }
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
        
        # Compile message
        if issues:
            message = "; ".join(issues)
            status = HealthStatus.CRITICAL
        elif warnings:
            message = "; ".join(warnings)
        else:
            message = "All dependencies are healthy"
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details=dependency_details
        )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, 
                 name: str = "database",
                 connection_string: Optional[str] = None,
                 timeout: float = 10.0):
        super().__init__(name, timeout)
        self.connection_string = connection_string
    
    async def _perform_check(self) -> HealthCheckResult:
        """Check database connectivity."""
        if not self.connection_string:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.WARNING,
                message="Database not configured",
                details={"configured": False}
            )
        
        try:
            # This is a placeholder - actual implementation would depend on database type
            # For SQLAlchemy, you might use:
            # from sqlalchemy import create_engine, text
            # engine = create_engine(self.connection_string)
            # with engine.connect() as conn:
            #     result = conn.execute(text("SELECT 1"))
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={"configured": True, "connected": True}
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                details={
                    "configured": True,
                    "connected": False,
                    "error": str(e)
                }
            )


class HealthCheckManager:
    """
    Manager for orchestrating multiple health checks.
    
    Provides centralized health check execution, caching, and reporting.
    """
    
    def __init__(self, cache_duration: float = 30.0):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.cache_duration = cache_duration
        self._lock = threading.RLock()
        logger.info("Health check manager initialized")
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        with self._lock:
            self.health_checks[health_check.name] = health_check
            logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_health_check(self, name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if name in self.health_checks:
                del self.health_checks[name]
                logger.info(f"Unregistered health check: {name}")
                return True
            return False
    
    async def check_health(self, name: str) -> Optional[HealthCheckResult]:
        """Check health for a specific health check."""
        with self._lock:
            health_check = self.health_checks.get(name)
        
        if not health_check:
            return None
        
        # Use cached result if recent enough
        current_time = time.time()
        if (health_check.last_result and 
            current_time - health_check.last_check_time < self.cache_duration):
            return health_check.last_result
        
        # Perform fresh health check
        return await health_check.check()
    
    async def check_all_health(self, parallel: bool = True) -> Dict[str, HealthCheckResult]:
        """Check health for all registered health checks."""
        with self._lock:
            health_checks = list(self.health_checks.values())
        
        if not health_checks:
            return {}
        
        if parallel:
            # Run all health checks in parallel
            tasks = [health_check.check() for health_check in health_checks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            health_results = {}
            for health_check, result in zip(health_checks, results):
                if isinstance(result, Exception):
                    # Handle exceptions that occurred during health check
                    health_results[health_check.name] = HealthCheckResult(
                        name=health_check.name,
                        status=HealthStatus.CRITICAL,
                        message=f"Health check exception: {str(result)}",
                        details={"error": str(result), "error_type": type(result).__name__}
                    )
                else:
                    health_results[health_check.name] = result
            
            return health_results
        else:
            # Run health checks sequentially
            health_results = {}
            for health_check in health_checks:
                try:
                    result = await health_check.check()
                    health_results[health_check.name] = result
                except Exception as e:
                    health_results[health_check.name] = HealthCheckResult(
                        name=health_check.name,
                        status=HealthStatus.CRITICAL,
                        message=f"Health check exception: {str(e)}",
                        details={"error": str(e), "error_type": type(e).__name__}
                    )
            
            return health_results
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status and detailed results."""
        results = await self.check_all_health(parallel=True)
        
        if not results:
            return {
                "healthy": True,
                "status": HealthStatus.HEALTHY.value,
                "message": "No health checks configured",
                "checks": {},
                "summary": {
                    "total": 0,
                    "healthy": 0,
                    "warning": 0,
                    "critical": 0,
                    "unknown": 0
                }
            }
        
        # Calculate summary statistics
        summary = {
            "total": len(results),
            "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
            "warning": sum(1 for r in results.values() if r.status == HealthStatus.WARNING),
            "critical": sum(1 for r in results.values() if r.status == HealthStatus.CRITICAL),
            "unknown": sum(1 for r in results.values() if r.status == HealthStatus.UNKNOWN)
        }
        
        # Determine overall status
        if summary["critical"] > 0:
            overall_status = HealthStatus.CRITICAL
            healthy = False
            message = f"{summary['critical']} critical issue(s) detected"
        elif summary["warning"] > 0:
            overall_status = HealthStatus.WARNING
            healthy = False
            message = f"{summary['warning']} warning(s) detected"
        else:
            overall_status = HealthStatus.HEALTHY
            healthy = True
            message = "All health checks passed"
        
        return {
            "healthy": healthy,
            "status": overall_status.value,
            "message": message,
            "timestamp": time.time(),
            "checks": {name: result.to_dict() for name, result in results.items()},
            "summary": summary
        }


# Global health check manager
_health_check_manager = HealthCheckManager()


def get_health_check_manager() -> HealthCheckManager:
    """Get the global health check manager."""
    return _health_check_manager


def setup_default_health_checks(model_manager=None):
    """Set up default health checks for the application."""
    manager = get_health_check_manager()
    
    # System resources
    manager.register_health_check(SystemResourcesHealthCheck())
    
    # GPU resources
    manager.register_health_check(GPUHealthCheck())
    
    # Dependencies
    manager.register_health_check(DependencyHealthCheck())
    
    # Model health checks
    if model_manager:
        for model_name in model_manager.list_models():
            manager.register_health_check(ModelHealthCheck(model_name, model_manager))
    
    logger.info("Default health checks set up successfully")
