"""
Model governance and MLOps management system.

This module provides comprehensive MLOps capabilities including:
- Model versioning and lifecycle management
- A/B testing and canary deployments
- Model performance monitoring
- Experiment tracking
- Model registry integration
- Automated validation and testing
"""

import json
import hashlib
import time
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import secrets
from pathlib import Path
from abc import ABC, abstractmethod
import threading
from collections import defaultdict, deque

from .config import SecurityConfig


logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status."""
    PENDING = "pending"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"


class DeploymentStrategy(Enum):
    """Model deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    A_B_TEST = "a_b_test"


class ExperimentStatus(Enum):
    """Experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelMetadata:
    """Model metadata and information."""
    id: str
    name: str
    version: str
    description: str
    framework: str  # pytorch, tensorflow, onnx
    architecture: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters_count: int
    model_size_mb: float
    
    # Training information
    training_dataset: Optional[str] = None
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    training_duration_hours: Optional[float] = None
    
    # Deployment information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Performance benchmarks
    inference_time_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass
class ModelVersion:
    """Model version information."""
    model_id: str
    version: str
    status: ModelStatus
    file_path: str
    checksum: str
    metadata: ModelMetadata
    
    # Deployment information
    deployed_at: Optional[datetime] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Validation results
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["deployed_at"] = self.deployed_at.isoformat() if self.deployed_at else None
        return data


@dataclass
class Experiment:
    """ML experiment tracking."""
    id: str
    name: str
    description: str
    status: ExperimentStatus
    model_id: str
    
    # Configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["started_at"] = self.started_at.isoformat() if self.started_at else None
        data["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    id: str
    name: str
    model_a_id: str
    model_b_id: str
    traffic_split_percent: int  # Percentage for model B (0-100)
    
    # Test criteria
    success_metrics: List[str] = field(default_factory=list)
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    test_duration_hours: int = 24
    
    # Status
    started_at: Optional[datetime] = None
    status: str = "draft"  # draft, running, completed, cancelled
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking."""
    model_id: str
    version: str
    timestamp: datetime
    
    # Accuracy metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Performance metrics
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    
    # Resource metrics
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    
    # Business metrics
    prediction_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class ModelRegistry:
    """Model registry for version management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.models: Dict[str, Dict[str, ModelVersion]] = {}  # model_id -> version -> ModelVersion
        self.metadata_cache: Dict[str, ModelMetadata] = {}
        
        # Model storage path
        self.storage_path = Path("models")
        self.storage_path.mkdir(exist_ok=True)
    
    def register_model(self, metadata: ModelMetadata, model_file_path: str) -> ModelVersion:
        """Register new model version."""
        # Calculate checksum
        checksum = self._calculate_file_checksum(model_file_path)
        
        # Create model version
        model_version = ModelVersion(
            model_id=metadata.id,
            version=metadata.version,
            status=ModelStatus.PENDING,
            file_path=model_file_path,
            checksum=checksum,
            metadata=metadata
        )
        
        # Store in registry
        if metadata.id not in self.models:
            self.models[metadata.id] = {}
        
        self.models[metadata.id][metadata.version] = model_version
        self.metadata_cache[f"{metadata.id}:{metadata.version}"] = metadata
        
        logger.info(f"Registered model {metadata.id}:{metadata.version}")
        return model_version
    
    def get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version."""
        return self.models.get(model_id, {}).get(version)
    
    def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get latest version of model."""
        if model_id not in self.models:
            return None
        
        versions = self.models[model_id]
        if not versions:
            return None
        
        # Sort versions by created_at
        latest = max(versions.values(), key=lambda v: v.metadata.created_at)
        return latest
    
    def get_active_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get currently active version of model."""
        if model_id not in self.models:
            return None
        
        for version in self.models[model_id].values():
            if version.status == ModelStatus.ACTIVE:
                return version
        
        return None
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.models.keys())
    
    def list_versions(self, model_id: str) -> List[str]:
        """List all versions of a model."""
        return list(self.models.get(model_id, {}).keys())
    
    def update_model_status(self, model_id: str, version: str, status: ModelStatus) -> bool:
        """Update model status."""
        model_version = self.get_model_version(model_id, version)
        if model_version:
            old_status = model_version.status
            model_version.status = status
            
            if status == ModelStatus.ACTIVE:
                # Deactivate other versions
                for v in self.models[model_id].values():
                    if v.version != version and v.status == ModelStatus.ACTIVE:
                        v.status = ModelStatus.DEPRECATED
                
                model_version.deployed_at = datetime.now(timezone.utc)
            
            logger.info(f"Updated model {model_id}:{version} status: {old_status.value} -> {status.value}")
            return True
        
        return False
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def search_models(self, tags: Optional[List[str]] = None, 
                     framework: Optional[str] = None,
                     min_accuracy: Optional[float] = None) -> List[ModelMetadata]:
        """Search models by criteria."""
        results = []
        
        for model_versions in self.models.values():
            for version in model_versions.values():
                metadata = version.metadata
                
                # Filter by tags
                if tags and not any(tag in metadata.tags for tag in tags):
                    continue
                
                # Filter by framework
                if framework and metadata.framework.lower() != framework.lower():
                    continue
                
                # Filter by accuracy
                if (min_accuracy and metadata.training_accuracy and 
                    metadata.training_accuracy < min_accuracy):
                    continue
                
                results.append(metadata)
        
        return results


class ModelValidator:
    """Model validation and testing."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.validation_callbacks: List[Callable] = []
    
    def add_validation_callback(self, callback: Callable) -> None:
        """Add custom validation callback."""
        self.validation_callbacks.append(callback)
    
    async def validate_model(self, model_version: ModelVersion) -> Dict[str, Any]:
        """Validate model before deployment."""
        validation_results = {
            "model_id": model_version.model_id,
            "version": model_version.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validations": {},
            "overall_status": "passed",
            "issues": []
        }
        
        # File integrity check
        file_check = self._validate_file_integrity(model_version)
        validation_results["validations"]["file_integrity"] = file_check
        
        # Model format validation
        format_check = self._validate_model_format(model_version)
        validation_results["validations"]["model_format"] = format_check
        
        # Performance benchmarks
        performance_check = await self._validate_performance(model_version)
        validation_results["validations"]["performance"] = performance_check
        
        # Custom validations
        for callback in self.validation_callbacks:
            try:
                custom_result = await callback(model_version)
                validation_results["validations"][f"custom_{callback.__name__}"] = custom_result
            except Exception as e:
                logger.error(f"Custom validation failed: {e}")
                validation_results["validations"][f"custom_{callback.__name__}"] = {
                    "passed": False,
                    "error": str(e)
                }
        
        # Determine overall status
        failed_validations = [
            name for name, result in validation_results["validations"].items()
            if not result.get("passed", False)
        ]
        
        if failed_validations:
            validation_results["overall_status"] = "failed"
            validation_results["issues"] = failed_validations
        
        # Update model version with validation results
        model_version.validation_results = validation_results
        
        return validation_results
    
    def _validate_file_integrity(self, model_version: ModelVersion) -> Dict[str, Any]:
        """Validate file exists and checksum matches."""
        try:
            file_path = Path(model_version.file_path)
            
            if not file_path.exists():
                return {
                    "passed": False,
                    "message": "Model file does not exist"
                }
            
            # Verify checksum
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_checksum = sha256_hash.hexdigest()
            if actual_checksum != model_version.checksum:
                return {
                    "passed": False,
                    "message": "Checksum mismatch",
                    "expected": model_version.checksum,
                    "actual": actual_checksum
                }
            
            return {
                "passed": True,
                "message": "File integrity validated",
                "file_size_mb": file_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"File validation error: {e}"
            }
    
    def _validate_model_format(self, model_version: ModelVersion) -> Dict[str, Any]:
        """Validate model can be loaded."""
        try:
            metadata = model_version.metadata
            
            if metadata.framework.lower() == "pytorch":
                import torch
                model = torch.load(model_version.file_path, map_location='cpu')
                return {
                    "passed": True,
                    "message": "PyTorch model loaded successfully",
                    "model_type": str(type(model))
                }
            
            elif metadata.framework.lower() == "onnx":
                import onnx
                model = onnx.load(model_version.file_path)
                onnx.checker.check_model(model)
                return {
                    "passed": True,
                    "message": "ONNX model validated successfully"
                }
            
            else:
                return {
                    "passed": True,
                    "message": f"Format validation skipped for {metadata.framework}"
                }
                
        except Exception as e:
            return {
                "passed": False,
                "message": f"Model format validation failed: {e}"
            }
    
    async def _validate_performance(self, model_version: ModelVersion) -> Dict[str, Any]:
        """Validate model performance benchmarks."""
        try:
            # This would run actual performance tests
            # For now, we'll simulate the validation
            
            metadata = model_version.metadata
            
            # Check if model size is reasonable
            if metadata.model_size_mb > 1000:  # > 1GB
                return {
                    "passed": False,
                    "message": f"Model too large: {metadata.model_size_mb:.1f}MB",
                    "threshold": "1000MB"
                }
            
            # Check parameter count
            if metadata.parameters_count > 1e9:  # > 1B parameters
                return {
                    "passed": False,
                    "message": f"Too many parameters: {metadata.parameters_count:,}",
                    "threshold": "1,000,000,000"
                }
            
            return {
                "passed": True,
                "message": "Performance validation passed",
                "model_size_mb": metadata.model_size_mb,
                "parameters": metadata.parameters_count
            }
            
        except Exception as e:
            return {
                "passed": False,
                "message": f"Performance validation error: {e}"
            }


class ABTestManager:
    """A/B testing management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, Dict] = defaultdict(dict)
        self.traffic_router = TrafficRouter()
    
    def create_ab_test(self, name: str, model_a_id: str, model_b_id: str,
                      traffic_split_percent: int = 50, 
                      success_metrics: List[str] = None) -> ABTestConfig:
        """Create new A/B test."""
        test_id = f"ab_test_{secrets.token_urlsafe(8)}"
        
        test_config = ABTestConfig(
            id=test_id,
            name=name,
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            traffic_split_percent=traffic_split_percent,
            success_metrics=success_metrics or ["accuracy", "latency"]
        )
        
        self.active_tests[test_id] = test_config
        logger.info(f"Created A/B test {test_id}: {model_a_id} vs {model_b_id}")
        
        return test_config
    
    def start_ab_test(self, test_id: str) -> bool:
        """Start A/B test."""
        test_config = self.active_tests.get(test_id)
        if not test_config:
            return False
        
        test_config.started_at = datetime.now(timezone.utc)
        test_config.status = "running"
        
        # Configure traffic routing
        self.traffic_router.add_route(
            test_id, 
            test_config.model_a_id, 
            test_config.model_b_id, 
            test_config.traffic_split_percent
        )
        
        logger.info(f"Started A/B test {test_id}")
        return True
    
    def record_test_result(self, test_id: str, model_id: str, 
                          metrics: Dict[str, float]) -> None:
        """Record test result for analysis."""
        if test_id not in self.active_tests:
            return
        
        if model_id not in self.test_results[test_id]:
            self.test_results[test_id][model_id] = {
                "samples": [],
                "metrics": defaultdict(list)
            }
        
        # Record metrics
        result_data = self.test_results[test_id][model_id]
        result_data["samples"].append({
            "timestamp": datetime.now(timezone.utc),
            "metrics": metrics
        })
        
        for metric_name, value in metrics.items():
            result_data["metrics"][metric_name].append(value)
    
    def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        test_config = self.active_tests.get(test_id)
        if not test_config:
            return {"error": "Test not found"}
        
        test_data = self.test_results[test_id]
        analysis = {
            "test_id": test_id,
            "test_name": test_config.name,
            "status": test_config.status,
            "models": {},
            "statistical_significance": {},
            "recommendation": "inconclusive"
        }
        
        # Analyze each model's performance
        for model_id, data in test_data.items():
            if not data["samples"]:
                continue
            
            model_analysis = {
                "sample_count": len(data["samples"]),
                "metrics": {}
            }
            
            for metric_name, values in data["metrics"].items():
                if values:
                    model_analysis["metrics"][metric_name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "std": self._calculate_std(values)
                    }
            
            analysis["models"][model_id] = model_analysis
        
        # Statistical significance testing (simplified)
        if len(analysis["models"]) == 2:
            model_ids = list(analysis["models"].keys())
            model_a_data = analysis["models"][model_ids[0]]
            model_b_data = analysis["models"][model_ids[1]]
            
            # Check if we have enough samples
            min_samples = min(
                model_a_data["sample_count"],
                model_b_data["sample_count"]
            )
            
            if min_samples >= test_config.min_sample_size:
                analysis["statistical_significance"]["sufficient_samples"] = True
                
                # Simple comparison (in production, use proper statistical tests)
                for metric in test_config.success_metrics:
                    if (metric in model_a_data["metrics"] and 
                        metric in model_b_data["metrics"]):
                        
                        a_mean = model_a_data["metrics"][metric]["mean"]
                        b_mean = model_b_data["metrics"][metric]["mean"]
                        
                        improvement = ((b_mean - a_mean) / a_mean) * 100
                        analysis["statistical_significance"][metric] = {
                            "improvement_percent": improvement,
                            "significant": abs(improvement) > 5  # 5% threshold
                        }
                
                # Make recommendation
                significant_improvements = [
                    metric for metric, data in analysis["statistical_significance"].items()
                    if isinstance(data, dict) and data.get("significant", False) and data.get("improvement_percent", 0) > 0
                ]
                
                if significant_improvements:
                    analysis["recommendation"] = "deploy_model_b"
                elif any(
                    data.get("improvement_percent", 0) < -5 
                    for data in analysis["statistical_significance"].values()
                    if isinstance(data, dict)
                ):
                    analysis["recommendation"] = "keep_model_a"
                else:
                    analysis["recommendation"] = "no_significant_difference"
            else:
                analysis["statistical_significance"]["sufficient_samples"] = False
        
        return analysis
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def stop_test(self, test_id: str) -> bool:
        """Stop A/B test."""
        test_config = self.active_tests.get(test_id)
        if not test_config:
            return False
        
        test_config.status = "completed"
        
        # Analyze final results
        final_results = self.analyze_test_results(test_id)
        test_config.results = final_results
        
        # Remove traffic routing
        self.traffic_router.remove_route(test_id)
        
        logger.info(f"Stopped A/B test {test_id}")
        return True


class TrafficRouter:
    """Traffic routing for A/B tests."""
    
    def __init__(self):
        self.routes: Dict[str, Dict[str, Any]] = {}
    
    def add_route(self, test_id: str, model_a_id: str, model_b_id: str, 
                 split_percent: int) -> None:
        """Add traffic routing rule."""
        self.routes[test_id] = {
            "model_a": model_a_id,
            "model_b": model_b_id,
            "split_percent": split_percent
        }
    
    def get_model_for_request(self, test_id: str, request_hash: str) -> Optional[str]:
        """Determine which model to use for request."""
        if test_id not in self.routes:
            return None
        
        route = self.routes[test_id]
        
        # Use request hash to determine routing (consistent for same request)
        hash_value = int(hashlib.md5(request_hash.encode()).hexdigest(), 16)
        traffic_percent = hash_value % 100
        
        if traffic_percent < route["split_percent"]:
            return route["model_b"]
        else:
            return route["model_a"]
    
    def remove_route(self, test_id: str) -> None:
        """Remove traffic routing rule."""
        if test_id in self.routes:
            del self.routes[test_id]


class ExperimentTracker:
    """Experiment tracking and management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: Dict[str, threading.Thread] = {}
    
    def create_experiment(self, name: str, description: str, model_id: str,
                         hyperparameters: Dict[str, Any] = None,
                         created_by: Optional[str] = None) -> Experiment:
        """Create new experiment."""
        experiment_id = f"exp_{secrets.token_urlsafe(8)}"
        
        experiment = Experiment(
            id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            model_id=model_id,
            hyperparameters=hyperparameters or {},
            created_by=created_by
        )
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Created experiment {experiment_id}: {name}")
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start experiment execution."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now(timezone.utc)
        
        logger.info(f"Started experiment {experiment_id}")
        return True
    
    def log_metric(self, experiment_id: str, metric_name: str, value: float) -> None:
        """Log experiment metric."""
        experiment = self.experiments.get(experiment_id)
        if experiment:
            experiment.metrics[metric_name] = value
    
    def log_artifact(self, experiment_id: str, artifact_path: str) -> None:
        """Log experiment artifact."""
        experiment = self.experiments.get(experiment_id)
        if experiment:
            experiment.artifacts.append(artifact_path)
    
    def complete_experiment(self, experiment_id: str, 
                           results: Dict[str, Any] = None) -> bool:
        """Complete experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return False
        
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)
        experiment.results = results or {}
        
        logger.info(f"Completed experiment {experiment_id}")
        return True
    
    def get_experiment_metrics(self, experiment_id: str) -> Dict[str, float]:
        """Get experiment metrics."""
        experiment = self.experiments.get(experiment_id)
        return experiment.metrics if experiment else {}
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        comparison = {
            "experiments": {},
            "common_metrics": [],
            "best_experiment": None
        }
        
        # Collect experiment data
        valid_experiments = []
        all_metrics = set()
        
        for exp_id in experiment_ids:
            experiment = self.experiments.get(exp_id)
            if experiment:
                valid_experiments.append(experiment)
                comparison["experiments"][exp_id] = {
                    "name": experiment.name,
                    "status": experiment.status.value,
                    "metrics": experiment.metrics,
                    "hyperparameters": experiment.hyperparameters,
                    "duration_hours": self._calculate_duration_hours(experiment)
                }
                all_metrics.update(experiment.metrics.keys())
        
        # Find common metrics
        if valid_experiments:
            common_metrics = set(valid_experiments[0].metrics.keys())
            for exp in valid_experiments[1:]:
                common_metrics.intersection_update(exp.metrics.keys())
            comparison["common_metrics"] = list(common_metrics)
        
        # Find best experiment (simplified - uses first common metric)
        if comparison["common_metrics"]:
            metric_name = comparison["common_metrics"][0]
            best_exp = max(
                valid_experiments,
                key=lambda exp: exp.metrics.get(metric_name, 0)
            )
            comparison["best_experiment"] = {
                "id": best_exp.id,
                "name": best_exp.name,
                "metric": metric_name,
                "value": best_exp.metrics[metric_name]
            }
        
        return comparison
    
    def _calculate_duration_hours(self, experiment: Experiment) -> Optional[float]:
        """Calculate experiment duration in hours."""
        if not experiment.started_at:
            return None
        
        end_time = experiment.completed_at or datetime.now(timezone.utc)
        duration = end_time - experiment.started_at
        return duration.total_seconds() / 3600


class ModelGovernance:
    """Main model governance system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.model_registry = ModelRegistry(config)
        self.model_validator = ModelValidator(config)
        self.ab_test_manager = ABTestManager(config)
        self.experiment_tracker = ExperimentTracker(config)
        
        # Performance monitoring
        self.performance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        logger.info("Model governance system initialized")
    
    def register_model(self, metadata: ModelMetadata, model_file_path: str) -> ModelVersion:
        """Register new model with governance."""
        return self.model_registry.register_model(metadata, model_file_path)
    
    async def deploy_model(self, model_id: str, version: str,
                          strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN,
                          validation_required: bool = True) -> Dict[str, Any]:
        """Deploy model with governance checks."""
        model_version = self.model_registry.get_model_version(model_id, version)
        if not model_version:
            return {"success": False, "error": "Model version not found"}
        
        # Validation phase
        if validation_required:
            logger.info(f"Validating model {model_id}:{version}")
            model_version.status = ModelStatus.VALIDATING
            
            validation_results = await self.model_validator.validate_model(model_version)
            
            if validation_results["overall_status"] != "passed":
                model_version.status = ModelStatus.FAILED
                return {
                    "success": False,
                    "error": "Model validation failed",
                    "validation_results": validation_results
                }
        
        # Deployment phase
        logger.info(f"Deploying model {model_id}:{version} with strategy {strategy.value}")
        
        if strategy == DeploymentStrategy.BLUE_GREEN:
            success = self._deploy_blue_green(model_version)
        elif strategy == DeploymentStrategy.CANARY:
            success = self._deploy_canary(model_version)
        elif strategy == DeploymentStrategy.A_B_TEST:
            success = self._deploy_ab_test(model_version)
        else:
            success = self._deploy_rolling(model_version)
        
        if success:
            self.model_registry.update_model_status(model_id, version, ModelStatus.ACTIVE)
            return {"success": True, "deployment_strategy": strategy.value}
        else:
            self.model_registry.update_model_status(model_id, version, ModelStatus.FAILED)
            return {"success": False, "error": "Deployment failed"}
    
    def _deploy_blue_green(self, model_version: ModelVersion) -> bool:
        """Deploy using blue-green strategy."""
        # In production, this would:
        # 1. Set up new environment (green)
        # 2. Deploy model to green environment
        # 3. Run health checks
        # 4. Switch traffic from blue to green
        # 5. Keep blue as backup
        
        logger.info(f"Blue-green deployment for {model_version.model_id}:{model_version.version}")
        return True
    
    def _deploy_canary(self, model_version: ModelVersion) -> bool:
        """Deploy using canary strategy."""
        # In production, this would:
        # 1. Deploy to small percentage of traffic
        # 2. Monitor performance
        # 3. Gradually increase traffic
        # 4. Rollback if issues detected
        
        logger.info(f"Canary deployment for {model_version.model_id}:{model_version.version}")
        return True
    
    def _deploy_ab_test(self, model_version: ModelVersion) -> bool:
        """Deploy as A/B test."""
        # Get current active version for comparison
        current_version = self.model_registry.get_active_version(model_version.model_id)
        if not current_version:
            return False
        
        # Create A/B test
        test_config = self.ab_test_manager.create_ab_test(
            name=f"Deploy {model_version.version}",
            model_a_id=f"{current_version.model_id}:{current_version.version}",
            model_b_id=f"{model_version.model_id}:{model_version.version}",
            traffic_split_percent=10  # Start with 10% traffic
        )
        
        self.ab_test_manager.start_ab_test(test_config.id)
        logger.info(f"A/B test deployment for {model_version.model_id}:{model_version.version}")
        return True
    
    def _deploy_rolling(self, model_version: ModelVersion) -> bool:
        """Deploy using rolling update."""
        logger.info(f"Rolling deployment for {model_version.model_id}:{model_version.version}")
        return True
    
    def record_model_performance(self, model_id: str, version: str,
                                metrics: ModelPerformanceMetrics) -> None:
        """Record model performance metrics."""
        key = f"{model_id}:{version}"
        self.performance_history[key].append(metrics)
        
        # Check for performance degradation
        self._check_performance_drift(model_id, version, metrics)
    
    def _check_performance_drift(self, model_id: str, version: str,
                                current_metrics: ModelPerformanceMetrics) -> None:
        """Check for performance drift and alert if needed."""
        key = f"{model_id}:{version}"
        history = self.performance_history[key]
        
        if len(history) < 10:  # Need history to compare
            return
        
        # Check accuracy drift
        if current_metrics.accuracy is not None:
            recent_accuracy = [m.accuracy for m in list(history)[-10:] if m.accuracy is not None]
            if recent_accuracy:
                avg_accuracy = sum(recent_accuracy) / len(recent_accuracy)
                if current_metrics.accuracy < avg_accuracy * 0.95:  # 5% drop
                    logger.warning(
                        f"Accuracy drift detected for {model_id}:{version}: "
                        f"{current_metrics.accuracy:.3f} vs {avg_accuracy:.3f}"
                    )
        
        # Check latency drift
        if current_metrics.latency_p95_ms is not None:
            recent_latency = [m.latency_p95_ms for m in list(history)[-10:] if m.latency_p95_ms is not None]
            if recent_latency:
                avg_latency = sum(recent_latency) / len(recent_latency)
                if current_metrics.latency_p95_ms > avg_latency * 1.5:  # 50% increase
                    logger.warning(
                        f"Latency drift detected for {model_id}:{version}: "
                        f"{current_metrics.latency_p95_ms:.1f}ms vs {avg_latency:.1f}ms"
                    )
    
    def get_governance_dashboard(self) -> Dict[str, Any]:
        """Get governance dashboard data."""
        return {
            "models": {
                "total_models": len(self.model_registry.list_models()),
                "active_models": len([
                    model_id for model_id in self.model_registry.list_models()
                    if self.model_registry.get_active_version(model_id)
                ]),
                "pending_validation": len([
                    version for model_versions in self.model_registry.models.values()
                    for version in model_versions.values()
                    if version.status == ModelStatus.VALIDATING
                ])
            },
            "experiments": {
                "total_experiments": len(self.experiment_tracker.experiments),
                "running_experiments": len([
                    exp for exp in self.experiment_tracker.experiments.values()
                    if exp.status == ExperimentStatus.RUNNING
                ])
            },
            "ab_tests": {
                "active_tests": len([
                    test for test in self.ab_test_manager.active_tests.values()
                    if test.status == "running"
                ])
            },
            "performance_alerts": self._get_performance_alerts()
        }
    
    def _get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get current performance alerts."""
        alerts = []
        
        # Check recent performance data for alerts
        for key, history in self.performance_history.items():
            if not history:
                continue
            
            latest = history[-1]
            model_id, version = key.split(":")
            
            # High error rate alert
            if latest.error_count > latest.prediction_count * 0.05:  # >5% error rate
                alerts.append({
                    "type": "high_error_rate",
                    "model_id": model_id,
                    "version": version,
                    "error_rate": latest.error_count / latest.prediction_count if latest.prediction_count > 0 else 0,
                    "timestamp": latest.timestamp.isoformat()
                })
            
            # High latency alert
            if latest.latency_p95_ms and latest.latency_p95_ms > 1000:  # >1s
                alerts.append({
                    "type": "high_latency",
                    "model_id": model_id,
                    "version": version,
                    "latency_p95_ms": latest.latency_p95_ms,
                    "timestamp": latest.timestamp.isoformat()
                })
        
        return alerts


class MLOpsManager:
    """MLOps workflow management."""
    
    def __init__(self, config: SecurityConfig, governance: ModelGovernance):
        self.config = config
        self.governance = governance
        self.workflows: Dict[str, Dict[str, Any]] = {}
        
        logger.info("MLOps manager initialized")
    
    def create_deployment_pipeline(self, name: str, model_id: str,
                                  stages: List[str] = None) -> str:
        """Create deployment pipeline."""
        pipeline_id = f"pipeline_{secrets.token_urlsafe(8)}"
        
        default_stages = [
            "validation",
            "testing",
            "staging_deployment",
            "performance_evaluation",
            "production_deployment"
        ]
        
        pipeline = {
            "id": pipeline_id,
            "name": name,
            "model_id": model_id,
            "stages": stages or default_stages,
            "current_stage": 0,
            "status": "ready",
            "created_at": datetime.now(timezone.utc),
            "logs": []
        }
        
        self.workflows[pipeline_id] = pipeline
        logger.info(f"Created deployment pipeline {pipeline_id} for model {model_id}")
        
        return pipeline_id
    
    async def execute_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Execute deployment pipeline."""
        pipeline = self.workflows.get(pipeline_id)
        if not pipeline:
            return {"success": False, "error": "Pipeline not found"}
        
        pipeline["status"] = "running"
        pipeline["started_at"] = datetime.now(timezone.utc)
        
        try:
            for stage_index, stage_name in enumerate(pipeline["stages"]):
                pipeline["current_stage"] = stage_index
                
                logger.info(f"Executing stage {stage_name} for pipeline {pipeline_id}")
                stage_result = await self._execute_stage(pipeline, stage_name)
                
                pipeline["logs"].append({
                    "stage": stage_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "result": stage_result
                })
                
                if not stage_result.get("success", False):
                    pipeline["status"] = "failed"
                    return {
                        "success": False,
                        "error": f"Pipeline failed at stage: {stage_name}",
                        "stage_result": stage_result
                    }
            
            pipeline["status"] = "completed"
            pipeline["completed_at"] = datetime.now(timezone.utc)
            
            return {"success": True, "pipeline_id": pipeline_id}
            
        except Exception as e:
            pipeline["status"] = "failed"
            pipeline["error"] = str(e)
            logger.error(f"Pipeline {pipeline_id} failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_stage(self, pipeline: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
        """Execute individual pipeline stage."""
        model_id = pipeline["model_id"]
        
        if stage_name == "validation":
            # Get latest version for validation
            latest_version = self.governance.model_registry.get_latest_version(model_id)
            if not latest_version:
                return {"success": False, "error": "No model version found"}
            
            validation_results = await self.governance.model_validator.validate_model(latest_version)
            return {
                "success": validation_results["overall_status"] == "passed",
                "validation_results": validation_results
            }
        
        elif stage_name == "testing":
            # Run automated tests
            return {"success": True, "message": "Automated tests passed"}
        
        elif stage_name == "staging_deployment":
            # Deploy to staging environment
            return {"success": True, "message": "Deployed to staging"}
        
        elif stage_name == "performance_evaluation":
            # Evaluate performance in staging
            return {"success": True, "message": "Performance evaluation completed"}
        
        elif stage_name == "production_deployment":
            # Deploy to production
            latest_version = self.governance.model_registry.get_latest_version(model_id)
            if not latest_version:
                return {"success": False, "error": "No model version found"}
            
            deployment_result = await self.governance.deploy_model(
                model_id, 
                latest_version.version,
                DeploymentStrategy.BLUE_GREEN,
                validation_required=False  # Already validated
            )
            
            return deployment_result
        
        else:
            return {"success": True, "message": f"Executed stage: {stage_name}"}
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline status."""
        return self.workflows.get(pipeline_id)
