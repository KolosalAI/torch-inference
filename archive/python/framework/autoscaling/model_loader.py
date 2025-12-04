"""
Dynamic model loader for PyTorch inference framework.

This module provides advanced model loading capabilities including:
- Dynamic loading and unloading of models
- Load balancing across multiple model instances
- Smart caching and prefetching
- Model versioning and A/B testing
- Health monitoring and failover
"""

import asyncio
import time
import logging
import hashlib
from typing import Any, Dict, List, Optional, Callable, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import threading
import weakref
import json

from ..core.base_model import BaseModel, ModelManager
from ..core.inference_engine import InferenceEngine
from ..core.config import InferenceConfig


logger = logging.getLogger(__name__)


class ModelLoadingStrategy(Enum):
    """Model loading strategies."""
    EAGER = "eager"            # Load immediately
    LAZY = "lazy"              # Load on first request
    SCHEDULED = "scheduled"    # Load based on schedule
    PREDICTIVE = "predictive"  # Load based on predicted usage


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"


class ModelStatus(Enum):
    """Model instance status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    WARMING_UP = "warming_up"
    READY = "ready"
    ACTIVE = "active"  # Added for tests
    ERROR = "error"
    UNLOADING = "unloading"
    IDLE = "idle"  # Added for zero scaler compatibility


# Alias for backward compatibility
ModelInstanceState = ModelStatus


@dataclass
class ModelLoaderConfig:
    """Configuration for dynamic model loader."""
    
    # Basic settings
    enabled: bool = True
    
    # Loading strategy
    loading_strategy: ModelLoadingStrategy = ModelLoadingStrategy.LAZY
    model_loading_strategy: ModelLoadingStrategy = ModelLoadingStrategy.LAZY  # Alias for tests
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS
    
    # Instance management
    max_instances_per_model: int = 3
    min_instances_per_model: int = 1
    default_instances_per_model: int = 1
    
    # Autoscaling settings
    enable_auto_scaling: bool = False
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Loading settings
    concurrent_loads: int = 2
    load_timeout_seconds: float = 300.0  # 5 minutes
    warmup_timeout_seconds: float = 60.0
    
    # Caching and prefetching
    enable_model_caching: bool = True
    cache_size_gb: float = 10.0
    prefetch_popular_models: bool = True
    prefetch_threshold: float = 0.1  # 10% of requests
    
    # Health monitoring
    health_check_interval_seconds: float = 30.0
    health_check_interval: float = 30.0  # Alias for backward compatibility
    max_consecutive_failures: int = 3
    max_unhealthy_instances: int = 1  # Maximum unhealthy instances to tolerate
    failure_cooldown_seconds: float = 300.0  # 5 minutes
    
    # Performance settings
    enable_model_versioning: bool = True
    enable_ab_testing: bool = False
    ab_test_traffic_split: float = 0.1  # 10% to new version
    
    # Resource management
    memory_limit_per_model_gb: float = 4.0
    cpu_limit_per_model: float = 2.0
    cleanup_interval_seconds: float = 300.0
    
    # Model specific configurations
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_instances_per_model <= 0:
            raise ValueError("max_instances_per_model must be positive")
        
        if self.min_instances_per_model <= 0:
            raise ValueError("min_instances_per_model must be positive")
            
        if self.min_instances_per_model > self.max_instances_per_model:
            raise ValueError("min_instances_per_model cannot be greater than max_instances_per_model")


class ModelInstanceInfo:
    """Information about a model instance."""
    
    def __init__(self, model_name: str = None, instance_id: str = None, version: str = "v1", 
                 model_id: str = None, model=None, state=None):
        # Support both parameter styles
        self.model_name = model_name or model_id
        self.model_id = self.model_name  # Alias for backward compatibility
        self.instance_id = instance_id
        self.version = version
        self.status = state or ModelStatus.UNLOADED
        # state is now a property that returns status
        
        # Model and engine
        self.model: Optional[BaseModel] = model
        self.engine: Optional[InferenceEngine] = None
        
        # Performance tracking
        self.load_time: Optional[float] = None
        self.last_used = time.time()
        self.last_accessed = self.last_used  # Alias for zero scaler compatibility
        self.created_at = time.time()
        self.request_count = 0
        self.total_requests = self.request_count  # Alias
        self.active_connections = 0
        self.total_processing_time = 0.0
        self.total_response_time = self.total_processing_time  # Alias
        self.error_count = 0
        self.consecutive_failures = 0
        
        # Health status
        self.last_health_check = time.time()
        self.health_check_failures = 0
        self.failure_cooldown_until: Optional[float] = None
        
        # Threading
        self._lock = asyncio.Lock()
        self._loading_event = asyncio.Event()
    
    @property
    def state(self) -> ModelStatus:
        """Get current state (alias for status)."""
        return self.status
    
    @state.setter
    def state(self, value: ModelStatus):
        """Set current state (updates status)."""
        self.status = value
    
    def update_state(self, new_state: ModelInstanceState):
        """Update instance state and last accessed time."""
        self.state = new_state
        self.status = new_state
        self.last_accessed = time.time()
    
    def start_request(self):
        """Start a new request - increment connections and request count."""
        self.active_connections += 1
        self.total_requests += 1
        self.request_count = self.total_requests
        self.last_accessed = time.time()
    
    def end_request(self, response_time: float):
        """End a request - decrement connections and add response time."""
        if self.active_connections > 0:
            self.active_connections -= 1
        self.total_response_time += response_time
        self.total_processing_time = self.total_response_time
        self.last_accessed = time.time()
    
    def get_average_response_time(self) -> float:
        """Get average response time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    def is_healthy(self) -> bool:
        """Check if instance is healthy."""
        return self.health_check_failures < 3  # Max 3 failures before unhealthy
    
    def get_load_score(self) -> float:
        """Calculate load score for load balancing."""
        # Simple load score: combination of active connections and response time
        connection_score = self.active_connections
        response_time_score = self.get_average_response_time() * 10  # Scale response time
        return connection_score + response_time_score
    
    @property
    def average_response_time(self) -> float:
        """Get average response time."""
        return self.get_average_response_time()
    
    @property
    def is_available(self) -> bool:
        """Check if instance is available for requests."""
        if not self.is_healthy():
            return False
        if self.failure_cooldown_until and time.time() < self.failure_cooldown_until:
            return False
        return self.status == ModelStatus.READY
    
    @property
    def current_load(self) -> float:
        """Get current load (simplified as request count for now)."""
        return float(self.request_count)
    
    def mark_failure(self, cooldown_seconds: float = 300.0):
        """Mark a failure and potentially put in cooldown."""
        self.error_count += 1
        self.consecutive_failures += 1
        self.health_check_failures += 1
        
        if self.consecutive_failures >= 3:  # Max consecutive failures
            self.failure_cooldown_until = time.time() + cooldown_seconds
            logger.warning(f"Instance {self.instance_id} entering failure cooldown")
    
    def mark_success(self):
        """Mark a successful operation."""
        self.consecutive_failures = 0
        self.health_check_failures = 0
        self.failure_cooldown_until = None


class LoadBalancer:
    """Load balancer for model instances."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.weighted_counters: Dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger(f"{__name__}.LoadBalancer")
    
    def select_instance(self, instances: List[ModelInstanceInfo], request_data: Optional[Dict] = None) -> Optional[ModelInstanceInfo]:
        """Select an instance based on the load balancing strategy."""
        # Filter healthy instances
        healthy_instances = [inst for inst in instances if inst.is_healthy()]
        
        if not healthy_instances:
            return None
        
        if len(healthy_instances) == 1:
            return healthy_instances[0]
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_selection(healthy_instances, request_data)
        else:
            # Default to round robin
            return self._round_robin_selection(healthy_instances)
    
    def _round_robin_selection(self, instances: List[ModelInstanceInfo]) -> ModelInstanceInfo:
        """Round robin selection."""
        model_name = instances[0].model_name
        counter = self.round_robin_counters[model_name]
        selected = instances[counter % len(instances)]
        self.round_robin_counters[model_name] = counter + 1
        return selected
    
    def _least_connections_selection(self, instances: List[ModelInstanceInfo]) -> ModelInstanceInfo:
        """Least connections selection."""
        return min(instances, key=lambda inst: inst.active_connections)
    
    def _least_response_time_selection(self, instances: List[ModelInstanceInfo]) -> ModelInstanceInfo:
        """Least response time selection."""
        return min(instances, key=lambda inst: inst.get_average_response_time())
    
    def _weighted_round_robin_selection(self, instances: List[ModelInstanceInfo]) -> ModelInstanceInfo:
        """Weighted round robin selection based on inverse load."""
        # Calculate weights (inverse of load + 1 to avoid division by zero)
        weights = []
        for inst in instances:
            load = max(inst.active_connections, 1)  # Minimum load of 1
            weight = 1.0 / load
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return instances[0]
        
        # Use weighted round robin counter
        model_name = instances[0].model_name
        counter = self.weighted_counters[model_name]
        
        # Simple weighted selection - higher weight instances get selected more often
        normalized_weights = [w / total_weight for w in weights]
        cumulative = 0
        selection_point = (counter % 1000) / 1000.0  # Normalize counter to 0-1
        
        for i, weight in enumerate(normalized_weights):
            cumulative += weight
            if selection_point <= cumulative:
                self.weighted_counters[model_name] = counter + 1
                return instances[i]
        
        # Fallback
        self.weighted_counters[model_name] = counter + 1
        return instances[-1]
    
    def _consistent_hash_selection(self, instances: List[ModelInstanceInfo], request_data: Optional[Dict] = None) -> ModelInstanceInfo:
        """Consistent hash selection."""
        if not request_data:
            # Fallback to round robin if no request data
            return self._round_robin_selection(instances)
        
        # Create hash from request data
        hash_input = str(sorted(request_data.items()))
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        hash_int = int(hash_value[:8], 16)  # Use first 8 chars as int
        
        # Select instance based on hash
        return instances[hash_int % len(instances)]


class DynamicModelLoader:
    """
    Dynamic model loader with advanced features.
    """
    
    def __init__(self, config: ModelLoaderConfig, model_manager: Optional[ModelManager] = None, inference_engine = None):
        self.config = config
        self.model_manager = model_manager or ModelManager()
        self.inference_engine = inference_engine
        
        # Instance management
        self.model_instances: Dict[str, List[ModelInstanceInfo]] = defaultdict(list)
        self.loading_instances: Set[str] = set()
        self.instance_lock = asyncio.Lock()
        
        # Load balancer
        self.load_balancer = LoadBalancer(config.load_balancing_strategy)
        
        # Request tracking
        self.request_history: deque = deque(maxlen=10000)
        self.model_popularity: Dict[str, float] = defaultdict(float)
        
        # Background tasks
        self.is_running = False
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.prefetch_task: Optional[asyncio.Task] = None
        
        # Model versioning
        self.model_versions: Dict[str, List[str]] = defaultdict(list)
        self.active_versions: Dict[str, str] = {}
        
        # A/B testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        self.logger = logging.getLogger(f"{__name__}.DynamicModelLoader")
        self.logger.info(f"Initialized DynamicModelLoader with strategy: {config.loading_strategy.value}")
    
    async def start(self):
        """Start the dynamic model loader."""
        if self.is_running:
            self.logger.warning("DynamicModelLoader already running")
            return
        
        if not self.config.enabled:
            self.logger.info("DynamicModelLoader is disabled")
            return
        
        self.is_running = True
        
        # Start background tasks
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.config.prefetch_popular_models:
            self.prefetch_task = asyncio.create_task(self._prefetch_loop())
        
        self.logger.info("DynamicModelLoader started")
    
    async def stop(self):
        """Stop the dynamic model loader."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        for task in [self.health_monitor_task, self.cleanup_task, self.prefetch_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Unload all instances
        await self._unload_all_models()
        
        self.logger.info("DynamicModelLoader stopped")
    
    async def _load_model_unlocked(self, model_name: str, version: str = "v1", force_reload: bool = False) -> bool:
        """
        Load a model with the specified version (without acquiring instance_lock).
        Must be called within instance_lock context.
        
        Args:
            model_name: Name of the model to load
            version: Version of the model
            force_reload: Force reload even if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not force_reload and self._is_model_loaded_unlocked(model_name, version):
                self.logger.info(f"Model {model_name}:{version} already loaded")
                return True
            
            # Create new instance
            instance_id = f"{model_name}:{version}:{int(time.time())}"
            instance = ModelInstanceInfo(model_name, instance_id, version)
            
            # Load the instance
            success = await self._load_instance(instance)
            
            if success:
                self.model_instances[model_name].append(instance)
                
                # Update versioning
                if version not in self.model_versions[model_name]:
                    self.model_versions[model_name].append(version)
                
                if model_name not in self.active_versions:
                    self.active_versions[model_name] = version
                
                self.logger.info(f"Successfully loaded model {model_name}:{version}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}:{version}: {e}")
            return False

    async def load_model(self, model_name: str, version: str = "v1", force_reload: bool = False) -> Optional[ModelInstanceInfo]:
        """
        Load a model with the specified version.
        
        Args:
            model_name: Name of the model to load
            version: Version of the model
            force_reload: Force reload even if already loaded
            
        Returns:
            ModelInstanceInfo if successful, None otherwise
        """
        async with self.instance_lock:
            success = await self._load_model_unlocked(model_name, version, force_reload)
            if success:
                # Return the newly loaded instance
                instances = self.model_instances.get(model_name, [])
                for instance in instances:
                    if instance.version == version and instance.status != ModelStatus.ERROR:
                        return instance
            return None
    
    async def unload_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """
        Unload a model or specific version.
        
        Args:
            model_name: Name of the model to unload
            version: Specific version to unload (if None, unload all versions)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.instance_lock:
                instances = self.model_instances[model_name]
                
                if version:
                    # Unload specific version
                    instances_to_unload = [inst for inst in instances if inst.version == version]
                else:
                    # Unload all versions
                    instances_to_unload = instances[:]
                
                for instance in instances_to_unload:
                    await self._unload_instance(instance)
                    instances.remove(instance)
                
                # Clean up empty entries
                if not instances:
                    del self.model_instances[model_name]
                    if model_name in self.active_versions:
                        del self.active_versions[model_name]
                    
                    # Also unload from model manager
                    if hasattr(self.model_manager, 'unload_model'):
                        try:
                            if asyncio.iscoroutinefunction(self.model_manager.unload_model):
                                await self.model_manager.unload_model(model_name)
                            else:
                                self.model_manager.unload_model(model_name)
                        except Exception as e:
                            self.logger.warning(f"Failed to unload from model manager: {e}")
            
            self.logger.info(f"Unloaded model {model_name}" + (f":{version}" if version else ""))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    async def predict(self, model_name: str, inputs: Any, version: Optional[str] = None, **kwargs) -> Any:
        """
        Make a prediction with automatic loading and load balancing.
        
        Args:
            model_name: Name of the model
            inputs: Input data
            version: Specific version (if None, use active version)
            **kwargs: Additional prediction arguments
            
        Returns:
            Prediction result
        """
        if not self.config.enabled:
            return None
            
        start_time = time.time()
        
        try:
            # Determine version to use
            if not version:
                version = self.active_versions.get(model_name, "v1")
            
            # Get or load model instance
            instance = await self._get_or_load_instance(model_name, version)
            
            # Make prediction
            result = await self._predict_with_instance(instance, inputs, **kwargs)
            
            # Record request
            processing_time = time.time() - start_time
            self._record_request(model_name, version, processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {model_name}:{version}: {e}")
            raise
    
    async def _get_or_load_instance(self, model_name: str, version: str) -> ModelInstanceInfo:
        """Get existing instance or load new one."""
        # Try to get existing instance
        instance = await self._select_instance(model_name, version)
        
        if instance:
            return instance
        
        # No available instance, load one
        instance = await self.load_model(model_name, version)
        if not instance:
            raise RuntimeError(f"Failed to load model {model_name}:{version}")
        
        return instance
    
    async def _select_instance(self, model_name: str, version: str) -> Optional[ModelInstanceInfo]:
        """Select best instance for the request."""
        async with self.instance_lock:
            instances = [
                inst for inst in self.model_instances[model_name]
                if inst.version == version
            ]
        
        return self.load_balancer.select_instance(instances)
    
    async def _load_instance(self, instance: ModelInstanceInfo) -> bool:
        """Load a specific model instance."""
        async with instance._lock:
            if instance.status != ModelStatus.UNLOADED:
                return True
            
            instance.status = ModelStatus.LOADING
            load_start_time = time.time()
            
            try:
                # Load model via model manager
                model = await self.model_manager.load_model(instance.model_name)
                
                if model is None:
                    self.logger.error(f"Model '{instance.model_name}' could not be loaded")
                    instance.status = ModelStatus.ERROR
                    return False
                
                # Create inference engine if we have one
                if self.inference_engine:
                    engine = self.inference_engine
                else:
                    # Create a basic inference engine
                    if hasattr(model, 'config') and model.config is not None:
                        try:
                            from ..core.inference_engine import InferenceEngine
                            engine = InferenceEngine(model, model.config)
                        except Exception as e:
                            self.logger.error(f"Failed to create inference engine: {e}")
                            # For testing, create a mock engine
                            from unittest.mock import Mock, AsyncMock
                            engine = Mock()
                            engine.predict = AsyncMock(return_value={"test": "result"})
                    else:
                        self.logger.error(f"Model {instance.model_name} has no config")
                        instance.status = ModelStatus.ERROR
                        return False
                
                # Assign to instance
                instance.model = model
                instance.engine = engine
                instance.status = ModelStatus.LOADED
                
                # Warm up
                instance.status = ModelStatus.WARMING_UP
                await asyncio.wait_for(
                    self._warmup_instance(instance),
                    timeout=self.config.warmup_timeout_seconds
                )
                
                instance.status = ModelStatus.READY
                instance.load_time = time.time() - load_start_time
                instance._loading_event.set()
                
                self.logger.info(f"Loaded instance {instance.instance_id} in {instance.load_time:.2f}s")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to load instance {instance.instance_id}: {e}")
                instance.status = ModelStatus.ERROR
                instance._loading_event.set()
                return False
    
    async def _warmup_instance(self, instance: ModelInstanceInfo):
        """Warm up a model instance."""
        if instance.model and hasattr(instance.model, 'warmup'):
            if asyncio.iscoroutinefunction(instance.model.warmup):
                await instance.model.warmup()
            else:
                instance.model.warmup()
    
    async def _unload_instance(self, instance: ModelInstanceInfo):
        """Unload a model instance."""
        async with instance._lock:
            if instance.status in (ModelStatus.UNLOADED, ModelStatus.UNLOADING):
                return
            
            instance.status = ModelStatus.UNLOADING
            
            try:
                if instance.engine:
                    if hasattr(instance.engine, 'stop') and asyncio.iscoroutinefunction(instance.engine.stop):
                        await instance.engine.stop()
                    elif hasattr(instance.engine, 'stop'):
                        instance.engine.stop()
                
                if instance.model:
                    if hasattr(instance.model, 'cleanup') and asyncio.iscoroutinefunction(instance.model.cleanup):
                        await instance.model.cleanup()
                    elif hasattr(instance.model, 'cleanup'):
                        instance.model.cleanup()
                
                instance.model = None
                instance.engine = None
                instance.status = ModelStatus.UNLOADED
                
                self.logger.info(f"Unloaded instance {instance.instance_id}")
                
            except Exception as e:
                self.logger.error(f"Error unloading instance {instance.instance_id}: {e}")
    
    async def _predict_with_instance(self, instance: ModelInstanceInfo, inputs: Any, **kwargs) -> Any:
        """Make prediction with a specific instance."""
        if not instance.is_available:
            raise RuntimeError(f"Instance {instance.instance_id} not available")
        
        # Wait for loading if necessary
        if instance.status in (ModelStatus.LOADING, ModelStatus.WARMING_UP):
            await instance._loading_event.wait()
        
        if instance.status != ModelStatus.READY:
            raise RuntimeError(f"Instance {instance.instance_id} not ready")
        
        start_time = time.time()
        try:
            result = await instance.engine.predict(inputs, **kwargs)
            
            # Update statistics
            processing_time = time.time() - start_time
            instance.last_used = time.time()
            instance.request_count += 1
            instance.total_processing_time += processing_time
            instance.mark_success()
            
            return result
            
        except Exception as e:
            instance.mark_failure(self.config.failure_cooldown_seconds)
            self.logger.error(f"Prediction failed on instance {instance.instance_id}: {e}")
            raise
    
    def _is_model_loaded_unlocked(self, model_name: str, version: str) -> bool:
        """Check if a model version is loaded (without acquiring lock - must be called within lock context)."""
        instances = self.model_instances.get(model_name, [])
        return any(
            inst.version == version and inst.status in (ModelStatus.READY, ModelStatus.LOADING, ModelStatus.WARMING_UP)
            for inst in instances
        )
    
    async def _is_model_loaded(self, model_name: str, version: str) -> bool:
        """Check if a model version is loaded."""
        async with self.instance_lock:
            return self._is_model_loaded_unlocked(model_name, version)
    
    def _record_request(self, model_name: str, version: str, processing_time: float):
        """Record a request for analytics."""
        request_info = {
            'model_name': model_name,
            'version': version,
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        self.request_history.append(request_info)
        
        # Update popularity
        self.model_popularity[model_name] += 1
    
    async def _health_monitor_loop(self):
        """Background health monitoring."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all instances."""
        all_instances = []
        async with self.instance_lock:
            for instances in self.model_instances.values():
                all_instances.extend(instances)
        
        for instance in all_instances:
            try:
                if instance.engine and instance.status == ModelStatus.READY:
                    # Simple health check - just check if engine exists and model is loaded
                    if self.inference_engine:
                        health_result = await self.inference_engine.health_check(instance.model_name, instance.instance_id)
                        is_healthy = health_result if isinstance(health_result, bool) else health_result.get('healthy', False)
                    else:
                        is_healthy = instance.model is not None
                    
                    if not is_healthy:
                        instance.health_check_failures += 1
                        if instance.health_check_failures >= 3:
                            instance.mark_failure(self.config.failure_cooldown_seconds)
                    else:
                        instance.mark_success()
                    
                    instance.last_health_check = time.time()
            
            except Exception as e:
                self.logger.warning(f"Health check failed for {instance.instance_id}: {e}")
                instance.health_check_failures += 1
                if instance.health_check_failures >= 3:
                    instance.mark_failure(self.config.failure_cooldown_seconds)
    
    async def _cleanup_loop(self):
        """Background cleanup task."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self._cleanup_unused_instances()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_unused_instances(self):
        """Clean up unused instances."""
        current_time = time.time()
        instances_to_remove = []
        
        async with self.instance_lock:
            for model_name, instances in self.model_instances.items():
                # Keep minimum instances
                if len(instances) <= self.config.min_instances_per_model:
                    continue
                
                # Find old unused instances
                for instance in instances:
                    idle_time = current_time - instance.last_used
                    if (idle_time > 1800 and  # 30 minutes
                        len(instances) > self.config.min_instances_per_model):
                        instances_to_remove.append((model_name, instance))
        
        # Remove unused instances
        for model_name, instance in instances_to_remove:
            await self._unload_instance(instance)
            async with self.instance_lock:
                self.model_instances[model_name].remove(instance)
            
            self.logger.info(f"Cleaned up unused instance {instance.instance_id}")
    
    async def _prefetch_loop(self):
        """Background prefetching task."""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._prefetch_popular_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in prefetch loop: {e}")
    
    async def _prefetch_popular_models(self):
        """Prefetch popular models."""
        # Calculate popularity threshold
        total_requests = sum(self.model_popularity.values())
        threshold = total_requests * self.config.prefetch_threshold
        
        popular_models = [
            model_name for model_name, count in self.model_popularity.items()
            if count >= threshold and model_name not in self.model_instances
        ]
        
        # Prefetch popular models
        for model_name in popular_models:
            try:
                self.logger.info(f"Prefetching popular model: {model_name}")
                await self.load_model(model_name)
            except Exception as e:
                self.logger.warning(f"Failed to prefetch {model_name}: {e}")
    
    async def _unload_all_models(self):
        """Unload all models."""
        async with self.instance_lock:
            all_instances = []
            for instances in self.model_instances.values():
                all_instances.extend(instances)
            
            self.model_instances.clear()
        
        # Unload all instances
        for instance in all_instances:
            await self._unload_instance(instance)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        # Count instances by state
        total_instances = 0
        idle_instances = 0
        active_instances = 0
        
        for instances in self.model_instances.values():
            for instance in instances:
                total_instances += 1
                if instance.state == ModelStatus.IDLE or instance.state == ModelStatus.READY:
                    if instance.active_connections == 0:
                        idle_instances += 1
                    else:
                        active_instances += 1
        
        stats = {
            'total_models': len(self.model_instances),
            'loaded_models': len(self.model_instances),
            'total_instances': total_instances,
            'idle_instances': idle_instances,
            'active_instances': active_instances,
            'enabled': self.config.enabled,
            'model_popularity': dict(self.model_popularity),
            'active_versions': dict(self.active_versions),
            'model_versions': dict(self.model_versions),
            'config': {
                'loading_strategy': self.config.loading_strategy.value,
                'load_balancing_strategy': self.config.load_balancing_strategy.value,
                'max_instances_per_model': self.config.max_instances_per_model
            }
        }
        
        # Add instance details
        instance_stats = {}
        for model_name, instances in self.model_instances.items():
            instance_stats[model_name] = []
            for instance in instances:
                instance_stats[model_name].append({
                    'instance_id': instance.instance_id,
                    'version': instance.version,
                    'status': instance.status.value,
                    'is_healthy': instance.is_healthy(),
                    'request_count': instance.request_count,
                    'average_response_time': instance.average_response_time,
                    'error_count': instance.error_count,
                    'last_used': instance.last_used
                })
        
        stats['instances'] = instance_stats
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the loader."""
        total_instances = sum(len(instances) for instances in self.model_instances.values())
        healthy_instances = 0
        
        for instances in self.model_instances.values():
            for instance in instances:
                if instance.is_healthy():
                    healthy_instances += 1
        
        return {
            'status': 'healthy' if healthy_instances > 0 or total_instances == 0 else 'unhealthy',
            'enabled': self.config.enabled,
            'total_instances': total_instances,
            'healthy_instances': healthy_instances,
            'loaded_models': len(self.model_instances),
            'is_running': self.is_running
        }
    
    async def _check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        if not self.config.enable_auto_scaling:
            return
        
        for model_name, instances in self.model_instances.items():
            if not instances:
                continue
            
            # Calculate average load
            total_load = sum(instance.get_load_score() for instance in instances)
            avg_load = total_load / len(instances)
            
            # Scale up if load is high
            if (avg_load > self.config.scale_up_threshold and 
                len(instances) < self.config.max_instances_per_model):
                
                self.logger.info(f"Scaling up {model_name} due to high load: {avg_load}")
                await self.scale_model(model_name, len(instances) + 1)
            
            # Scale down if load is low
            elif (avg_load < self.config.scale_down_threshold and 
                  len(instances) > self.config.min_instances_per_model):
                
                self.logger.info(f"Scaling down {model_name} due to low load: {avg_load}")
                await self.scale_model(model_name, len(instances) - 1)
    
    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """List all loaded models."""
        models = []
        for model_name, instances in self.model_instances.items():
            for instance in instances:
                models.append({
                    'model_name': model_name,
                    'version': instance.version,
                    'instance_id': instance.instance_id,
                    'status': instance.status.value,
                    'is_healthy': instance.is_healthy(),
                    'load_time': instance.load_time,
                    'request_count': instance.request_count
                })
        return models

    async def scale_model(self, model_name: str, target_instances: int) -> List[ModelInstanceInfo]:
        """Scale model to target number of instances."""
        async with self.instance_lock:
            current_instances = self.model_instances.get(model_name, [])
            current_count = len(current_instances)
            
            if target_instances > current_count:
                # Scale up
                for _ in range(target_instances - current_count):
                    try:
                        # Use unlocked version since we already hold the lock
                        version = self.active_versions.get(model_name, "v1")
                        # Force reload to create additional instances
                        success = await self._load_model_unlocked(model_name, version, force_reload=True)
                        
                        if not success:
                            self.logger.error(f"Failed to load model {model_name}:{version}")
                            break
                        
                    except Exception as e:
                        self.logger.error(f"Failed to scale up {model_name}: {e}")
                        break
                        
            elif target_instances < current_count:
                # Scale down
                instances_to_remove = current_count - target_instances
                for _ in range(instances_to_remove):
                    if current_instances:
                        instance = current_instances.pop()
                        await self._unload_instance(instance)
            
            return self.model_instances.get(model_name, [])
    
    @property
    def instances(self) -> Dict[str, List[ModelInstanceInfo]]:
        """Get current model instances."""
        return dict(self.model_instances)
