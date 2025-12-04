"""
Distributed Caching System with Redis Cluster Support

Provides:
- Redis Cluster integration
- Distributed cache with automatic failover
- Cache partitioning and sharding
- Cache warming and invalidation strategies
- High availability caching
"""

import asyncio
import logging
import json
import pickle
import hashlib
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import threading
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Cache backend types."""
    MEMORY = "memory"
    REDIS = "redis"
    REDIS_CLUSTER = "redis_cluster"
    MEMCACHED = "memcached"


class SerializationFormat(Enum):
    """Serialization formats."""
    PICKLE = "pickle"
    JSON = "json"
    MSGPACK = "msgpack"


class CacheConsistency(Enum):
    """Cache consistency levels."""
    EVENTUAL = "eventual"
    STRONG = "strong"
    WEAK = "weak"


@dataclass
class CacheConfig:
    """Cache configuration."""
    backend: CacheBackend = CacheBackend.MEMORY
    serialization: SerializationFormat = SerializationFormat.PICKLE
    consistency: CacheConsistency = CacheConsistency.EVENTUAL
    default_ttl: float = 3600.0  # 1 hour
    max_key_size: int = 250
    max_value_size: int = 1024 * 1024 * 10  # 10MB
    compression_enabled: bool = True
    compression_threshold: int = 1024  # Compress values larger than 1KB


@dataclass
class CacheEntry:
    """Distributed cache entry."""
    key: str
    value: Any
    ttl: Optional[float]
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    node_id: Optional[str] = None


@dataclass
class CacheNode:
    """Cache node information."""
    node_id: str
    host: str
    port: int
    is_primary: bool = True
    is_healthy: bool = True
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    connection_pool: Optional[Any] = None


class CacheSerializer:
    """Handles serialization/deserialization for cache values."""
    
    def __init__(self, format_type: SerializationFormat, compression_enabled: bool = True):
        self.format_type = format_type
        self.compression_enabled = compression_enabled
        
        if compression_enabled:
            try:
                import zlib
                self.compressor = zlib
            except ImportError:
                logger.warning("zlib not available, disabling compression")
                self.compression_enabled = False
    
    def serialize(self, value: Any) -> bytes:
        """Serialize value to bytes."""
        if self.format_type == SerializationFormat.PICKLE:
            data = pickle.dumps(value)
        elif self.format_type == SerializationFormat.JSON:
            data = json.dumps(value, default=str).encode('utf-8')
        elif self.format_type == SerializationFormat.MSGPACK:
            try:
                import msgpack
                data = msgpack.packb(value, default=str)
            except ImportError:
                logger.warning("msgpack not available, falling back to pickle")
                data = pickle.dumps(value)
        else:
            raise ValueError(f"Unsupported serialization format: {self.format_type}")
        
        # Apply compression if enabled and data is large enough
        if self.compression_enabled and len(data) > 1024:
            compressed = self.compressor.compress(data)
            # Only use compressed version if it's actually smaller
            if len(compressed) < len(data):
                return b'COMPRESSED:' + compressed
        
        return data
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to value."""
        # Check if data is compressed
        if data.startswith(b'COMPRESSED:'):
            if self.compression_enabled:
                data = self.compressor.decompress(data[11:])  # Remove 'COMPRESSED:' prefix
            else:
                raise ValueError("Compressed data found but compression not enabled")
        
        if self.format_type == SerializationFormat.PICKLE:
            return pickle.loads(data)
        elif self.format_type == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        elif self.format_type == SerializationFormat.MSGPACK:
            try:
                import msgpack
                return msgpack.unpackb(data)
            except ImportError:
                return pickle.loads(data)
        else:
            raise ValueError(f"Unsupported serialization format: {self.format_type}")


class DistributedCache(ABC):
    """Abstract base class for distributed cache implementations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class RedisClusterCache(DistributedCache):
    """Redis Cluster distributed cache implementation."""
    
    def __init__(self, config: CacheConfig, cluster_nodes: List[Dict[str, Any]]):
        self.config = config
        self.cluster_nodes = cluster_nodes
        self.serializer = CacheSerializer(config.serialization, config.compression_enabled)
        
        # Redis cluster client
        self._cluster_client: Optional[Any] = None
        self._connection_pool: Optional[Any] = None
        
        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        self._error_count = 0
        self._lock = threading.RLock()
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info(f"Redis cluster cache initialized with {len(cluster_nodes)} nodes")
    
    async def start(self):
        """Initialize Redis cluster connection."""
        try:
            import redis.asyncio as redis
            from redis.asyncio import RedisCluster
            
            # Create cluster client
            self._cluster_client = RedisCluster(
                startup_nodes=self.cluster_nodes,
                decode_responses=False,  # We handle our own serialization
                skip_full_coverage_check=True,
                health_check_interval=30,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                max_connections_per_node=20
            )
            
            # Test connection
            await self._cluster_client.ping()
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._monitor_cluster_health())
            
            logger.info("Redis cluster cache started successfully")
            
        except ImportError:
            raise ImportError("redis package required for Redis cluster support")
        except Exception as e:
            logger.error(f"Failed to start Redis cluster cache: {e}")
            raise
    
    async def stop(self):
        """Stop Redis cluster connection."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._cluster_client:
            await self._cluster_client.close()
        
        logger.info("Redis cluster cache stopped")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cluster."""
        if not self._cluster_client:
            return None
        
        try:
            # Validate key
            if not self._validate_key(key):
                return None
            
            data = await self._cluster_client.get(key)
            if data is None:
                with self._lock:
                    self._miss_count += 1
                return None
            
            # Deserialize
            value = self.serializer.deserialize(data)
            
            with self._lock:
                self._hit_count += 1
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting key {key} from Redis cluster: {e}")
            with self._lock:
                self._error_count += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in Redis cluster."""
        if not self._cluster_client:
            return False
        
        try:
            # Validate key and value
            if not self._validate_key(key):
                return False
            
            # Serialize value
            serialized_data = self.serializer.serialize(value)
            
            # Check value size
            if len(serialized_data) > self.config.max_value_size:
                logger.warning(f"Value too large for key {key}: {len(serialized_data)} bytes")
                return False
            
            # Set TTL
            if ttl is None:
                ttl = self.config.default_ttl
            
            # Set in Redis
            if ttl > 0:
                await self._cluster_client.setex(key, int(ttl), serialized_data)
            else:
                await self._cluster_client.set(key, serialized_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting key {key} in Redis cluster: {e}")
            with self._lock:
                self._error_count += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cluster."""
        if not self._cluster_client:
            return False
        
        try:
            result = await self._cluster_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error deleting key {key} from Redis cluster: {e}")
            with self._lock:
                self._error_count += 1
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cluster."""
        if not self._cluster_client:
            return False
        
        try:
            result = await self._cluster_client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Error checking existence of key {key} in Redis cluster: {e}")
            with self._lock:
                self._error_count += 1
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries (use with caution)."""
        if not self._cluster_client:
            return False
        
        try:
            # This is potentially dangerous in a shared cluster
            # Consider implementing namespace-based clearing instead
            logger.warning("Clearing entire Redis cluster cache")
            await self._cluster_client.flushall()
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Redis cluster cache: {e}")
            with self._lock:
                self._error_count += 1
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                'backend': 'redis_cluster',
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'error_count': self._error_count,
                'hit_rate_percent': hit_rate,
                'total_requests': total_requests
            }
        
        # Add cluster-specific stats
        if self._cluster_client:
            try:
                cluster_info = await self._cluster_client.cluster_info()
                cluster_nodes = await self._cluster_client.cluster_nodes()
                
                stats.update({
                    'cluster_state': cluster_info.get('cluster_state'),
                    'cluster_slots_assigned': cluster_info.get('cluster_slots_assigned'),
                    'cluster_slots_ok': cluster_info.get('cluster_slots_ok'),
                    'cluster_known_nodes': cluster_info.get('cluster_known_nodes'),
                    'nodes_count': len(cluster_nodes),
                    'healthy_nodes': sum(1 for node in cluster_nodes.values() if node.get('flags', []).count('fail') == 0)
                })
            except Exception as e:
                logger.error(f"Error getting cluster stats: {e}")
        
        return stats
    
    def _validate_key(self, key: str) -> bool:
        """Validate cache key."""
        if not key or len(key) > self.config.max_key_size:
            return False
        
        # Check for invalid characters
        invalid_chars = [' ', '\n', '\r', '\t']
        if any(char in key for char in invalid_chars):
            return False
        
        return True
    
    async def _monitor_cluster_health(self):
        """Monitor Redis cluster health."""
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                try:
                    if self._cluster_client:
                        # Ping cluster
                        await self._cluster_client.ping()
                        
                        # Check cluster state
                        cluster_info = await self._cluster_client.cluster_info()
                        state = cluster_info.get('cluster_state')
                        
                        if state != 'ok':
                            logger.warning(f"Redis cluster state is not OK: {state}")
                        
                except Exception as e:
                    logger.error(f"Redis cluster health check failed: {e}")
                    with self._lock:
                        self._error_count += 1
        
        except asyncio.CancelledError:
            pass


class CacheWarmingStrategy:
    """Cache warming strategies for distributed cache."""
    
    def __init__(self, cache: DistributedCache):
        self.cache = cache
        self._warming_tasks: Dict[str, asyncio.Task] = {}
    
    async def warm_cache_proactive(self, keys_generator: Callable[[], List[str]], 
                                 value_loader: Callable[[str], Any], 
                                 ttl: Optional[float] = None):
        """Proactively warm cache with predicted keys."""
        try:
            keys = keys_generator()
            logger.info(f"Starting proactive cache warming for {len(keys)} keys")
            
            # Warm cache in batches to avoid overwhelming the system
            batch_size = 50
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                tasks = []
                
                for key in batch:
                    task = asyncio.create_task(self._warm_single_key(key, value_loader, ttl))
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            logger.info("Proactive cache warming completed")
            
        except Exception as e:
            logger.error(f"Error in proactive cache warming: {e}")
    
    async def warm_cache_reactive(self, key: str, value_loader: Callable[[str], Any], 
                                ttl: Optional[float] = None) -> Any:
        """Reactively warm cache on cache miss."""
        # Check if key is already being warmed
        if key in self._warming_tasks:
            try:
                return await self._warming_tasks[key]
            except Exception:
                pass
        
        # Start warming task
        warming_task = asyncio.create_task(self._warm_single_key(key, value_loader, ttl))
        self._warming_tasks[key] = warming_task
        
        try:
            value = await warming_task
            return value
        finally:
            self._warming_tasks.pop(key, None)
    
    async def _warm_single_key(self, key: str, value_loader: Callable[[str], Any], 
                             ttl: Optional[float] = None) -> Any:
        """Warm a single cache key."""
        try:
            # Check if already cached
            existing_value = await self.cache.get(key)
            if existing_value is not None:
                return existing_value
            
            # Load value
            if asyncio.iscoroutinefunction(value_loader):
                value = await value_loader(key)
            else:
                value = value_loader(key)
            
            # Cache the value
            await self.cache.set(key, value, ttl)
            
            logger.debug(f"Warmed cache key: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Error warming cache key {key}: {e}")
            raise


class DistributedCacheManager:
    """
    Distributed caching system with Redis Cluster support.
    
    Features:
    - Redis Cluster integration with automatic failover
    - Cache warming strategies (proactive and reactive)
    - Serialization with compression
    - Health monitoring and statistics
    - Cache invalidation patterns
    """
    
    def __init__(self, config: CacheConfig, cluster_nodes: Optional[List[Dict[str, Any]]] = None):
        self.config = config
        
        # Initialize cache backend
        if config.backend == CacheBackend.REDIS_CLUSTER:
            if not cluster_nodes:
                raise ValueError("Redis cluster nodes required for Redis cluster backend")
            self.cache = RedisClusterCache(config, cluster_nodes)
        else:
            raise ValueError(f"Unsupported cache backend: {config.backend}")
        
        # Cache warming
        self.warming_strategy = CacheWarmingStrategy(self.cache)
        
        # Invalidation patterns
        self._invalidation_patterns: Dict[str, List[str]] = defaultdict(list)
        
        logger.info(f"Distributed cache manager initialized with backend: {config.backend.value}")
    
    async def start(self):
        """Start the distributed cache."""
        await self.cache.start()
        logger.info("Distributed cache manager started")
    
    async def stop(self):
        """Stop the distributed cache."""
        await self.cache.stop()
        logger.info("Distributed cache manager stopped")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value with optional default."""
        value = await self.cache.get(key)
        return value if value is not None else default
    
    async def get_or_set(self, key: str, value_loader: Callable, ttl: Optional[float] = None) -> Any:
        """Get value or set it using the loader function."""
        value = await self.cache.get(key)
        if value is not None:
            return value
        
        # Use reactive warming
        return await self.warming_strategy.warm_cache_reactive(key, value_loader, ttl)
    
    async def set_with_pattern(self, key: str, value: Any, ttl: Optional[float] = None, 
                             invalidation_pattern: Optional[str] = None) -> bool:
        """Set value with optional invalidation pattern."""
        result = await self.cache.set(key, value, ttl)
        
        if result and invalidation_pattern:
            self._invalidation_patterns[invalidation_pattern].append(key)
        
        return result
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        keys = self._invalidation_patterns.get(pattern, [])
        
        if not keys:
            return 0
        
        invalidated_count = 0
        for key in keys:
            if await self.cache.delete(key):
                invalidated_count += 1
        
        # Clear the pattern
        self._invalidation_patterns[pattern] = []
        
        logger.info(f"Invalidated {invalidated_count} keys for pattern: {pattern}")
        return invalidated_count
    
    async def warm_cache_batch(self, key_value_pairs: List[Tuple[str, Any]], 
                             ttl: Optional[float] = None) -> int:
        """Warm cache with batch of key-value pairs."""
        success_count = 0
        
        for key, value in key_value_pairs:
            if await self.cache.set(key, value, ttl):
                success_count += 1
        
        logger.info(f"Warmed {success_count}/{len(key_value_pairs)} cache entries")
        return success_count
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        cache_stats = await self.cache.get_stats()
        
        stats = {
            'cache_stats': cache_stats,
            'config': {
                'backend': self.config.backend.value,
                'serialization': self.config.serialization.value,
                'consistency': self.config.consistency.value,
                'default_ttl': self.config.default_ttl,
                'compression_enabled': self.config.compression_enabled
            },
            'invalidation_patterns': {
                pattern: len(keys) for pattern, keys in self._invalidation_patterns.items()
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return stats


# Global distributed cache manager
_distributed_cache_manager: Optional[DistributedCacheManager] = None


def get_distributed_cache_manager() -> DistributedCacheManager:
    """Get the global distributed cache manager."""
    global _distributed_cache_manager
    if _distributed_cache_manager is None:
        # Default configuration for Redis cluster
        config = CacheConfig(
            backend=CacheBackend.REDIS_CLUSTER,
            serialization=SerializationFormat.PICKLE,
            consistency=CacheConsistency.EVENTUAL,
            default_ttl=3600.0,
            compression_enabled=True
        )
        
        # Default cluster nodes (should be configured via environment)
        cluster_nodes = [
            {"host": "localhost", "port": 7000},
            {"host": "localhost", "port": 7001},
            {"host": "localhost", "port": 7002},
        ]
        
        _distributed_cache_manager = DistributedCacheManager(config, cluster_nodes)
    
    return _distributed_cache_manager


def create_cache_config(backend: CacheBackend = CacheBackend.REDIS_CLUSTER,
                       cluster_nodes: Optional[List[Dict[str, Any]]] = None,
                       **kwargs) -> Tuple[CacheConfig, List[Dict[str, Any]]]:
    """Create cache configuration with sensible defaults."""
    config = CacheConfig(backend=backend, **kwargs)
    
    if cluster_nodes is None:
        cluster_nodes = [
            {"host": "localhost", "port": 7000},
            {"host": "localhost", "port": 7001}, 
            {"host": "localhost", "port": 7002},
        ]
    
    return config, cluster_nodes
