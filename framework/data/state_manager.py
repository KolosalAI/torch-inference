"""
Advanced Data & State Management System

Provides:
- Model versioning and rollback capabilities
- Model registry integration
- Intelligent caching strategies
- State persistence and recovery
- Data lineage tracking
"""

import asyncio
import logging
import json
import hashlib
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import threading
from collections import defaultdict, OrderedDict
import tempfile
import shutil
import sqlite3
import weakref

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model status in registry."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    STAGING = "staging"
    FAILED = "failed"


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive strategy based on usage patterns


@dataclass
class CachePolicy:
    """Cache policy configuration."""
    strategy: CacheStrategy
    max_size_mb: int = 1024
    ttl_seconds: int = 3600
    max_items: int = 1000
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate cache policy."""
        if self.max_size_mb <= 0:
            raise ValueError("max_size_mb must be positive")
        if self.strategy == CacheStrategy.TTL and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive for TTL strategy")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert Enum to its value string
        data["strategy"] = self.strategy.value if isinstance(self.strategy, Enum) else str(self.strategy)
        return data


@dataclass
class ModelVersion:
    """Model version information."""
    model_id: str
    version: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ModelStatus = ModelStatus.STAGING
    checksum: str = ""
    size_bytes: int = 0
    
    def __post_init__(self):
        """Validate model version fields."""
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.version:
            raise ValueError("version cannot be empty")
        if not self.path:
            raise ValueError("path cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime to ISO format string
        if isinstance(data.get('created_at'), datetime):
            data['created_at'] = data['created_at'].isoformat()
        elif hasattr(self.created_at, 'isoformat'):
            data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        # Handle datetime conversion
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def _parse_version(self, version: str) -> List[int]:
        """Parse version string into comparable integers."""
        try:
            return [int(part) for part in version.split('.')]
        except ValueError:
            return [0]
    
    def compare_version(self, other: 'ModelVersion') -> int:
        """Compare versions. Returns -1 if self < other, 0 if equal, 1 if self > other."""
        self_parts = self._parse_version(self.version)
        other_parts = other._parse_version(other.version)
        
        # Pad shorter version with zeros
        max_len = max(len(self_parts), len(other_parts))
        self_parts = self_parts + [0] * (max_len - len(self_parts))
        other_parts = other_parts + [0] * (max_len - len(other_parts))
        
        if self_parts < other_parts:
            return -1
        elif self_parts > other_parts:
            return 1
        else:
            return 0
    
    def is_compatible(self, required_version: str) -> bool:
        """Check if this version is compatible with required version."""
        self_parts = self._parse_version(self.version)
        required_parts = self._parse_version(required_version)
        # Major version must match
        if len(self_parts) >= 1 and len(required_parts) >= 1:
            if self_parts[0] != required_parts[0]:
                return False
        # Compatible if the required version is greater than or equal to this version
        # (i.e., newer runtime expecting same-major works with this version)
        return ModelVersion("req", required_version, "-").compare_version(self) >= 0
    
    def __lt__(self, other: 'ModelVersion') -> bool:
        """Compare versions for sorting."""
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return self._parse_version(self.version) < self._parse_version(other.version)


@dataclass
class ModelMetadata:
    """Model metadata information."""
    name: str
    version: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    framework: str = "pytorch"
    architecture: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: ModelStatus = ModelStatus.STAGING
    size_bytes: int = 0
    checksum: str = ""
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    author: str = ""
    license: str = ""


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl


@dataclass
class DataLineage:
    """Data lineage tracking information."""
    operation_id: str
    operation_type: str
    input_data_id: str
    output_data_id: str
    model_name: str
    model_version: str
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelStorage(ABC):
    """Abstract base class for model storage backends."""
    
    @abstractmethod
    async def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Save model and return storage path."""
        pass
    
    @abstractmethod
    async def load_model(self, name: str, version: str) -> Tuple[Any, ModelMetadata]:
        """Load model and metadata."""
        pass
    
    @abstractmethod
    async def delete_model(self, name: str, version: str) -> bool:
        """Delete model."""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[ModelMetadata]:
        """List all models."""
        pass


class FileSystemModelStorage(ModelStorage):
    """File system-based model storage."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"File system model storage initialized: {self.base_path}")
    
    async def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Save model to file system."""
        model_dir = self.base_path / metadata.name / metadata.version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        metadata_path = model_dir / "metadata.json"
        
        # Calculate checksum
        model_data = pickle.dumps(model)
        metadata.checksum = hashlib.sha256(model_data).hexdigest()
        metadata.size_bytes = len(model_data)
        metadata.updated_at = datetime.utcnow()
        
        # Save files
        with open(model_path, 'wb') as f:
            f.write(model_data)
        
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, default=str, indent=2)
        
        logger.info(f"Saved model {metadata.name} v{metadata.version} to {model_dir}")
        return str(model_path)
    
    async def load_model(self, name: str, version: str) -> Tuple[Any, ModelMetadata]:
        """Load model from file system."""
        model_dir = self.base_path / name / version
        model_path = model_dir / "model.pkl"
        metadata_path = model_dir / "metadata.json"
        
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Model {name} v{version} not found")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Convert string dates back to datetime
        for date_field in ['created_at', 'updated_at']:
            if date_field in metadata_dict:
                metadata_dict[date_field] = datetime.fromisoformat(metadata_dict[date_field])
        
        metadata = ModelMetadata(**metadata_dict)
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model {name} v{version}")
        return model, metadata
    
    async def delete_model(self, name: str, version: str) -> bool:
        """Delete model from file system."""
        model_dir = self.base_path / name / version
        
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model {name} v{version}")
            return True
        
        return False
    
    async def list_models(self) -> List[ModelMetadata]:
        """List all models in storage."""
        models = []
        
        for model_name_dir in self.base_path.iterdir():
            if not model_name_dir.is_dir():
                continue
            
            for version_dir in model_name_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata_dict = json.load(f)
                        
                        # Convert string dates back to datetime
                        for date_field in ['created_at', 'updated_at']:
                            if date_field in metadata_dict:
                                metadata_dict[date_field] = datetime.fromisoformat(metadata_dict[date_field])
                        
                        models.append(ModelMetadata(**metadata_dict))
                    
                    except Exception as e:
                        logger.error(f"Error loading metadata from {metadata_path}: {e}")
        
        return models


class IntelligentCache:
    """Intelligent caching system with multiple eviction policies."""
    
    def __init__(self, cache_policy: CachePolicy = None):
        if cache_policy is None:
            cache_policy = CachePolicy(strategy=CacheStrategy.LRU)
        
        self.policy = cache_policy
        self.max_size_bytes = cache_policy.max_size_mb * 1024 * 1024  # Convert MB to bytes
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._current_size = 0
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.RLock()
        # Track access times for test visibility
        self._access_times: Dict[str, float] = {}
        
        # Cleanup task for TTL entries
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"Intelligent cache initialized: {self.policy.strategy.value}, max size: {self.max_size_bytes} bytes")
    
    @property
    def _current_size_mb(self) -> float:
        """Get current size in MB for compatibility."""
        return self._current_size / (1024 * 1024)
    
    async def start(self):
        """Start cache maintenance tasks."""
        if self._cleanup_task:
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_entries())
        logger.info("Cache maintenance started")
    
    async def stop(self):
        """Stop cache maintenance tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        with self._lock:
            self._cache.clear()
            self._current_size = 0
        
        logger.info("Cache maintenance stopped")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._miss_count += 1
                return None
            
            # Check TTL expiration
            if entry.is_expired():
                self._remove_entry(key)
                self._miss_count += 1
                return None
            
            # Update access statistics
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            self._hit_count += 1
            self._access_times[key] = time.time()
            
            # Move to end for LRU
            if self.policy.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)
            
            return entry.value
    
    async def put(self, key: str, value: Any, size_mb: Optional[float] = None, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        # Calculate size
        try:
            if size_mb is not None:
                size = int(size_mb * 1024 * 1024)  # Convert MB to bytes
            else:
                size = len(pickle.dumps(value))
        except Exception:
            # Fallback size estimation
            if size_mb is not None:
                size = int(size_mb * 1024 * 1024)
            else:
                size = len(str(value))
        
        # Set TTL based on policy if not provided
        if ttl is None and self.policy.strategy == CacheStrategy.TTL:
            ttl = float(self.policy.ttl_seconds)
        
        with self._lock:
            # Check if value is too large
            if size > self.max_size_bytes:
                logger.warning(f"Value too large for cache: {size} bytes")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Make space if needed (size and item count)
            while (self._current_size + size > self.max_size_bytes or len(self._cache) >= self.policy.max_items) and self._cache:
                self._evict_entry()
            
            # Add new entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl=ttl
            )
            
            self._cache[key] = entry
            self._current_size += size
            self._access_times[key] = time.time()
            
            logger.debug(f"Cached entry: {key} ({size} bytes)")
            return True
    
    async def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                return False
            return True
    
    async def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def _remove_entry(self, key: str):
        """Remove entry and update size."""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size -= entry.size_bytes
            self._access_times.pop(key, None)
    
    def _evict_entry(self):
        """Evict an entry based on the eviction policy."""
        if not self._cache:
            return
        
        if self.policy.strategy == CacheStrategy.LRU:
            # Remove least recently used (first in OrderedDict)
            key = next(iter(self._cache))
            self._remove_entry(key)
        
        elif self.policy.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            self._remove_entry(key)
        
        elif self.policy.strategy == CacheStrategy.FIFO:
            # Remove first inserted
            key = next(iter(self._cache))
            self._remove_entry(key)
        
        elif self.policy.strategy == CacheStrategy.TTL:
            # Remove expired entries first, then oldest
            now = datetime.utcnow()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            if expired_keys:
                self._remove_entry(expired_keys[0])
            else:
                # Fallback to oldest entry
                key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
                self._remove_entry(key)
        
        elif self.policy.strategy == CacheStrategy.ADAPTIVE:
            # Evict based on a combination of frequency (low first) and recency (old first)
            def score(k: str) -> Tuple[int, float]:
                e = self._cache[k]
                # lower access_count and older last_accessed should be evicted first
                return (e.access_count, e.last_accessed.timestamp())
            key = min(self._cache.keys(), key=score)
            self._remove_entry(key)
    
    async def _cleanup_expired_entries(self):
        """Cleanup expired TTL entries."""
        try:
            while True:
                await asyncio.sleep(60.0)  # Check every minute
                
                with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        except asyncio.CancelledError:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total_requests) if total_requests > 0 else 0.0
            miss_rate = (self._miss_count / total_requests) if total_requests > 0 else 0.0
            
            # Handle policy properly
            policy_value = self.policy.strategy.value if hasattr(self.policy, 'strategy') else str(self.policy)
            
            return {
                'policy': policy_value,
                'size_mb': self._current_size / (1024 * 1024),
                'max_size_mb': self.policy.max_size_mb,
                'item_count': len(self._cache),
                'max_items': self.policy.max_items,
                'hits': self._hit_count,
                'misses': self._miss_count,
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
            }
    
    async def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
            self._access_times.clear()
            logger.info("Cache cleared")


class ModelRegistry:
    """Model registry for version management and metadata tracking."""
    
    def __init__(self, storage: ModelStorage, db_path: Optional[str] = None):
        self.storage = storage
        
        # Initialize SQLite database for metadata
        if db_path is None:
            db_path = tempfile.mktemp(suffix='.db')
        self.db_path = db_path
        self._init_database()
        
        # In-memory cache for active models
        self._model_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        
        # In-memory structures expected by tests
        self._models: Dict[str, ModelVersion] = {}
        self._versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        
        logger.info(f"Model registry initialized with storage: {type(storage).__name__}")
    
    def _init_database(self):
        """Initialize the metadata database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,  -- JSON encoded
                    framework TEXT,
                    architecture TEXT,
                    input_schema TEXT,  -- JSON encoded
                    output_schema TEXT,  -- JSON encoded
                    created_at TEXT,
                    updated_at TEXT,
                    status TEXT,
                    size_bytes INTEGER,
                    checksum TEXT,
                    performance_metrics TEXT,  -- JSON encoded
                    dependencies TEXT,  -- JSON encoded
                    author TEXT,
                    license TEXT,
                    PRIMARY KEY (name, version)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    deployed_at TEXT,
                    environment TEXT,
                    config TEXT,  -- JSON encoded
                    status TEXT,
                    PRIMARY KEY (name, version, environment)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_status ON models(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at)
            """)
    
    async def register_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Register a model in the registry."""
        # Save model to storage
        storage_path = await self.storage.save_model(model, metadata)
        
        # Save metadata to database
        self._save_metadata_to_db(metadata)
        
        # Cache the model
        with self._cache_lock:
            cache_key = f"{metadata.name}:{metadata.version}"
            self._model_cache[cache_key] = model
        
        logger.info(f"Registered model {metadata.name} v{metadata.version}")
        return storage_path
    
    def _save_metadata_to_db(self, metadata: ModelMetadata):
        """Save model metadata to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models (
                    name, version, description, tags, framework, architecture,
                    input_schema, output_schema, created_at, updated_at, status,
                    size_bytes, checksum, performance_metrics, dependencies,
                    author, license
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.name, metadata.version, metadata.description,
                json.dumps(metadata.tags), metadata.framework, metadata.architecture,
                json.dumps(metadata.input_schema), json.dumps(metadata.output_schema),
                metadata.created_at.isoformat(), metadata.updated_at.isoformat(),
                metadata.status.value, metadata.size_bytes, metadata.checksum,
                json.dumps(metadata.performance_metrics), json.dumps(metadata.dependencies),
                metadata.author, metadata.license
            ))
    
    async def get_model(self, name: str, version: str = "latest") -> Tuple[Any, ModelMetadata]:
        """Get a model from the registry."""
        # Handle "latest" version
        if version == "latest":
            version = await self._get_latest_version(name)
            if not version:
                raise ValueError(f"No versions found for model {name}")
        
        # Check cache first
        with self._cache_lock:
            cache_key = f"{name}:{version}"
            if cache_key in self._model_cache:
                metadata = self._get_metadata_from_db(name, version)
                if metadata:
                    return self._model_cache[cache_key], metadata
        
        # Load from storage
        model, metadata = await self.storage.load_model(name, version)
        
        # Cache the model
        with self._cache_lock:
            self._model_cache[cache_key] = model
        
        return model, metadata
    
    def _get_metadata_from_db(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Get model metadata from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM models WHERE name = ? AND version = ?",
                (name, version)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            
            # Convert JSON fields
            json_fields = ['tags', 'input_schema', 'output_schema', 'performance_metrics', 'dependencies']
            for field in json_fields:
                if data[field]:
                    data[field] = json.loads(data[field])
                else:
                    data[field] = {}
            
            # Convert dates
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            data['status'] = ModelStatus(data['status'])
            
            return ModelMetadata(**data)
    
    async def _get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version of a model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT version FROM models WHERE name = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
                (name, ModelStatus.ACTIVE.value)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    
    async def list_models(self, status: Optional[ModelStatus] = None) -> List[str]:
        """List models in the registry (names only)."""
        # Ignore status for in-memory simplified implementation
        return sorted(list(self._versions.keys()))
    
    async def update_model_status(self, name: str, version: str, status: ModelStatus) -> bool:
        """Update model status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE models SET status = ?, updated_at = ? WHERE name = ? AND version = ?",
                (status.value, datetime.utcnow().isoformat(), name, version)
            )
            
            if cursor.rowcount > 0:
                logger.info(f"Updated model {name} v{version} status to {status.value}")
                return True
            
            return False
    
    async def rollback_model(self, name: str, target_version: str) -> bool:
        """Rollback model to a specific version."""
        # First, check if target version exists and is not failed
        metadata = self._get_metadata_from_db(name, target_version)
        if not metadata:
            raise ValueError(f"Target version {target_version} not found")
        
        if metadata.status == ModelStatus.FAILED:
            raise ValueError(f"Cannot rollback to failed version {target_version}")
        
        # Set current active versions to deprecated
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE models SET status = ?, updated_at = ? WHERE name = ? AND status = ?",
                (ModelStatus.DEPRECATED.value, datetime.utcnow().isoformat(), name, ModelStatus.ACTIVE.value)
            )
            
            # Set target version to active
            conn.execute(
                "UPDATE models SET status = ?, updated_at = ? WHERE name = ? AND version = ?",
                (ModelStatus.ACTIVE.value, datetime.utcnow().isoformat(), name, target_version)
            )
        
        # Clear cache for this model
        with self._cache_lock:
            keys_to_remove = [key for key in self._model_cache.keys() if key.startswith(f"{name}:")]
            for key in keys_to_remove:
                del self._model_cache[key]
        
        logger.info(f"Rolled back model {name} to version {target_version}")
        return True
    
    async def delete_model(self, name: str, version: str) -> bool:
        """Delete a model version."""
        # Delete from storage
        await self.storage.delete_model(name, version)
        
        # Delete from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM models WHERE name = ? AND version = ?",
                (name, version)
            )
            
            success = cursor.rowcount > 0
        
        # Remove from cache
        with self._cache_lock:
            cache_key = f"{name}:{version}"
            self._model_cache.pop(cache_key, None)
        
        if success:
            logger.info(f"Deleted model {name} v{version}")
        
        return success
    
    async def register_version(self, version: ModelVersion) -> None:
        """Register a model version."""
        # Prevent duplicates
        existing_versions = [v.version for v in self._versions.get(version.model_id, [])]
        if version.version in existing_versions:
            raise ValueError("Model version already exists")

        # Track in-memory
        self._versions[version.model_id].append(version)
        # Track latest pointer for convenience
        current_latest = self._models.get(version.model_id)
        if not current_latest or ModelVersion(version.model_id, version.version, version.path).compare_version(current_latest) > 0:
            self._models[version.model_id] = version

        # Persist minimal metadata to DB for durability
        md = ModelMetadata(
            name=version.model_id,
            version=version.version,
            description=version.metadata.get("description", f"Model {version.model_id} version {version.version}"),
            created_at=version.created_at,
            updated_at=version.created_at,
            status=version.status,
            size_bytes=version.size_bytes,
            checksum=version.checksum,
        )
        self._save_metadata_to_db(md)
    
    async def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        versions = self._versions.get(model_id, [])
        if not versions:
            return None
        # Sort using compare
        latest = max(versions, key=lambda v: v._parse_version(v.version))
        return latest
    
    async def get_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get a specific version of a model."""
        for v in self._versions.get(model_id, []):
            if v.version == version:
                return v
        return None
    
    async def list_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model."""
        return list(self._versions.get(model_id, []))
    
    async def list_all_models(self) -> List[str]:
        """List all model names in the registry."""
        return sorted(list(self._versions.keys()))
    
    async def remove_version(self, model_id: str, version: str) -> bool:
        """Remove a specific model version."""
        removed = False
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM models WHERE name = ? AND version = ?", (model_id, version))
            # Update in-memory
            versions = self._versions.get(model_id, [])
            new_versions = [v for v in versions if v.version != version]
            if len(new_versions) != len(versions):
                self._versions[model_id] = new_versions
                removed = True
                # Update latest pointer
                if new_versions:
                    self._models[model_id] = max(new_versions, key=lambda v: v._parse_version(v.version))
                else:
                    self._models.pop(model_id, None)
            # Remove from model cache
            with self._cache_lock:
                cache_key = f"{model_id}:{version}"
                self._model_cache.pop(cache_key, None)
            return removed
        except Exception as e:
            logger.error(f"Error removing model version {model_id}:{version}: {e}")
            return False

    async def remove_model(self, model_id: str) -> int:
        """Remove all versions of a model and return removed count."""
        count = len(self._versions.get(model_id, []))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM models WHERE name = ?", (model_id,))
            # Clear in-memory
            self._versions.pop(model_id, None)
            self._models.pop(model_id, None)
            return count
        except Exception as e:
            logger.error(f"Error removing model {model_id}: {e}")
            return 0
    
    async def search_by_metadata(self, criteria: Dict[str, Any]) -> List[ModelVersion]:
        """Search models by metadata criteria."""
        results: List[ModelVersion] = []
        # Filter using in-memory metadata stored in ModelVersion.metadata
        for model_id, vers in self._versions.items():
            for v in vers:
                match = True
                for key, cond in criteria.items():
                    if isinstance(cond, dict):
                        val = v.metadata.get(key)
                        if val is None:
                            match = False
                            break
                        if "$gte" in cond and not (val >= cond["$gte"]):
                            match = False
                            break
                        if "$lte" in cond and not (val <= cond["$lte"]):
                            match = False
                            break
                    else:
                        if v.metadata.get(key) != cond:
                            match = False
                            break
                if match:
                    results.append(v)
        return results
    
    async def save_to_disk(self) -> None:
        """Save registry state to disk (persist in-memory versions)."""
        data = {}
        for model_id, versions in self._versions.items():
            data[model_id] = [
                {
                    "model_id": v.model_id,
                    "version": v.version,
                    "path": v.path,
                    "metadata": v.metadata,
                    "created_at": v.created_at.isoformat(),
                    "status": v.status.value,
                    "checksum": v.checksum,
                    "size_bytes": v.size_bytes,
                }
                for v in versions
            ]
        json_path = str(Path(self.storage.base_path) / "registry_versions.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    
    async def load_from_disk(self) -> None:
        """Load registry state from disk (restore in-memory versions)."""
        json_path = str(Path(self.storage.base_path) / "registry_versions.json")
        if not Path(json_path).exists():
            return
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._versions.clear()
        self._models.clear()
        for model_id, versions in data.items():
            lst: List[ModelVersion] = []
            for vd in versions:
                mv = ModelVersion(
                    model_id=vd["model_id"],
                    version=vd["version"],
                    path=vd["path"],
                    metadata=vd.get("metadata", {}),
                    created_at=datetime.fromisoformat(vd["created_at"]),
                    status=ModelStatus(vd.get("status", ModelStatus.STAGING.value)),
                    checksum=vd.get("checksum", ""),
                    size_bytes=vd.get("size_bytes", 0),
                )
                lst.append(mv)
            self._versions[model_id] = lst
            if lst:
                self._models[model_id] = max(lst, key=lambda v: v._parse_version(v.version))


class DataStateManager:
    """
    Comprehensive data and state management system.
    
    Features:
    - Model versioning and rollback
    - Model registry integration
    - Intelligent caching strategies
    - State persistence and recovery
    - Data lineage tracking
    """
    
    def __init__(self, storage_path: str = "models", cache_size_mb: int = 1024, cache_policy: Optional[CachePolicy] = None):
        # Initialize components
        self.storage = FileSystemModelStorage(storage_path)
        self.registry = ModelRegistry(self.storage)
        if cache_policy is None:
            cache_policy = CachePolicy(
                strategy=CacheStrategy.LRU,
                max_size_mb=cache_size_mb,
                ttl_seconds=3600,
                max_items=1000
            )
        self.cache = IntelligentCache(cache_policy=cache_policy)
        
        # Data lineage tracking
        self._lineage_db_path = Path(storage_path) / "lineage.db"
        self._init_lineage_database()
        
        # State management
        self._state: Dict[str, Any] = {}
        self._state_lock = threading.RLock()
        
        logger.info("Data state manager initialized")
    
    def _init_lineage_database(self):
        """Initialize the data lineage database."""
        with sqlite3.connect(self._lineage_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage (
                    operation_id TEXT PRIMARY KEY,
                    operation_type TEXT NOT NULL,
                    input_data_id TEXT,
                    output_data_id TEXT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    parameters TEXT,  -- JSON encoded
                    metadata TEXT     -- JSON encoded
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_model ON lineage(model_name, model_version)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_timestamp ON lineage(timestamp)
            """)
    
    async def start(self):
        """Start all components."""
        await self.cache.start()
        logger.info("Data state manager started")
    
    async def stop(self):
        """Stop all components."""
        await self.cache.stop()
        logger.info("Data state manager stopped")
    
    async def register_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Register a new model."""
        return await self.registry.register_model(model, metadata)
    
    async def get_model(self, name: str, version: str = "latest") -> Tuple[Any, ModelMetadata]:
        """Get a model with caching."""
        cache_key = f"model:{name}:{version}"
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Load from registry
        model, metadata = await self.registry.get_model(name, version)
        
        # Cache the result
        await self.cache.put(cache_key, (model, metadata), ttl=3600.0)  # Cache for 1 hour
        
        return model, metadata
    
    async def rollback_model(self, name: str, target_version: str) -> bool:
        """Rollback a model to a specific version."""
        # Clear related cache entries
        cache_keys_to_clear = [
            f"model:{name}:latest",
            f"model:{name}:{target_version}"
        ]
        
        for key in cache_keys_to_clear:
            await self.cache.remove(key)
        
        return await self.registry.rollback_model(name, target_version)
    
    def record_lineage(self, lineage: DataLineage):
        """Record data lineage."""
        with sqlite3.connect(self._lineage_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO lineage (
                    operation_id, operation_type, input_data_id, output_data_id,
                    model_name, model_version, timestamp, parameters, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lineage.operation_id, lineage.operation_type, lineage.input_data_id,
                lineage.output_data_id, lineage.model_name, lineage.model_version,
                lineage.timestamp.isoformat(), json.dumps(lineage.parameters),
                json.dumps(lineage.metadata)
            ))
    
    def get_lineage(self, model_name: str, model_version: Optional[str] = None, 
                   limit: int = 100) -> List[DataLineage]:
        """Get data lineage for a model."""
        with sqlite3.connect(self._lineage_db_path) as conn:
            if model_version:
                cursor = conn.execute(
                    "SELECT * FROM lineage WHERE model_name = ? AND model_version = ? ORDER BY timestamp DESC LIMIT ?",
                    (model_name, model_version, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM lineage WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?",
                    (model_name, limit)
                )
            
            lineages = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                
                # Convert JSON fields
                data['parameters'] = json.loads(data['parameters']) if data['parameters'] else {}
                data['metadata'] = json.loads(data['metadata']) if data['metadata'] else {}
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                
                lineages.append(DataLineage(**data))
            
            return lineages
    
    def save_kv_state(self, key: str, value: Any):
        """Save application key/value state (internal helper)."""
        with self._state_lock:
            self._state[key] = value
        # Also cache it
        # Fire and forget; best-effort
        try:
            asyncio.create_task(self.cache.put(f"state:{key}", value, ttl=7200.0))
        except RuntimeError:
            # No running loop; ignore
            pass
        logger.debug(f"Saved state: {key}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get application state."""
        # Try cache first
        cache_key = f"state:{key}"
        # Note: synchronous path, so do not await; cache may not be available here
        cached_value = None
        if cached_value is not None:
            return cached_value
        
        # Try in-memory state
        with self._state_lock:
            value = self._state.get(key, default)
        
        if value is not default:
            try:
                asyncio.create_task(self.cache.put(cache_key, value, ttl=7200.0))
            except RuntimeError:
                pass
        
        return value
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'cache': self.cache.get_stats(),
            'registry': {
                'storage_type': type(self.storage).__name__,
                'cache_size': len(self.registry._model_cache)
            },
            'state': {
                'keys_count': len(self._state)
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def register_model_version(self, version: ModelVersion) -> None:
        """Register a model version."""
        await self.registry.register_version(version)
    
    async def cache_model_data(self, key: str, data: Any, size_mb: float) -> None:
        """Cache model data."""
        # Use cache policy TTL if it's a TTL strategy
        ttl = None
        if self.cache.policy.strategy == CacheStrategy.TTL:
            ttl = float(self.cache.policy.ttl_seconds)
        await self.cache.put(key, data, size_mb=size_mb, ttl=ttl)
    
    async def get_cached_model_data(self, key: str) -> Any:
        """Get cached model data."""
        return await self.cache.get(key)
    
    async def cleanup_expired_cache(self) -> int:
        """Cleanup expired cache entries and return count."""
        with self.cache._lock:
            expired_keys = [
                key for key, entry in self.cache._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self.cache._remove_entry(key)
            
            return len(expired_keys)
    
    def get_state_stats(self) -> Dict[str, Any]:
        """Get state management statistics."""
        # Aggregate totals from in-memory registry
        total_models = len(self.registry._versions)
        total_versions = sum(len(vs) for vs in self.registry._versions.values())
        cache_stats = self.cache.get_stats()
        return {
            'cache': cache_stats,
            'registry': {
                'total_models': total_models,
                'total_versions': total_versions,
            },
            'state': {
                'keys_count': len(self._state)
            }
        }

    # Convenience API expected by tests
    @property
    def model_registry(self) -> ModelRegistry:
        return self.registry

    async def get_model_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        return await self.registry.get_version(model_id, version)

    async def list_model_versions(self, model_id: str) -> List[ModelVersion]:
        return await self.registry.list_versions(model_id)

    async def list_all_models(self) -> List[str]:
        return await self.registry.list_all_models()

    async def get_latest_model_version(self, model_id: str) -> Optional[ModelVersion]:
        return await self.registry.get_latest_version(model_id)

    async def remove_model_version(self, model_id: str, version: str) -> bool:
        return await self.registry.remove_version(model_id, version)

    async def preload_model_to_cache(self, model_id: str, version: str) -> bool:
        try:
            data = await self._load_model_data(model_id, version)
            await self.cache.put(f"{model_id}:{version}", data, size_mb=1)
            return True
        except Exception as e:
            logger.error(f"Failed to preload {model_id}:{version} - {e}")
            return False

    async def warm_cache(self, model_ids: List[str]) -> int:
        count = 0
        for mid in model_ids:
            latest = await self.registry.get_latest_version(mid)
            if not latest:
                continue
            if await self.preload_model_to_cache(mid, latest.version):
                count += 1
        return count

    async def save_state(self) -> None:
        await self.registry.save_to_disk()

    async def load_state(self) -> None:
        await self.registry.load_from_disk()

    async def _load_model_data(self, model_id: str, version: str) -> Any:
        """Placeholder for model data loading (can be mocked in tests)."""
        return {"model": f"{model_id}_{version}"}


# Global data state manager
_data_state_manager: Optional[DataStateManager] = None


def get_data_state_manager() -> DataStateManager:
    """Get the global data state manager."""
    global _data_state_manager
    if _data_state_manager is None:
        _data_state_manager = DataStateManager()
    return _data_state_manager


# Alias for compatibility with tests
StateManager = DataStateManager
