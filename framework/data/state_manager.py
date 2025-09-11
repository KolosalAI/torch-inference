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


@dataclass
class CachePolicy:
    """Cache policy configuration."""
    strategy: CacheStrategy
    max_size_mb: int = 1024
    ttl_seconds: int = 3600
    max_items: int = 1000
    
    def __post_init__(self):
        """Validate cache policy."""
        if self.max_size_mb <= 0:
            raise ValueError("max_size_mb must be positive")
        if self.strategy == CacheStrategy.TTL and self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive for TTL strategy")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelVersion:
    """Model version information."""
    model_id: str
    version: str
    path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ModelStatus = ModelStatus.STAGING
    
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
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        # Handle datetime conversion
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def __lt__(self, other: 'ModelVersion') -> bool:
        """Compare versions for sorting."""
        if not isinstance(other, ModelVersion):
            return NotImplemented
        return self._parse_version(self.version) < self._parse_version(other.version)
    
    def _parse_version(self, version_str: str) -> Tuple[int, ...]:
        """Parse version string for comparison."""
        try:
            return tuple(map(int, version_str.split('.')))
        except ValueError:
            # Fallback to string comparison
            return (0,)


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
        
        # Cleanup task for TTL entries
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"Intelligent cache initialized: {self.policy.strategy.value}, max size: {self.max_size_bytes} bytes")
    
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
    
    def get(self, key: str) -> Optional[Any]:
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
            
            # Move to end for LRU
            if self.policy.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        # Calculate size
        try:
            size = len(pickle.dumps(value))
        except Exception:
            # Fallback size estimation
            size = len(str(value))
        
        with self._lock:
            # Check if value is too large
            if size > self.max_size_bytes:
                logger.warning(f"Value too large for cache: {size} bytes")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Make space if needed
            while self._current_size + size > self.max_size_bytes and self._cache:
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
            
            logger.debug(f"Cached entry: {key} ({size} bytes)")
            return True
    
    def remove(self, key: str) -> bool:
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
            hit_rate = (self._hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'policy': self.policy.value,
                'max_size_bytes': self.max_size_bytes,
                'current_size_bytes': self._current_size,
                'utilization_percent': (self._current_size / self.max_size_bytes) * 100,
                'entry_count': len(self._cache),
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'hit_rate_percent': hit_rate
            }
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_size = 0
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
    
    async def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List models in the registry."""
        with sqlite3.connect(self.db_path) as conn:
            if status:
                cursor = conn.execute(
                    "SELECT * FROM models WHERE status = ? ORDER BY name, created_at DESC",
                    (status.value,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM models ORDER BY name, created_at DESC"
                )
            
            models = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                
                # Convert JSON fields
                json_fields = ['tags', 'input_schema', 'output_schema', 'performance_metrics', 'dependencies']
                for field in json_fields:
                    if data[field]:
                        data[field] = json.loads(data[field])
                    else:
                        data[field] = {} if field != 'dependencies' else []
                
                # Convert dates and status
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                data['status'] = ModelStatus(data['status'])
                
                models.append(ModelMetadata(**data))
            
            return models
    
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
    
    def __init__(self, storage_path: str = "models", cache_size_mb: int = 1024):
        # Initialize components
        self.storage = FileSystemModelStorage(storage_path)
        self.registry = ModelRegistry(self.storage)
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
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Load from registry
        model, metadata = await self.registry.get_model(name, version)
        
        # Cache the result
        self.cache.put(cache_key, (model, metadata), ttl=3600.0)  # Cache for 1 hour
        
        return model, metadata
    
    async def rollback_model(self, name: str, target_version: str) -> bool:
        """Rollback a model to a specific version."""
        # Clear related cache entries
        cache_keys_to_clear = [
            f"model:{name}:latest",
            f"model:{name}:{target_version}"
        ]
        
        for key in cache_keys_to_clear:
            self.cache.remove(key)
        
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
    
    def save_state(self, key: str, value: Any):
        """Save application state."""
        with self._state_lock:
            self._state[key] = value
            
        # Also cache it
        self.cache.put(f"state:{key}", value, ttl=7200.0)  # Cache for 2 hours
        
        logger.debug(f"Saved state: {key}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get application state."""
        # Try cache first
        cache_key = f"state:{key}"
        cached_value = self.cache.get(cache_key)
        if cached_value is not None:
            return cached_value
        
        # Try in-memory state
        with self._state_lock:
            value = self._state.get(key, default)
        
        # Cache the value
        if value is not default:
            self.cache.put(cache_key, value, ttl=7200.0)
        
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
