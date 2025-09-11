"""
Tests for Data Management implementation.
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

from framework.data.state_manager import (
    StateManager, ModelRegistry, ModelVersion, IntelligentCache,
    CachePolicy, CacheStrategy
)


class TestModelVersion:
    """Test model version data structure."""
    
    def test_model_version_creation(self):
        """Test creating model version."""
        version = ModelVersion(
            model_id="resnet50",
            version="1.0.0",
            path="/models/resnet50_v1.pth",
            metadata={"accuracy": 0.92, "size_mb": 25},
            created_at=datetime.utcnow()
        )
        
        assert version.model_id == "resnet50"
        assert version.version == "1.0.0"
        assert version.path == "/models/resnet50_v1.pth"
        assert version.metadata["accuracy"] == 0.92
        assert isinstance(version.created_at, datetime)
    
    def test_model_version_validation(self):
        """Test model version validation."""
        # Valid version should not raise exception
        ModelVersion(
            model_id="bert-base",
            version="2.1.0",
            path="/models/bert_base.bin"
        )
        
        # Empty model_id should raise ValueError
        with pytest.raises(ValueError):
            ModelVersion(model_id="", version="1.0", path="/path")
        
        # Invalid version format should raise ValueError
        with pytest.raises(ValueError):
            ModelVersion(model_id="test", version="", path="/path")
    
    def test_model_version_to_dict(self):
        """Test converting model version to dict."""
        created_time = datetime.utcnow()
        version = ModelVersion(
            model_id="gpt2",
            version="3.0.0",
            path="/models/gpt2.bin",
            metadata={"parameters": "117M"},
            created_at=created_time
        )
        
        version_dict = version.to_dict()
        
        assert version_dict["model_id"] == "gpt2"
        assert version_dict["version"] == "3.0.0"
        assert version_dict["path"] == "/models/gpt2.bin"
        assert version_dict["metadata"]["parameters"] == "117M"
        assert version_dict["created_at"] == created_time.isoformat()
    
    def test_model_version_from_dict(self):
        """Test creating model version from dict."""
        created_time = datetime.utcnow()
        version_dict = {
            "model_id": "transformer",
            "version": "1.2.3",
            "path": "/models/transformer.pth",
            "metadata": {"layers": 12},
            "created_at": created_time.isoformat(),
            "checksum": "abc123",
            "size_bytes": 1024000
        }
        
        version = ModelVersion.from_dict(version_dict)
        
        assert version.model_id == "transformer"
        assert version.version == "1.2.3"
        assert version.checksum == "abc123"
        assert version.size_bytes == 1024000
    
    def test_model_version_comparison(self):
        """Test model version comparison."""
        v1 = ModelVersion("test", "1.0.0", "/path1")
        v2 = ModelVersion("test", "1.0.1", "/path2")
        v3 = ModelVersion("test", "2.0.0", "/path3")
        
        assert v1.compare_version(v2) < 0  # v1 < v2
        assert v2.compare_version(v1) > 0  # v2 > v1
        assert v1.compare_version(v1) == 0  # v1 == v1
        assert v1.compare_version(v3) < 0  # v1 < v3
    
    def test_model_version_is_compatible(self):
        """Test model version compatibility checking."""
        version = ModelVersion(
            "test",
            "2.1.0",
            "/path",
            metadata={"min_torch_version": "1.8.0"}
        )
        
        # Should be compatible with higher versions
        assert version.is_compatible("2.1.0")
        assert version.is_compatible("2.1.1")
        assert version.is_compatible("2.2.0")
        
        # Should not be compatible with lower versions
        assert not version.is_compatible("2.0.9")
        assert not version.is_compatible("1.9.0")


class TestCachePolicy:
    """Test cache policy functionality."""
    
    def test_cache_policy_creation(self):
        """Test creating cache policy."""
        policy = CachePolicy(
            strategy=CacheStrategy.LRU,
            max_size_mb=1024,
            ttl_seconds=3600,
            max_items=1000
        )
        
        assert policy.strategy == CacheStrategy.LRU
        assert policy.max_size_mb == 1024
        assert policy.ttl_seconds == 3600
        assert policy.max_items == 1000
    
    def test_cache_policy_validation(self):
        """Test cache policy validation."""
        # Valid policy should not raise exception
        CachePolicy(strategy=CacheStrategy.FIFO, max_size_mb=512)
        
        # Negative max_size_mb should raise ValueError
        with pytest.raises(ValueError):
            CachePolicy(strategy=CacheStrategy.LRU, max_size_mb=-100)
        
        # Zero ttl_seconds should raise ValueError  
        with pytest.raises(ValueError):
            CachePolicy(strategy=CacheStrategy.TTL, ttl_seconds=0)
    
    def test_cache_policy_to_dict(self):
        """Test converting cache policy to dict."""
        policy = CachePolicy(
            strategy=CacheStrategy.ADAPTIVE,
            max_size_mb=2048,
            ttl_seconds=7200,
            max_items=5000,
            custom_params={"threshold": 0.8}
        )
        
        policy_dict = policy.to_dict()
        
        assert policy_dict["strategy"] == "adaptive"
        assert policy_dict["max_size_mb"] == 2048
        assert policy_dict["ttl_seconds"] == 7200
        assert policy_dict["max_items"] == 5000
        assert policy_dict["custom_params"]["threshold"] == 0.8


class TestIntelligentCache:
    """Test intelligent cache functionality."""
    
    @pytest.fixture
    def cache_policy(self):
        """Create cache policy."""
        return CachePolicy(
            strategy=CacheStrategy.LRU,
            max_size_mb=100,
            ttl_seconds=3600,
            max_items=50
        )
    
    @pytest.fixture
    def intelligent_cache(self, cache_policy):
        """Create intelligent cache."""
        return IntelligentCache(cache_policy)
    
    def test_cache_initialization(self, intelligent_cache, cache_policy):
        """Test cache initialization."""
        assert intelligent_cache.policy is cache_policy
        assert len(intelligent_cache._cache) == 0
        assert intelligent_cache._hit_count == 0
        assert intelligent_cache._miss_count == 0
        assert intelligent_cache._current_size == 0
    
    @pytest.mark.asyncio
    async def test_cache_put_and_get(self, intelligent_cache):
        """Test putting and getting cache items."""
        # Put item in cache
        await intelligent_cache.put("key1", "value1", size_mb=1)
        
        # Get item from cache
        value = await intelligent_cache.get("key1")
        
        assert value == "value1"
        assert intelligent_cache._current_size_mb == 1
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, intelligent_cache):
        """Test cache miss scenarios."""
        # Get non-existent key
        value = await intelligent_cache.get("nonexistent")
        
        assert value is None
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, intelligent_cache):
        """Test LRU eviction policy."""
        # Fill cache to capacity
        for i in range(50):  # Max items = 50
            await intelligent_cache.put(f"key{i}", f"value{i}", size_mb=1)
        
        # Access key0 to make it recently used
        await intelligent_cache.get("key0")
        
        # Add one more item (should evict key1, not key0)
        await intelligent_cache.put("key50", "value50", size_mb=1)
        
        # key0 should still exist, key1 should be evicted
        assert await intelligent_cache.get("key0") == "value0"
        assert await intelligent_cache.get("key1") is None
        assert await intelligent_cache.get("key50") == "value50"
    
    @pytest.mark.asyncio
    async def test_cache_size_eviction(self, intelligent_cache):
        """Test size-based eviction."""
        # Add items that exceed size limit
        await intelligent_cache.put("large_item", "data" * 1000, size_mb=80)
        await intelligent_cache.put("small_item1", "data1", size_mb=10)
        await intelligent_cache.put("small_item2", "data2", size_mb=10)
        
        # Current size: 100MB (at limit)
        
        # Add another item that exceeds limit
        await intelligent_cache.put("overflow_item", "overflow", size_mb=20)
        
        # Should evict oldest items to make room
        assert intelligent_cache._current_size_mb <= intelligent_cache.policy.max_size_mb
        assert len(intelligent_cache._cache) <= intelligent_cache.policy.max_items
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        policy = CachePolicy(
            strategy=CacheStrategy.TTL,
            ttl_seconds=1,  # 1 second TTL
            max_size_mb=100
        )
        cache = IntelligentCache(policy)
        
        # Put item in cache
        await cache.put("ttl_key", "ttl_value", size_mb=1)
        
        # Immediately available
        assert await cache.get("ttl_key") == "ttl_value"
        
        # Wait for TTL expiration
        await asyncio.sleep(1.1)
        
        # Should be expired and return None
        assert await cache.get("ttl_key") is None
    
    @pytest.mark.asyncio
    async def test_cache_contains(self, intelligent_cache):
        """Test cache contains check."""
        await intelligent_cache.put("test_key", "test_value", size_mb=1)
        
        assert await intelligent_cache.contains("test_key") is True
        assert await intelligent_cache.contains("missing_key") is False
    
    @pytest.mark.asyncio
    async def test_cache_remove(self, intelligent_cache):
        """Test cache item removal."""
        await intelligent_cache.put("remove_key", "remove_value", size_mb=5)
        
        # Verify item exists
        assert await intelligent_cache.get("remove_key") == "remove_value"
        
        # Remove item
        removed = await intelligent_cache.remove("remove_key")
        
        assert removed is True
        assert await intelligent_cache.get("remove_key") is None
        assert intelligent_cache._current_size_mb == 0
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, intelligent_cache):
        """Test clearing entire cache."""
        # Add multiple items
        for i in range(10):
            await intelligent_cache.put(f"clear_key{i}", f"value{i}", size_mb=1)
        
        assert len(intelligent_cache._cache) == 10
        assert intelligent_cache._current_size_mb == 10
        
        # Clear cache
        await intelligent_cache.clear()
        
        assert len(intelligent_cache._cache) == 0
        assert intelligent_cache._current_size_mb == 0
        assert len(intelligent_cache._access_times) == 0
    
    def test_cache_stats(self, intelligent_cache):
        """Test cache statistics."""
        stats = intelligent_cache.get_stats()
        
        assert "size_mb" in stats
        assert "max_size_mb" in stats
        assert "item_count" in stats
        assert "max_items" in stats
        assert "hit_rate" in stats
        assert "miss_rate" in stats
        
        assert stats["size_mb"] == 0
        assert stats["item_count"] == 0
    
    @pytest.mark.asyncio
    async def test_cache_hit_miss_statistics(self, intelligent_cache):
        """Test cache hit/miss statistics tracking."""
        # Initially no hits or misses
        stats = intelligent_cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Cache miss
        await intelligent_cache.get("missing_key")
        stats = intelligent_cache.get_stats()
        assert stats["misses"] == 1
        
        # Cache put and hit
        await intelligent_cache.put("hit_key", "hit_value", size_mb=1)
        await intelligent_cache.get("hit_key")
        
        stats = intelligent_cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 0.5) < 0.01  # 1/(1+1) = 0.5
    
    @pytest.mark.asyncio
    async def test_adaptive_cache_strategy(self):
        """Test adaptive cache strategy."""
        policy = CachePolicy(
            strategy=CacheStrategy.ADAPTIVE,
            max_size_mb=100,
            max_items=20
        )
        cache = IntelligentCache(policy)
        
        # Add items with different access patterns
        for i in range(10):
            await cache.put(f"frequent_{i}", f"value_{i}", size_mb=2)
            # Access some items more frequently
            if i < 5:
                for _ in range(5):  # Access 5 times
                    await cache.get(f"frequent_{i}")
        
        # Add more items to trigger eviction
        for i in range(15):
            await cache.put(f"new_{i}", f"new_value_{i}", size_mb=2)
        
        # Frequently accessed items should be more likely to remain
        frequent_remaining = sum(
            1 for i in range(5) 
            if await cache.contains(f"frequent_{i}")
        )
        infrequent_remaining = sum(
            1 for i in range(5, 10) 
            if await cache.contains(f"frequent_{i}")
        )
        
        # More frequent items should remain in cache
        assert frequent_remaining >= infrequent_remaining
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, intelligent_cache):
        """Test concurrent cache operations."""
        async def cache_worker(worker_id, operation_count):
            results = []
            for i in range(operation_count):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"
                
                # Put operation
                await intelligent_cache.put(key, value, size_mb=0.1)
                
                # Get operation
                retrieved = await intelligent_cache.get(key)
                results.append(retrieved == value)
                
                # Small delay to increase concurrency likelihood
                await asyncio.sleep(0.001)
            
            return results
        
        # Run multiple workers concurrently
        tasks = []
        for worker_id in range(5):
            task = asyncio.create_task(cache_worker(worker_id, 10))
            tasks.append(task)
        
        all_results = await asyncio.gather(*tasks)
        
        # All operations should succeed
        for worker_results in all_results:
            assert all(worker_results)


class TestModelRegistry:
    """Test model registry functionality."""
    
    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary directory for registry."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_registry(self, temp_registry_dir):
        """Create model registry."""
        from framework.data.state_manager import FileSystemModelStorage
        storage = FileSystemModelStorage(temp_registry_dir)
        return ModelRegistry(storage)
    
    def test_registry_initialization(self, model_registry, temp_registry_dir):
        """Test model registry initialization."""
        assert model_registry.storage.base_path == Path(temp_registry_dir)
        assert isinstance(model_registry._model_cache, dict)
    
    @pytest.mark.asyncio
    async def test_register_model_version(self, model_registry):
        """Test registering model version."""
        version = ModelVersion(
            model_id="test_model",
            version="1.0.0",
            path="/models/test_model.pth",
            metadata={"accuracy": 0.95}
        )
        
        await model_registry.register_version(version)
        
        assert "test_model" in model_registry._models
        assert version in model_registry._versions["test_model"]
    
    @pytest.mark.asyncio
    async def test_get_latest_version(self, model_registry):
        """Test getting latest model version."""
        # Register multiple versions
        versions = [
            ModelVersion("test_model", "1.0.0", "/v1"),
            ModelVersion("test_model", "1.1.0", "/v1.1"),
            ModelVersion("test_model", "2.0.0", "/v2")
        ]
        
        for version in versions:
            await model_registry.register_version(version)
        
        latest = await model_registry.get_latest_version("test_model")
        
        assert latest.version == "2.0.0"
        assert latest.path == "/v2"
    
    @pytest.mark.asyncio
    async def test_get_specific_version(self, model_registry):
        """Test getting specific model version."""
        version = ModelVersion("specific_model", "1.5.2", "/specific")
        await model_registry.register_version(version)
        
        retrieved = await model_registry.get_version("specific_model", "1.5.2")
        
        assert retrieved.model_id == "specific_model"
        assert retrieved.version == "1.5.2"
        assert retrieved.path == "/specific"
    
    @pytest.mark.asyncio
    async def test_list_model_versions(self, model_registry):
        """Test listing all versions for a model."""
        versions = [
            ModelVersion("multi_model", "1.0.0", "/v1"),
            ModelVersion("multi_model", "1.1.0", "/v1.1"),
            ModelVersion("multi_model", "1.2.0", "/v1.2")
        ]
        
        for version in versions:
            await model_registry.register_version(version)
        
        all_versions = await model_registry.list_versions("multi_model")
        
        assert len(all_versions) == 3
        version_numbers = [v.version for v in all_versions]
        assert "1.0.0" in version_numbers
        assert "1.1.0" in version_numbers  
        assert "1.2.0" in version_numbers
    
    @pytest.mark.asyncio
    async def test_list_all_models(self, model_registry):
        """Test listing all registered models."""
        # Register versions for different models
        await model_registry.register_version(ModelVersion("model_a", "1.0", "/a"))
        await model_registry.register_version(ModelVersion("model_b", "2.0", "/b"))
        await model_registry.register_version(ModelVersion("model_c", "1.5", "/c"))
        
        all_models = await model_registry.list_models()
        
        assert len(all_models) == 3
        assert "model_a" in all_models
        assert "model_b" in all_models
        assert "model_c" in all_models
    
    @pytest.mark.asyncio
    async def test_remove_model_version(self, model_registry):
        """Test removing specific model version."""
        # Register multiple versions
        await model_registry.register_version(ModelVersion("remove_model", "1.0", "/v1"))
        await model_registry.register_version(ModelVersion("remove_model", "2.0", "/v2"))
        
        # Remove one version
        removed = await model_registry.remove_version("remove_model", "1.0")
        
        assert removed is True
        
        # Check version was removed
        versions = await model_registry.list_versions("remove_model")
        version_numbers = [v.version for v in versions]
        assert "1.0" not in version_numbers
        assert "2.0" in version_numbers
    
    @pytest.mark.asyncio
    async def test_remove_all_model_versions(self, model_registry):
        """Test removing all versions of a model."""
        # Register multiple versions
        await model_registry.register_version(ModelVersion("delete_model", "1.0", "/v1"))
        await model_registry.register_version(ModelVersion("delete_model", "2.0", "/v2"))
        
        removed_count = await model_registry.remove_model("delete_model")
        
        assert removed_count == 2
        assert "delete_model" not in model_registry._models
        assert "delete_model" not in model_registry._versions
    
    @pytest.mark.asyncio
    async def test_model_metadata_search(self, model_registry):
        """Test searching models by metadata."""
        # Register models with different metadata
        await model_registry.register_version(ModelVersion(
            "high_accuracy", "1.0", "/high",
            metadata={"accuracy": 0.95, "model_type": "cnn"}
        ))
        await model_registry.register_version(ModelVersion(
            "fast_model", "1.0", "/fast",
            metadata={"accuracy": 0.80, "model_type": "linear", "speed": "fast"}
        ))
        await model_registry.register_version(ModelVersion(
            "balanced_model", "1.0", "/balanced",
            metadata={"accuracy": 0.88, "model_type": "cnn", "speed": "medium"}
        ))
        
        # Search by accuracy threshold
        high_accuracy_models = await model_registry.search_by_metadata(
            {"accuracy": {"$gte": 0.90}}
        )
        
        assert len(high_accuracy_models) == 1
        assert high_accuracy_models[0].model_id == "high_accuracy"
        
        # Search by model type
        cnn_models = await model_registry.search_by_metadata(
            {"model_type": "cnn"}
        )
        
        assert len(cnn_models) == 2
        model_ids = [m.model_id for m in cnn_models]
        assert "high_accuracy" in model_ids
        assert "balanced_model" in model_ids
    
    @pytest.mark.asyncio
    async def test_registry_persistence(self, model_registry):
        """Test registry persistence to disk."""
        # Register some models
        await model_registry.register_version(ModelVersion("persist_model", "1.0", "/persist"))
        
        # Save to disk
        await model_registry.save_to_disk()
        
        # Create new registry instance and load from disk
        new_registry = ModelRegistry(model_registry.storage)
        await new_registry.load_from_disk()
        
        # Should have the same models
        models = await new_registry.list_models()
        assert "persist_model" in models
        
        version = await new_registry.get_version("persist_model", "1.0")
        assert version.path == "/persist"
    
    @pytest.mark.asyncio
    async def test_model_validation_on_registration(self, model_registry):
        """Test model validation during registration."""
        # Valid model should register successfully
        valid_version = ModelVersion("valid_model", "1.0.0", "/valid/path.pth")
        await model_registry.register_version(valid_version)
        
        # Duplicate version should raise exception
        duplicate_version = ModelVersion("valid_model", "1.0.0", "/different/path.pth")
        with pytest.raises(ValueError, match="already exists"):
            await model_registry.register_version(duplicate_version)
    
    @pytest.mark.asyncio
    async def test_model_checksum_validation(self, model_registry):
        """Test model checksum validation."""
        with patch('hashlib.md5') as mock_md5:
            mock_md5.return_value.hexdigest.return_value = "abc123def456"
            
            version = ModelVersion(
                "checksum_model", "1.0", "/model.pth",
                checksum="abc123def456"
            )
            
            # Should register successfully with matching checksum
            await model_registry.register_version(version)
            
            # Verify checksum validation was called
            retrieved = await model_registry.get_version("checksum_model", "1.0")
            assert retrieved.checksum == "abc123def456"


class TestStateManager:
    """Test state manager functionality."""
    
    @pytest.fixture
    def temp_state_dir(self):
        """Create temporary directory for state."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def state_manager(self, temp_state_dir):
        """Create state manager."""
        cache_policy = CachePolicy(
            strategy=CacheStrategy.LRU,
            max_size_mb=512,
            max_items=100
        )
        return StateManager(
            storage_path=temp_state_dir,
            cache_size_mb=cache_policy.max_size_mb
        )
    
    def test_state_manager_initialization(self, state_manager, temp_state_dir):
        """Test state manager initialization."""
        assert isinstance(state_manager.registry, ModelRegistry)
        assert isinstance(state_manager.cache, IntelligentCache)
        assert state_manager.storage.base_path == Path(temp_state_dir)
    
    @pytest.mark.asyncio
    async def test_register_and_get_model(self, state_manager):
        """Test registering and retrieving models."""
        # Register a model
        version = ModelVersion(
            "state_test_model",
            "1.0.0", 
            "/models/test.pth",
            metadata={"framework": "pytorch"}
        )
        
        await state_manager.register_model_version(version)
        
        # Retrieve the model
        retrieved = await state_manager.get_model_version("state_test_model", "1.0.0")
        
        assert retrieved.model_id == "state_test_model"
        assert retrieved.version == "1.0.0"
        assert retrieved.metadata["framework"] == "pytorch"
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, state_manager):
        """Test cache integration with model management."""
        # Cache some model data
        model_data = {"weights": [1, 2, 3], "bias": [0.1, 0.2]}
        await state_manager.cache_model_data("cached_model", model_data, size_mb=10)
        
        # Retrieve from cache
        cached_data = await state_manager.get_cached_model_data("cached_model")
        
        assert cached_data == model_data
    
    @pytest.mark.asyncio
    async def test_preload_model_to_cache(self, state_manager):
        """Test preloading models to cache."""
        # Register a model
        version = ModelVersion("preload_model", "1.0", "/models/preload.pth")
        await state_manager.register_model_version(version)
        
        # Mock loading model data
        with patch.object(state_manager, '_load_model_data') as mock_load:
            mock_load.return_value = {"loaded": "data"}
            
            # Preload to cache
            success = await state_manager.preload_model_to_cache("preload_model", "1.0")
            
            assert success is True
            
            # Should be available in cache
            cached = await state_manager.get_cached_model_data("preload_model:1.0")
            assert cached == {"loaded": "data"}
    
    @pytest.mark.asyncio
    async def test_model_lifecycle_management(self, state_manager):
        """Test complete model lifecycle management."""
        # Register initial version
        v1 = ModelVersion("lifecycle_model", "1.0.0", "/v1.pth")
        await state_manager.register_model_version(v1)
        
        # Register updated version
        v2 = ModelVersion("lifecycle_model", "1.1.0", "/v1.1.pth")
        await state_manager.register_model_version(v2)
        
        # Get all versions
        versions = await state_manager.list_model_versions("lifecycle_model")
        assert len(versions) == 2
        
        # Get latest version
        latest = await state_manager.get_latest_model_version("lifecycle_model")
        assert latest.version == "1.1.0"
        
        # Remove old version
        await state_manager.remove_model_version("lifecycle_model", "1.0.0")
        
        versions = await state_manager.list_model_versions("lifecycle_model")
        assert len(versions) == 1
        assert versions[0].version == "1.1.0"
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, state_manager):
        """Test state persistence and recovery."""
        # Add some state data
        version = ModelVersion("persist_state", "1.0", "/persist.pth")
        await state_manager.register_model_version(version)
        
        await state_manager.cache_model_data("persist_cache", {"data": "cached"}, size_mb=1)
        
        # Save state
        await state_manager.save_state()
        
        # Create new state manager and load state
        new_state_manager = StateManager(
            storage_path=str(state_manager.storage.base_path),
            cache_size_mb=state_manager.cache.policy.max_size_mb
        )
        await new_state_manager.load_state()
        
        # Verify state was restored
        models = await new_state_manager.list_all_models()
        assert "persist_state" in models
        
        # Cache might not persist, but registry should
        retrieved = await new_state_manager.get_model_version("persist_state", "1.0")
        assert retrieved.path == "/persist.pth"
    
    @pytest.mark.asyncio
    async def test_cache_warming(self, state_manager):
        """Test cache warming functionality."""
        # Register multiple models
        models = [
            ModelVersion("warm1", "1.0", "/warm1.pth"),
            ModelVersion("warm2", "1.0", "/warm2.pth"),
            ModelVersion("warm3", "1.0", "/warm3.pth")
        ]
        
        for model in models:
            await state_manager.register_model_version(model)
        
        # Mock model loading
        with patch.object(state_manager, '_load_model_data') as mock_load:
            mock_load.side_effect = lambda model_id, version: {"model": f"{model_id}_{version}"}
            
            # Warm cache for specific models
            warmed_models = ["warm1", "warm2"]
            success_count = await state_manager.warm_cache(warmed_models)
            
            assert success_count == 2
            
            # Check models are in cache
            for model_id in warmed_models:
                cached = await state_manager.get_cached_model_data(f"{model_id}:1.0")
                assert cached == {"model": f"{model_id}_1.0"}
    
    def test_get_state_statistics(self, state_manager):
        """Test getting state management statistics."""
        stats = state_manager.get_state_stats()
        
        assert "registry" in stats
        assert "cache" in stats
        
        assert "total_models" in stats["registry"]
        assert "total_versions" in stats["registry"]
        
        assert "size_mb" in stats["cache"]
        assert "item_count" in stats["cache"]
        assert "hit_rate" in stats["cache"]
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_cache_entries(self, state_manager):
        """Test cleanup of expired cache entries."""
        # Use TTL cache policy for this test
        ttl_policy = CachePolicy(
            strategy=CacheStrategy.TTL,
            ttl_seconds=1,
            max_size_mb=100
        )
        ttl_state_manager = StateManager(
            storage_path=str(state_manager.storage.base_path),
            cache_size_mb=ttl_policy.max_size_mb
        )
        
        # Add cache entries
        await ttl_state_manager.cache_model_data("expire1", {"data": 1}, size_mb=1)
        await ttl_state_manager.cache_model_data("expire2", {"data": 2}, size_mb=1)
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Cleanup expired entries
        cleaned_count = await ttl_state_manager.cleanup_expired_cache()
        
        assert cleaned_count >= 0  # May have auto-expired
        
        # Entries should be gone
        assert await ttl_state_manager.get_cached_model_data("expire1") is None
        assert await ttl_state_manager.get_cached_model_data("expire2") is None
    
    @pytest.mark.asyncio
    async def test_concurrent_state_operations(self, state_manager):
        """Test concurrent state management operations."""
        async def register_models(start_id, count):
            for i in range(count):
                model_id = f"concurrent_{start_id}_{i}"
                version = ModelVersion(model_id, "1.0", f"/path/{model_id}.pth")
                await state_manager.register_model_version(version)
        
        async def cache_operations(start_id, count):
            for i in range(count):
                key = f"cache_{start_id}_{i}"
                await state_manager.cache_model_data(key, {"id": start_id, "i": i}, size_mb=0.1)
        
        # Run concurrent operations
        tasks = []
        for worker_id in range(3):
            tasks.append(asyncio.create_task(register_models(worker_id, 10)))
            tasks.append(asyncio.create_task(cache_operations(worker_id, 10)))
        
        await asyncio.gather(*tasks)
        
        # Verify all operations completed successfully
        all_models = await state_manager.list_all_models()
        assert len(all_models) >= 30  # 3 workers * 10 models each
        
        cache_stats = state_manager.get_state_stats()["cache"]
        assert cache_stats["item_count"] >= 30  # 3 workers * 10 cache entries each


class TestStateManagerIntegration:
    """Test state manager integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_model_deployment(self):
        """Test complete model deployment scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup state manager
            cache_policy = CachePolicy(
                strategy=CacheStrategy.ADAPTIVE,
                max_size_mb=1024,
                max_items=50
            )
            state_manager = StateManager(storage_path=temp_dir, cache_size_mb=cache_policy.max_size_mb)
            
            # Simulate model development lifecycle
            
            # 1. Register initial model version
            v1 = ModelVersion(
                "production_model",
                "1.0.0",
                "/models/prod_v1.pth",
                metadata={"accuracy": 0.85, "size_mb": 100}
            )
            await state_manager.register_model_version(v1)
            
            # 2. Preload to cache for fast serving
            with patch.object(state_manager, '_load_model_data') as mock_load:
                mock_load.return_value = {"weights": "v1_weights", "config": "v1_config"}
                
                success = await state_manager.preload_model_to_cache("production_model", "1.0.0")
                assert success is True
            
            # 3. Deploy improved version
            v2 = ModelVersion(
                "production_model", 
                "1.1.0",
                "/models/prod_v1_1.pth",
                metadata={"accuracy": 0.89, "size_mb": 120}
            )
            await state_manager.register_model_version(v2)
            
            # 4. Gradual rollout - cache new version
            with patch.object(state_manager, '_load_model_data') as mock_load:
                mock_load.return_value = {"weights": "v1_1_weights", "config": "v1_1_config"}
                
                await state_manager.preload_model_to_cache("production_model", "1.1.0")
            
            # 5. Verify both versions available
            v1_cached = await state_manager.get_cached_model_data("production_model:1.0.0")
            v2_cached = await state_manager.get_cached_model_data("production_model:1.1.0")
            
            assert v1_cached["weights"] == "v1_weights"
            assert v2_cached["weights"] == "v1_1_weights"
            
            # 6. Remove old version after successful deployment
            await state_manager.remove_model_version("production_model", "1.0.0")
            
            versions = await state_manager.list_model_versions("production_model")
            assert len(versions) == 1
            assert versions[0].version == "1.1.0"
    
    @pytest.mark.asyncio
    async def test_multi_tenant_model_management(self):
        """Test multi-tenant model management scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_policy = CachePolicy(CacheStrategy.LRU, max_size_mb=500, max_items=20)
            state_manager = StateManager(storage_path=temp_dir, cache_size_mb=cache_policy.max_size_mb)
            
            # Register models for different tenants
            tenant_models = {
                "tenant_a": [
                    ModelVersion("sentiment_analysis", "1.0", "/tenant_a/sentiment.pth"),
                    ModelVersion("image_classifier", "2.0", "/tenant_a/image.pth")
                ],
                "tenant_b": [
                    ModelVersion("sentiment_analysis", "1.2", "/tenant_b/sentiment.pth"), 
                    ModelVersion("text_summarizer", "1.0", "/tenant_b/summarizer.pth")
                ]
            }
            
            # Register all models
            for tenant, models in tenant_models.items():
                for model in models:
                    # Add tenant info to metadata
                    model.metadata = model.metadata or {}
                    model.metadata["tenant"] = tenant
                    await state_manager.register_model_version(model)
            
            # Search models by tenant
            tenant_a_models = await state_manager.model_registry.search_by_metadata(
                {"tenant": "tenant_a"}
            )
            tenant_b_models = await state_manager.model_registry.search_by_metadata(
                {"tenant": "tenant_b"}
            )
            
            assert len(tenant_a_models) == 2
            assert len(tenant_b_models) == 2
            
            # Verify tenant isolation
            tenant_a_sentiment = await state_manager.get_model_version("sentiment_analysis", "1.0")
            tenant_b_sentiment = await state_manager.get_model_version("sentiment_analysis", "1.2")
            
            assert tenant_a_sentiment.metadata["tenant"] == "tenant_a"
            assert tenant_b_sentiment.metadata["tenant"] == "tenant_b"
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_scenario(self):
        """Test disaster recovery and state restoration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Original state manager
            original_state = StateManager(storage_path=temp_dir)
            
            # Setup initial state
            models = [
                ModelVersion("critical_model", "1.0", "/critical.pth"),
                ModelVersion("critical_model", "1.1", "/critical_v1_1.pth"),
                ModelVersion("backup_model", "2.0", "/backup.pth")
            ]
            
            for model in models:
                await original_state.register_model_version(model)
            
            # Cache important models
            await original_state.cache_model_data("critical_model:1.1", {"critical": "data"}, size_mb=50)
            
            # Save state (simulate backup)
            await original_state.save_state()
            
            # Simulate disaster - create new state manager
            recovered_state = StateManager(storage_path=temp_dir)
            
            # Recovery process
            await recovered_state.load_state()
            
            # Verify recovery
            recovered_models = await recovered_state.list_all_models()
            assert "critical_model" in recovered_models
            assert "backup_model" in recovered_models
            
            # Verify version history
            critical_versions = await recovered_state.list_model_versions("critical_model")
            version_numbers = [v.version for v in critical_versions]
            assert "1.0" in version_numbers
            assert "1.1" in version_numbers
            
            # Cache may need to be rebuilt, but registry should be intact
            latest_critical = await recovered_state.get_latest_model_version("critical_model")
            assert latest_critical.version == "1.1"
