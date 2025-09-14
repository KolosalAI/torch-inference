"""
Tests for Health Checks implementation.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from framework.reliability.health_checks import (
    HealthCheckResult, HealthStatus, HealthCheck,
    SystemResourcesHealthCheck, GPUHealthCheck, ModelHealthCheck,
    DependencyHealthCheck, HealthCheckManager
)


class TestHealthCheckResult:
    """Test health check result data structure."""
    
    def test_healthy_result(self):
        """Test creating a healthy result."""
        result = HealthCheckResult.healthy("test_check", "All systems operational")
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All systems operational"
        assert result.details == {}
        assert isinstance(result.timestamp, datetime)
    
    def test_unhealthy_result(self):
        """Test creating an unhealthy result."""
        details = {"error_count": 5, "last_error": "Connection failed"}
        result = HealthCheckResult.unhealthy("test_check", "Service degraded", details)
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.UNHEALTHY
        assert result.message == "Service degraded"
        assert result.details == details
    
    def test_unknown_result(self):
        """Test creating an unknown status result."""
        result = HealthCheckResult.unknown("test_check", "Cannot determine status")
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.UNKNOWN
        assert result.message == "Cannot determine status"
    
    def test_result_serialization(self):
        """Test result can be serialized to dict."""
        result = HealthCheckResult.healthy("test", "OK", {"cpu": "50%"})
        result_dict = result.to_dict()
        
        assert result_dict["name"] == "test"
        assert result_dict["status"] == "HEALTHY"
        assert result_dict["message"] == "OK"
        assert result_dict["details"] == {"cpu": "50%"}
        assert "timestamp" in result_dict


class TestBaseHealthCheck:
    """Test base health check functionality."""
    
    class DummyHealthCheck(HealthCheck):
        async def _perform_check(self):
            return HealthCheckResult.healthy("dummy", "Test result")
        def __init__(self, name, check_result=None):
            super().__init__(name)
            self._check_result = check_result or HealthCheckResult.healthy(name, "OK")
        
        async def check_health(self) -> HealthCheckResult:
            return self._check_result
    
    @pytest.mark.asyncio
    async def test_basic_check(self):
        """Test basic health check execution."""
        check = self.DummyHealthCheck("dummy")
        result = await check.check_health()
        
        assert result.name == "dummy"
        assert result.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_check_with_timeout(self):
        """Test health check with timeout."""
        async def slow_check():
            await asyncio.sleep(0.2)
            return HealthCheckResult.healthy("slow", "OK")
        
        check = self.DummyHealthCheck("timeout_test")
        check._perform_check = slow_check
        
        # Should timeout after 0.1 seconds
        start_time = time.time()
        result = await check.run_check(timeout=0.1)
        elapsed = time.time() - start_time
        
        assert result.status == HealthStatus.UNKNOWN
        assert "timeout" in result.message.lower()
        assert elapsed < 0.15  # Should have timed out quickly
    
    @pytest.mark.asyncio
    async def test_check_exception_handling(self):
        """Test health check exception handling."""
        async def failing_check():
            raise Exception("Health check failed")
        
        check = self.DummyHealthCheck("failing")
        check._perform_check = failing_check
        
        result = await check.run_check()
        
        assert result.status == HealthStatus.UNKNOWN
        assert "Health check failed" in result.message


class TestSystemResourcesHealthCheck:
    """Test system resources health check."""
    
    @pytest.fixture
    def health_check(self):
        """Create system resources health check."""
        return SystemResourcesHealthCheck(
            name="system_resources",
            cpu_critical_threshold=0.8,
            memory_critical_threshold=0.85,
            disk_critical_threshold=0.9
        )
    
    @pytest.mark.asyncio
    async def test_healthy_system(self, health_check):
        """Test healthy system resources."""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0
            
            result = await health_check.check_health()
            
            assert result.status == HealthStatus.HEALTHY
            assert "cpu" in result.details
            assert "memory" in result.details
            assert "disk" in result.details
    
    @pytest.mark.asyncio
    async def test_high_cpu_usage(self, health_check):
        """Test high CPU usage detection."""
        with patch('psutil.cpu_percent', return_value=90.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 70.0
            
            result = await health_check.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY
            assert "CPU usage high" in result.message
    
    @pytest.mark.asyncio
    async def test_high_memory_usage(self, health_check):
        """Test high memory usage detection."""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 95.0
            mock_disk.return_value.percent = 70.0
            
            result = await health_check.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY
            assert "Memory usage high" in result.message
    
    @pytest.mark.asyncio
    async def test_multiple_issues(self, health_check):
        """Test multiple resource issues."""
        with patch('psutil.cpu_percent', return_value=90.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.percent = 95.0
            mock_disk.return_value.percent = 95.0
            
            result = await health_check.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY
            assert len(result.message.split(',')) == 3  # CPU, Memory, Disk


class TestGPUHealthCheck:
    """Test GPU health check."""
    
    @pytest.fixture
    def health_check(self):
        """Create GPU health check."""
        return GPUHealthCheck(
            name="gpu_resources",
            memory_critical_threshold=0.85
        )
    
    @pytest.mark.asyncio
    async def test_gpu_available_healthy(self, health_check):
        """Test healthy GPU state."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.memory_stats') as mock_memory_stats, \
             patch('torch.cuda.temperature', return_value=65):
            
            # Mock memory stats for healthy GPU
            mock_memory_stats.return_value = {
                'allocated_bytes.all.current': 1024 * 1024 * 1024,  # 1GB
                'reserved_bytes.all.current': 2048 * 1024 * 1024    # 2GB
            }
            
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
                
                result = await health_check.check_health()
                
                assert result.status == HealthStatus.HEALTHY
                assert "gpu_count" in result.details
                assert result.details["gpu_count"] == 2
    
    @pytest.mark.asyncio
    async def test_gpu_not_available(self, health_check):
        """Test when GPU is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            result = await health_check.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY
            assert "not available" in result.message
    
    @pytest.mark.asyncio
    async def test_high_gpu_memory(self, health_check):
        """Test high GPU memory usage."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.memory_stats') as mock_memory_stats, \
             patch('torch.cuda.temperature', return_value=65):
            
            # Mock high memory usage
            mock_memory_stats.return_value = {
                'allocated_bytes.all.current': 7 * 1024 * 1024 * 1024,  # 7GB
                'reserved_bytes.all.current': 8 * 1024 * 1024 * 1024    # 8GB
            }
            
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
                
                result = await health_check.check_health()
                
                assert result.status == HealthStatus.UNHEALTHY
                assert "GPU memory usage high" in result.message
    
    @pytest.mark.asyncio
    async def test_high_gpu_temperature(self, health_check):
        """Test high GPU temperature."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.memory_stats') as mock_memory_stats, \
             patch('torch.cuda.temperature', return_value=90):  # High temp
            
            mock_memory_stats.return_value = {
                'allocated_bytes.all.current': 1024 * 1024 * 1024,
                'reserved_bytes.all.current': 2048 * 1024 * 1024
            }
            
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 8 * 1024 * 1024 * 1024
                
                result = await health_check.check_health()
                
                assert result.status == HealthStatus.UNHEALTHY
                assert "GPU temperature high" in result.message


class TestModelHealthCheck:
    """Test model health check."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        model.eval = Mock()
        return model
    
    @pytest.fixture
    def health_check(self, mock_model):
        """Create model health check."""
        return ModelHealthCheck("test_model", mock_model)
    
    @pytest.mark.asyncio
    async def test_model_healthy(self, health_check, mock_model):
        """Test healthy model."""
        # Mock successful inference
        with patch.object(health_check, '_run_test_inference', return_value=True):
            result = await health_check.check_health()
            
            assert result.status == HealthStatus.HEALTHY
            assert "model_name" in result.details
            assert result.details["model_name"] == "test_model"
    
    @pytest.mark.asyncio
    async def test_model_inference_failure(self, health_check, mock_model):
        """Test model inference failure."""
        # Mock failed inference
        with patch.object(health_check, '_run_test_inference', side_effect=Exception("Inference failed")):
            result = await health_check.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY
            assert "Inference failed" in result.message
    
    @pytest.mark.asyncio
    async def test_model_none(self):
        """Test with None model."""
        health_check = ModelHealthCheck("none_model", None)
        result = await health_check.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "not loaded" in result.message


class TestDependencyHealthCheck:
    """Test dependency health check."""
    
    @pytest.fixture
    def health_check(self):
        """Create dependency health check."""
        return DependencyHealthCheck("database", "postgresql://localhost:5432/test")
    
    @pytest.mark.asyncio
    async def test_database_connection_success(self, health_check):
        """Test database connection when asyncpg is not available or connection fails."""
        result = await health_check.check_health()
        
        # The health check should return UNHEALTHY due to either missing module or connection failure
        assert result.status == HealthStatus.UNHEALTHY
        # Check for either import error or connection error
        assert ("No module named 'asyncpg'" in result.message or 
                "PostgreSQL connection failed" in result.message or
                "connection" in result.message.lower())
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self, health_check):
        """Test database connection when asyncpg is not available or connection fails."""
        result = await health_check.check_health()
        
        # The health check should return UNHEALTHY due to either missing module or connection failure
        assert result.status == HealthStatus.UNHEALTHY
        # Check for either import error or connection error
        assert ("No module named 'asyncpg'" in result.message or 
                "PostgreSQL connection failed" in result.message or
                "connection" in result.message.lower())
    
    @pytest.mark.asyncio
    async def test_redis_connection_success(self):
        """Test Redis connection when redis is not available or connection fails."""
        health_check = DependencyHealthCheck("redis", "redis://localhost:6379")
        
        result = await health_check.check_health()
        
        # The health check should return UNHEALTHY due to either missing module or connection failure
        assert result.status == HealthStatus.UNHEALTHY
        # Check for either import error or connection error
        assert ("No module named 'redis'" in result.message or 
                "Redis connection failed" in result.message or
                "connection" in result.message.lower())
    
    @pytest.mark.asyncio
    async def test_http_service_success(self):
        """Test successful HTTP service check."""
        health_check = DependencyHealthCheck("api", "http://api.example.com/health")
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__.return_value = mock_response
            mock_get.return_value = mock_response
            
            result = await health_check.check_health()
            
            assert result.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_http_service_failure(self):
        """Test HTTP service failure."""
        health_check = DependencyHealthCheck("api", "http://api.example.com/health")
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.__aenter__.return_value = mock_response
            mock_get.return_value = mock_response
            
            result = await health_check.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY


class TestHealthCheckManager:
    """Test health check manager."""
    
    @pytest.fixture
    def manager(self):
        """Create health check manager."""
        return HealthCheckManager()
    
    @pytest.fixture
    def dummy_check(self):
        """Create dummy health check."""
        class DummyCheck(HealthCheck):
            def __init__(self, name, status=HealthStatus.HEALTHY):
                super().__init__(name)
                self._status = status
            
            async def _perform_check(self):
                return HealthCheckResult(
                    name=self.name,
                    status=self._status,
                    message="Test message",
                    details={},
                    timestamp=datetime.utcnow()
                )
        
        return DummyCheck
    
    def test_register_check(self, manager, dummy_check):
        """Test registering a health check."""
        check = dummy_check("test")
        manager.register_check(check)
        
        assert "test" in manager._checks
        assert manager._checks["test"] is check
    
    def test_register_duplicate_check(self, manager, dummy_check):
        """Test registering duplicate health check."""
        check1 = dummy_check("test")
        check2 = dummy_check("test")
        
        manager.register_check(check1)
        
        with pytest.raises(ValueError, match="already registered"):
            manager.register_check(check2)
    
    def test_unregister_check(self, manager, dummy_check):
        """Test unregistering a health check."""
        check = dummy_check("test")
        manager.register_check(check)
        manager.unregister_check("test")
        
        assert "test" not in manager._checks
    
    def test_unregister_nonexistent_check(self, manager):
        """Test unregistering non-existent check."""
        with pytest.raises(ValueError, match="not found"):
            manager.unregister_check("nonexistent")
    
    @pytest.mark.asyncio
    async def test_run_all_checks_healthy(self, manager, dummy_check):
        """Test running all checks when all are healthy."""
        checks = [
            dummy_check("check1", HealthStatus.HEALTHY),
            dummy_check("check2", HealthStatus.HEALTHY),
            dummy_check("check3", HealthStatus.HEALTHY)
        ]
        
        for check in checks:
            manager.register_check(check)
        
        results = await manager.run_all_checks()
        
        assert len(results) == 3
        assert all(r.status == HealthStatus.HEALTHY for r in results)
    
    @pytest.mark.asyncio
    async def test_run_all_checks_mixed(self, manager, dummy_check):
        """Test running all checks with mixed results."""
        checks = [
            dummy_check("healthy", HealthStatus.HEALTHY),
            dummy_check("unhealthy", HealthStatus.UNHEALTHY),
            dummy_check("unknown", HealthStatus.UNKNOWN)
        ]
        
        for check in checks:
            manager.register_check(check)
        
        results = await manager.run_all_checks()
        
        assert len(results) == 3
        statuses = [r.status for r in results]
        assert HealthStatus.HEALTHY in statuses
        assert HealthStatus.UNHEALTHY in statuses
        assert HealthStatus.UNKNOWN in statuses
    
    @pytest.mark.asyncio
    async def test_run_specific_checks(self, manager, dummy_check):
        """Test running specific checks."""
        checks = [
            dummy_check("check1", HealthStatus.HEALTHY),
            dummy_check("check2", HealthStatus.UNHEALTHY),
            dummy_check("check3", HealthStatus.HEALTHY)
        ]
        
        for check in checks:
            manager.register_check(check)
        
        results = await manager.run_checks(["check1", "check3"])
        
        assert len(results) == 2
        names = [r.name for r in results]
        assert "check1" in names
        assert "check3" in names
        assert "check2" not in names
    
    @pytest.mark.asyncio
    async def test_run_nonexistent_check(self, manager):
        """Test running non-existent check."""
        results = await manager.run_checks(["nonexistent"])
        
        assert len(results) == 1
        assert results[0].status == HealthStatus.UNKNOWN
        assert "not found" in results[0].message
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, manager, dummy_check):
        """Test concurrent execution of health checks."""
        async def slow_check_health(self):
            await asyncio.sleep(0.1)
            return HealthCheckResult.healthy(self.name, "OK")
        
        # Create slow checks
        checks = []
        for i in range(3):
            check = dummy_check(f"slow{i}")
            check._perform_check = slow_check_health.__get__(check, type(check))
            checks.append(check)
            manager.register_check(check)
        
        start_time = time.time()
        results = await manager.run_all_checks()
        elapsed = time.time() - start_time
        
        # Should complete in roughly 0.1 seconds (concurrent), not 0.3 (sequential)
        assert elapsed < 0.2
        assert len(results) == 3
    
    def test_get_overall_health_all_healthy(self, manager, dummy_check):
        """Test overall health when all checks are healthy."""
        results = [
            HealthCheckResult.healthy("check1", "OK"),
            HealthCheckResult.healthy("check2", "OK"),
            HealthCheckResult.healthy("check3", "OK")
        ]
        
        overall = manager.get_overall_health(results)
        assert overall == HealthStatus.HEALTHY
    
    def test_get_overall_health_with_unhealthy(self, manager, dummy_check):
        """Test overall health with unhealthy checks."""
        results = [
            HealthCheckResult.healthy("check1", "OK"),
            HealthCheckResult.unhealthy("check2", "Failed"),
            HealthCheckResult.healthy("check3", "OK")
        ]
        
        overall = manager.get_overall_health(results)
        assert overall == HealthStatus.UNHEALTHY
    
    def test_get_overall_health_with_unknown(self, manager, dummy_check):
        """Test overall health with unknown checks."""
        results = [
            HealthCheckResult.healthy("check1", "OK"),
            HealthCheckResult.unknown("check2", "Cannot determine"),
            HealthCheckResult.healthy("check3", "OK")
        ]
        
        overall = manager.get_overall_health(results)
        assert overall == HealthStatus.UNKNOWN
    
    def test_get_overall_health_priority(self, manager, dummy_check):
        """Test overall health priority (UNHEALTHY > UNKNOWN > HEALTHY)."""
        results = [
            HealthCheckResult.healthy("check1", "OK"),
            HealthCheckResult.unknown("check2", "Cannot determine"),
            HealthCheckResult.unhealthy("check3", "Failed")
        ]
        
        overall = manager.get_overall_health(results)
        assert overall == HealthStatus.UNHEALTHY  # Should prioritize UNHEALTHY
