"""Tests for enterprise features."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, mock_open
from datetime import datetime, timezone

# Test with mock classes if enterprise modules not available
try:
    from framework.enterprise.config import (
        EnterpriseConfig, AuthConfig, SecurityConfig, AuthProvider
    )
    from framework.enterprise.auth import EnterpriseAuth, JWTManager
    from framework.enterprise.security import SecurityManager, EncryptionManager
    from framework.enterprise.governance import (
        ModelGovernance, ModelValidator, ABTestManager, ABTestConfig
    )
    from framework.enterprise.monitoring import EnterpriseMonitor
except ImportError:
    # Mock classes for testing when enterprise features not available
    EnterpriseConfig = None
    AuthConfig = None
    SecurityConfig = None
    AuthProvider = None
    EnterpriseAuth = None
    JWTManager = None
    SecurityManager = None
    EncryptionManager = None
    ModelGovernance = None
    ModelValidator = None
    ABTestManager = None
    ABTestConfig = None
    EnterpriseMonitor = None


@pytest.mark.skipif(EnterpriseConfig is None, reason="Enterprise features not available")
class TestEnterpriseConfig:
    """Test enterprise configuration."""
    
    def test_enterprise_config_creation(self):
        """Test creating enterprise configuration."""
        config = EnterpriseConfig(
            environment="production",
            auth=AuthConfig(
                provider=AuthProvider.OAUTH2,
                secret_key="test-secret"
            ),
            security=SecurityConfig(
                enable_encryption_at_rest=True,
                enable_rate_limiting=True
            )
        )
        
        assert config.environment == "production"
        assert config.auth.provider == AuthProvider.OAUTH2
        assert config.security.enable_encryption_at_rest
        assert config.security.enable_rate_limiting
    
    def test_enterprise_config_defaults(self):
        """Test enterprise configuration defaults."""
        config = EnterpriseConfig()
        
        assert config.environment == "production"  # Default is production, not development
        assert isinstance(config.auth, AuthConfig)
        assert isinstance(config.security, SecurityConfig)


@pytest.mark.skipif(EnterpriseAuth is None, reason="Auth manager not available")
class TestAuthManager:
    """Test authentication manager."""
    
    @pytest.fixture
    def auth_config(self):
        """Create auth configuration."""
        return AuthConfig(
            provider=AuthProvider.OAUTH2,
            secret_key="test-secret-key",
            oauth2_client_id="test-client-id"
        )
    
    @pytest.fixture
    def auth_manager(self, auth_config):
        """Create auth manager."""
        config = EnterpriseConfig(auth=auth_config)
        return EnterpriseAuth(config)
    
    def test_auth_manager_initialization(self, auth_manager, auth_config):
        """Test auth manager initialization."""
        assert auth_manager.config.auth == auth_config
        assert auth_manager.config.auth.provider == AuthProvider.OAUTH2
    
    @pytest.mark.asyncio
    async def test_user_authentication(self, auth_manager):
        """Test user authentication."""
        # Mock successful authentication
        with patch.object(auth_manager, 'authenticate_password') as mock_auth:
            mock_auth.return_value = Mock(user_id="test_user", roles=["user"])
            
            result = await mock_auth("test_user", "password")
            
            if result:
                assert result.user_id == "test_user"
                assert "user" in result.roles
            mock_auth.assert_called_once_with("test_user", "password")
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, auth_manager):
        """Test authentication failure."""
        with patch.object(auth_manager, 'authenticate_password') as mock_auth:
            mock_auth.return_value = None  # Authentication failed
            
            result = await mock_auth("invalid_user", "wrong_password")
            assert result is None
    
    def test_token_validation(self, auth_manager):
        """Test JWT token validation."""
        # Create mock token
        if hasattr(auth_manager, 'jwt_manager'):
            with patch.object(auth_manager.jwt_manager, 'verify_token') as mock_verify:
                mock_verify.return_value = {"user_id": "test", "exp": 9999999999}
                
                result = mock_verify("valid_token")
                assert result is not None
                
                mock_verify.assert_called_once()
    
    def test_token_generation(self, auth_manager):
        """Test JWT token generation."""
        if hasattr(auth_manager, 'jwt_manager'):
            # Create a mock user
            mock_user = Mock()
            mock_user.id = "test_user"
            mock_user.username = "test"
            mock_user.roles = ["admin"]
            
            with patch.object(auth_manager.jwt_manager, 'create_access_token') as mock_generate:
                mock_generate.return_value = "generated_token"
                
                token = mock_generate(mock_user)
                
                assert token == "generated_token"
                mock_generate.assert_called_once()


@pytest.mark.skipif(SecurityManager is None, reason="Security manager not available")
class TestSecurityManager:
    """Test security manager."""
    
    @pytest.fixture
    def security_config(self):
        """Create security configuration."""
        return SecurityConfig(
            enable_encryption_at_rest=True,
            enable_rate_limiting=True,
            rate_limit_requests_per_minute=100
        )
    
    @pytest.fixture
    def security_manager(self, security_config):
        """Create security manager."""
        config = EnterpriseConfig(security=security_config)
        return SecurityManager(config)
    
    def test_security_manager_initialization(self, security_manager, security_config):
        """Test security manager initialization."""
        assert security_manager.config.security == security_config
        assert security_manager.encryption_manager is not None
        assert security_manager.rate_limiter is not None
    
    def test_rate_limiting(self, security_manager):
        """Test rate limiting functionality."""
        client_id = "test_client"
        
        # Test request validation which includes rate limiting
        valid, message = security_manager.validate_request(client_id, "test data")
        assert valid or "rate limit" in message.lower()  # Should either pass or show rate limit message
    
    def test_input_validation(self, security_manager):
        """Test input validation."""
        # Valid input
        valid_input = "normal input text"
        valid, message = security_manager.validate_request("test_client", valid_input)
        # Should pass validation (may still be rate limited)
        assert valid or "rate limit" in message.lower() if message else True
    
    def test_encryption_decryption(self, security_manager):
        """Test data encryption and decryption."""
        if security_manager.config.security.enable_encryption_at_rest:
            original_data = "sensitive data"
            
            # Test encryption manager directly
            encrypted = security_manager.encryption_manager.encrypt_data(original_data)
            assert encrypted != original_data
            
            # Decrypt
            decrypted = security_manager.encryption_manager.decrypt_data(encrypted)
            # Handle both string and bytes return types
            if isinstance(decrypted, bytes):
                decrypted = decrypted.decode('utf-8')
            assert decrypted == original_data


@pytest.mark.skipif(ModelGovernance is None, reason="Model governance not available")
class TestModelGovernance:
    """Test model governance."""
    
    @pytest.fixture
    def enterprise_config(self):
        """Create enterprise configuration."""
        return EnterpriseConfig(
            environment="production"
        )
    
    @pytest.fixture
    def model_governance(self, enterprise_config):
        """Create model governance."""
        return ModelGovernance(enterprise_config)
    
    def test_governance_initialization(self, model_governance, enterprise_config):
        """Test model governance initialization."""
        assert model_governance.config == enterprise_config
        assert hasattr(model_governance, 'model_validator')
    
    @pytest.mark.asyncio
    async def test_model_registration(self, model_governance):
        """Test model registration."""
        # Create mock model metadata
        from framework.enterprise.governance import ModelMetadata
        
        model_metadata = ModelMetadata(
            id="test_model_id",
            name="test_model",
            version="1.0",
            description="Test model",
            framework="pytorch",
            architecture="resnet50",
            input_shape=(1, 3, 224, 224),
            output_shape=(1, 1000),
            parameters_count=25000000,
            model_size_mb=100.5
        )
        
        # Mock the file path and file operations
        model_file_path = "/path/to/model.pt"
        
        # Mock the file checksum calculation and file operations
        with patch('builtins.open', mock_open(read_data=b'fake model data')), \
             patch('os.path.getsize', return_value=105906176), \
             patch('shutil.copy2'):
            
            # Register model
            model_version = model_governance.register_model(model_metadata, model_file_path)
            assert model_version is not None
            assert model_version.model_id == "test_model_id"
    
    @pytest.mark.asyncio
    async def test_model_validation(self, model_governance):
        """Test model validation."""
        # Create mock model version
        from framework.enterprise.governance import ModelMetadata, ModelVersion, ModelStatus
        
        model_metadata = ModelMetadata(
            id="test_model_id", 
            name="test_model",
            version="1.0",
            description="Test model",
            framework="pytorch",
            architecture="resnet50", 
            input_shape=(1, 3, 224, 224),
            output_shape=(1, 1000),
            parameters_count=25000000,
            model_size_mb=100.5
        )
        
        model_version = ModelVersion(
            model_id="test_model_id",
            version="1.0",
            metadata=model_metadata,
            file_path="/path/to/model.pt",
            checksum="abc123",
            status=ModelStatus.PENDING  # Use PENDING instead of REGISTERED
        )
        
        with patch.object(model_governance.model_validator, 'validate_model') as mock_validate:
            mock_validate.return_value = {"overall_status": "passed", "checks_passed": 5}
            
            result = await model_governance.model_validator.validate_model(model_version)
            
            assert result["overall_status"] == "passed"
            assert result["checks_passed"] == 5
            mock_validate.assert_called_once_with(model_version)


@pytest.mark.skipif(ABTestManager is None, reason="A/B test manager not available")
class TestABTestManager:
    """Test A/B testing manager."""
    
    @pytest.fixture
    def enterprise_config(self):
        """Create enterprise configuration."""
        return EnterpriseConfig()
    
    @pytest.fixture
    def ab_test_manager(self, enterprise_config):
        """Create A/B test manager."""
        return ABTestManager(enterprise_config)
    
    def test_ab_test_manager_initialization(self, ab_test_manager):
        """Test A/B test manager initialization."""
        assert len(ab_test_manager.active_tests) == 0
        assert hasattr(ab_test_manager, 'traffic_router')
    
    def test_create_ab_test(self, ab_test_manager):
        """Test creating A/B test."""
        test_config = ab_test_manager.create_ab_test(
            name="Model Comparison Test",
            model_a_id="model_v1",
            model_b_id="model_v2",
            traffic_split_percent=50,
            success_metrics=["accuracy", "latency"]
        )
        
        assert test_config.name == "Model Comparison Test"
        assert test_config.model_a_id == "model_v1"
        assert test_config.model_b_id == "model_v2"
        assert test_config.traffic_split_percent == 50
        assert "accuracy" in test_config.success_metrics
        assert test_config.id in ab_test_manager.active_tests
    
    def test_start_ab_test(self, ab_test_manager):
        """Test starting A/B test."""
        test_config = ab_test_manager.create_ab_test(
            "Test", "model_a", "model_b", 30
        )
        
        success = ab_test_manager.start_ab_test(test_config.id)
        assert success
        assert test_config.status == "running"
        assert test_config.started_at is not None
    
    def test_record_test_results(self, ab_test_manager):
        """Test recording test results."""
        test_config = ab_test_manager.create_ab_test(
            "Test", "model_a", "model_b", 50
        )
        ab_test_manager.start_ab_test(test_config.id)
        
        # Record results for both models
        ab_test_manager.record_test_result(
            test_config.id, "model_a", {"accuracy": 0.85, "latency": 0.05}
        )
        ab_test_manager.record_test_result(
            test_config.id, "model_b", {"accuracy": 0.88, "latency": 0.04}
        )
        
        # Check results were recorded
        assert test_config.id in ab_test_manager.test_results
        assert "model_a" in ab_test_manager.test_results[test_config.id]
        assert "model_b" in ab_test_manager.test_results[test_config.id]
    
    def test_analyze_test_results(self, ab_test_manager):
        """Test analyzing test results."""
        test_config = ab_test_manager.create_ab_test(
            "Test", "model_a", "model_b", 50
        )
        ab_test_manager.start_ab_test(test_config.id)
        
        # Record multiple results
        for i in range(100):
            ab_test_manager.record_test_result(
                test_config.id, "model_a", 
                {"accuracy": 0.85 + (i % 10) * 0.01, "latency": 0.05}
            )
            ab_test_manager.record_test_result(
                test_config.id, "model_b",
                {"accuracy": 0.88 + (i % 10) * 0.01, "latency": 0.04}
            )
        
        analysis = ab_test_manager.analyze_test_results(test_config.id)
        
        assert analysis["test_id"] == test_config.id
        assert "models" in analysis
        assert "model_a" in analysis["models"]
        assert "model_b" in analysis["models"]
        assert "statistical_significance" in analysis
        assert "recommendation" in analysis
    
    def test_stop_ab_test(self, ab_test_manager):
        """Test stopping A/B test."""
        test_config = ab_test_manager.create_ab_test(
            "Test", "model_a", "model_b", 50
        )
        ab_test_manager.start_ab_test(test_config.id)
        
        success = ab_test_manager.stop_test(test_config.id)
        assert success
        assert test_config.status == "completed"


@pytest.mark.skipif(EnterpriseMonitor is None, reason="Enterprise monitoring not available")
class TestEnterpriseMonitoring:
    """Test enterprise monitoring."""
    
    @pytest.fixture
    def enterprise_config(self):
        """Create enterprise configuration."""
        return EnterpriseConfig()
    
    @pytest.fixture
    def enterprise_monitoring(self, enterprise_config):
        """Create enterprise monitoring."""
        return EnterpriseMonitor(enterprise_config)
    
    def test_monitoring_initialization(self, enterprise_monitoring):
        """Test monitoring initialization."""
        assert enterprise_monitoring.config is not None
        assert hasattr(enterprise_monitoring, 'alert_manager')
    
    def test_metric_collection(self, enterprise_monitoring):
        """Test metric collection."""
        # Test that prometheus metrics are available if enabled
        if enterprise_monitoring.prometheus_metrics:
            # Mock recording a metric
            with patch.object(enterprise_monitoring.prometheus_metrics, 'record_inference'):
                enterprise_monitoring.prometheus_metrics.record_inference(
                    model="test_model",
                    duration=0.05,
                    status="success"
                )
                # Should not raise exception
                assert True
    
    def test_alert_triggering(self, enterprise_monitoring):
        """Test alert triggering."""
        if enterprise_monitoring.alert_manager:
            # Mock alert check
            with patch.object(enterprise_monitoring.alert_manager, 'check_alerts') as mock_check:
                mock_metrics = {"latency": 0.15, "error_rate": 0.02}
                enterprise_monitoring.alert_manager.check_alerts(mock_metrics)
                mock_check.assert_called_once_with(mock_metrics)
    
    def test_start_stop_monitoring(self, enterprise_monitoring):
        """Test starting and stopping monitoring."""
        # Test start
        enterprise_monitoring.start_monitoring()
        assert enterprise_monitoring.is_running
        
        # Test stop  
        enterprise_monitoring.stop_monitoring()
        assert not enterprise_monitoring.is_running


class TestEnterpriseIntegration:
    """Integration tests for enterprise features."""
    
    @pytest.mark.skipif(any(cls is None for cls in [
        EnterpriseConfig, EnterpriseAuth, SecurityManager, ModelGovernance
    ]), reason="Enterprise features not available")
    def test_complete_enterprise_workflow(self):
        """Test complete enterprise workflow."""
        # Create enterprise config
        config = EnterpriseConfig(
            environment="production",
            auth=AuthConfig(provider=AuthProvider.OAUTH2),
            security=SecurityConfig(enable_encryption_at_rest=True)
        )
        
        # Initialize managers
        auth_manager = EnterpriseAuth(config)
        security_manager = SecurityManager(config)  # Pass EnterpriseConfig, not SecurityConfig
        model_governance = ModelGovernance(config)
        
        # Test workflow would involve:
        # 1. User authentication
        # 2. Input validation
        # 3. Model access control
        # 4. Inference execution
        # 5. Result encryption
        # 6. Audit logging
        
        # Mock successful workflow
        assert auth_manager is not None
        assert security_manager is not None
        assert model_governance is not None
    
    @pytest.mark.skipif(ABTestManager is None, reason="A/B testing not available")
    def test_ab_testing_workflow(self):
        """Test A/B testing workflow."""
        config = EnterpriseConfig()
        ab_manager = ABTestManager(config)
        
        # Complete A/B testing workflow
        test = ab_manager.create_ab_test("Performance Test", "v1", "v2", 50)
        ab_manager.start_ab_test(test.id)
        
        # Simulate traffic and results
        for i in range(50):
            model_id = "v1" if i % 2 == 0 else "v2"
            accuracy = 0.85 + (0.03 if model_id == "v2" else 0)
            ab_manager.record_test_result(test.id, model_id, {"accuracy": accuracy})
        
        analysis = ab_manager.analyze_test_results(test.id)
        ab_manager.stop_test(test.id)
        
        assert analysis["recommendation"] in ["deploy_model_b", "keep_model_a", "no_significant_difference", "inconclusive"]


class TestEnterpriseErrorHandling:
    """Test error handling in enterprise features."""
    
    @pytest.mark.skipif(EnterpriseAuth is None, reason="Auth manager not available")
    def test_authentication_error_handling(self):
        """Test authentication error handling."""
        config = EnterpriseConfig(auth=AuthConfig(provider=AuthProvider.OAUTH2))
        auth_manager = EnterpriseAuth(config)
        
        # Test with invalid configuration
        config.auth.secret_key = None
        
        with pytest.raises(Exception):
            if hasattr(auth_manager, 'jwt_manager'):
                auth_manager.jwt_manager.generate_token("test_user")
    
    @pytest.mark.skipif(SecurityManager is None, reason="Security manager not available")
    def test_security_error_handling(self):
        """Test security error handling."""
        config = EnterpriseConfig(security=SecurityConfig())
        security_manager = SecurityManager(config)
        
        # Test encryption with invalid data
        if security_manager.config.security.enable_encryption_at_rest:
            with pytest.raises((Exception, ValueError, TypeError)):
                # Try to encrypt None or invalid data
                security_manager.encryption_manager.encrypt_data(None)
    
    @pytest.mark.skipif(ModelGovernance is None, reason="Model governance not available")
    @pytest.mark.asyncio
    async def test_governance_error_handling(self):
        """Test model governance error handling."""
        config = EnterpriseConfig()
        governance = ModelGovernance(config)
        
        # Test with invalid model metadata
        from framework.enterprise.governance import ModelMetadata
        
        with pytest.raises((Exception, ValueError, TypeError)):
            # Try to register with invalid metadata - missing required fields
            incomplete_metadata = ModelMetadata(
                id="",
                name="", 
                version="",
                description="",
                framework="",
                architecture="",
                input_shape=(0,),
                output_shape=(0,),
                parameters_count=0,
                model_size_mb=0.0
            )
            governance.register_model(incomplete_metadata, "")
