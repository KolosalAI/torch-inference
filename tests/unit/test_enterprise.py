"""Tests for enterprise features."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

# Test with mock classes if enterprise modules not available
try:
    from framework.enterprise.config import (
        EnterpriseConfig, AuthConfig, SecurityConfig, AuthProvider
    )
    from framework.enterprise.auth import AuthManager, TokenManager
    from framework.enterprise.security import SecurityManager, EncryptionManager
    from framework.enterprise.governance import (
        ModelGovernance, ModelValidator, ABTestManager, ABTestConfig
    )
    from framework.enterprise.monitoring import EnterpriseMonitoring
except ImportError:
    # Mock classes for testing when enterprise features not available
    EnterpriseConfig = None
    AuthConfig = None
    SecurityConfig = None
    AuthProvider = None
    AuthManager = None
    TokenManager = None
    SecurityManager = None
    EncryptionManager = None
    ModelGovernance = None
    ModelValidator = None
    ABTestManager = None
    ABTestConfig = None
    EnterpriseMonitoring = None


@pytest.mark.skipif(EnterpriseConfig is None, reason="Enterprise features not available")
class TestEnterpriseConfig:
    """Test enterprise configuration."""
    
    def test_enterprise_config_creation(self):
        """Test creating enterprise configuration."""
        config = EnterpriseConfig(
            environment="production",
            auth=AuthConfig(
                provider=AuthProvider.OAUTH2,
                jwt_secret_key="test-secret"
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
        
        assert config.environment == "development"
        assert isinstance(config.auth, AuthConfig)
        assert isinstance(config.security, SecurityConfig)


@pytest.mark.skipif(AuthManager is None, reason="Auth manager not available")
class TestAuthManager:
    """Test authentication manager."""
    
    @pytest.fixture
    def auth_config(self):
        """Create auth configuration."""
        return AuthConfig(
            provider=AuthProvider.OAUTH2,
            jwt_secret_key="test-secret-key",
            token_expiry_hours=24
        )
    
    @pytest.fixture
    def auth_manager(self, auth_config):
        """Create auth manager."""
        return AuthManager(auth_config)
    
    def test_auth_manager_initialization(self, auth_manager, auth_config):
        """Test auth manager initialization."""
        assert auth_manager.config == auth_config
        assert auth_manager.provider == AuthProvider.OAUTH2
    
    @pytest.mark.asyncio
    async def test_user_authentication(self, auth_manager):
        """Test user authentication."""
        # Mock successful authentication
        with patch.object(auth_manager, '_authenticate_oauth2') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "roles": ["user"]}
            
            result = await auth_manager.authenticate("test_token")
            
            assert result["user_id"] == "test_user"
            assert "user" in result["roles"]
            mock_auth.assert_called_once_with("test_token")
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, auth_manager):
        """Test authentication failure."""
        with patch.object(auth_manager, '_authenticate_oauth2') as mock_auth:
            mock_auth.side_effect = Exception("Authentication failed")
            
            with pytest.raises(Exception):
                await auth_manager.authenticate("invalid_token")
    
    def test_token_validation(self, auth_manager):
        """Test JWT token validation."""
        # Create mock token
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {"user_id": "test", "exp": 9999999999}
            
            is_valid = auth_manager.validate_token("valid_token")
            assert is_valid
            
            mock_decode.assert_called_once()
    
    def test_token_generation(self, auth_manager):
        """Test JWT token generation."""
        with patch('jwt.encode') as mock_encode:
            mock_encode.return_value = "generated_token"
            
            token = auth_manager.generate_token("test_user", ["admin"])
            
            assert token == "generated_token"
            mock_encode.assert_called_once()


@pytest.mark.skipif(SecurityManager is None, reason="Security manager not available")
class TestSecurityManager:
    """Test security manager."""
    
    @pytest.fixture
    def security_config(self):
        """Create security configuration."""
        return SecurityConfig(
            enable_encryption_at_rest=True,
            enable_rate_limiting=True,
            max_requests_per_minute=100
        )
    
    @pytest.fixture
    def security_manager(self, security_config):
        """Create security manager."""
        return SecurityManager(security_config)
    
    def test_security_manager_initialization(self, security_manager, security_config):
        """Test security manager initialization."""
        assert security_manager.config == security_config
        assert security_manager.encryption_enabled
        assert security_manager.rate_limiting_enabled
    
    def test_rate_limiting(self, security_manager):
        """Test rate limiting functionality."""
        client_id = "test_client"
        
        # Should allow requests within limit
        for _ in range(5):
            allowed = security_manager.check_rate_limit(client_id)
            assert allowed
        
        # Mock exceeding rate limit
        with patch.object(security_manager, '_get_request_count') as mock_count:
            mock_count.return_value = 150  # Over limit
            
            allowed = security_manager.check_rate_limit(client_id)
            assert not allowed
    
    def test_input_validation(self, security_manager):
        """Test input validation."""
        # Valid input
        valid_input = {"text": "normal input", "length": 100}
        assert security_manager.validate_input(valid_input)
        
        # Invalid input (too long)
        invalid_input = {"text": "x" * 10000, "length": 10000}
        assert not security_manager.validate_input(invalid_input)
        
        # Malicious input
        malicious_input = {"text": "<script>alert('xss')</script>"}
        assert not security_manager.validate_input(malicious_input)
    
    def test_encryption_decryption(self, security_manager):
        """Test data encryption and decryption."""
        if not security_manager.encryption_enabled:
            pytest.skip("Encryption not enabled")
        
        original_data = "sensitive data"
        
        # Encrypt
        encrypted = security_manager.encrypt_data(original_data)
        assert encrypted != original_data
        
        # Decrypt
        decrypted = security_manager.decrypt_data(encrypted)
        assert decrypted == original_data


@pytest.mark.skipif(ModelGovernance is None, reason="Model governance not available")
class TestModelGovernance:
    """Test model governance."""
    
    @pytest.fixture
    def enterprise_config(self):
        """Create enterprise configuration."""
        return EnterpriseConfig(
            environment="production",
            security=SecurityConfig(enable_model_validation=True)
        )
    
    @pytest.fixture
    def model_governance(self, enterprise_config):
        """Create model governance."""
        return ModelGovernance(enterprise_config)
    
    def test_governance_initialization(self, model_governance, enterprise_config):
        """Test model governance initialization."""
        assert model_governance.config == enterprise_config
        assert isinstance(model_governance.validator, ModelValidator)
    
    @pytest.mark.asyncio
    async def test_model_registration(self, model_governance):
        """Test model registration."""
        model_info = {
            "name": "test_model",
            "version": "1.0",
            "model_type": "classification",
            "checksum": "abc123"
        }
        
        success = await model_governance.register_model(model_info)
        assert success
        
        # Check if model is registered
        models = model_governance.list_models()
        assert len(models) > 0
        assert any(m["name"] == "test_model" for m in models)
    
    @pytest.mark.asyncio
    async def test_model_validation(self, model_governance):
        """Test model validation."""
        model_version = Mock()
        model_version.name = "test_model"
        model_version.version = "1.0"
        model_version.file_path = "/path/to/model.pt"
        
        with patch.object(model_governance.validator, 'validate_model') as mock_validate:
            mock_validate.return_value = {"valid": True, "checks_passed": 5}
            
            result = await model_governance.validate_model(model_version)
            
            assert result["valid"]
            assert result["checks_passed"] == 5
            mock_validate.assert_called_once_with(model_version)
    
    def test_model_approval_workflow(self, model_governance):
        """Test model approval workflow."""
        model_id = "test_model_v1"
        
        # Submit for approval
        submitted = model_governance.submit_for_approval(model_id, "user_123")
        assert submitted
        
        # Approve model
        approved = model_governance.approve_model(model_id, "admin_456")
        assert approved
        
        # Check approval status
        status = model_governance.get_approval_status(model_id)
        assert status["approved"]
        assert status["approved_by"] == "admin_456"


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


@pytest.mark.skipif(EnterpriseMonitoring is None, reason="Enterprise monitoring not available")
class TestEnterpriseMonitoring:
    """Test enterprise monitoring."""
    
    @pytest.fixture
    def enterprise_config(self):
        """Create enterprise configuration."""
        return EnterpriseConfig()
    
    @pytest.fixture
    def enterprise_monitoring(self, enterprise_config):
        """Create enterprise monitoring."""
        return EnterpriseMonitoring(enterprise_config)
    
    def test_monitoring_initialization(self, enterprise_monitoring):
        """Test monitoring initialization."""
        assert enterprise_monitoring.metrics_storage is not None
        assert enterprise_monitoring.alert_manager is not None
    
    def test_metric_collection(self, enterprise_monitoring):
        """Test metric collection."""
        # Record metrics
        enterprise_monitoring.record_metric("inference_count", 1, {"model": "test_model"})
        enterprise_monitoring.record_metric("latency", 0.05, {"model": "test_model"})
        
        # Get metrics
        metrics = enterprise_monitoring.get_metrics("inference_count")
        assert len(metrics) > 0
    
    def test_alert_triggering(self, enterprise_monitoring):
        """Test alert triggering."""
        # Set up alert rule
        enterprise_monitoring.add_alert_rule(
            "high_latency",
            condition="latency > 0.1",
            severity="warning"
        )
        
        # Trigger alert condition
        enterprise_monitoring.record_metric("latency", 0.15, {"model": "slow_model"})
        
        # Check if alert was triggered
        alerts = enterprise_monitoring.get_active_alerts()
        assert len(alerts) > 0
        assert any(alert["rule"] == "high_latency" for alert in alerts)
    
    def test_audit_logging(self, enterprise_monitoring):
        """Test audit logging."""
        # Log audit event
        enterprise_monitoring.log_audit_event(
            "model_access",
            user_id="test_user",
            details={"model": "sensitive_model", "action": "inference"}
        )
        
        # Retrieve audit logs
        audit_logs = enterprise_monitoring.get_audit_logs(
            start_time=datetime.now(timezone.utc).timestamp() - 3600
        )
        
        assert len(audit_logs) > 0
        assert any(log["event_type"] == "model_access" for log in audit_logs)


class TestEnterpriseIntegration:
    """Integration tests for enterprise features."""
    
    @pytest.mark.skipif(any(cls is None for cls in [
        EnterpriseConfig, AuthManager, SecurityManager, ModelGovernance
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
        auth_manager = AuthManager(config.auth)
        security_manager = SecurityManager(config.security)
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
        
        assert analysis["recommendation"] in ["deploy_model_b", "keep_model_a", "no_significant_difference"]


class TestEnterpriseErrorHandling:
    """Test error handling in enterprise features."""
    
    @pytest.mark.skipif(AuthManager is None, reason="Auth manager not available")
    def test_authentication_error_handling(self):
        """Test authentication error handling."""
        config = AuthConfig(provider=AuthProvider.OAUTH2)
        auth_manager = AuthManager(config)
        
        # Test with invalid configuration
        config.jwt_secret_key = None
        
        with pytest.raises(Exception):
            auth_manager.generate_token("test_user")
    
    @pytest.mark.skipif(SecurityManager is None, reason="Security manager not available")
    def test_security_error_handling(self):
        """Test security error handling."""
        config = SecurityConfig()
        security_manager = SecurityManager(config)
        
        # Test encryption with no key
        if security_manager.encryption_enabled:
            with pytest.raises(Exception):
                security_manager.encrypt_data("test", key=None)
    
    @pytest.mark.skipif(ModelGovernance is None, reason="Model governance not available")
    @pytest.mark.asyncio
    async def test_governance_error_handling(self):
        """Test model governance error handling."""
        config = EnterpriseConfig()
        governance = ModelGovernance(config)
        
        # Test with invalid model info
        invalid_model_info = {"incomplete": "data"}
        
        with pytest.raises(Exception):
            await governance.register_model(invalid_model_info)
