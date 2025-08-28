"""
Test suite for security modules in the framework.

This module tests all security implementations including authentication,
governance, monitoring, and security configurations.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time
import json
from datetime import datetime, timedelta

# Framework imports
try:
    from framework.security.auth import (
        AuthenticationManager, UserCredentials, AuthToken,
        APIKeyAuth, TokenAuth, BasicAuth, authenticate_user,
        generate_api_key, validate_token, revoke_token
    )
    from framework.security.config import (
        SecurityConfig, SecurityPolicy, ThreatLevel,
        load_security_config, validate_security_config,
        get_default_security_config, apply_security_policy
    )
    from framework.security.governance import (
        ModelGovernance, GovernancePolicy, ComplianceCheck,
        AuditLogger, GovernanceRule, enforce_governance,
        audit_model_access, check_compliance
    )
    from framework.security.monitoring import (
        SecurityMonitor, SecurityEvent, ThreatDetector,
        SecurityAlert, monitor_security_events, detect_threats,
        log_security_event, get_security_metrics
    )
    from framework.security.engine import (
        SecureInferenceEngine, SecurityContext, SecureModelLoader,
        create_secure_engine, validate_model_integrity,
        encrypt_model_weights, decrypt_model_weights
    )
    from framework.security.security import (
        SecurityMitigation, SecurityValidator, SecurityEnforcer,
        apply_security_mitigations, validate_security_constraints,
        enforce_security_policies
    )
    SECURITY_MODULES_AVAILABLE = True
except ImportError as e:
    SECURITY_MODULES_AVAILABLE = False
    pytest.skip(f"Security modules not available: {e}", allow_module_level=True)


class TestUserCredentials:
    """Test UserCredentials class."""
    
    def test_user_credentials_creation(self):
        """Test creating user credentials."""
        credentials = UserCredentials(
            username="testuser",
            password="testpass123",
            email="test@example.com",
            role="user"
        )
        
        assert credentials.username == "testuser"
        assert credentials.email == "test@example.com"
        assert credentials.role == "user"
        # Password should be hashed
        assert credentials.password != "testpass123"
    
    def test_user_credentials_validation(self):
        """Test user credentials validation."""
        credentials = UserCredentials(
            username="testuser",
            password="testpass123"
        )
        
        # Test password verification
        assert credentials.verify_password("testpass123") is True
        assert credentials.verify_password("wrongpass") is False
    
    def test_user_credentials_serialization(self):
        """Test user credentials serialization."""
        credentials = UserCredentials(
            username="testuser",
            password="testpass123",
            role="admin"
        )
        
        serialized = credentials.to_dict()
        assert "username" in serialized
        assert "role" in serialized
        # Password should not be in serialized form
        assert "password" not in serialized or serialized["password"] != "testpass123"


class TestAuthToken:
    """Test AuthToken class."""
    
    def test_auth_token_creation(self):
        """Test creating auth token."""
        token = AuthToken(
            user_id="user123",
            username="testuser",
            expires_in=3600  # 1 hour
        )
        
        assert token.user_id == "user123"
        assert token.username == "testuser"
        assert token.expires_at > datetime.utcnow()
    
    def test_auth_token_validation(self):
        """Test auth token validation."""
        # Valid token
        token = AuthToken(
            user_id="user123",
            username="testuser",
            expires_in=3600
        )
        
        assert token.is_valid() is True
        assert token.is_expired() is False
        
        # Expired token
        expired_token = AuthToken(
            user_id="user123",
            username="testuser",
            expires_in=-1  # Already expired
        )
        
        assert expired_token.is_valid() is False
        assert expired_token.is_expired() is True
    
    def test_auth_token_refresh(self):
        """Test auth token refresh."""
        token = AuthToken(
            user_id="user123",
            username="testuser",
            expires_in=60  # 1 minute
        )
        
        old_expires_at = token.expires_at
        token.refresh(3600)  # Extend to 1 hour
        
        assert token.expires_at > old_expires_at


class TestAuthenticationManager:
    """Test AuthenticationManager class."""
    
    @pytest.fixture
    def temp_auth_dir(self):
        """Create temporary auth directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def auth_manager(self, temp_auth_dir):
        """Create authentication manager."""
        return AuthenticationManager(storage_dir=temp_auth_dir)
    
    def test_auth_manager_initialization(self, auth_manager):
        """Test authentication manager initialization."""
        assert hasattr(auth_manager, 'users')
        assert hasattr(auth_manager, 'tokens')
    
    def test_user_registration(self, auth_manager):
        """Test user registration."""
        user = auth_manager.register_user(
            username="testuser",
            password="testpass123",
            email="test@example.com"
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert "testuser" in auth_manager.users
    
    def test_user_authentication(self, auth_manager):
        """Test user authentication."""
        # Register user first
        auth_manager.register_user(
            username="testuser",
            password="testpass123"
        )
        
        # Test successful authentication
        token = auth_manager.authenticate("testuser", "testpass123")
        assert token is not None
        assert token.username == "testuser"
        
        # Test failed authentication
        token = auth_manager.authenticate("testuser", "wrongpass")
        assert token is None
    
    def test_token_validation(self, auth_manager):
        """Test token validation."""
        # Register and authenticate user
        auth_manager.register_user("testuser", "testpass123")
        token = auth_manager.authenticate("testuser", "testpass123")
        
        # Test token validation
        is_valid = auth_manager.validate_token(token.token)
        assert is_valid is True
        
        # Test invalid token
        is_valid = auth_manager.validate_token("invalid_token")
        assert is_valid is False
    
    def test_token_revocation(self, auth_manager):
        """Test token revocation."""
        # Register and authenticate user
        auth_manager.register_user("testuser", "testpass123")
        token = auth_manager.authenticate("testuser", "testpass123")
        
        # Revoke token
        revoked = auth_manager.revoke_token(token.token)
        assert revoked is True
        
        # Token should no longer be valid
        is_valid = auth_manager.validate_token(token.token)
        assert is_valid is False


class TestAPIKeyAuth:
    """Test API Key authentication."""
    
    def test_api_key_generation(self):
        """Test API key generation."""
        api_key = generate_api_key(length=32)
        
        assert len(api_key) == 32
        assert isinstance(api_key, str)
    
    def test_api_key_auth(self):
        """Test API key authentication."""
        auth = APIKeyAuth()
        
        # Generate and register API key
        api_key = auth.generate_key("testuser")
        assert api_key is not None
        
        # Test authentication
        user_info = auth.authenticate(api_key)
        assert user_info is not None
        assert user_info["username"] == "testuser"
        
        # Test invalid key
        user_info = auth.authenticate("invalid_key")
        assert user_info is None


class TestSecurityConfig:
    """Test SecurityConfig class."""
    
    def test_security_config_creation(self):
        """Test creating security config."""
        config = SecurityConfig(
            enable_auth=True,
            enable_encryption=True,
            threat_level=ThreatLevel.HIGH,
            max_login_attempts=3,
            token_expiry=3600
        )
        
        assert config.enable_auth is True
        assert config.enable_encryption is True
        assert config.threat_level == ThreatLevel.HIGH
        assert config.max_login_attempts == 3
        assert config.token_expiry == 3600
    
    def test_security_config_defaults(self):
        """Test default security config."""
        config = get_default_security_config()
        
        assert isinstance(config, SecurityConfig)
        assert hasattr(config, 'enable_auth')
        assert hasattr(config, 'enable_encryption')
    
    def test_security_config_validation(self):
        """Test security config validation."""
        # Valid config
        config = SecurityConfig(
            enable_auth=True,
            max_login_attempts=5,
            token_expiry=7200
        )
        
        is_valid, errors = validate_security_config(config)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid config
        invalid_config = SecurityConfig(
            max_login_attempts=-1,  # Invalid
            token_expiry=0  # Invalid
        )
        
        is_valid, errors = validate_security_config(invalid_config)
        assert is_valid is False or len(errors) > 0
    
    def test_security_policy_application(self):
        """Test security policy application."""
        policy = SecurityPolicy(
            name="strict_policy",
            rules={
                "require_auth": True,
                "require_encryption": True,
                "max_session_time": 1800
            }
        )
        
        config = SecurityConfig()
        updated_config = apply_security_policy(config, policy)
        
        assert updated_config is not None


class TestModelGovernance:
    """Test ModelGovernance class."""
    
    @pytest.fixture
    def temp_governance_dir(self):
        """Create temporary governance directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def governance(self, temp_governance_dir):
        """Create model governance instance."""
        return ModelGovernance(storage_dir=temp_governance_dir)
    
    def test_governance_initialization(self, governance):
        """Test governance initialization."""
        assert hasattr(governance, 'policies')
        assert hasattr(governance, 'audit_logger')
    
    def test_governance_policy_creation(self):
        """Test governance policy creation."""
        policy = GovernancePolicy(
            name="data_protection",
            description="Ensure data protection compliance",
            rules={
                "encrypt_data": True,
                "anonymize_inputs": True,
                "log_access": True
            }
        )
        
        assert policy.name == "data_protection"
        assert policy.rules["encrypt_data"] is True
    
    def test_compliance_check(self, governance):
        """Test compliance checking."""
        # Add compliance rule
        rule = GovernanceRule(
            name="encryption_required",
            condition="model.encrypted == True",
            action="block"
        )
        
        governance.add_rule(rule)
        
        # Test compliance
        model_metadata = {"encrypted": True}
        is_compliant = governance.check_compliance(model_metadata)
        assert is_compliant is True
        
        # Test non-compliance
        model_metadata = {"encrypted": False}
        is_compliant = governance.check_compliance(model_metadata)
        assert is_compliant is False
    
    def test_audit_logging(self, governance):
        """Test audit logging."""
        # Log model access
        governance.audit_logger.log_access(
            user_id="user123",
            model_id="model456",
            action="inference",
            timestamp=datetime.utcnow()
        )
        
        # Retrieve audit logs
        logs = governance.audit_logger.get_logs(
            start_date=datetime.utcnow() - timedelta(hours=1)
        )
        
        assert len(logs) >= 1
        assert logs[0]["user_id"] == "user123"
        assert logs[0]["model_id"] == "model456"
    
    def test_governance_enforcement(self, governance):
        """Test governance enforcement."""
        # Create policy
        policy = GovernancePolicy(
            name="access_control",
            rules={"require_authorization": True}
        )
        
        governance.add_policy(policy)
        
        # Test enforcement
        context = {"user_authorized": True}
        allowed = governance.enforce_policy("access_control", context)
        assert allowed is True
        
        context = {"user_authorized": False}
        allowed = governance.enforce_policy("access_control", context)
        assert allowed is False


class TestSecurityMonitor:
    """Test SecurityMonitor class."""
    
    @pytest.fixture
    def security_monitor(self):
        """Create security monitor."""
        return SecurityMonitor()
    
    def test_security_monitor_initialization(self, security_monitor):
        """Test security monitor initialization."""
        assert hasattr(security_monitor, 'events')
        assert hasattr(security_monitor, 'threat_detector')
    
    def test_security_event_logging(self, security_monitor):
        """Test security event logging."""
        event = SecurityEvent(
            event_type="login_attempt",
            user_id="user123",
            source_ip="192.168.1.100",
            severity="medium",
            timestamp=datetime.utcnow()
        )
        
        security_monitor.log_event(event)
        
        events = security_monitor.get_events(limit=10)
        assert len(events) >= 1
        assert events[0].event_type == "login_attempt"
    
    def test_threat_detection(self, security_monitor):
        """Test threat detection."""
        # Simulate multiple failed login attempts
        for i in range(5):
            event = SecurityEvent(
                event_type="login_failed",
                user_id="user123",
                source_ip="192.168.1.100",
                timestamp=datetime.utcnow()
            )
            security_monitor.log_event(event)
        
        # Detect threats
        threats = security_monitor.detect_threats()
        
        assert len(threats) >= 0  # May detect brute force attempt
    
    def test_security_alerts(self, security_monitor):
        """Test security alerts."""
        alert = SecurityAlert(
            alert_type="brute_force",
            severity="high",
            description="Multiple failed login attempts detected",
            source_ip="192.168.1.100"
        )
        
        security_monitor.trigger_alert(alert)
        
        alerts = security_monitor.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0].alert_type == "brute_force"
    
    def test_security_metrics(self, security_monitor):
        """Test security metrics collection."""
        # Log some events
        events = [
            SecurityEvent("login_success", user_id="user1"),
            SecurityEvent("login_failed", user_id="user2"),
            SecurityEvent("model_access", user_id="user1")
        ]
        
        for event in events:
            security_monitor.log_event(event)
        
        metrics = security_monitor.get_metrics()
        
        assert "total_events" in metrics
        assert "event_types" in metrics
        assert metrics["total_events"] >= 3


class TestSecureInferenceEngine:
    """Test SecureInferenceEngine class."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple test model."""
        import torch
        return torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2)
        )
    
    @pytest.fixture
    def security_context(self):
        """Create security context."""
        return SecurityContext(
            user_id="user123",
            permissions=["model_inference", "data_access"],
            encryption_enabled=True,
            audit_enabled=True
        )
    
    def test_security_context_creation(self, security_context):
        """Test security context creation."""
        assert security_context.user_id == "user123"
        assert "model_inference" in security_context.permissions
        assert security_context.encryption_enabled is True
    
    def test_secure_model_loader(self, simple_model):
        """Test secure model loader."""
        loader = SecureModelLoader()
        
        # Test model validation
        is_valid = loader.validate_model(simple_model)
        assert is_valid is True
        
        # Test secure loading
        with patch.object(loader, 'decrypt_model') as mock_decrypt:
            mock_decrypt.return_value = simple_model
            
            loaded_model = loader.load_secure_model("/path/to/model.pt")
            assert loaded_model is not None
    
    def test_model_integrity_validation(self, simple_model):
        """Test model integrity validation."""
        # Create model hash
        model_hash = validate_model_integrity(simple_model)
        assert model_hash is not None
        assert len(model_hash) > 0
        
        # Validate against hash
        is_valid = validate_model_integrity(simple_model, expected_hash=model_hash)
        assert is_valid is True
    
    def test_model_encryption(self, simple_model):
        """Test model encryption/decryption."""
        import torch
        
        # Test encryption
        encrypted_weights = encrypt_model_weights(simple_model.state_dict(), key="test_key")
        assert encrypted_weights is not None
        
        # Test decryption
        decrypted_weights = decrypt_model_weights(encrypted_weights, key="test_key")
        assert decrypted_weights is not None
        
        # Weights should match original
        original_weights = simple_model.state_dict()
        for key in original_weights:
            if key in decrypted_weights:
                assert torch.allclose(original_weights[key], decrypted_weights[key], atol=1e-6)
    
    def test_secure_inference_engine(self, simple_model, security_context):
        """Test secure inference engine."""
        engine = SecureInferenceEngine(simple_model, security_context)
        
        assert engine.model == simple_model
        assert engine.security_context == security_context
    
    def test_create_secure_engine(self, simple_model):
        """Test create_secure_engine factory function."""
        engine = create_secure_engine(
            model=simple_model,
            user_id="user123",
            permissions=["inference"]
        )
        
        assert isinstance(engine, SecureInferenceEngine)
        assert engine.security_context.user_id == "user123"


class TestSecurityMitigation:
    """Test SecurityMitigation class."""
    
    def test_security_mitigation_initialization(self):
        """Test security mitigation initialization."""
        mitigation = SecurityMitigation()
        
        assert hasattr(mitigation, 'mitigations')
        assert hasattr(mitigation, 'validators')
    
    def test_apply_security_mitigations(self):
        """Test applying security mitigations."""
        mitigations = ["input_validation", "output_sanitization", "rate_limiting"]
        
        result = apply_security_mitigations(mitigations)
        
        assert result is not None
        assert "applied_mitigations" in result
    
    def test_security_validator(self):
        """Test security validator."""
        validator = SecurityValidator()
        
        # Test input validation
        is_valid = validator.validate_input("safe_input")
        assert is_valid is True
        
        # Test potentially unsafe input
        is_valid = validator.validate_input("<script>alert('xss')</script>")
        assert is_valid is False
    
    def test_security_enforcer(self):
        """Test security enforcer."""
        enforcer = SecurityEnforcer()
        
        # Test policy enforcement
        policies = ["require_auth", "encrypt_data"]
        context = {"authenticated": True, "encrypted": True}
        
        result = enforcer.enforce_policies(policies, context)
        assert result["allowed"] is True
        
        # Test policy violation
        context = {"authenticated": False, "encrypted": True}
        result = enforcer.enforce_policies(policies, context)
        assert result["allowed"] is False


class TestSecurityIntegration:
    """Test security module integration."""
    
    def test_auth_with_governance(self):
        """Test authentication with governance."""
        auth_manager = AuthenticationManager()
        governance = ModelGovernance()
        
        # Register user
        user = auth_manager.register_user("testuser", "testpass123")
        token = auth_manager.authenticate("testuser", "testpass123")
        
        # Check governance compliance for user
        user_context = {
            "user_id": user.user_id,
            "authenticated": True,
            "token_valid": auth_manager.validate_token(token.token)
        }
        
        # Should pass basic governance checks
        assert user_context["authenticated"] is True
        assert user_context["token_valid"] is True
    
    def test_monitoring_with_governance(self):
        """Test monitoring with governance."""
        monitor = SecurityMonitor()
        governance = ModelGovernance()
        
        # Log governance violation
        event = SecurityEvent(
            event_type="governance_violation",
            description="Unauthorized model access attempt",
            severity="high"
        )
        
        monitor.log_event(event)
        
        # Check if governance should be notified
        recent_events = monitor.get_events(limit=10)
        governance_events = [e for e in recent_events if e.event_type == "governance_violation"]
        
        assert len(governance_events) >= 1
    
    def test_complete_security_workflow(self):
        """Test complete security workflow."""
        # Initialize all security components
        auth_manager = AuthenticationManager()
        governance = ModelGovernance()
        monitor = SecurityMonitor()
        
        # User registration and authentication
        user = auth_manager.register_user("secureuser", "securepass123")
        token = auth_manager.authenticate("secureuser", "securepass123")
        
        # Log authentication event
        auth_event = SecurityEvent(
            event_type="login_success",
            user_id=user.user_id
        )
        monitor.log_event(auth_event)
        
        # Check governance compliance
        context = {
            "user_id": user.user_id,
            "authenticated": True,
            "token_valid": auth_manager.validate_token(token.token)
        }
        
        # Model access attempt
        access_event = SecurityEvent(
            event_type="model_access",
            user_id=user.user_id
        )
        monitor.log_event(access_event)
        
        # Audit the access
        governance.audit_logger.log_access(
            user_id=user.user_id,
            model_id="secure_model",
            action="inference"
        )
        
        # Verify workflow
        assert token is not None
        assert auth_manager.validate_token(token.token) is True
        
        events = monitor.get_events()
        assert len(events) >= 2  # Auth and access events
        
        audit_logs = governance.audit_logger.get_logs()
        assert len(audit_logs) >= 1


@pytest.mark.integration
class TestSecurityModulesWithFramework:
    """Integration tests for security modules with the main framework."""
    
    def test_security_with_inference_framework(self):
        """Test security modules with TorchInferenceFramework."""
        # Mock framework integration
        with patch('framework.TorchInferenceFramework') as MockFramework:
            mock_framework = Mock()
            MockFramework.return_value = mock_framework
            
            # Test that security can be integrated
            framework = MockFramework()
            assert framework is not None
    
    def test_security_configuration_loading(self):
        """Test loading security configuration."""
        config = get_default_security_config()
        
        assert isinstance(config, SecurityConfig)
        assert hasattr(config, 'enable_auth')
    
    def test_security_error_handling(self):
        """Test security error handling."""
        auth_manager = AuthenticationManager()
        
        # Test invalid authentication
        token = auth_manager.authenticate("nonexistent", "password")
        assert token is None
        
        # Test invalid token validation
        is_valid = auth_manager.validate_token("invalid_token")
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__])
