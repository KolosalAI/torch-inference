"""
Tests for Advanced Configuration Management implementation.
"""

import pytest
import json
import yaml
import os
import tempfile
import time
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from framework.config.advanced_config import (
    ConfigurationManager, FeatureFlagManager, SecretManager,
    ConfigurationError, SecretNotFoundError
)


class TestConfigurationManager:
    """Test configuration manager functionality."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_config_files(self, temp_config_dir):
        """Create sample configuration files."""
        # Base config
        base_config = {
            "app_name": "torch-inference",
            "debug": False,
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "inference_db"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "timeout": 30
            }
        }
        
        # Development overrides
        dev_config = {
            "debug": True,
            "database": {
                "host": "dev-db.example.com",
                "name": "inference_dev"
            },
            "api": {
                "port": 8001
            }
        }
        
        # Production overrides
        prod_config = {
            "database": {
                "host": "prod-db.example.com",
                "name": "inference_prod"
            },
            "api": {
                "workers": 4,
                "timeout": 60
            }
        }
        
        # Write config files
        base_file = temp_config_dir / "base.yaml"
        dev_file = temp_config_dir / "development.yaml"
        prod_file = temp_config_dir / "production.yaml"
        
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        with open(dev_file, 'w') as f:
            yaml.dump(dev_config, f)
        
        with open(prod_file, 'w') as f:
            yaml.dump(prod_config, f)
        
        return {
            "base": base_file,
            "development": dev_file,
            "production": prod_file
        }
    
    def test_config_manager_initialization(self, temp_config_dir):
        """Test configuration manager initialization."""
        config_manager = ConfigurationManager(str(temp_config_dir))
        
        assert config_manager.config_dir == str(temp_config_dir)
        assert config_manager.environment == "development"  # Default
        assert config_manager._config == {}
        assert config_manager._watchers == {}
    
    def test_load_configuration_development(self, temp_config_dir, sample_config_files):
        """Test loading development configuration."""
        config_manager = ConfigurationManager(str(temp_config_dir), environment="development")
        config_manager.load_configuration()
        
        # Should have base config merged with development overrides
        assert config_manager.get("app_name") == "torch-inference"
        assert config_manager.get("debug") is True  # Overridden in dev
        assert config_manager.get("database.host") == "dev-db.example.com"  # Overridden
        assert config_manager.get("database.port") == 5432  # From base
        assert config_manager.get("database.name") == "inference_dev"  # Overridden
        assert config_manager.get("api.port") == 8001  # Overridden in dev
        assert config_manager.get("api.timeout") == 30  # From base
    
    def test_load_configuration_production(self, temp_config_dir, sample_config_files):
        """Test loading production configuration."""
        config_manager = ConfigurationManager(str(temp_config_dir), environment="production")
        config_manager.load_configuration()
        
        # Should have base config merged with production overrides
        assert config_manager.get("debug") is False  # From base (not overridden)
        assert config_manager.get("database.host") == "prod-db.example.com"  # Overridden
        assert config_manager.get("database.name") == "inference_prod"  # Overridden
        assert config_manager.get("api.port") == 8000  # From base (not overridden)
        assert config_manager.get("api.workers") == 4  # Only in production
        assert config_manager.get("api.timeout") == 60  # Overridden in prod
    
    def test_get_configuration_values(self, temp_config_dir, sample_config_files):
        """Test getting configuration values."""
        config_manager = ConfigurationManager(str(temp_config_dir))
        config_manager.load_configuration()
        
        # Test simple key access
        assert config_manager.get("app_name") == "torch-inference"
        
        # Test nested key access with dot notation
        assert config_manager.get("database.host") == "dev-db.example.com"
        assert config_manager.get("api.port") == 8001
        
        # Test default values
        assert config_manager.get("nonexistent_key", "default") == "default"
        assert config_manager.get("nested.nonexistent", "default") == "default"
        
        # Test getting entire sections
        database_config = config_manager.get("database")
        assert isinstance(database_config, dict)
        assert database_config["host"] == "dev-db.example.com"
    
    def test_set_configuration_values(self, temp_config_dir, sample_config_files):
        """Test setting configuration values."""
        config_manager = ConfigurationManager(str(temp_config_dir))
        config_manager.load_configuration()
        
        # Set simple value
        config_manager.set("new_key", "new_value")
        assert config_manager.get("new_key") == "new_value"
        
        # Set nested value with dot notation
        config_manager.set("api.new_setting", "nested_value")
        assert config_manager.get("api.new_setting") == "nested_value"
        
        # Update existing value
        config_manager.set("debug", True)
        assert config_manager.get("debug") is True
    
    def test_environment_variable_substitution(self, temp_config_dir):
        """Test environment variable substitution."""
        # Create config with environment variables
        config_with_env = {
            "database": {
                "host": "${DB_HOST}",
                "port": "${DB_PORT:5432}",  # With default
                "user": "${DB_USER}",
                "password": "${DB_PASSWORD}"
            },
            "api": {
                "secret_key": "${SECRET_KEY}",
                "workers": "${WORKERS:2}"  # With default
            }
        }
        
        config_file = temp_config_dir / "base.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_with_env, f)
        
        # Set environment variables
        with patch.dict(os.environ, {
            'DB_HOST': 'env-db.example.com',
            'DB_USER': 'env_user',
            'DB_PASSWORD': 'env_password',
            'SECRET_KEY': 'env_secret_key'
            # DB_PORT and WORKERS not set, should use defaults
        }):
            config_manager = ConfigurationManager(str(temp_config_dir))
            config_manager.load_configuration()
            
            assert config_manager.get("database.host") == "env-db.example.com"
            assert config_manager.get("database.port") == 5432  # Default
            assert config_manager.get("database.user") == "env_user"
            assert config_manager.get("database.password") == "env_password"
            assert config_manager.get("api.secret_key") == "env_secret_key"
            assert config_manager.get("api.workers") == 2  # Default
    
    def test_missing_environment_variable(self, temp_config_dir):
        """Test handling of missing environment variables."""
        config_with_missing_env = {
            "required_value": "${MISSING_REQUIRED_VAR}",
            "optional_value": "${MISSING_OPTIONAL_VAR:default_value}"
        }
        
        config_file = temp_config_dir / "base.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_with_missing_env, f)
        
        config_manager = ConfigurationManager(str(temp_config_dir))
        config_manager.load_configuration()
        
        # Without secret manager, missing vars should remain as placeholders
        assert config_manager.get("required_value") == "${MISSING_REQUIRED_VAR}"
        assert config_manager.get("optional_value") == "default_value"  # Default still works
    
    def test_reload_configuration(self, temp_config_dir, sample_config_files):
        """Test configuration reloading."""
        config_manager = ConfigurationManager(str(temp_config_dir))
        config_manager.load_configuration()
        
        original_debug = config_manager.get("debug")
        
        # Modify the configuration file
        dev_file = sample_config_files["development"]
        with open(dev_file, 'r') as f:
            dev_config = yaml.safe_load(f)
        
        dev_config["debug"] = not original_debug  # Flip the debug setting
        
        with open(dev_file, 'w') as f:
            yaml.dump(dev_config, f)
        
        # Reload configuration
        config_manager.reload_configuration()
        
        # Should have new value
        assert config_manager.get("debug") == (not original_debug)
    
    def test_configuration_validation(self, temp_config_dir):
        """Test configuration validation."""
        # Define validation schema
        schema = {
            "type": "object",
            "properties": {
                "app_name": {"type": "string"},
                "debug": {"type": "boolean"},
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535}
                    },
                    "required": ["host", "port"]
                }
            },
            "required": ["app_name", "database"]
        }
        
        # Valid configuration
        valid_config = {
            "app_name": "test-app",
            "debug": True,
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }
        
        config_file = temp_config_dir / "base.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigurationManager(str(temp_config_dir))
        config_manager.set_validation_schema(schema)
        
        # Should load without error
        config_manager.load_configuration()
        
        # Test invalid configuration
        config_manager.set("database.port", "invalid_port")  # Should be integer
        
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            config_manager.validate_configuration()
    
    def test_configuration_file_watching(self, temp_config_dir, sample_config_files):
        """Test configuration file watching for changes."""
        config_manager = ConfigurationManager(str(temp_config_dir))
        config_manager.load_configuration()
        
        reload_called = False
        
        def mock_reload():
            nonlocal reload_called
            reload_called = True
        
        # Mock the reload method to track calls
        config_manager.reload_configuration = mock_reload
        
        # Start watching (in a real implementation, this would use file system events)
        config_manager.start_watching()
        
        # Simulate file change by calling the internal handler
        if hasattr(config_manager, '_on_config_file_change'):
            config_manager._on_config_file_change(str(sample_config_files["development"]))
            assert reload_called
        
        config_manager.stop_watching()
    
    def test_configuration_to_dict(self, temp_config_dir, sample_config_files):
        """Test converting configuration to dictionary."""
        config_manager = ConfigurationManager(str(temp_config_dir))
        config_manager.load_configuration()
        
        config_dict = config_manager.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "torch-inference"
        assert config_dict["debug"] is True
        assert isinstance(config_dict["database"], dict)
        assert config_dict["database"]["host"] == "dev-db.example.com"


class TestFeatureFlagManager:
    """Test feature flag manager functionality."""
    
    @pytest.fixture
    def temp_flags_file(self):
        """Create temporary feature flags file."""
        flags_data = {
            "feature_flags": {
                "new_model_api": {
                    "enabled": True,
                    "description": "New model API endpoint",
                    "environments": ["development", "staging"],
                    "rollout_percentage": 100
                },
                "experimental_cache": {
                    "enabled": False,
                    "description": "Experimental caching layer",
                    "environments": ["development"],
                    "rollout_percentage": 0
                },
                "advanced_metrics": {
                    "enabled": True,
                    "description": "Advanced metrics collection",
                    "environments": ["development", "production"],
                    "rollout_percentage": 50
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(flags_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_feature_flag_manager_initialization(self, temp_flags_file):
        """Test feature flag manager initialization."""
        flag_manager = FeatureFlagManager(temp_flags_file, environment="development")
        
        assert flag_manager.flags_file == temp_flags_file
        assert flag_manager.environment == "development"
        assert len(flag_manager._flags) > 0
    
    def test_is_feature_enabled_simple(self, temp_flags_file):
        """Test simple feature flag checking."""
        flag_manager = FeatureFlagManager(temp_flags_file, environment="development")
        
        # Feature enabled in development
        assert flag_manager.is_feature_enabled("new_model_api") is True
        
        # Feature disabled
        assert flag_manager.is_feature_enabled("experimental_cache") is False
        
        # Non-existent feature
        assert flag_manager.is_feature_enabled("nonexistent_feature") is False
    
    def test_is_feature_enabled_environment_specific(self, temp_flags_file):
        """Test environment-specific feature flags."""
        dev_manager = FeatureFlagManager(temp_flags_file, environment="development")
        prod_manager = FeatureFlagManager(temp_flags_file, environment="production")
        
        # Feature enabled in both environments
        assert dev_manager.is_feature_enabled("advanced_metrics") is True
        assert prod_manager.is_feature_enabled("advanced_metrics") is True
        
        # Feature enabled only in development
        assert dev_manager.is_feature_enabled("new_model_api") is True
        assert prod_manager.is_feature_enabled("new_model_api") is False
    
    def test_is_feature_enabled_rollout_percentage(self, temp_flags_file):
        """Test rollout percentage feature flags."""
        flag_manager = FeatureFlagManager(temp_flags_file, environment="production")
        
        # Mock random to test rollout percentage
        with patch('random.randint') as mock_random:
            # User should get feature (random < rollout_percentage)
            mock_random.return_value = 25  # 25 < 50% rollout
            assert flag_manager.is_feature_enabled("advanced_metrics", user_id="user1") is True
            
            # User should not get feature (random >= rollout_percentage)
            mock_random.return_value = 75  # 75 >= 50% rollout
            assert flag_manager.is_feature_enabled("advanced_metrics", user_id="user2") is False
    
    def test_is_feature_enabled_user_consistency(self, temp_flags_file):
        """Test that feature flags are consistent for the same user."""
        flag_manager = FeatureFlagManager(temp_flags_file, environment="production")
        
        # Same user should get consistent results
        user_id = "consistent-user"
        
        result1 = flag_manager.is_feature_enabled("advanced_metrics", user_id=user_id)
        result2 = flag_manager.is_feature_enabled("advanced_metrics", user_id=user_id)
        result3 = flag_manager.is_feature_enabled("advanced_metrics", user_id=user_id)
        
        assert result1 == result2 == result3
    
    def test_get_feature_info(self, temp_flags_file):
        """Test getting feature flag information."""
        flag_manager = FeatureFlagManager(temp_flags_file)
        
        info = flag_manager.get_feature_info("new_model_api")
        
        assert info is not None
        assert info["enabled"] is True
        assert info["description"] == "New model API endpoint"
        assert "development" in info["environments"]
        assert info["rollout_percentage"] == 100
    
    def test_get_all_features(self, temp_flags_file):
        """Test getting all feature flags."""
        flag_manager = FeatureFlagManager(temp_flags_file, environment="development")
        
        all_features = flag_manager.get_all_features()
        
        assert isinstance(all_features, dict)
        assert "new_model_api" in all_features
        assert "experimental_cache" in all_features
        assert "advanced_metrics" in all_features
        
        # Check that each feature includes enabled status for current environment
        for feature_name, feature_info in all_features.items():
            assert "enabled_in_environment" in feature_info
            assert isinstance(feature_info["enabled_in_environment"], bool)
    
    def test_reload_feature_flags(self, temp_flags_file):
        """Test reloading feature flags."""
        flag_manager = FeatureFlagManager(temp_flags_file)
        
        # Check initial state
        assert flag_manager.is_feature_enabled("experimental_cache") is False
        
        # Modify the flags file
        with open(temp_flags_file, 'r') as f:
            flags_data = yaml.safe_load(f)
        
        flags_data["feature_flags"]["experimental_cache"]["enabled"] = True
        
        with open(temp_flags_file, 'w') as f:
            yaml.dump(flags_data, f)
        
        # Reload flags
        flag_manager.reload_flags()
        
        # Check new state
        assert flag_manager.is_feature_enabled("experimental_cache") is True
    
    def test_feature_flag_file_not_found(self):
        """Test handling of missing feature flags file."""
        with pytest.raises(FileNotFoundError):
            FeatureFlagManager("/nonexistent/flags.yaml")
    
    def test_invalid_feature_flag_format(self):
        """Test handling of invalid feature flag file format."""
        invalid_flags = {"invalid": "format"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_flags, f)
            temp_file = f.name
        
        try:
            flag_manager = FeatureFlagManager(temp_file)
            # Should handle gracefully and have empty flags
            assert len(flag_manager._flags) == 0
        finally:
            os.unlink(temp_file)


class TestSecretManager:
    """Test secret manager functionality."""
    
    @pytest.fixture
    def temp_secrets_file(self):
        """Create temporary secrets file."""
        secrets_data = {
            "database_password": "super_secret_db_password",
            "api_key": "sk-test-api-key-12345",
            "jwt_secret": "jwt-signing-secret-key",
            "encryption_key": "32-byte-encryption-key-here!!"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(secrets_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_secret_manager_initialization_file(self, temp_secrets_file):
        """Test secret manager initialization with file source."""
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": temp_secrets_file
        })
        
        assert secret_manager.source_type == "file"
        assert len(secret_manager._secrets) > 0
    
    def test_secret_manager_initialization_env(self):
        """Test secret manager initialization with environment source."""
        with patch.dict(os.environ, {
            'SECRET_DB_PASSWORD': 'env_db_password',
            'SECRET_API_KEY': 'env_api_key'
        }):
            secret_manager = SecretManager(source_type="env", source_config={
                "prefix": "SECRET_"
            })
            
            assert secret_manager.source_type == "env"
            assert secret_manager.get_secret("DB_PASSWORD") == "env_db_password"
            assert secret_manager.get_secret("API_KEY") == "env_api_key"
    
    def test_get_secret_file_source(self, temp_secrets_file):
        """Test getting secrets from file source."""
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": temp_secrets_file
        })
        
        assert secret_manager.get_secret("database_password") == "super_secret_db_password"
        assert secret_manager.get_secret("api_key") == "sk-test-api-key-12345"
        assert secret_manager.get_secret("jwt_secret") == "jwt-signing-secret-key"
    
    def test_get_secret_not_found(self, temp_secrets_file):
        """Test getting non-existent secret."""
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": temp_secrets_file
        })
        
        with pytest.raises(SecretNotFoundError, match="Secret 'nonexistent' not found"):
            secret_manager.get_secret("nonexistent")
    
    def test_get_secret_with_default(self, temp_secrets_file):
        """Test getting secret with default value."""
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": temp_secrets_file
        })
        
        # Existing secret should return actual value
        assert secret_manager.get_secret("api_key", "default") == "sk-test-api-key-12345"
        
        # Non-existent secret should return default
        assert secret_manager.get_secret("nonexistent", "default_value") == "default_value"
    
    def test_has_secret(self, temp_secrets_file):
        """Test checking if secret exists."""
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": temp_secrets_file
        })
        
        assert secret_manager.has_secret("database_password") is True
        assert secret_manager.has_secret("nonexistent") is False
    
    def test_list_secrets(self, temp_secrets_file):
        """Test listing available secrets."""
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": temp_secrets_file
        })
        
        secrets = secret_manager.list_secrets()
        
        assert "database_password" in secrets
        assert "api_key" in secrets
        assert "jwt_secret" in secrets
        assert "encryption_key" in secrets
        assert len(secrets) == 4
    
    def test_refresh_secrets(self, temp_secrets_file):
        """Test refreshing secrets from source."""
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": temp_secrets_file
        })
        
        # Check initial state
        assert secret_manager.has_secret("new_secret") is False
        
        # Modify secrets file
        with open(temp_secrets_file, 'r') as f:
            secrets_data = json.load(f)
        
        secrets_data["new_secret"] = "new_secret_value"
        
        with open(temp_secrets_file, 'w') as f:
            json.dump(secrets_data, f)
        
        # Refresh secrets
        secret_manager.refresh_secrets()
        
        # Check new secret is available
        assert secret_manager.has_secret("new_secret") is True
        assert secret_manager.get_secret("new_secret") == "new_secret_value"
    
    @patch('boto3.client')
    def test_aws_secrets_manager(self, mock_boto_client):
        """Test AWS Secrets Manager integration."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock AWS response
        mock_client.get_secret_value.return_value = {
            'SecretString': json.dumps({
                'database_password': 'aws_db_password',
                'api_key': 'aws_api_key'
            })
        }
        
        secret_manager = SecretManager(source_type="aws", source_config={
            "secret_name": "prod/inference/secrets",
            "region": "us-east-1"
        })
        
        assert secret_manager.get_secret("database_password") == "aws_db_password"
        assert secret_manager.get_secret("api_key") == "aws_api_key"
        
        mock_client.get_secret_value.assert_called_with(SecretId="prod/inference/secrets")
    
    def test_secret_caching(self, temp_secrets_file):
        """Test secret caching behavior."""
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": temp_secrets_file
        })
        
        # First access should load from file
        secret1 = secret_manager.get_secret("database_password")
        
        # Modify file
        with open(temp_secrets_file, 'w') as f:
            json.dump({"database_password": "modified_password"}, f)
        
        # Second access should return cached value (not modified)
        secret2 = secret_manager.get_secret("database_password")
        assert secret1 == secret2 == "super_secret_db_password"
        
        # After refresh, should get new value
        secret_manager.refresh_secrets()
        secret3 = secret_manager.get_secret("database_password")
        assert secret3 == "modified_password"


class TestConfigurationIntegration:
    """Test integration between configuration, feature flags, and secrets."""
    
    @pytest.fixture
    def integrated_config_setup(self):
        """Set up integrated configuration environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Configuration files
            base_config = {
                "app_name": "integrated-test",
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "user": "app_user",
                    "password": "${DB_PASSWORD}"  # From secrets
                },
                "features": {
                    "enabled": ["basic_auth"]
                }
            }
            
            config_file = temp_path / "base.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(base_config, f)
            
            # Feature flags
            feature_flags = {
                "feature_flags": {
                    "advanced_auth": {
                        "enabled": True,
                        "environments": ["development"],
                        "rollout_percentage": 100
                    }
                }
            }
            
            flags_file = temp_path / "flags.yaml"
            with open(flags_file, 'w') as f:
                yaml.dump(feature_flags, f)
            
            # Secrets
            secrets = {
                "DB_PASSWORD": "integrated_test_password",
                "JWT_SECRET": "integrated_jwt_secret"
            }
            
            secrets_file = temp_path / "secrets.json"
            with open(secrets_file, 'w') as f:
                json.dump(secrets, f)
            
            yield {
                "config_dir": str(temp_path),
                "flags_file": str(flags_file),
                "secrets_file": str(secrets_file)
            }
    
    def test_integrated_configuration_loading(self, integrated_config_setup):
        """Test loading configuration with secrets substitution."""
        setup = integrated_config_setup
        
        # Set up secret manager
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": setup["secrets_file"]
        })
        
        # Set up configuration manager with secret manager
        config_manager = ConfigurationManager(setup["config_dir"])
        config_manager.set_secret_manager(secret_manager)
        config_manager.load_configuration()
        
        # Configuration should have secrets substituted
        assert config_manager.get("database.password") == "integrated_test_password"
        assert config_manager.get("database.host") == "localhost"
        assert config_manager.get("app_name") == "integrated-test"
    
    def test_integrated_feature_flag_usage(self, integrated_config_setup):
        """Test using feature flags with configuration."""
        setup = integrated_config_setup
        
        config_manager = ConfigurationManager(setup["config_dir"])
        config_manager.load_configuration()
        
        flag_manager = FeatureFlagManager(setup["flags_file"], environment="development")
        
        # Base features from config
        base_features = config_manager.get("features.enabled", [])
        assert "basic_auth" in base_features
        
        # Dynamic features from feature flags
        dynamic_features = []
        if flag_manager.is_feature_enabled("advanced_auth"):
            dynamic_features.append("advanced_auth")
        
        # Combined feature set
        all_features = base_features + dynamic_features
        assert "basic_auth" in all_features
        assert "advanced_auth" in all_features
    
    def test_configuration_hot_reload_with_secrets(self, integrated_config_setup):
        """Test configuration hot reload with secrets changes."""
        setup = integrated_config_setup
        
        secret_manager = SecretManager(source_type="file", source_config={
            "file_path": setup["secrets_file"]
        })
        
        config_manager = ConfigurationManager(setup["config_dir"])
        config_manager.set_secret_manager(secret_manager)
        config_manager.load_configuration()
        
        # Initial password
        initial_password = config_manager.get("database.password")
        assert initial_password == "integrated_test_password"
        
        # Update secrets file
        new_secrets = {
            "DB_PASSWORD": "updated_password",
            "JWT_SECRET": "updated_jwt_secret"
        }
        
        with open(setup["secrets_file"], 'w') as f:
            json.dump(new_secrets, f)
        
        # Refresh secrets and reload config
        secret_manager.refresh_secrets()
        config_manager.reload_configuration()
        
        # Should have updated password
        updated_password = config_manager.get("database.password")
        assert updated_password == "updated_password"
