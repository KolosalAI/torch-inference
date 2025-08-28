"""
Enterprise configuration management with advanced features.

This module provides comprehensive configuration for enterprise deployments including:
- Multi-environment support
- Secrets management
- Security policies
- Compliance settings
- Integration configurations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
import os
from datetime import timedelta

from ..core.config import InferenceConfig


class AuthProvider(Enum):
    """Supported authentication providers."""
    JWT = "jwt"
    OAUTH2 = "oauth2"
    SAML = "saml"
    OIDC = "oidc"
    ACTIVE_DIRECTORY = "active_directory"
    LDAP = "ldap"


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"


class ComplianceStandard(Enum):
    """Compliance standards."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    FIPS_140_2 = "fips_140_2"


@dataclass
class AuthConfig:
    """Authentication configuration."""
    provider: AuthProvider = AuthProvider.JWT
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # OAuth2/OIDC settings
    oauth2_client_id: str = ""
    oauth2_client_secret: str = ""
    oauth2_server_url: str = ""
    oauth2_scopes: List[str] = field(default_factory=list)
    
    # SAML settings
    saml_sp_entity_id: str = ""
    saml_idp_url: str = ""
    saml_x509_cert: str = ""
    
    # Active Directory/LDAP
    ldap_server_url: str = ""
    ldap_bind_dn: str = ""
    ldap_bind_password: str = ""
    ldap_user_search_base: str = ""
    
    # Multi-factor authentication
    enable_mfa: bool = False
    mfa_issuer: str = "TorchInference"
    
    # API key settings
    enable_api_keys: bool = True
    api_key_header: str = "X-API-Key"
    api_key_expiry_days: int = 365


@dataclass
class RBACConfig:
    """Role-based access control configuration."""
    enable_rbac: bool = True
    default_role: str = "user"
    admin_users: List[str] = field(default_factory=list)
    
    # Role definitions
    roles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "admin": {
            "permissions": ["*"],
            "description": "Full system access"
        },
        "model_manager": {
            "permissions": [
                "model:create", "model:read", "model:update", "model:delete",
                "inference:predict", "metrics:read"
            ],
            "description": "Model management and inference"
        },
        "data_scientist": {
            "permissions": [
                "model:read", "inference:predict", "metrics:read",
                "experiment:create", "experiment:read"
            ],
            "description": "Model usage and experimentation"
        },
        "user": {
            "permissions": ["inference:predict"],
            "description": "Basic inference access"
        }
    })
    
    # Resource-based permissions
    resource_permissions: Dict[str, List[str]] = field(default_factory=dict)
    tenant_isolation: bool = True


@dataclass
class SecuritySettings:
    """Security configuration."""
    # Encryption settings
    enable_encryption_at_rest: bool = True
    enable_encryption_in_transit: bool = True
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    encryption_key_rotation_days: int = 90
    
    # Input validation
    max_request_size_mb: int = 100
    allowed_file_types: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
    ])
    enable_content_type_validation: bool = True
    enable_file_signature_validation: bool = True
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20
    
    # Security headers
    enable_security_headers: bool = True
    cors_allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allowed_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    
    # Audit logging
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90
    log_sensitive_data: bool = False
    
    # Threat detection
    enable_anomaly_detection: bool = True
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    # Secrets management
    secrets_provider: str = "env"  # env, vault, aws_secrets, azure_keyvault
    vault_url: str = ""
    vault_token: str = ""


@dataclass
class MonitoringConfig:
    """Enterprise monitoring configuration."""
    # Basic monitoring
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True
    
    # Metrics configuration
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    metrics_retention_days: int = 30
    
    # Distributed tracing
    tracing_service_name: str = "torch-inference"
    tracing_sampling_rate: float = 0.1
    jaeger_endpoint: str = ""
    zipkin_endpoint: str = ""
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "json"
    log_retention_days: int = 30
    enable_structured_logging: bool = True
    
    # Alerting
    enable_alerting: bool = True
    alert_channels: List[str] = field(default_factory=list)  # slack, email, pagerduty
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.05,
        "latency_p95_ms": 1000,
        "memory_usage_percent": 85,
        "cpu_usage_percent": 80,
        "disk_usage_percent": 90
    })
    
    # Health checks
    health_check_interval_seconds: int = 30
    readiness_timeout_seconds: int = 10
    liveness_timeout_seconds: int = 5


@dataclass
class ComplianceConfig:
    """Compliance and governance configuration."""
    enabled_standards: List[ComplianceStandard] = field(default_factory=list)
    
    # GDPR settings
    gdpr_data_retention_days: int = 365
    gdpr_enable_right_to_be_forgotten: bool = True
    gdpr_enable_data_portability: bool = True
    gdpr_consent_required: bool = True
    
    # Data privacy
    enable_data_anonymization: bool = False
    pii_detection_enabled: bool = True
    data_classification_levels: List[str] = field(default_factory=lambda: [
        "public", "internal", "confidential", "restricted"
    ])
    
    # Audit requirements
    audit_log_immutability: bool = True
    audit_log_encryption: bool = True
    audit_log_backup_enabled: bool = True
    
    # Model governance
    model_approval_required: bool = False
    model_version_control: bool = True
    model_performance_monitoring: bool = True
    model_bias_detection: bool = False


@dataclass
class ScalingConfig:
    """Auto-scaling and resource management."""
    enable_auto_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    
    # CPU-based scaling
    cpu_target_utilization: int = 70
    cpu_scale_up_threshold: int = 80
    cpu_scale_down_threshold: int = 50
    
    # Memory-based scaling
    memory_target_utilization: int = 70
    memory_scale_up_threshold: int = 80
    memory_scale_down_threshold: int = 50
    
    # Custom metrics scaling
    custom_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Scaling behavior
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    scale_up_increment: int = 2
    scale_down_decrement: int = 1


@dataclass
class IntegrationConfig:
    """Enterprise integration configuration."""
    # Database connections
    database_url: str = ""
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Message queues
    message_broker_url: str = ""
    enable_async_processing: bool = True
    task_queue_name: str = "inference_tasks"
    result_backend_url: str = ""
    
    # Caching
    cache_backend: str = "redis"  # redis, memcached, memory
    cache_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_max_size_mb: int = 1024
    
    # External APIs
    model_registry_url: str = ""
    experiment_tracking_url: str = ""
    feature_store_url: str = ""
    
    # Kubernetes integration
    kubernetes_namespace: str = "default"
    kubernetes_service_account: str = ""
    enable_kubernetes_discovery: bool = False


@dataclass
class SecurityConfig:
    """Main security configuration."""
    # Basic inference config
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Security features
    auth: AuthConfig = field(default_factory=AuthConfig)
    rbac: RBACConfig = field(default_factory=RBACConfig)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    # Deployment settings
    environment: str = "production"  # development, staging, production
    tenant_id: Optional[str] = None
    deployment_id: str = ""
    version: str = "1.0.0"
    
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Basic settings
        config.environment = os.getenv("ENVIRONMENT", "production")
        config.tenant_id = os.getenv("TENANT_ID")
        config.deployment_id = os.getenv("DEPLOYMENT_ID", "")
        
        # Authentication
        config.auth.secret_key = os.getenv("JWT_SECRET_KEY", "")
        config.auth.oauth2_client_id = os.getenv("OAUTH2_CLIENT_ID", "")
        config.auth.oauth2_client_secret = os.getenv("OAUTH2_CLIENT_SECRET", "")
        
        # Security
        config.security.enable_encryption_at_rest = os.getenv("ENABLE_ENCRYPTION_AT_REST", "true").lower() == "true"
        config.security.rate_limit_requests_per_minute = int(os.getenv("RATE_LIMIT_RPM", "100"))
        
        # Monitoring
        config.monitoring.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "")
        config.monitoring.metrics_port = int(os.getenv("METRICS_PORT", "9090"))
        
        # Integration
        config.integration.database_url = os.getenv("DATABASE_URL", "")
        config.integration.cache_url = os.getenv("CACHE_URL", "redis://localhost:6379/0")
        config.integration.message_broker_url = os.getenv("MESSAGE_BROKER_URL", "")
        
        return config
    
    def validate(self) -> None:
        """Validate enterprise configuration."""
        # Validate authentication
        if self.auth.provider != AuthProvider.JWT and not self.auth.secret_key:
            raise ValueError("Secret key is required for authentication")
        
        # Validate security
        if self.security.enable_encryption_at_rest and not self.security.encryption_algorithm:
            raise ValueError("Encryption algorithm must be specified")
        
        # Validate monitoring
        if self.monitoring.enable_tracing and not (self.monitoring.jaeger_endpoint or self.monitoring.zipkin_endpoint):
            raise ValueError("Tracing endpoint must be configured when tracing is enabled")
        
        # Validate RBAC
        if self.rbac.enable_rbac and not self.rbac.roles:
            raise ValueError("Roles must be defined when RBAC is enabled")
        
        # Validate compliance
        for standard in self.compliance.enabled_standards:
            if standard == ComplianceStandard.GDPR:
                if not self.compliance.gdpr_consent_required:
                    raise ValueError("GDPR compliance requires consent management")
    
    def get_secrets(self) -> Dict[str, str]:
        """Get sensitive configuration values."""
        secrets = {}
        
        if self.security.secrets_provider == "env":
            secrets.update({
                "jwt_secret_key": self.auth.secret_key,
                "oauth2_client_secret": self.auth.oauth2_client_secret,
                "database_url": self.integration.database_url,
                "cache_url": self.integration.cache_url
            })
        
        return secrets
    
    def export_for_deployment(self) -> Dict[str, Any]:
        """Export configuration for deployment (excluding secrets)."""
        config_dict = {
            "environment": self.environment,
            "tenant_id": self.tenant_id,
            "deployment_id": self.deployment_id,
            "version": self.version,
            "auth": {
                "provider": self.auth.provider.value,
                "algorithm": self.auth.algorithm,
                "access_token_expire_minutes": self.auth.access_token_expire_minutes,
                "enable_mfa": self.auth.enable_mfa,
                "enable_api_keys": self.auth.enable_api_keys
            },
            "security": {
                "enable_encryption_at_rest": self.security.enable_encryption_at_rest,
                "enable_rate_limiting": self.security.enable_rate_limiting,
                "rate_limit_requests_per_minute": self.security.rate_limit_requests_per_minute,
                "enable_audit_logging": self.security.enable_audit_logging
            },
            "monitoring": {
                "enable_metrics": self.monitoring.enable_metrics,
                "enable_tracing": self.monitoring.enable_tracing,
                "metrics_port": self.monitoring.metrics_port,
                "log_level": self.monitoring.log_level
            },
            "scaling": {
                "enable_auto_scaling": self.scaling.enable_auto_scaling,
                "min_replicas": self.scaling.min_replicas,
                "max_replicas": self.scaling.max_replicas
            }
        }
        
        return config_dict


# Factory functions for common configurations
def create_development_config() -> SecurityConfig:
    """Create configuration for development environment."""
    config = SecurityConfig()
    config.environment = "development"
    config.auth.access_token_expire_minutes = 480  # 8 hours
    config.security.enable_rate_limiting = False
    config.security.enable_audit_logging = False
    config.monitoring.log_level = "DEBUG"
    config.scaling.enable_auto_scaling = False
    return config


def create_staging_config() -> SecurityConfig:
    """Create configuration for staging environment."""
    config = SecurityConfig()
    config.environment = "staging"
    config.security.rate_limit_requests_per_minute = 500
    config.monitoring.tracing_sampling_rate = 0.5
    config.scaling.max_replicas = 5
    return config


def create_production_config() -> SecurityConfig:
    """Create configuration for production environment."""
    config = SecurityConfig()
    config.environment = "production"
    config.auth.enable_mfa = True
    config.security.enable_encryption_at_rest = True
    config.security.enable_audit_logging = True
    config.compliance.enabled_standards = [ComplianceStandard.GDPR]
    config.monitoring.enable_alerting = True
    config.scaling.enable_auto_scaling = True
    return config
