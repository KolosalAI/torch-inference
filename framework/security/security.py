"""
Enterprise security management system.

This module provides comprehensive security features including:
- Encryption at rest and in transit
- Input validation and sanitization
- Rate limiting and DDoS protection
- Audit logging
- Threat detection
- Security headers
"""

import hashlib
import hmac
import secrets
import time
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import json
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    magic = None

from .config import SecurityConfig, EncryptionAlgorithm


logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    AUTHENTICATION_ERROR = "auth_error"
    AUTHORIZATION_ERROR = "authz_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_INPUT = "invalid_input"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    MODEL_ACCESS = "model_access"
    ADMIN_ACTION = "admin_action"


@dataclass
class SecurityAlert:
    """Security alert model."""
    id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "resolved": self.resolved
        }


@dataclass
class AuditLogEntry:
    """Audit log entry."""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    tenant_id: Optional[str] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "tenant_id": self.tenant_id,
            "success": self.success
        }


class EncryptionManager:
    """Encryption and decryption management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.algorithm = config.security.encryption_algorithm
        self._initialize_encryption()
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption keys and ciphers."""
        # Generate or load encryption key
        self.encryption_key = self._get_or_generate_key()
        
        if self.algorithm == EncryptionAlgorithm.AES_256_GCM:
            self.cipher_suite = Fernet(base64.urlsafe_b64encode(self.encryption_key[:32]))
        else:
            # For other algorithms, implement as needed
            self.cipher_suite = Fernet(base64.urlsafe_b64encode(self.encryption_key[:32]))
    
    def _get_or_generate_key(self) -> bytes:
        """Get existing encryption key or generate new one."""
        # In production, this would load from secure key management system
        key_file = "encryption.key"
        try:
            with open(key_file, "rb") as f:
                return f.read()
        except FileNotFoundError:
            key = secrets.token_bytes(32)  # 256-bit key
            with open(key_file, "wb") as f:
                f.write(key)
            return key
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt data and return base64 encoded result."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.cipher_suite.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> bytes:
        """Decrypt base64 encoded data."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        return self.cipher_suite.decrypt(encrypted_bytes)
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt file and return path to encrypted file."""
        if output_path is None:
            output_path = f"{file_path}.enc"
        
        with open(file_path, "rb") as f:
            data = f.read()
        
        encrypted_data = self.cipher_suite.encrypt(data)
        
        with open(output_path, "wb") as f:
            f.write(encrypted_data)
        
        return output_path
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt file and return path to decrypted file."""
        if output_path is None:
            output_path = encrypted_file_path.replace(".enc", "")
        
        with open(encrypted_file_path, "rb") as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        
        with open(output_path, "wb") as f:
            f.write(decrypted_data)
        
        return output_path
    
    def generate_hash(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> str:
        """Generate secure hash of data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if salt is None:
            salt = secrets.token_bytes(32)
        
        hash_obj = hashlib.pbkdf2_hmac('sha256', data, salt, 100000)
        return base64.urlsafe_b64encode(salt + hash_obj).decode('utf-8')
    
    def verify_hash(self, data: Union[str, bytes], hash_value: str) -> bool:
        """Verify data against hash."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        try:
            hash_bytes = base64.urlsafe_b64decode(hash_value.encode('utf-8'))
            salt = hash_bytes[:32]
            stored_hash = hash_bytes[32:]
            
            new_hash = hashlib.pbkdf2_hmac('sha256', data, salt, 100000)
            return hmac.compare_digest(stored_hash, new_hash)
        except Exception:
            return False


class InputValidator:
    """Input validation and sanitization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.max_size = config.security.max_request_size_mb * 1024 * 1024
        self.allowed_extensions = set(config.security.allowed_file_types)
        
        # Compile regex patterns for common attacks
        self.sql_injection_pattern = re.compile(
            r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bDROP\b|\bUPDATE\b)",
            re.IGNORECASE
        )
        self.xss_pattern = re.compile(
            r"(<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>|javascript:|on\w+\s*=)",
            re.IGNORECASE
        )
        self.path_traversal_pattern = re.compile(r"\.\.[/\\]")
        
    def validate_file_upload(self, file_data: bytes, filename: str) -> Tuple[bool, Optional[str]]:
        """Validate uploaded file."""
        # Check file size
        if len(file_data) > self.max_size:
            return False, f"File size exceeds maximum allowed size ({self.max_size} bytes)"
        
        # Check file extension
        file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if file_ext not in self.allowed_extensions:
            return False, f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}"
        
        # Check file signature if validation is enabled
        if self.config.security.enable_file_signature_validation:
            if not self._validate_file_signature(file_data, file_ext):
                return False, "File signature does not match extension"
        
        return True, None
    
    def _validate_file_signature(self, file_data: bytes, expected_ext: str) -> bool:
        """Validate file signature matches extension."""
        if not file_data:
            return False
        
        # Common file signatures
        signatures = {
            '.jpg': [b'\xff\xd8\xff'],
            '.jpeg': [b'\xff\xd8\xff'],
            '.png': [b'\x89PNG\r\n\x1a\n'],
            '.bmp': [b'BM'],
            '.tiff': [b'II*\x00', b'MM\x00*'],
            '.webp': [b'RIFF', b'WEBP']
        }
        
        expected_sigs = signatures.get(expected_ext, [])
        for sig in expected_sigs:
            if file_data.startswith(sig):
                return True
        
        return len(expected_sigs) == 0  # Allow if no signature defined
    
    def validate_input_text(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate text input for security threats."""
        # Check for SQL injection
        if self.sql_injection_pattern.search(text):
            return False, "Potential SQL injection detected"
        
        # Check for XSS
        if self.xss_pattern.search(text):
            return False, "Potential XSS attack detected"
        
        # Check for path traversal
        if self.path_traversal_pattern.search(text):
            return False, "Path traversal attempt detected"
        
        return True, None
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text."""
        # Remove potential HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Escape special characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#x27;')
        
        return text
    
    def validate_json(self, json_data: str, max_depth: int = 10) -> Tuple[bool, Optional[str]]:
        """Validate JSON data."""
        try:
            data = json.loads(json_data)
            
            # Check nesting depth to prevent DoS
            def check_depth(obj, depth=0):
                if depth > max_depth:
                    return False
                if isinstance(obj, dict):
                    return all(check_depth(v, depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_depth(item, depth + 1) for item in obj)
                return True
            
            if not check_depth(data):
                return False, f"JSON nesting depth exceeds maximum ({max_depth})"
            
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.requests_per_minute = config.security.rate_limit_requests_per_minute
        self.burst_size = config.security.rate_limit_burst_size
        
        # Token bucket implementation
        self.buckets: Dict[str, Dict[str, Any]] = {}
        
        # Sliding window for rate limiting
        self.request_windows: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, client_id: str, endpoint: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limits."""
        current_time = time.time()
        key = f"{client_id}:{endpoint}" if endpoint else client_id
        
        # Initialize bucket if not exists
        if key not in self.buckets:
            self.buckets[key] = {
                'tokens': self.burst_size,
                'last_refill': current_time
            }
        
        bucket = self.buckets[key]
        
        # Refill tokens based on time elapsed
        time_elapsed = current_time - bucket['last_refill']
        tokens_to_add = time_elapsed * (self.requests_per_minute / 60.0)
        bucket['tokens'] = min(self.burst_size, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = current_time
        
        # Check if request can be processed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True, {
                'allowed': True,
                'remaining_tokens': int(bucket['tokens']),
                'retry_after': 0
            }
        else:
            retry_after = (1 - bucket['tokens']) / (self.requests_per_minute / 60.0)
            return False, {
                'allowed': False,
                'remaining_tokens': 0,
                'retry_after': int(retry_after)
            }
    
    def get_usage_stats(self, client_id: str) -> Dict[str, Any]:
        """Get rate limiting statistics for client."""
        current_time = time.time()
        window = self.request_windows[client_id]
        
        # Clean old requests (older than 1 minute)
        while window and current_time - window[0] > 60:
            window.popleft()
        
        return {
            'requests_last_minute': len(window),
            'remaining_capacity': max(0, self.requests_per_minute - len(window)),
            'reset_time': current_time + 60 - (window[0] if window else current_time)
        }


class ThreatDetector:
    """Threat detection and anomaly analysis."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.anomaly_thresholds = {
            'failed_login_rate': 5,  # per minute
            'unusual_access_pattern': 100,  # requests per minute
            'suspicious_user_agent': ['bot', 'crawler', 'scanner']
        }
    
    def analyze_login_attempt(self, user_id: Optional[str], ip_address: str, 
                             success: bool, user_agent: Optional[str] = None) -> Optional[SecurityAlert]:
        """Analyze login attempt for threats."""
        current_time = datetime.now(timezone.utc)
        
        if not success:
            # Track failed attempts by IP
            self.failed_attempts[ip_address].append(current_time)
            
            # Clean old attempts (older than 1 hour)
            cutoff = current_time - timedelta(hours=1)
            self.failed_attempts[ip_address] = [
                attempt for attempt in self.failed_attempts[ip_address]
                if attempt > cutoff
            ]
            
            # Check for brute force attack
            recent_failures = [
                attempt for attempt in self.failed_attempts[ip_address]
                if attempt > current_time - timedelta(minutes=1)
            ]
            
            if len(recent_failures) >= self.config.security.max_failed_attempts:
                return SecurityAlert(
                    id=f"threat_{secrets.token_urlsafe(8)}",
                    event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                    threat_level=ThreatLevel.HIGH,
                    message=f"Brute force attack detected from IP {ip_address}",
                    details={
                        'failed_attempts': len(recent_failures),
                        'time_window': '1 minute',
                        'action_recommended': 'block_ip'
                    },
                    ip_address=ip_address,
                    user_id=user_id
                )
        
        # Check user agent for suspicious patterns
        if user_agent:
            for suspicious_pattern in self.anomaly_thresholds['suspicious_user_agent']:
                if suspicious_pattern.lower() in user_agent.lower():
                    return SecurityAlert(
                        id=f"threat_{secrets.token_urlsafe(8)}",
                        event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                        threat_level=ThreatLevel.MEDIUM,
                        message=f"Suspicious user agent detected: {user_agent}",
                        details={
                            'user_agent': user_agent,
                            'pattern_matched': suspicious_pattern
                        },
                        ip_address=ip_address,
                        user_id=user_id
                    )
        
        return None
    
    def analyze_request_pattern(self, client_id: str, endpoint: str, 
                              request_count: int, time_window: int) -> Optional[SecurityAlert]:
        """Analyze request patterns for anomalies."""
        requests_per_minute = (request_count / time_window) * 60
        
        if requests_per_minute > self.anomaly_thresholds['unusual_access_pattern']:
            return SecurityAlert(
                id=f"threat_{secrets.token_urlsafe(8)}",
                event_type=SecurityEvent.SUSPICIOUS_ACTIVITY,
                threat_level=ThreatLevel.MEDIUM,
                message=f"Unusual request pattern detected for client {client_id}",
                details={
                    'endpoint': endpoint,
                    'requests_per_minute': requests_per_minute,
                    'threshold': self.anomaly_thresholds['unusual_access_pattern']
                },
                user_id=client_id
            )
        
        return None


class AuditLogger:
    """Audit logging system."""
    
    def __init__(self, config: SecurityConfig, encryption_manager: EncryptionManager):
        self.config = config
        self.encryption_manager = encryption_manager
        self.log_entries: List[AuditLogEntry] = []
        self.max_entries = 10000  # Keep last 10k entries in memory
    
    def log_action(self, user_id: Optional[str], action: str, resource: str,
                  details: Dict[str, Any], ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None, tenant_id: Optional[str] = None,
                  success: bool = True) -> AuditLogEntry:
        """Log audit entry."""
        entry = AuditLogEntry(
            id=f"audit_{secrets.token_urlsafe(16)}",
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            action=action,
            resource=resource,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            tenant_id=tenant_id,
            success=success
        )
        
        self.log_entries.append(entry)
        
        # Maintain memory limit
        if len(self.log_entries) > self.max_entries:
            self.log_entries = self.log_entries[-self.max_entries:]
        
        # Log to file/database in production
        self._persist_log_entry(entry)
        
        return entry
    
    def _persist_log_entry(self, entry: AuditLogEntry) -> None:
        """Persist audit log entry (encrypted if configured)."""
        log_data = json.dumps(entry.to_dict())
        
        if self.config.compliance.audit_log_encryption:
            log_data = self.encryption_manager.encrypt_data(log_data)
        
        # In production, write to secure audit log file or database
        logger.info(f"AUDIT: {log_data}")
    
    def search_logs(self, user_id: Optional[str] = None, action: Optional[str] = None,
                   resource: Optional[str] = None, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None, limit: int = 100) -> List[AuditLogEntry]:
        """Search audit logs."""
        filtered_entries = []
        
        for entry in self.log_entries:
            # Apply filters
            if user_id and entry.user_id != user_id:
                continue
            if action and entry.action != action:
                continue
            if resource and entry.resource != resource:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            
            filtered_entries.append(entry)
            
            if len(filtered_entries) >= limit:
                break
        
        return filtered_entries
    
    def get_user_activity(self, user_id: str, hours: int = 24) -> List[AuditLogEntry]:
        """Get recent activity for user."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return self.search_logs(user_id=user_id, start_time=start_time)


class SecurityManager:
    """Main security management system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption_manager = EncryptionManager(config)
        self.input_validator = InputValidator(config)
        self.rate_limiter = RateLimiter(config)
        self.threat_detector = ThreatDetector(config)
        self.audit_logger = AuditLogger(config, self.encryption_manager)
        
        # Security alerts
        self.active_alerts: List[SecurityAlert] = []
        self.alert_callbacks: List[callable] = []
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add callback for security alerts."""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert: SecurityAlert) -> None:
        """Trigger security alert."""
        self.active_alerts.append(alert)
        logger.warning(f"Security Alert: {alert.message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def validate_request(self, client_id: str, request_data: Any,
                        endpoint: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Validate incoming request."""
        # Rate limiting check
        allowed, rate_info = self.rate_limiter.is_allowed(client_id, endpoint)
        if not allowed:
            self.audit_logger.log_action(
                user_id=client_id,
                action="rate_limit_exceeded",
                resource=endpoint or "unknown",
                details=rate_info,
                success=False
            )
            return False, f"Rate limit exceeded. Try again in {rate_info['retry_after']} seconds."
        
        # Input validation
        if isinstance(request_data, str):
            valid, error = self.input_validator.validate_input_text(request_data)
            if not valid:
                alert = SecurityAlert(
                    id=f"security_{secrets.token_urlsafe(8)}",
                    event_type=SecurityEvent.INVALID_INPUT,
                    threat_level=ThreatLevel.MEDIUM,
                    message=f"Invalid input detected: {error}",
                    details={"input_preview": request_data[:100]},
                    user_id=client_id
                )
                self._trigger_alert(alert)
                return False, error
        
        return True, None
    
    def log_security_event(self, event_type: SecurityEvent, user_id: Optional[str],
                          message: str, details: Dict[str, Any],
                          ip_address: Optional[str] = None) -> None:
        """Log security event."""
        self.audit_logger.log_action(
            user_id=user_id,
            action=event_type.value,
            resource="security_event",
            details={"message": message, **details},
            ip_address=ip_address
        )
    
    def analyze_login(self, user_id: Optional[str], ip_address: str,
                     success: bool, user_agent: Optional[str] = None) -> None:
        """Analyze login attempt for security threats."""
        alert = self.threat_detector.analyze_login_attempt(
            user_id, ip_address, success, user_agent
        )
        
        if alert:
            self._trigger_alert(alert)
        
        # Log the login attempt
        self.audit_logger.log_action(
            user_id=user_id,
            action="login_attempt",
            resource="authentication",
            details={
                "success": success,
                "user_agent": user_agent
            },
            ip_address=ip_address,
            success=success
        )
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics."""
        current_time = datetime.now(timezone.utc)
        last_hour = current_time - timedelta(hours=1)
        last_24h = current_time - timedelta(hours=24)
        
        # Get recent audit logs
        recent_logs = [
            entry for entry in self.audit_logger.log_entries
            if entry.timestamp > last_hour
        ]
        
        failed_logins = len([
            entry for entry in recent_logs
            if entry.action == "login_attempt" and not entry.success
        ])
        
        return {
            "active_alerts": len([a for a in self.active_alerts if not a.resolved]),
            "failed_logins_last_hour": failed_logins,
            "total_audit_entries": len(self.audit_logger.log_entries),
            "threat_detection_enabled": True,
            "encryption_enabled": self.config.security.enable_encryption_at_rest,
            "rate_limiting_enabled": self.config.security.enable_rate_limiting
        }
    
    def get_active_threats(self) -> List[SecurityAlert]:
        """Get active security threats."""
        return [alert for alert in self.active_alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve security alert."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
