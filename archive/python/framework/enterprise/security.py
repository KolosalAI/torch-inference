"""
Enterprise security features for multi-GPU inference.
Provides authentication, authorization, rate limiting, and security monitoring.
"""

import hashlib
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
import ipaddress

# Optional imports for JWT and bcrypt
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    bcrypt = None

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"

class Permission(Enum):
    """System permissions."""
    READ_INFERENCE = "read_inference"
    WRITE_INFERENCE = "write_inference"
    READ_MODELS = "read_models"
    WRITE_MODELS = "write_models"
    READ_METRICS = "read_metrics"
    WRITE_CONFIG = "write_config"
    ADMIN_USERS = "admin_users"
    ADMIN_SYSTEM = "admin_system"

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class User:
    """User account information."""
    username: str
    password_hash: str
    role: UserRole
    permissions: Set[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: datetime
    event_type: str
    severity: str
    user_id: Optional[str]
    ip_address: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RateLimitRule:
    """Rate limiting rule."""
    name: str
    requests_per_window: int
    window_seconds: int
    endpoints: List[str] = field(default_factory=list)
    user_roles: List[UserRole] = field(default_factory=list)

@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    level: SecurityLevel = SecurityLevel.MEDIUM
    require_authentication: bool = True
    require_authorization: bool = True
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    password_policy: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.password_policy is None:
            self.password_policy = {
                'min_length': 8,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_symbols': False
            }

class SecurityManager:
    """Enterprise security manager."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.jwt_secret = self.config.get('jwt_secret', secrets.token_urlsafe(32))
        self.jwt_algorithm = self.config.get('jwt_algorithm', 'HS256')
        self.jwt_expiry_hours = self.config.get('jwt_expiry_hours', 24)
        
        # User management
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, RateLimitRule] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Security monitoring
        self.security_events: deque = deque(maxlen=10000)
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        
        # Callbacks
        self.security_callbacks: List[Callable[[SecurityEvent], None]] = []
        
        # Locks
        self.user_lock = threading.RLock()
        self.session_lock = threading.RLock()
        self.rate_limit_lock = threading.RLock()
        
        self._setup_default_permissions()
        self._setup_default_rate_limits()
        self._create_default_admin()
    
    def _setup_default_permissions(self):
        """Setup default role permissions."""
        self.role_permissions = {
            UserRole.ADMIN: {
                Permission.READ_INFERENCE, Permission.WRITE_INFERENCE,
                Permission.READ_MODELS, Permission.WRITE_MODELS,
                Permission.READ_METRICS, Permission.WRITE_CONFIG,
                Permission.ADMIN_USERS, Permission.ADMIN_SYSTEM
            },
            UserRole.USER: {
                Permission.READ_INFERENCE, Permission.WRITE_INFERENCE,
                Permission.READ_MODELS, Permission.READ_METRICS
            },
            UserRole.READONLY: {
                Permission.READ_INFERENCE, Permission.READ_MODELS,
                Permission.READ_METRICS
            },
            UserRole.SERVICE: {
                Permission.READ_INFERENCE, Permission.WRITE_INFERENCE,
                Permission.READ_MODELS
            }
        }
    
    def _setup_default_rate_limits(self):
        """Setup default rate limiting rules."""
        self.rate_limits = {
            'inference_requests': RateLimitRule(
                name='inference_requests',
                requests_per_window=100,
                window_seconds=60,
                endpoints=['/api/v1/inference', '/api/v1/predict'],
                user_roles=[UserRole.USER, UserRole.SERVICE]
            ),
            'admin_operations': RateLimitRule(
                name='admin_operations',
                requests_per_window=20,
                window_seconds=60,
                endpoints=['/api/v1/admin/*'],
                user_roles=[UserRole.ADMIN]
            ),
            'model_operations': RateLimitRule(
                name='model_operations',
                requests_per_window=10,
                window_seconds=60,
                endpoints=['/api/v1/models/*'],
                user_roles=[UserRole.USER, UserRole.ADMIN]
            )
        }
    
    def _create_default_admin(self):
        """Create default admin user if none exists."""
        if not self.users:
            admin_password = self.config.get('default_admin_password', 'admin123')
            self.create_user(
                username='admin',
                password=admin_password,
                role=UserRole.ADMIN
            )
            logger.info("Created default admin user")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt if available, otherwise simple hash."""
        if BCRYPT_AVAILABLE:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        else:
            # Fallback to simple hash (not secure for production)
            return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        if BCRYPT_AVAILABLE and hashed.startswith('$2'):
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        else:
            # Fallback verification
            return hashlib.sha256(password.encode('utf-8')).hexdigest() == hashed
    
    def create_user(self, username: str, password: str, role: UserRole,
                   custom_permissions: Set[Permission] = None) -> bool:
        """Create a new user."""
        with self.user_lock:
            if username in self.users:
                return False
            
            permissions = custom_permissions or self.role_permissions.get(role, set())
            
            user = User(
                username=username,
                password_hash=self._hash_password(password),
                role=role,
                permissions=permissions,
                created_at=datetime.now(timezone.utc)
            )
            
            self.users[username] = user
            
            self._log_security_event(
                'user_created',
                'info',
                None,
                '127.0.0.1',
                f"User '{username}' created with role '{role.value}'"
            )
            
            return True
    
    def authenticate_user(self, username: str, password: str, ip_address: str = '127.0.0.1') -> Optional[str]:
        """Authenticate user and return JWT token."""
        if ip_address in self.blocked_ips:
            self._log_security_event(
                'blocked_ip_attempt',
                'warning',
                username,
                ip_address,
                f"Authentication attempt from blocked IP"
            )
            return None
        
        with self.user_lock:
            user = self.users.get(username)
            
            if not user or not user.is_active:
                self._record_failed_attempt(username, ip_address)
                return None
            
            if not self._verify_password(password, user.password_hash):
                self._record_failed_attempt(username, ip_address)
                return None
            
            # Clear failed attempts on successful login
            if username in self.failed_attempts:
                del self.failed_attempts[username]
            
            # Update last login
            user.last_login = datetime.now(timezone.utc)
            
            # Generate JWT token
            token = self._generate_jwt_token(user)
            
            # Create session
            session_id = secrets.token_urlsafe(32)
            with self.session_lock:
                self.sessions[session_id] = {
                    'username': username,
                    'token': token,
                    'created_at': datetime.now(timezone.utc),
                    'ip_address': ip_address,
                    'last_activity': datetime.now(timezone.utc)
                }
            
            self._log_security_event(
                'user_login',
                'info',
                username,
                ip_address,
                f"User '{username}' logged in successfully"
            )
            
            return token
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user."""
        if not JWT_AVAILABLE:
            # Simple token fallback
            return f"simple_token_{user.username}_{secrets.token_urlsafe(16)}"
        
        payload = {
            'username': user.username,
            'role': user.role.value,
            'permissions': [p.value for p in user.permissions],
            'iat': datetime.now(timezone.utc),
            'exp': datetime.now(timezone.utc) + timedelta(hours=self.jwt_expiry_hours)
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        if not JWT_AVAILABLE:
            # Simple token verification
            if token.startswith('simple_token_'):
                parts = token.split('_')
                if len(parts) >= 3:
                    username = parts[2]
                    if username in self.users and self.users[username].is_active:
                        user = self.users[username]
                        return {
                            'username': user.username,
                            'role': user.role.value,
                            'permissions': [p.value for p in user.permissions]
                        }
            return None
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check if user still exists and is active
            username = payload.get('username')
            if username in self.users and self.users[username].is_active:
                return payload
            
            return None
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def _record_failed_attempt(self, username: str, ip_address: str):
        """Record failed authentication attempt."""
        current_time = datetime.now(timezone.utc)
        
        # Record failed attempt
        self.failed_attempts[username].append(current_time)
        
        # Check for brute force attempts
        recent_attempts = [
            attempt for attempt in self.failed_attempts[username]
            if current_time - attempt < timedelta(minutes=15)
        ]
        
        if len(recent_attempts) >= 5:
            # Block IP after 5 failed attempts in 15 minutes
            self.blocked_ips.add(ip_address)
            
            self._log_security_event(
                'brute_force_detected',
                'critical',
                username,
                ip_address,
                f"Brute force attack detected, blocking IP {ip_address}"
            )
        
        self._log_security_event(
            'authentication_failed',
            'warning',
            username,
            ip_address,
            f"Failed authentication attempt for user '{username}'"
        )
    
    def check_permission(self, token: str, required_permission: Permission) -> bool:
        """Check if token has required permission."""
        payload = self.verify_token(token)
        if not payload:
            return False
        
        user_permissions = payload.get('permissions', [])
        return required_permission.value in user_permissions
    
    def check_rate_limit(self, endpoint: str, user_token: str, ip_address: str) -> bool:
        """Check if request is within rate limits."""
        payload = self.verify_token(user_token)
        if not payload:
            return False
        
        user_role = UserRole(payload.get('role', 'user'))
        current_time = time.time()
        
        with self.rate_limit_lock:
            # Find applicable rate limit rules
            applicable_rules = []
            for rule in self.rate_limits.values():
                if (not rule.endpoints or any(self._match_endpoint(endpoint, pattern) for pattern in rule.endpoints)) and \
                   (not rule.user_roles or user_role in rule.user_roles):
                    applicable_rules.append(rule)
            
            # Check each applicable rule
            for rule in applicable_rules:
                key = f"{rule.name}:{payload['username']}:{ip_address}"
                request_times = self.request_history[key]
                
                # Remove old requests outside the window
                cutoff_time = current_time - rule.window_seconds
                while request_times and request_times[0] < cutoff_time:
                    request_times.popleft()
                
                # Check if limit exceeded
                if len(request_times) >= rule.requests_per_window:
                    self._log_security_event(
                        'rate_limit_exceeded',
                        'warning',
                        payload['username'],
                        ip_address,
                        f"Rate limit exceeded for rule '{rule.name}' on endpoint '{endpoint}'"
                    )
                    return False
                
                # Record this request
                request_times.append(current_time)
            
            return True
    
    def _match_endpoint(self, endpoint: str, pattern: str) -> bool:
        """Check if endpoint matches pattern (supports wildcards)."""
        if pattern.endswith('*'):
            return endpoint.startswith(pattern[:-1])
        return endpoint == pattern
    
    def create_api_key(self, username: str, description: str = "") -> Optional[str]:
        """Create API key for user."""
        with self.user_lock:
            user = self.users.get(username)
            if not user:
                return None
            
            api_key = f"sk-{secrets.token_urlsafe(32)}"
            
            # Store API key (in production, this should be hashed)
            if 'api_keys' not in user.metadata:
                user.metadata['api_keys'] = {}
            
            user.metadata['api_keys'][api_key] = {
                'description': description,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'last_used': None
            }
            
            self._log_security_event(
                'api_key_created',
                'info',
                username,
                '127.0.0.1',
                f"API key created for user '{username}'"
            )
            
            return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify API key and return username."""
        with self.user_lock:
            for username, user in self.users.items():
                if 'api_keys' in user.metadata and api_key in user.metadata['api_keys']:
                    # Update last used
                    user.metadata['api_keys'][api_key]['last_used'] = datetime.now(timezone.utc).isoformat()
                    return username
            
            return None
    
    def revoke_api_key(self, username: str, api_key: str) -> bool:
        """Revoke API key."""
        with self.user_lock:
            user = self.users.get(username)
            if not user or 'api_keys' not in user.metadata:
                return False
            
            if api_key in user.metadata['api_keys']:
                del user.metadata['api_keys'][api_key]
                
                self._log_security_event(
                    'api_key_revoked',
                    'info',
                    username,
                    '127.0.0.1',
                    f"API key revoked for user '{username}'"
                )
                
                return True
            
            return False
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password."""
        with self.user_lock:
            user = self.users.get(username)
            if not user:
                return False
            
            if not self._verify_password(old_password, user.password_hash):
                return False
            
            user.password_hash = self._hash_password(new_password)
            
            self._log_security_event(
                'password_changed',
                'info',
                username,
                '127.0.0.1',
                f"Password changed for user '{username}'"
            )
            
            return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user account."""
        with self.user_lock:
            user = self.users.get(username)
            if not user:
                return False
            
            user.is_active = False
            
            # Invalidate all sessions
            with self.session_lock:
                sessions_to_remove = [
                    session_id for session_id, session in self.sessions.items()
                    if session['username'] == username
                ]
                for session_id in sessions_to_remove:
                    del self.sessions[session_id]
            
            self._log_security_event(
                'user_deactivated',
                'info',
                None,
                '127.0.0.1',
                f"User '{username}' deactivated"
            )
            
            return True
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock IP address."""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            
            self._log_security_event(
                'ip_unblocked',
                'info',
                None,
                ip_address,
                f"IP address {ip_address} unblocked"
            )
            
            return True
        
        return False
    
    def _log_security_event(self, event_type: str, severity: str, user_id: Optional[str],
                           ip_address: str, description: str, metadata: Dict[str, Any] = None):
        """Log security event."""
        event = SecurityEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Trigger callbacks
        for callback in self.security_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Security callback failed: {e}")
    
    def add_security_callback(self, callback: Callable[[SecurityEvent], None]):
        """Add security event callback."""
        self.security_callbacks.append(callback)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary."""
        current_time = datetime.now(timezone.utc)
        
        # Recent events (last 24 hours)
        recent_events = [
            event for event in self.security_events
            if current_time - event.timestamp < timedelta(hours=24)
        ]
        
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event.event_type] += 1
        
        return {
            'total_users': len(self.users),
            'active_users': sum(1 for user in self.users.values() if user.is_active),
            'blocked_ips': list(self.blocked_ips),
            'active_sessions': len(self.sessions),
            'recent_events_24h': len(recent_events),
            'event_breakdown': dict(event_counts),
            'failed_attempt_summary': {
                username: len(attempts)
                for username, attempts in self.failed_attempts.items()
            }
        }
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information."""
        with self.user_lock:
            user = self.users.get(username)
            if not user:
                return None
            
            return {
                'username': user.username,
                'role': user.role.value,
                'permissions': [p.value for p in user.permissions],
                'created_at': user.created_at.isoformat(),
                'last_login': user.last_login.isoformat() if user.last_login else None,
                'is_active': user.is_active,
                'api_keys': len(user.metadata.get('api_keys', {}))
            }
    
    def list_users(self) -> List[Dict[str, Any]]:
        """List all users."""
        with self.user_lock:
            return [
                {
                    'username': user.username,
                    'role': user.role.value,
                    'is_active': user.is_active,
                    'created_at': user.created_at.isoformat(),
                    'last_login': user.last_login.isoformat() if user.last_login else None
                }
                for user in self.users.values()
            ]
    
    def cleanup_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now(timezone.utc)
        expired_sessions = []
        
        with self.session_lock:
            for session_id, session in self.sessions.items():
                # Remove sessions older than JWT expiry
                if current_time - session['created_at'] > timedelta(hours=self.jwt_expiry_hours):
                    expired_sessions.append(session_id)
                # Remove inactive sessions (no activity for 4 hours)
                elif current_time - session['last_activity'] > timedelta(hours=4):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def export_security_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Export security events for analysis."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'severity': event.severity,
                'user_id': event.user_id,
                'ip_address': event.ip_address,
                'description': event.description,
                'metadata': event.metadata
            }
            for event in self.security_events
            if event.timestamp > cutoff_time
        ]
    
    def cleanup(self):
        """Clean up security manager resources."""
        self.cleanup_sessions()
        logger.info("Security manager cleanup completed")


# Legacy compatibility
class EnterpriseSecurity:
    """Legacy enterprise security class for compatibility."""
    
    def __init__(self, policy: SecurityPolicy = None):
        if policy is None:
            policy = SecurityPolicy()
        
        self.policy = policy
        self.logger = logging.getLogger(__name__)
        self.audit_log = []
        
        # Create underlying security manager
        self.security_manager = SecurityManager({
            'jwt_secret': secrets.token_urlsafe(32),
            'default_admin_password': 'admin123'
        })
    
    def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate a security request."""
        # Simplified validation
        if self.policy.require_authentication:
            if 'user' not in request:
                return False
        
        self.log_audit_event("request_validated", request.get('user', 'anonymous'))
        return True
    
    def log_audit_event(self, event: str, user: str, details: Optional[Dict[str, Any]] = None):
        """Log an audit event."""
        if not self.policy.enable_audit_logging:
            return
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'user': user,
            'details': details or {}
        }
        
        self.audit_log.append(audit_entry)
        self.logger.info(f"Audit: {event} by {user}")
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return self.audit_log.copy()
