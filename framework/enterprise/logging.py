"""
Enterprise logging system for multi-GPU inference.
Provides structured logging, audit trails, and compliance features.
"""

import logging
import logging.handlers
import json
import time
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
import threading
import queue
from pathlib import Path

class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """Log categories for enterprise logging."""
    SYSTEM = "system"
    SECURITY = "security"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    INFERENCE = "inference"
    ERROR = "error"
    API = "api"
    COMPLIANCE = "compliance"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    category: str
    message: str
    component: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)

@dataclass
class AuditEvent:
    """Audit event entry."""
    timestamp: str
    event_type: str
    user_id: str
    resource: str
    action: str
    outcome: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class EnterpriseLogger:
    """Enterprise-grade logging system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.log_level = LogLevel(self.config.get('level', 'INFO'))
        self.log_directory = Path(self.config.get('directory', 'logs'))
        self.max_file_size = self.config.get('max_file_size_mb', 100) * 1024 * 1024
        self.backup_count = self.config.get('backup_count', 5)
        self.enable_json = self.config.get('json_format', True)
        self.enable_audit = self.config.get('audit_logging', True)
        self.enable_compliance = self.config.get('compliance_logging', True)
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.loggers: Dict[str, logging.Logger] = {}
        self.audit_logger: Optional[logging.Logger] = None
        self.compliance_logger: Optional[logging.Logger] = None
        
        # Async logging
        self.log_queue = queue.Queue()
        self.logging_active = False
        self.logging_thread: Optional[threading.Thread] = None
        
        self._setup_loggers()
        self._start_async_logging()
    
    def _setup_loggers(self):
        """Setup category-specific loggers."""
        for category in LogCategory:
            logger = logging.getLogger(f"enterprise.{category.value}")
            logger.setLevel(getattr(logging, self.log_level.value))
            
            # Remove existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create rotating file handler
            log_file = self.log_directory / f"{category.value}.log"
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            
            # Set formatter
            if self.enable_json:
                formatter = JsonFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            logger.propagate = False
            
            self.loggers[category.value] = logger
        
        # Setup audit logger
        if self.enable_audit:
            self.audit_logger = logging.getLogger("enterprise.audit")
            self.audit_logger.setLevel(logging.INFO)
            
            audit_file = self.log_directory / "audit.log"
            audit_handler = logging.handlers.RotatingFileHandler(
                audit_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            audit_handler.setFormatter(JsonFormatter())
            self.audit_logger.addHandler(audit_handler)
            self.audit_logger.propagate = False
        
        # Setup compliance logger
        if self.enable_compliance:
            self.compliance_logger = logging.getLogger("enterprise.compliance")
            self.compliance_logger.setLevel(logging.INFO)
            
            compliance_file = self.log_directory / "compliance.log"
            compliance_handler = logging.handlers.RotatingFileHandler(
                compliance_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            compliance_handler.setFormatter(JsonFormatter())
            self.compliance_logger.addHandler(compliance_handler)
            self.compliance_logger.propagate = False
    
    def _start_async_logging(self):
        """Start async logging thread."""
        self.logging_active = True
        self.logging_thread = threading.Thread(
            target=self._async_logging_worker,
            daemon=True
        )
        self.logging_thread.start()
    
    def _async_logging_worker(self):
        """Async logging worker thread."""
        while self.logging_active:
            try:
                log_entry = self.log_queue.get(timeout=1.0)
                self._write_log_entry(log_entry)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Async logging error: {e}")  # Fallback logging
    
    def _write_log_entry(self, log_entry: LogEntry):
        """Write log entry to appropriate logger."""
        logger = self.loggers.get(log_entry.category)
        if not logger:
            logger = self.loggers.get(LogCategory.SYSTEM.value)
        
        level = getattr(logging, log_entry.level.upper(), logging.INFO)
        
        if self.enable_json:
            # Log as JSON
            logger.log(level, log_entry.to_json())
        else:
            # Log as formatted message
            message = f"[{log_entry.component}]"
            if log_entry.request_id:
                message += f" [{log_entry.request_id}]"
            message += f" {log_entry.message}"
            
            logger.log(level, message)
    
    def log(self, level: LogLevel, category: LogCategory, message: str, 
            component: str, request_id: Optional[str] = None,
            user_id: Optional[str] = None, session_id: Optional[str] = None,
            metadata: Dict[str, Any] = None):
        """Log a message."""
        log_entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            category=category.value,
            message=message,
            component=component,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        # Add to async queue
        try:
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            # Fallback to synchronous logging
            self._write_log_entry(log_entry)
    
    def debug(self, category: LogCategory, message: str, component: str, **kwargs):
        """Log debug message."""
        self.log(LogLevel.DEBUG, category, message, component, **kwargs)
    
    def info(self, category: LogCategory, message: str, component: str, **kwargs):
        """Log info message."""
        self.log(LogLevel.INFO, category, message, component, **kwargs)
    
    def warning(self, category: LogCategory, message: str, component: str, **kwargs):
        """Log warning message."""
        self.log(LogLevel.WARNING, category, message, component, **kwargs)
    
    def error(self, category: LogCategory, message: str, component: str, **kwargs):
        """Log error message."""
        self.log(LogLevel.ERROR, category, message, component, **kwargs)
    
    def critical(self, category: LogCategory, message: str, component: str, **kwargs):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, category, message, component, **kwargs)
    
    def audit(self, event_type: str, user_id: str, resource: str, action: str,
              outcome: str, details: Dict[str, Any], ip_address: Optional[str] = None,
              user_agent: Optional[str] = None):
        """Log audit event."""
        if not self.audit_logger:
            return
        
        audit_event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.audit_logger.info(json.dumps(asdict(audit_event), default=str))
    
    def compliance(self, regulation: str, event: str, details: Dict[str, Any],
                  compliance_status: str = "compliant"):
        """Log compliance event."""
        if not self.compliance_logger:
            return
        
        compliance_event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regulation': regulation,
            'event': event,
            'compliance_status': compliance_status,
            'details': details
        }
        
        self.compliance_logger.info(json.dumps(compliance_event, default=str))
    
    def performance(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Log performance metrics."""
        self.info(
            LogCategory.PERFORMANCE,
            f"Operation '{operation}' completed in {duration:.3f}s",
            "performance_monitor",
            metadata={
                'operation': operation,
                'duration_seconds': duration,
                **(metadata or {})
            }
        )
    
    def inference_request(self, request_id: str, model: str, input_shape: List[int],
                         duration: float, success: bool, error: Optional[str] = None):
        """Log inference request."""
        outcome = "success" if success else "failure"
        message = f"Inference request {outcome}: {model}"
        
        metadata = {
            'model': model,
            'input_shape': input_shape,
            'duration_seconds': duration,
            'success': success
        }
        
        if error:
            metadata['error'] = error
        
        level = LogLevel.INFO if success else LogLevel.ERROR
        self.log(
            level,
            LogCategory.INFERENCE,
            message,
            "inference_engine",
            request_id=request_id,
            metadata=metadata
        )
    
    def security_event(self, event_type: str, severity: str, description: str,
                      user_id: Optional[str] = None, ip_address: Optional[str] = None,
                      metadata: Dict[str, Any] = None):
        """Log security event."""
        level = LogLevel.CRITICAL if severity == "high" else LogLevel.WARNING
        
        self.log(
            level,
            LogCategory.SECURITY,
            f"Security event: {description}",
            "security_monitor",
            user_id=user_id,
            metadata={
                'event_type': event_type,
                'severity': severity,
                'ip_address': ip_address,
                **(metadata or {})
            }
        )
    
    def api_request(self, request_id: str, method: str, endpoint: str,
                   status_code: int, duration: float, user_id: Optional[str] = None,
                   ip_address: Optional[str] = None):
        """Log API request."""
        message = f"{method} {endpoint} - {status_code} ({duration:.3f}s)"
        level = LogLevel.ERROR if status_code >= 400 else LogLevel.INFO
        
        self.log(
            level,
            LogCategory.API,
            message,
            "api_gateway",
            request_id=request_id,
            user_id=user_id,
            metadata={
                'method': method,
                'endpoint': endpoint,
                'status_code': status_code,
                'duration_seconds': duration,
                'ip_address': ip_address
            }
        )
    
    def get_log_files(self) -> Dict[str, str]:
        """Get paths to all log files."""
        log_files = {}
        
        for category in LogCategory:
            log_file = self.log_directory / f"{category.value}.log"
            if log_file.exists():
                log_files[category.value] = str(log_file)
        
        if self.audit_logger:
            audit_file = self.log_directory / "audit.log"
            if audit_file.exists():
                log_files["audit"] = str(audit_file)
        
        if self.compliance_logger:
            compliance_file = self.log_directory / "compliance.log"
            if compliance_file.exists():
                log_files["compliance"] = str(compliance_file)
        
        return log_files
    
    def search_logs(self, category: str, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search logs for specific patterns."""
        log_file = self.log_directory / f"{category}.log"
        
        if not log_file.exists():
            return []
        
        results = []
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if query.lower() in line.lower():
                        if self.enable_json:
                            try:
                                results.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                results.append({'raw_line': line.strip()})
                        else:
                            results.append({'raw_line': line.strip()})
                        
                        if len(results) >= limit:
                            break
        except Exception as e:
            self.error(
                LogCategory.SYSTEM,
                f"Failed to search logs: {e}",
                "enterprise_logger"
            )
        
        return results
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'log_directory': str(self.log_directory),
            'categories': {},
            'total_files': 0,
            'total_size_mb': 0.0
        }
        
        for category in LogCategory:
            log_file = self.log_directory / f"{category.value}.log"
            if log_file.exists():
                file_size = log_file.stat().st_size
                stats['categories'][category.value] = {
                    'file_path': str(log_file),
                    'size_mb': file_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                }
                stats['total_size_mb'] += file_size / (1024 * 1024)
                stats['total_files'] += 1
        
        # Add special log files
        for log_name in ['audit', 'compliance']:
            log_file = self.log_directory / f"{log_name}.log"
            if log_file.exists():
                file_size = log_file.stat().st_size
                stats['categories'][log_name] = {
                    'file_path': str(log_file),
                    'size_mb': file_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                }
                stats['total_size_mb'] += file_size / (1024 * 1024)
                stats['total_files'] += 1
        
        return stats
    
    def rotate_logs(self):
        """Manually rotate all log files."""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()
        
        if self.audit_logger:
            for handler in self.audit_logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()
        
        if self.compliance_logger:
            for handler in self.compliance_logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()
    
    def cleanup(self):
        """Clean up logging resources."""
        self.logging_active = False
        
        if self.logging_thread:
            self.logging_thread.join(timeout=2.0)
        
        # Close all handlers
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.close()
        
        if self.audit_logger:
            for handler in self.audit_logger.handlers:
                handler.close()
        
        if self.compliance_logger:
            for handler in self.compliance_logger.handlers:
                handler.close()

class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        
        # Add extra fields
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
        
        return json.dumps(log_entry, default=str)

# Global enterprise logger instance
_enterprise_logger: Optional[EnterpriseLogger] = None

def get_enterprise_logger(config: Dict[str, Any] = None) -> EnterpriseLogger:
    """Get global enterprise logger instance."""
    global _enterprise_logger
    
    if _enterprise_logger is None:
        _enterprise_logger = EnterpriseLogger(config)
    
    return _enterprise_logger

def setup_enterprise_logging(config: Dict[str, Any] = None):
    """Setup enterprise logging."""
    global _enterprise_logger
    _enterprise_logger = EnterpriseLogger(config)
    return _enterprise_logger
