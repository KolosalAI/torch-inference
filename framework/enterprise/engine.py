"""
Enterprise inference engine with comprehensive enterprise features.

This module integrates all enterprise components to provide:
- Secure authentication and authorization
- Advanced monitoring and observability
- Model governance and MLOps
- High availability and scalability
- Compliance and audit capabilities
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager

from ..core.inference_engine import InferenceEngine
from ..core.base_model import BaseModel
from .config import EnterpriseConfig
from .auth import EnterpriseAuth, User, Permission
from .security import SecurityManager, SecurityEvent
from .monitoring import EnterpriseMonitor
from .governance import ModelGovernance, ModelPerformanceMetrics


logger = logging.getLogger(__name__)


@dataclass
class EnterpriseInferenceRequest:
    """Enterprise inference request with security context."""
    id: str
    inputs: Any
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    trace_id: Optional[str] = None
    permissions: List[str] = None
    priority: int = 0
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []


@dataclass
class EnterpriseInferenceResponse:
    """Enterprise inference response with metadata."""
    request_id: str
    result: Any
    model_id: str
    model_version: str
    processing_time_ms: float
    timestamp: datetime
    trace_id: Optional[str] = None
    compliance_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.compliance_metadata is None:
            self.compliance_metadata = {}


class EnterpriseInferenceEngine:
    """
    Enterprise-grade inference engine with comprehensive security, monitoring,
    and governance features.
    """
    
    def __init__(self, model: BaseModel, config: EnterpriseConfig):
        self.model = model
        self.config = config
        
        # Initialize base inference engine
        self.base_engine = InferenceEngine(model, config.inference)
        
        # Initialize enterprise components
        self.auth = EnterpriseAuth(config)
        self.security_manager = SecurityManager(config)
        self.monitor = EnterpriseMonitor(config)
        self.governance = ModelGovernance(config)
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Setup integration
        self._setup_integration()
        
        logger.info("Enterprise inference engine initialized")
    
    def _setup_integration(self) -> None:
        """Setup integration between components."""
        # Add security alert callbacks
        self.security_manager.add_alert_callback(self._handle_security_alert)
        
        # Add monitoring alert callbacks
        if self.monitor.alert_manager:
            self.monitor.alert_manager.add_alert_callback(self._handle_monitoring_alert)
        
        # Setup audit logging for model access
        self._setup_audit_logging()
    
    def _setup_audit_logging(self) -> None:
        """Setup audit logging for compliance."""
        # This would typically integrate with external audit systems
        logger.info("Audit logging configured")
    
    def _handle_security_alert(self, alert) -> None:
        """Handle security alerts."""
        logger.warning(f"Security Alert: {alert.message}")
        
        # In production, this would:
        # - Send notifications to security team
        # - Update threat intelligence
        # - Trigger automated responses
    
    def _handle_monitoring_alert(self, alert) -> None:
        """Handle monitoring alerts."""
        logger.warning(f"Monitoring Alert: {alert.message}")
        
        # In production, this would:
        # - Send notifications to operations team
        # - Trigger auto-scaling if needed
        # - Update dashboards
    
    async def start(self) -> None:
        """Start the enterprise inference engine."""
        logger.info("Starting enterprise inference engine...")
        
        # Start base engine
        await self.base_engine.start()
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        logger.info("Enterprise inference engine started successfully")
    
    async def stop(self) -> None:
        """Stop the enterprise inference engine."""
        logger.info("Stopping enterprise inference engine...")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Stop base engine
        await self.base_engine.stop()
        
        logger.info("Enterprise inference engine stopped")
    
    async def predict(self, request: EnterpriseInferenceRequest) -> EnterpriseInferenceResponse:
        """
        Perform secure, monitored inference with full enterprise features.
        
        Args:
            request: Enterprise inference request with security context
            
        Returns:
            Enterprise inference response with metadata
            
        Raises:
            PermissionError: If user lacks required permissions
            SecurityError: If security validation fails
            ValidationError: If input validation fails
        """
        start_time = time.time()
        self.request_count += 1
        
        # Create trace context
        if self.monitor.distributed_tracing:
            trace_context = self.monitor.distributed_tracing.get_current_trace_context()
            if trace_context:
                request.trace_id = trace_context.trace_id
        
        try:
            # 1. Authentication & Authorization
            await self._authenticate_and_authorize(request)
            
            # 2. Security validation
            await self._validate_security(request)
            
            # 3. Input validation and sanitization
            validated_inputs = await self._validate_and_sanitize_inputs(request)
            
            # 4. Rate limiting check
            await self._check_rate_limits(request)
            
            # 5. Model selection and routing (for A/B tests)
            model_info = await self._select_model(request)
            
            # 6. Perform inference
            result = await self._perform_inference(validated_inputs, request)
            
            # 7. Post-process and validate outputs
            processed_result = await self._process_outputs(result, request)
            
            # 8. Record metrics and audit logs
            processing_time = (time.time() - start_time) * 1000
            await self._record_success_metrics(request, processing_time, model_info)
            
            # 9. Create enterprise response
            response = EnterpriseInferenceResponse(
                request_id=request.id,
                result=processed_result,
                model_id=model_info.get("id", "unknown"),
                model_version=model_info.get("version", "unknown"),
                processing_time_ms=processing_time,
                timestamp=datetime.now(timezone.utc),
                trace_id=request.trace_id,
                compliance_metadata=self._create_compliance_metadata(request)
            )
            
            return response
            
        except Exception as e:
            # Record error metrics
            self.error_count += 1
            processing_time = (time.time() - start_time) * 1000
            await self._record_error_metrics(request, str(e), processing_time)
            
            # Log security event if relevant
            if isinstance(e, PermissionError):
                self.security_manager.log_security_event(
                    SecurityEvent.AUTHORIZATION_ERROR,
                    request.user_id,
                    f"Permission denied: {e}",
                    {"request_id": request.id, "permission_required": str(e)},
                    request.ip_address
                )
            
            raise
    
    async def predict_batch(self, requests: List[EnterpriseInferenceRequest]) -> List[EnterpriseInferenceResponse]:
        """Perform batch inference with enterprise features."""
        if not requests:
            return []
        
        # Process requests concurrently while respecting rate limits
        semaphore = asyncio.Semaphore(self.config.scaling.max_replicas)
        
        async def process_single_request(req):
            async with semaphore:
                return await self.predict(req)
        
        tasks = [process_single_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch request {requests[i].id} failed: {response}")
                # Could create error response here
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    async def _authenticate_and_authorize(self, request: EnterpriseInferenceRequest) -> User:
        """Authenticate user and check permissions."""
        if not request.user_id and not request.session_id:
            raise PermissionError("Authentication required")
        
        # Get user from session or user_id
        user = None
        if request.session_id:
            session = self.auth.session_manager.get_session(request.session_id)
            if not session or not session.is_valid():
                raise PermissionError("Invalid or expired session")
            user = self.auth.rbac_manager.get_user(session.user_id)
        elif request.user_id:
            user = self.auth.rbac_manager.get_user(request.user_id)
        
        if not user or not user.is_active:
            raise PermissionError("User not found or inactive")
        
        # Check inference permission
        if not user.has_permission(Permission.INFERENCE_PREDICT, self.auth.rbac_manager):
            raise PermissionError("Insufficient permissions for inference")
        
        # Check tenant isolation
        if (self.config.rbac.tenant_isolation and request.tenant_id and 
            user.tenant_id != request.tenant_id):
            raise PermissionError("Tenant access denied")
        
        # Update request with user permissions
        request.permissions = self.auth.rbac_manager._get_user_permissions(user.id)
        
        return user
    
    async def _validate_security(self, request: EnterpriseInferenceRequest) -> None:
        """Perform security validation."""
        # Validate request against security policies
        client_id = request.user_id or request.ip_address or "anonymous"
        endpoint = "inference"
        
        is_valid, error_msg = self.security_manager.validate_request(
            client_id, request.inputs, endpoint
        )
        
        if not is_valid:
            raise ValueError(f"Security validation failed: {error_msg}")
    
    async def _validate_and_sanitize_inputs(self, request: EnterpriseInferenceRequest) -> Any:
        """Validate and sanitize input data."""
        # Input validation based on model requirements
        if hasattr(self.model, 'validate_input'):
            is_valid, error_msg = self.model.validate_input(request.inputs)
            if not is_valid:
                raise ValueError(f"Input validation failed: {error_msg}")
        
        # Security-based input sanitization
        if hasattr(request.inputs, 'items'):  # Dictionary-like
            sanitized = {}
            for key, value in request.inputs.items():
                if isinstance(value, str):
                    sanitized[key] = self.security_manager.input_validator.sanitize_input(value)
                else:
                    sanitized[key] = value
            return sanitized
        
        return request.inputs
    
    async def _check_rate_limits(self, request: EnterpriseInferenceRequest) -> None:
        """Check rate limiting."""
        client_id = request.user_id or request.ip_address or "anonymous"
        
        allowed, rate_info = self.security_manager.rate_limiter.is_allowed(client_id, "inference")
        
        if not allowed:
            raise RuntimeError(f"Rate limit exceeded. Retry after {rate_info['retry_after']} seconds")
    
    async def _select_model(self, request: EnterpriseInferenceRequest) -> Dict[str, Any]:
        """Select model for inference (handles A/B testing)."""
        # In production, this would check for active A/B tests
        # and route requests accordingly
        
        return {
            "id": self.model.model_info.get("id", "default"),
            "version": self.model.model_info.get("version", "1.0.0"),
            "name": self.model.model_info.get("name", "default_model")
        }
    
    async def _perform_inference(self, inputs: Any, request: EnterpriseInferenceRequest) -> Any:
        """Perform the actual inference."""
        # Add tracing information
        if self.monitor.distributed_tracing:
            span = self.monitor.distributed_tracing.create_span("model_inference")
            span.set_attribute("user_id", request.user_id or "anonymous")
            span.set_attribute("model_id", self.model.model_info.get("id", "unknown"))
            
        # Perform inference using base engine
        result = await self.base_engine.predict(inputs, request.priority, request.timeout)
        
        return result
    
    async def _process_outputs(self, result: Any, request: EnterpriseInferenceRequest) -> Any:
        """Post-process and validate outputs."""
        # Output sanitization for security
        if self.config.security.sanitize_outputs:
            # Apply output sanitization rules
            if isinstance(result, dict):
                sanitized = {}
                for key, value in result.items():
                    if isinstance(value, str):
                        sanitized[key] = self.security_manager.input_validator.sanitize_input(value)
                    else:
                        sanitized[key] = value
                return sanitized
        
        return result
    
    async def _record_success_metrics(self, request: EnterpriseInferenceRequest, 
                                    processing_time: float, model_info: Dict[str, Any]) -> None:
        """Record successful inference metrics."""
        # Monitor HTTP-style request
        self.monitor.record_request(
            method="POST",
            endpoint="/predict",
            status="200",
            duration=processing_time / 1000,  # Convert to seconds
            user_id=request.user_id
        )
        
        # Monitor inference-specific metrics
        self.monitor.record_inference(
            model=model_info["id"],
            duration=processing_time / 1000,
            status="success",
            tenant=request.tenant_id or "default"
        )
        
        # Record model performance for governance
        performance_metrics = ModelPerformanceMetrics(
            model_id=model_info["id"],
            version=model_info["version"],
            timestamp=datetime.now(timezone.utc),
            latency_p95_ms=processing_time,  # Single request latency
            throughput_rps=1.0 / (processing_time / 1000),
            prediction_count=1,
            error_count=0
        )
        
        self.governance.record_model_performance(
            model_info["id"],
            model_info["version"],
            performance_metrics
        )
        
        # Audit logging
        self.security_manager.audit_logger.log_action(
            user_id=request.user_id,
            action="inference_request",
            resource=f"model:{model_info['id']}",
            details={
                "model_version": model_info["version"],
                "processing_time_ms": processing_time,
                "input_size": len(str(request.inputs)) if request.inputs else 0
            },
            ip_address=request.ip_address,
            user_agent=request.user_agent,
            tenant_id=request.tenant_id
        )
    
    async def _record_error_metrics(self, request: EnterpriseInferenceRequest, 
                                  error: str, processing_time: float) -> None:
        """Record error metrics."""
        # Monitor HTTP-style error
        self.monitor.record_request(
            method="POST",
            endpoint="/predict",
            status="500",
            duration=processing_time / 1000,
            user_id=request.user_id
        )
        
        # Monitor inference error
        model_info = await self._select_model(request)
        self.monitor.record_inference(
            model=model_info["id"],
            duration=processing_time / 1000,
            status="error",
            tenant=request.tenant_id or "default"
        )
        
        # Record error in governance
        performance_metrics = ModelPerformanceMetrics(
            model_id=model_info["id"],
            version=model_info["version"],
            timestamp=datetime.now(timezone.utc),
            latency_p95_ms=processing_time,
            prediction_count=0,
            error_count=1
        )
        
        self.governance.record_model_performance(
            model_info["id"],
            model_info["version"],
            performance_metrics
        )
        
        # Audit logging
        self.security_manager.audit_logger.log_action(
            user_id=request.user_id,
            action="inference_error",
            resource=f"model:{model_info['id']}",
            details={
                "error": error,
                "processing_time_ms": processing_time
            },
            ip_address=request.ip_address,
            tenant_id=request.tenant_id,
            success=False
        )
    
    def _create_compliance_metadata(self, request: EnterpriseInferenceRequest) -> Dict[str, Any]:
        """Create compliance metadata for response."""
        metadata = {
            "data_retention_policy": "standard",
            "processing_purpose": "ml_inference",
            "user_consent": True,  # Would check actual consent in production
        }
        
        # Add GDPR-specific metadata if enabled
        if any(std.value == "gdpr" for std in self.config.compliance.enabled_standards):
            metadata.update({
                "gdpr_lawful_basis": "legitimate_interest",
                "data_controller": "torch_inference_service",
                "retention_period_days": self.config.compliance.gdpr_data_retention_days
            })
        
        return metadata
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "components": {},
            "metrics": {},
            "version": self.config.version
        }
        
        # Check base engine health
        base_health = await self.base_engine.health_check()
        health_status["components"]["inference_engine"] = base_health
        
        # Check enterprise components
        health_status["components"]["monitoring"] = self.monitor.get_health_status()
        health_status["components"]["security"] = self.security_manager.get_security_metrics()
        
        # Overall metrics
        health_status["metrics"] = {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "requests_per_second": self.request_count / (time.time() - self.start_time)
        }
        
        # Determine overall health
        if health_status["metrics"]["error_rate"] > 0.1:  # >10% error rate
            health_status["status"] = "unhealthy"
        elif any(not comp.get("healthy", True) for comp in health_status["components"].values()):
            health_status["status"] = "degraded"
        
        return health_status
    
    def get_enterprise_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_health": self.monitor.get_health_status(),
            "security_metrics": self.security_manager.get_security_metrics(),
            "monitoring_data": self.monitor.get_monitoring_dashboard_data(),
            "governance_data": self.governance.get_governance_dashboard(),
            "performance_metrics": {
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "uptime_hours": (time.time() - self.start_time) / 3600
            }
        }
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "compliance_standards": [std.value for std in self.config.compliance.enabled_standards],
            "audit_summary": {
                "total_audit_entries": len(self.security_manager.audit_logger.log_entries),
                "security_events": len([
                    entry for entry in self.security_manager.audit_logger.log_entries
                    if "security" in entry.action
                ]),
                "model_accesses": len([
                    entry for entry in self.security_manager.audit_logger.log_entries
                    if entry.action == "inference_request"
                ])
            },
            "data_governance": {
                "models_under_governance": len(self.governance.model_registry.list_models()),
                "active_experiments": len([
                    exp for exp in self.governance.experiment_tracker.experiments.values()
                    if exp.status.value == "running"
                ])
            },
            "security_posture": self.security_manager.get_security_metrics()
        }
    
    @asynccontextmanager
    async def enterprise_context(self):
        """Context manager for enterprise engine lifecycle."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


# Factory functions for creating enterprise engines
async def create_enterprise_engine(model: BaseModel, config: EnterpriseConfig) -> EnterpriseInferenceEngine:
    """Create and initialize enterprise inference engine."""
    engine = EnterpriseInferenceEngine(model, config)
    await engine.start()
    return engine


def create_enterprise_config_from_env() -> EnterpriseConfig:
    """Create enterprise configuration from environment variables."""
    config = EnterpriseConfig.from_env()
    config.validate()
    return config
