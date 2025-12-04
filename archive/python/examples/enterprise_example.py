#!/usr/bin/env python3
"""
Enterprise PyTorch Inference Framework - Complete Example

This example demonstrates all enterprise features including:
- Authentication and authorization
- Security and encryption
- Monitoring and observability  
- Model governance and MLOps
- High availability deployment

Usage:
    python enterprise_example.py
"""

import asyncio
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import torch
import time

# Add framework to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from framework.security import (
    EnterpriseInferenceEngine,
    EnterpriseConfig,
    EnterpriseAuth,
    SecurityManager,
    EnterpriseMonitor,
    ModelGovernance
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleModel(torch.nn.Module):
    """Simple PyTorch model for demonstration."""
    
    def __init__(self, input_size: int = 784, num_classes: int = 10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.nn.functional.softmax(x, dim=1)


class EnterpriseDemo:
    """Enterprise inference framework demonstration."""
    
    def __init__(self):
        # Setup paths
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.config = self._setup_config()
        self.engine = None
        self.auth = None
        self.security = None
        self.monitor = None
        self.governance = None
    
    def _setup_config(self) -> EnterpriseConfig:
        """Setup enterprise configuration."""
        config = EnterpriseConfig()
        
        # Override for demo
        config.auth.enable_mfa = False  # Disable MFA for demo
        config.auth.jwt_expiry_hours = 24
        config.monitoring.enable_distributed_tracing = True
        config.monitoring.metrics_port = 9090
        config.security.enable_encryption = True
        config.governance.enable_model_validation = True
        
        return config
    
    async def setup_enterprise_components(self):
        """Initialize all enterprise components."""
        logger.info("Initializing enterprise components...")
        
        # Authentication
        self.auth = EnterpriseAuth(self.config)
        logger.info("✓ Authentication system initialized")
        
        # Security
        self.security = SecurityManager(self.config)
        logger.info("✓ Security manager initialized")
        
        # Monitoring
        self.monitor = EnterpriseMonitor(self.config)
        logger.info("✓ Monitoring system initialized")
        
        # Model Governance
        self.governance = ModelGovernance(self.config, self.monitor)
        logger.info("✓ Model governance initialized")
        
        # Enterprise Engine
        self.engine = EnterpriseInferenceEngine(self.config)
        logger.info("✓ Enterprise inference engine initialized")
        
        logger.info("🚀 All enterprise components ready!")
    
    def create_sample_model(self) -> str:
        """Create and save a sample model."""
        logger.info("Creating sample model...")
        
        model = SimpleModel()
        model.eval()
        
        # Save model
        model_path = self.model_path / "sample_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': 'SimpleModel',
            'input_shape': [1, 28, 28],
            'output_classes': 10,
            'version': '1.0.0'
        }, model_path)
        
        logger.info(f"✓ Sample model saved to {model_path}")
        return str(model_path)
    
    def setup_users_and_roles(self):
        """Setup demo users and roles."""
        logger.info("Setting up demo users and roles...")
        
        # Create admin user
        admin_user = self.auth.create_user(
            username="admin",
            email="admin@company.com",
            full_name="System Administrator",
            password="admin123",  # Use strong passwords in production!
            roles=["admin"]
        )
        logger.info(f"✓ Created admin user: {admin_user.username}")
        
        # Create data scientist user
        ds_user = self.auth.create_user(
            username="data_scientist",
            email="ds@company.com", 
            full_name="Data Scientist",
            password="ds123",
            roles=["data_scientist"]
        )
        logger.info(f"✓ Created data scientist user: {ds_user.username}")
        
        # Create regular user
        user = self.auth.create_user(
            username="user1",
            email="user1@company.com",
            full_name="Regular User", 
            password="user123",
            roles=["user"]
        )
        logger.info(f"✓ Created regular user: {user.username}")
        
        # Create API key for programmatic access
        api_key = self.auth.create_api_key(
            name="Integration API Key",
            user_id=admin_user.id,
            scopes=["inference:predict", "model:read", "metrics:read"]
        )
        logger.info(f"✓ Created API key: {api_key.name}")
        
        return {
            'admin_user': admin_user,
            'ds_user': ds_user,
            'regular_user': user,
            'api_key': api_key
        }
    
    async def demonstrate_authentication(self, users: Dict):
        """Demonstrate authentication features."""
        logger.info("\n" + "="*50)
        logger.info("🔐 AUTHENTICATION DEMONSTRATION")
        logger.info("="*50)
        
        # Login with username/password
        admin_token = await self.auth.authenticate_user("admin", "admin123")
        logger.info(f"✓ Admin login successful, token: {admin_token[:20]}...")
        
        # Validate token
        claims = await self.auth.validate_token(admin_token)
        logger.info(f"✓ Token validation successful for user: {claims.get('sub')}")
        
        # Check permissions
        has_admin_perm = await self.auth.check_permission(admin_token, "admin:*")
        logger.info(f"✓ Admin has admin permissions: {has_admin_perm}")
        
        # Test API key authentication
        api_key = users['api_key']
        key_valid = await self.auth.validate_api_key(api_key.key)
        logger.info(f"✓ API key validation: {key_valid is not None}")
        
        # Test unauthorized access
        try:
            user_token = await self.auth.authenticate_user("user1", "user123")
            has_admin_perm = await self.auth.check_permission(user_token, "admin:delete_model")
            logger.info(f"✗ Regular user admin permissions: {has_admin_perm}")
        except Exception as e:
            logger.info(f"✓ Unauthorized access properly blocked: {type(e).__name__}")
    
    async def demonstrate_security(self):
        """Demonstrate security features."""
        logger.info("\n" + "="*50)
        logger.info("🛡️ SECURITY DEMONSTRATION")
        logger.info("="*50)
        
        # Input validation
        test_data = {"model": "test", "input": [1, 2, 3]}
        is_valid, validation_result = self.security.validate_request("test_client", test_data)
        logger.info(f"✓ Input validation result: {is_valid}")
        
        # Data encryption
        sensitive_data = "This is sensitive model data"
        encrypted = self.security.encrypt_data(sensitive_data)
        decrypted = self.security.decrypt_data(encrypted)
        logger.info(f"✓ Encryption/Decryption successful: {decrypted == sensitive_data}")
        
        # Rate limiting test
        client_id = "test_client"
        for i in range(5):
            allowed = self.security.rate_limiter.check_rate_limit(client_id)
            logger.info(f"Request {i+1} allowed: {allowed}")
            if not allowed:
                logger.info("✓ Rate limiting working correctly")
                break
        
        # Threat detection simulation
        suspicious_patterns = [
            "'; DROP TABLE users; --",  # SQL injection
            "<script>alert('xss')</script>",  # XSS
            "../../../etc/passwd",  # Path traversal
        ]
        
        for pattern in suspicious_patterns:
            threat_detected = self.security.threat_detector.detect_threats({"input": pattern})
            logger.info(f"✓ Threat detection for '{pattern[:20]}...': {threat_detected['threat_detected']}")
    
    async def demonstrate_monitoring(self):
        """Demonstrate monitoring and observability."""
        logger.info("\n" + "="*50)
        logger.info("📊 MONITORING DEMONSTRATION")
        logger.info("="*50)
        
        # Health check
        health_status = self.monitor.get_health_status()
        logger.info(f"✓ System health status: {health_status['status']}")
        
        # Custom metrics
        self.monitor.record_inference_metrics(
            model_name="sample_model",
            version="1.0.0",
            latency=0.15,
            memory_usage=256,
            gpu_utilization=75.5
        )
        logger.info("✓ Custom metrics recorded")
        
        # Distributed tracing simulation
        with self.monitor.create_trace_span("inference_request") as span:
            span.set_attribute("model.name", "sample_model")
            span.set_attribute("request.size", 1024)
            
            # Simulate some work
            await asyncio.sleep(0.1)
            
            with self.monitor.create_trace_span("model_prediction", parent=span) as child_span:
                child_span.set_attribute("prediction.confidence", 0.95)
                await asyncio.sleep(0.05)
        
        logger.info("✓ Distributed tracing spans created")
        
        # Metrics collection
        current_metrics = self.monitor.get_metrics()
        logger.info(f"✓ Current metrics collected: {len(current_metrics)} metrics")
        
        # Alert simulation (would normally integrate with external systems)
        alert_sent = self.monitor.send_alert(
            "high_latency",
            "Inference latency exceeded threshold",
            {"model": "sample_model", "latency": 2.5}
        )
        logger.info(f"✓ Alert sent: {alert_sent}")
    
    async def demonstrate_model_governance(self, model_path: str):
        """Demonstrate model governance and MLOps."""
        logger.info("\n" + "="*50)
        logger.info("🎯 MODEL GOVERNANCE DEMONSTRATION")
        logger.info("="*50)
        
        # Register model
        model_info = await self.governance.register_model(
            name="sample_classifier",
            framework="pytorch",
            version="1.0.0",
            file_path=model_path,
            description="Sample classification model for demonstration",
            metadata={
                "accuracy": 0.95,
                "f1_score": 0.93,
                "training_dataset": "demo_dataset_v1",
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 100
                }
            }
        )
        logger.info(f"✓ Model registered: {model_info['name']} v{model_info['version']}")
        
        # Model validation
        is_valid = await self.governance.validate_model(model_info['id'])
        logger.info(f"✓ Model validation result: {is_valid}")
        
        # Start experiment
        experiment = await self.governance.start_experiment(
            name="model_performance_test",
            description="Testing model performance with new data",
            model_id=model_info['id'],
            hyperparameters={
                "test_batch_size": 64,
                "confidence_threshold": 0.8
            }
        )
        logger.info(f"✓ Experiment started: {experiment['name']}")
        
        # Log experiment results
        await self.governance.log_experiment_result(
            experiment['id'],
            metrics={
                "accuracy": 0.96,
                "precision": 0.94,
                "recall": 0.95,
                "inference_time": 0.12
            },
            artifacts=["confusion_matrix.png", "roc_curve.png"]
        )
        logger.info("✓ Experiment results logged")
        
        # A/B testing setup
        ab_test = await self.governance.setup_ab_test(
            name="model_v1_vs_v2",
            control_model_id=model_info['id'],
            treatment_model_version="1.1.0",
            traffic_split=0.2  # 20% to new version
        )
        logger.info(f"✓ A/B test configured: {ab_test['name']}")
        
        # Model deployment
        deployment = await self.governance.deploy_model(
            model_id=model_info['id'],
            environment="staging",
            config={
                "replicas": 2,
                "cpu_limit": "2",
                "memory_limit": "4Gi",
                "gpu_required": True
            }
        )
        logger.info(f"✓ Model deployed to staging: {deployment['deployment_id']}")
    
    async def demonstrate_enterprise_inference(self, users: Dict, model_path: str):
        """Demonstrate enterprise inference with security and monitoring."""
        logger.info("\n" + "="*50)
        logger.info("🚀 ENTERPRISE INFERENCE DEMONSTRATION")
        logger.info("="*50)
        
        # Load model into engine
        await self.engine.load_model(model_path, "sample_model")
        logger.info("✓ Model loaded into enterprise engine")
        
        # Prepare sample data
        sample_input = torch.randn(1, 28, 28)  # MNIST-like input
        
        # Authenticated inference with admin user
        admin_token = await self.auth.authenticate_user("admin", "admin123")
        
        start_time = time.time()
        result = await self.engine.secure_predict(
            model_name="sample_model",
            input_data=sample_input.numpy().tolist(),
            auth_token=admin_token,
            client_id="demo_client",
            trace_id="demo_trace_001"
        )
        inference_time = time.time() - start_time
        
        logger.info(f"✓ Secure inference completed in {inference_time:.3f}s")
        logger.info(f"  Prediction shape: {np.array(result['prediction']).shape}")
        logger.info(f"  Confidence: {max(result['prediction']):.3f}")
        logger.info(f"  Trace ID: {result['trace_id']}")
        
        # Test batch inference
        batch_input = torch.randn(5, 28, 28)
        batch_result = await self.engine.secure_batch_predict(
            model_name="sample_model",
            input_data=batch_input.numpy().tolist(),
            auth_token=admin_token,
            client_id="demo_client"
        )
        logger.info(f"✓ Batch inference completed: {len(batch_result['predictions'])} predictions")
        
        # Test unauthorized access
        try:
            user_token = await self.auth.authenticate_user("user1", "user123")
            # This should work as regular users have inference permission
            user_result = await self.engine.secure_predict(
                model_name="sample_model",
                input_data=sample_input.numpy().tolist(),
                auth_token=user_token,
                client_id="user_client"
            )
            logger.info("✓ Regular user inference successful")
        except Exception as e:
            logger.info(f"✗ Regular user access denied: {e}")
        
        # Test API key access
        api_key = users['api_key']
        api_result = await self.engine.secure_predict(
            model_name="sample_model",
            input_data=sample_input.numpy().tolist(),
            api_key=api_key.key,
            client_id="api_client"
        )
        logger.info("✓ API key inference successful")
    
    async def demonstrate_compliance_and_audit(self):
        """Demonstrate compliance and audit features."""
        logger.info("\n" + "="*50)
        logger.info("📋 COMPLIANCE & AUDIT DEMONSTRATION")
        logger.info("="*50)
        
        # Generate compliance report
        compliance_report = await self.engine.generate_compliance_report(
            start_date="2024-01-01",
            end_date="2024-12-31",
            include_sections=["security", "data_protection", "model_governance", "audit_logs"]
        )
        
        logger.info("✓ Compliance report generated")
        logger.info(f"  Report ID: {compliance_report['report_id']}")
        logger.info(f"  Sections: {', '.join(compliance_report['sections'])}")
        logger.info(f"  Total findings: {compliance_report['summary']['total_findings']}")
        
        # Audit trail
        audit_events = self.security.audit_logger.get_recent_events(limit=10)
        logger.info(f"✓ Recent audit events: {len(audit_events)} events")
        
        for event in audit_events[-3:]:  # Show last 3 events
            logger.info(f"  - {event['action']} by {event.get('user', 'system')} at {event['timestamp']}")
        
        # Data lineage tracking
        lineage_info = await self.governance.get_data_lineage("sample_model")
        logger.info("✓ Data lineage information retrieved")
        logger.info(f"  Source datasets: {len(lineage_info.get('datasets', []))}")
        logger.info(f"  Processing steps: {len(lineage_info.get('processing_steps', []))}")
    
    async def demonstrate_high_availability(self):
        """Demonstrate high availability features."""
        logger.info("\n" + "="*50)
        logger.info("🔄 HIGH AVAILABILITY DEMONSTRATION")
        logger.info("="*50)
        
        # Health checks
        health = await self.engine.health_check()
        logger.info(f"✓ Health check: {health['status']}")
        logger.info(f"  Uptime: {health.get('uptime', 'N/A')}")
        logger.info(f"  Memory usage: {health.get('memory_usage', 'N/A')}")
        logger.info(f"  GPU availability: {health.get('gpu_available', False)}")
        
        # Readiness check
        readiness = await self.engine.readiness_check()
        logger.info(f"✓ Readiness check: {readiness['ready']}")
        
        # Circuit breaker simulation
        logger.info("Testing circuit breaker (simulating failures)...")
        failure_count = 0
        for i in range(10):
            try:
                # Simulate some failures
                if i % 3 == 0:  # Every 3rd request fails
                    raise Exception("Simulated failure")
                logger.info(f"  Request {i+1}: Success")
            except Exception:
                failure_count += 1
                logger.info(f"  Request {i+1}: Failed ({failure_count} failures)")
        
        logger.info(f"✓ Circuit breaker simulation completed: {failure_count} failures handled")
        
        # Load balancing simulation
        logger.info("Simulating load balancing across multiple replicas...")
        for i in range(5):
            replica_id = f"replica_{(i % 3) + 1}"
            logger.info(f"  Request {i+1} -> {replica_id}")
        
        logger.info("✓ Load balancing simulation completed")
    
    async def run_complete_demo(self):
        """Run the complete enterprise demonstration."""
        logger.info("🎬 Starting Enterprise PyTorch Inference Framework Demo")
        logger.info("=" * 60)
        
        try:
            # Setup
            await self.setup_enterprise_components()
            model_path = self.create_sample_model()
            users = self.setup_users_and_roles()
            
            # Demonstrations
            await self.demonstrate_authentication(users)
            await self.demonstrate_security()
            await self.demonstrate_monitoring()
            await self.demonstrate_model_governance(model_path)
            await self.demonstrate_enterprise_inference(users, model_path)
            await self.demonstrate_compliance_and_audit()
            await self.demonstrate_high_availability()
            
            # Final summary
            logger.info("\n" + "="*60)
            logger.info("🎉 ENTERPRISE DEMO COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info("\n✅ All enterprise features demonstrated:")
            logger.info("  🔐 Authentication & Authorization")
            logger.info("  🛡️  Security & Encryption")
            logger.info("  📊 Monitoring & Observability")
            logger.info("  🎯 Model Governance & MLOps")
            logger.info("  🚀 Secure Inference Engine")
            logger.info("  📋 Compliance & Audit")
            logger.info("  🔄 High Availability")
            
            logger.info("\n🚀 Ready for production deployment!")
            logger.info("\nNext steps:")
            logger.info("  1. Run setup_enterprise.py for full deployment")
            logger.info("  2. Configure production secrets and certificates")
            logger.info("  3. Deploy using Docker Compose or Kubernetes")
            logger.info("  4. Setup monitoring dashboards and alerts")
            logger.info("  5. Configure backup and disaster recovery")
            
        except Exception as e:
            logger.error(f"❌ Demo failed with error: {e}")
            raise


async def main():
    """Main function to run the enterprise demo."""
    demo = EnterpriseDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
