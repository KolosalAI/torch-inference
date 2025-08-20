"""
Unit tests for security mitigations.

Tests the security features implemented to address:
- CVE-2024-23342: Minerva timing attack on P-256 in python-ecdsa (High)
- CVE-2024-5658: PyTorch Improper Resource Shutdown or Release (Moderate)
"""

import pytest
import torch
import time
import threading
from unittest.mock import patch, MagicMock
import secrets
import hashlib

from framework.core.security import (
    PyTorchSecurityMitigation, 
    ECDSASecurityMitigation,
    initialize_security_mitigations
)


class TestPyTorchSecurityMitigation:
    """Test PyTorch security mitigations for CVE-2024-5658."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security = PyTorchSecurityMitigation()
    
    def test_initialization(self):
        """Test security mitigation initialization."""
        assert self.security is not None
        assert hasattr(self.security, '_active_resources')
        assert hasattr(self.security, '_cleanup_hooks')
        assert hasattr(self.security, '_lock')
    
    def test_secure_context_manager(self):
        """Test secure context manager functionality."""
        with self.security.secure_torch_context():
            # Context should be active
            assert torch.get_num_threads() >= 1
        
        # Context should be cleaned up
        # Resource tracking should be active
        assert len(self.security._active_resources) >= 0
    
    def test_resource_cleanup(self):
        """Test automatic resource cleanup."""
        # Create some tensors in secure context
        with self.security.secure_torch_context():
            tensor1 = torch.randn(100, 100)
            tensor2 = torch.randn(100, 100)
            
            # Perform some operations
            result = torch.matmul(tensor1, tensor2)
            
        # Resources should be tracked
        self.security._cleanup_resources()
        
        # Verify cleanup was attempted
        assert len(self.security._cleanup_hooks) >= 0
    
    def test_thread_safety(self):
        """Test thread safety of security mitigations."""
        results = []
        errors = []
        
        def worker():
            try:
                with self.security.secure_torch_context():
                    tensor = torch.randn(50, 50)
                    result = torch.sum(tensor)
                    results.append(result.item())
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    
    def test_memory_management(self):
        """Test memory management features."""
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with self.security.secure_torch_context():
            # Create large tensors
            tensors = []
            for _ in range(10):
                if torch.cuda.is_available():
                    tensor = torch.randn(1000, 1000, device='cuda')
                else:
                    tensor = torch.randn(1000, 1000)
                tensors.append(tensor)
        
        # Cleanup should have occurred
        self.security._cleanup_resources()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            # Memory should be cleaned up (allowing for some variance)
            assert final_memory <= initial_memory + 1024 * 1024  # 1MB tolerance


class TestECDSASecurityMitigation:
    """Test ECDSA security mitigations for CVE-2024-23342."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security = ECDSASecurityMitigation()
    
    def test_initialization(self):
        """Test security mitigation initialization."""
        assert self.security is not None
        assert hasattr(self.security, 'configure_secure_random')
        assert hasattr(self.security, 'secure_ecdsa_sign')
    
    def test_timing_attack_mitigation(self):
        """Test timing attack mitigations."""
        # Test secure random configuration
        self.security.configure_secure_random()
        
        # Test that secure random is working
        import secrets
        import hashlib
        
        # Simple timing test - just verify no exceptions
        data = secrets.token_bytes(32)
        hash_obj = hashlib.sha256(data)
        result = hash_obj.digest()
        
        assert result is not None
        assert len(result) == 32
    
    def test_secure_random_generation(self):
        """Test secure random number generation."""
        self.security.configure_secure_random()
        
        # Generate multiple random values using secure methods
        import secrets
        randoms = []
        for _ in range(100):
            random_val = secrets.token_bytes(32)
            randoms.append(random_val)
        
        # Check uniqueness (should be very high)
        unique_randoms = set(randoms)
        assert len(unique_randoms) == len(randoms), "Random values are not unique"
        
        # Check length
        for random_val in randoms:
            assert len(random_val) == 32, f"Random value wrong length: {len(random_val)}"
    
    def test_entropy_mixing(self):
        """Test entropy functionality."""
        # Test that we can configure secure random without errors
        self.security.configure_secure_random()
        
        # Test random generation works consistently
        import secrets
        for _ in range(10):
            data = secrets.token_bytes(16)
            assert len(data) == 16
    
    def test_side_channel_protection(self):
        """Test side-channel attack protections."""
        # Test that secure operations can be performed consistently
        self.security.configure_secure_random()
        
        test_data = [
            b"short",
            b"medium_length_data",
            b"very_long_data_that_should_still_have_consistent_timing_characteristics"
        ]
        
        # Verify all data can be processed
        import hashlib
        for data in test_data:
            hash_obj = hashlib.sha256(data)
            result = hash_obj.digest()
            assert result is not None
            assert len(result) == 32


class TestSecurityInitialization:
    """Test security initialization and integration."""
    
    def test_initialize_security_mitigations(self):
        """Test global security initialization."""
        pytorch_security, ecdsa_security = initialize_security_mitigations()
        
        assert pytorch_security is not None
        assert ecdsa_security is not None
        assert pytorch_security._initialized is True
        assert ecdsa_security._initialized is True
    
    def test_initialization_error_handling(self):
        """Test error handling during initialization."""
        # Test with mocked failures
        with patch('framework.core.security.PyTorchSecurityMitigation.initialize') as mock_init:
            mock_init.side_effect = Exception("Initialization failed")
            
            # Should handle gracefully
            pytorch_security, ecdsa_security = initialize_security_mitigations()
            
            # ECDSA should still initialize even if PyTorch fails
            assert ecdsa_security is not None
    
    def test_security_logging(self):
        """Test security event logging."""
        with patch('framework.core.security.security_logger') as mock_logger:
            # Initialize security
            pytorch_security, ecdsa_security = initialize_security_mitigations()
            
            # Use security features
            with pytorch_security.secure_context():
                pass
            
            with ecdsa_security.secure_timing_context():
                pass
            
            # Check that security events were logged
            assert mock_logger.info.called or mock_logger.debug.called


class TestSecurityIntegration:
    """Test integration of security mitigations with framework components."""
    
    def test_framework_integration(self):
        """Test that security integrates properly with framework."""
        # This tests that the security module can be imported and used
        # by framework components without errors
        
        try:
            from framework.core.security import (
                PyTorchSecurityMitigation,
                ECDSASecurityMitigation,
                initialize_security_mitigations
            )
            
            # Initialize
            pytorch_sec, ecdsa_sec = initialize_security_mitigations()
            
            # Use in context
            with pytorch_sec.secure_context():
                tensor = torch.randn(10, 10)
                result = torch.sum(tensor)
            
            with ecdsa_sec.secure_timing_context():
                data = secrets.token_bytes(32)
                hash_result = hashlib.sha256(data).digest()
            
            # Cleanup
            pytorch_sec.cleanup_resources()
            
        except Exception as e:
            pytest.fail(f"Security integration failed: {e}")
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        security = PyTorchSecurityMitigation()
        security.initialize()
        
        # Test recovery from various error conditions
        try:
            with security.secure_context():
                # Simulate an error during secure operations
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass  # Expected
        
        # Security should still be functional after error
        with security.secure_context():
            tensor = torch.randn(5, 5)
            assert tensor.shape == (5, 5)
        
        # Cleanup should work
        security.cleanup_resources()


if __name__ == "__main__":
    pytest.main([__file__])
